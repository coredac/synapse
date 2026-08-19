"""Programming model for computations placed on a TileArray.

This module defines the typed frontend representation of a tile-array program.
It records computation independently of MLIR. Compiler lowering later converts
the recorded program into Taskflow and Neura operations.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from .spatial import Tile, TileArray


class TileArrayScalarType(str, Enum):
    """The scalar types supported by the TileArray programming model.
    This is intentionally independent of MLIR types. The lowering converts
    these frontend types into the corresponding MLIR types.

    Additional scalar types can be added here as the language grows.
    """

    I32 = "i32"
    F32 = "f32"


@dataclass(frozen=True)
class TileArrayValue:
    """A typed value produced by one tile-array operation."""

    id: int
    dtype: TileArrayScalarType
    _builder: TileArrayBuilder = field(repr=False)


# ---------------------------------------------------------------
# Typed tile-array operations
# ---------------------------------------------------------------


@dataclass(frozen=True)
class ConstantOp:
    """A scalar constant produced by a tile-array operation."""

    result: TileArrayValue
    value: int | float
    tile: Tile


@dataclass(frozen=True)
class AddOp:
    """A scalar addition executed by a tile-array operation."""

    result: TileArrayValue
    lhs: TileArrayValue
    rhs: TileArrayValue
    tile: Tile


# This union explicitly lists every operation currently supported by the
# tile-array frontend. Future operations such as MacOp and GatherOp should be
# added here.
TileArrayOp: TypeAlias = ConstantOp | AddOp


@dataclass(frozen=True)
class TileArrayProgram:
    """A tile-array program produced by TileArrayBuilder."""

    array: TileArray
    operations: tuple[TileArrayOp, ...]


# ---------------------------------------------------------------
# Internal program builder
# ---------------------------------------------------------------
class TileArrayBuilder:
    """A tile-array program builder.

    The builder records typed operations in user-program order. Once build()
    is called, it returns a TileArrayProgram.
    """

    def __init__(self):
        self._array: TileArray | None = None
        self._operations: list[TileArrayOp] = []
        self._next_value_id = 0
        self._token = None
        self._is_built = False

    def __enter__(self):
        """Make this builder active for tile-array DSL calls."""
        if _active_builder.get() is not None:
            raise RuntimeError("Cannot enter a nested TileArrayBuilder context")
        if self._token is not None:
            raise RuntimeError("TileArrayBuilder context is already active")
        self._token = _active_builder.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Restore the previously active builder."""
        if self._token is None:
            raise RuntimeError("TileArrayBuilder is not active")

        _active_builder.reset(self._token)
        self._token = None

    def _new_value(self, *, dtype: TileArrayScalarType) -> TileArrayValue:
        """Allocate the next value identifier."""
        result = TileArrayValue(id=self._next_value_id, dtype=dtype, _builder=self)
        self._next_value_id += 1
        return result

    def _bind_tile_array(self, tile: Tile) -> None:
        """Bind the program to one TileArray."""
        if not isinstance(tile, Tile):
            raise TypeError("tile must be a Tile")
        if self._array is None:
            self._array = tile.array
            return
        if tile.array is not self._array:
            raise ValueError(
                "all operations in a TileArrayProgram must use tiles from the same TileArray"
            )

    def _check_operand(self, operand: TileArrayValue) -> None:
        """Verify that an operand was produced by this builder."""
        if not isinstance(operand, TileArrayValue):
            raise TypeError("operation operand must be a TileArrayValue")

        if operand._builder is not self:
            raise ValueError(
                "operation operand belongs to a different TileArrayProgram"
            )

    def _check_can_emit(self) -> None:
        """Reject operations emitted after the program has been finalized."""
        if self._is_built:
            raise RuntimeError(
                "cannot emit operations after building a TileArrayProgram"
            )

    def emit_constant(
        self, value: int | float, *, dtype: TileArrayScalarType, tile: Tile
    ) -> TileArrayValue:
        """Record one scalar constant operation."""
        self._check_can_emit()
        self._bind_tile_array(tile)
        result = self._new_value(dtype=dtype)

        self._operations.append(ConstantOp(result=result, value=value, tile=tile))
        return result

    def emit_add(
        self, lhs: TileArrayValue, rhs: TileArrayValue, *, tile: Tile
    ) -> TileArrayValue:
        """Record one scalar addition operation."""
        self._check_can_emit()
        self._bind_tile_array(tile)
        self._check_operand(lhs)
        self._check_operand(rhs)

        if lhs.dtype != rhs.dtype:
            raise TypeError("add operands must have the same tile-array value type")

        result = self._new_value(dtype=lhs.dtype)

        self._operations.append(AddOp(result=result, lhs=lhs, rhs=rhs, tile=tile))
        return result

    def build(self) -> TileArrayProgram:
        """Finish recording and return a program."""
        if self._token is not None:
            raise RuntimeError(
                "cannot build a TileArrayProgram while its builder is active"
            )
        if self._array is None:
            raise RuntimeError(
                "cannot build an empty TileArrayProgram without a TileArray"
            )
        self._is_built = True
        return TileArrayProgram(array=self._array, operations=tuple(self._operations))


# The active builder is compiler-internal state. Public DSL calls use it to
# find the builder created by frontend lowering.
_active_builder: ContextVar[TileArrayBuilder | None] = ContextVar(
    "active_tile_array_builder",
    default=None,
)


def _require_active_builder() -> TileArrayBuilder:
    """Return the active builder."""
    builder = _active_builder.get()

    if builder is None:
        raise RuntimeError(
            "tile-array operations must be called while lowering a Synapse program"
        )

    return builder


# ---------------------------------------------------------------
# User-facing tile-array program DSL
# ---------------------------------------------------------------
def constant(
    value: int | float, *, tile: Tile, dtype: TileArrayScalarType | None = None
) -> TileArrayValue:
    """Create a scalar constant on one hardware tile.

    Integer literals default to i32. Floating-point literals default to f32.
    Use an explicit dtype when a different representation is required:
        constant(1.0, tile=tile, dtype=TileArrayScalarType.F32)
    """
    if isinstance(value, bool):
        raise TypeError("boolean constants are not supported yet")

    if dtype is None:
        if isinstance(value, int):
            dtype = TileArrayScalarType.I32
        elif isinstance(value, float):
            dtype = TileArrayScalarType.F32
        else:
            raise TypeError(
                "constant currently supports integer and floating-point values"
            )

    if not isinstance(dtype, TileArrayScalarType):
        raise TypeError("dtype must be a TileArrayScalarType")

    if dtype == TileArrayScalarType.I32 and not isinstance(value, int):
        raise TypeError("an i32 constant requires an integer value")

    if dtype in (TileArrayScalarType.F32,) and not isinstance(value, (int, float)):
        raise TypeError("a floating-point constant requires a numeric value")

    return _require_active_builder().emit_constant(
        value,
        dtype=dtype,
        tile=tile,
    )


def add(
    lhs: TileArrayValue,
    rhs: TileArrayValue,
    *,
    tile: Tile,
) -> TileArrayValue:
    """Create a scalar addition on one hardware tile."""

    return _require_active_builder().emit_add(
        lhs,
        rhs,
        tile=tile,
    )
