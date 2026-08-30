"""Programming model for computations placed on a TileArray.

This module defines the typed frontend representation of a tile-array program.
It records computation independently of MLIR. Compiler lowering later converts
the recorded program into Taskflow and Neura operations.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

from .spatial import Tile, TileArray
from .types import DType, f32, i32


@dataclass(frozen=True)
class TileArrayValue:
    """A typed value produced by one tile-array operation."""

    id: int
    dtype: DType
    _builder: TileArrayBuilder = field(repr=False)


# ---------------------------------------------------------------
# Typed tile-array operations
# ---------------------------------------------------------------
@dataclass(frozen=True)
class TileArrayOp:
    """Base class for operations executed on a TileArray.

    ``operands`` may contain any number of input values. Constants use an
    empty tuple, while operations such as add consume input values.
    """

    result: TileArrayValue
    operands: tuple[TileArrayValue, ...]
    tile: Tile


@dataclass(frozen=True)
class ConstantOp(TileArrayOp):
    """A scalar constant produced by a tile-array operation."""

    value: int | float

    def __post_init__(self) -> None:
        """Validate the operation-specific operands and scalar value."""
        if self.operands:
            raise ValueError(
                f"ConstantOp requires zero operands, but got {len(self.operands)}"
            )

        dtype = self.result.dtype
        if not isinstance(dtype, DType):
            raise TypeError("ConstantOp result must use a DType")

        if isinstance(self.value, bool):
            raise TypeError("boolean constants are not supported yet")

        if dtype == DType.I32 and not isinstance(self.value, int):
            raise TypeError("an i32 constant requires an integer value")

        if dtype == DType.F32 and not isinstance(self.value, (int, float)):
            raise TypeError("an f32 constant requires a numeric value")


@dataclass(frozen=True)
class AddOp(TileArrayOp):
    """A scalar addition executed by a tile-array operation."""

    def __post_init__(self) -> None:
        """Validate the operation-specific arity and scalar types."""
        if len(self.operands) != 2:
            raise ValueError(
                f"AddOp requires two operands, but got {len(self.operands)}"
            )

        lhs, rhs = self.operands

        if lhs.dtype != rhs.dtype:
            raise TypeError("AddOp operands must have the same scalar type")

        if self.result.dtype != lhs.dtype:
            raise TypeError("AddOp result type must match its operand type")


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

    def _validate_operand_for_builder(self, operand: TileArrayValue) -> None:
        """Validate that an operand was produced by this builder."""
        if not isinstance(operand, TileArrayValue):
            raise TypeError("operation operand must be a TileArrayValue")

        if operand._builder is not self:
            raise ValueError(
                "operation operand belongs to a different TileArrayProgram"
            )

    def _ensure_not_built(self) -> None:
        """Reject operations emitted after the program has been built."""
        if self._is_built:
            raise RuntimeError(
                "cannot emit operations after building a TileArrayProgram"
            )

    def emit(
        self,
        *,
        operands: tuple[TileArrayValue, ...],
        result_dtype: DType,
        tile: Tile,
        create_operation: Callable[[TileArrayValue], TileArrayOp],
    ) -> TileArrayValue:
        """Create and record one tile-array operation."""
        self._ensure_not_built()
        self._bind_tile_array(tile)

        for operand in operands:
            self._validate_operand_for_builder(operand)

        # Commit the value ID only after operation construction and validation
        # succeed, so a rejected operation does not consume a value ID.
        result = TileArrayValue(
            id=self._next_value_id,
            dtype=result_dtype,
            _builder=self,
        )
        operation = create_operation(result)

        if not isinstance(operation, TileArrayOp):
            raise TypeError("create_operation must return a TileArrayOp")
        if operation.result is not result:
            raise ValueError("create_operation must use the provided result value")
        if operation.operands != operands:
            raise ValueError("create_operation must use the provided operands")
        if operation.tile is not tile:
            raise ValueError("create_operation must use the provided tile")

        self._operations.append(operation)
        self._next_value_id += 1
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
    value: int | float, *, tile: Tile, dtype: DType | None = None
) -> TileArrayValue:
    """Create a scalar constant on one hardware tile.

    Integer literals default to i32. Floating-point literals default to f32.
    Use an explicit dtype when a different representation is required:
        constant(1.0, tile=tile, dtype=DType.F32)
    """
    if dtype is None:
        if type(value) is int:
            dtype = DType.I32
        elif type(value) is float:
            dtype = DType.F32
        else:
            raise TypeError(
                "constant currently supports integer and floating-point values"
            )

    builder = _require_active_builder()

    return builder.emit(
        operands=(),
        result_dtype=dtype,
        tile=tile,
        create_operation=lambda result: ConstantOp(
            result=result,
            operands=(),
            tile=tile,
            value=value,
        ),
    )


def add(
    lhs: TileArrayValue,
    rhs: TileArrayValue,
    *,
    tile: Tile,
) -> TileArrayValue:
    """Create a scalar addition on one hardware tile."""

    builder = _require_active_builder()
    operands = (lhs, rhs)

    return builder.emit(
        operands=operands,
        result_dtype=lhs.dtype,
        tile=tile,
        create_operation=lambda result: AddOp(
            result=result,
            operands=operands,
            tile=tile,
        ),
    )
