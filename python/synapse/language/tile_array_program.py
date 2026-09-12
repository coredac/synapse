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
from .tensor import Tensor, TensorAccess
from .types import DType


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

    results: tuple[TileArrayValue, ...]
    operands: tuple[TileArrayValue, ...]
    tile: Tile


@dataclass(frozen=True)
class LoadOp(TileArrayOp):
    """Loads through an explicit address or a configured tensor access."""

    source: TensorAccess | None = None

    def __post_init__(self) -> None:
        """Validates the selected address form and result type."""
        if len(self.results) != 1:
            raise ValueError("LoadOp requires exactly one result")
        if self.source is None:
            if len(self.operands) != 1:
                raise ValueError("dynamic LoadOp requires one address operand")
            if self.operands[0].dtype != DType.I32:
                raise TypeError("dynamic addresses currently require i32")
        else:
            if self.operands:
                raise ValueError("configured LoadOp has no address operand")
            if self.results[0].dtype != self.source.dtype:
                raise TypeError("LoadOp result type must match its source")

    @property
    def addr(self) -> TileArrayValue | None:
        """Returns the explicit address when the load is dynamic."""
        return self.operands[0] if self.operands else None


@dataclass(frozen=True)
class StoreOp(TileArrayOp):
    """Stores through an explicit address or a configured tensor access."""

    target: TensorAccess | None = None

    def __post_init__(self) -> None:
        """Validates the value and selected address form."""
        if self.results:
            raise ValueError("StoreOp produces no results")
        if self.target is None:
            if len(self.operands) != 2:
                raise ValueError("dynamic StoreOp requires value and address operands")
            if self.operands[1].dtype != DType.I32:
                raise TypeError("dynamic addresses currently require i32")
        else:
            if len(self.operands) != 1:
                raise ValueError("configured StoreOp requires one value operand")
            if self.operands[0].dtype != self.target.dtype:
                raise TypeError("StoreOp value type must match its target")

    @property
    def addr(self) -> TileArrayValue | None:
        """Returns the explicit address when the store is dynamic."""
        return self.operands[1] if len(self.operands) == 2 else None


@dataclass(frozen=True)
class ConstantOp(TileArrayOp):
    """A scalar constant produced by a tile-array operation."""

    value: int | float

    def __post_init__(self) -> None:
        """Validates the operation-specific operands and scalar value."""
        if len(self.results) != 1:
            raise ValueError("ConstantOp requires exactly one result")

        if self.operands:
            raise ValueError(
                f"ConstantOp requires zero operands, but got {len(self.operands)}"
            )

        result = self.results[0]
        dtype = result.dtype
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
        """Validates the operation-specific arity and scalar types."""
        if len(self.results) != 1:
            raise ValueError("AddOp requires exactly one result")

        if len(self.operands) != 2:
            raise ValueError(
                f"AddOp requires two operands, but got {len(self.operands)}"
            )

        result = self.results[0]
        lhs, rhs = self.operands

        if lhs.dtype != rhs.dtype:
            raise TypeError("AddOp operands must have the same scalar type")

        if result.dtype != lhs.dtype:
            raise TypeError("AddOp result type must match its operand type")


@dataclass(frozen=True)
class MacOp(TileArrayOp):
    """A configured MAC using one stationary scalar value."""

    stationary_value: TensorAccess

    @property
    def accumulated(self) -> TileArrayValue:
        """Returns the accumulated output value."""

        return self.results[0]

    @property
    def forwarded(self) -> TileArrayValue:
        """Returns the value forwarded from input0."""

        return self.results[1]

    def __post_init__(self) -> None:
        """Validates configured MAC operands and stationary data."""

        if len(self.results) != 2:
            raise ValueError("MacOp requires accumulated and forwarded results")
        if len(self.operands) not in (1, 2):
            raise ValueError("MacOp requires a flowing input and optional partial sum")
        if not self.stationary_value.is_scalar:
            raise ValueError("stationary data must identify one scalar")
        if any(operand.dtype != self.accumulated.dtype for operand in self.operands):
            raise TypeError("MacOp operands and result must have the same dtype")
        if self.stationary_value.dtype != self.accumulated.dtype:
            raise TypeError("MacOp stationary data and result must have the same dtype")
        if self.forwarded.dtype != self.accumulated.dtype:
            raise TypeError("MacOp results must have the same dtype")


@dataclass(frozen=True)
class StationaryBinding:
    """Stationary data assigned to the tiles of one template program."""

    source: Tensor
    tile_values: tuple[tuple[Tile, TensorAccess], ...]


@dataclass(frozen=True)
class TileArrayProgram:
    """A tile-array program produced by TileArrayBuilder."""

    array: TileArray
    arguments: tuple[Tensor, ...]
    operations: tuple[TileArrayOp, ...]
    stationary: StationaryBinding | None


# ---------------------------------------------------------------
# Internal program builder
# ---------------------------------------------------------------
class TileArrayBuilder:
    """A tile-array program builder.

    The builder records typed operations in user-program order. Once build()
    is called, it returns a TileArrayProgram.
    """

    def __init__(self, arguments: tuple[Tensor, ...] = ()):
        self._arguments = arguments
        self._array: TileArray | None = None
        self._operations: list[TileArrayOp] = []
        self._next_value_id = 0
        self._token = None
        self._is_built = False

    def __enter__(self):
        """Makes this builder active for tile-array DSL calls."""
        if _active_builder.get() is not None:
            raise RuntimeError("Cannot enter a nested TileArrayBuilder context")
        if self._token is not None:
            raise RuntimeError("TileArrayBuilder context is already active")
        self._token = _active_builder.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Restores the previously active builder."""
        if self._token is None:
            raise RuntimeError("TileArrayBuilder is not active")

        _active_builder.reset(self._token)
        self._token = None

    def _bind_tile_array(self, array: TileArray) -> None:
        """Binds the program to one TileArray."""
        if self._array is None:
            self._array = array
            return
        if array is not self._array:
            raise ValueError(
                "all operations in a TileArrayProgram must use tiles from the same TileArray"
            )

    def _validate_operand_for_builder(self, operand: TileArrayValue) -> None:
        """Validates that an operand was produced by this builder."""
        if not isinstance(operand, TileArrayValue):
            raise TypeError("operation operand must be a TileArrayValue")

        if operand._builder is not self:
            raise ValueError(
                "operation operand belongs to a different TileArrayProgram"
            )

    def _ensure_not_built(self) -> None:
        """Rejects operations emitted after the program has been built."""
        if self._is_built:
            raise RuntimeError(
                "cannot emit operations after building a TileArrayProgram"
            )

    def emit(
        self,
        *,
        operands: tuple[TileArrayValue, ...],
        result_dtypes: tuple[DType, ...],
        tile: Tile,
        create_operation: Callable[[tuple[TileArrayValue, ...]], TileArrayOp],
    ) -> tuple[TileArrayValue, ...]:
        """Creates and records one tile-array operation."""
        self._ensure_not_built()
        self._bind_tile_array(tile.array)

        for operand in operands:
            self._validate_operand_for_builder(operand)

        # Commits the value ID only after operation construction and validation
        # succeed, so a rejected operation does not consume a value ID.
        results = tuple(
            TileArrayValue(id=self._next_value_id + index, dtype=dtype, _builder=self)
            for index, dtype in enumerate(result_dtypes)
        )

        operation = create_operation(results)

        if not isinstance(operation, TileArrayOp):
            raise TypeError("create_operation must return a TileArrayOp")
        if len(operation.results) != len(results) or any(
            actual is not expected
            for actual, expected in zip(operation.results, results)
        ):
            raise ValueError("create_operation must use the provided result values")
        if operation.operands != operands:
            raise ValueError("create_operation must use the provided operands")
        if operation.tile is not tile:
            raise ValueError("create_operation must use the provided tile")

        self._operations.append(operation)
        self._next_value_id += len(results)
        return results

    def build(self) -> TileArrayProgram:
        """Finishes recording and returns a program."""
        if self._token is not None:
            raise RuntimeError(
                "cannot build a TileArrayProgram while its builder is active"
            )
        if self._array is None:
            raise RuntimeError(
                "cannot build an empty TileArrayProgram without a TileArray"
            )

        mac_operations = [
            operation for operation in self._operations if isinstance(operation, MacOp)
        ]

        stationary = None

        if mac_operations:
            stationary_source = mac_operations[0].stationary_value.source

            if any(
                operation.stationary_value.source is not stationary_source
                for operation in mac_operations
            ):
                raise ValueError("all configured MACs must use one stationary source")
            stationary = StationaryBinding(
                source=stationary_source,
                tile_values=tuple(
                    (operation.tile, operation.stationary_value)
                    for operation in mac_operations
                ),
            )

        self._is_built = True
        return TileArrayProgram(
            array=self._array,
            arguments=self._arguments,
            operations=tuple(self._operations),
            stationary=stationary,
        )


# The active builder is compiler-internal state. Public DSL calls use it to
# find the builder created by frontend lowering.
_active_builder: ContextVar[TileArrayBuilder | None] = ContextVar(
    "active_tile_array_builder",
    default=None,
)


def _require_active_builder() -> TileArrayBuilder:
    """Returns the active builder."""
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
    """Creates a scalar constant on one hardware tile.

    Integer literals default to i32. Floating-point literals default to f32.
    An explicit dtype selects a different representation:
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
        result_dtypes=(dtype,),
        tile=tile,
        create_operation=lambda results: ConstantOp(
            results=results,
            operands=(),
            tile=tile,
            value=value,
        ),
    )[0]


def add(
    lhs: TileArrayValue,
    rhs: TileArrayValue,
    *,
    tile: Tile,
) -> TileArrayValue:
    """Creates a scalar addition on one hardware tile."""

    builder = _require_active_builder()
    operands = (lhs, rhs)

    return builder.emit(
        operands=operands,
        result_dtypes=(lhs.dtype,),
        tile=tile,
        create_operation=lambda results: AddOp(
            results=results,
            operands=operands,
            tile=tile,
        ),
    )[0]


def load(
    source: TensorAccess | None = None,
    *,
    addr: TileArrayValue | None = None,
    dtype: DType | None = None,
    tile: Tile,
) -> TileArrayValue:
    """Loads from a tensor access or an explicit target address.

    Configured accesses enumerate logical indices in lexicographic order,
    with the last varying dimension advancing fastest. Dynamic addresses
    already use the target address representation; they are not tensor indices.
    """
    if (source is None) == (addr is None):
        raise ValueError("load requires exactly one of source and addr")
    if source is not None:
        if not isinstance(source, TensorAccess):
            raise TypeError("load source must be a TensorAccess")
        if dtype is not None and dtype != source.dtype:
            raise TypeError("load dtype must match its source")
        dtype = source.dtype
    if not isinstance(dtype, DType):
        raise TypeError("dynamic load requires an explicit DType")

    operands = () if addr is None else (addr,)
    return _require_active_builder().emit(
        operands=operands,
        result_dtypes=(dtype,),
        tile=tile,
        create_operation=lambda results: LoadOp(
            results=results,
            operands=operands,
            tile=tile,
            source=source,
        ),
    )[0]


def store(
    value: TileArrayValue,
    *,
    target: TensorAccess | None = None,
    addr: TileArrayValue | None = None,
    tile: Tile,
) -> None:
    """Stores a value through a tensor access or explicit target address."""
    if (target is None) == (addr is None):
        raise ValueError("store requires exactly one of target and addr")
    if target is not None and not isinstance(target, TensorAccess):
        raise TypeError("store target must be a TensorAccess")
    operands = (value,) if addr is None else (value, addr)
    _require_active_builder().emit(
        operands=operands,
        result_dtypes=(),
        tile=tile,
        create_operation=lambda results: StoreOp(
            results=results,
            operands=operands,
            tile=tile,
            target=target,
        ),
    )


def mac(
    input0: TileArrayValue,
    input1: TileArrayValue | None = None,
    *,
    stationary: TensorAccess,
    tile: Tile,
) -> tuple[TileArrayValue, TileArrayValue]:
    """Creates one configured MAC operation."""

    if not isinstance(stationary, TensorAccess):
        raise TypeError("mac stationary data must be a tensor access")

    operands = (input0,) if input1 is None else (input0, input1)

    builder = _require_active_builder()

    accumulated, forwarded = builder.emit(
        operands=operands,
        result_dtypes=(input0.dtype, input0.dtype),
        tile=tile,
        create_operation=lambda results: MacOp(
            results=results,
            operands=operands,
            tile=tile,
            stationary_value=stationary,
        ),
    )

    return accumulated, forwarded
