"""Programming model for computations placed on a TileArray.

This module defines the typed frontend representation of a tile-array program.
It records computation independently of MLIR. Compiler lowering later converts
the recorded program into Taskflow and Neura operations.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum

from .spatial import Port, Tile, TileArray
from .tensor import Tensor, TensorAccess
from .types import DType


class StationaryMode(Enum):
    """The dataflow operand retained locally by configured MACs."""

    WEIGHT = "weight"
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True)
class TileArrayValue:
    """A typed value produced by one tile-array operation."""

    id: int
    dtype: DType
    _builder: TileArrayBuilder = field(repr=False)


@dataclass(frozen=True)
class InputPortBinding:
    """A tensor slice bound to one TileArray input Port."""

    input_des: TileArrayValue
    input_src: TensorAccess
    port: Port


@dataclass(frozen=True)
class OutputPortBinding:
    """A TileArray value bound to a tensor output slice and Port."""

    output_src: TileArrayValue
    output_des: TensorAccess
    port: Port


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
class MacOp(TileArrayOp):
    """A configured MAC using one stationary scalar value."""

    stationary_value: TensorAccess
    stationary_mode: StationaryMode

    def __post_init__(self) -> None:
        """Validates configured MAC operands and stationary data."""

        if self.stationary_mode in (
            StationaryMode.WEIGHT,
            StationaryMode.INPUT,
        ) and len(self.operands) not in (1, 2):
            raise ValueError(
                "weight/input-stationary MacOp requires "
                "an input and optional partial sum"
            )

        if self.stationary_mode == StationaryMode.OUTPUT and len(self.operands) != 2:
            raise ValueError("output-stationary MacOp requires two multiplicands")

        if not self.stationary_value.is_scalar:
            raise ValueError("stationary data must identify one scalar")
        if any(operand.dtype != self.result.dtype for operand in self.operands):
            raise TypeError("MacOp operands and result must have the same dtype")
        if self.stationary_value.dtype != self.result.dtype:
            raise TypeError("MacOp stationary data and result must have the same dtype")


@dataclass(frozen=True)
class StationaryBinding:
    """Stationary data assigned to the tiles of one template program."""

    mode: StationaryMode
    source: Tensor
    tile_values: tuple[tuple[Tile, TensorAccess], ...]


@dataclass(frozen=True)
class TileArrayProgram:
    """A tile-array program produced by TileArrayBuilder."""

    array: TileArray
    arguments: tuple[Tensor, ...]
    input_ports: tuple[InputPortBinding, ...]
    output_ports: tuple[OutputPortBinding, ...]
    operations: tuple[TileArrayOp, ...]
    template_name: str | None
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
        self._input_ports: list[InputPortBinding] = []
        self._output_ports: list[OutputPortBinding] = []
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
        result_dtype: DType,
        tile: Tile,
        create_operation: Callable[[TileArrayValue], TileArrayOp],
    ) -> TileArrayValue:
        """Creates and record one tile-array operation."""
        self._ensure_not_built()
        self._bind_tile_array(tile.array)

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

    def add_input_port(self, *, source: TensorAccess, port: Port) -> TileArrayValue:
        """Creates one scalar kernel input and bind it to a Port."""
        self._ensure_not_built()
        if source.is_scalar:
            raise ValueError("an input Port requires a tensor slice")

        if sum(isinstance(index, slice) for index in source.indices) != 1:
            raise ValueError("an input Port currently supports one varying dimension")

        self._bind_tile_array(port.array)

        result = TileArrayValue(
            id=self._next_value_id, dtype=source.dtype, _builder=self
        )
        self._next_value_id += 1
        self._input_ports.append(
            InputPortBinding(input_src=source, input_des=result, port=port)
        )
        return result

    def add_output_port(
        self, *, value: TileArrayValue, target: TensorAccess, port: Port
    ) -> None:
        """Bind one kernel result to an output slice and Port."""
        self._ensure_not_built()
        self._validate_operand_for_builder(value)

        if target.is_scalar:
            raise ValueError("an output Port requires a tensor slice")
        if sum(isinstance(index, slice) for index in target.indices) != 1:
            raise ValueError("an output Port currently supports one varying dimension")

        if target.dtype != value.dtype:
            raise TypeError("output value and target must have the same dtype")

        self._bind_tile_array(port.array)

        self._output_ports.append(
            OutputPortBinding(output_src=value, output_des=target, port=port)
        )

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

        mac_operations = [
            operation for operation in self._operations if isinstance(operation, MacOp)
        ]

        stationary = None
        template_name = None

        if mac_operations:
            stationary_mode = mac_operations[0].stationary_mode
            stationary_source = mac_operations[0].stationary_value.source

            if any(
                operation.stationary_mode != stationary_mode
                for operation in mac_operations
            ):
                raise ValueError(
                    "all configured MACs in one template must use one stationary mode"
                )
            if any(
                operation.stationary_value.source is not stationary_source
                for operation in mac_operations
            ):
                raise ValueError("all configured MACs must use one stationary source")
            stationary = StationaryBinding(
                mode=stationary_mode,
                source=stationary_source,
                tile_values=tuple(
                    (operation.tile, operation.stationary_value)
                    for operation in mac_operations
                ),
            )
            # Configured MAC networks currently use the systolic template.
            # Template registration will replace this inference later.
            template_name = "systolic_array"

        self._is_built = True
        return TileArrayProgram(
            array=self._array,
            arguments=self._arguments,
            input_ports=tuple(self._input_ports),
            output_ports=tuple(self._output_ports),
            operations=tuple(self._operations),
            template_name=template_name,
            stationary=stationary,
        )


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


def input_port(source: TensorAccess, *, port: Port) -> TileArrayValue:
    """Reads one tensor slice through a TileArray input port."""
    if not isinstance(source, TensorAccess):
        raise TypeError("input_port source must be a tensor access")
    if not isinstance(port, Port):
        raise TypeError("input_port port must be a Port")
    return _require_active_builder().add_input_port(source=source, port=port)


def output_port(value: TileArrayValue, *, target: TensorAccess, port: Port) -> None:
    """Write one TileArray value stream through an output Port."""

    if not isinstance(target, TensorAccess):
        raise TypeError("output_port target must be a tensor access")

    if not isinstance(port, Port):
        raise TypeError("output_port port must be a Port")

    _require_active_builder().add_output_port(value=value, target=target, port=port)


def mac(
    input0: TileArrayValue,
    input1: TileArrayValue | None,
    *,
    stationary: TensorAccess,
    mode: StationaryMode,
    tile: Tile,
) -> TileArrayValue:
    """Create one configured MAC operation.

    Operand meanings depend on the selected stationary mode:

    - WEIGHT: input0 is activation, input1 is an optional partial sum.
    - INPUT: input0 is weight, input1 is an optional partial sum.
    - OUTPUT: input0 and input1 are the two multiplicands.
    """

    if not isinstance(stationary, TensorAccess):
        raise TypeError("mac stationary data must be a tensor access")

    if not isinstance(mode, StationaryMode):
        raise TypeError("mac mode must be a StationaryMode")

    operands = (input0,) if input1 is None else (input0, input1)

    builder = _require_active_builder()

    return builder.emit(
        operands=operands,
        result_dtype=input0.dtype,
        tile=tile,
        create_operation=lambda result: MacOp(
            result=result,
            operands=operands,
            tile=tile,
            stationary_value=stationary,
            stationary_mode=mode,
        ),
    )
