"""Lowers Synapse Python programs to compiler input IR."""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatchmethod
from inspect import signature
from itertools import product
from math import prod
from typing import TYPE_CHECKING, cast

from synapse.language.spatial import Tile
from synapse.language.tensor import Tensor, TensorAccess
from synapse.language.tile_array_program import (
    AddOp,
    ConstantOp,
    LoadOp,
    MacOp,
    StoreOp,
    TileArrayBuilder,
    TileArrayOp,
    TileArrayProgram,
)
from synapse.language.types import DType, TensorType

if TYPE_CHECKING:
    from taskflow_mlir.ir import AffineMapAttr, DenseI64ArrayAttr, DictAttr, Value


def lower(program_fn: Callable, *, argument_types: tuple[TensorType, ...] = ()) -> str:
    """Lowers a standalone TileArray program to pre-mapping Taskflow and Neura IR."""
    program = build_tile_array_program(program_fn, argument_types=argument_types)
    return TileArrayProgramLowering(program).lower_to_single_task(program_fn.__name__)


def build_tile_array_program(
    program_fn: Callable, *, argument_types: tuple[TensorType, ...] = ()
) -> TileArrayProgram:
    """Runs a TileArray function and records its operations and tensor accesses.

    Direct compilation and pattern rewriting share this construction step.
    It creates a TileArrayProgram without importing or constructing MLIR.
    """
    # Program arguments currently carry tensor types. Scalar argument capture
    # will use Taskflow value dependencies when that frontend path is added.

    function_signature = signature(program_fn)
    parameter_names = tuple(function_signature.parameters)

    if len(parameter_names) != len(argument_types):
        raise TypeError(
            f"program {program_fn.__name__} expects {len(parameter_names)} arguments, "
            f"but got {len(argument_types)} argument types"
        )

    if any(
        not isinstance(argument_type, TensorType) for argument_type in argument_types
    ):
        raise TypeError("program argument types must be TensorType values")

    arguments = [
        Tensor(name=name, type=argument_type)
        for name, argument_type in zip(parameter_names, argument_types)
    ]

    builder = TileArrayBuilder(arguments=tuple(arguments))

    # Executes the user's tile-array DSL while recording its operations.
    with builder:
        program_fn(*arguments)

    return builder.build()


class TileArrayProgramLowering:
    """Lowers a TileArrayProgram into the current MLIR context."""

    def __init__(self, program: TileArrayProgram):
        self.program = program
        self.kernel_input_indices = {
            argument: index for index, argument in enumerate(program.arguments)
        }
        read_sources = {
            operation.source.source
            for operation in program.operations
            if isinstance(operation, LoadOp) and operation.source is not None
        }
        write_sources = {
            operation.target.source
            for operation in program.operations
            if isinstance(operation, StoreOp) and operation.target is not None
        }
        if program.stationary is not None:
            read_sources.add(program.stationary.source)
        if not (read_sources | write_sources).issubset(self.kernel_input_indices):
            raise ValueError("memory accesses must reference program arguments")
        self.has_dynamic_memory = any(
            isinstance(operation, (LoadOp, StoreOp)) and operation.addr is not None
            for operation in program.operations
        )
        # Raw addresses may alias any captured buffer. Task dependencies remain
        # conservative until address provenance becomes available.
        if self.has_dynamic_memory:
            read_sources.update(program.arguments)
            write_sources.update(program.arguments)
        self.read_arguments = tuple(
            argument for argument in program.arguments if argument in read_sources
        )
        self.write_arguments = tuple(
            argument for argument in program.arguments if argument in write_sources
        )

    # The standalone path is used only when a TileArray program is lowered
    # without an existing Taskflow graph.
    def lower_to_single_task(self, program_name: str) -> str:
        """Creates a complete module containing one TileArray task."""

        from taskflow_mlir.dialects import func, neura, taskflow
        from taskflow_mlir.ir import Context, InsertionPoint, Location, Module

        with Context(), Location.unknown():
            taskflow.register_dialect()
            neura.register_dialect()

            module = Module.create()
            argument_types = [
                self.get_memref_type(argument.type)
                for argument in self.program.arguments
            ]
            result_types = [
                self.get_memref_type(argument.type) for argument in self.write_arguments
            ]

            with InsertionPoint(module.body):
                function = func.FuncOp(
                    program_name,
                    (argument_types, result_types),
                )
                function_block = function.add_entry_block()

            function_values = dict(
                zip(self.program.arguments, function_block.arguments)
            )
            read_values = [
                function_values[argument] for argument in self.read_arguments
            ]
            write_values = [
                function_values[argument] for argument in self.write_arguments
            ]

            used_arguments = set(self.read_arguments) | set(self.write_arguments)
            other_arguments = tuple(
                argument
                for argument in self.program.arguments
                if argument not in used_arguments
            )
            other_values = [function_values[argument] for argument in other_arguments]

            with InsertionPoint(function_block):
                task = taskflow.TaskflowTaskOp(
                    done_reads=[],
                    done_writes=[value.type for value in write_values],
                    value_outputs=[],
                    will_reads=read_values,
                    will_writes=write_values,
                    value_inputs=other_values,
                    task_name=program_name,
                    original_read_memrefs=read_values,
                    original_write_memrefs=write_values,
                )
                task_block = task.body.blocks.append(
                    *[value.type for value in read_values],
                    *[value.type for value in write_values],
                    *[value.type for value in other_values],
                )

                func.ReturnOp(task.done_writes)

            task_arguments = dict(
                zip(
                    self.read_arguments + self.write_arguments + other_arguments,
                    task_block.arguments,
                )
            )

            with InsertionPoint(task_block):
                self.lower_to_kernel(task_arguments)

                taskflow.TaskflowYieldOp(
                    done_reads=[],
                    done_writes=[
                        task_arguments[argument] for argument in self.write_arguments
                    ],
                    value_results=[],
                )

            if not module.operation.verify():
                raise RuntimeError("generated Taskflow/Neura module is invalid")

            return str(module)

    def lower_to_kernel(self, argument_values: dict[Tensor, Value]) -> None:
        """Creates one kernel at the caller's insertion point in an existing task.

        The caller owns the current context, location, task, and terminator.
        Buffer arguments retain their program order for configuration indices.
        """
        from taskflow_mlir.dialects import neura
        from taskflow_mlir.ir import InsertionPoint, StringAttr

        inputs = []
        for argument in self.program.arguments:
            value = argument_values[argument]
            if value.type != self.get_memref_type(argument.type):
                raise TypeError(f"unexpected memref type for {argument.name}")
            inputs.append(value)

        # Configuration validation precedes IR insertion.
        metadata = self.get_kernel_metadata()
        memory_configs: dict[int, tuple[TensorAccess, DenseI64ArrayAttr]] = {}
        for index, operation in enumerate(self.program.operations):
            if isinstance(operation, LoadOp):
                access = operation.source
            elif isinstance(operation, StoreOp):
                access = operation.target
            else:
                continue
            if access is not None:
                memory_configs[index] = (access, self.get_memory_offsets(access))
        kernel = neura.KernelOp(
            outputs=[],
            inputs=inputs,
            iter_args_init=[],
            accelerator=StringAttr.get("neura"),
            kernel_metadata=metadata,
        )
        block = kernel.body.blocks.append(*[value.type for value in inputs])
        values_by_id: dict[int, Value] = {}
        with InsertionPoint(block):
            for index, operation in enumerate(self.program.operations):
                operands = tuple(values_by_id[value.id] for value in operation.operands)
                memory_config = memory_configs.get(index)
                if memory_config is not None:
                    access, _ = memory_config
                    # The memref supplies a launch-time base rather than a
                    # routed address value. Its SSA use stays inside the kernel.
                    base = block.arguments[self.kernel_input_indices[access.source]]
                    operands += (base,)
                result_types = tuple(
                    self.get_mlir_type(value.dtype) for value in operation.results
                )
                lowered = self.lower_operation(operation, operands, result_types)
                lowered.operation.attributes["placement"] = self.get_placement(
                    operation.tile
                )
                if memory_config is not None:
                    lowered.operation.attributes["constants"] = memory_config[1]
                results = tuple(lowered.results)
                if len(results) != len(operation.results):
                    raise RuntimeError(
                        "frontend and MLIR operation result counts differ"
                    )
                for source, result in zip(operation.results, results):
                    values_by_id[source.id] = result
            neura.YieldOp(iter_args_next=[], results_=[])

    def get_mlir_type(self, dtype: DType):
        """Translates a frontend data type into an MLIR type."""

        from taskflow_mlir.ir import F32Type, IntegerType

        if dtype == DType.I32:
            return IntegerType.get_signless(32)

        if dtype == DType.F32:
            return F32Type.get()

        raise NotImplementedError(f"unsupported tile-array data type: {dtype.value}")

    def get_memref_type(self, tensor_type: TensorType):
        """Translates a TensorType into a Taskflow MemRef type."""

        from taskflow_mlir.ir import MemRefType

        return MemRefType.get(
            list(tensor_type.shape), self.get_mlir_type(tensor_type.dtype)
        )

    def get_memory_offsets(self, access: TensorAccess) -> DenseI64ArrayAttr:
        """Converts a static access to Neura's constant element-offset array."""
        from taskflow_mlir.ir import DenseI64ArrayAttr

        shape = access.source.type.shape
        dimensions = [
            range(*index.indices(size)) if isinstance(index, slice) else (index,)
            for size, index in zip(shape, access.indices)
        ]
        strides = [prod(shape[index + 1 :]) for index in range(len(shape))]
        offsets = [
            sum(index * stride for index, stride in zip(indices, strides))
            for indices in product(*dimensions)
        ]
        if not offsets:
            raise ValueError("configured address queues cannot be empty")
        return DenseI64ArrayAttr.get(offsets)

    def get_stationary_map(self) -> AffineMapAttr:
        """Infers a translated weight-stationary map from tile bindings."""
        from taskflow_mlir.ir import AffineExpr, AffineMap, AffineMapAttr

        stationary = self.program.stationary
        if stationary is None or not stationary.tile_values:
            raise ValueError("stationary bindings are required")
        tile, access = stationary.tile_values[0]
        if len(access.indices) != 2 or not access.is_scalar:
            raise ValueError(
                "stationary mapping currently requires static rank-2 elements"
            )
        k_index, column_index = access.indices
        if not isinstance(k_index, int) or not isinstance(column_index, int):
            raise TypeError("stationary indices must be static integers")
        row_origin = k_index + tile.y
        column_shift = column_index - tile.x
        for tile, access in stationary.tile_values:
            if access.indices != (row_origin - tile.y, tile.x + column_shift):
                raise ValueError("stationary accesses do not form a translated WS map")
        x = AffineExpr.get_dim(0)
        y = AffineExpr.get_dim(1)
        row = AffineExpr.get_add(
            AffineExpr.get_constant(row_origin),
            AffineExpr.get_mul(AffineExpr.get_constant(-1), y),
        )
        column = AffineExpr.get_add(x, AffineExpr.get_constant(column_shift))
        return AffineMapAttr.get(AffineMap.get(2, 0, [row, column]))

    def get_kernel_metadata(self) -> DictAttr | None:
        """Materializes the supported Neura stationary implementation metadata."""
        from taskflow_mlir.ir import DictAttr, IntegerAttr, StringAttr

        stationary = self.program.stationary
        if stationary is None:
            return None
        return DictAttr.get(
            {
                "kind": StringAttr.get("template"),
                "template": DictAttr.get(
                    {
                        "name": StringAttr.get("systolic_array"),
                        "stationary": DictAttr.get(
                            {
                                "kernel_input": IntegerAttr.get(
                                    self.get_mlir_type(DType.I32),
                                    self.kernel_input_indices[stationary.source],
                                ),
                                "map": self.get_stationary_map(),
                            }
                        ),
                    }
                ),
            }
        )

    def get_placement(self, tile: Tile) -> DictAttr:
        """Builds Neura placement directly from a Tile coordinate."""

        from taskflow_mlir.ir import DictAttr, IntegerAttr

        i32_type = self.get_mlir_type(DType.I32)

        return DictAttr.get(
            {
                "x": IntegerAttr.get(i32_type, tile.x),
                "y": IntegerAttr.get(i32_type, tile.y),
            }
        )

    def get_constant_attribute(
        self,
        operation: ConstantOp,
        result_type,
    ):
        """Builds the typed MLIR attribute for a constant value."""

        from taskflow_mlir.ir import FloatAttr, IntegerAttr

        result = operation.results[0]

        if result.dtype == DType.I32:
            return IntegerAttr.get(result_type, cast(int, operation.value))

        if result.dtype == DType.F32:
            return FloatAttr.get(result_type, float(operation.value))

        raise NotImplementedError(f"unsupported constant type: {result.dtype.value}")

    @singledispatchmethod
    def lower_operation(self, operation: TileArrayOp, operands, result_types):
        """Lowers one frontend TileArray operation to a Neura operation.

        The caller handles common lowering such as resolving operands,
        attaching placement, and recording the resulting SSA value.
        """
        raise NotImplementedError(
            f"unsupported tile-array operation: {type(operation).__name__}"
        )

    @lower_operation.register
    def lower_constant(self, operation: ConstantOp, operands, result_types):
        """Lowers a ConstantOp to neura.constant."""

        from taskflow_mlir.dialects import neura

        return neura.ConstantOp(
            result_types[0], self.get_constant_attribute(operation, result_types[0])
        )

    @lower_operation.register
    def lower_add(self, operation: AddOp, operands, result_types):
        """Lowers a frontend AddOp to neura.add."""

        from taskflow_mlir.dialects import neura

        lhs, rhs = operands

        return neura.AddOp(result_types[0], lhs, rhs=rhs)

    @lower_operation.register
    def lower_mac(self, operation: MacOp, operands, result_types):
        """Lowers a configured MacOp to neura.mac."""

        from taskflow_mlir.dialects import neura

        input0 = operands[0]

        input1 = operands[1] if len(operands) == 2 else None
        accumulated_type, forwarded_type = result_types

        return neura.MacOp(
            accumulated_type,
            forwarded_type,
            input0,
            input1=input1,
        )

    @lower_operation.register
    def lower_load(self, operation: LoadOp, operands, result_types):
        """Lowers a configured or explicitly addressed load."""
        from taskflow_mlir.dialects import neura

        return neura.LoadOp(result_types[0], addr=operands[0] if operands else None)

    @lower_operation.register
    def lower_store(self, operation: StoreOp, operands, result_types):
        """Lowers a configured or explicitly addressed store."""
        from taskflow_mlir.dialects import neura

        return neura.StoreOp(
            operands[0], addr=operands[1] if len(operands) == 2 else None
        )
