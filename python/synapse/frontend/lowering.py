"""Lower Synapse Python programs to compiler input IR."""

from collections.abc import Callable
from functools import singledispatch
from inspect import signature
from typing import cast

from synapse.language.spatial import Tile
from synapse.language.tensor import Tensor, TensorAccess
from synapse.language.tile_array_program import (
    AddOp,
    ConstantOp,
    MacOp,
    TileArrayBuilder,
    TileArrayOp,
    TileArrayProgram,
)
from synapse.language.types import DType, TensorType


def lower(program_fn: Callable, *, argument_types: tuple[TensorType, ...] = ()) -> str:
    """Lowers one tile-array program to pre-mapping Taskflow and Neura IR."""
    # TODO: Generalize program arguments beyond Tensor. Accept both DType and
    # TensorType, create Scalar or Tensor symbolic values accordingly, and
    # lower scalar dependencies through Taskflow value_inputs/value_outputs.

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

    program = builder.build()

    return _lower_tile_array_program(program_name=program_fn.__name__, program=program)


def _lower_tile_array_program(
    *,
    program_name: str,
    program: TileArrayProgram,
) -> str:
    """Convert a TileArrayProgram into pre-mapping Taskflow and Neura IR.

    MLIR imports remain local so users can import ``synapse.language`` without
    requiring the compiled Amoeba Python bindings.
    """

    from taskflow_mlir.dialects import func, neura, taskflow
    from taskflow_mlir.ir import (
        AffineExpr,
        AffineMap,
        AffineMapAttr,
        ArrayAttr,
        Context,
        DictAttr,
        F32Type,
        FloatAttr,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        Location,
        MemRefType,
        Module,
        StringAttr,
    )

    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()

        i32_type = IntegerType.get_signless(32)

        def get_mlir_type(dtype: DType):
            """Translate a frontend data type into an MLIR type."""

            if dtype == DType.I32:
                return i32_type

            if dtype == DType.F32:
                return F32Type.get()

            raise NotImplementedError(
                f"unsupported tile-array data type: {dtype.value}"
            )

        def get_memref_type(tensor_type: TensorType):
            """Translates a TensorType into a Taskflow MemRef type."""
            return MemRefType.get(
                list(tensor_type.shape), get_mlir_type(tensor_type.dtype)
            )

        def get_access_map(access: TensorAccess) -> AffineMapAttr:
            """Builds the affine map that enumerates one tensor access."""

            results = []
            next_dimension = 0

            for index in access.indices:
                if isinstance(index, slice):
                    results.append(AffineExpr.get_dim(next_dimension))
                    next_dimension += 1
                else:
                    results.append(AffineExpr.get_constant(index))

            return AffineMapAttr.get(AffineMap.get(next_dimension, 0, results))

        def get_stationary_map() -> AffineMapAttr:
            """Builds and validates the stationary Tile-to-data map."""
            if program.stationary is None:
                raise RuntimeError("template program requires stationary data")

            for tile, access in program.stationary.tile_values:
                expected_indices = (program.array.y_tiles - 1 - tile.y, tile.x)

                if access.indices != expected_indices:
                    raise ValueError("weight-stationary GEMM requires B[K - 1 - y, x]")

            x = AffineExpr.get_dim(0)
            y = AffineExpr.get_dim(1)

            last_row = AffineExpr.get_constant(program.array.y_tiles - 1)

            negative_y = AffineExpr.get_mul(AffineExpr.get_constant(-1), y)

            weight_row = AffineExpr.get_add(last_row, negative_y)

            return AffineMapAttr.get(AffineMap.get(2, 0, [weight_row, x]))

        def get_constant_attribute(
            operation: ConstantOp,
            result_type,
        ):
            """Build the typed MLIR attribute for a constant value."""
            result = operation.results[0]

            if result.dtype == DType.I32:
                return IntegerAttr.get(result_type, cast(int, operation.value))

            if result.dtype == DType.F32:
                return FloatAttr.get(result_type, float(operation.value))

            raise NotImplementedError(
                f"unsupported constant type: {result.dtype.value}"
            )

        def get_placement(tile: Tile) -> DictAttr:
            """Build Neura placement directly from a Tile coordinate."""

            return DictAttr.get(
                {
                    "x": IntegerAttr.get(i32_type, tile.x),
                    "y": IntegerAttr.get(i32_type, tile.y),
                }
            )

        def get_kernel_metadata() -> DictAttr | None:
            """Materialize template, stationary, and Port metadata."""

            if program.template_name is None:
                return None

            if program.stationary is None:
                raise RuntimeError("template program requires stationary metadata")

            input_ports = ArrayAttr.get(
                [
                    DictAttr.get(
                        {
                            "kernel_input": IntegerAttr.get(i32_type, index),
                            "direction": StringAttr.get(binding.port.direction),
                            "x": IntegerAttr.get(i32_type, binding.port.x),
                            "y": IntegerAttr.get(i32_type, binding.port.y),
                        }
                    )
                    for index, binding in enumerate(program.input_ports)
                ]
            )

            output_ports = ArrayAttr.get(
                [
                    DictAttr.get(
                        {
                            "kernel_result": IntegerAttr.get(i32_type, index),
                            "direction": StringAttr.get(binding.port.direction),
                            "x": IntegerAttr.get(i32_type, binding.port.x),
                            "y": IntegerAttr.get(i32_type, binding.port.y),
                        }
                    )
                    for index, binding in enumerate(program.output_ports)
                ]
            )

            stationary = DictAttr.get(
                {
                    "kernel_input": IntegerAttr.get(i32_type, len(program.input_ports)),
                    "map": get_stationary_map(),
                }
            )

            template = DictAttr.get(
                {
                    "input_ports": input_ports,
                    "name": StringAttr.get(program.template_name),
                    "output_ports": output_ports,
                    "stationary": stationary,
                }
            )

            return DictAttr.get(
                {
                    "kind": StringAttr.get("template"),
                    "template": template,
                }
            )

        @singledispatch
        def lower_operation(operation: TileArrayOp, operands, result_types):
            """Lower one frontend TileArray operation to a Neura operation.

            The caller handles common lowering such as resolving operands,
            attaching placement, and recording the resulting SSA value.
            """
            raise NotImplementedError(
                f"unsupported tile-array operation: {type(operation).__name__}"
            )

        @lower_operation.register
        def lower_constant(operation: ConstantOp, operands, result_types):
            """Lower a ConstantOp to neura.constant."""
            return neura.ConstantOp(
                result_types[0], get_constant_attribute(operation, result_types[0])
            )

        @lower_operation.register
        def lower_add(operation: AddOp, operands, result_types):
            """Lower a frontend AddOp to neura.add."""
            lhs, rhs = operands

            return neura.AddOp(result_types[0], lhs, rhs=rhs)

        @lower_operation.register
        def lower_mac(operation: MacOp, operands, result_types):
            """Lower a configured MacOp to neura.mac."""

            input0 = operands[0]

            input1 = operands[1] if len(operands) == 2 else None
            accumulated_type, forwarded_type = result_types

            return neura.MacOp(
                accumulated_type,
                forwarded_type,
                input0,
                input1=input1,
            )

        read_sources = {binding.input_src.source for binding in program.input_ports}

        if program.stationary is not None:
            read_sources.add(program.stationary.source)

        write_sources = {binding.output_des.source for binding in program.output_ports}

        read_arguments = tuple(
            argument for argument in program.arguments if argument in read_sources
        )

        write_arguments = tuple(
            argument for argument in program.arguments if argument in write_sources
        )

        function_argument_types = [
            get_memref_type(argument.type) for argument in program.arguments
        ]

        function_result_types = [
            get_memref_type(argument.type) for argument in write_arguments
        ]

        module = Module.create()

        with InsertionPoint(module.body):
            function = func.FuncOp(
                program_name,
                (function_argument_types, function_result_types),
            )

            function_block = function.add_entry_block()

        function_values = dict(zip(program.arguments, function_block.arguments))

        read_values = [function_values[argument] for argument in read_arguments]

        write_values = [function_values[argument] for argument in write_arguments]

        with InsertionPoint(function_block):
            task = taskflow.TaskflowTaskOp(
                done_reads=[],
                done_writes=[value.type for value in write_values],
                value_outputs=[],
                will_reads=read_values,
                will_writes=write_values,
                value_inputs=[],
                task_name=program_name,
                original_read_memrefs=read_values,
                original_write_memrefs=write_values,
            )

            task_block = task.body.blocks.append(
                *[value.type for value in read_values],
                *[value.type for value in write_values],
            )

            func.ReturnOp(task.done_writes)

        task_arguments = dict(
            zip(read_arguments + write_arguments, task_block.arguments)
        )

        with InsertionPoint(task_block):
            stream_values_by_id = {}

            for argument in read_arguments:
                bindings = [
                    binding
                    for binding in program.input_ports
                    if binding.input_src.source is argument
                ]

                if not bindings:
                    continue

                stream_read = taskflow.TaskflowStreamReadOp(
                    [get_mlir_type(binding.input_des.dtype) for binding in bindings],
                    task_arguments[argument],
                    ArrayAttr.get(
                        [get_access_map(binding.input_src) for binding in bindings]
                    ),
                )

                for binding, value in zip(bindings, stream_read.values):
                    stream_values_by_id[binding.input_des.id] = value

            kernel_input_values = [
                stream_values_by_id[binding.input_des.id]
                for binding in program.input_ports
            ]

            kernel_input_types = [value.type for value in kernel_input_values]

            if program.stationary is not None:
                stationary_value = task_arguments[program.stationary.source]

                kernel_input_values.append(stationary_value)

                kernel_input_types.append(stationary_value.type)

            kernel_output_types = [
                get_mlir_type(binding.output_src.dtype)
                for binding in program.output_ports
            ]

            kernel = neura.KernelOp(
                outputs=kernel_output_types,
                inputs=kernel_input_values,
                iter_args_init=[],
                accelerator=StringAttr.get("neura"),
                kernel_metadata=get_kernel_metadata(),
            )

            kernel_block = kernel.body.blocks.append(*kernel_input_types)

            if program.output_ports:
                taskflow.TaskflowStreamWriteOp(
                    list(kernel.results),
                    task_arguments[write_arguments[0]],
                    ArrayAttr.get(
                        [
                            get_access_map(binding.output_des)
                            for binding in program.output_ports
                        ]
                    ),
                )

            taskflow.TaskflowYieldOp(
                done_reads=[],
                done_writes=[task_arguments[argument] for argument in write_arguments],
                value_results=[],
            )

        values_by_id = {
            binding.input_des.id: kernel_block.arguments[index]
            for index, binding in enumerate(program.input_ports)
        }

        with InsertionPoint(kernel_block):
            for operation in program.operations:
                result_types = tuple(
                    get_mlir_type(result.dtype) for result in operation.results
                )

                mlir_operands = tuple(
                    values_by_id[operand.id] for operand in operation.operands
                )

                mlir_operation = lower_operation(operation, mlir_operands, result_types)

                mlir_operation.operation.attributes["placement"] = get_placement(
                    operation.tile
                )

                mlir_results = tuple(mlir_operation.results)

                if len(mlir_results) != len(operation.results):
                    raise RuntimeError(
                        "frontend and MLIR operation result counts differ"
                    )

                for frontend_result, mlir_result in zip(
                    operation.results, mlir_results
                ):
                    values_by_id[frontend_result.id] = mlir_result

            neura.YieldOp(
                iter_args_next=[],
                results_=[
                    values_by_id[binding.output_src.id]
                    for binding in program.output_ports
                ],
            )

        if not module.operation.verify():
            raise RuntimeError("generated Taskflow/Neura module is invalid")

        return str(module)
