"""Lower Synapse Python programs to compiler input IR."""

from collections.abc import Callable
from functools import singledispatch
from typing import cast

from synapse.language.spatial import Tile
from synapse.language.tile_array_program import (
    AddOp,
    ConstantOp,
    TileArrayBuilder,
    TileArrayOp,
    TileArrayProgram,
)
from synapse.language.types import DType


def lower(program_fn: Callable) -> str:
    """Lower one tile-array program to pre-mapping Taskflow and Neura IR."""

    builder = TileArrayBuilder()

    # Execute the user's tile-array DSL while recording its operations.
    with builder:
        program_fn()

    program = builder.build()

    return _lower_tile_array_program(
        program_name=program_fn.__name__,
        program=program,
    )


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
        Context,
        DictAttr,
        F32Type,
        FloatAttr,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        Location,
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

        def get_constant_attribute(
            operation: ConstantOp,
            result_type,
        ):
            """Build the typed MLIR attribute for a constant value."""

            if operation.result.dtype == DType.I32:
                return IntegerAttr.get(result_type, cast(int, operation.value))

            if operation.result.dtype == DType.F32:
                return FloatAttr.get(result_type, float(operation.value))

            raise NotImplementedError(
                f"unsupported constant type: {operation.result.dtype.value}"
            )

        def get_placement(tile: Tile) -> DictAttr:
            """Build Neura placement directly from a Tile coordinate."""

            return DictAttr.get(
                {
                    "x": IntegerAttr.get(i32_type, tile.x),
                    "y": IntegerAttr.get(i32_type, tile.y),
                }
            )

        @singledispatch
        def lower_operation(operation: TileArrayOp, operands, result_type):
            """Lower one frontend TileArray operation to a Neura operation.

            The caller handles common lowering such as resolving operands,
            attaching placement, and recording the resulting SSA value.
            """
            raise NotImplementedError(
                f"unsupported tile-array operation: {type(operation).__name__}"
            )

        @lower_operation.register
        def lower_constant(operation: ConstantOp, operands, result_type):
            """Lower a ConstantOp to neura.constant."""
            return neura.ConstantOp(
                result_type, get_constant_attribute(operation, result_type)
            )

        @lower_operation.register
        def lower_add(operation: AddOp, operands, result_type):
            """Lower a frontend AddOp to neura.add."""
            lhs, rhs = operands

            return neura.AddOp(result_type, lhs, rhs=rhs)

        module = Module.create()

        # This milestone lowers one Python function into one task containing
        # one manually placed Neura kernel.
        with InsertionPoint(module.body):
            function = func.FuncOp(program_name, ([], []))
            function_block = function.add_entry_block()

        with InsertionPoint(function_block):
            task = taskflow.TaskflowTaskOp(
                done_reads=[],
                done_writes=[],
                value_outputs=[],
                will_reads=[],
                will_writes=[],
                value_inputs=[],
                task_name=program_name,
                original_read_memrefs=[],
                original_write_memrefs=[],
            )
            task_block = task.body.blocks.append()

            func.ReturnOp([])

        with InsertionPoint(task_block):
            kernel = neura.KernelOp(
                outputs=[],
                inputs=[],
                iter_args_init=[],
                accelerator=StringAttr.get("neura"),
            )
            kernel_block = kernel.body.blocks.append()

            taskflow.TaskflowYieldOp(
                done_reads=[],
                done_writes=[],
                value_results=[],
            )

        # Map frontend value IDs to the MLIR SSA values produced while
        # lowering the recorded operations.
        values_by_id = {}

        with InsertionPoint(kernel_block):
            for operation in program.operations:
                result_type = get_mlir_type(operation.result.dtype)

                mlir_operands = tuple(
                    values_by_id[operand.id] for operand in operation.operands
                )

                mlir_operation = lower_operation(operation, mlir_operands, result_type)

                mlir_operation.operation.attributes["placement"] = get_placement(
                    operation.tile
                )

                values_by_id[operation.result.id] = mlir_operation.result

            neura.YieldOp(
                iter_args_next=[],
                results_=[],
            )

        if not module.operation.verify():
            raise RuntimeError("generated Taskflow/Neura module is invalid")

        return str(module)
