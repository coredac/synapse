"""Applies user-defined replacement patterns to MLIR modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from taskflow_mlir.ir import (
    BlockArgument,
    F32Type,
    InsertionPoint,
    IntegerType,
    MemRefType,
    Module,
    OpView,
    Value,
)

from synapse.language.types import BufferType, DType
from synapse.patterns import TileArrayProgramReplacePattern
from synapse.patterns.gemm_pattern import (
    AffineGemmPattern,
    LinalgGemmPattern,
    LinalgGenericGemmPattern,
)

from .pattern_rewrite import PatternRewriter

_DEFAULT_REPLACE_PATTERNS: tuple[type[TileArrayProgramReplacePattern], ...] = (
    LinalgGemmPattern,
    LinalgGenericGemmPattern,
    AffineGemmPattern,
)


class PatternReplacer(PatternRewriter):
    """Adds TileArray implementation replacement to the shared IR mutations."""

    def replace_with_tile_array_program(
        self,
        operation: OpView,
        *,
        program: Callable,
        arguments: tuple[Value, ...],
    ) -> bool:
        """Replaces a compatible buffer computation with a TileArray kernel.

        The pattern establishes applicability and semantic preconditions. The
        shared path adapts argument types and validates the generated kernel
        before changing the source IR.
        """
        from taskflow_mlir.dialects import func

        from synapse.frontend.lowering import (
            TileArrayProgramLowering,
            build_tile_array_program,
        )

        if operation.operation != self._root.operation:
            raise ValueError("TileArray replacement requires the pattern root")
        buffer_types = _task_buffer_types(operation, arguments)
        if buffer_types is None:
            return False

        tile_program = build_tile_array_program(program, argument_types=buffer_types)
        lowering = TileArrayProgramLowering(tile_program)

        staged = Module.create()
        with InsertionPoint(staged.body):
            function = func.FuncOp(
                "replacement", ([value.type for value in arguments], [])
            )
            block = function.add_entry_block()
        with InsertionPoint(block):
            lowering.lower_to_kernel(dict(zip(tile_program.arguments, block.arguments)))
            func.ReturnOp([])
        if not staged.operation.verify():
            raise ValueError("replacement kernel failed verification")

        kernel = block.operations[0]
        for index, value in enumerate(arguments):
            kernel.operation.operands[index] = value
        kernel.operation.move_before(operation.operation)
        operation.erase()
        return True


def apply_replacements(
    module: Module,
    patterns: Sequence[type[TileArrayProgramReplacePattern]] | None = None,
) -> int:
    """Tries built-in replacements before user patterns at each operation."""

    return _apply_replacements(
        module.operation.opview,
        (*_DEFAULT_REPLACE_PATTERNS, *(patterns or ())),
    )


def _apply_replacements(
    operation: OpView,
    patterns: Sequence[type[TileArrayProgramReplacePattern]],
) -> int:
    """Applies patterns recursively, stopping below a replaced operation."""

    for pattern in patterns:
        if not isinstance(operation, pattern.root):
            continue

        replacer = PatternReplacer(operation)

        if pattern.match_and_replace(operation, replacer):
            return 1

    replace_count = 0

    for region in operation.regions:
        for block in region.blocks:
            for nested_operation in tuple(block.operations):
                replace_count += _apply_replacements(
                    nested_operation,
                    patterns,
                )

    return replace_count


def _task_buffer_types(operation, arguments):
    """Returns supported buffer types, or None when the task boundary is unsuitable."""
    parent = operation.operation.parent
    if parent is None or parent.name != "taskflow.task" or len(operation.results):
        return None
    block = parent.regions[0].blocks[0]
    buffer_types = []
    for value in arguments:
        if (
            not BlockArgument.isinstance(value)
            or BlockArgument(value).owner != block
            or not MemRefType.isinstance(value.type)
        ):
            return None
        memref = MemRefType(value.type)
        if not memref.has_static_shape or any(size <= 0 for size in memref.shape):
            return None
        if memref.element_type == IntegerType.get_signless(32):
            dtype = DType.I32
        elif memref.element_type == F32Type.get():
            dtype = DType.F32
        else:
            return None
        if memref != MemRefType.get(list(memref.shape), memref.element_type):
            return None
        buffer_types.append(BufferType(tuple(memref.shape), dtype))
    return tuple(buffer_types)
