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

from synapse.language.types import DType, TensorType
from synapse.patterns import TileArrayProgramPattern


class PatternReplacer:
    """Mutates IR on behalf of one successfully matched pattern."""

    def __init__(self, root: OpView):
        self._root = root

    @property
    def ip(self) -> InsertionPoint:
        """Returns an insertion point immediately before the pattern root."""

        return InsertionPoint(self._root)

    def erase_op(self, operation: OpView) -> None:
        """Erases an operation that has no live results."""

        for index in range(len(operation.results)):
            if any(operation.results[index].uses):
                raise ValueError("cannot erase an operation with live results")

        operation.erase()

    def replace_op(self, operation: OpView, replacement: OpView) -> None:
        """Replaces an operation and redirects its SSA results."""

        old_results = tuple(
            operation.results[index] for index in range(len(operation.results))
        )
        new_results = tuple(
            replacement.results[index] for index in range(len(replacement.results))
        )

        if len(old_results) != len(new_results):
            raise ValueError("replacement must produce the same number of results")

        if any(old.type != new.type for old, new in zip(old_results, new_results)):
            raise TypeError("replacement must preserve result types")

        for old_result, new_result in zip(
            old_results,
            new_results,
        ):
            old_result.replace_all_uses_with(new_result)

        operation.erase()

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
        inferred_types = _task_argument_types(operation, arguments)
        if inferred_types is None:
            return False

        tile_program = build_tile_array_program(program, argument_types=inferred_types)
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


def apply_patterns(
    module: Module,
    patterns: Sequence[type[TileArrayProgramPattern]],
) -> int:
    """Walks a module and applies the first matching pattern at each operation."""

    return _apply_patterns(
        module.operation.opview,
        patterns,
    )


def _apply_patterns(
    operation: OpView,
    patterns: Sequence[type[TileArrayProgramPattern]],
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
                replace_count += _apply_patterns(
                    nested_operation,
                    patterns,
                )

    return replace_count


def _task_argument_types(operation, arguments):
    """Returns supported capture types, or None when the task boundary is unsuitable."""
    parent = operation.operation.parent
    if parent is None or parent.name != "taskflow.task" or len(operation.results):
        return None
    block = parent.regions[0].blocks[0]
    types = []
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
        types.append(TensorType(tuple(memref.shape), dtype))
    return tuple(types)
