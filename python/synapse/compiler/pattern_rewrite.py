"""Runs computation IR rewrite passes to a fixed point."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from taskflow_mlir.ir import InsertionPoint, Module, OpView, Value

from synapse.patterns import IRRewritePattern

_DEFAULT_REWRITE_PATTERNS: tuple[type[IRRewritePattern], ...] = ()


class PatternRewriter:
    """Provides shared IR mutation operations for a matched pattern."""

    def __init__(
        self, root: OpView, *, on_erase: Callable[[OpView], None] | None = None
    ):
        self._root = root
        self._on_erase = on_erase

    @property
    def ip(self) -> InsertionPoint:
        """Returns an insertion point immediately before the pattern root."""
        return InsertionPoint(self._root)

    def erase_op(self, operation: OpView) -> None:
        """Erases an operation that has no live results."""
        for index in range(len(operation.results)):
            if any(operation.results[index].uses):
                raise ValueError("cannot erase an operation with live results")
        if self._on_erase is not None:
            self._on_erase(operation)
        operation.erase()

    def replace_op(self, operation: OpView, replacement: OpView) -> None:
        """Replaces an operation and redirects its SSA results."""
        self.replace_op_with_values(
            operation,
            tuple(
                replacement.results[index] for index in range(len(replacement.results))
            ),
        )

    def replace_op_with_values(
        self, operation: OpView, replacements: Sequence[Value]
    ) -> None:
        """Redirects an operation's results to existing SSA values and erases it."""
        results = tuple(
            operation.results[index] for index in range(len(operation.results))
        )
        if len(results) != len(replacements):
            raise ValueError("replacement must produce the same number of results")
        if any(old.type != new.type for old, new in zip(results, replacements)):
            raise TypeError("replacement must preserve result types")
        for old, new in zip(results, replacements):
            old.replace_all_uses_with(new)
        self.erase_op(operation)


def apply_rewrites(
    scope: Module | OpView,
    patterns: Sequence[type[IRRewritePattern]] = (),
    *,
    max_iterations: int = 1000,
) -> int:
    """Runs built-in passes followed by user passes until a round makes no changes."""
    root = scope.operation.opview if isinstance(scope, Module) else scope
    rewrite_patterns = (*_DEFAULT_REWRITE_PATTERNS, *patterns)
    rewrite_count = 0
    for _ in range(max_iterations):
        round_count = 0
        for pattern in rewrite_patterns:
            round_count += _apply_pattern_pass(root, pattern)
        rewrite_count += round_count
        # The fixed-point is reached when a complete round of passes makes no changes.
        if round_count == 0:
            return rewrite_count
    raise RuntimeError(
        f"IR rewrites did not converge within {max_iterations} iterations"
    )


def _apply_pattern_pass(scope: OpView, pattern: type[IRRewritePattern]) -> int:
    """Applies one rule to all surviving operations in a parent-first snapshot."""
    # Reverse insertion preserves parent-first traversal when popitem removes entries.
    pending = {
        operation.operation: operation
        for operation in reversed(tuple(_walk_operations(scope)))
    }

    def forget_erased(operation: OpView) -> None:
        """Removes the erased operation and its descendants from the pending pass."""
        pending.pop(operation.operation, None)
        for nested_operation in _walk_operations(operation):
            pending.pop(nested_operation.operation, None)

    rewrite_count = 0
    while pending:
        _, operation = pending.popitem()
        if isinstance(operation, pattern.root):
            rewriter = PatternRewriter(operation, on_erase=forget_erased)
            if pattern.match_and_rewrite(operation, rewriter):
                rewrite_count += 1
    return rewrite_count


def _walk_operations(scope: OpView) -> Iterator[OpView]:
    """Visits contained operations in parent-first order."""
    for region in scope.regions:
        for block in region.blocks:
            for operation in block.operations:
                yield operation
                yield from _walk_operations(operation)
