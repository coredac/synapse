"""Public API for Synapse rewrite patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from taskflow_mlir.ir import OpView

    from synapse.compiler.pattern_rewriter import PatternRewriter


class TileArrayRewritePattern(ABC):
    """Matches source IR and rewrites it into a TileArray implementation."""

    root: ClassVar[type[OpView] | tuple[type[OpView], ...]]

    @classmethod
    @abstractmethod
    def match_and_rewrite(
        cls,
        operation: OpView,
        rewriter: PatternRewriter,
    ) -> bool:
        """Matches one root operation and rewrites it when supported."""

        raise NotImplementedError
