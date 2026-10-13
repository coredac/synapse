"""Public interfaces for IR rewrites and TileArray implementation replacement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from taskflow_mlir.ir import OpView

    from synapse.compiler.pattern_replace import PatternReplacer
    from synapse.compiler.pattern_rewrite import PatternRewriter


class IRRewritePattern(ABC):
    """Rewrites computation IR without selecting a TileArray implementation."""

    root: ClassVar[type[OpView] | tuple[type[OpView], ...]]

    @classmethod
    @abstractmethod
    def match_and_rewrite(
        cls,
        operation: OpView,
        rewriter: PatternRewriter,
    ) -> bool:
        """Returns True after a rewrite; a False result leaves the IR unchanged."""
        raise NotImplementedError


class TileArrayProgramReplacePattern(ABC):
    """Matches source IR and replaces it with a TileArray program."""

    root: ClassVar[type[OpView] | tuple[type[OpView], ...]]

    @classmethod
    @abstractmethod
    def match_and_replace(
        cls,
        operation: OpView,
        replacer: PatternReplacer,
    ) -> bool:
        """Matches one root operation and replaces it when supported."""

        raise NotImplementedError
