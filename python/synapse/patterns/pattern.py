"""Public API for Synapse replacement patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from taskflow_mlir.ir import OpView

    from synapse.compiler.pattern_replacement import PatternReplacer


class TileArrayProgramPattern(ABC):
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
