"""Tensor values used by the Synapse programming language."""

from __future__ import annotations

from dataclasses import dataclass

from .types import DType, TensorType

TensorIndex = int | slice


@dataclass(frozen=True)
class Tensor:
    """A symbolic tensor value passed to a Synapse program."""

    name: str
    type: TensorType

    def __getitem__(
        self, indices: TensorIndex | tuple[TensorIndex, ...]
    ) -> TensorAccess:
        """Describes a scalar element or full-dimensional slice of this tensor."""
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.type.shape):
            raise IndexError(
                f"expected {len(self.type.shape)} indices, but got {len(indices)}"
            )

        for dimension, index in zip(self.type.shape, indices):
            if type(index) is int:
                if not (0 <= index < dimension):
                    raise IndexError(
                        f"index {index} is outside dimension size {dimension}"
                    )
                continue

            if isinstance(index, slice):
                if index != slice(None):
                    raise ValueError("only full slices are supported initially")
                continue

            raise TypeError("indices must be integers or full slices")

        return TensorAccess(source=self, indices=indices)


@dataclass(frozen=True)
class TensorAccess:
    """A symbolic element or slice selected from a Tensor."""

    source: Tensor
    indices: tuple[TensorIndex, ...]

    @property
    def dtype(self) -> DType:
        """Returns the scalar element type of the tensor access."""
        return self.source.type.dtype

    @property
    def is_scalar(self) -> bool:
        """Returns whether this access identifies one scalar element."""

        return all(type(index) is int for index in self.indices)
