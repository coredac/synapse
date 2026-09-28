"""Values exposed by the Synapse programming language."""

from __future__ import annotations

from dataclasses import dataclass

from .types import BufferType, DType, SynapseType


class SynapseValue:
    """Base class for values in the Synapse language."""


@dataclass(frozen=True, eq=False)
class Buffer(SynapseValue):
    """A symbolic buffer value passed to a TileArray program."""

    name: str
    type: BufferType

    def __getitem__(
        self,
        indices: int | slice | tuple[int | slice, ...],
    ) -> BufferSlice:
        """Selects an element or static slice of this buffer."""
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.type.shape):
            raise IndexError(
                f"expected {len(self.type.shape)} indices, but got {len(indices)}"
            )

        shape = []
        for dimension, index in zip(self.type.shape, indices):
            if type(index) is int:
                if not 0 <= index < dimension:
                    raise IndexError(
                        f"index {index} is outside dimension size {dimension}"
                    )
            elif isinstance(index, slice):
                shape.append(len(range(*index.indices(dimension))))
            else:
                raise TypeError("indices must be integers or static slices")

        selected_type: SynapseType
        if shape:
            selected_type = BufferType(tuple(shape), self.type.dtype)
        else:
            selected_type = self.type.dtype

        return BufferSlice(self, indices, selected_type)


@dataclass(frozen=True)
class BufferSlice(SynapseValue):
    """A static selection of one buffer value."""

    source: Buffer
    indices: tuple[int | slice, ...]
    type: SynapseType

    @property
    def dtype(self) -> DType:
        """Returns the selected buffer element type."""
        return self.source.type.dtype

    @property
    def is_scalar(self) -> bool:
        """Reports whether the selection identifies one element."""
        return isinstance(self.type, DType)
