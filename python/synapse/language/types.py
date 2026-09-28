"""Types exposed by the Synapse programming language.

These types describe program values independently of a particular hardware
hierarchy or compiler IR. Compiler lowering later decides whether a shaped
value becomes a tensor, MemRef, stream source, or another backend type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SynapseType:
    """Base class for types in the Synapse language."""


class DType(SynapseType, Enum):
    """A data type supported by Synapse."""

    I32 = "i32"
    F32 = "f32"

    def __getitem__(self, shape: int | tuple[int, ...]) -> BufferType:
        """Create a buffer type with this data element type."""

        if not isinstance(shape, tuple):
            shape = (shape,)

        return BufferType(shape=shape, dtype=self)

    def __str__(self) -> str:
        """Return the source-level spelling of this type."""

        return self.value


@dataclass(frozen=True)
class BufferType(SynapseType):
    """The shape and element type of a buffer."""

    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        """Validates the dimensions and element type."""

        if not self.shape:
            raise TypeError("a buffer type requires at least one dimension")

        if not all(
            type(dimension) is int and dimension > 0 for dimension in self.shape
        ):
            raise TypeError("buffer dimensions must be positive integers")

        if not isinstance(self.dtype, DType):
            raise TypeError("dtype must be a DType")


i32 = DType.I32
f32 = DType.F32
