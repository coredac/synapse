"""Types exposed by the Synapse programming language.

These types describe program values independently of a particular hardware
hierarchy or compiler IR. Compiler lowering later decides whether a shaped
value becomes a tensor, MemRef, stream source, or another backend type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DType(Enum):
    """A data type supported by Synapse."""

    I32 = "i32"
    F32 = "f32"

    def __getitem__(self, shape: int | tuple[int, ...]) -> ShapedType:
        """Create a shaped type with this data element type."""

        if not isinstance(shape, tuple):
            shape = (shape,)

        return ShapedType(
            shape=shape,
            dtype=self,
        )

    def __str__(self) -> str:
        """Return the source-level spelling of this type."""

        return self.value


@dataclass(frozen=True)
class ShapedType:
    """The shape and scalar element type of a Synapse program value.

    ShapedType does not prescribe whether compiler lowering uses a tensor,
    MemRef, or another backend representation.
    """

    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        """Validate the dimensions and scalar element type."""

        if not self.shape:
            raise TypeError("a shaped type requires at least one dimension")

        if not all(
            type(dimension) is int and dimension > 0 for dimension in self.shape
        ):
            raise TypeError("shape dimensions must be positive integers")

        if not isinstance(self.dtype, DType):
            raise TypeError("dtype must be a DType")


i32 = DType.I32
f32 = DType.F32
