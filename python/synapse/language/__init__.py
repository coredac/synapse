"""Public Synapse language API."""

from .spatial import TileArray
from .tile_array_program import TileArrayScalarType, add, constant

i32 = TileArrayScalarType.I32
f32 = TileArrayScalarType.F32

__all__ = [
    "TileArray",
    "TileArrayScalarType",
    "add",
    "constant",
    "f32",
    "i32",
]
