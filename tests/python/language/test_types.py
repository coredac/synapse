import synapse.language as synl
from synapse.language.types import DType, ShapedType


def test_dtypes_create_shaped_types():
    matrix = synl.i32[4, 4]
    vector = synl.f32[8]

    assert synl.i32 == DType.I32
    assert synl.f32 == DType.F32

    assert matrix == ShapedType(
        shape=(4, 4),
        dtype=synl.i32,
    )

    assert vector == ShapedType(
        shape=(8,),
        dtype=synl.f32,
    )
