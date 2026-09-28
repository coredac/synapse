import synapse.language as synl
from synapse.language.values import Buffer, SynapseValue


def test_buffer_value_has_a_separate_type():
    buffer_type = synl.BufferType(shape=(3, 4), dtype=synl.i32)
    buffer = Buffer(name="A", type=buffer_type)

    assert isinstance(buffer, SynapseValue)
    assert buffer.type is buffer_type
    assert buffer.type.shape == (3, 4)
    assert buffer.type.dtype == synl.i32


def test_buffer_subscripts_produce_typed_slices():
    buffer = Buffer("A", synl.i32[3, 4])

    scalar = buffer[1, 2]
    column = buffer[:, 1]

    assert scalar.source is buffer
    assert scalar.indices == (1, 2)
    assert scalar.type == synl.i32
    assert scalar.is_scalar

    assert column.source is buffer
    assert column.indices == (slice(None), 1)
    assert column.type == synl.i32[3]
