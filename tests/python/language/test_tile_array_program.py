import synapse.language as synl


def test_records_tile_array_program():
    array = synl.TileArray(4, 4)
    builder = synl.tile_array_program.TileArrayBuilder()

    with builder:
        lhs = synl.constant(1, tile=array[0, 0])
        rhs = synl.constant(2, tile=array[0, 2])
        result = synl.add(lhs, rhs, tile=array[0, 1])

    program = builder.build()
    lhs_op, rhs_op, add_op = program.operations

    assert program.array is array
    assert isinstance(lhs_op, synl.tile_array_program.ConstantOp)
    assert isinstance(rhs_op, synl.tile_array_program.ConstantOp)
    assert isinstance(add_op, synl.tile_array_program.AddOp)
    assert [(value.id, value.dtype) for value in (lhs, rhs, result)] == [
        (0, synl.i32),
        (1, synl.i32),
        (2, synl.i32),
    ]
    assert [op.tile for op in program.operations] == [
        array[0, 0],
        array[0, 2],
        array[0, 1],
    ]
    assert (add_op.lhs, add_op.rhs) == (lhs, rhs)


def test_infers_supported_scalar_types():
    array = synl.TileArray(1, 3)
    builder = synl.tile_array_program.TileArrayBuilder()

    with builder:
        integer = synl.constant(1, tile=array[0, 0])
        floating = synl.constant(1.0, tile=array[0, 1])
        explicit_f32 = synl.constant(
            1,
            tile=array[0, 2],
            dtype=synl.f32,
        )

    assert integer.dtype == synl.i32
    assert floating.dtype == synl.f32
    assert explicit_f32.dtype == synl.f32
