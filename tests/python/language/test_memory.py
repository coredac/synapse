"""Checks general memory operations independently of Neura bindings."""

import pytest
import synapse.language as synl
from synapse.frontend.lowering import build_tile_array_program
from synapse.language.tile_array_program import LoadOp, StoreOp, TileArrayBuilder
from synapse.library import ws_gemm_3x3


def test_dynamic_addresses_are_explicit_operands():
    array = synl.TileArray(4, 4)
    builder = TileArrayBuilder()
    with builder:
        addr = synl.constant(16, tile=array[1, 1])
        value = synl.load(addr=addr, dtype=synl.f32, tile=array[0, 1])
        synl.store(value, addr=addr, tile=array[1, 0])
    _, read, write = builder.build().operations
    assert isinstance(read, LoadOp)
    assert read.addr is addr
    assert read.operands == (addr,)
    assert read.source is None
    assert isinstance(write, StoreOp)
    assert write.operands == (value, addr)
    assert write.addr is addr
    assert write.results == ()


def test_memory_configuration_retains_static_accesses():
    tensor = synl.Tensor("A", synl.i32[3, 4])
    array = synl.TileArray(4, 4)
    builder = TileArrayBuilder((tensor,))
    with builder:
        value = synl.load(tensor[1, 2], tile=array[0, 1])
        synl.store(value, target=tensor[:, ::2], tile=array[1, 0])
    read, write = builder.build().operations
    assert isinstance(read, LoadOp)
    assert isinstance(write, StoreOp)
    assert read.source is not None
    assert write.target is not None
    assert read.source.indices == (1, 2)
    assert read.operands == ()
    assert write.target.indices == (slice(None), slice(None, None, 2))
    assert write.operands == (value,)


def test_rejected_memory_forms_do_not_consume_value_ids():
    tensor = synl.Tensor("A", synl.i32[3, 3])
    array = synl.TileArray(4, 4)
    builder = TileArrayBuilder((tensor,))
    with builder:
        addr = synl.constant(0, tile=array[1, 1])
        with pytest.raises(ValueError, match="exactly one"):
            synl.load(tensor[:, 0], addr=addr, tile=array[0, 1])
        with pytest.raises(TypeError, match="explicit DType"):
            synl.load(addr=addr, tile=array[0, 1])
        with pytest.raises(TypeError, match="must match"):
            synl.load(tensor[:, 0], dtype=synl.f32, tile=array[0, 1])
        value = synl.load(addr=addr, dtype=synl.i32, tile=array[0, 1])
        assert value.id == 1
        with pytest.raises(ValueError, match="exactly one"):
            synl.store(value, target=tensor[:, 0], addr=addr, tile=array[1, 0])
    assert len(builder.build().operations) == 2


def test_gemm_program_records_load_mac_store_network():
    program = build_tile_array_program(
        ws_gemm_3x3, argument_types=(synl.i32[3, 3],) * 3
    )
    assert sum(isinstance(op, LoadOp) for op in program.operations) == 3
    assert sum(isinstance(op, StoreOp) for op in program.operations) == 3
    assert len(program.operations) == 15
    assert program.stationary is not None
    assert program.stationary.source is program.arguments[1]
