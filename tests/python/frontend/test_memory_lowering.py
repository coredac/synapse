"""Checks configured offset semantics and explicit address lowering."""

import pytest
import synapse.language as synl
from synapse.frontend.lowering import (
    TileArrayProgramLowering,
    build_tile_array_program,
    lower,
)
from synapse.language.tile_array_program import LoadOp
from taskflow_mlir.dialects import neura, taskflow
from taskflow_mlir.ir import Context, DenseI64ArrayAttr, Location, Module


def configured_accesses(A: synl.Tensor):
    array = synl.TileArray(4, 4)
    synl.load(A[1, 2], tile=array[0, 1])
    synl.load(A[:, ::2], tile=array[0, 2])
    synl.load(A[::-1, 1], tile=array[0, 3])


def dynamic_accesses():
    array = synl.TileArray(4, 4)
    addr = synl.constant(16, tile=array[1, 1])
    value = synl.load(addr=addr, dtype=synl.i32, tile=array[0, 1])
    synl.store(value, addr=addr, tile=array[1, 0])


def test_static_offsets_cover_scalar_strided_and_multidimensional_accesses():
    program = build_tile_array_program(
        configured_accesses, argument_types=(synl.i32[3, 4],)
    )
    with Context(), Location.unknown():
        lowering = TileArrayProgramLowering(program)
        offsets = []
        for operation in program.operations:
            assert isinstance(operation, LoadOp)
            assert operation.source is not None
            offsets.append(tuple(lowering.get_memory_offsets(operation.source)))
    assert offsets == [(6,), (0, 2, 4, 6, 8, 10), (9, 5, 1)]


def test_dynamic_ir_uses_address_operands_without_memory_configuration():
    source = lower(dynamic_accesses)
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        task = module.body.operations[0].regions[0].blocks[0].operations[0]
        kernel = task.regions[0].blocks[0].operations[0]
        constant, read, write, _ = tuple(kernel.regions[0].blocks[0].operations)
        assert tuple(read.operands[index] for index in range(len(read.operands))) == (
            constant.results[0],
        )
        assert tuple(write.operands[index] for index in range(len(write.operands))) == (
            read.results[0],
            constant.results[0],
        )
        assert len(kernel.results) == 0
        assert module.operation.verify()
    assert "memory_access" not in source


def test_empty_configured_queue_is_rejected():
    def empty(A: synl.Tensor):
        synl.load(A[0:0, 0], tile=synl.TileArray(4, 4)[0, 1])

    with pytest.raises(ValueError, match="cannot be empty"):
        lower(empty, argument_types=(synl.i32[3, 3],))


def test_dynamic_addresses_reach_backend_mapping():
    import synapse

    mapped = synapse.compile(dynamic_accesses, target="neura")
    assert '"neura.load"' in mapped
    assert '"neura.store"' in mapped
    assert "mapping_locs" in mapped
    assert "memory_access" not in mapped


def test_memory_bases_survive_task_argument_grouping():
    def program(A: synl.Tensor, unused: synl.Tensor, C: synl.Tensor):
        array = synl.TileArray(4, 4)
        value = synl.load(A[:, 0], tile=array[0, 1])
        synl.store(value, target=C[:, 0], tile=array[1, 0])

    source = lower(program, argument_types=(synl.i32[3, 3],) * 3)
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        task = module.body.operations[0].regions[0].blocks[0].operations[0]
        block = task.regions[0].blocks[0]
        kernel = block.operations[0]
        assert isinstance(kernel, neura.KernelOp)
        assert tuple(
            kernel.operands[index] for index in range(len(kernel.operands))
        ) == (
            block.arguments[0],
            block.arguments[2],
            block.arguments[1],
        )
        kernel_block = kernel.regions[0].blocks[0]
        read, store, _ = tuple(kernel_block.operations)
        assert tuple(read.operands[index] for index in range(len(read.operands))) == (
            kernel_block.arguments[0],
        )
        assert tuple(store.operands[index] for index in range(len(store.operands))) == (
            read.results[0],
            kernel_block.arguments[2],
        )
        assert tuple(DenseI64ArrayAttr(read.operation.attributes["constants"])) == (
            0,
            3,
            6,
        )
        assert tuple(DenseI64ArrayAttr(store.operation.attributes["constants"])) == (
            0,
            3,
            6,
        )
        assert "memory_access" not in str(kernel)
        assert module.operation.verify()
