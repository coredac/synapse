"""Checks GEMM matching, semantic rejection, and task-preserving replacement."""

import pytest
import synapse
from synapse.compiler.pattern_rewriter import apply_patterns
from synapse.patterns.gemm_pattern import (
    AffineGemmPattern,
    LinalgGemmPattern,
    LinalgGenericGemmPattern,
)
from taskflow_mlir.dialects import linalg, neura, taskflow
from taskflow_mlir.ir import Context, Location, Module

LINALG_GEMM = """
  linalg.matmul ins(%a, %b : memref<3x3xi32>, memref<3x3xi32>)
                outs(%c : memref<3x3xi32>)
"""

GENERIC_GEMM = """
  linalg.generic {
    indexing_maps = [affine_map<(i,j,k)->(i,k)>,
                     affine_map<(i,j,k)->(k,j)>,
                     affine_map<(i,j,k)->(i,j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%a, %b : memref<3x3xi32>, memref<3x3xi32>)
    outs(%c : memref<3x3xi32>) {
  ^bb0(%lhs: i32, %rhs: i32, %acc: i32):
    %product = arith.muli %lhs, %rhs : i32
    %sum = arith.addi %product, %acc : i32
    linalg.yield %sum : i32
  }
"""

AFFINE_GEMM = """
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 3 {
        %lhs = affine.load %a[%i, %k] : memref<3x3xi32>
        %rhs = affine.load %b[%k, %j] : memref<3x3xi32>
        %acc = affine.load %c[%i, %j] : memref<3x3xi32>
        %product = arith.muli %lhs, %rhs : i32
        %sum = arith.addi %acc, %product : i32
        affine.store %sum, %c[%i, %j] : memref<3x3xi32>
      }
    }
  }
"""


def task_source(body=LINALG_GEMM):
    """Wraps a semantic kernel with an independently allocated output buffer."""
    return """
module {
  func.func @gemm(%A: memref<3x3xi32>, %B: memref<3x3xi32>) -> memref<3x3xi32> {
    %C = memref.alloc() : memref<3x3xi32>
    %done = taskflow.task @gemm
      will_reads(%A, %B : memref<3x3xi32>, memref<3x3xi32>)
      will_writes(%C : memref<3x3xi32>)
      [original_read_memrefs(%A, %B : memref<3x3xi32>, memref<3x3xi32>),
       original_write_memrefs(%C : memref<3x3xi32>)]
      : (memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) -> memref<3x3xi32> {
    ^bb0(%a: memref<3x3xi32>, %b: memref<3x3xi32>, %c: memref<3x3xi32>):
      %zero = arith.constant 0 : i32
      linalg.fill ins(%zero : i32) outs(%c : memref<3x3xi32>)
      BODY
      taskflow.yield done_writes(%c : memref<3x3xi32>)
    }
    return %done : memref<3x3xi32>
  }
}
""".replace("BODY", body)


@pytest.mark.parametrize(
    "body,pattern",
    [
        (LINALG_GEMM, LinalgGemmPattern),
        (GENERIC_GEMM, LinalgGenericGemmPattern),
        (AFFINE_GEMM, AffineGemmPattern),
    ],
)
def test_rewrites_gemm_inside_existing_task(body, pattern):
    import synapse.language as synl
    from synapse.frontend.lowering import lower
    from synapse.library import ws_gemm_3x3

    direct_source = lower(ws_gemm_3x3, argument_types=(synl.i32[3, 3],) * 3)
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(task_source(body))
        function = module.body.operations[0]
        task = function.regions[0].blocks[0].operations[1]
        block = task.regions[0].blocks[0]
        terminator = tuple(block.operations)[-1]
        result = task.results[0]
        assert apply_patterns(module, [pattern]) == 1
        assert module.operation.verify()
        assert task.results[0] == result
        assert tuple(block.operations)[-1] == terminator
        kernel = next(
            op for op in block.operations if op.operation.name == "neura.kernel"
        )
        assert isinstance(kernel, neura.KernelOp)
        assert len(kernel.results) == 0
        assert tuple(
            kernel.operands[index] for index in range(len(kernel.operands))
        ) == tuple(block.arguments[index] for index in range(len(block.arguments)))
        direct = Module.parse(direct_source)
        direct_task = direct.body.operations[0].regions[0].blocks[0].operations[0]
        direct_kernel = direct_task.regions[0].blocks[0].operations[0]
        assert kernel.operation.get_asm(
            use_local_scope=True
        ) == direct_kernel.operation.get_asm(use_local_scope=True)
        assert str(module).count('"neura.load"') == 3
        assert str(module).count('"neura.mac"') == 9
        assert str(module).count('"neura.store"') == 3


@pytest.mark.parametrize(
    "source",
    [
        task_source().replace("constant 0", "constant 1"),
        task_source().replace(
            "linalg.fill ins(%zero : i32) outs(%c : memref<3x3xi32>)", ""
        ),
        task_source().replace("3x3xi32", "2x2xi32"),
        task_source()
        .replace("%C = memref.alloc() : memref<3x3xi32>", "")
        .replace("%C", "%A"),
        task_source(GENERIC_GEMM).replace("arith.muli", "arith.subi"),
        task_source(AFFINE_GEMM).replace("%a[%i, %k]", "%a[%k, %i]"),
        task_source(AFFINE_GEMM).replace("%k = 0 to 3", "%k = 0 to 2"),
    ],
)
def test_rejected_candidates_preserve_original_ir(source):
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        before = str(module)
        assert (
            apply_patterns(
                module, [LinalgGemmPattern, LinalgGenericGemmPattern, AffineGemmPattern]
            )
            == 0
        )
        assert module.operation.verify()
        assert str(module) == before


def test_public_compile_accepts_task_ir():
    mapped = synapse.compile(task_source(), target="neura")
    assert "compiled_ii = 1" in mapped
    assert "linalg.matmul" not in mapped


def test_affine_match_accepts_actual_linalg_lowering():
    import subprocess
    from pathlib import Path

    executable = (
        Path(__file__).resolve().parents[3]
        / "build/amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt"
    )
    lowered = subprocess.run(
        [str(executable), "--convert-linalg-to-affine-loops"],
        input=task_source(),
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    rewritten = synapse.rewrite(lowered, patterns=[AffineGemmPattern])
    assert "neura.kernel" in rewritten
    assert "arith.muli" not in rewritten


def test_failed_tile_array_lowering_preserves_source_module():
    import synapse.language as synl
    from synapse.compiler.pattern_rewriter import PatternRewriter

    def invalid_program(A: synl.Tensor, B: synl.Tensor, C: synl.Tensor):
        synl.load(A[0:0, 0], tile=synl.TileArray(4, 4)[0, 1])

    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(task_source())
        task = module.body.operations[0].regions[0].blocks[0].operations[1]
        root = task.regions[0].blocks[0].operations[2]
        assert isinstance(root, linalg.MatmulOp)
        before = str(module)
        with pytest.raises(ValueError, match="cannot be empty"):
            PatternRewriter(root).replace_with_tile_array(
                root,
                program=invalid_program,
                arguments=tuple(
                    root.operands[index] for index in range(len(root.operands))
                ),
            )
        assert str(module) == before
        assert module.operation.verify()


def test_unknown_output_aliasing_is_not_assumed_safe():
    source = (
        task_source()
        .replace("%B: memref<3x3xi32>)", "%B: memref<3x3xi32>, %C: memref<3x3xi32>)")
        .replace("%C = memref.alloc() : memref<3x3xi32>", "")
    )
    assert "neura.kernel" not in synapse.rewrite(source)


def test_intervening_memory_write_invalidates_zero_initialization():
    body = (
        """
      %one = arith.constant 1 : i32
      %index = arith.constant 0 : index
      memref.store %one, %c[%index, %index] : memref<3x3xi32>
    """
        + LINALG_GEMM
    )
    assert "neura.kernel" not in synapse.rewrite(task_source(body))
