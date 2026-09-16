"""Tests for the Synapse MLIR pattern driver."""

from typing import cast

import pytest
import synapse
import synapse.language as synl
from synapse.compiler.pattern_rewriter import (
    PatternRewriter,
    apply_patterns,
)
from synapse.patterns import TileArrayRewritePattern
from taskflow_mlir.dialects import arith, func, linalg, neura, taskflow
from taskflow_mlir.ir import Context, Location, Module


class AddToMultiplyPattern(TileArrayRewritePattern):
    """Replaces integer addition to verify the rewrite mechanism."""

    root = arith.AddIOp

    @classmethod
    def match_and_rewrite(
        cls,
        operation: arith.AddIOp,
        rewriter: PatternRewriter,
    ) -> bool:
        """Replaces arith.addi with arith.muli."""

        with rewriter.ip:
            replacement = arith.MulIOp(
                operation.lhs,
                operation.rhs,
            )

        rewriter.replace_op(
            operation,
            replacement,
        )

        return True


def test_applies_pattern_to_matching_operation():
    source = """
    module {
      func.func @compute(%lhs: i32, %rhs: i32) -> i32 {
        %result = arith.addi %lhs, %rhs : i32
        return %result : i32
      }
    }
    """

    with Context(), Location.unknown():
        module = Module.parse(source)

        rewrite_count = apply_patterns(
            module,
            patterns=[AddToMultiplyPattern],
        )

        assert module.operation.verify()
        rewritten = str(module)

    assert rewrite_count == 1
    assert "arith.addi" not in rewritten
    assert "arith.muli" in rewritten


def copy_program(A: synl.Tensor, C: synl.Tensor):
    """Copies a tensor through configured loads and stores."""
    array = synl.TileArray(4, 4)
    for x in range(1, A.type.shape[1] + 1):
        value = synl.load(A[:, x - 1], tile=array[0, x])
        synl.store(value, target=C[:, x - 1], tile=array[x, 0])


class CopyPattern(TileArrayRewritePattern):
    """Checks and replaces a copy in one user-defined callback."""

    root = linalg.CopyOp

    @classmethod
    def match_and_rewrite(cls, operation, rewriter) -> bool:
        """Checks the buffer types and inserts the copy implementation."""
        operation = cast(linalg.CopyOp, operation)
        (source,) = operation.inputs
        (target,) = operation.outputs
        if source.type != target.type:
            return False
        return rewriter.replace_with_tile_array(
            operation,
            program=copy_program,
            arguments=(source, target),
        )


def _copy_source():
    """Provides a bufferized copy with an independently allocated destination."""
    return """
    module {
      func.func @copy(%A: memref<3x3xi32>) -> memref<3x3xi32> {
        %C = memref.alloc() : memref<3x3xi32>
        %done = taskflow.task @copy
          will_reads(%A : memref<3x3xi32>)
          will_writes(%C : memref<3x3xi32>)
          [original_read_memrefs(%A : memref<3x3xi32>),
           original_write_memrefs(%C : memref<3x3xi32>)]
          : (memref<3x3xi32>, memref<3x3xi32>) -> memref<3x3xi32> {
        ^bb0(%a: memref<3x3xi32>, %c: memref<3x3xi32>):
          linalg.copy ins(%a : memref<3x3xi32>) outs(%c : memref<3x3xi32>)
          taskflow.yield done_writes(%c : memref<3x3xi32>)
        }
        return %done : memref<3x3xi32>
      }
    }
    """


@pytest.mark.parametrize("dtype", ["i32", "f32"])
def test_custom_pattern_checks_and_rewrites_in_one_callback(dtype):
    source = _copy_source().replace("3x3xi32", f"3x3x{dtype}")
    rewritten = synapse.rewrite(source, patterns=[CopyPattern])
    assert "linalg.copy" not in rewritten
    assert rewritten.count('"neura.load"') == 3
    assert rewritten.count('"neura.store"') == 3


def test_alias_checks_apply_to_custom_rewrites():
    source = (
        _copy_source()
        .replace("%C = memref.alloc() : memref<3x3xi32>", "")
        .replace("%C", "%A")
    )
    assert "neura.kernel" not in synapse.rewrite(source, patterns=[CopyPattern])


def test_root_filter_runs_before_the_pattern_callback():
    class UnrelatedPattern(TileArrayRewritePattern):
        root = arith.AddIOp

        @classmethod
        def match_and_rewrite(cls, operation, rewriter) -> bool:
            """Detects a callback invoked for an unrelated root type."""
            raise AssertionError("the root filter must reject this operation")

    assert "linalg.copy" in synapse.rewrite(_copy_source(), patterns=[UnrelatedPattern])


def test_failed_user_check_preserves_the_original_operation():
    class RejectingPattern(TileArrayRewritePattern):
        root = linalg.CopyOp

        @classmethod
        def match_and_rewrite(cls, operation, rewriter) -> bool:
            """Declines a candidate after the root type has matched."""
            return False

    rewritten = synapse.rewrite(_copy_source(), patterns=[RejectingPattern])
    assert "linalg.copy" in rewritten
    assert "neura.kernel" not in rewritten


def test_invalid_user_implementation_remains_an_error():
    def invalid(A: synl.Tensor, C: synl.Tensor):
        raise ValueError("invalid user program")

    class InvalidPattern(TileArrayRewritePattern):
        root = linalg.CopyOp

        @classmethod
        def match_and_rewrite(cls, operation, rewriter) -> bool:
            """Passes an invalid implementation to the staged lowering path."""
            operation = cast(linalg.CopyOp, operation)
            return rewriter.replace_with_tile_array(
                operation,
                program=invalid,
                arguments=tuple(operation.inputs) + tuple(operation.outputs),
            )

    with pytest.raises(ValueError, match="invalid user program"):
        synapse.rewrite(_copy_source(), patterns=[InvalidPattern])
