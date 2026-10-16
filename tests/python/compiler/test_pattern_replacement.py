"""Tests for the Synapse MLIR pattern driver."""

from collections.abc import Sequence
from typing import cast

import pytest
import synapse.language as synl
from synapse.compiler.pattern_replace import (
    apply_replacements,
)
from synapse.patterns import TileArrayProgramReplacePattern
from taskflow_mlir.dialects import linalg, neura, taskflow
from taskflow_mlir.ir import Context, Location, Module


def _apply_to_source(
    source: str,
    patterns: Sequence[type[TileArrayProgramReplacePattern]],
) -> str:
    """Applies selected patterns to parsed test IR."""
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        apply_replacements(module, patterns)
        assert module.operation.verify()
        return str(module)


def copy_program(A: synl.Buffer, C: synl.Buffer):
    """Copies a buffer through configured loads and stores."""
    array = synl.TileArray(4, 4)
    for x in range(1, A.type.shape[1] + 1):
        value = synl.load(A[:, x - 1], tile=array[0, x])
        synl.store(value, target=C[:, x - 1], tile=array[x, 0])


class CopyPattern(TileArrayProgramReplacePattern):
    """Checks and replaces a copy in one user-defined callback."""

    root = linalg.CopyOp

    @classmethod
    def match_and_replace(cls, operation, replacer) -> bool:
        """Checks the buffer types and inserts the copy implementation."""
        operation = cast(linalg.CopyOp, operation)
        (source,) = operation.inputs
        (target,) = operation.outputs
        if source.type != target.type:
            return False
        return replacer.replace_with_tile_array_program(
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
def test_custom_pattern_checks_and_replaces_in_one_callback(dtype):
    source = _copy_source().replace("3x3xi32", f"3x3x{dtype}")
    replaced = _apply_to_source(source, [CopyPattern])
    assert "linalg.copy" not in replaced
    assert replaced.count('"neura.load"') == 3
    assert replaced.count('"neura.store"') == 3


def test_root_filter_runs_before_the_pattern_callback():
    class UnrelatedPattern(TileArrayProgramReplacePattern):
        root = linalg.MatmulOp

        @classmethod
        def match_and_replace(cls, operation, replacer) -> bool:
            """Detects a callback invoked for an unrelated root type."""
            raise AssertionError("the root filter must reject this operation")

    assert "linalg.copy" in _apply_to_source(_copy_source(), [UnrelatedPattern])


def test_failed_user_check_preserves_the_original_operation():
    class RejectingPattern(TileArrayProgramReplacePattern):
        root = linalg.CopyOp

        @classmethod
        def match_and_replace(cls, operation, replacer) -> bool:
            """Declines a candidate after the root type has matched."""
            return False

    replaced = _apply_to_source(_copy_source(), [RejectingPattern])
    assert "linalg.copy" in replaced
    assert "neura.kernel" not in replaced


def test_invalid_user_implementation_remains_an_error():
    def invalid(A: synl.Buffer, C: synl.Buffer):
        raise ValueError("invalid user program")

    class InvalidPattern(TileArrayProgramReplacePattern):
        root = linalg.CopyOp

        @classmethod
        def match_and_replace(cls, operation, replacer) -> bool:
            """Passes an invalid implementation to the staged lowering path."""
            operation = cast(linalg.CopyOp, operation)
            return replacer.replace_with_tile_array_program(
                operation,
                program=invalid,
                arguments=tuple(operation.inputs) + tuple(operation.outputs),
            )

    with pytest.raises(ValueError, match="invalid user program"):
        _apply_to_source(_copy_source(), [InvalidPattern])
