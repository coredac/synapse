"""Top-Level Synapse Compilation Flow."""

from collections.abc import Callable, Sequence
from pathlib import Path

from synapse.frontend.lowering import lower
from synapse.language.types import BufferType
from synapse.patterns import TileArrayProgramPattern

_NEURA_BACKEND_PIPELINE = """builtin.module(
  convert-affine-to-taskflow,
  func.func(construct-hyperblock-from-task),
  classify-task-and-counter,
  convert-taskflow-to-neura,
  lower-affine,
  convert-scf-to-cf,
  convert-cf-to-llvm,
  assign-accelerator,
  lower-memref-to-neura,
  lower-arith-to-neura,
  lower-builtin-to-neura,
  lower-llvm-to-neura,
  canonicalize-return,
  canonicalize-cast,
  promote-input-arg-to-const,
  fold-constant,
  canonicalize-live-in,
  leverage-predicated-value,
  transform-ctrl-to-data-flow,
  fold-constant,
  insert-data-mov,
  map-to-accelerator
)"""


def compile(
    program: Callable | str,
    *,
    target: str,
    argument_types: tuple[BufferType, ...] = (),
    patterns: Sequence[type[TileArrayProgramPattern]] | None = None,
    architecture_spec: str | Path | None = None,
) -> str:
    """Compiles a TileArray function or bufferized task IR for the backend."""

    # The current compilation path targets Neura.
    if target != "neura":
        raise ValueError(f"unsupported compilation target: {target}")
    if isinstance(program, str):
        if argument_types:
            raise ValueError("IR inputs already carry their argument types")
        source = program
    else:
        if patterns is not None:
            raise ValueError("replacement patterns apply to IR inputs")
        source = lower(program, argument_types=argument_types)

    if architecture_spec is None:
        architecture_spec = (
            Path(__file__).resolve().parents[3]
            / "mlir"
            / "amoeba"
            / "thirdparty"
            / "neura"
            / "test"
            / "arch_spec"
            / "architecture.yaml"
        )
    architecture_spec = Path(architecture_spec)
    if not architecture_spec.is_file():
        raise FileNotFoundError(
            f"Neura architecture specification does not exist: {architecture_spec}"
        )
    return _compile_task_graph(
        source,
        patterns=patterns,
        architecture_spec=architecture_spec,
    )


def _compile_task_graph(
    source: str,
    *,
    patterns: Sequence[type[TileArrayProgramPattern]] | None,
    architecture_spec: Path,
) -> str:
    """Applies patterns before and after Linalg-to-Affine conversion."""
    from taskflow_mlir.dialects import neura, taskflow
    from taskflow_mlir.ir import Context, Location, Module
    from taskflow_mlir.passmanager import PassManager

    from synapse.compiler.pattern_replacement import apply_patterns
    from synapse.patterns.gemm_pattern import (
        AffineGemmPattern,
        LinalgGemmPattern,
        LinalgGenericGemmPattern,
    )

    if patterns is None:
        replacement_patterns = (
            LinalgGemmPattern,
            LinalgGenericGemmPattern,
            AffineGemmPattern,
        )
    else:
        replacement_patterns = patterns

    taskflow.set_neura_architecture_spec(str(architecture_spec))

    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        apply_patterns(module, replacement_patterns)
        PassManager.parse(
            "builtin.module(func.func(convert-linalg-to-affine-loops))"
        ).run(module.operation)
        apply_patterns(module, replacement_patterns)
        PassManager.parse(_NEURA_BACKEND_PIPELINE).run(module.operation)
        if not module.operation.verify():
            raise ValueError("compiled task graph failed verification")
        return str(module)
