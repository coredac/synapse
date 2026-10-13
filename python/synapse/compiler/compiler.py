"""Top-Level Synapse Compilation Flow."""

from collections.abc import Callable, Sequence
from pathlib import Path

from synapse.frontend.lowering import lower
from synapse.language.types import BufferType
from synapse.patterns import IRRewritePattern, TileArrayProgramReplacePattern

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
    replace_patterns: Sequence[type[TileArrayProgramReplacePattern]] | None = None,
    rewrite_patterns: Sequence[type[IRRewritePattern]] = (),
    architecture_spec: str | Path | None = None,
) -> str:
    """Compiles a TileArray function or bufferized task IR for the backend.

    rewrite_patterns extend built-in IR rewrites at each source level.
    replace_patterns extend built-in TileArray implementation patterns.
    """

    # The current compilation path targets Neura.
    if target != "neura":
        raise ValueError(f"unsupported compilation target: {target}")
    if isinstance(program, str):
        if argument_types:
            raise ValueError("IR inputs already carry their argument types")
        source = program
    else:
        if replace_patterns is not None:
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
        replace_patterns=replace_patterns,
        rewrite_patterns=rewrite_patterns,
        architecture_spec=architecture_spec,
    )


def _compile_task_graph(
    source: str,
    *,
    replace_patterns: Sequence[type[TileArrayProgramReplacePattern]] | None,
    rewrite_patterns: Sequence[type[IRRewritePattern]],
    architecture_spec: Path,
) -> str:
    """Normalizes source IR, selects templates, and runs the Neura backend."""
    from taskflow_mlir.dialects import neura, taskflow
    from taskflow_mlir.ir import Context, Location, Module
    from taskflow_mlir.passmanager import PassManager

    from synapse.compiler.pattern_replace import apply_replacements
    from synapse.compiler.pattern_rewrite import apply_rewrites

    taskflow.set_neura_architecture_spec(str(architecture_spec))

    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        apply_rewrites(module, rewrite_patterns)
        apply_replacements(module, replace_patterns)
        PassManager.parse(
            "builtin.module(func.func(convert-linalg-to-affine-loops))"
        ).run(module.operation)
        apply_rewrites(module, rewrite_patterns)
        apply_replacements(module, replace_patterns)
        PassManager.parse(_NEURA_BACKEND_PIPELINE).run(module.operation)
        if not module.operation.verify():
            raise ValueError("compiled task graph failed verification")
        return str(module)
