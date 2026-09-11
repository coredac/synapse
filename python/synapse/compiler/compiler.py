"""Top-Level Synapse Compilation Flow."""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from synapse.frontend.lowering import lower
from synapse.language.types import TensorType
from synapse.patterns import TileArrayRewritePattern


def compile(
    program: Callable | str,
    *,
    target: str,
    argument_types: tuple[TensorType, ...] = (),
    patterns: Sequence[type[TileArrayRewritePattern]] | None = None,
) -> str:
    """Compiles a TileArray function or bufferized task IR for the backend."""

    # The current compilation path targets Neura.
    if target != "neura":
        raise ValueError(f"unsupported compilation target: {target}")
    if isinstance(program, str):
        if argument_types:
            raise ValueError("IR inputs already carry their argument types")
        neura_ir = rewrite(program, patterns=patterns)
    else:
        if patterns is not None:
            raise ValueError("rewrite patterns apply to IR inputs")
        neura_ir = lower(program, argument_types=argument_types)
    return _run_neura_backend(neura_ir)


def rewrite(
    source: str,
    *,
    patterns: Sequence[type[TileArrayRewritePattern]] | None = None,
) -> str:
    """Applies patterns to task IR while preserving unmatched computations."""
    from taskflow_mlir.dialects import neura, taskflow
    from taskflow_mlir.ir import Context, Location, Module

    from synapse.compiler.pattern_rewriter import apply_patterns
    from synapse.patterns.gemm_pattern import (
        AffineGemmPattern,
        LinalgGemmPattern,
        LinalgGenericGemmPattern,
    )

    if patterns is None:
        patterns = [LinalgGemmPattern, LinalgGenericGemmPattern, AffineGemmPattern]
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()
        module = Module.parse(source)
        apply_patterns(module, patterns)
        if not module.operation.verify():
            raise ValueError("rewritten module failed verification")
        return str(module)


def _run_neura_backend(neura_ir: str) -> str:
    """Legalizes values, inserts data movement, and runs template mapping."""

    repository_root = Path(__file__).resolve().parents[3]
    amoeba_opt = (
        repository_root
        / "build"
        / "amoeba"
        / "tools"
        / "mlir-amoeba-opt"
        / "mlir-amoeba-opt"
    )

    if not amoeba_opt.is_file():
        raise FileNotFoundError(f"Amoeba compiler is not built: {amoeba_opt}")

    architecture_spec = (
        repository_root
        / "mlir"
        / "amoeba"
        / "thirdparty"
        / "neura"
        / "test"
        / "arch_spec"
        / "architecture.yaml"
    )

    with TemporaryDirectory(prefix="synapse-") as temporary_directory:
        output_path = Path(temporary_directory) / "mapped.mlir"

        command = [
            str(amoeba_opt),
            f"--neura-architecture-spec={architecture_spec}",
            "--promote-input-arg-to-const",
            "--leverage-predicated-value",
            "--insert-data-mov",
            (
                "--map-to-accelerator="
                "mapping-strategy=template "
                "mapping-mode=spatial-only"
            ),
            "-o",
            str(output_path),
        ]

        completed = subprocess.run(
            command,
            input=neura_ir,
            capture_output=True,
            text=True,
            check=False,
            cwd=temporary_directory,
        )

        if completed.returncode != 0:
            diagnostics = "\n".join(
                output for output in (completed.stdout, completed.stderr) if output
            )

            raise RuntimeError(f"Neura backend compilation failed:\n{diagnostics}")

        return output_path.read_text(encoding="utf-8")
