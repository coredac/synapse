"""Top-Level Synapse Compilation Flow."""

import subprocess
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from synapse.frontend.lowering import lower
from synapse.language.types import TensorType


def compile(
    program: Callable,
    *,
    target: str,
    argument_types: tuple[TensorType, ...] = (),
) -> str:
    """Compile a Synapse program for the selected backend."""

    # We only support the Neura backend for now, so we raise an error if the user tries to compile for any other target.
    if target != "neura":
        raise ValueError(f"unsupported compilation target: {target}")
    # TODO: Support the amoeba backend.

    neura_ir = lower(program, argument_types=argument_types)
    return _run_neura_backend(neura_ir)


def _run_neura_backend(neura_ir: str) -> str:
    """Legalize Neura values, insert data movement, and run template mapping."""

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

    with TemporaryDirectory(prefix="synapse-") as temporary_directory:
        output_path = Path(temporary_directory) / "mapped.mlir"

        command = [
            str(amoeba_opt),
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
        )

        if completed.returncode != 0:
            diagnostics = "\n".join(
                output for output in (completed.stdout, completed.stderr) if output
            )

            raise RuntimeError(f"Neura backend compilation failed:\n{diagnostics}")

        return output_path.read_text(encoding="utf-8")
