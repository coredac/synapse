# SYNAPSE

SYNAPSE is a prototype programming model for hierarchical data-driven
accelerator and SoC design.

The first implementation target is Taskflow IR:

```text
SYNAPSE Python model
  -> TaskflowDialect MLIR
```

The repository is intentionally narrower than Triton: SYNAPSE owns the
programming model, AST frontend, internal graph IR, and Taskflow lowering.
Compilation from Taskflow IR to NEURA or lower accelerator targets is handled
by the external NEURA/Taskflow compiler stack under `mlir/`.

- `python/synapse/frontend`: Python source capture, AST type/shape inference,
  and AST-to-Graph construction.
- `python/synapse/language`: user-facing tensor program API plus lower-level
  graph/task/kernel escape hatches.
- `python/synapse/compiler`: compile driver and compiler targets for Taskflow,
  currently only `taskflow`.
- `python/synapse/runtime`: launch descriptors and future runtime contracts.
- `mlir`: external compiler submodules, currently `mlir/neura`.
- `examples`: small source examples.
- `test`: future lit/FileCheck tests.

Run a smoke example:

```bash
PYTHONPATH=synapse/python python synapse/examples/matmul_relu.py
```

The default programming style is a symbolic tensor program:

```python
@syn.program
def matmul_relu(
    A: syn.Tensor((64, 64), "f32", layout=syn.tile(16, 16)),
    B: syn.Tensor((64, 64), "f32", layout=syn.tile(16, 16)),
):
    C = syn.matmul(A, B, target=["cgra", "systolic"])
    return syn.relu(C, target="cgra")
```

The frontend parses the Python source, runs a small type/shape inference pass,
and then builds a SYNAPSE graph. The compiler turns that graph into Taskflow
tasks, channels, layouts, and hardware intent metadata. Users should not need
to manually construct `taskflow.task` or `Graph.task(...)` for normal programs.

Compile to Taskflow IR:

```bash
PYTHONPATH=synapse/python \
  python -m synapse.tools.synapse_compile synapse/examples/matmul_relu.py \
  --target taskflow
```

To compile Taskflow IR to NEURA, use the NEURA/Taskflow compiler submodule
under `mlir/neura`. SYNAPSE does not expose that path as a frontend demo target.
