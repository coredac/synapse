# SYNAPSE Code Architecture

SYNAPSE is the programming-model repository. It owns the user-facing Python
model, AST frontend, internal graph IR, and lowering to Taskflow IR.

It does not maintain a parallel C++ MLIR compiler tree.

## Repository Boundary

```text
SYNAPSE/
  python/synapse/language   User-facing programming model
  python/synapse/frontend   Python AST parser, inference, and graph builder
  python/synapse/compiler   Compiler driver and Taskflow compile target
  python/synapse/runtime    Launch descriptors and future runtime contracts
  mlir/neura                External NEURA/Taskflow compiler submodule
  python/synapse/tools      Command-line tools
  examples                  Programming model examples
  test                      Python and Taskflow compatibility tests

mlir/neura/
  include/TaskflowDialect   Canonical SYNAPSE task-level IR
  include/NeuraDialect      Instruction-level dataflow IR
  lib/Conversion            Taskflow/NEURA lowering passes
  lib/TaskflowDialect       Task graph orchestration and optimization
  lib/NeuraDialect          NEURA transforms, mapping, codegen
```

## Dialect Decision

We do not maintain both `SynapseDialect` and `TaskflowDialect`.

`TaskflowDialect` should become the canonical task-level IR for SYNAPSE. The
programming model name is SYNAPSE; the compiler IR is the existing Taskflow
dialect, extended as needed.

This avoids a redundant chain like:

```text
synapse.task -> taskflow.task -> neura.kernel
```

Instead:

```text
SYNAPSE Python model
  -> TaskflowDialect + synapse/taskflow attrs
```

If the public IR name later needs to match the programming model, the
TaskflowDialect in `mlir/neura` can be renamed to SynapseDialect. Semantically,
it should remain one dialect, not two overlapping dialects.

## Compiler Package

`python/synapse/compiler` uses standard compiler terminology:

- `driver.py`: public `compile(...)` entry and compile-target registry.
- `taskflow.py`: compiles the Python model to Taskflow MLIR.
- `base.py`: shared compile result and compiler-target protocol.

Public SYNAPSE APIs use compiler/target terminology. The only frontend demo
target is `taskflow`.

## MLIR Submodule Boundary

Compilation after Taskflow IR is intentionally outside the SYNAPSE frontend
demo. That compiler stack lives in `mlir/neura`, registered as a git submodule
from `https://github.com/coredac/neura`.
