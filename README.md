# SYNAPSE

SYNAPSE is a programming model and compiler frontend for
task-level dataflow systems.

## Repository Layout

```text
python/synapse/          Python package source
python/synapse/frontend  Source capture and frontend parser utilities
examples/                Small frontend examples
tests/                   Pytest tests
mlir/neura/              Downstream NEURA / Taskflow compiler stack
```

## Setup

Create or activate a Python environment, then install SYNAPSE in editable mode
from the repository root:

```bash
cd $PROJECT_PATH/synapse
python -m pip install -e .
```

Editable install only needs to be done once per environment. After that,
changes under `python/synapse/` are picked up directly.

For tests, install `pytest`:

```bash
python -m pip install pytest
```

## Run The GEMM Parser Example

```bash
cd $PROJECT_PATH/synapse
python examples/gemm.py
```

The example parses a plain Python GEMM function and prints its Python AST. At
this stage, the GEMM function is not executed; SYNAPSE only reads its source.

## Run Tests

```bash
cd $PROJECT_PATH/synapse
pytest -q
```

The current tests check that the frontend parser can capture a Python GEMM
function and expose the expected AST nodes.
