# SYNAPSE

SYNAPSE is a Python programming model and compiler frontend for spatial
dataflow systems. Its first backend target is a multi-CGRA architecture in
which each task may contain a program explicitly placed on a CGRA tile array.

The current single-task compiler path lowers a Python TileArray program into:

```text
func.func
  taskflow.task
    neura.kernel
      placed Neura operations
```

The Neura backend then legalizes predicated values, inserts data movement, and
maps the placed operations to tiles, links, registers, and time steps.

## Repository Layout

```text
python/synapse/
  language/              User-facing spatial and TileArray language APIs
  frontend/              Python program capture and lowering to MLIR
  compiler/              Backend compiler orchestration
examples/                Small frontend examples
tests/python/            Python unit and compiler-integration tests
mlir/amoeba/             Pinned Amoeba compiler dependency
  thirdparty/neura/      Pinned Neura backend dependency managed by Amoeba
```

## Requirements

Frontend and language development requires Python 3.10 or newer.

The compiler-integration path additionally requires:

- Python 3.11;
- CMake, Ninja, Clang, LLD, and ccache;
- `pybind11==2.13.6` and `nanobind==2.15.0`; and
- LLVM/MLIR at commit
  [`6146a88f60492b520a36f8f8f3231e15f3cc6082`](https://github.com/llvm/llvm-project/commit/6146a88f60492b520a36f8f8f3231e15f3cc6082).

This is the same LLVM revision and Python binding configuration used by the
pinned Amoeba workflow.

## Checkout

After cloning SYNAPSE, initialize Amoeba and its direct Neura dependency:

```bash
git submodule update --init mlir/amoeba
git -C mlir/amoeba submodule update --init thirdparty/neura
```

These commands intentionally avoid downloading Neura's nested benchmark
submodules, which are not required to build the SYNAPSE compiler path.

## Python Setup

Create or activate a Python environment, then install SYNAPSE in editable mode
from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

Editable install only needs to be done once per environment. After that,
changes under `python/synapse/` are picked up directly.

## Build LLVM and MLIR

Install the Python dependencies into the same Python 3.11 environment that
will configure LLVM and Amoeba:

```bash
python -m pip install pybind11==2.13.6 nanobind==2.15.0
```

Choose a location for LLVM, clone it, and check out the pinned revision:

```bash
export SYNAPSE_LLVM_PROJECT=/absolute/path/to/llvm-project

git clone https://github.com/llvm/llvm-project.git "${SYNAPSE_LLVM_PROJECT}"
git -C "${SYNAPSE_LLVM_PROJECT}" checkout \
  6146a88f60492b520a36f8f8f3231e15f3cc6082
```

Configure and build LLVM/MLIR with Python bindings enabled:

```bash
cmake -G Ninja \
  -S "${SYNAPSE_LLVM_PROJECT}/llvm" \
  -B "${SYNAPSE_LLVM_PROJECT}/build" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build "${SYNAPSE_LLVM_PROJECT}/build" --parallel 2
```

## Build Amoeba and Neura

From the SYNAPSE repository root, point Amoeba at the LLVM build and create the
compiler artifacts under `build/amoeba`:

```bash
export LLVM_BUILD_DIR="${SYNAPSE_LLVM_PROJECT}/build"

cmake -G Ninja \
  -S mlir/amoeba \
  -B build/amoeba \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

cmake --build build/amoeba \
  --target mlir-amoeba-opt AmoebaPythonModules \
  --parallel 2
```

The build produces:

```text
build/amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt
build/amoeba/python_packages/amoeba_core/taskflow_mlir/
```

## Run Tests

Run the frontend and language tests without building LLVM:

```bash
python -m pytest -q tests/python/frontend tests/python/language
```

After building Amoeba, run the compiler-integration tests:

```bash
python -m pytest -q tests/python/compiler
```

Or run the complete Python suite:

```bash
python -m pytest -q tests/python
```

## Run the GEMM Parser Example

```bash
python examples/gemm.py
```

This example captures a plain Python GEMM function and prints its Python AST.
The TileArray compiler path is exercised by the tests under
`tests/python/compiler`.

## Continuous Integration

The GitHub Actions workflow contains two layers:

- `python-unit-tests` runs frontend and language tests on Python 3.10 and 3.11
  without building LLVM.
- `compiler-integration` uses Python 3.11, checks out the pinned Amoeba and
  Neura revisions, downloads the pinned LLVM commit, builds LLVM/MLIR with
  Python bindings, builds `mlir-amoeba-opt` and `AmoebaPythonModules`, and runs
  the compiler tests.

LLVM and ccache artifacts are cached using the pinned LLVM revision so later
workflow runs do not rebuild the entire dependency stack unnecessarily.
