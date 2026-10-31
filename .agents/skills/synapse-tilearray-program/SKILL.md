---
name: synapse-tilearray-program
description: Write or update Synapse TileArray kernel programs using Buffer slices, explicit tile placement, and dataflow operations. Use for kernel implementation or adapting a program to a supplied architecture and computation specification.
---

# Synapse TileArray programs

Implement the supplied computation as a reusable Python TileArray function.
Resolve repository paths from the directory containing `pyproject.toml`.

## Read the relevant sources

- [Public language API](../../../python/synapse/language/__init__.py), [operation semantics](../../../python/synapse/language/tile_array_program.py), and [buffer values and slices](../../../python/synapse/language/values.py).
- [WS GEMM example](../../../python/synapse/library/gemm.py) and [lowering and mapping tests](../../../tests/python/compiler/test_systolic_gemm.py).
- Read the supplied architecture specification before selecting tile coordinates and operations.

## Implement the program

1. Establish argument shapes and dtypes, computation semantics, output overwrite or accumulation behavior, and any supplied preconditions.
2. Define a Python callable with `synl.Buffer` arguments and supplied `BufferType` values such as `synl.i32[3, 3]`. Python loops construct the recorded program; they are not runtime loops in the generated kernel.
3. Create a `synl.TileArray` consistent with the target architecture. Place memory and compute operations on supported tiles, and connect data through returned `TileArrayValue` operands.
4. Keep natural buffer indexing such as `synl.load(A[:, k], tile=array[0, y])`. These slices describe configured, ordered memory accesses; they do not execute NumPy loads.
5. Use the existing `load`, `store`, `constant`, `add`, and `mac` operations. In `mac`, the first result is accumulated data and the second forwards the flowing input. The stationary argument selects one scalar buffer element.
6. Derive placements, indices, and data connections from the requested computation. Keep logical indices distinct from physical tile coordinates and distinguish dimension sizes from element coordinates.
7. Keep the kernel reusable through direct compilation and pattern replacement. For requested pattern integration, use [synapse-patterns](../synapse-patterns/SKILL.md).

## Verify and deliver

- Use `build_tile_array_program(kernel_fn, argument_types=...)` from the frontend to inspect the recorded operations when needed.
- Use `lower(kernel_fn, argument_types=...)` for pre-mapping IR, and `synapse.compile(kernel_fn, target="neura", argument_types=..., architecture_spec=...)` for mapping.
- Follow existing tests to check meaningful IR and code generation behavior. Run focused tests in an environment that provides `taskflow_mlir`; check Python typing when Python code changes.
- Use an existing execution validator when available. Otherwise report numerical behavior as unverified; compiler and mapping checks establish different properties.
- Follow repository naming conventions and use descriptive comments in the third person. Keep changes reviewable and within the requested scope.
- Deliver the kernel source and validation results, including any unsupported conditions or remaining failures.
