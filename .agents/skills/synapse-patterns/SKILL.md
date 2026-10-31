---
name: synapse-patterns
description: Write or update Synapse MLIR patterns for computation rewrites and TileArray program replacement. Use for IR matching, rewrite rules, or connecting an existing kernel to source IR.
---

# Synapse patterns

Implement the requested computation match and IR change through existing pattern APIs.
Resolve repository paths from the directory containing `pyproject.toml`.

## Choose the pattern

- Use `IRRewritePattern.match_and_rewrite(operation, rewriter)` to transform computation IR while preserving its semantics, such as CONV to im2col and matmul.
- Use `TileArrayProgramReplacePattern.match_and_replace(operation, replacer)` to select a TileArray kernel implementation for a matched computation.
- Declare `root` as the relevant MLIR OpView class. The driver checks this type before calling the pattern.
- If a replacement requires a new kernel, use [synapse-tilearray-program](../synapse-tilearray-program/SKILL.md) for that implementation.

## Implement the rule

1. Read [pattern interfaces](../../../python/synapse/patterns/pattern.py) and the relevant example: [GEMM replacements](../../../python/synapse/patterns/gemm_pattern.py) or [CONV rewrite](../../../tests/python/compiler/test_conv_rewrite.py).
2. Establish the computation semantics and supported shapes and dtypes. Keep computation-specific checks in the concrete pattern; reuse shared compiler operations instead of duplicating them.
3. Perform all rejection checks before mutation. Return `False` without changing IR; return `True` only after a change.
4. Use the rewriter or replacer insertion point and mutation helpers. For kernel replacement, call `replace_with_tile_array_program(operation, program=kernel_fn, arguments=...)`.
5. Preserve overwrite versus accumulation behavior. For example, Linalg matmul computes `C += A @ B`, while the current WS GEMM program computes `C = A @ B`.
6. Make rewrite rules converge. `apply_rewrites` repeats complete pattern passes until a round makes no changes.

The current kernel replacement helper accepts result-free operations directly inside `taskflow.task`, with arguments that are static, default-layout i32/f32 memref captures of that task.
A newly created intermediate buffer or view is not automatically an accepted task capture.
User patterns extend the built-in sets; built-in replacements are tried first.

## Verify and deliver

- Add small input and expected-output IR examples to the corresponding test file. Check verified rewritten IR, and unchanged IR for relevant rejected candidates.
- For replacement, confirm the intended kernel was selected; compilation through Counter/General fallback alone does not establish replacement success.
- Follow [rewrite tests](../../../tests/python/compiler/test_pattern_rewrite.py) or [replacement tests](../../../tests/python/compiler/test_pattern_replacement.py).
- Use `apply_rewrites` or `apply_replacements` for focused tests. Use `synapse.compile(task_ir, target="neura", rewrite_patterns=[...], replace_patterns=[...], architecture_spec=...)` when checking compiler integration.
- Run tests appropriate to the change in an environment that provides `taskflow_mlir`; check Python typing when Python code changes.
- Follow repository naming conventions and use descriptive comments in the third person. Keep changes reviewable and within the requested scope.
- Report what changed, checks performed, and limitations. IR verification and mapping success do not establish numerical correctness.
