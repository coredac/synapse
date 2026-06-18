import ast

from synapse.frontend.parser import dump_ast, parse_function


def gemm(A, B, C):
    for i in range(128):
        for j in range(128):
            acc = 0.0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_parse_function_captures_gemm_source():
    parsed = parse_function(gemm)

    assert parsed.name == "gemm"
    assert "def gemm" in parsed.source
    assert isinstance(parsed.tree, ast.Module)

    fn = parsed.tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert fn.name == "gemm"
    assert [arg.arg for arg in fn.args.args] == ["A", "B", "C"]


def test_dump_ast_contains_gemm_nodes():
    dumped = dump_ast(gemm)

    assert "FunctionDef" in dumped
    assert "For" in dumped
    assert "Call" in dumped
    assert "AugAssign" in dumped
    assert "Subscript" in dumped
    assert "BinOp" in dumped
    assert "Assign" in dumped