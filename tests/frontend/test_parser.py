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


# Full dump of dump_ast(gemm):
#
# Module(
#   body=[
#     FunctionDef(
#       name='gemm',
#       args=arguments(
#         posonlyargs=[],
#         args=[
#           arg(arg='A'),
#           arg(arg='B'),
#           arg(arg='C')],
#         kwonlyargs=[],
#         kw_defaults=[],
#         defaults=[]),
#       body=[
#         For(
#           target=Name(id='i', ctx=Store()),
#           iter=Call(
#             func=Name(id='range', ctx=Load()),
#             args=[
#               Constant(value=128)],
#             keywords=[]),
#           body=[
#             For(
#               target=Name(id='j', ctx=Store()),
#               iter=Call(
#                 func=Name(id='range', ctx=Load()),
#                 args=[
#                   Constant(value=128)],
#                 keywords=[]),
#               body=[
#                 Assign(
#                   targets=[
#                     Name(id='acc', ctx=Store())],
#                   value=Constant(value=0.0)),
#                 For(
#                   target=Name(id='k', ctx=Store()),
#                   iter=Call(
#                     func=Name(id='range', ctx=Load()),
#                     args=[
#                       Constant(value=128)],
#                     keywords=[]),
#                   body=[
#                     AugAssign(
#                       target=Name(id='acc', ctx=Store()),
#                       op=Add(),
#                       value=BinOp(
#                         left=Subscript(
#                           value=Name(id='A', ctx=Load()),
#                           slice=Tuple(
#                             elts=[
#                               Name(id='i', ctx=Load()),
#                               Name(id='k', ctx=Load())],
#                             ctx=Load()),
#                           ctx=Load()),
#                         op=Mult(),
#                         right=Subscript(
#                           value=Name(id='B', ctx=Load()),
#                           slice=Tuple(
#                             elts=[
#                               Name(id='k', ctx=Load()),
#                               Name(id='j', ctx=Load())],
#                             ctx=Load()),
#                           ctx=Load())))],
#                   orelse=[]),
#                 Assign(
#                   targets=[
#                     Subscript(
#                       value=Name(id='C', ctx=Load()),
#                       slice=Tuple(
#                         elts=[
#                           Name(id='i', ctx=Load()),
#                           Name(id='j', ctx=Load())],
#                         ctx=Load()),
#                       ctx=Store())],
#                   value=Name(id='acc', ctx=Load()))],
#               orelse=[])],
#           orelse=[])],
#       decorator_list=[],
#       type_params=[])],
#   type_ignores=[])
def test_dump_ast_contains_gemm_nodes():
    dumped = dump_ast(gemm)

    assert "FunctionDef" in dumped
    assert "For" in dumped
    assert "Call" in dumped
    assert "AugAssign" in dumped
    assert "Subscript" in dumped
    assert "BinOp" in dumped
    assert "Assign" in dumped