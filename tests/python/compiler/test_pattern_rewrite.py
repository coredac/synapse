"""Checks greedy IR rewriting separately from TileArray replacement."""

import pytest
from synapse.compiler import pattern_rewrite
from synapse.compiler.pattern_rewrite import PatternRewriter, apply_rewrites
from synapse.patterns import IRRewritePattern
from taskflow_mlir.dialects import arith, func
from taskflow_mlir.ir import Context, IntegerAttr, Location, Module, OpResult, OpView

SOURCE = """
module {
  func.func @identity(%x: i32) -> i32 {
    %identity = arith.constant IDENTITY : i32
    %first = arith.OP %x, %identity : i32
    %second = arith.OP %first, %identity : i32
    return %second : i32
  }
}
"""


class IntegerIdentityPattern(IRRewritePattern):
    """Eliminates scalar integer addition by zero and multiplication by one."""

    root = (arith.AddIOp, arith.MulIOp)

    @classmethod
    def match_and_rewrite(cls, operation: OpView, rewriter: PatternRewriter) -> bool:
        rhs = operation.operands[1]
        if not OpResult.isinstance(rhs):
            return False
        constant = OpResult(rhs).owner
        if constant.name != "arith.constant":
            return False
        value = constant.attributes["value"]
        identity = 0 if operation.operation.name == "arith.addi" else 1
        if not IntegerAttr.isinstance(value) or IntegerAttr(value).value != identity:
            return False
        rewriter.replace_op_with_values(operation, (operation.operands[0],))
        return True


class RemoveUnusedConstantPattern(IRRewritePattern):
    """Removes the unused constant after a function's nested identities disappear."""

    root = func.FuncOp

    @classmethod
    def match_and_rewrite(cls, operation: OpView, rewriter: PatternRewriter) -> bool:
        body = tuple(operation.regions[0].blocks[0].operations)
        if len(body) != 2 or body[0].operation.name != "arith.constant":
            return False
        if any(body[0].results[0].uses):
            return False
        rewriter.erase_op(body[0])
        return True


@pytest.mark.parametrize("op,identity", [("addi", "0"), ("muli", "1")])
@pytest.mark.parametrize("whole_module", [True, False])
def test_rewrites_values_and_revisits_parent_operations(op, identity, whole_module):
    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", op).replace("IDENTITY", identity))
        function = module.body.operations[0]
        count = apply_rewrites(
            module if whole_module else function,
            [RemoveUnusedConstantPattern, IntegerIdentityPattern],
        )
        assert count == (3 if whole_module else 2)
        assert module.operation.verify()
        block = function.regions[0].blocks[0]
        body = tuple(block.operations)
        assert len(body) == (1 if whole_module else 2)
        assert body[-1].operands[0] == block.arguments[0]
        assert apply_rewrites(module, [IntegerIdentityPattern]) == 0


def test_rejected_rewrite_preserves_ir():
    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "1"))
        original = str(module)
        assert apply_rewrites(module, [IntegerIdentityPattern]) == 0
        assert str(module) == original
        assert module.operation.verify()


def test_root_filter_and_pattern_order():
    class UnexpectedPattern(IRRewritePattern):
        root = (arith.AddIOp, arith.SubIOp)

        @classmethod
        def match_and_rewrite(
            cls, operation: OpView, rewriter: PatternRewriter
        ) -> bool:
            raise AssertionError(
                "the earlier rule or root filter should reject this callback"
            )

    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "0"))
        assert apply_rewrites(module, [IntegerIdentityPattern, UnexpectedPattern]) == 2
        assert module.operation.verify()


def test_builtin_rewrites_precede_added_user_rules(monkeypatch: pytest.MonkeyPatch):
    class UnexpectedUserPattern(IRRewritePattern):
        root = arith.AddIOp

        @classmethod
        def match_and_rewrite(
            cls, operation: OpView, rewriter: PatternRewriter
        ) -> bool:
            raise AssertionError("the built-in rule should consume the addition")

    monkeypatch.setattr(
        pattern_rewrite, "_DEFAULT_REWRITE_PATTERNS", (IntegerIdentityPattern,)
    )
    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "0"))
        assert apply_rewrites(module, [UnexpectedUserPattern]) == 2
        assert "arith.addi" not in str(module)
        assert module.operation.verify()


def test_complete_pass_rewrites_multiple_operations_in_one_round():
    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "0"))
        assert apply_rewrites(module, [IntegerIdentityPattern], max_iterations=2) == 2
        assert "arith.addi" not in str(module)
        assert module.operation.verify()


def test_erasing_a_container_removes_its_pending_body():
    class RemoveUnusedFunctionPattern(IRRewritePattern):
        """Erases a private test function and detects visits to its deleted body."""

        root = (func.FuncOp, arith.AddIOp)

        @classmethod
        def match_and_rewrite(
            cls, operation: OpView, rewriter: PatternRewriter
        ) -> bool:
            if operation.operation.name != "func.func":
                raise AssertionError("the erased function body should not be visited")
            rewriter.erase_op(operation)
            return True

    with Context(), Location.unknown():
        source = SOURCE.replace("OP", "addi").replace("IDENTITY", "0")
        module = Module.parse(source.replace("func.func @", "func.func private @"))
        assert apply_rewrites(module, [RemoveUnusedFunctionPattern]) == 1
        assert len(tuple(module.body.operations)) == 0
        assert module.operation.verify()


def test_rewrite_can_erase_a_pending_sibling():
    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "0"))
        block = module.body.operations[0].regions[0].blocks[0]

        class RemoveIdentityPairPattern(IRRewritePattern):
            """Removes two captured integer identities in one match."""

            root = arith.AddIOp

            @classmethod
            def match_and_rewrite(
                cls, operation: OpView, rewriter: PatternRewriter
            ) -> bool:
                value = operation.operands[0]
                rewriter.replace_op_with_values(block.operations[2], (value,))
                rewriter.replace_op_with_values(operation, (value,))
                return True

        assert apply_rewrites(module, [RemoveIdentityPairPattern]) == 1
        assert "arith.addi" not in str(module)
        assert tuple(block.operations)[-1].operands[0] == block.arguments[0]
        assert module.operation.verify()


def test_rewrites_operation_and_redirects_uses():
    class AddToMultiplyPattern(IRRewritePattern):
        """Rewrites repeated integer addition as multiplication by two."""

        root = arith.AddIOp

        @classmethod
        def match_and_rewrite(
            cls, operation: OpView, rewriter: PatternRewriter
        ) -> bool:
            if operation.operands[0] != operation.operands[1]:
                return False
            type_ = operation.results[0].type
            with rewriter.ip:
                two = arith.ConstantOp(type_, IntegerAttr.get(type_, 2))
                replacement = arith.MulIOp(operation.operands[0], two.result)
            rewriter.replace_op(operation, replacement)
            return True

    source = """
    module {
      func.func @compute(%x: i32) -> i32 {
        %result = arith.addi %x, %x : i32
        return %result : i32
      }
    }
    """
    with Context(), Location.unknown():
        module = Module.parse(source)
        assert apply_rewrites(module, [AddToMultiplyPattern]) == 1
        assert "arith.addi" not in str(module)
        assert "arith.muli" in str(module)
        block = module.body.operations[0].regions[0].blocks[0]
        assert block.operations[2].operands[0] == block.operations[1].results[0]
        assert module.operation.verify()


def test_nonconverging_rewrites_stop_at_iteration_limit():
    class SwapAddOperandsPattern(IRRewritePattern):
        """Creates an equivalent addition with reversed operands on every match."""

        root = arith.AddIOp

        @classmethod
        def match_and_rewrite(
            cls, operation: OpView, rewriter: PatternRewriter
        ) -> bool:
            with rewriter.ip:
                replacement = arith.AddIOp(operation.operands[1], operation.operands[0])
            rewriter.replace_op(operation, replacement)
            return True

    with Context(), Location.unknown():
        module = Module.parse(SOURCE.replace("OP", "addi").replace("IDENTITY", "0"))
        with pytest.raises(RuntimeError, match="did not converge within 4 iterations"):
            apply_rewrites(module, [SwapAddOperandsPattern], max_iterations=4)
        assert module.operation.verify()
