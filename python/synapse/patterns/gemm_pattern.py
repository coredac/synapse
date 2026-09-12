"""Matches GEMM source forms to the weight-stationary implementation."""

from typing import Protocol, cast

from taskflow_mlir.dialects import affine, linalg
from taskflow_mlir.ir import (
    AffineDimExpr,
    AffineMap,
    AffineMapAttr,
    ArrayAttr,
    Attribute,
    IntegerAttr,
    IntegerType,
    MemRefType,
    OpResult,
)

from synapse.library import ws_gemm_3x3
from synapse.patterns import TileArrayRewritePattern


class _AffineMapAttrValue(Protocol):
    """Describes the value property omitted by the pinned MLIR attribute stub."""

    @property
    def value(self) -> AffineMap: ...


def _loop(operation, extent):
    """Recognizes one canonical GEMM loop over the required extent."""
    if operation.operation.name != "affine.for" or len(operation.operands):
        return None
    attrs = operation.operation.attributes
    if (
        attrs["lowerBoundMap"] != AffineMapAttr.get(AffineMap.get_constant(0))
        or attrs["upperBoundMap"] != AffineMapAttr.get(AffineMap.get_constant(extent))
        or IntegerAttr(attrs["step"]).value != 1
    ):
        return None
    block = operation.regions[0].blocks[0]
    return block if len(block.arguments) == 1 and not len(operation.results) else None


def _indices(operation, operands):
    """Resolves the projected indices of a GEMM load or store."""
    access_map = cast(
        _AffineMapAttrValue, AffineMapAttr(operation.operation.attributes["map"])
    ).value
    if access_map.n_symbols or access_map.n_dims != len(operands):
        return None
    if any(not AffineDimExpr.isinstance(expr) for expr in access_map.results):
        return None
    return tuple(operands[AffineDimExpr(expr).position] for expr in access_map.results)


def _zero(value):
    """Recognizes an integer zero used by GEMM initialization."""
    if not OpResult.isinstance(value):
        return False
    owner = OpResult(value).owner
    return (
        owner.name == "arith.constant"
        and IntegerAttr.isinstance(owner.attributes["value"])
        and IntegerAttr(owner.attributes["value"]).value == 0
    )


def _has_zero_output(operation, output):
    """Checks the overwrite template's precondition on the initial accumulator.

    Named matmul and the recognized loop bodies compute C += A @ B, whereas
    the WS program computes C = A @ B. A preceding zero fill makes them agree.
    """
    parent = operation.operation.parent
    if parent is None:
        return False
    previous = None
    found = False
    for region in parent.regions:
        for block in region.blocks:
            previous = None
            for sibling in block.operations:
                if sibling.operation == operation.operation:
                    found = True
                    break
                if sibling.operation.name != "arith.constant":
                    previous = sibling
            if found:
                break
        if found:
            break
    if previous is None:
        return False
    if isinstance(previous, linalg.FillOp):
        return tuple(previous.outputs) == (output,) and _zero(previous.inputs[0])

    rows, columns = MemRefType(output.type).shape
    outer = _loop(previous, rows)
    if outer is None or len(tuple(outer.operations)) != 2:
        return False
    inner = _loop(outer.operations[0], columns)
    if inner is None or len(tuple(inner.operations)) != 2:
        return False
    store = inner.operations[0]
    return (
        store.operation.name == "affine.store"
        and store.operands[1] == output
        and _zero(store.operands[0])
        and _indices(
            store,
            tuple(store.operands[index] for index in range(len(store.operands)))[2:],
        )
        == (outer.arguments[0], inner.arguments[0])
    )


class LinalgGemmPattern(TileArrayRewritePattern):
    """Matches named matmul using its declared semantics and operand types."""

    root = linalg.MatmulOp

    @classmethod
    def match_and_rewrite(cls, operation, rewriter) -> bool:
        """Checks the selected matmul and replaces it with the WS kernel."""
        # The driver has already checked the declared root operation type.
        operation = cast(linalg.MatmulOp, operation)
        arguments = tuple(operation.inputs) + tuple(operation.outputs)
        expected = MemRefType.get([3, 3], IntegerType.get_signless(32))
        if len(operation.results) or len(arguments) != 3:
            return False
        if any(value.type != expected for value in arguments):
            return False
        if not _has_zero_output(operation, arguments[2]):
            return False
        return rewriter.replace_with_tile_array(
            operation,
            program=ws_gemm_3x3,
            arguments=arguments,
        )


class LinalgGenericGemmPattern(TileArrayRewritePattern):
    """Matches contraction maps and the multiply-add dataflow in a generic op."""

    root = linalg.GenericOp

    @classmethod
    def match_and_rewrite(cls, operation, rewriter) -> bool:
        """Checks the contraction dataflow and replaces it with the WS kernel."""
        # The driver has already checked the declared root operation type.
        operation = cast(linalg.GenericOp, operation)
        attrs = operation.operation.attributes
        maps = (
            "affine_map<(i,j,k)->(i,k)>",
            "affine_map<(i,j,k)->(k,j)>",
            "affine_map<(i,j,k)->(i,j)>",
        )
        if tuple(ArrayAttr(attrs["indexing_maps"])) != tuple(
            Attribute.parse(text) for text in maps
        ):
            return False
        if tuple(ArrayAttr(attrs["iterator_types"])) != tuple(
            Attribute.parse(f"#linalg.iterator_type<{kind}>")
            for kind in ("parallel", "parallel", "reduction")
        ):
            return False
        block = operation.regions[0].blocks[0]
        body = tuple(block.operations)
        if len(block.arguments) != 3 or len(body) != 3:
            return False
        multiply, add, terminator = body
        if (
            multiply.operation.name != "arith.muli"
            or tuple(
                multiply.operands[index] for index in range(len(multiply.operands))
            )
            != tuple(block.arguments[index] for index in range(2))
            or add.operation.name != "arith.addi"
            or set(add.operands[index] for index in range(len(add.operands)))
            != {multiply.results[0], block.arguments[2]}
            or terminator.operation.name != "linalg.yield"
            or tuple(
                terminator.operands[index] for index in range(len(terminator.operands))
            )
            != (add.results[0],)
        ):
            return False
        arguments = tuple(operation.inputs) + tuple(operation.outputs)
        expected = MemRefType.get([3, 3], IntegerType.get_signless(32))
        if len(operation.results) or len(arguments) != 3:
            return False
        if any(value.type != expected for value in arguments):
            return False
        if not _has_zero_output(operation, arguments[2]):
            return False
        return rewriter.replace_with_tile_array(
            operation,
            program=ws_gemm_3x3,
            arguments=arguments,
        )


class AffineGemmPattern(TileArrayRewritePattern):
    """Matches matrix accesses and multiply-add dataflow in a canonical loop nest."""

    root = affine.AffineForOp

    @classmethod
    def match_and_rewrite(cls, operation, rewriter) -> bool:
        """Checks the loop dataflow and replaces it with the WS kernel."""
        outer = _loop(operation, 3)
        if outer is None or len(tuple(outer.operations)) != 2:
            return False
        middle = _loop(outer.operations[0], 3)
        if middle is None or len(tuple(middle.operations)) != 2:
            return False
        inner = _loop(middle.operations[0], 3)
        if inner is None:
            return False
        body = tuple(inner.operations)
        if tuple(op.operation.name for op in body) != (
            "affine.load",
            "affine.load",
            "affine.load",
            "arith.muli",
            "arith.addi",
            "affine.store",
            "affine.yield",
        ):
            return False
        lhs, rhs, current, multiply, add, store, _ = body
        i, j, k = outer.arguments[0], middle.arguments[0], inner.arguments[0]
        if (
            _indices(
                lhs,
                tuple(lhs.operands[index] for index in range(len(lhs.operands)))[1:],
            )
            != (i, k)
            or _indices(
                rhs,
                tuple(rhs.operands[index] for index in range(len(rhs.operands)))[1:],
            )
            != (k, j)
            or _indices(
                current,
                tuple(
                    current.operands[index] for index in range(len(current.operands))
                )[1:],
            )
            != (i, j)
            or _indices(
                store,
                tuple(store.operands[index] for index in range(len(store.operands)))[
                    2:
                ],
            )
            != (i, j)
            or tuple(
                multiply.operands[index] for index in range(len(multiply.operands))
            )
            != (lhs.results[0], rhs.results[0])
            or set(add.operands[index] for index in range(len(add.operands)))
            != {current.results[0], multiply.results[0]}
            or tuple(store.operands[index] for index in range(len(store.operands)))[:2]
            != (add.results[0], current.operands[0])
        ):
            return False
        arguments = (lhs.operands[0], rhs.operands[0], current.operands[0])
        expected = MemRefType.get([3, 3], IntegerType.get_signless(32))
        if any(value.type != expected for value in arguments):
            return False
        if not _has_zero_output(operation, arguments[2]):
            return False
        return rewriter.replace_with_tile_array(
            operation,
            program=ws_gemm_3x3,
            arguments=arguments,
        )
