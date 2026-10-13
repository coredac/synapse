"""Checks configurable CONV to im2col and matmul with before/after IR."""

from math import prod
from typing import ClassVar

import pytest
from synapse.compiler.pattern_rewrite import PatternRewriter, apply_rewrites
from synapse.patterns import IRRewritePattern
from taskflow_mlir.dialects import arith, linalg, memref
from taskflow_mlir.ir import (
    AffineDimExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    ArrayAttr,
    Attribute,
    Context,
    DenseIntElementsAttr,
    F32Type,
    InsertionPoint,
    IntegerType,
    Location,
    MemRefType,
    Module,
    OpView,
)


def _unflatten_index(index: AffineExpr, shape: list[int]) -> list[AffineExpr]:
    """Builds row-major coordinates from one flat index.

    Shape [1, 1, 3] gives [0, 0, index]. Shape [2, 3] gives
    [index floordiv 3, index mod 3]. The index ranges over prod(shape).
    """
    coordinates = []
    flat_extent = dimension_stride = prod(shape)
    for extent in shape:
        dimension_stride //= extent
        coordinate = AffineExpr.get_floor_div(index, dimension_stride)
        if extent == 1:
            coordinate = AffineExpr.get_constant(0)
        elif dimension_stride * extent != flat_extent:
            # Later dimensions wrap after one full cycle of their extent.
            coordinate = AffineExpr.get_mod(coordinate, extent)
        coordinates.append(coordinate)
    return coordinates


class ConvToMatmulPattern(IRRewritePattern):
    """Rewrites convolution as im2col followed by C += patches @ weights.

    Each patches row contains the input window for one output position.
    Filter and output buffers become matrix views without data copies.
    """

    root = linalg.Conv2DNhwcHwcfOp
    # The target dimensions are (M, N, K) for C[M,N] += A[M,K] @ B[K,N].
    gemm_shape: ClassVar[tuple[int, int, int]] = (3, 3, 3)

    @classmethod
    def match_and_rewrite(cls, operation: OpView, rewriter: PatternRewriter) -> bool:
        """Matches the configured matrix dimensions and emits the decomposition."""
        # The named convolution has input, filter, and output buffer operands.
        input_buffer, filter_buffer, output_buffer = (
            operation.operands[index] for index in range(3)
        )
        buffers = (input_buffer, filter_buffer, output_buffer)
        if any(not MemRefType.isinstance(buffer.type) for buffer in buffers):
            return False
        buffer_types = tuple(MemRefType(buffer.type) for buffer in buffers)
        if any(
            not type_.has_static_shape
            or type_ != MemRefType.get(type_.shape, type_.element_type)
            for type_ in buffer_types
        ):
            return False
        input_type, filter_type, output_type = buffer_types
        dtype = input_type.element_type
        is_integer = IntegerType.isinstance(dtype)
        if any(type_.element_type != dtype for type_ in buffer_types) or not (
            is_integer or F32Type.isinstance(dtype)
        ):
            return False

        # GEMM dimensions count output positions, output channels, and window elements.
        batch_size, output_height, output_width, output_channels = output_type.shape
        kernel_height, kernel_width, input_channels, _ = filter_type.shape
        rows = batch_size * output_height * output_width
        columns = output_channels
        reduction = kernel_height * kernel_width * input_channels
        if (rows, columns, reduction) != cls.gemm_shape:
            return False

        # Each patches[row, k] reads one element from an input window.
        row, k = AffineDimExpr.get(0), AffineDimExpr.get(1)
        batch_index, output_y, output_x = _unflatten_index(
            row, [batch_size, output_height, output_width]
        )
        kernel_y, kernel_x, channel_index = _unflatten_index(
            k, [kernel_height, kernel_width, input_channels]
        )
        strides = DenseIntElementsAttr(operation.attributes["strides"])
        dilations = DenseIntElementsAttr(operation.attributes["dilations"])
        stride_y, stride_x = strides[0], strides[1]
        dilation_y, dilation_x = dilations[0], dilations[1]
        input_y = output_y * stride_y + kernel_y * dilation_y
        input_x = output_x * stride_x + kernel_x * dilation_x
        input_map = AffineMap.get(2, 0, [batch_index, input_y, input_x, channel_index])

        with rewriter.ip:
            # Im2col copies overlapping input windows into a rows-by-reduction matrix.
            patches = memref.AllocaOp(MemRefType.get([rows, reduction], dtype), [], [])
            im2col = linalg.GenericOp(
                [],
                inputs=[input_buffer],
                outputs=[patches.result],
                indexing_maps=ArrayAttr.get(
                    [
                        AffineMapAttr.get(input_map),
                        AffineMapAttr.get(AffineMap.get_identity(2)),
                    ]
                ),
                iterator_types=ArrayAttr.get(
                    [Attribute.parse("#linalg.iterator_type<parallel>")] * 2
                ),
            )
            copy_body = im2col.regions[0].blocks.append(dtype, dtype)
            with InsertionPoint(copy_body):
                linalg.YieldOp([copy_body.arguments[0]])

            # Matrix views merge dimensions 0-2 and preserve output channels.
            weights = memref.CollapseShapeOp(
                MemRefType.get([reduction, columns], dtype),
                filter_buffer,
                [[0, 1, 2], [3]],
            )
            result = memref.CollapseShapeOp(
                MemRefType.get([rows, columns], dtype), output_buffer, [[0, 1, 2], [3]]
            )

            # Matmul accumulates patches[m, k] * weights[k, n] into result[m, n].
            matmul = linalg.MatmulOp(
                [],
                inputs=[patches.result, weights.result],
                outputs=[result.result],
                indexing_maps=ArrayAttr.get(
                    [
                        Attribute.parse("affine_map<(m,n,k)->(m,k)>"),
                        Attribute.parse("affine_map<(m,n,k)->(k,n)>"),
                        Attribute.parse("affine_map<(m,n,k)->(m,n)>"),
                    ]
                ),
            )
            matmul_body = matmul.regions[0].blocks.append(dtype, dtype, dtype)
            patch_value = matmul_body.arguments[0]
            weight_value = matmul_body.arguments[1]
            accumulator = matmul_body.arguments[2]
            multiply, add = (
                (arith.MulIOp, arith.AddIOp)
                if is_integer
                else (arith.MulFOp, arith.AddFOp)
            )
            with InsertionPoint(matmul_body):
                multiplied = multiply(patch_value, weight_value)
                accumulated = add(accumulator, multiplied.result)
                linalg.YieldOp([accumulated.result])
        rewriter.erase_op(operation)
        return True


CONV_IR = """
module {
  func.func @conv(%input: memref<1x1x5x1xi32>,
                  %filter: memref<1x3x1x3xi32>,
                  %output: memref<1x1x3x3xi32>) {
    linalg.conv_2d_nhwc_hwcf
      {strides = dense<1> : tensor<2xi64>,
       dilations = dense<1> : tensor<2xi64>}
      ins(%input, %filter : memref<1x1x5x1xi32>, memref<1x3x1x3xi32>)
      outs(%output : memref<1x1x3x3xi32>)
    return
  }
}
"""

IM2COL_MATMUL_IR = """
module {
  func.func @conv(%input: memref<1x1x5x1xi32>,
                  %filter: memref<1x3x1x3xi32>,
                  %output: memref<1x1x3x3xi32>) {
    %patches = memref.alloca() : memref<3x3xi32>
    linalg.generic {
      indexing_maps = [affine_map<(m,k)->(0,0,m+k,0)>,
                       affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<1x1x5x1xi32>) outs(%patches : memref<3x3xi32>) {
    ^bb0(%element: i32, %old: i32):
      linalg.yield %element : i32
    }
    %weights = memref.collapse_shape %filter [[0, 1, 2], [3]]
      : memref<1x3x1x3xi32> into memref<3x3xi32>
    %result = memref.collapse_shape %output [[0, 1, 2], [3]]
      : memref<1x1x3x3xi32> into memref<3x3xi32>
    linalg.matmul ins(%patches, %weights : memref<3x3xi32>, memref<3x3xi32>)
                  outs(%result : memref<3x3xi32>)
    return
  }
}
"""

POINTWISE_CONV_IR = """
module {
  func.func @conv(%input: memref<1x1x3x3xi32>,
                  %filter: memref<1x1x3x3xi32>,
                  %output: memref<1x1x3x3xi32>) {
    linalg.conv_2d_nhwc_hwcf
      {strides = dense<1> : tensor<2xi64>,
       dilations = dense<1> : tensor<2xi64>}
      ins(%input, %filter : memref<1x1x3x3xi32>, memref<1x1x3x3xi32>)
      outs(%output : memref<1x1x3x3xi32>)
    return
  }
}
"""

POINTWISE_MATMUL_IR = """
module {
  func.func @conv(%input: memref<1x1x3x3xi32>,
                  %filter: memref<1x1x3x3xi32>,
                  %output: memref<1x1x3x3xi32>) {
    %patches = memref.alloca() : memref<3x3xi32>
    linalg.generic {
      indexing_maps = [affine_map<(m,k)->(0,0,m,k)>,
                       affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<1x1x3x3xi32>) outs(%patches : memref<3x3xi32>) {
    ^bb0(%element: i32, %old: i32):
      linalg.yield %element : i32
    }
    %weights = memref.collapse_shape %filter [[0, 1, 2], [3]]
      : memref<1x1x3x3xi32> into memref<3x3xi32>
    %result = memref.collapse_shape %output [[0, 1, 2], [3]]
      : memref<1x1x3x3xi32> into memref<3x3xi32>
    linalg.matmul ins(%patches, %weights : memref<3x3xi32>, memref<3x3xi32>)
                  outs(%result : memref<3x3xi32>)
    return
  }
}
"""

FOUR_ROW_MATMUL_IR = """
module {
  func.func @conv(%input: memref<1x1x6x1xi32>,
                  %filter: memref<1x3x1x3xi32>,
                  %output: memref<1x1x4x3xi32>) {
    %patches = memref.alloca() : memref<4x3xi32>
    linalg.generic {
      indexing_maps = [affine_map<(m,k)->(0,0,m+k,0)>,
                       affine_map<(m,k)->(m,k)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<1x1x6x1xi32>) outs(%patches : memref<4x3xi32>) {
    ^bb0(%element: i32, %old: i32):
      linalg.yield %element : i32
    }
    %weights = memref.collapse_shape %filter [[0, 1, 2], [3]]
      : memref<1x3x1x3xi32> into memref<3x3xi32>
    %result = memref.collapse_shape %output [[0, 1, 2], [3]]
      : memref<1x1x4x3xi32> into memref<4x3xi32>
    linalg.matmul ins(%patches, %weights : memref<4x3xi32>, memref<3x3xi32>)
                  outs(%result : memref<4x3xi32>)
    return
  }
}
"""


@pytest.mark.parametrize(
    "source, expected_ir",
    [
        (CONV_IR, IM2COL_MATMUL_IR),
        (POINTWISE_CONV_IR, POINTWISE_MATMUL_IR),
        (CONV_IR.replace("i32", "f32"), IM2COL_MATMUL_IR.replace("i32", "f32")),
        (
            CONV_IR.replace("1x1x5x1", "1x1x7x1").replace(
                "strides = dense<1>", "strides = dense<[1, 2]>"
            ),
            IM2COL_MATMUL_IR.replace("1x1x5x1", "1x1x7x1").replace("m+k", "m*2+k"),
        ),
        (
            CONV_IR.replace("1x1x5x1", "1x1x7x1").replace(
                "dilations = dense<1>", "dilations = dense<[1, 2]>"
            ),
            IM2COL_MATMUL_IR.replace("1x1x5x1", "1x1x7x1").replace("m+k", "m+k*2"),
        ),
    ],
    ids=["sliding-window", "pointwise", "f32", "stride-2", "dilation-2"],
)
def test_conv_rewrite_matches_expected_ir(source, expected_ir):
    with Context(), Location.unknown():
        module = Module.parse(source)
        expected = Module.parse(expected_ir)
        assert apply_rewrites(module, [ConvToMatmulPattern]) == 1
        assert module.operation.verify()
        assert module.operation.get_asm(
            print_generic_op_form=True
        ) == expected.operation.get_asm(print_generic_op_form=True)
        assert apply_rewrites(module, [ConvToMatmulPattern]) == 0


def test_unmatched_gemm_shape_preserves_original_ir():
    source = CONV_IR.replace("1x1x5x1", "1x1x6x1").replace("1x1x3x3", "1x1x4x3")
    with Context(), Location.unknown():
        module = Module.parse(source)
        original = str(module)
        assert apply_rewrites(module, [ConvToMatmulPattern]) == 0
        assert str(module) == original


def test_gemm_shape_can_be_configured():
    class FourRowConvToMatmulPattern(ConvToMatmulPattern):
        gemm_shape = (4, 3, 3)

    source = CONV_IR.replace("1x1x5x1", "1x1x6x1").replace("1x1x3x3", "1x1x4x3")
    with Context(), Location.unknown():
        module = Module.parse(source)
        expected = Module.parse(FOUR_ROW_MATMUL_IR)
        assert apply_rewrites(module, [FourRowConvToMatmulPattern]) == 1
        assert module.operation.verify()
        assert module.operation.get_asm(
            print_generic_op_form=True
        ) == expected.operation.get_asm(print_generic_op_form=True)
