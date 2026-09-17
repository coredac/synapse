"""Explicit spatial implementations shared by direct programs and patterns."""

import synapse.language as synl
from synapse.language.tile_array_program import TileArrayValue


def ws_gemm_3x3(A: synl.Tensor, B: synl.Tensor, C: synl.Tensor):
    """Computes C = A @ B using nine MACs and six memory tiles.

    C is overwritten and must not overlap either input buffer. The matching
    patterns prove this precondition before selecting the implementation.
    """
    if any(tensor.type != synl.i32[3, 3] for tensor in (A, B, C)):
        raise ValueError("ws_gemm_3x3 requires three 3x3 i32 tensors")

    array = synl.TileArray(x_tiles=4, y_tiles=4)
    partial_sums: list[TileArrayValue] = []

    # Physical MAC rows run north to south while reduction indices increase.
    for y in range(3, 0, -1):
        k = 3 - y
        flowing = synl.load(A[:, k], tile=array[0, y])
        next_partial_sums: list[TileArrayValue] = []

        # The west column contains loads; MAC columns start at x=1.
        for x in range(1, 4):
            accumulated, flowing = synl.mac(
                flowing,
                partial_sums[x - 1] if partial_sums else None,
                stationary=B[k, x - 1],
                tile=array[x, y],
            )
            next_partial_sums.append(accumulated)
        partial_sums = next_partial_sums

    # Each south memory tile writes one output column.
    for x, accumulated in enumerate(partial_sums, start=1):
        synl.store(accumulated, target=C[:, x - 1], tile=array[x, 0])
