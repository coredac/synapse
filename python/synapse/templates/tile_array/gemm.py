"""Reusable GEMM implementations authored with the Synapse language."""

import synapse.language as synl


def ws_gemm_4x4(
    A: synl.Tensor,
    B: synl.Tensor,
    C: synl.Tensor,
):
    """Describes a fixed 4x4 weight-stationary GEMM."""

    array = synl.TileArray(x_tiles=4, y_tiles=4)

    partial_sums = []

    for k in range(array.y_tiles):
        y = array.y_tiles - 1 - k

        # A[:, k] enters from the west boundary ports and flows east across this row.
        flowing = synl.input_port(
            A[:, k],
            port=array.west_ports[y],
        )

        next_partial_sums = []

        for x in range(array.x_tiles):
            accumulated, flowing = synl.mac(
                flowing,
                partial_sums[x] if partial_sums else None,
                stationary=B[k, x],
                tile=array[x, y],
            )

            next_partial_sums.append(accumulated)

        partial_sums = next_partial_sums

    for x, accumulated in enumerate(partial_sums):
        # Each south Port writes one result column into C.
        synl.output_port(
            accumulated,
            target=C[:, x],
            port=array.south_ports[x],
        )
