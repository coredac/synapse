"""Reusable GEMM implementations authored with the Synapse language."""

import synapse.language as synl


def ws_gemm_4x4(
    A: synl.Tensor,
    B: synl.Tensor,
    C: synl.Tensor,
):
    """Describe a fixed 4x4 weight-stationary GEMM."""

    array = synl.TileArray(x_tiles=4, y_tiles=4)

    partial_sums = []

    for k in range(array.y_tiles):
        y = array.y_tiles - 1 - k

        # Stream one column of A through the corresponding west Port.
        activation = synl.input_port(
            A[:, k],
            port=array.west_ports[y],
        )

        # Each Tile keeps one value from B stationary while partial sums
        # propagate toward the south boundary.
        partial_sums = [
            synl.mac(
                activation,
                partial_sums[x] if partial_sums else None,
                stationary=B[k, x],
                mode=synl.StationaryMode.WEIGHT,
                tile=array[x, y],
            )
            for x in range(array.x_tiles)
        ]

    for x, result in enumerate(partial_sums):
        # Each south Port writes one result column into C.
        synl.output_port(
            result,
            target=C[:, x],
            port=array.south_ports[x],
        )
