"""Hardware spatial features exposed by the Synapse language.
This file exposes spatial structures that programmers can use to organize
computation and data movement according to the target hardware hierarchy.

TileArray currently exposes the two-dimensional tile array of a CGRA.

Future spatial abstractions may expose inter-core structures, such as the
core array of a multi-CGRA, AMD AIE/NPU, or Tenstorrent.
"""


class Tile:
    """A hardware tile in a CGRA TileArray.

    A Tile is owned by a TileArray. Its row and column identify the
    corresponding position in the target CGRA tile array.
    """

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col


class TileArray:
    """A parameterized two-dimensional tile array.

    In the initial implementation, ``rows`` and ``cols`` must match the
    dimensions of the target CGRA. A tile accessed as ``array[row, col]``
    corresponds directly to the tile at that hardware coordinate.

    Example:
        array = TileArray(rows=4, cols=4)
        tile = array[1, 2]
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

        self._tiles = [
            [Tile(row=row, col=col) for col in range(cols)] for row in range(rows)
        ]

    def tiles(self):
        """Iterate over all tiles in the array.

        The iteration order is a Python programming convenience and
        does not specify sequential hardware execution.
        """
        for row_tiles in self._tiles:
            yield from row_tiles

    def __getitem__(self, coordinate: tuple[int, int]) -> Tile:
        """
        Return the tile at the given ``(row, col)`` coordinate
        """
        row, col = coordinate
        return self._tiles[row][col]
