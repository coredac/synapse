"""Hardware spatial features exposed by the Synapse language.
This file exposes spatial structures that programmers can use to organize
computation and data movement according to the target hardware hierarchy.

TileArray currently exposes the two-dimensional tile array of a CGRA.

Future spatial abstractions may expose inter-core structures, such as the
core array of a multi-CGRA, AMD AIE/NPU, or Tenstorrent.
"""


class Tile:
    """A hardware tile in a CGRA TileArray.

    A Tile is owned by a TileArray. Its ``x`` and ``y`` coordinates identify
    the corresponding position in the target CGRA tile array.
    """

    def __init__(self, x: int, y: int, array: "TileArray"):
        self.array = array
        self.x = x
        self.y = y


class TileArray:
    """A parameterized two-dimensional tile array.

    In the initial implementation, ``x_tiles`` and ``y_tiles`` must match the
    dimensions of the target CGRA. A tile accessed as ``array[x, y]``
    corresponds directly to the tile at that hardware coordinate.

    Coordinates follow Neura's convention: increasing ``x`` moves east
    (right), and increasing ``y`` moves north (up).

    Example:
        array = TileArray(x_tiles=4, y_tiles=4)
        tile = array[1, 2]
    """

    def __init__(self, x_tiles: int, y_tiles: int):
        self.x_tiles = x_tiles
        self.y_tiles = y_tiles

        self._tiles = [
            [Tile(x=x, y=y, array=self) for x in range(x_tiles)]
            for y in range(y_tiles)
        ]

    def tiles(self):
        """Iterate over all tiles in the array.

        The iteration order is a Python programming convenience and
        does not specify sequential hardware execution.
        """
        for y_row in self._tiles:
            yield from y_row

    def __getitem__(self, coordinate: tuple[int, int]) -> Tile:
        """Return the tile at the given ``(x, y)`` coordinate."""

        x, y = coordinate
        return self._tiles[y][x]
