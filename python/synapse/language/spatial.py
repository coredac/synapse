"""Hardware spatial features exposed by the Synapse language.
This file exposes spatial structures that programmers can use to organize
computation and data movement according to the target hardware hierarchy.

TileArray currently exposes the two-dimensional tile array of a CGRA.

Future spatial abstractions may expose inter-core structures, such as the
core array of a multi-CGRA, AMD AIE/NPU, or Tenstorrent.
"""

from typing import Literal


class Tile:
    """A hardware tile in a CGRA TileArray.

    A Tile is owned by a TileArray. Its ``x`` and ``y`` coordinates identify
    the corresponding position in the target CGRA tile array.
    """

    def __init__(self, x: int, y: int, array: "TileArray"):
        self.array = array
        self.x = x
        self.y = y


PortDirection = Literal["west", "east", "north", "south"]


class Port:
    """A boundary data port of a CGRA TileArray.

    ``direction`` identifies the array boundary. The ``x`` and ``y``
    coordinates identify the boundary tile attached to this Port.

    Ports expose hardware connectivity to the programming model. They do not
    prescribe when data is transferred or introduce clock-based scheduling.
    """

    def __init__(
        self,
        *,
        direction: PortDirection,
        x: int,
        y: int,
        array: "TileArray",
    ):
        self.direction = direction
        self.x = x
        self.y = y
        self.array = array

    @property
    def tile(self) -> Tile:
        """Return the boundary tile attached to this Port."""

        return self.array[self.x, self.y]


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

        self.tiles = tuple(
            Tile(x=x, y=y, array=self) for y in range(y_tiles) for x in range(x_tiles)
        )

        self.west_ports = tuple(
            Port(direction="west", x=0, y=y, array=self) for y in range(y_tiles)
        )

        self.east_ports = tuple(
            Port(direction="east", x=x_tiles - 1, y=y, array=self)
            for y in range(y_tiles)
        )

        self.north_ports = tuple(
            Port(direction="north", x=x, y=y_tiles - 1, array=self)
            for x in range(x_tiles)
        )

        self.south_ports = tuple(
            Port(direction="south", x=x, y=0, array=self) for x in range(x_tiles)
        )

    def __getitem__(self, coordinate: tuple[int, int]) -> Tile:
        """Return the tile at the given ``(x, y)`` coordinate."""

        x, y = coordinate

        if not (0 <= x < self.x_tiles and 0 <= y < self.y_tiles):
            raise IndexError(
                f"tile coordinate ({x}, {y}) is outside "
                f"TileArray({self.x_tiles}, {self.y_tiles})"
            )

        return self.tiles[y * self.x_tiles + x]
