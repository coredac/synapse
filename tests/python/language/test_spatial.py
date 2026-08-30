import synapse.language as synl


def test_tile_array_is_parameterized():
    array_2x3 = synl.TileArray(x_tiles=2, y_tiles=3)
    array_4x4 = synl.TileArray(x_tiles=4, y_tiles=4)

    assert array_2x3.x_tiles == 2
    assert array_2x3.y_tiles == 3

    assert array_4x4.x_tiles == 4
    assert array_4x4.y_tiles == 4


def test_tile_array_exposes_tiles():
    array = synl.TileArray(x_tiles=2, y_tiles=3)

    coordinates = {(tile.x, tile.y) for tile in array.tiles}

    assert coordinates == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_access_tile_array_by_coordinate():
    array = synl.TileArray(x_tiles=2, y_tiles=3)
    tile = array[1, 2]

    assert tile.array is array
    assert tile.x == 1
    assert tile.y == 2

    assert tile is array[1, 2]

    enumerated_tile = next(
        candidate for candidate in array.tiles if candidate.x == 1 and candidate.y == 2
    )

    assert tile is enumerated_tile


def test_tile_array_exposes_boundary_ports():
    array = synl.TileArray(x_tiles=2, y_tiles=3)

    assert [(port.direction, port.x, port.y) for port in array.west_ports] == [
        ("west", 0, 0),
        ("west", 0, 1),
        ("west", 0, 2),
    ]

    assert [(port.direction, port.x, port.y) for port in array.east_ports] == [
        ("east", 1, 0),
        ("east", 1, 1),
        ("east", 1, 2),
    ]

    assert [(port.direction, port.x, port.y) for port in array.north_ports] == [
        ("north", 0, 2),
        ("north", 1, 2),
    ]

    assert [(port.direction, port.x, port.y) for port in array.south_ports] == [
        ("south", 0, 0),
        ("south", 1, 0),
    ]

    assert array.west_ports[2].array is array
    assert array.west_ports[2].tile is array[0, 2]
    assert array.south_ports[1].tile is array[1, 0]
