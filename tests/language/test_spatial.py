import synapse.language as synl


def test_tile_array_is_parameterized():
    array_2x3 = synl.TileArray(rows=2, cols=3)
    array_4x4 = synl.TileArray(rows=4, cols=4)

    assert array_2x3.rows == 2
    assert array_2x3.cols == 3

    assert array_4x4.rows == 4
    assert array_4x4.cols == 4


def test_tile_array_exposes_logical_tiles():
    array = synl.TileArray(2, 3)

    coordinates = {(tile.row, tile.col) for tile in array.tiles()}

    assert coordinates == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_access_tile_array_by_coordinate():
    array = synl.TileArray(2, 3)
    tile = array[1, 2]

    assert tile.row == 1
    assert tile.col == 2

    assert tile is array[1, 2]

    enumerated_tile = next(
        candidate
        for candidate in array.tiles()
        if candidate.row == 1 and candidate.col == 2
    )

    assert tile is enumerated_tile
