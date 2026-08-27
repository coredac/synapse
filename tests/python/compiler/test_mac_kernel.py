import synapse
import synapse.language as synl
from synapse.frontend.lowering import lower

EXPECTED_IR = """
module {
  func.func @f32_mac() {
    taskflow.task @f32_mac : () -> () {
      neura.kernel attributes {accelerator = "neura"} {
        %0 = "neura.constant"() <{value = 2.000000e+00 : f32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> f32
        %1 = "neura.constant"() <{value = 3.000000e+00 : f32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> f32
        %2 = "neura.constant"() <{value = 0.000000e+00 : f32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> f32
        %3 = "neura.fmul_fadd"(%0, %1, %2) {placement = {x = 0 : i32, y = 0 : i32}} : (f32, f32, f32) -> f32
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}
""".strip()


def f32_mac():
    array = synl.TileArray(x_tiles=1, y_tiles=1)

    lhs = synl.constant(2.0, tile=array[0, 0])
    rhs = synl.constant(3.0, tile=array[0, 0])
    accumulator = synl.constant(0.0, tile=array[0, 0])

    synl.mac(
        lhs,
        rhs,
        accumulator,
        tile=array[0, 0],
    )


def mapped_f32_mac():
    array = synl.TileArray(x_tiles=2, y_tiles=2)

    lhs = synl.constant(2.0, tile=array[0, 0])
    rhs = synl.constant(3.0, tile=array[1, 0])
    accumulator = synl.constant(0.0, tile=array[0, 1])

    synl.mac(
        lhs,
        rhs,
        accumulator,
        tile=array[1, 1],
    )


def test_f32_mac_lowers_to_neura():
    actual = lower(f32_mac)

    assert actual.strip() == EXPECTED_IR


def test_f32_mac_compiles_to_mapped_neura():
    actual = synapse.compile(
        mapped_f32_mac,
        target="neura",
    )

    assert '"neura.fmul_fadd"' in actual
    assert 'mapping_strategy = "template"' in actual
    assert 'resource = "link"' in actual
    assert "placement =" not in actual
