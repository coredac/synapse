import synapse
import synapse.language as synl
from synapse.frontend.lowering import lower

PRE_MAPPING_IR = """
module {
  func.func @add_constant() {
    taskflow.task @add_constant : () -> () {
      neura.kernel attributes {accelerator = "neura"} {
        %0 = "neura.constant"() <{value = 1 : i32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> i32
        %1 = "neura.constant"() <{value = 2 : i32}> {placement = {x = 2 : i32, y = 0 : i32}} : () -> i32
        %2 = "neura.add"(%0, %1) {placement = {x = 1 : i32, y = 0 : i32}} : (i32, i32) -> i32
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}
""".strip()


MAPPED_IR = """
module {
  func.func @add_constant() {
    taskflow.task @add_constant : () -> () {
      neura.kernel attributes {accelerator = "neura", mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
        %0 = "neura.constant"() <{value = 1 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
        %1 = "neura.constant"() <{value = 2 : i32}> {dfg_id = 1 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
        %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %3 = "neura.data_mov"(%1) {dfg_id = 4 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %4 = "neura.add"(%2, %3) {dfg_id = 5 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        neura.yield {dfg_id = 2 : i32}
      }
      taskflow.yield
    }
    return
  }
}
""".strip()


def add_constant():
    array = synl.TileArray(x_tiles=4, y_tiles=4)

    lhs = synl.constant(1, tile=array[0, 0])
    rhs = synl.constant(2, tile=array[2, 0])

    synl.add(lhs, rhs, tile=array[1, 0])


def test_add_constant_lowers_to_placed_neura_ir():
    actual = lower(add_constant)

    assert actual.strip() == PRE_MAPPING_IR


def test_add_constant_compiles_to_mapped_neura_ir():
    actual = synapse.compile(add_constant, target="neura")

    assert actual.strip() == MAPPED_IR
