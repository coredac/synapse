// This file defines the expected post-mapping Neura IR for the
// Synapse single-task spatial program.

module {
  func.func @single_task() {
    taskflow.task @Task_0 : () -> () {
      neura.kernel attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "manual", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 2 : i32, y_tiles = 1 : i32}} {
        // Tile(row=0, col=0) produces one value at step 0.
        %source = "neura.constant"() <{value = 1 : i32}> {dfg_id = 0 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
        
        // Move the value east to Tile(row=0, col=1) and keep it in local register 0.
        %moved = "neura.data_mov"(%source) {dfg_id = 1 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        
        // Tile(row=0, col=1) consumes the transferred value at step 2.
        %result = "neura.add"(%moved) {dfg_id = 2 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 0 : i32}], rhs_value = 1 : i32} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        
        neura.yield
      }
      taskflow.yield
    }
    return
  }
}