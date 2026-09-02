import synapse
import synapse.language as synl
from synapse.frontend import lowering
from synapse.templates.tile_array import ws_gemm_4x4

PRE_MAPPING_IR = """
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0, 1)>
#map2 = affine_map<(d0) -> (d0, 2)>
#map3 = affine_map<(d0) -> (d0, 3)>
#map4 = affine_map<(d0, d1) -> (-d1 + 3, d0)>
module {
  func.func @ws_gemm_4x4(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>) -> memref<4x4xi32> {
    %done_writes = taskflow.task @ws_gemm_4x4 will_reads(%arg0, %arg1 : memref<4x4xi32>, memref<4x4xi32>) will_writes(%arg2 : memref<4x4xi32>) [original_read_memrefs(%arg0, %arg1 : memref<4x4xi32>, memref<4x4xi32>), original_write_memrefs(%arg2 : memref<4x4xi32>)] : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> (memref<4x4xi32>) {
    ^bb0(%arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>):
      %0:4 = taskflow.stream_read %arg3 maps [#map, #map1, #map2, #map3] : memref<4x4xi32> -> (i32, i32, i32, i32)
      %1:4 = neura.kernel inputs(%0#0, %0#1, %0#2, %0#3, %arg4 : i32, i32, i32, i32, memref<4x4xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {input_ports = [{direction = "west", kernel_input = 0 : i32, x = 0 : i32, y = 3 : i32}, {direction = "west", kernel_input = 1 : i32, x = 0 : i32, y = 2 : i32}, {direction = "west", kernel_input = 2 : i32, x = 0 : i32, y = 1 : i32}, {direction = "west", kernel_input = 3 : i32, x = 0 : i32, y = 0 : i32}], name = "systolic_array", output_ports = [{direction = "south", kernel_result = 0 : i32, x = 0 : i32, y = 0 : i32}, {direction = "south", kernel_result = 1 : i32, x = 1 : i32, y = 0 : i32}, {direction = "south", kernel_result = 2 : i32, x = 2 : i32, y = 0 : i32}, {direction = "south", kernel_result = 3 : i32, x = 3 : i32, y = 0 : i32}], stationary = {kernel_input = 4 : i32, map = #map4, mode = "weight"}}}} {
      ^bb0(%arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: memref<4x4xi32>):
        %2 = "neura.mac"(%arg6) <{stationary = "weight"}> {placement = {x = 0 : i32, y = 3 : i32}} : (i32) -> i32
        %3 = "neura.mac"(%arg6) <{stationary = "weight"}> {placement = {x = 1 : i32, y = 3 : i32}} : (i32) -> i32
        %4 = "neura.mac"(%arg6) <{stationary = "weight"}> {placement = {x = 2 : i32, y = 3 : i32}} : (i32) -> i32
        %5 = "neura.mac"(%arg6) <{stationary = "weight"}> {placement = {x = 3 : i32, y = 3 : i32}} : (i32) -> i32
        %6 = "neura.mac"(%arg7, %2) <{stationary = "weight"}> {placement = {x = 0 : i32, y = 2 : i32}} : (i32, i32) -> i32
        %7 = "neura.mac"(%arg7, %3) <{stationary = "weight"}> {placement = {x = 1 : i32, y = 2 : i32}} : (i32, i32) -> i32
        %8 = "neura.mac"(%arg7, %4) <{stationary = "weight"}> {placement = {x = 2 : i32, y = 2 : i32}} : (i32, i32) -> i32
        %9 = "neura.mac"(%arg7, %5) <{stationary = "weight"}> {placement = {x = 3 : i32, y = 2 : i32}} : (i32, i32) -> i32
        %10 = "neura.mac"(%arg8, %6) <{stationary = "weight"}> {placement = {x = 0 : i32, y = 1 : i32}} : (i32, i32) -> i32
        %11 = "neura.mac"(%arg8, %7) <{stationary = "weight"}> {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> i32
        %12 = "neura.mac"(%arg8, %8) <{stationary = "weight"}> {placement = {x = 2 : i32, y = 1 : i32}} : (i32, i32) -> i32
        %13 = "neura.mac"(%arg8, %9) <{stationary = "weight"}> {placement = {x = 3 : i32, y = 1 : i32}} : (i32, i32) -> i32
        %14 = "neura.mac"(%arg9, %10) <{stationary = "weight"}> {placement = {x = 0 : i32, y = 0 : i32}} : (i32, i32) -> i32
        %15 = "neura.mac"(%arg9, %11) <{stationary = "weight"}> {placement = {x = 1 : i32, y = 0 : i32}} : (i32, i32) -> i32
        %16 = "neura.mac"(%arg9, %12) <{stationary = "weight"}> {placement = {x = 2 : i32, y = 0 : i32}} : (i32, i32) -> i32
        %17 = "neura.mac"(%arg9, %13) <{stationary = "weight"}> {placement = {x = 3 : i32, y = 0 : i32}} : (i32, i32) -> i32
        neura.yield results(%14, %15, %16, %17 : i32, i32, i32, i32)
      } : i32, i32, i32, i32
      taskflow.stream_write(%1#0, %1#1, %1#2, %1#3 : i32, i32, i32, i32) to %arg5 maps [#map, #map1, #map2, #map3] : memref<4x4xi32>
      taskflow.yield done_writes(%arg5 : memref<4x4xi32>)
    }
    return %done_writes : memref<4x4xi32>
  }
}
""".strip()


MAPPED_IR = """
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0, 1)>
#map2 = affine_map<(d0) -> (d0, 2)>
#map3 = affine_map<(d0) -> (d0, 3)>
#map4 = affine_map<(d0, d1) -> (-d1 + 3, d0)>
module {
  func.func @ws_gemm_4x4(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>) -> memref<4x4xi32> {
    %done_writes = taskflow.task @ws_gemm_4x4 will_reads(%arg0, %arg1 : memref<4x4xi32>, memref<4x4xi32>) will_writes(%arg2 : memref<4x4xi32>) [original_read_memrefs(%arg0, %arg1 : memref<4x4xi32>, memref<4x4xi32>), original_write_memrefs(%arg2 : memref<4x4xi32>)] : (memref<4x4xi32>, memref<4x4xi32>, memref<4x4xi32>) -> (memref<4x4xi32>) {
    ^bb0(%arg3: memref<4x4xi32>, %arg4: memref<4x4xi32>, %arg5: memref<4x4xi32>):
      %0:4 = taskflow.stream_read %arg3 maps [#map, #map1, #map2, #map3] : memref<4x4xi32> -> (i32, i32, i32, i32)
      %1:4 = neura.kernel inputs(%0#0, %0#1, %0#2, %0#3, %arg4 : i32, i32, i32, i32, memref<4x4xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {input_ports = [{direction = "west", kernel_input = 0 : i32, x = 0 : i32, y = 3 : i32}, {direction = "west", kernel_input = 1 : i32, x = 0 : i32, y = 2 : i32}, {direction = "west", kernel_input = 2 : i32, x = 0 : i32, y = 1 : i32}, {direction = "west", kernel_input = 3 : i32, x = 0 : i32, y = 0 : i32}], name = "systolic_array", output_ports = [{direction = "south", kernel_result = 0 : i32, x = 0 : i32, y = 0 : i32}, {direction = "south", kernel_result = 1 : i32, x = 1 : i32, y = 0 : i32}, {direction = "south", kernel_result = 2 : i32, x = 2 : i32, y = 0 : i32}, {direction = "south", kernel_result = 3 : i32, x = 3 : i32, y = 0 : i32}], stationary = {kernel_input = 4 : i32, map = #map4, mode = "weight"}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
      ^bb0(%arg6: !neura.data<i32, i1>, %arg7: !neura.data<i32, i1>, %arg8: !neura.data<i32, i1>, %arg9: !neura.data<i32, i1>, %arg10: !neura.data<memref<4x4xi32>, i1>):
        %2 = "neura.data_mov"(%arg6) {dfg_id = 0 : i32, mapping_locs = [{direction = "west", id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "port", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %3 = "neura.mac"(%2) <{stationary = "weight"}> {dfg_id = 16 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %4 = "neura.data_mov"(%arg6) {dfg_id = 1 : i32, mapping_locs = [{direction = "west", id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "port", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}, {id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %5 = "neura.mac"(%4) <{stationary = "weight"}> {dfg_id = 17 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %6 = "neura.data_mov"(%arg6) {dfg_id = 2 : i32, mapping_locs = [{direction = "west", id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "port", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}, {id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 41 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %7 = "neura.mac"(%6) <{stationary = "weight"}> {dfg_id = 18 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %8 = "neura.data_mov"(%arg6) {dfg_id = 3 : i32, mapping_locs = [{direction = "west", id = 20 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, io = "input", resource = "port", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}, {id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 41 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}, {id = 44 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %9 = "neura.mac"(%8) <{stationary = "weight"}> {dfg_id = 19 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %10 = "neura.data_mov"(%arg7) {dfg_id = 4 : i32, mapping_locs = [{direction = "west", id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "port", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %11 = "neura.data_mov"(%3) {dfg_id = 20 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %12 = "neura.mac"(%10, %11) <{stationary = "weight"}> {dfg_id = 24 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %13 = "neura.data_mov"(%arg7) {dfg_id = 5 : i32, mapping_locs = [{direction = "west", id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "port", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}, {id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %14 = "neura.data_mov"(%5) {dfg_id = 21 : i32, mapping_locs = [{id = 42 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %15 = "neura.mac"(%13, %14) <{stationary = "weight"}> {dfg_id = 25 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %16 = "neura.data_mov"(%arg7) {dfg_id = 6 : i32, mapping_locs = [{direction = "west", id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "port", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}, {id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}, {id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %17 = "neura.data_mov"(%7) {dfg_id = 22 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %18 = "neura.mac"(%16, %17) <{stationary = "weight"}> {dfg_id = 26 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %19 = "neura.data_mov"(%arg7) {dfg_id = 7 : i32, mapping_locs = [{direction = "west", id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, io = "input", resource = "port", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}, {id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}, {id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}, {id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %20 = "neura.data_mov"(%9) {dfg_id = 23 : i32, mapping_locs = [{id = 47 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %21 = "neura.mac"(%19, %20) <{stationary = "weight"}> {dfg_id = 27 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %22 = "neura.data_mov"(%arg8) {dfg_id = 8 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, io = "input", resource = "port", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %23 = "neura.data_mov"(%12) {dfg_id = 28 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %24 = "neura.mac"(%22, %23) <{stationary = "weight"}> {dfg_id = 32 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %25 = "neura.data_mov"(%arg8) {dfg_id = 9 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, io = "input", resource = "port", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}, {id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %26 = "neura.data_mov"(%15) {dfg_id = 29 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %27 = "neura.mac"(%25, %26) <{stationary = "weight"}> {dfg_id = 33 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %28 = "neura.data_mov"(%arg8) {dfg_id = 10 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, io = "input", resource = "port", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}, {id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}, {id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %29 = "neura.data_mov"(%18) {dfg_id = 30 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %30 = "neura.mac"(%28, %29) <{stationary = "weight"}> {dfg_id = 34 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %31 = "neura.data_mov"(%arg8) {dfg_id = 11 : i32, mapping_locs = [{direction = "west", id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, io = "input", resource = "port", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}, {id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}, {id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}, {id = 18 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %32 = "neura.data_mov"(%21) {dfg_id = 31 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %33 = "neura.mac"(%31, %32) <{stationary = "weight"}> {dfg_id = 35 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %34 = "neura.data_mov"(%arg9) {dfg_id = 12 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "input", resource = "port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %35 = "neura.data_mov"(%24) {dfg_id = 36 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %36 = "neura.mac"(%34, %35) <{stationary = "weight"}> {dfg_id = 40 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %37 = "neura.data_mov"(%arg9) {dfg_id = 13 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "input", resource = "port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}, {id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %38 = "neura.data_mov"(%27) {dfg_id = 37 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %39 = "neura.mac"(%37, %38) <{stationary = "weight"}> {dfg_id = 41 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %40 = "neura.data_mov"(%arg9) {dfg_id = 14 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "input", resource = "port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}, {id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}, {id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %41 = "neura.data_mov"(%30) {dfg_id = 38 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %42 = "neura.mac"(%40, %41) <{stationary = "weight"}> {dfg_id = 42 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %43 = "neura.data_mov"(%arg9) {dfg_id = 15 : i32, mapping_locs = [{direction = "west", id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "input", resource = "port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}, {id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}, {id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}, {id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %44 = "neura.data_mov"(%33) {dfg_id = 39 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %45 = "neura.mac"(%43, %44) <{stationary = "weight"}> {dfg_id = 43 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %46 = "neura.data_mov"(%36) {dfg_id = 44 : i32, mapping_locs = [{direction = "south", id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, io = "output", resource = "port", time_step = 3 : i32, x = 0 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %47 = "neura.data_mov"(%39) {dfg_id = 45 : i32, mapping_locs = [{direction = "south", id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, io = "output", resource = "port", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %48 = "neura.data_mov"(%42) {dfg_id = 46 : i32, mapping_locs = [{direction = "south", id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, io = "output", resource = "port", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %49 = "neura.data_mov"(%45) {dfg_id = 47 : i32, mapping_locs = [{direction = "south", id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, io = "output", resource = "port", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        neura.yield results(%46, %47, %48, %49 : !neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) {dfg_id = 48 : i32}
      } : i32, i32, i32, i32
      taskflow.stream_write(%1#0, %1#1, %1#2, %1#3 : i32, i32, i32, i32) to %arg5 maps [#map, #map1, #map2, #map3] : memref<4x4xi32>
      taskflow.yield done_writes(%arg5 : memref<4x4xi32>)
    }
    return %done_writes : memref<4x4xi32>
  }
}
""".strip()


def test_lowers_systolic_gemm_to_exact_pre_mapping_ir():
    actual = lowering.lower(
        ws_gemm_4x4,
        argument_types=(
            synl.i32[4, 4],
            synl.i32[4, 4],
            synl.i32[4, 4],
        ),
    )

    assert actual.strip() == PRE_MAPPING_IR


def test_compiles_systolic_gemm_to_exact_mapped_ir():
    actual = synapse.compile(
        ws_gemm_4x4,
        target="neura",
        argument_types=(
            synl.i32[4, 4],
            synl.i32[4, 4],
            synl.i32[4, 4],
        ),
    )

    assert actual.strip() == MAPPED_IR
