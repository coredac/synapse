import synapse.language as synl
from synapse.frontend import lowering

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


def ws_gemm_4x4(A: synl.Tensor, B: synl.Tensor, C: synl.Tensor):
    array = synl.TileArray(x_tiles=4, y_tiles=4)

    partial_sums = []

    for k in range(array.y_tiles):
        y = array.y_tiles - 1 - k

        # A[:, k] is streamed through one west boundary Port.
        activation = synl.input_port(
            A[:, k],
            port=array.west_ports[y],
        )

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
        # Each south Port writes one column of C.
        synl.output_port(
            result,
            target=C[:, x],
            port=array.south_ports[x],
        )


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
