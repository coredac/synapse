"""Checks complete standalone GEMM IR and code generation."""

import subprocess
from pathlib import Path

import synapse
import synapse.language as synl
from synapse.frontend import lowering
from synapse.library import ws_gemm_3x3

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NEURA_ROOT = REPOSITORY_ROOT / "mlir/amoeba/thirdparty/neura"

PRE_MAPPING_IR = """
#map = affine_map<(d0, d1) -> (-d1 + 3, d0 - 1)>
module {
  func.func @ws_gemm_3x3(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) -> memref<3x3xi32> {
    %done_writes = taskflow.task @ws_gemm_3x3 will_reads(%arg0, %arg1 : memref<3x3xi32>, memref<3x3xi32>) will_writes(%arg2 : memref<3x3xi32>) [original_read_memrefs(%arg0, %arg1 : memref<3x3xi32>, memref<3x3xi32>), original_write_memrefs(%arg2 : memref<3x3xi32>)] : (memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) -> (memref<3x3xi32>) {
    ^bb0(%arg3: memref<3x3xi32>, %arg4: memref<3x3xi32>, %arg5: memref<3x3xi32>):
      neura.kernel inputs(%arg3, %arg4, %arg5 : memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {name = "systolic_array", stationary = {kernel_input = 1 : i32, map = #map}}}} {
      ^bb0(%arg6: memref<3x3xi32>, %arg7: memref<3x3xi32>, %arg8: memref<3x3xi32>):
        %0 = "neura.load"(%arg6) {constants = array<i64: 0, 3, 6>, placement = {x = 0 : i32, y = 3 : i32}} : (memref<3x3xi32>) -> i32
        %result, %forwarded = "neura.mac"(%0) {placement = {x = 1 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
        %result_0, %forwarded_1 = "neura.mac"(%forwarded) {placement = {x = 2 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
        %result_2, %forwarded_3 = "neura.mac"(%forwarded_1) {placement = {x = 3 : i32, y = 3 : i32}} : (i32) -> (i32, i32)
        %1 = "neura.load"(%arg6) {constants = array<i64: 1, 4, 7>, placement = {x = 0 : i32, y = 2 : i32}} : (memref<3x3xi32>) -> i32
        %result_4, %forwarded_5 = "neura.mac"(%1, %result) {placement = {x = 1 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
        %result_6, %forwarded_7 = "neura.mac"(%forwarded_5, %result_0) {placement = {x = 2 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
        %result_8, %forwarded_9 = "neura.mac"(%forwarded_7, %result_2) {placement = {x = 3 : i32, y = 2 : i32}} : (i32, i32) -> (i32, i32)
        %2 = "neura.load"(%arg6) {constants = array<i64: 2, 5, 8>, placement = {x = 0 : i32, y = 1 : i32}} : (memref<3x3xi32>) -> i32
        %result_10, %forwarded_11 = "neura.mac"(%2, %result_4) {placement = {x = 1 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
        %result_12, %forwarded_13 = "neura.mac"(%forwarded_11, %result_6) {placement = {x = 2 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
        %result_14, %forwarded_15 = "neura.mac"(%forwarded_13, %result_8) {placement = {x = 3 : i32, y = 1 : i32}} : (i32, i32) -> (i32, i32)
        "neura.store"(%result_10, %arg8) {constants = array<i64: 0, 3, 6>, placement = {x = 1 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
        "neura.store"(%result_12, %arg8) {constants = array<i64: 1, 4, 7>, placement = {x = 2 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
        "neura.store"(%result_14, %arg8) {constants = array<i64: 2, 5, 8>, placement = {x = 3 : i32, y = 0 : i32}} : (i32, memref<3x3xi32>) -> ()
        neura.yield
      }
      taskflow.yield done_writes(%arg5 : memref<3x3xi32>)
    }
    return %done_writes : memref<3x3xi32>
  }
}
""".strip()


MAPPED_IR = """
#map = affine_map<(d0, d1) -> (-d1 + 3, d0 - 1)>
module {
  func.func @ws_gemm_3x3(%arg0: memref<3x3xi32>, %arg1: memref<3x3xi32>, %arg2: memref<3x3xi32>) -> memref<3x3xi32> {
    %done_writes = taskflow.task @ws_gemm_3x3 will_reads(%arg0, %arg1 : memref<3x3xi32>, memref<3x3xi32>) will_writes(%arg2 : memref<3x3xi32>) [original_read_memrefs(%arg0, %arg1 : memref<3x3xi32>, memref<3x3xi32>), original_write_memrefs(%arg2 : memref<3x3xi32>)] : (memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) -> (memref<3x3xi32>) {
    ^bb0(%arg3: memref<3x3xi32>, %arg4: memref<3x3xi32>, %arg5: memref<3x3xi32>):
      neura.kernel inputs(%arg3, %arg4, %arg5 : memref<3x3xi32>, memref<3x3xi32>, memref<3x3xi32>) attributes {accelerator = "neura", kernel_metadata = {kind = "template", template = {name = "systolic_array", stationary = {kernel_input = 1 : i32, map = #map}}}, mapping_info = {compiled_ii = 1 : i32, mapping_mode = "spatial-only", mapping_strategy = "template", rec_mii = 1 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {
      ^bb0(%arg6: !neura.data<memref<3x3xi32>, i1>, %arg7: !neura.data<memref<3x3xi32>, i1>, %arg8: !neura.data<memref<3x3xi32>, i1>):
        %0 = "neura.load"(%arg6) {constants = array<i64: 0, 3, 6>, dfg_id = 0 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) {dfg_id = 4 : i32, mapping_locs = [{id = 38 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result, %forwarded = "neura.mac"(%1) {dfg_id = 7 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %2 = "neura.data_mov"(%forwarded) {dfg_id = 9 : i32, mapping_locs = [{id = 41 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_0, %forwarded_1 = "neura.mac"(%2) {dfg_id = 11 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %3 = "neura.data_mov"(%forwarded_1) {dfg_id = 15 : i32, mapping_locs = [{id = 44 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_2, %forwarded_3 = "neura.mac"(%3) {dfg_id = 18 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %4 = "neura.load"(%arg6) {constants = array<i64: 1, 4, 7>, dfg_id = 1 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
        %5 = "neura.data_mov"(%4) {dfg_id = 5 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %6 = "neura.data_mov"(%result) {dfg_id = 8 : i32, mapping_locs = [{id = 42 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_4, %forwarded_5 = "neura.mac"(%5, %6) {dfg_id = 10 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %7 = "neura.data_mov"(%forwarded_5) {dfg_id = 13 : i32, mapping_locs = [{id = 28 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %8 = "neura.data_mov"(%result_0) {dfg_id = 14 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_6, %forwarded_7 = "neura.mac"(%7, %8) {dfg_id = 17 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %9 = "neura.data_mov"(%forwarded_7) {dfg_id = 22 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %10 = "neura.data_mov"(%result_2) {dfg_id = 23 : i32, mapping_locs = [{id = 47 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_8, %forwarded_9 = "neura.mac"(%9, %10) {dfg_id = 26 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %11 = "neura.load"(%arg6) {constants = array<i64: 2, 5, 8>, dfg_id = 2 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<memref<3x3xi32>, i1>) -> !neura.data<i32, i1>
        %12 = "neura.data_mov"(%11) {dfg_id = 6 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %13 = "neura.data_mov"(%result_4) {dfg_id = 12 : i32, mapping_locs = [{id = 29 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_10, %forwarded_11 = "neura.mac"(%12, %13) {dfg_id = 16 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %14 = "neura.data_mov"(%forwarded_11) {dfg_id = 20 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %15 = "neura.data_mov"(%result_6) {dfg_id = 21 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_12, %forwarded_13 = "neura.mac"(%14, %15) {dfg_id = 25 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %16 = "neura.data_mov"(%forwarded_13) {dfg_id = 28 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %17 = "neura.data_mov"(%result_8) {dfg_id = 29 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %result_14, %forwarded_15 = "neura.mac"(%16, %17) {dfg_id = 31 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> (!neura.data<i32, i1>, !neura.data<i32, i1>)
        %18 = "neura.data_mov"(%result_10) {dfg_id = 19 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 0 : i32, invalid_iterations = 3 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        "neura.store"(%18, %arg8) {constants = array<i64: 0, 3, 6>, dfg_id = 24 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
        %19 = "neura.data_mov"(%result_12) {dfg_id = 27 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 0 : i32, invalid_iterations = 4 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        "neura.store"(%19, %arg8) {constants = array<i64: 1, 4, 7>, dfg_id = 30 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
        %20 = "neura.data_mov"(%result_14) {dfg_id = 32 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 0 : i32, invalid_iterations = 5 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        "neura.store"(%20, %arg8) {constants = array<i64: 2, 5, 8>, dfg_id = 33 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 0 : i32, invalid_iterations = 6 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<memref<3x3xi32>, i1>) -> ()
        neura.yield {dfg_id = 3 : i32}
      }
      taskflow.yield done_writes(%arg5 : memref<3x3xi32>)
    }
    return %done_writes : memref<3x3xi32>
  }
}
""".strip()


def test_lowers_systolic_gemm_to_exact_pre_mapping_ir():
    actual = lowering.lower(ws_gemm_3x3, argument_types=(synl.i32[3, 3],) * 3)
    assert actual.strip() == PRE_MAPPING_IR


def test_compiles_systolic_gemm_to_exact_mapped_ir():
    actual = synapse.compile(
        ws_gemm_3x3, target="neura", argument_types=(synl.i32[3, 3],) * 3
    )
    assert actual.strip() == MAPPED_IR


def test_generates_configured_memory_and_mac_instructions(tmp_path):
    mapped = synapse.compile(
        ws_gemm_3x3, target="neura", argument_types=(synl.i32[3, 3],) * 3
    )
    command = [
        str(REPOSITORY_ROOT / "build/amoeba/tools/mlir-amoeba-opt/mlir-amoeba-opt"),
        f"--neura-architecture-spec={NEURA_ROOT / 'test/arch_spec/architecture.yaml'}",
        "--generate-code",
        "-o",
        str(tmp_path / "mapped.mlir"),
    ]
    completed = subprocess.run(
        command, input=mapped, text=True, capture_output=True, cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr
    assembly = (tmp_path / "tmp-generated-instructions.asm").read_text()
    assert assembly.count("  LOAD,") == 3
    assert assembly.count("  STORE,") == 3
    assert assembly.count("  MUL_ADD,") == 6
    assert assembly.count("  MUL,") == 3
    assert "address(arg0, 0, 3, 6)" in assembly
    assert "address(arg2, 2, 5, 8)" in assembly
    assert "value(arg1, 8)" in assembly

    invalid = mapped.replace("array<i64: 0, 3, 6>", "array<i64: 0, 3, 9>")
    completed = subprocess.run(
        command, input=invalid, text=True, capture_output=True, cwd=tmp_path
    )
    assert completed.returncode != 0
    assert "constant element offset is out of bounds" in completed.stderr
