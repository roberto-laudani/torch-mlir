// RUN: torch-mlir-opt %s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// -----
// bits=4, f32 lhs, f32 unpacked zero-points, no bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_bits4_zp_f32
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,4],f32>
func.func @test_matmul_n_bits_bits4_zp_f32(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %{{.+}} {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=4, f32 lhs, f32 unpacked zero-points, f32 bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_bits4_zp_f32_bias
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,4],f32>
// CHECK-SAME: %[[BIAS:.+]]: !torch.vtensor<[64],f32>
func.func @test_matmul_n_bits_bits4_zp_f32_bias(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,4],f32>,
    %arg4: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %[[BIAS]] {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %none = torch.constant.none
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,4],f32>, !torch.none, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=4, f32 lhs, packed ui8 zero-points, f32 bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_bits4_zp_ui8_bias
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,2],ui8>
// CHECK-SAME: %[[BIAS:.+]]: !torch.vtensor<[64],f32>
func.func @test_matmul_n_bits_bits4_zp_ui8_bias(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,2],ui8>,
    %arg4: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %[[BIAS]] {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %none = torch.constant.none
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,2],ui8>, !torch.none, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=4, f32 lhs, no zero-points, f32 bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_bits4_no_zp_bias
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>
// CHECK-SAME: %[[BIAS:.+]]: !torch.vtensor<[64],f32>
func.func @test_matmul_n_bits_bits4_no_zp_bias(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %{{.+}}, %[[BIAS]] {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %none = torch.constant.none
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %none, %none, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.none, !torch.none, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=8, f32 lhs, f32 unpacked zero-points, no bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_bits8
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,32],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,4],f32>
func.func @test_matmul_n_bits_bits8(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,32],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %{{.+}} {accuracy_level = 0 : si64, bits = 8 : si64, block_size = 32 : si64}
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 8 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,32],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=4, f32 lhs, packed ui8 zero-points, f32 bias, accuracy_level=4
// CHECK-LABEL: @test_matmul_n_bits_accuracy_level_4
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,2],ui8>
// CHECK-SAME: %[[BIAS:.+]]: !torch.vtensor<[64],f32>
func.func @test_matmul_n_bits_accuracy_level_4(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,2],ui8>,
    %arg4: !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %[[BIAS]] {accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64}
  %none = torch.constant.none
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3, %none, %arg4) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 4 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,2],ui8>, !torch.none, !torch.vtensor<[64],f32>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// bits=4, f16 lhs, f16 unpacked zero-points, no bias, accuracy_level=0
// CHECK-LABEL: @test_matmul_n_bits_f16
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f16>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f16>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,4],f16>
func.func @test_matmul_n_bits_f16(
    %arg0: !torch.vtensor<[?,?,128],f16>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f16>, %arg3: !torch.vtensor<[64,4],f16>) -> !torch.vtensor<[?,?,64],f16>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %{{.+}} {accuracy_level = 2 : si64, bits = 4 : si64, block_size = 32 : si64}
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 2 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f16>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f16>, !torch.vtensor<[64,4],f16>) -> !torch.vtensor<[?,?,64],f16>
  return %0 : !torch.vtensor<[?,?,64],f16>
}

// -----
// 1d scales and zero-points
// CHECK-LABEL: @test_matmul_n_bits_1d_scales_zp
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[?,?,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[256],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[128],ui8>
func.func @test_matmul_n_bits_1d_scales_zp(
    %arg0: !torch.vtensor<[?,?,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[256],f32>, %arg3: !torch.vtensor<[128],ui8>) -> !torch.vtensor<[?,?,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK-DAG: %[[SCALES_RESHAPED:.+]] = torch.aten.reshape %[[SCALES]], %{{.+}} : !torch.vtensor<[256],f32>, !torch.list<int> -> !torch.vtensor<[64,4],f32>
  // CHECK-DAG: %[[ZP_RESHAPED:.+]] = torch.aten.reshape %[[ZERO_POINTS]], %{{.+}} : !torch.vtensor<[128],ui8>, !torch.list<int> -> !torch.vtensor<[64,2],ui8>
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES_RESHAPED]], %[[ZP_RESHAPED]], %{{.+}} {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[?,?,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[256],f32>, !torch.vtensor<[128],ui8>) -> !torch.vtensor<[?,?,64],f32>
  return %0 : !torch.vtensor<[?,?,64],f32>
}

// -----
// Fully static lhs shape, bits=4, f32 unpacked zero-points, no bias
// CHECK-LABEL: @test_matmul_n_bits_static_shape
// CHECK-SAME: %[[A:.+]]: !torch.vtensor<[1,16,128],f32>
// CHECK-SAME: %[[B:.+]]: !torch.vtensor<[64,4,16],ui8>
// CHECK-SAME: %[[SCALES:.+]]: !torch.vtensor<[64,4],f32>, %[[ZERO_POINTS:.+]]: !torch.vtensor<[64,4],f32>
func.func @test_matmul_n_bits_static_shape(
    %arg0: !torch.vtensor<[1,16,128],f32>, %arg1: !torch.vtensor<[64,4,16],ui8>,
    %arg2: !torch.vtensor<[64,4],f32>, %arg3: !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[1,16,64],f32>
  attributes {
    torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64,
    torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
    torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulNBits"
  // CHECK: torch.matmul_n_bits %[[A]], %[[B]], %[[SCALES]], %[[ZERO_POINTS]], %{{.+}} {accuracy_level = 0 : si64, bits = 4 : si64, block_size = 32 : si64}
  %0 = torch.operator "onnx.MatMulNBits"(%arg0, %arg1, %arg2, %arg3) {torch.onnx.K = 128 : si64, torch.onnx.N = 64 : si64, torch.onnx.accuracy_level = 0 : si64, torch.onnx.bits = 4 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[1,16,128],f32>, !torch.vtensor<[64,4,16],ui8>, !torch.vtensor<[64,4],f32>, !torch.vtensor<[64,4],f32>) -> !torch.vtensor<[1,16,64],f32>
  return %0 : !torch.vtensor<[1,16,64],f32>
}
