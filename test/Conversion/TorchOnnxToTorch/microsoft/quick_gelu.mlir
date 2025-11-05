// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_quick_gelu_float32
func.func @test_quick_gelu_float32(%arg0: !torch.vtensor<[?,?,16],f32>) -> !torch.vtensor<[?,?,16],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK: %[[ALPHA_EXTRACT:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[SCALED_ARG:.+]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_EXTRACT]]
  // CHECK: %[[SIGMOID:.+]] = torch.aten.sigmoid %[[SCALED_ARG]]
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %arg0, %[[SIGMOID]]
  // CHECK: return %[[RESULT]] : !torch.vtensor<[?,?,16],f32>
  %0 = torch.operator "onnx.QuickGelu"(%arg0) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[?,?,16],f32>) -> !torch.vtensor<[?,?,16],f32>
  return %0 : !torch.vtensor<[?,?,16],f32> 
}

// -----

// CHECK-LABEL: @test_quick_gelu_float16
func.func @test_quick_gelu_float16(%arg0: !torch.vtensor<[?,128],f16>) -> !torch.vtensor<[?,128],f16>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK: %[[ALPHA_EXTRACT:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[SCALED_ARG:.+]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_EXTRACT]]
  // CHECK: %[[SIGMOID:.+]] = torch.aten.sigmoid %[[SCALED_ARG]]
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %arg0, %[[SIGMOID]]
  // CHECK: return %[[RESULT]] : !torch.vtensor<[?,128],f16>
  %0 = torch.operator "onnx.QuickGelu"(%arg0) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[?,128],f16>) -> !torch.vtensor<[?,128],f16>
  return %0 : !torch.vtensor<[?,128],f16>
}

// -----

// CHECK-LABEL: @test_quick_gelu_bfloat16
func.func @test_quick_gelu_bfloat16(%arg0: !torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,4],bf16>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK: %[[ALPHA_EXTRACT:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[SCALED_ARG:.+]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_EXTRACT]]
  // CHECK: %[[SIGMOID:.+]] = torch.aten.sigmoid %[[SCALED_ARG]]
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %arg0, %[[SIGMOID]]
  // CHECK: return %[[RESULT]] : !torch.vtensor<[2,4],bf16>
  %0 = torch.operator "onnx.QuickGelu"(%arg0) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,4],bf16>
  return %0 : !torch.vtensor<[2,4],bf16>
}

// -----

// CHECK-LABEL: @test_quick_gelu_float64
func.func @test_quick_gelu_float64(%arg0: !torch.vtensor<[4,4,16],f64>) -> !torch.vtensor<[4,4,16],f64>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK: %[[ALPHA_EXTRACT:.+]] = torch.constant.float 1.000000e+00
  // CHECK: %[[SCALED_ARG:.+]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_EXTRACT]]
  // CHECK: %[[SIGMOID:.+]] = torch.aten.sigmoid %[[SCALED_ARG]]
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %arg0, %[[SIGMOID]]
  // CHECK: return %[[RESULT]] : !torch.vtensor<[4,4,16],f64>
  %0 = torch.operator "onnx.QuickGelu"(%arg0) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[4,4,16],f64>) -> !torch.vtensor<[4,4,16],f64>
  return %0 : !torch.vtensor<[4,4,16],f64>
}

// -----

// CHECK-LABEL: @test_quick_gelu_float64_alpha_0p5
func.func @test_quick_gelu_float64_alpha_0p5(%arg0: !torch.vtensor<[4,4,16],f64>) -> !torch.vtensor<[4,4,16],f64>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK: %[[ALPHA_EXTRACT:.+]] = torch.constant.float 5.000000e-01
  // CHECK: %[[SCALED_ARG:.+]] = torch.aten.mul.Scalar %arg0, %[[ALPHA_EXTRACT]]
  // CHECK: %[[SIGMOID:.+]] = torch.aten.sigmoid %[[SCALED_ARG]]
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %arg0, %[[SIGMOID]]
  // CHECK: return %[[RESULT]] : !torch.vtensor<[4,4,16],f64>
  %0 = torch.operator "onnx.QuickGelu"(%arg0) {torch.onnx.alpha = 5.000000e-01 : f32} : (!torch.vtensor<[4,4,16],f64>) -> !torch.vtensor<[4,4,16],f64>
  return %0 : !torch.vtensor<[4,4,16],f64>
}
