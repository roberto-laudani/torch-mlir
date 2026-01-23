// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_quantized_matmul_to_float
// CHECK-SAME: %[[LHS:.+]]: !torch.vtensor<[4,3],ui8>, %[[RHS:.+]]: !torch.vtensor<[3,2],ui8>,
// CHECK-SAME: %[[LHS_SCALE:.+]]: !torch.vtensor<[],f32>, %[[RHS_SCALE:.+]]: !torch.vtensor<[],f32>,
// CHECK-SAME: %[[LHS_ZP:.+]]: !torch.vtensor<[],ui8>, %[[RHS_ZP:.+]]: !torch.vtensor<[],ui8>
func.func @test_quantized_matmul_to_float(
    %arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],ui8>,
    %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>,
    %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,2],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulIntegerToFloat"
  // Emit torch.quantized_matmul op
  // CHECK: %[[MATMUL_INT:.+]] = torch.quantized_matmul %[[LHS]], %[[RHS]], %[[LHS_ZP]], %[[RHS_ZP]] : !torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],ui8> -> !torch.vtensor<[4,2],si32>
  // Convert to float
  // CHECK: %[[INT_RESULT:.+]] = torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : !torch.vtensor<[4,2],si32>{{.+}} -> !torch.vtensor<[4,2],f32>
  // Multiply scales
  // CHECK: %[[SCALE_PROD:.+]] = torch.aten.mul.Tensor %[[LHS_SCALE]], %[[RHS_SCALE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // Apply scale to result
  // CHECK: %[[RESULT:.+]] = torch.aten.mul.Tensor %[[INT_RESULT]], %[[SCALE_PROD]] : !torch.vtensor<[4,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[4,2],f32>
  // CHECK: return %[[RESULT]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) :
           (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>,
            !torch.vtensor<[],ui8>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// CHECK-LABEL: @test_quantized_matmul_to_float_with_bias
// CHECK-SAME: %[[LHS:.+]]: !torch.vtensor<[4,3],ui8>, %[[RHS:.+]]: !torch.vtensor<[3,2],si8>,
// CHECK-SAME: %[[LHS_SCALE:.+]]: !torch.vtensor<[],f32>, %[[RHS_SCALE:.+]]: !torch.vtensor<[],f32>,
// CHECK-SAME: %[[LHS_ZP:.+]]: !torch.vtensor<[],ui8>, %[[RHS_ZP:.+]]: !torch.vtensor<[],si8>,
// CHECK-SAME: %[[BIAS:.+]]: !torch.vtensor<[2],f32>
func.func @test_quantized_matmul_to_float_with_bias(
    %arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],si8>,
    %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>,
    %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.vtensor<[],si8>,
    %arg6: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4,2],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  // CHECK-NOT: torch.operator "onnx.MatMulIntegerToFloat"
  // Emit torch.quantized_matmul op
  // CHECK: %[[MATMUL_INT:.+]] = torch.quantized_matmul %[[LHS]], %[[RHS]], %[[LHS_ZP]], %[[RHS_ZP]] : !torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8> -> !torch.vtensor<[4,2],si32>
  // Convert to float
  // CHECK: torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : !torch.vtensor<[4,2],si32>{{.+}} -> !torch.vtensor<[4,2],f32>
  // Multiply scales
  // CHECK: %[[SCALE_PROD:.+]] = torch.aten.mul.Tensor %[[LHS_SCALE]], %[[RHS_SCALE]] : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  // Apply scale
  // CHECK: %[[SCALED:.+]] = torch.aten.mul.Tensor %{{.+}}, %[[SCALE_PROD]] : !torch.vtensor<[4,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[4,2],f32>
  // Add bias
  // CHECK: %[[RESULT:.+]] = torch.aten.add.Tensor %[[SCALED]], %[[BIAS]], %{{.+}} : !torch.vtensor<[4,2],f32>, !torch.vtensor<[2],f32>, !torch.float -> !torch.vtensor<[4,2],f32>
  // CHECK: return %[[RESULT]]
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) :
           (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>,
            !torch.vtensor<[],ui8>, !torch.vtensor<[],si8>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}
