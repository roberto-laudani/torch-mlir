// ============================================================================
//
// This test is dedicated to com.microsoft.MatMulIntegerToFloat - for now,
// the pre-existing support for MatMulInteger is tested within a common
// 'simple_ops*' LIT file.
//
// ============================================================================

// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// TODO: Test cases should be split out into more dedicated scenarios.

// CHECK-LABEL: @test_matmul_integer_to_float_dynamic_with_bias_0d_args
// CHECK-SAME: (%[[LHS_ARG:.+]]: !torch.vtensor<[?,?,3],ui8>, %[[RHS_ARG:.+]]: !torch.vtensor<[3,8],ui8>,
// CHECK-SAME:  %[[LHS_SCALE_ARG:.+]]: !torch.vtensor<[],f32>, %[[RHS_SCALE_ARG:.+]]: !torch.vtensor<[],f32>,
// CHECK-SAME:  %[[LHS_ZP_ARG:.+]]: !torch.vtensor<[],ui8>, %[[RHS_ZP_ARG:.+]]: !torch.vtensor<[],ui8>,
// CHECK-SAME:  %[[BIAS_ARG:.+]]: !torch.vtensor<[8],f32>)
func.func @test_matmul_integer_to_float_dynamic_with_bias_0d_args(
    %arg0: !torch.vtensor<[?,?,3],ui8>, %arg1: !torch.vtensor<[3,8],ui8>,
    %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[],f32>,
    %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.vtensor<[],ui8>,
    %arg6: !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,?,8],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!torch.vtensor<[?,?,3],ui8>, !torch.vtensor<[3,8],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[],ui8>, !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,?,8],f32>
  // CHECK:      %[[INT_1_ALPHA:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[LHS_MINUS_ZP:.+]] = torch.aten.sub.Tensor %[[LHS_ARG]], %[[LHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,3],ui8>
  // CHECK-DAG:  %[[RHS_MINUS_ZP:.+]] = torch.aten.sub.Tensor %[[RHS_ARG]], %[[RHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[3,8],ui8>
  // CHECK-DAG:  %[[MATMUL_INT:.+]] = torch.aten.matmul %[[LHS_MINUS_ZP]], %[[RHS_MINUS_ZP]] : {{.*}} -> !torch.vtensor<[?,?,8],si32>
  // CHECK-DAG:  %[[MATMUL_FP_CAST:.+]] = torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK-DAG:  %[[SCALE_PRODUCT:.+]] = torch.aten.mul.Tensor %[[LHS_SCALE_ARG]], %[[RHS_SCALE_ARG]] : {{.*}} -> !torch.vtensor<[],f32>
  // CHECK:      %[[MATMUL_DEQ_RES:.+]] = torch.aten.mul.Tensor %[[MATMUL_FP_CAST]], %[[SCALE_PRODUCT]] : {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK:      %[[RES_BIASED:.+]] = torch.aten.add.Tensor %[[MATMUL_DEQ_RES]], %[[BIAS_ARG]], {{.*}} : {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK:      return %[[RES_BIASED:.+]] : !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}

// -----

// CHECK-LABEL: @test_matmul_integer_to_float_mixed_signedness_0d_args
// CHECK-SAME: (%[[LHS_ARG:.+]]: !torch.vtensor<[4,3],ui8>, %[[RHS_ARG:.+]]: !torch.vtensor<[3,2],si8>,
// CHECK-SAME:  %[[LHS_SCALE_ARG:.+]]: !torch.vtensor<[],f32>, %[[RHS_SCALE_ARG:.+]]: !torch.vtensor<[2],f32>,
// CHECK-SAME:  %[[LHS_ZP_ARG:.+]]: !torch.vtensor<[],ui8>, %[[RHS_ZP_ARG:.+]]: !torch.vtensor<[2],si8>)
func.func @test_matmul_integer_to_float_mixed_signedness_0d_args(
    %arg0: !torch.vtensor<[4,3],ui8>, %arg1: !torch.vtensor<[3,2],si8>,
    %arg2: !torch.vtensor<[],f32>, %arg3: !torch.vtensor<[2],f32>,
    %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.vtensor<[2],si8>) -> !torch.vtensor<[4,2],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) :
           (!torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[2],f32>,
            !torch.vtensor<[],ui8>, !torch.vtensor<[2],si8>) -> !torch.vtensor<[4,2],f32>
  // CHECK:      %[[INT_1_ALPHA:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[LHS_MINUS_ZP:.+]] = torch.aten.sub.Tensor %[[LHS_ARG]], %[[LHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[4,3],ui8>
  // CHECK-DAG:  %[[RHS_MINUS_ZP:.+]] = torch.aten.sub.Tensor %[[RHS_ARG]], %[[RHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[3,2],si8>
  // CHECK-DAG:  %[[MATMUL_INT:.+]] = torch.aten.matmul %[[LHS_MINUS_ZP]], %[[RHS_MINUS_ZP]] : {{.*}} -> !torch.vtensor<[4,2],si32>
  // CHECK-DAG:  %[[MATMUL_FP_CAST:.+]] = torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : {{.*}} -> !torch.vtensor<[4,2],f32>
  // CHECK-DAG:  %[[SCALE_PRODUCT:.+]] = torch.aten.mul.Tensor %[[LHS_SCALE_ARG]], %[[RHS_SCALE_ARG]] : {{.*}} -> !torch.vtensor<[2],f32>
  // CHECK:      %[[MATMUL_DEQ_RES:.+]] = torch.aten.mul.Tensor %[[MATMUL_FP_CAST]], %[[SCALE_PRODUCT]] : {{.*}} -> !torch.vtensor<[4,2],f32>
  // CHECK:      return %[[MATMUL_DEQ_RES]] : !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// CHECK-LABEL: @test_matmul_integer_to_float_fp16_scalar_args
// CHECK-SAME: (%[[LHS_ARG:.+]]: !torch.vtensor<[4,3],si8>, %[[RHS_ARG:.+]]: !torch.vtensor<[3,2],si8>,
// CHECK-SAME:  %[[LHS_SCALE_ARG:.+]]: !torch.float, %[[RHS_SCALE_ARG:.+]]: !torch.float,
// CHECK-SAME:  %[[LHS_ZP_ARG:.+]]: !torch.int, %[[RHS_ZP_ARG:.+]]: !torch.int)
func.func @test_matmul_integer_to_float_fp16_scalar_args(
    %arg0: !torch.vtensor<[4,3],si8>, %arg1: !torch.vtensor<[3,2],si8>,
    %arg2: !torch.float, %arg3: !torch.float,
    %arg4: !torch.int, %arg5: !torch.int) -> !torch.vtensor<[4,2],f16>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {

  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.vtensor<[4,3],si8>, !torch.vtensor<[3,2],si8>, !torch.float, !torch.float, !torch.int, !torch.int) -> !torch.vtensor<[4,2],f16>
  // CHECK:      %[[INT_1_ALPHA:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[LHS_MINUS_ZP:.+]] = torch.aten.sub.Scalar %[[LHS_ARG]], %[[LHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[4,3],si8>
  // CHECK-DAG:  %[[RHS_MINUS_ZP:.+]] = torch.aten.sub.Scalar %[[RHS_ARG]], %[[RHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[3,2],si8>
  // CHECK-DAG:  %[[MATMUL_INT:.+]] = torch.aten.matmul %[[LHS_MINUS_ZP]], %[[RHS_MINUS_ZP]] : {{.*}} -> !torch.vtensor<[4,2],si32>
  // CHECK-DAG:  %[[MATMUL_FP_CAST:.+]] = torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : {{.*}} -> !torch.vtensor<[4,2],f16>
  // CHECK-DAG:  %[[SCALE_PRODUCT:.+]] = torch.aten.mul %[[LHS_SCALE_ARG]], %[[RHS_SCALE_ARG]] : {{.*}} -> !torch.float
  // CHECK:      %[[MATMUL_DEQ_RES:.+]] = torch.aten.mul.Scalar %[[MATMUL_FP_CAST]], %[[SCALE_PRODUCT]] : {{.*}} -> !torch.vtensor<[4,2],f16>
  // CHECK:      return %[[MATMUL_DEQ_RES]] : !torch.vtensor<[4,2],f16>
  return %0 : !torch.vtensor<[4,2],f16>
}

// -----

// CHECK-LABEL: @test_matmul_integer_to_float_dynamic_mixed_args
// CHECK-SAME: (%[[LHS_ARG:.+]]: !torch.vtensor<[?,?,3],ui8>, %[[RHS_ARG:.+]]: !torch.vtensor<[3,8],ui8>,
// CHECK-SAME:  %[[LHS_SCALE_ARG:.+]]: !torch.float, %[[RHS_SCALE_ARG:.+]]: !torch.vtensor<[8],f32>,
// CHECK-SAME:  %[[LHS_ZP_ARG:.+]]: !torch.vtensor<[],ui8>, %[[RHS_ZP_ARG:.+]]: !torch.int)
func.func @test_matmul_integer_to_float_dynamic_mixed_args(
    %arg0: !torch.vtensor<[?,?,3],ui8>, %arg1: !torch.vtensor<[3,8],ui8>,
    %arg2: !torch.float, %arg3: !torch.vtensor<[8],f32>,
    %arg4: !torch.vtensor<[],ui8>, %arg5: !torch.int) -> !torch.vtensor<[?,?,8],f32>
  attributes {
    torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.MatMulIntegerToFloat"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!torch.vtensor<[?,?,3],ui8>, !torch.vtensor<[3,8],ui8>, !torch.float, !torch.vtensor<[8],f32>, !torch.vtensor<[],ui8>, !torch.int) -> !torch.vtensor<[?,?,8],f32>
  // CHECK:      %[[INT_1_ALPHA:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[LHS_MINUS_ZP:.+]] = torch.aten.sub.Tensor %[[LHS_ARG]], %[[LHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,3],ui8>
  // CHECK-DAG:  %[[RHS_MINUS_ZP:.+]] = torch.aten.sub.Scalar %[[RHS_ARG]], %[[RHS_ZP_ARG]], %[[INT_1_ALPHA]] : {{.*}} -> !torch.vtensor<[3,8],ui8>
  // CHECK-DAG:  %[[MATMUL_INT:.+]] = torch.aten.matmul %[[LHS_MINUS_ZP]], %[[RHS_MINUS_ZP]] : {{.*}} -> !torch.vtensor<[?,?,8],si32>
  // CHECK-DAG:  %[[MATMUL_FP_CAST:.+]] = torch.aten.to.dtype %[[MATMUL_INT]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK-DAG:  %[[SCALE_PRODUCT:.+]] = torch.aten.mul.Scalar %[[RHS_SCALE_ARG]], %[[LHS_SCALE_ARG]] : {{.*}} -> !torch.vtensor<[8],f32>
  // CHECK:      %[[MATMUL_DEQ_RES:.+]] = torch.aten.mul.Tensor %[[MATMUL_FP_CAST]], %[[SCALE_PRODUCT]] : {{.*}} -> !torch.vtensor<[?,?,8],f32>
  // CHECK:      return %[[MATMUL_DEQ_RES]] : !torch.vtensor<[?,?,8],f32>
  return %0 : !torch.vtensor<[?,?,8],f32>
}
