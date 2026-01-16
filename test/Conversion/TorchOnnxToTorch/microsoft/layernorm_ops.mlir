// ============================================================================
//
// This test is dedicated to various com.microsoft versions of
// LayerNormalization using RMS normalization (no mean subtraction).
//
// RMS norm decomposition: y = x * rsqrt(mean(x²) + eps) * scale
//
// ============================================================================

// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_simplified_layer_norm_dynamic
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_dynamic(%arg0: !torch.vtensor<[?,?,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,?,2048],f32>
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 5.000000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>) -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[EPS:.+]] = torch.constant.float 5.000000e-01
  // CHECK:      %[[DIM:.+]] = torch.constant.int 2
  // CHECK:      %[[DIM_LIST:.+]] = torch.prim.ListConstruct %[[DIM]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INPUT]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,2048],f32> -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[TRUE:.+]] = torch.constant.bool true
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim %[[X_SQ]], %[[DIM_LIST]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[ONE:.+]] = torch.constant.int 1
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]], %[[ONE]] : !torch.vtensor<[?,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]] : !torch.vtensor<[?,?,1],f32> -> !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INV_STD]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32> -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[WEIGHT]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32> -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,?,2048],f32>
  return %0 : !torch.vtensor<[?,?,2048],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_single_result
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,2,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_single_result(%arg0: !torch.vtensor<[?,2,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,2,2048],f32>
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 2.500000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,2,2048],f32>, !torch.vtensor<[2048],f32>) -> !torch.vtensor<[?,2,2048],f32>
  // CHECK:      %[[EPS:.+]] = torch.constant.float 2.500000e-01
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INPUT]]
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]]
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]]
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INV_STD]]
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[WEIGHT]]
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,2,2048],f32>
  return %0 : !torch.vtensor<[?,2,2048],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_with_inv_std_var
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[2,4,8],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[8],f32>)
func.func @test_simplified_layer_norm_with_inv_std_var(%arg0: !torch.vtensor<[2,4,8],f32>, %arg1: !torch.vtensor<[8],f32>)
  -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0:2 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 1.250000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  // CHECK:      %[[EPS:.+]] = torch.constant.float 1.250000e-01
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INPUT]]
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]]
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]] : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[INPUT]], %[[INV_STD]]
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[WEIGHT]]
  // CHECK:      return %[[RES]], %[[INV_STD]] : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
  return %0#0, %0#1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_single_output
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SKIP:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SCALE:.+]]: !torch.vtensor<[2048],f32>, %[[BIAS:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_skip_simplified_layer_norm_single_output(
  %arg0: !torch.vtensor<[?,?,2048],f32>,
  %arg1: !torch.vtensor<[?,?,2048],f32>,
  %arg2: !torch.vtensor<[2048],f32>,
  %arg3: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,?,2048],f32>
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2, %arg3)
           {torch.onnx.epsilon = 5.000000e-01 : f32}
           : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>, !torch.vtensor<[2048],f32>)
           -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_SUM_TMP:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[SKIP_SUM:.+]] = torch.aten.add.Tensor %[[SKIP_SUM_TMP]], %[[BIAS]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[EPS:.+]] = torch.constant.float 5.000000e-01
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[SKIP_SUM]]
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]]
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]]
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[INV_STD]]
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[SCALE]]
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,?,2048],f32>
  return %0 : !torch.vtensor<[?,?,2048],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_two_outputs
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SKIP:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SCALE:.+]]: !torch.vtensor<[8],f32>)
func.func @test_skip_simplified_layer_norm_two_outputs(
  %arg0: !torch.vtensor<[2,4,8],f32>,
  %arg1: !torch.vtensor<[2,4,8],f32>,
  %arg2: !torch.vtensor<[8],f32>)
  -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:2 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2)
           {torch.onnx.epsilon = 1.250000e-01 : f32}
           : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>)
           -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_SUM:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[2,4,8],f32>
  // CHECK:      %[[EPS:.+]] = torch.constant.float 1.250000e-01
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[SKIP_SUM]]
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim {{.*}} -> !torch.vtensor<[2,4,1],f32>
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]]
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]]
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[INV_STD]]
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[SCALE]]
  // CHECK:      return %[[RES]], %[[MEAN_SQ]] : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
  return %0#0, %0#1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_with_all_intermediates
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SKIP:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SCALE:.+]]: !torch.vtensor<[8],f32>)
func.func @test_skip_simplified_layer_norm_with_all_intermediates(
  %arg0: !torch.vtensor<[2,4,8],f32>,
  %arg1: !torch.vtensor<[2,4,8],f32>,
  %arg2: !torch.vtensor<[8],f32>)
  -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,8],f32>)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:4 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2)
           {torch.onnx.epsilon = 6.250000e-02 : f32}
           : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>)
           -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,8],f32>)
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_SUM:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[2,4,8],f32>
  // CHECK:      %[[EPS:.+]] = torch.constant.float 6.250000e-02
  // CHECK:      %[[X_SQ:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[SKIP_SUM]]
  // CHECK:      %[[MEAN_SQ:.+]] = torch.aten.mean.dim {{.*}} -> !torch.vtensor<[2,4,1],f32>
  // CHECK:      %[[MEAN_PLUS_EPS:.+]] = torch.aten.add.Scalar %[[MEAN_SQ]], %[[EPS]]
  // CHECK:      %[[INV_STD:.+]] = torch.aten.rsqrt %[[MEAN_PLUS_EPS]] : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
  // CHECK:      %[[NORM:.+]] = torch.aten.mul.Tensor %[[SKIP_SUM]], %[[INV_STD]]
  // CHECK:      %[[RES:.+]] = torch.aten.mul.Tensor %[[NORM]], %[[SCALE]]
  // CHECK:      return %[[RES]], %[[MEAN_SQ]], %[[INV_STD]], %[[SKIP_SUM]] : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,8],f32>
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,8],f32>
}
