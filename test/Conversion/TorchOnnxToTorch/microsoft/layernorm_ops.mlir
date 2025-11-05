// ============================================================================
//
// This test is dedicated to various com.microsoft versions of
// LayerNormalization.
// For now, the core onnx.LayerNormalization op itself is tested within a
// common 'simple_ops*' LIT file.
//
// ============================================================================

// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_simplified_layer_norm_dynamic
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_dynamic(%arg0: !torch.vtensor<[?,?,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>)
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0:3 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 5.000000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>) -> (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 5.000000e-01
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  // CHECK:      return %[[RES]], %[[MEAN]], %[[RSTD]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  return %0#0, %0#1, %0#2 : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_dynbatch_axis1_single_result
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,2,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_dynbatch_axis1_single_result(%arg0: !torch.vtensor<[?,2,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,2,2048],f32>
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = 1 : si64, torch.onnx.epsilon = 2.500000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,2,2048],f32>, !torch.vtensor<[2048],f32>) -> !torch.vtensor<[?,2,2048],f32>
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 2.500000e-01
  // CHECK:      %[[INT_2:.+]] = torch.constant.int 2
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2]], %[[INT_2048]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,2,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,2,2048],f32>, !torch.vtensor<[?,1,1],f32>, !torch.vtensor<[?,1,1],f32>
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,2,2048],f32>
  return %0 : !torch.vtensor<[?,2,2048],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_axis1_dynamic_norm
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_axis1_dynamic_norm(
  %arg0: !torch.vtensor<[?,?,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,?,2048],f32>
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = 1 : si64, torch.onnx.epsilon = 6.250000e-02 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>) -> !torch.vtensor<[?,?,2048],f32>
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[EPS:.+]] = torch.constant.float 6.250000e-02
  // CHECK-DAG:  %[[IDX1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[SZ1:.+]] = torch.aten.size.int %[[INPUT]], %[[IDX1]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[SZ1]], %[[INT_2048]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %{{.*}}, %{{.*}} = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,1,1],f32>, !torch.vtensor<[?,1,1],f32>
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,?,2048],f32>
  return %0 : !torch.vtensor<[?,?,2048],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_static_axis2_only_mean
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[2,4,8],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[8],f32>)
func.func @test_simplified_layer_norm_static_axis2_only_mean(%arg0: !torch.vtensor<[2,4,8],f32>, %arg1: !torch.vtensor<[8],f32>)
  -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0:2 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = 2 : si64, torch.onnx.epsilon = 1.250000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>) -> (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 1.250000e-01
  // CHECK:      %[[INT_8:.+]] = torch.constant.int 8
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_8]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[2,4,8],f32>, !torch.list<int>, !torch.vtensor<[8],f32>, !torch.none, !torch.float -> !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>
  // CHECK:      return %[[RES]], %[[MEAN]] : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
  return %0#0, %0#1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_dynamic_none_results
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_dynamic_none_results(%arg0: !torch.vtensor<[?,?,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> !torch.vtensor<[?,?,2048],f32>
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0:3 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 7.500000e-01 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>) -> (!torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.none)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 7.500000e-01
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[NONE_0:.+]] = torch.constant.none
  // CHECK:      %[[NONE_1:.+]] = torch.constant.none
  // CHECK:      return %[[RES]] : !torch.vtensor<[?,?,2048],f32>
  return %0#0 : !torch.vtensor<[?,?,2048],f32>
}

// -----

// CHECK-LABEL: @test_simplified_layer_norm_dynamic_only_isd
// CHECK-SAME:  (%[[INPUT:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[WEIGHT:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_simplified_layer_norm_dynamic_only_isd(%arg0: !torch.vtensor<[?,?,2048],f32>, %arg1: !torch.vtensor<[2048],f32>)
  -> (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>)
  attributes {
    torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""
  } {
  %0:3 = torch.operator "onnx.SimplifiedLayerNormalization"(%arg0, %arg1) {
      torch.onnx.axis = -1 : si64, torch.onnx.epsilon = 6.250000e-02 : f32, torch.onnx.stash_type = 1 : si64
  } : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>) -> (!torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.vtensor<[?,?,1],f32>)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 6.250000e-02
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[NORM_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[INPUT]], %[[NORM_SHAPE]], %[[WEIGHT]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[NONE_0:.+]] = torch.constant.none
  // CHECK:      return %[[RES]], %[[RSTD]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>
  return %0#0, %0#2 : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_dynamic_all_results
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SKIP:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SCALE:.+]]: !torch.vtensor<[2048],f32>, %[[BIAS:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_skip_simplified_layer_norm_dynamic_all_results(
  %arg0: !torch.vtensor<[?,?,2048],f32>,
  %arg1: !torch.vtensor<[?,?,2048],f32>,
  %arg2: !torch.vtensor<[2048],f32>,
  %arg3: !torch.vtensor<[2048],f32>)
  -> (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:4 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2, %arg3)
           {torch.onnx.epsilon = 5.000000e-01 : f32}
           : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>, !torch.vtensor<[2048],f32>)
           -> (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>)
  // CHECK:      %[[EPS:.+]] = torch.constant.float 5.000000e-01
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[X]], %[[SHAPE]], %[[SCALE]], %[[BIAS]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.vtensor<[2048],f32>, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_RES_TMP:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      %[[SKIP_RES:.+]] = torch.aten.add.Tensor %[[SKIP_RES_TMP]], %[[BIAS]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      return %[[RES]], %[[MEAN]], %[[RSTD]], %[[SKIP_RES]] : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_static_no_optional_results
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SKIP:.+]]: !torch.vtensor<[2,4,8],f32>, %[[SCALE:.+]]: !torch.vtensor<[8],f32>)
func.func @test_skip_simplified_layer_norm_static_no_optional_results(
  %arg0: !torch.vtensor<[2,4,8],f32>,
  %arg1: !torch.vtensor<[2,4,8],f32>,
  %arg2: !torch.vtensor<[8],f32>)
  -> (!torch.vtensor<[2,4,8],f32>, !torch.none, !torch.none, !torch.none)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:4 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2)
           {torch.onnx.epsilon = 1.250000e-01 : f32}
           : (!torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[8],f32>)
           -> (!torch.vtensor<[2,4,8],f32>, !torch.none, !torch.none, !torch.none)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 1.250000e-01
  // CHECK:      %[[INT_8:.+]] = torch.constant.int 8
  // CHECK:      %[[SHAPE:.+]] = torch.prim.ListConstruct %[[INT_8]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[X]], %[[SHAPE]], %[[SCALE]], %[[NONE]], %[[EPS]] : !torch.vtensor<[2,4,8],f32>, !torch.list<int>, !torch.vtensor<[8],f32>, !torch.none, !torch.float -> !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32>
  // CHECK:      %[[NONE_MEAN:.+]] = torch.constant.none
  // CHECK:      %[[NONE_RSTD:.+]] = torch.constant.none
  // CHECK:      %[[NONE_SKIP:.+]] = torch.constant.none
  // CHECK:      return %[[RES]], %[[NONE_MEAN]], %[[NONE_RSTD]], %[[NONE_SKIP]] : !torch.vtensor<[2,4,8],f32>, !torch.none, !torch.none, !torch.none
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[2,4,8],f32>, !torch.none, !torch.none, !torch.none
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_dynbatch_none_stats
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[?,8,2048],f32>, %[[SKIP:.+]]: !torch.vtensor<[?,8,2048],f32>, %[[SCALE:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_skip_simplified_layer_norm_dynbatch_none_stats(
  %arg0: !torch.vtensor<[?,8,2048],f32>,
  %arg1: !torch.vtensor<[?,8,2048],f32>,
  %arg2: !torch.vtensor<[2048],f32>)
  -> (!torch.vtensor<[?,8,2048],f32>, !torch.none, !torch.none, !torch.vtensor<[?,8,2048],f32>)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:4 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2)
           {torch.onnx.epsilon = 2.500000e-01 : f32}
           : (!torch.vtensor<[?,8,2048],f32>, !torch.vtensor<[?,8,2048],f32>, !torch.vtensor<[2048],f32>)
           -> (!torch.vtensor<[?,8,2048],f32>, !torch.none, !torch.none, !torch.vtensor<[?,8,2048],f32>)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 2.500000e-01
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[X]], %[[SHAPE]], %[[SCALE]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,8,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,8,2048],f32>, !torch.vtensor<[?,8,1],f32>, !torch.vtensor<[?,8,1],f32>
  // CHECK:      %[[NONE_MEAN:.+]] = torch.constant.none
  // CHECK:      %[[NONE_RSTD:.+]] = torch.constant.none
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_RES:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,8,2048],f32>
  // CHECK:      return %[[RES]], %[[NONE_MEAN]], %[[NONE_RSTD]], %[[SKIP_RES]] : !torch.vtensor<[?,8,2048],f32>, !torch.none, !torch.none, !torch.vtensor<[?,8,2048],f32>
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[?,8,2048],f32>, !torch.none, !torch.none, !torch.vtensor<[?,8,2048],f32>
}

// -----

// CHECK-LABEL: @test_skip_simplified_layer_norm_dynamic_no_mean
// CHECK-SAME:  (%[[X:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SKIP:.+]]: !torch.vtensor<[?,?,2048],f32>, %[[SCALE:.+]]: !torch.vtensor<[2048],f32>)
func.func @test_skip_simplified_layer_norm_dynamic_no_mean(
  %arg0: !torch.vtensor<[?,?,2048],f32>,
  %arg1: !torch.vtensor<[?,?,2048],f32>,
  %arg2: !torch.vtensor<[2048],f32>)
  -> (!torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>)
  attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  %0:4 = torch.operator "onnx.SkipSimplifiedLayerNormalization"(%arg0, %arg1, %arg2)
           {torch.onnx.epsilon = 6.250000e-02 : f32}
           : (!torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[2048],f32>)
           -> (!torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>)
  // CHECK:      %[[NONE:.+]] = torch.constant.none
  // CHECK:      %[[EPS:.+]] = torch.constant.float 6.250000e-02
  // CHECK:      %[[INT_2048:.+]] = torch.constant.int 2048
  // CHECK:      %[[SHAPE:.+]] = torch.prim.ListConstruct %[[INT_2048]] : (!torch.int) -> !torch.list<int>
  // CHECK:      %[[RES:.+]], %[[MEAN:.+]], %[[RSTD:.+]] = torch.aten.native_layer_norm %[[X]], %[[SHAPE]], %[[SCALE]], %[[NONE]], %[[EPS]] : !torch.vtensor<[?,?,2048],f32>, !torch.list<int>, !torch.vtensor<[2048],f32>, !torch.none, !torch.float -> !torch.vtensor<[?,?,2048],f32>, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,1],f32>
  // CHECK:      %[[NONE_MEAN:.+]] = torch.constant.none
  // CHECK:      %[[ALPHA:.+]] = torch.constant.float 1.000000e+00
  // CHECK:      %[[SKIP_RES:.+]] = torch.aten.add.Tensor %[[X]], %[[SKIP]], %[[ALPHA]] : {{.*}} -> !torch.vtensor<[?,?,2048],f32>
  // CHECK:      return %[[RES]], %[[NONE_MEAN]], %[[RSTD]], %[[SKIP_RES]] : !torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>
  return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[?,?,2048],f32>, !torch.none, !torch.vtensor<[?,?,1],f32>, !torch.vtensor<[?,?,2048],f32>
}
