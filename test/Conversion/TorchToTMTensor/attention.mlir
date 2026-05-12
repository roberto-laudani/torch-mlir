// RUN: torch-mlir-opt <%s -convert-torch-to-tmtensor -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test_group_query_attention
// CHECK-SAME:  (%[[Q:.+]]: !torch.vtensor<[?,?,16],f32>, %[[K:.+]]: !torch.vtensor<[?,?,16],f32>, %[[V:.+]]: !torch.vtensor<[?,?,16],f32>, %[[PAST_K:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V:.+]]: !torch.vtensor<[?,2,?,8],f32>)
func.func @test_group_query_attention(
  %arg0: !torch.vtensor<[?,?,16],f32>,
  %arg1: !torch.vtensor<[?,?,16],f32>,
  %arg2: !torch.vtensor<[?,?,16],f32>,
  %arg3: !torch.vtensor<[?,2,?,8],f32>,
  %arg4: !torch.vtensor<[?,2,?,8],f32>)
  -> (!torch.vtensor<[?,?,16],f32>,
      !torch.vtensor<[?,2,?,8],f32>,
      !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[FILL_OUT:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[INIT_TENSOR]] : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  // CHECK-DAG: %[[ATTN:.+]] = tm_tensor.attention ins(%[[Q_IN:.+]], %[[K_IN:.+]], %[[V_IN:.+]] : tensor<?x?x8xf32>, tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%[[FILL_OUT]] : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  // CHECK-DAG: %[[Q_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK-DAG: %[[K_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK-DAG: %[[V_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK: %[[BATCH_DIV:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
  // CHECK: tensor.expand_shape %[[ATTN]] {{\[\[}}0, 1], [2], [3]] output_shape {{\[}}%[[BATCH_DIV]], 2, %{{.*}}, 8] : tensor<?x?x8xf32> into tensor<?x2x?x8xf32>
  %0 = torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  %1 = torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  %int-1 = torch.constant.int -1
  %int-1_0 = torch.constant.int -1
  %int16 = torch.constant.int 16
  %int8 = torch.constant.int 8
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %2 = torch.prim.ListConstruct %int-1, %int2, %int-1_0, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.reshape %arg0, %2 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %4 = torch.prim.ListConstruct %int-1, %int2_1, %int-1_0, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.reshape %arg1, %4 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %6 = torch.aten.reshape %arg2, %4 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %7 = torch.aten.scaled_dot_product_attention %3, %5, %6, %none, %float0.000000e00, %false, %none, %true : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,2,?,8],f32>
  %8 = torch.prim.ListConstruct %int-1, %int-1_0, %int16 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %9 = torch.aten.reshape %7, %8 : !torch.vtensor<[?,2,?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  return %9, %arg3, %arg4 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}

// -----

// COM: Verify that the scale parameter gets forwarded as an attribute to tm_tensor.attention

// CHECK-LABEL: @test_group_query_attention_scaled
// CHECK-SAME:  (%[[Q:.+]]: !torch.vtensor<[?,?,16],f32>, %[[K:.+]]: !torch.vtensor<[?,?,16],f32>, %[[V:.+]]: !torch.vtensor<[?,?,16],f32>, %[[PAST_K:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V:.+]]: !torch.vtensor<[?,2,?,8],f32>)
func.func @test_group_query_attention_scaled(
  %arg0: !torch.vtensor<[?,?,16],f32>,
  %arg1: !torch.vtensor<[?,?,16],f32>,
  %arg2: !torch.vtensor<[?,?,16],f32>,
  %arg3: !torch.vtensor<[?,2,?,8],f32>,
  %arg4: !torch.vtensor<[?,2,?,8],f32>)
  -> (!torch.vtensor<[?,?,16],f32>,
      !torch.vtensor<[?,2,?,8],f32>,
      !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[FILL_OUT:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[INIT_TENSOR]] : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  // scale = 1/sqrt(8) ≈ 0.353553. The lowering requires this to match
  // 1/sqrt(headDim); see torch-mlir #4419.
  // CHECK-DAG: %[[ATTN:.+]] = tm_tensor.attention {scale = 0.35355339059327379 : f64} ins(%[[Q_IN:.+]], %[[K_IN:.+]], %[[V_IN:.+]] : tensor<?x?x8xf32>, tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%[[FILL_OUT]] : tensor<?x?x8xf32>) -> tensor<?x?x8xf32>
  // CHECK-DAG: %[[Q_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK-DAG: %[[K_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK-DAG: %[[V_IN]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2], [3]] : tensor<?x2x?x8xf32> into tensor<?x?x8xf32>
  // CHECK: %[[BATCH_DIV:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
  // CHECK: tensor.expand_shape %[[ATTN]] {{\[\[}}0, 1], [2], [3]] output_shape {{\[}}%[[BATCH_DIV]], 2, %{{.*}}, 8] : tensor<?x?x8xf32> into tensor<?x2x?x8xf32>
  %0 = torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  %1 = torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  %int-1 = torch.constant.int -1
  %int-1_0 = torch.constant.int -1
  %int16 = torch.constant.int 16
  %int8 = torch.constant.int 8
  %int2 = torch.constant.int 2
  %int2_1 = torch.constant.int 2
  %2 = torch.prim.ListConstruct %int-1, %int2, %int-1_0, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.aten.reshape %arg0, %2 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %4 = torch.prim.ListConstruct %int-1, %int2_1, %int-1_0, %int8 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.reshape %arg1, %4 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %6 = torch.aten.reshape %arg2, %4 : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  %none = torch.constant.none
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %scale = torch.constant.float 0.35355339059327379
  %7 = torch.aten.scaled_dot_product_attention %3, %5, %6, %none, %float0.000000e00, %false, %scale, %true : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[?,2,?,8],f32>
  %8 = torch.prim.ListConstruct %int-1, %int-1_0, %int16 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %9 = torch.aten.reshape %7, %8 : !torch.vtensor<[?,2,?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  return %9, %arg3, %arg4 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}

// -----

// COM: Test causal mask with Q_len < K_len (KV cache decode scenario).
// COM: This verifies lower-right aligned causal mask generation where offset = K_len - Q_len.

// CHECK-LABEL: @test_causal_mask_kv_cache
// CHECK-SAME:  (%[[Q:.+]]: !torch.vtensor<[1,2,1,8],f32>, %[[K:.+]]: !torch.vtensor<[1,2,4,8],f32>, %[[V:.+]]: !torch.vtensor<[1,2,4,8],f32>)
func.func @test_causal_mask_kv_cache(
  %arg0: !torch.vtensor<[1,2,1,8],f32>,
  %arg1: !torch.vtensor<[1,2,4,8],f32>,
  %arg2: !torch.vtensor<[1,2,4,8],f32>)
  -> !torch.vtensor<[1,2,1,8],f32>
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK: %[[MASK_EMPTY:.*]] = tensor.empty() : tensor<1x2x1x4xf32>
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[MASK_FILLED:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[MASK_EMPTY]] : tensor<1x2x1x4xf32>) -> tensor<1x2x1x4xf32>
  // COM: Verify lower-right aligned causal mask: offset = K_len - Q_len = 4 - 1 = 3
  // CHECK: %[[Q_LEN:.*]] = tensor.dim %[[MASK_FILLED]], %{{.*}} : tensor<1x2x1x4xf32>
  // CHECK: %[[K_LEN:.*]] = tensor.dim %[[MASK_FILLED]], %{{.*}} : tensor<1x2x1x4xf32>
  // CHECK: %[[OFFSET:.*]] = arith.subi %[[K_LEN]], %[[Q_LEN]] : index
  // CHECK: linalg.generic
  // COM: Inside the generic op, verify (i + offset) >= j comparison
  // CHECK: linalg.index 2
  // CHECK: linalg.index 3
  // CHECK: arith.addi
  // CHECK: arith.cmpi sge
  %none = torch.constant.none
  %true = torch.constant.bool true
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2, %none, %float0.000000e00, %true, %none, %true : !torch.vtensor<[1,2,1,8],f32>, !torch.vtensor<[1,2,4,8],f32>, !torch.vtensor<[1,2,4,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,2,1,8],f32>
  return %0 : !torch.vtensor<[1,2,1,8],f32>
}
