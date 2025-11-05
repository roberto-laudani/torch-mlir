// ============================================================================
//
// This test is dedicated to com.microsoft.GroupQueryAttention with dynamically
// sized inputs - for now, the pre-existing support for the operator is still
// tested within a common 'simple_ops*' LIT file.
//
// ============================================================================

// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_group_query_attention
// CHECK-SAME:  (%[[QUERY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[KEY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[VALUE_ARG:.+]]: !torch.vtensor<[?,?,16],f32>,
// CHECK-SAME:   %[[PAST_K_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>)
func.func @test_group_query_attention(
  %arg0: !torch.vtensor<[?,?,16],f32>,
  %arg1: !torch.vtensor<[?,?,16],f32>,
  %arg2: !torch.vtensor<[?,?,16],f32>,
  %arg3: !torch.vtensor<[?,2,?,8],f32>,
  %arg4: !torch.vtensor<[?,2,?,8],f32>)
  -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  // CHECK-DAG:  %[[INT_M1_A:.+]] = torch.constant.int -1
  // CHECK-DAG:  %[[INT_M1_B:.+]] = torch.constant.int -1
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:  %[[INT_8:.+]]  = torch.constant.int 8
  // CHECK-DAG:  %[[INT_2_A:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[INT_2_B:.+]] = torch.constant.int 2
  // CHECK:      %[[Q_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_A]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[QUERY_RESHAPE:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[KV_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_B]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG:  %[[KEY_RESHAPE:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:  %[[VALUE_RESHAPE:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG:  %[[TRUE:.+]]  = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]]  = torch.constant.float 0.000000e+00
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[QUERY_RESHAPE]], %[[KEY_RESHAPE]], %[[VALUE_RESHAPE]], %[[NONE]], %[[ZERO]], %[[FALSE]], %[[NONE]], %[[TRUE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_M1_B]], %[[INT_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN]], %[[OUT_SHAPE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  // CHECK:      return %[[OUT]], %[[PAST_K_ARG]], %[[PAST_V_ARG]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
  %2:3 = torch.operator "onnx.GroupQueryAttention"(
           %arg0, %arg1, %arg2, %arg3, %arg4, %0, %1)
           {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64}
           : (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>,
              !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>,
              !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>)
           -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  return %2#0, %2#1, %2#2 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}

// -----

// CHECK-LABEL: @test_group_query_attention_scaled
// CHECK-SAME:  (%[[QUERY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[KEY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[VALUE_ARG:.+]]: !torch.vtensor<[?,?,16],f32>,
// CHECK-SAME:   %[[PAST_K_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>)
func.func @test_group_query_attention_scaled(
  %arg0: !torch.vtensor<[?,?,16],f32>,
  %arg1: !torch.vtensor<[?,?,16],f32>,
  %arg2: !torch.vtensor<[?,?,16],f32>,
  %arg3: !torch.vtensor<[?,2,?,8],f32>,
  %arg4: !torch.vtensor<[?,2,?,8],f32>)
  -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  // CHECK-DAG:  %[[INT_M1_A:.+]] = torch.constant.int -1
  // CHECK-DAG:  %[[INT_M1_B:.+]] = torch.constant.int -1
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:  %[[INT_8:.+]]  = torch.constant.int 8
  // CHECK-DAG:  %[[INT_2_A:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[INT_2_B:.+]] = torch.constant.int 2
  // CHECK:      %[[Q_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_A]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[QUERY_RESHAPE:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[KV_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_B]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG:  %[[KEY_RESHAPE:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:  %[[VALUE_RESHAPE:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG:  %[[TRUE:.+]]  = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]]  = torch.constant.float 0.000000e+00
  // CHECK-DAG:  %[[SCALE:.+]] = torch.constant.float 0.088388346135616302
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[QUERY_RESHAPE]], %[[KEY_RESHAPE]], %[[VALUE_RESHAPE]], %[[NONE]], %[[ZERO]], %[[FALSE]], %[[SCALE]], %[[TRUE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.float, !torch.bool -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_M1_B]], %[[INT_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN]], %[[OUT_SHAPE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  // CHECK:      return %[[OUT]], %[[PAST_K_ARG]], %[[PAST_V_ARG]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
  %2:3 = torch.operator "onnx.GroupQueryAttention"(
           %arg0, %arg1, %arg2, %arg3, %arg4, %0, %1)
           {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64, torch.onnx.scale = 0.0883883461 : f32}
           : (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>,
              !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>,
              !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>)
           -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  return %2#0, %2#1, %2#2 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}

// -----

// CHECK-LABEL: @test_group_query_attention_rotary
// CHECK-SAME:  (%[[QUERY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[KEY_ARG:.+]]: !torch.vtensor<[?,?,16],f32>, %[[VALUE_ARG:.+]]: !torch.vtensor<[?,?,16],f32>,
// CHECK-SAME:   %[[PAST_K_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>,
// CHECK-SAME:   %[[COS_ARG:.+]]: !torch.vtensor<[?,?],si64>, %[[SIN_ARG:.+]]: !torch.vtensor<[?,?],si64>)
// A dynamic rotary-embedding variant where sin/cos caches are inputs.
func.func @test_group_query_attention_rotary(
  %query: !torch.vtensor<[?,?,16],f32>,
  %key: !torch.vtensor<[?,?,16],f32>,
  %value: !torch.vtensor<[?,?,16],f32>,
  %past_k: !torch.vtensor<[?,2,?,8],f32>,
  %past_v: !torch.vtensor<[?,2,?,8],f32>,
  %cos_cache: !torch.vtensor<[?,?],si64>,
  %sin_cache: !torch.vtensor<[?,?],si64>)
  -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %seqlens_k = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  %total_seq = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  // CHECK-DAG:   %[[INT_M1_A:.+]] = torch.constant.int -1
  // CHECK-DAG:   %[[INT_M1_B:.+]] = torch.constant.int -1
  // CHECK-DAG:   %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:   %[[INT_8:.+]]  = torch.constant.int 8
  // CHECK-DAG:   %[[INT_2_A:.+]] = torch.constant.int 2
  // CHECK-DAG:   %[[INT_2_B:.+]] = torch.constant.int 2
  // CHECK:       %[[Q_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_A]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:       %[[QUERY_RESHAPE:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:       %[[KV_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_2_B]], %[[INT_M1_B]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG:   %[[KEY_RESHAPE:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:   %[[VALUE_RESHAPE:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_SHAPE]] : !torch.vtensor<[?,?,16],f32>, !torch.list<int> -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:   %[[QUERY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[QUERY_RESHAPE]], %{{.*}}, %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>, {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:   %[[KEY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[KEY_RESHAPE]], %{{.*}}, %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>, {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK-DAG:   %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:   %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG:   %[[TRUE:.+]]  = torch.constant.bool true
  // CHECK:       %[[ZERO:.+]]  = torch.constant.float 0.000000e+00
  // CHECK:       %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[QUERY_ROTARY]], %[[KEY_ROTARY]], %[[VALUE_RESHAPE]], %[[NONE]], %[[ZERO]], %[[FALSE]], %[[NONE]], %[[TRUE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:       %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_M1_A]], %[[INT_M1_B]], %[[INT_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:       %[[OUT:.+]] = torch.aten.reshape %[[ATTN]], %[[OUT_SHAPE]] : !torch.vtensor<[?,2,?,8],f32>, !torch.list<int> -> !torch.vtensor<[?,?,16],f32>
  // CHECK:       return %[[OUT]], %[[PAST_K_ARG]], %[[PAST_V_ARG]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
  %res:3 = torch.operator "onnx.GroupQueryAttention"(%query, %key, %value, %past_k, %past_v, %seqlens_k, %total_seq, %cos_cache, %sin_cache)
           {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64, torch.onnx.do_rotary = 1 : si64}
           : (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>,
              !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>,
              !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>,
              !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>)
           -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  return %res#0, %res#1, %res#2 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}
