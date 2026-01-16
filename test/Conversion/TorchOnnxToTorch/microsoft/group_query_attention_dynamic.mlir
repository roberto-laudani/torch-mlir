// ============================================================================
//
// This test is dedicated to com.microsoft.GroupQueryAttention with dynamically
// sized inputs. Tests the reshape+permute pattern and in-place KV cache updates.
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
  // Dynamic batch and sequence size extraction
  // CHECK-DAG:  %[[INT_0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[INT_1:.+]] = torch.constant.int 1
  // CHECK:      %[[BATCH_SIZE:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_0]] : {{.*}} -> !torch.int
  // CHECK:      %[[SEQ_LEN:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_1]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:  %[[INT_8:.+]] = torch.constant.int 8
  // CHECK-DAG:  %[[INT_2:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[INT_2_0:.+]] = torch.constant.int 2
  // Reshape Q from [batch, seq, hidden] to [batch, seq, heads, head_dim]
  // CHECK:      %[[Q_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[Q_RESHAPED:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // Permute Q from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
  // CHECK:      %[[PERMUTE_DIMS:.+]] = torch.prim.ListConstruct
  // CHECK:      %[[Q_PERMUTED:.+]] = torch.aten.permute %[[Q_RESHAPED]], %[[PERMUTE_DIMS]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape K from [batch, seq, kv_hidden] to [batch, seq, kv_heads, head_dim]
  // CHECK:      %[[KV_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2_0]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[K_RESHAPED:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // Permute K
  // CHECK:      %[[K_PERMUTED:.+]] = torch.aten.permute %[[K_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape V
  // CHECK:      %[[V_RESHAPED:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // Permute V
  // CHECK:      %[[V_PERMUTED:.+]] = torch.aten.permute %[[V_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Constants for attention
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[TRUE:.+]] = torch.constant.bool true
  // KV cache operations: compute past_seq_len, then slice_scatter to update cache
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[K_PERMUTED]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Slice to get active portion of KV cache
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], {{.*}}, {{.*}}, %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], {{.*}}, {{.*}}, %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Scaled dot product attention with is_causal=true
  // CHECK-DAG:  %[[TRUE_7:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[Q_PERMUTED]], %[[ACTIVE_K]], %[[ACTIVE_V]], %[[NONE]], %[[ZERO]], %[[TRUE]], %[[NONE]], %[[TRUE_7]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Output: permute back to [batch, seq, heads, head_dim] then reshape to [batch, seq, hidden]
  // CHECK:      %[[ATTN_PERMUTED:.+]] = torch.aten.permute %[[ATTN]], {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN_PERMUTED]], %[[OUT_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,16],f32>
  // CHECK:      return %[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
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
  // Dynamic dimensions
  // CHECK-DAG:  %[[INT_0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[INT_1:.+]] = torch.constant.int 1
  // CHECK:      %[[BATCH_SIZE:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_0]] : {{.*}} -> !torch.int
  // CHECK:      %[[SEQ_LEN:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_1]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:  %[[INT_8:.+]] = torch.constant.int 8
  // CHECK-DAG:  %[[INT_2:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[INT_2_0:.+]] = torch.constant.int 2
  // Reshape+permute pattern for Q
  // CHECK:      %[[Q_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2]], %[[INT_8]]
  // CHECK:      %[[Q_RESHAPED:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[PERMUTE_DIMS:.+]] = torch.prim.ListConstruct
  // CHECK:      %[[Q_PERMUTED:.+]] = torch.aten.permute %[[Q_RESHAPED]], %[[PERMUTE_DIMS]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape+permute for K
  // CHECK:      %[[KV_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2_0]], %[[INT_8]]
  // CHECK:      %[[K_RESHAPED:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[K_PERMUTED:.+]] = torch.aten.permute %[[K_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape+permute for V
  // CHECK:      %[[V_RESHAPED:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[V_PERMUTED:.+]] = torch.aten.permute %[[V_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Constants
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[TRUE:.+]] = torch.constant.bool true
  // KV cache operations
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[K_PERMUTED]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Attention with custom scale and is_causal=true
  // CHECK-DAG:  %[[TRUE_7:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK-DAG:  %[[SCALE:.+]] = torch.constant.float 0.088388346135616302
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[Q_PERMUTED]], %[[ACTIVE_K]], %[[ACTIVE_V]], %[[NONE]], %[[ZERO]], %[[TRUE]], %[[SCALE]], %[[TRUE_7]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Output reshape
  // CHECK:      %[[ATTN_PERMUTED:.+]] = torch.aten.permute %[[ATTN]], {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_16]]
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN_PERMUTED]], %[[OUT_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,16],f32>
  // CHECK:      return %[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
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
  // CHECK-DAG:  torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  // CHECK-DAG:  torch.vtensor.literal(dense<1> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  // Dynamic dimensions
  // CHECK-DAG:  %[[INT_0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[INT_1:.+]] = torch.constant.int 1
  // CHECK:      %[[BATCH_SIZE:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_0]] : {{.*}} -> !torch.int
  // CHECK:      %[[SEQ_LEN:.+]] = torch.aten.size.int %[[QUERY_ARG]], %[[INT_1]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK-DAG:  %[[INT_8:.+]] = torch.constant.int 8
  // CHECK-DAG:  %[[INT_2:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[INT_2_0:.+]] = torch.constant.int 2
  // Reshape+permute pattern for Q
  // CHECK:      %[[Q_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2]], %[[INT_8]]
  // CHECK:      %[[Q_RESHAPED:.+]] = torch.aten.reshape %[[QUERY_ARG]], %[[Q_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[PERMUTE_DIMS:.+]] = torch.prim.ListConstruct
  // CHECK:      %[[Q_PERMUTED:.+]] = torch.aten.permute %[[Q_RESHAPED]], %[[PERMUTE_DIMS]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape+permute for K
  // CHECK:      %[[KV_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2_0]], %[[INT_8]]
  // CHECK:      %[[K_RESHAPED:.+]] = torch.aten.reshape %[[KEY_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[K_PERMUTED:.+]] = torch.aten.permute %[[K_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape+permute for V
  // CHECK:      %[[V_RESHAPED:.+]] = torch.aten.reshape %[[VALUE_ARG]], %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[V_PERMUTED:.+]] = torch.aten.permute %[[V_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Constants
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[TRUE:.+]] = torch.constant.bool true
  // Position IDs computation and rotary embeddings
  // CHECK:      %[[QUERY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[Q_PERMUTED]], {{.*}}, %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[KEY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[K_PERMUTED]], {{.*}}, %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // KV cache operations with rotary-embedded key
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[KEY_ROTARY]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], {{.*}}, %[[PAST_SEQ]], %[[TOTAL_SEQ]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Attention with is_causal=true
  // CHECK-DAG:  %[[TRUE_11:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[QUERY_ROTARY]], %[[ACTIVE_K]], %[[ACTIVE_V]], %[[NONE]], %[[ZERO]], %[[TRUE]], %[[NONE]], %[[TRUE_11]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Output reshape
  // CHECK:      %[[ATTN_PERMUTED:.+]] = torch.aten.permute %[[ATTN]], {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_16]]
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN_PERMUTED]], %[[OUT_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,16],f32>
  // CHECK:      return %[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
  %res:3 = torch.operator "onnx.GroupQueryAttention"(%query, %key, %value, %past_k, %past_v, %seqlens_k, %total_seq, %cos_cache, %sin_cache)
           {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 2 : si64, torch.onnx.do_rotary = 1 : si64}
           : (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>,
              !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>,
              !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>,
              !torch.vtensor<[?,?],si64>, !torch.vtensor<[?,?],si64>)
           -> (!torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  return %res#0, %res#1, %res#2 : !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}
