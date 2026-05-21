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
  // KV cache in-place update via slice_scatter + slice the active
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[STEP:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[CACHE_DIM:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[START:.+]] = torch.constant.int 0
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[K_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
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
  // KV cache in-place update + active slice along cache seq dim (= 2).
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[STEP:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[CACHE_DIM:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[START:.+]] = torch.constant.int 0
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[K_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
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
  // CHECK-DAG:  %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG:  %[[TRUE:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ROTARY_INT_1:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[INT_4:.+]] = torch.constant.int 4
  // CHECK-DAG:  %[[ROTARY_INT_0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[FLOAT_1:.+]] = torch.constant.float 1.000000e+00
  //
  // Position IDs computation:
  //   Step 1: Convert seqlens_k from si32 to si64
  // CHECK:      %[[SEQLENS_K_I64:.+]] = torch.aten.to.dtype %{{.*}}, %[[INT_4]], %[[FALSE]], %[[FALSE]], %[[NONE]] : {{.*}} -> !torch.vtensor<[1],si64>
  //   Step 2: total_seqlens = seqlens_k + 1
  // CHECK:      %[[TOTAL_SEQLENS:.+]] = torch.aten.add.Scalar %[[SEQLENS_K_I64]], %[[ROTARY_INT_1]], %[[ROTARY_INT_1]] : {{.*}} -> !torch.vtensor<[1],si64>
  //   Step 3: past_seqlens = total_seqlens - sequence_length
  // CHECK:      %[[PAST_SEQLENS:.+]] = torch.aten.sub.Scalar %[[TOTAL_SEQLENS]], %[[SEQ_LEN]], %[[ROTARY_INT_1]] : {{.*}} -> !torch.vtensor<[1],si64>
  //   Step 4: arange(sequence_length)
  // CHECK:      %[[ARANGE:.+]] = torch.aten.arange %[[SEQ_LEN]], %[[INT_4]], %[[NONE]], %[[NONE]], %[[NONE]] : {{.*}} -> !torch.vtensor<[?],si64>
  //   Step 5: repeat to [batch_size, seq_len]
  // CHECK:      %[[REPEAT_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[ROTARY_INT_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[POS_IDS_INIT:.+]] = torch.aten.repeat %[[ARANGE]], %[[REPEAT_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?],si64>
  //   Step 6: past_seqlens.view(-1, 1)
  // CHECK-DAG:  %[[INT_NEG1:.+]] = torch.constant.int -1
  // CHECK:      %[[VIEW_SHAPE:.+]] = torch.prim.ListConstruct %[[INT_NEG1]], %[[ROTARY_INT_1]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[PAST_SEQLENS_VIEW:.+]] = torch.aten.view %[[PAST_SEQLENS]], %[[VIEW_SHAPE]] : {{.*}} -> !torch.vtensor<[?,1],si64>
  //   Step 7: pos_ids = pos_ids + past_seqlens
  // CHECK:      %[[POS_IDS:.+]] = torch.aten.add.Tensor %[[POS_IDS_INIT]], %[[PAST_SEQLENS_VIEW]], %[[ROTARY_INT_1]] : {{.*}} -> !torch.vtensor<[?,?],si64>
  //
  // Rotary embeddings applied to Q and K with computed position IDs
  // CHECK:      %[[QUERY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[Q_PERMUTED]], %[[POS_IDS]], %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[KEY_ROTARY:.+]] = torch.onnx.rotary_embedding %[[K_PERMUTED]], %[[POS_IDS]], %[[COS_ARG]], %[[SIN_ARG]], {{.*}} : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  //
  // KV cache in-place update with rotary-embedded key + active slice
  // along cache seq dim (= 2).
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[STEP:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[CACHE_DIM:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[START:.+]] = torch.constant.int 0
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[KEY_ROTARY]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Attention with is_causal=true
  // CHECK-DAG:  %[[TRUE_GQA:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[QUERY_ROTARY]], %[[ACTIVE_K]], %[[ACTIVE_V]], %[[NONE]], %[[ZERO]], %[[TRUE]], %[[NONE]], %[[TRUE_GQA]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
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

// -----

// CHECK-LABEL: @test_group_query_attention_packed_qkv
// CHECK-SAME:  (%[[PACKED_ARG:.+]]: !torch.vtensor<[?,?,64],f32>,
// CHECK-SAME:   %[[PAST_K_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>, %[[PAST_V_ARG:.+]]: !torch.vtensor<[?,2,?,8],f32>)
// Packed QKV: key and value are none, query contains [batch, seq, (num_heads + 2*kv_num_heads) * head_size]
// with num_heads=4, kv_num_heads=2, head_size=8 => packed_hidden=64, q_hidden=32, kv_hidden=16
func.func @test_group_query_attention_packed_qkv(
  %packed: !torch.vtensor<[?,?,64],f32>,
  %past_k: !torch.vtensor<[?,2,?,8],f32>,
  %past_v: !torch.vtensor<[?,2,?,8],f32>)
  -> (!torch.vtensor<[?,?,32],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  %none_key = torch.constant.none
  %none_value = torch.constant.none
  %seqlens_k = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  %total_seq = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi32>} : () -> !torch.vtensor<[1],si32>
  // Split packed QKV into Q [?,?,32], K [?,?,16], V [?,?,16]
  // CHECK-DAG:  %[[INT_32:.+]] = torch.constant.int 32
  // CHECK-DAG:  %[[INT_16:.+]] = torch.constant.int 16
  // CHECK:      %[[SPLIT_SIZES:.+]] = torch.prim.ListConstruct %[[INT_32]], %[[INT_16]], %[[INT_16]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG:  %[[INT_2_SPLIT:.+]] = torch.constant.int 2
  // CHECK:      %[[SPLIT_RESULT:.+]] = torch.aten.split_with_sizes %[[PACKED_ARG]], %[[SPLIT_SIZES]], %[[INT_2_SPLIT]] : {{.*}} -> !torch.list<vtensor<[?,?,?],f32>>
  // CHECK:      %[[QKV:.+]]:3 = torch.prim.ListUnpack %[[SPLIT_RESULT]] : !torch.list<vtensor<[?,?,?],f32>> -> !torch.vtensor<[?,?,32],f32>, !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[?,?,16],f32>
  // Dynamic batch and sequence size extraction
  // CHECK-DAG:  %[[INT_0:.+]] = torch.constant.int 0
  // CHECK-DAG:  %[[INT_1:.+]] = torch.constant.int 1
  // CHECK:      %[[BATCH_SIZE:.+]] = torch.aten.size.int %[[QKV]]#0, %[[INT_0]] : {{.*}} -> !torch.int
  // CHECK:      %[[SEQ_LEN:.+]] = torch.aten.size.int %[[QKV]]#0, %[[INT_1]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[INT_32_1:.+]] = torch.constant.int 32
  // CHECK-DAG:  %[[INT_8:.+]] = torch.constant.int 8
  // CHECK-DAG:  %[[INT_4:.+]] = torch.constant.int 4
  // CHECK-DAG:  %[[INT_2:.+]] = torch.constant.int 2
  // Reshape Q from [batch, seq, 32] to [batch, seq, 4, 8] then permute to [batch, 4, seq, 8]
  // CHECK:      %[[Q_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_4]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[Q_RESHAPED:.+]] = torch.aten.reshape %[[QKV]]#0, %[[Q_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,4,8],f32>
  // CHECK:      %[[PERMUTE_DIMS:.+]] = torch.prim.ListConstruct
  // CHECK:      %[[Q_PERMUTED:.+]] = torch.aten.permute %[[Q_RESHAPED]], %[[PERMUTE_DIMS]] : {{.*}} -> !torch.vtensor<[?,4,?,8],f32>
  // Reshape K from [batch, seq, 16] to [batch, seq, 2, 8] then permute
  // CHECK:      %[[KV_RESHAPE_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_2]], %[[INT_8]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[K_RESHAPED:.+]] = torch.aten.reshape %[[QKV]]#1, %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[K_PERMUTED:.+]] = torch.aten.permute %[[K_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Reshape V
  // CHECK:      %[[V_RESHAPED:.+]] = torch.aten.reshape %[[QKV]]#2, %[[KV_RESHAPE_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,2,8],f32>
  // CHECK:      %[[V_PERMUTED:.+]] = torch.aten.permute %[[V_RESHAPED]], {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Constants for attention
  // CHECK-DAG:  %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG:  %[[TRUE:.+]] = torch.constant.bool true
  // KV cache in-place update + active slice along cache seq dim (= 2).
  // CHECK:      %[[TOTAL_SEQ:.+]] = torch.aten.item {{.*}} : {{.*}} -> !torch.int
  // CHECK:      %[[PAST_SEQ:.+]] = torch.aten.sub.int %[[TOTAL_SEQ]], %[[SEQ_LEN]] : {{.*}} -> !torch.int
  // CHECK-DAG:  %[[STEP:.+]] = torch.constant.int 1
  // CHECK-DAG:  %[[CACHE_DIM:.+]] = torch.constant.int 2
  // CHECK-DAG:  %[[START:.+]] = torch.constant.int 0
  // CHECK:      %[[PRESENT_K:.+]] = torch.aten.slice_scatter %[[PAST_K_ARG]], %[[K_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[PRESENT_V:.+]] = torch.aten.slice_scatter %[[PAST_V_ARG]], %[[V_PERMUTED]], %[[CACHE_DIM]], %[[PAST_SEQ]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_K:.+]] = torch.aten.slice.Tensor %[[PRESENT_K]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // CHECK:      %[[ACTIVE_V:.+]] = torch.aten.slice.Tensor %[[PRESENT_V]], %[[CACHE_DIM]], %[[START]], %[[TOTAL_SEQ]], %[[STEP]] : {{.*}} -> !torch.vtensor<[?,2,?,8],f32>
  // Scaled dot product attention
  // CHECK-DAG:  %[[TRUE_7:.+]] = torch.constant.bool true
  // CHECK-DAG:  %[[ZERO:.+]] = torch.constant.float 0.000000e+00
  // CHECK:      %[[ATTN:.+]] = torch.aten.scaled_dot_product_attention %[[Q_PERMUTED]], %[[ACTIVE_K]], %[[ACTIVE_V]], %[[NONE]], %[[ZERO]], %[[TRUE]], %[[NONE]], %[[TRUE_7]] : {{.*}} -> !torch.vtensor<[?,4,?,8],f32>
  // Output reshape: permute back then reshape to [batch, seq, 32]
  // CHECK:      %[[ATTN_PERMUTED:.+]] = torch.aten.permute %[[ATTN]], {{.*}} -> !torch.vtensor<[?,?,4,8],f32>
  // CHECK:      %[[OUT_SHAPE:.+]] = torch.prim.ListConstruct %[[BATCH_SIZE]], %[[SEQ_LEN]], %[[INT_32_1]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      %[[OUT:.+]] = torch.aten.reshape %[[ATTN_PERMUTED]], %[[OUT_SHAPE]] : {{.*}} -> !torch.vtensor<[?,?,32],f32>
  // CHECK:      return %[[OUT]], %[[PRESENT_K]], %[[PRESENT_V]] : !torch.vtensor<[?,?,32],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
  %res:3 = torch.operator "onnx.GroupQueryAttention"(
             %packed, %none_key, %none_value, %past_k, %past_v, %seqlens_k, %total_seq)
             {torch.onnx.kv_num_heads = 2 : si64, torch.onnx.num_heads = 4 : si64}
             : (!torch.vtensor<[?,?,64],f32>, !torch.none, !torch.none,
                !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>,
                !torch.vtensor<[1],si32>, !torch.vtensor<[1],si32>)
             -> (!torch.vtensor<[?,?,32],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>)
  return %res#0, %res#1, %res#2 : !torch.vtensor<[?,?,32],f32>, !torch.vtensor<[?,2,?,8],f32>, !torch.vtensor<[?,2,?,8],f32>
}
