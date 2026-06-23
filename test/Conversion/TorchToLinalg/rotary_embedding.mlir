// RUN: torch-mlir-opt <%s -convert-torch-to-linalg --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_rotary_embedding(
// CHECK-SAME:    %[[X_VT:.*]]: !torch.vtensor<[1,4,2,16],f32>,
// CHECK-SAME:    %[[POS_VT:.*]]: !torch.vtensor<[1,2],si64>,
// CHECK-SAME:    %[[COS_VT:.*]]: !torch.vtensor<[4,8],f32>,
// CHECK-SAME:    %[[SIN_VT:.*]]: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[1,4,2,16],f32>
func.func @test_rotary_embedding(%arg0: !torch.vtensor<[1,4,2,16],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[4,8],f32>, %arg3: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[1,4,2,16],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[SIN_T:.*]] = torch_c.to_builtin_tensor %[[SIN_VT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
  // CHECK-DAG: %[[COS_T:.*]] = torch_c.to_builtin_tensor %[[COS_VT]] : !torch.vtensor<[4,8],f32> -> tensor<4x8xf32>
  // CHECK-DAG: %[[POS_T:.*]] = torch_c.to_builtin_tensor %[[POS_VT]] : !torch.vtensor<[1,2],si64> -> tensor<1x2xi64>
  // CHECK-DAG: %[[X_T:.*]]   = torch_c.to_builtin_tensor %[[X_VT]] : !torch.vtensor<[1,4,2,16],f32> -> tensor<1x4x2x16xf32>

  // CHECK:     %[[INIT_TENSOR:.*]] = tensor.empty(
  // CHECK:     %[[ZERO_F32:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:     %[[FILLED_TENSOR:.*]] = linalg.fill ins(%[[ZERO_F32]] : f32) outs(%[[INIT_TENSOR]] : tensor<1x4x2x16xf32>) -> tensor<1x4x2x16xf32>

  // CHECK-DAG: %[[EMBED_HALF_BASE_IDX:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[EMBED_SIZE_IDX:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[ONE_F32:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[NEG_ONE_F32:.*]] = arith.constant -1.000000e+00 : f32

  // CHECK:     %[[GENERIC:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[X_T]], %[[POS_T]] : tensor<1x4x2x16xf32>, tensor<1x2xi64>)
  // CHECK-SAME:   outs(%[[FILLED_TENSOR]] : tensor<1x4x2x16xf32>) {
  // CHECK:     ^bb0(%[[X_ELEM:.*]]: f32, %[[POS_ELEM:.*]]: i64, %[[OUT_INIT:.*]]: f32):
  // CHECK:       %[[BATCH_INDEX:.*]] = linalg.index 0 : index
  // CHECK:       %[[HEAD_INDEX:.*]] = linalg.index 1 : index
  // CHECK:       %[[SEQUENCE_INDEX:.*]] = linalg.index 2 : index
  // CHECK:       %[[EMBED_INDEX:.*]]  = linalg.index 3 : index
  // CHECK:       %[[EMBED_MOD3:.*]] = arith.remsi %[[EMBED_INDEX]], %[[EMBED_HALF_BASE_IDX]] : index
  // CHECK:       %[[IS_UPPER_HALF:.*]] = arith.cmpi sge, %[[EMBED_INDEX]], %[[EMBED_HALF_BASE_IDX]] : index
  // CHECK:       %[[EMBED_PLUS3:.*]] = arith.addi %[[EMBED_INDEX]], %[[EMBED_HALF_BASE_IDX]] : index
  // CHECK:       %[[PAIR_EMBED_INDEX:.*]] = arith.remsi %[[EMBED_PLUS3]], %[[EMBED_SIZE_IDX]] : index
  // CHECK:       %[[POS_INDEX:.*]] = arith.index_cast %[[POS_ELEM]] : i64 to index
  // CHECK:       %[[COS_ELEM:.*]] = tensor.extract %[[COS_T]][%[[POS_INDEX]], %[[EMBED_MOD3]]] : tensor<4x8xf32>
  // CHECK:       %[[SIN_ELEM:.*]] = tensor.extract %[[SIN_T]][%[[POS_INDEX]], %[[EMBED_MOD3]]] : tensor<4x8xf32>
  // CHECK:       %[[PAIR_X:.*]] = tensor.extract %[[X_T]][%[[BATCH_INDEX]], %[[HEAD_INDEX]], %[[SEQUENCE_INDEX]], %[[PAIR_EMBED_INDEX]]] : tensor<1x4x2x16xf32>
  // CHECK:       %[[SIGN:.*]] = arith.select %[[IS_UPPER_HALF]], %[[ONE_F32]], %[[NEG_ONE_F32]] : f32
  // CHECK:       %[[PROD1:.*]] = arith.mulf %[[X_ELEM]], %[[COS_ELEM]] : f32
  // CHECK:       %[[PROD2:.*]] = arith.mulf %[[PAIR_X]], %[[SIN_ELEM]] : f32
  // CHECK:       %[[PROD2S:.*]] = arith.mulf %[[PROD2]], %[[SIGN]] : f32
  // CHECK:       %[[SUM:.*]] = arith.addf %[[PROD1]], %[[PROD2S]] : f32
  // CHECK:       linalg.yield %[[SUM]] : f32
  // CHECK:     } -> tensor<1x4x2x16xf32>

  // CHECK:     %[[CAST:.*]] = tensor.cast %[[GENERIC]] : tensor<1x4x2x16xf32> to tensor<1x4x2x16xf32>
  // CHECK:     %[[RET_VT:.*]] = torch_c.from_builtin_tensor %[[CAST]] : tensor<1x4x2x16xf32> -> !torch.vtensor<[1,4,2,16],f32>
  // CHECK:     return %[[RET_VT]] : !torch.vtensor<[1,4,2,16],f32>
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[1,4,2,16],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[4,8],f32>, !torch.vtensor<[4,8],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[1,4,2,16],f32>
  return %0 : !torch.vtensor<[1,4,2,16],f32>
}

// -----

// CHECK-LABEL: func.func @test_rotary_embedding_dynamic(
// CHECK-SAME:    %[[X_VT_D:.*]]: !torch.vtensor<[?,3,?,6],f32>,
// CHECK-SAME:    %[[POS_VT_D:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:    %[[COS_VT_D:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:    %[[SIN_VT_D:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,3,?,6],f32>
func.func @test_rotary_embedding_dynamic(%arg0: !torch.vtensor<[?,3,?,6],f32>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,3,?,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[SIN_T_D:.*]] = torch_c.to_builtin_tensor %[[SIN_VT_D]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK-DAG: %[[COS_T_D:.*]] = torch_c.to_builtin_tensor %[[COS_VT_D]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK-DAG: %[[POS_T_D:.*]] = torch_c.to_builtin_tensor %[[POS_VT_D]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
  // CHECK-DAG: %[[X_T_D:.*]]   = torch_c.to_builtin_tensor %[[X_VT_D]] : !torch.vtensor<[?,3,?,6],f32> -> tensor<?x3x?x6xf32>

  // CHECK:     %[[INIT_TENSOR_D:.*]] = tensor.empty(
  // CHECK:     %[[ZERO_F32_D:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:     %[[FILLED_TENSOR_D:.*]] = linalg.fill ins(%[[ZERO_F32_D]] : f32) outs(%[[INIT_TENSOR_D]] : tensor<?x3x?x6xf32>) -> tensor<?x3x?x6xf32>

  // CHECK-DAG: %[[EMBED_HALF_BASE_IDX_D:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[EMBED_SIZE_IDX_D:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[ONE_F32_D:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[NEG_ONE_F32_D:.*]] = arith.constant -1.000000e+00 : f32

  // CHECK:     %[[GENERIC_D:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[X_T_D]], %[[POS_T_D]] : tensor<?x3x?x6xf32>, tensor<?x?xi64>)
  // CHECK-SAME:   outs(%[[FILLED_TENSOR_D]] : tensor<?x3x?x6xf32>) {
  // CHECK:     ^bb0(%[[X_ELEM_D:.*]]: f32, %[[POS_ELEM_D:.*]]: i64, %[[OUT_INIT_D:.*]]: f32):
  // CHECK:       %[[BATCH_INDEX_D:.*]] = linalg.index 0 : index
  // CHECK:       %[[HEAD_INDEX_D:.*]] = linalg.index 1 : index
  // CHECK:       %[[SEQUENCE_INDEX_D:.*]] = linalg.index 2 : index
  // CHECK:       %[[EMBED_INDEX_D:.*]]  = linalg.index 3 : index
  // CHECK:       %[[EMBED_MOD3_D:.*]] = arith.remsi %[[EMBED_INDEX_D]], %[[EMBED_HALF_BASE_IDX_D]] : index
  // CHECK:       %[[IS_UPPER_HALF_D:.*]] = arith.cmpi sge, %[[EMBED_INDEX_D]], %[[EMBED_HALF_BASE_IDX_D]] : index
  // CHECK:       %[[EMBED_PLUS3_D:.*]] = arith.addi %[[EMBED_INDEX_D]], %[[EMBED_HALF_BASE_IDX_D]] : index
  // CHECK:       %[[PAIR_EMBED_INDEX_D:.*]] = arith.remsi %[[EMBED_PLUS3_D]], %[[EMBED_SIZE_IDX_D]] : index
  // CHECK:       %[[POS_INDEX_D:.*]] = arith.index_cast %[[POS_ELEM_D]] : i64 to index
  // CHECK:       %[[COS_ELEM_D:.*]] = tensor.extract %[[COS_T_D]][%[[POS_INDEX_D]], %[[EMBED_MOD3_D]]] : tensor<4x3xf32>
  // CHECK:       %[[SIN_ELEM_D:.*]] = tensor.extract %[[SIN_T_D]][%[[POS_INDEX_D]], %[[EMBED_MOD3_D]]] : tensor<4x3xf32>
  // CHECK:       %[[PAIR_X_D:.*]] = tensor.extract %[[X_T_D]][%[[BATCH_INDEX_D]], %[[HEAD_INDEX_D]], %[[SEQUENCE_INDEX_D]], %[[PAIR_EMBED_INDEX_D]]] : tensor<?x3x?x6xf32>
  // CHECK:       %[[SIGN_D:.*]] = arith.select %[[IS_UPPER_HALF_D]], %[[ONE_F32_D]], %[[NEG_ONE_F32_D]] : f32
  // CHECK:       %[[PROD1_D:.*]] = arith.mulf %[[X_ELEM_D]], %[[COS_ELEM_D]] : f32
  // CHECK:       %[[PROD2_D:.*]] = arith.mulf %[[PAIR_X_D]], %[[SIN_ELEM_D]] : f32
  // CHECK:       %[[PROD2S_D:.*]] = arith.mulf %[[PROD2_D]], %[[SIGN_D]] : f32
  // CHECK:       %[[SUM_D:.*]] = arith.addf %[[PROD1_D]], %[[PROD2S_D]] : f32
  // CHECK:       linalg.yield %[[SUM_D]] : f32
  // CHECK:     } -> tensor<?x3x?x6xf32>

  // CHECK:     %[[CAST_D:.*]] = tensor.cast %[[GENERIC_D]] : tensor<?x3x?x6xf32> to tensor<?x3x?x6xf32>
  // CHECK:     %[[RET_VT_D:.*]] = torch_c.from_builtin_tensor %[[CAST_D]] : tensor<?x3x?x6xf32> -> !torch.vtensor<[?,3,?,6],f32>
  // CHECK:     return %[[RET_VT_D]] : !torch.vtensor<[?,3,?,6],f32>
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[?,3,?,6],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[?,3,?,6],f32>
  return %0 : !torch.vtensor<[?,3,?,6],f32>
}

// -----

// CHECK-LABEL: func.func @test_rotary_embedding_3d_dynamic(
// CHECK-SAME:    %[[X3D_VT:.*]]: !torch.vtensor<[?,?,24],f32>,
// CHECK-SAME:    %[[POS_VT3:.*]]: !torch.vtensor<[?,?],si64>,
// CHECK-SAME:    %[[COS_VT3:.*]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:    %[[SIN_VT3:.*]]: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,?,24],f32>
func.func @test_rotary_embedding_3d_dynamic(%arg0: !torch.vtensor<[?,?,24],f32>, %arg1: !torch.vtensor<[?,?],si64>, %arg2: !torch.vtensor<[4,3],f32>, %arg3: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[?,?,24],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
  // CHECK-DAG: %[[SIN_T3:.*]] = torch_c.to_builtin_tensor %[[SIN_VT3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK-DAG: %[[COS_T3:.*]] = torch_c.to_builtin_tensor %[[COS_VT3]] : !torch.vtensor<[4,3],f32> -> tensor<4x3xf32>
  // CHECK-DAG: %[[POS_T3:.*]] = torch_c.to_builtin_tensor %[[POS_VT3]] : !torch.vtensor<[?,?],si64> -> tensor<?x?xi64>
  // CHECK-DAG: %[[X3D_T:.*]]   = torch_c.to_builtin_tensor %[[X3D_VT]] : !torch.vtensor<[?,?,24],f32> -> tensor<?x?x24xf32>

  // COM: Expand to 4D with num_heads determined from the cos/sin cache shapes.
  // CHECK:     %[[EXPANDED:.*]] = tensor.expand_shape %[[X3D_T]] {{\[\[}}0], [1], [2, 3]] output_shape [%{{.*}}, %{{.*}}, 4, 6] : tensor<?x?x24xf32> into tensor<?x?x4x6xf32>
  // CHECK:     %[[TRANSPOSE_INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x4x?x6xf32>
  // CHECK:     %[[RESHAPED4D:.*]] = linalg.transpose ins(%[[EXPANDED]] : tensor<?x?x4x6xf32>) outs(%[[TRANSPOSE_INIT]] : tensor<?x4x?x6xf32>) permutation = [0, 2, 1, 3]

  // CHECK:     %[[INIT3D:.*]] = tensor.empty(
  // CHECK:     %[[ZERO3D:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:     %[[FILLED3D:.*]] = linalg.fill ins(%[[ZERO3D]] : f32) outs(%[[INIT3D]] : tensor<?x4x?x6xf32>) -> tensor<?x4x?x6xf32>

  // CHECK-DAG: %[[EMBED_HALF_BASE_IDX3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[EMBED_SIZE_IDX3:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[ONE3:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-DAG: %[[NEG_ONE3:.*]] = arith.constant -1.000000e+00 : f32

  // CHECK:     %[[GENERIC_3D:.*]] = linalg.generic
  // CHECK-SAME:   ins(%[[RESHAPED4D]], %[[POS_T3]] : tensor<?x4x?x6xf32>, tensor<?x?xi64>)
  // CHECK-SAME:   outs(%[[FILLED3D]] : tensor<?x4x?x6xf32>) {
  // CHECK:     ^bb0(%[[X_ELEM3:.*]]: f32, %[[POS_ELEM3:.*]]: i64, %[[OUT_INIT3:.*]]: f32):
  // CHECK:       %[[BATCH_IDX3:.*]] = linalg.index 0 : index
  // CHECK:       %[[HEAD_IDX3:.*]] = linalg.index 1 : index
  // CHECK:       %[[SEQ_IDX3:.*]] = linalg.index 2 : index
  // CHECK:       %[[EMBED_IDX3:.*]]  = linalg.index 3 : index
  // CHECK:       %[[EMBED_MOD3_3:.*]] = arith.remsi %[[EMBED_IDX3]], %[[EMBED_HALF_BASE_IDX3]] : index
  // CHECK:       %[[IS_UPPER3:.*]] = arith.cmpi sge, %[[EMBED_IDX3]], %[[EMBED_HALF_BASE_IDX3]] : index
  // CHECK:       %[[EMBED_PLUS3_3:.*]] = arith.addi %[[EMBED_IDX3]], %[[EMBED_HALF_BASE_IDX3]] : index
  // CHECK:       %[[PAIR_EMBED_IDX3:.*]] = arith.remsi %[[EMBED_PLUS3_3]], %[[EMBED_SIZE_IDX3]] : index
  // CHECK:       %[[POS_IDX3:.*]] = arith.index_cast %[[POS_ELEM3]] : i64 to index
  // CHECK:       %[[COS_ELEM3:.*]] = tensor.extract %[[COS_T3]][%[[POS_IDX3]], %[[EMBED_MOD3_3]]] : tensor<4x3xf32>
  // CHECK:       %[[SIN_ELEM3:.*]] = tensor.extract %[[SIN_T3]][%[[POS_IDX3]], %[[EMBED_MOD3_3]]] : tensor<4x3xf32>
  // CHECK:       %[[PAIR_X3:.*]] = tensor.extract %[[RESHAPED4D]][%[[BATCH_IDX3]], %[[HEAD_IDX3]], %[[SEQ_IDX3]], %[[PAIR_EMBED_IDX3]]] : tensor<?x4x?x6xf32>
  // CHECK:       %[[SIGN3:.*]] = arith.select %[[IS_UPPER3]], %[[ONE3]], %[[NEG_ONE3]] : f32
  // CHECK:       %[[PROD1_3:.*]] = arith.mulf %[[X_ELEM3]], %[[COS_ELEM3]] : f32
  // CHECK:       %[[PROD2_3:.*]] = arith.mulf %[[PAIR_X3]], %[[SIN_ELEM3]] : f32
  // CHECK:       %[[PROD2S_3:.*]] = arith.mulf %[[PROD2_3]], %[[SIGN3]] : f32
  // CHECK:       %[[SUM3:.*]] = arith.addf %[[PROD1_3]], %[[PROD2S_3]] : f32
  // CHECK:       linalg.yield %[[SUM3]] : f32
  // CHECK:     } -> tensor<?x4x?x6xf32>

  // COM: Restore the original 3D shape.
  // CHECK:     %[[CAST3:.*]] = tensor.cast %[[GENERIC_3D]] : tensor<?x4x?x6xf32> to tensor<?x4x?x6xf32>
  // CHECK:     %[[TRANSPOSE_INIT2:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4x6xf32>
  // CHECK:     %[[RESHAPE_BACK:.*]] = linalg.transpose ins(%[[CAST3]] : tensor<?x4x?x6xf32>) outs(%[[TRANSPOSE_INIT2]] : tensor<?x?x4x6xf32>) permutation = [0, 2, 1, 3]
  // CHECK:     %[[COLLAPSED3:.*]] = tensor.collapse_shape %[[RESHAPE_BACK]] {{\[\[}}0], [1], [2, 3]] : tensor<?x?x4x6xf32> into tensor<?x?x24xf32>
  // CHECK:     %[[RET3_VT:.*]] = torch_c.from_builtin_tensor %[[COLLAPSED3]] : tensor<?x?x24xf32> -> !torch.vtensor<[?,?,24],f32>
  // CHECK:     return %[[RET3_VT]] : !torch.vtensor<[?,?,24],f32>
  %int0 = torch.constant.int 0
  %int0_0 = torch.constant.int 0
  %int0_1 = torch.constant.int 0
  %int0_2 = torch.constant.int 0
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %0 = torch.onnx.rotary_embedding %arg0, %arg1, %arg2, %arg3, %int0, %int0_0, %int0_1, %int0_2, %float1.000000e00 : !torch.vtensor<[?,?,24],f32>, !torch.vtensor<[?,?],si64>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4,3],f32>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.float -> !torch.vtensor<[?,?,24],f32>
  return %0 : !torch.vtensor<[?,?,24],f32>
}
