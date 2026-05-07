// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file --mlir-print-local-scope | FileCheck %s

// -----

// Integer (sub-byte-unpacked) zero-points, with bias, dynamic batch and M dimensions.
// CHECK-LABEL: func.func @test_block_dequant_matmul_int_zp_with_bias
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,16],f32> -> tensor<?x?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,2,8],ui8> -> tensor<8x2x8xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,2],ui8> -> tensor<8x2xi8>
// CHECK-DAG:   %[[BIAS:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[8],f32> -> tensor<8xf32>

// Step 1: Dequantize rhs: (rhs - zp) * scale over (N, KB, BS).
// CHECK:       %[[DEQUANT_INIT:.*]] = tensor.empty() : tensor<8x2x8xf32>
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[RHS]], %[[SCALES]], %[[ZP]] : tensor<8x2x8xi8>, tensor<8x2xf32>, tensor<8x2xi8>)
// CHECK-SAME:  outs(%[[DEQUANT_INIT]] : tensor<8x2x8xf32>)
// CHECK:         ^bb0(%[[IN_R:.*]]: i8, %[[IN_S:.*]]: f32, %[[IN_Z:.*]]: i8, %[[OUT_INIT:.*]]: f32):
// CHECK:         %[[R_F:.*]] = arith.uitofp %[[IN_R]] : i8 to f32
// CHECK:         %[[Z_F:.*]] = arith.uitofp %[[IN_Z]] : i8 to f32
// CHECK:         %[[SHIFTED:.*]] = arith.subf %[[R_F]], %[[Z_F]] : f32
// CHECK:         %[[SCALED:.*]] = arith.mulf %[[SHIFTED]], %[[IN_S]] : f32
// CHECK:         linalg.yield %[[SCALED]] : f32
// CHECK:       } -> tensor<8x2x8xf32>

// Step 2: Expand lhs (Batch, M, K) -> (Batch, M, KB, BS).
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[BATCH_DIM:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x16xf32>
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[M_DIM:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x16xf32>
// CHECK:       %[[LHS_EXPANDED:.*]] = tensor.expand_shape %[[LHS]] {{\[\[}}0], [1], [2, 3]] output_shape [%[[BATCH_DIM]], %[[M_DIM]], 2, 8] : tensor<?x?x16xf32> into tensor<?x?x2x8xf32>

// Step 3: Blocked matmul (Batch, M, N) parallel x (KB, BS) reduction.
// CHECK:       %[[C0_MM:.*]] = arith.constant 0 : index
// CHECK:       %[[BATCH_DIM_MM:.*]] = tensor.dim %[[LHS]], %[[C0_MM]] : tensor<?x?x16xf32>
// CHECK:       %[[C1_MM:.*]] = arith.constant 1 : index
// CHECK:       %[[M_DIM_MM:.*]] = tensor.dim %[[LHS]], %[[C1_MM]] : tensor<?x?x16xf32>
// CHECK:       %[[MATMUL_EMPTY:.*]] = tensor.empty(%[[BATCH_DIM_MM]], %[[M_DIM_MM]]) : tensor<?x?x8xf32>
// CHECK:       %[[MATMUL_ZERO:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[MATMUL_EMPTY]] : tensor<?x?x8xf32>)
// CHECK:       %[[MATMUL:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[LHS_EXPANDED]], %[[DEQUANT]] : tensor<?x?x2x8xf32>, tensor<8x2x8xf32>)
// CHECK-SAME:  outs(%[[MATMUL_ZERO]] : tensor<?x?x8xf32>)
// CHECK:         ^bb0(%[[IN_LHS:.*]]: f32, %[[IN_DEQ:.*]]: f32, %[[ACC:.*]]: f32):
// CHECK:         %[[PROD:.*]] = arith.mulf %[[IN_LHS]], %[[IN_DEQ]] : f32
// CHECK:         %[[NEW_ACC:.*]] = arith.addf %[[PROD]], %[[ACC]] : f32
// CHECK:         linalg.yield %[[NEW_ACC]] : f32
// CHECK:       } -> tensor<?x?x8xf32>

// Step 4: Bias broadcast and accumulation.
// CHECK:       %[[C0_B:.*]] = arith.constant 0 : index
// CHECK:       %[[BATCH_DIM_B:.*]] = tensor.dim %[[MATMUL]], %[[C0_B]] : tensor<?x?x8xf32>
// CHECK:       %[[C1_B:.*]] = arith.constant 1 : index
// CHECK:       %[[M_DIM_B:.*]] = tensor.dim %[[MATMUL]], %[[C1_B]] : tensor<?x?x8xf32>
// CHECK:       %[[BIAS_EMPTY:.*]] = tensor.empty(%[[BATCH_DIM_B]], %[[M_DIM_B]]) : tensor<?x?x8xf32>
// CHECK:       %[[BIASED:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[BIAS]], %[[MATMUL]] : tensor<8xf32>, tensor<?x?x8xf32>)
// CHECK-SAME:  outs(%[[BIAS_EMPTY]] : tensor<?x?x8xf32>)
// CHECK:         ^bb0(%[[IN_BIAS:.*]]: f32, %[[IN_MAT:.*]]: f32, %[[OUT_INIT:.*]]: f32):
// CHECK:         %[[SUM:.*]] = arith.addf %[[IN_BIAS]], %[[IN_MAT]] : f32
// CHECK:         linalg.yield %[[SUM]] : f32
// CHECK:       } -> tensor<?x?x8xf32>

// Step 5 (skipped): no shape refinement (result type matches matmul output).
// CHECK-NOT:   tensor.cast

func.func @test_block_dequant_matmul_int_zp_with_bias(
    %arg0: !torch.vtensor<[?,?,16],f32>, %arg1: !torch.vtensor<[8,2,8],ui8>,
    %arg2: !torch.vtensor<[8,2],f32>, %arg3: !torch.vtensor<[8,2],ui8>,
    %arg4: !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,?,8],f32> {
    %result = torch.block_dequant_matmul %arg0, %arg1, %arg2, %arg3, %arg4 :
        !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[8,2,8],ui8>, !torch.vtensor<[8,2],f32>,
        !torch.vtensor<[8,2],ui8>, !torch.vtensor<[8],f32> -> !torch.vtensor<[?,?,8],f32>
    return %result : !torch.vtensor<[?,?,8],f32>
}

// -----

// Per-block float zero-points (never sub-byte-packed).
// CHECK-LABEL: func.func @test_block_dequant_matmul_float_zp
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,16],f32> -> tensor<?x?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,2,8],ui8> -> tensor<8x2x8xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[BIAS:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[8],f32> -> tensor<8xf32>

// Step 1: Dequant; only rhs goes through uitofp, zp is consumed as f32.
// CHECK:       %[[DEQUANT_INIT:.*]] = tensor.empty() : tensor<8x2x8xf32>
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[RHS]], %[[SCALES]], %[[ZP]] : tensor<8x2x8xi8>, tensor<8x2xf32>, tensor<8x2xf32>)
// CHECK-SAME:  outs(%[[DEQUANT_INIT]] : tensor<8x2x8xf32>)
// CHECK:         ^bb0(%[[IN_R:.*]]: i8, %[[IN_S:.*]]: f32, %[[IN_Z:.*]]: f32, %[[OUT_INIT:.*]]: f32):
// CHECK:         %[[R_F:.*]] = arith.uitofp %[[IN_R]] : i8 to f32
// CHECK-NOT:     arith.uitofp
// CHECK:         %[[SHIFTED:.*]] = arith.subf %[[R_F]], %[[IN_Z]] : f32
// CHECK:         %[[SCALED:.*]] = arith.mulf %[[SHIFTED]], %[[IN_S]] : f32
// CHECK:         linalg.yield %[[SCALED]] : f32
// CHECK:       } -> tensor<8x2x8xf32>

// Step 2: Expand lhs.
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[BATCH_DIM:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x16xf32>
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[M_DIM:.*]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x16xf32>
// CHECK:       %[[LHS_EXPANDED:.*]] = tensor.expand_shape %[[LHS]] {{\[\[}}0], [1], [2, 3]] output_shape [%[[BATCH_DIM]], %[[M_DIM]], 2, 8] : tensor<?x?x16xf32> into tensor<?x?x2x8xf32>

// Step 3: Matmul.
// CHECK:       %[[MATMUL:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[LHS_EXPANDED]], %[[DEQUANT]] : tensor<?x?x2x8xf32>, tensor<8x2x8xf32>)

// Step 4: Bias add.
// CHECK:       %[[BIASED:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[BIAS]], %[[MATMUL]] : tensor<8xf32>, tensor<?x?x8xf32>)

// CHECK-NOT:   tensor.cast

func.func @test_block_dequant_matmul_float_zp(
    %arg0: !torch.vtensor<[?,?,16],f32>, %arg1: !torch.vtensor<[8,2,8],ui8>,
    %arg2: !torch.vtensor<[8,2],f32>, %arg3: !torch.vtensor<[8,2],f32>,
    %arg4: !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,?,8],f32> {
  %result = torch.block_dequant_matmul %arg0, %arg1, %arg2, %arg3, %arg4 :
    !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[8,2,8],ui8>, !torch.vtensor<[8,2],f32>,
    !torch.vtensor<[8,2],f32>, !torch.vtensor<[8],f32> -> !torch.vtensor<[?,?,8],f32>
  return %result : !torch.vtensor<[?,?,8],f32>
}

// -----

// CHECK-LABEL: func.func @test_block_dequant_matmul_int_zp_no_bias
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,16],f32> -> tensor<?x?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,2,8],ui8> -> tensor<8x2x8xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,2],ui8> -> tensor<8x2xi8>

// Step 1: Dequant.
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[RHS]], %[[SCALES]], %[[ZP]] : tensor<8x2x8xi8>, tensor<8x2xf32>, tensor<8x2xi8>)

// Step 2: Expand lhs.
// CHECK:       tensor.expand_shape

// Step 3: Matmul.
// CHECK:       %[[MATMUL:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]

// Step 4 (skipped): no bias-broadcast generic should appear.
// CHECK-NOT:   affine_map<(d0, d1, d2) -> (d2)>
// CHECK-NOT:   tensor.cast

func.func @test_block_dequant_matmul_int_zp_no_bias(
    %arg0: !torch.vtensor<[?,?,16],f32>, %arg1: !torch.vtensor<[8,2,8],ui8>,
    %arg2: !torch.vtensor<[8,2],f32>, %arg3: !torch.vtensor<[8,2],ui8>) -> !torch.vtensor<[?,?,8],f32> {
  %none = torch.constant.none
  %result = torch.block_dequant_matmul %arg0, %arg1, %arg2, %arg3, %none :
    !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[8,2,8],ui8>, !torch.vtensor<[8,2],f32>,
    !torch.vtensor<[8,2],ui8>, !torch.none -> !torch.vtensor<[?,?,8],f32>
  return %result : !torch.vtensor<[?,?,8],f32>
}

// -----

// 2D lhs (batchRank == 0): no leading batch dim.
// CHECK-LABEL: func.func @test_block_dequant_matmul_no_batch
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,16],f32> -> tensor<?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,2,8],ui8> -> tensor<8x2x8xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,2],ui8> -> tensor<8x2xi8>
// CHECK-DAG:   %[[BIAS:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[8],f32> -> tensor<8xf32>

// Step 1: Dequant rhs (rhs-only domain, batchRank-independent).
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[RHS]], %[[SCALES]], %[[ZP]] : tensor<8x2x8xi8>, tensor<8x2xf32>, tensor<8x2xi8>)

// Step 2: Expand lhs (M, K) -> (M, KB, BS).
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[M_DIM:.*]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x16xf32>
// CHECK:       %[[LHS_EXPANDED:.*]] = tensor.expand_shape %[[LHS]] {{\[\[}}0], [1, 2]] output_shape [%[[M_DIM]], 2, 8] : tensor<?x16xf32> into tensor<?x2x8xf32>

// Step 3: MatMul
// CHECK:       %[[MATMUL:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[LHS_EXPANDED]], %[[DEQUANT]] : tensor<?x2x8xf32>, tensor<8x2x8xf32>)

// Step 4: Bias add.
// CHECK:       %[[BIASED:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1) -> (d1)>,
// CHECK-SAME:   affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-SAME:   affine_map<(d0, d1) -> (d0, d1)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[BIAS]], %[[MATMUL]] : tensor<8xf32>, tensor<?x8xf32>)

// CHECK-NOT:   tensor.cast

func.func @test_block_dequant_matmul_no_batch(
    %arg0: !torch.vtensor<[?,16],f32>, %arg1: !torch.vtensor<[8,2,8],ui8>,
    %arg2: !torch.vtensor<[8,2],f32>, %arg3: !torch.vtensor<[8,2],ui8>,
    %arg4: !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,8],f32> {
  %result = torch.block_dequant_matmul %arg0, %arg1, %arg2, %arg3, %arg4 :
    !torch.vtensor<[?,16],f32>, !torch.vtensor<[8,2,8],ui8>, !torch.vtensor<[8,2],f32>,
    !torch.vtensor<[8,2],ui8>, !torch.vtensor<[8],f32> -> !torch.vtensor<[?,8],f32>
  return %result : !torch.vtensor<[?,8],f32>
}
