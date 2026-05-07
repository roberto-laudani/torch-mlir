// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file --mlir-print-local-scope | FileCheck %s

// -----

// bits=4, integer zero-points, with bias. 
// CHECK-LABEL: func.func @test_block_quant_matmul_bits4_with_bias
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,16],f32> -> tensor<?x?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,2,8],ui8> -> tensor<8x2x8xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,2],f32> -> tensor<8x2xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,2],ui8> -> tensor<8x2xi8>
// CHECK-DAG:   %[[BIAS:.*]] = torch_c.to_builtin_tensor %arg4 : !torch.vtensor<[8],f32> -> tensor<8xf32>

// Hoisted constants for symmetric int8 quantization of A.
// CHECK-DAG:   %[[C127:.*]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:   %[[CMIN128:.*]] = arith.constant -1.280000e+02 : f32

// Step 1a: Reshape lhs (Batch, M, K) -> (Batch, M, KB, BS).
// CHECK:       %[[LHS_EXPANDED:.*]] = tensor.expand_shape %[[LHS]] {{\[\[}}0], [1], [2, 3]] output_shape [%{{.*}}, %{{.*}}, 2, 8] : tensor<?x?x16xf32> into tensor<?x?x2x8xf32>

// Step 1b: Block-wise max(|A|): (Batch, M, KB, BS) -> (Batch, M, KB), reduction over BS.
// CHECK:       %[[MAXABS_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x2xf32>
// CHECK:       %[[MAXABS_ZERO:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[MAXABS_EMPTY]] : tensor<?x?x2xf32>)
// CHECK:       %[[MAXABS:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[LHS_EXPANDED]] : tensor<?x?x2x8xf32>)
// CHECK-SAME:  outs(%[[MAXABS_ZERO]] : tensor<?x?x2xf32>)
// CHECK:         ^bb0(%[[IN_A:.*]]: f32, %[[OUT_ACC:.*]]: f32):
// CHECK:         %[[ABS:.*]] = math.absf %[[IN_A]] : f32
// CHECK:         %[[MAX:.*]] = arith.maximumf %[[ABS]], %[[OUT_ACC]] : f32
// CHECK:         linalg.yield %[[MAX]] : f32
// CHECK:       } -> tensor<?x?x2xf32>

// Step 1c: lhsScales = maxAbs / 127.
// CHECK:       %[[LHS_SCALES_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x2xf32>
// CHECK:       %[[LHS_SCALES:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[MAXABS]] : tensor<?x?x2xf32>)
// CHECK-SAME:  outs(%[[LHS_SCALES_EMPTY]] : tensor<?x?x2xf32>)
// CHECK:         ^bb0(%[[IN_MAXABS:.*]]: f32, %{{.*}}: f32):
// CHECK:         %[[SCALE:.*]] = arith.divf %[[IN_MAXABS]], %[[C127]] : f32
// CHECK:         linalg.yield %[[SCALE]] : f32
// CHECK:       } -> tensor<?x?x2xf32>

// Step 1c': lhsInvScales = (maxAbs == 0) ? 0 : 127 / maxAbs (zero-guard).
// CHECK:       %[[LHS_INV_SCALES:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[MAXABS]] : tensor<?x?x2xf32>)
// CHECK:         %[[CZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[ISZERO:.*]] = arith.cmpf oeq, %{{.*}}, %[[CZERO]] : f32
// CHECK:         %[[INV:.*]] = arith.divf %[[C127]], %{{.*}} : f32
// CHECK:         %[[SAFE:.*]] = arith.select %[[ISZERO]], %[[CZERO]], %[[INV]] : f32
// CHECK:         linalg.yield %[[SAFE]] : f32
// CHECK:       } -> tensor<?x?x2xf32>

// Step 1d: Quantize lhs: q = clamp(round(lhs * inv_scale), [-128, 127]) -> i8.
// CHECK:       %[[LHS_Q_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x2x8xi8>
// CHECK:       %[[LHS_Q:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[LHS_EXPANDED]], %[[LHS_INV_SCALES]] : tensor<?x?x2x8xf32>, tensor<?x?x2xf32>)
// CHECK-SAME:  outs(%[[LHS_Q_EMPTY]] : tensor<?x?x2x8xi8>)
// CHECK:         ^bb0(%[[IN_LHS:.*]]: f32, %[[IN_INV:.*]]: f32, %{{.*}}: i8):
// CHECK:         %[[MUL:.*]] = arith.mulf %[[IN_LHS]], %[[IN_INV]] : f32
// CHECK:         %[[ROUND:.*]] = math.roundeven %[[MUL]] : f32
// CHECK:         %[[CLAMP_LO:.*]] = arith.maximumf %[[ROUND]], %[[CMIN128]] : f32
// CHECK:         %[[CLAMP_HI:.*]] = arith.minimumf %[[CLAMP_LO]], %[[C127]] : f32
// CHECK:         %[[Q:.*]] = arith.fptosi %[[CLAMP_HI]] : f32 to i8
// CHECK:         linalg.yield %[[Q]] : i8
// CHECK:       } -> tensor<?x?x2x8xi8>

// Step 2: Integer matmul with fused zp subtraction.
// rhs, and the zero-points (broadcast over BS via (dN, dKB)). The body
// extends i8 -> i32 for all three, subtracts (rhs - zp) at i32, then mac.
// CHECK:       %[[INT_ACC_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x8x2xi32>
// CHECK:       %[[INT_ACC_ZERO:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[INT_ACC_EMPTY]] : tensor<?x?x8x2xi32>)
// CHECK:       %[[INT_ACC:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[LHS_Q]], %[[RHS]], %[[ZP]] : tensor<?x?x2x8xi8>, tensor<8x2x8xi8>, tensor<8x2xi8>)
// CHECK-SAME:  outs(%[[INT_ACC_ZERO]] : tensor<?x?x8x2xi32>)
// CHECK:         ^bb0(%[[IN_L:.*]]: i8, %[[IN_R:.*]]: i8, %[[IN_Z:.*]]: i8, %[[IN_ACC:.*]]: i32):
// CHECK:         %[[L_I32:.*]] = arith.extsi %[[IN_L]] : i8 to i32
// CHECK:         %[[R_I32:.*]] = arith.extui %[[IN_R]] : i8 to i32
// CHECK:         %[[Z_I32:.*]] = arith.extui %[[IN_Z]] : i8 to i32
// CHECK:         %[[DIFF:.*]] = arith.subi %[[R_I32]], %[[Z_I32]] : i32
// CHECK:         %[[PROD:.*]] = arith.muli %[[L_I32]], %[[DIFF]] : i32
// CHECK:         %[[NEW_ACC:.*]] = arith.addi %[[PROD]], %[[IN_ACC]] : i32
// CHECK:         linalg.yield %[[NEW_ACC]] : i32
// CHECK:       } -> tensor<?x?x8x2xi32>

// Step 3: Dequant: sitofp(intAcc) * lhsScale * rhsScale, KB-reduction.
// CHECK:       %[[DEQ_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x8xf32>
// CHECK:       %[[DEQ_ZERO:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[DEQ_EMPTY]] : tensor<?x?x8xf32>)
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[INT_ACC]], %[[LHS_SCALES]], %[[SCALES]] : tensor<?x?x8x2xi32>, tensor<?x?x2xf32>, tensor<8x2xf32>)
// CHECK-SAME:  outs(%[[DEQ_ZERO]] : tensor<?x?x8xf32>)
// CHECK:         ^bb0(%[[I:.*]]: i32, %[[LS:.*]]: f32, %[[RS:.*]]: f32, %[[ACC:.*]]: f32):
// CHECK:         %[[F:.*]] = arith.sitofp %[[I]] : i32 to f32
// CHECK:         %[[P1:.*]] = arith.mulf %[[F]], %[[LS]] : f32
// CHECK:         %[[P2:.*]] = arith.mulf %[[P1]], %[[RS]] : f32
// CHECK:         %[[NEW:.*]] = arith.addf %[[P2]], %[[ACC]] : f32
// CHECK:         linalg.yield %[[NEW]] : f32
// CHECK:       } -> tensor<?x?x8xf32>

// Step 4: Bias broadcast and accumulation.
// CHECK:       %[[BIAS_EMPTY:.*]] = tensor.empty({{.*}}) : tensor<?x?x8xf32>
// CHECK:       %[[BIASED:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2) -> (d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
// CHECK-SAME:   affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[BIAS]], %[[DEQUANT]] : tensor<8xf32>, tensor<?x?x8xf32>)
// CHECK-SAME:  outs(%[[BIAS_EMPTY]] : tensor<?x?x8xf32>)
// CHECK:         ^bb0(%[[IN_BIAS:.*]]: f32, %[[IN_DEQ:.*]]: f32, %{{.*}}: f32):
// CHECK:         %[[SUM:.*]] = arith.addf %[[IN_BIAS]], %[[IN_DEQ]] : f32
// CHECK:         linalg.yield %[[SUM]] : f32
// CHECK:       } -> tensor<?x?x8xf32>

func.func @test_block_quant_matmul_bits4_with_bias(
    %arg0: !torch.vtensor<[?,?,16],f32>, %arg1: !torch.vtensor<[8,2,8],ui8>,
    %arg2: !torch.vtensor<[8,2],f32>, %arg3: !torch.vtensor<[8,2],ui8>,
    %arg4: !torch.vtensor<[8],f32>) -> !torch.vtensor<[?,?,8],f32> {
    %result = torch.block_quant_matmul %arg0, %arg1, %arg2, %arg3, %arg4 {bits = 4 : si64} :
        !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[8,2,8],ui8>, !torch.vtensor<[8,2],f32>,
        !torch.vtensor<[8,2],ui8>, !torch.vtensor<[8],f32> -> !torch.vtensor<[?,?,8],f32>
    return %result : !torch.vtensor<[?,?,8],f32>
}

// -----

// bits=8, integer zero-points, no bias.
// CHECK-LABEL: func.func @test_block_quant_matmul_bits8_no_bias
// CHECK-DAG:   %[[LHS:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?,16],f32> -> tensor<?x?x16xf32>
// CHECK-DAG:   %[[RHS:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[8,4,4],ui8> -> tensor<8x4x4xi8>
// CHECK-DAG:   %[[SCALES:.*]] = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[8,4],f32> -> tensor<8x4xf32>
// CHECK-DAG:   %[[ZP:.*]] = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[8,4],ui8> -> tensor<8x4xi8>

// CHECK:       %[[LHS_Q_INIT:.*]] = tensor.empty({{.*}}) : tensor<?x?x4x4xi8>
// CHECK:       %[[LHS_Q:.*]] = linalg.generic
// CHECK-SAME:    outs(%[[LHS_Q_INIT]] : tensor<?x?x4x4xi8>)
// CHECK:         arith.fptosi
// CHECK:       } -> tensor<?x?x4x4xi8>

// Step 2: Integer matmul with fused zp subtraction (i8 x i8 -> i32).
// CHECK:       %[[INT_ACC:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[LHS_Q]], %[[RHS]], %[[ZP]] : tensor<?x?x4x4xi8>, tensor<8x4x4xi8>, tensor<8x4xi8>)
// CHECK:         ^bb0(%[[IN_L:.*]]: i8, %[[IN_R:.*]]: i8, %[[IN_Z:.*]]: i8, %[[IN_ACC:.*]]: i32):
// CHECK:         %[[L_I32:.*]] = arith.extsi %[[IN_L]] : i8 to i32
// CHECK:         %[[R_I32:.*]] = arith.extui %[[IN_R]] : i8 to i32
// CHECK:         %[[Z_I32:.*]] = arith.extui %[[IN_Z]] : i8 to i32
// CHECK:         %[[DIFF:.*]] = arith.subi %[[R_I32]], %[[Z_I32]] : i32
// CHECK:         %[[PROD:.*]] = arith.muli %[[L_I32]], %[[DIFF]] : i32
// CHECK:         %[[NEW_ACC:.*]] = arith.addi %[[PROD]], %[[IN_ACC]] : i32
// CHECK:         linalg.yield %[[NEW_ACC]] : i32
// CHECK:       } -> tensor<?x?x8x4xi32>

// Step 3: Dequant.
// CHECK:       %[[DEQUANT:.*]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:  [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
// CHECK-SAME:   affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK:         arith.sitofp
// CHECK:         arith.mulf
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:       } -> tensor<?x?x8xf32>

// Step 4 (skipped): no bias.
// CHECK-NOT:   affine_map<(d0, d1, d2) -> (d2)>

func.func @test_block_quant_matmul_bits8_no_bias(
    %arg0: !torch.vtensor<[?,?,16],f32>, %arg1: !torch.vtensor<[8,4,4],ui8>,
    %arg2: !torch.vtensor<[8,4],f32>, %arg3: !torch.vtensor<[8,4],ui8>) -> !torch.vtensor<[?,?,8],f32> {
  %none = torch.constant.none
  %result = torch.block_quant_matmul %arg0, %arg1, %arg2, %arg3, %none {bits = 8 : si64} :
    !torch.vtensor<[?,?,16],f32>, !torch.vtensor<[8,4,4],ui8>, !torch.vtensor<[8,4],f32>,
    !torch.vtensor<[8,4],ui8>, !torch.none -> !torch.vtensor<[?,?,8],f32>
  return %result : !torch.vtensor<[?,?,8],f32>
}
