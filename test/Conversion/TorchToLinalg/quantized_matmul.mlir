// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file --mlir-print-local-scope -verify-diagnostics | FileCheck %s

// Scalar zero points: both map to () i.e. broadcast to all elements.
// CHECK-LABEL: func.func @quantized_matmul_si8_scalar_zp
func.func @quantized_matmul_si8_scalar_zp(
    %lhs: !torch.vtensor<[4,3],si8>,
    %rhs: !torch.vtensor<[3,2],si8>,
    %lhs_zp: !torch.vtensor<[],si8>,
    %rhs_zp: !torch.vtensor<[],si8>) -> !torch.vtensor<[4,2],si32> {
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%[[A:.*]]: i8, %[[B:.*]]: i8, %[[ZPA:.*]]: i8, %[[ZPB:.*]]: i8, %[[ACC:.*]]: i32):
  // CHECK:        %[[AEXT:.*]] = arith.extsi %[[A]] : i8 to i32
  // CHECK:        %[[BEXT:.*]] = arith.extsi %[[B]] : i8 to i32
  // CHECK:        %[[ZPAEXT:.*]] = arith.extsi %[[ZPA]] : i8 to i32
  // CHECK:        %[[ASUB:.*]] = arith.subi %[[AEXT]], %[[ZPAEXT]] : i32
  // CHECK:        %[[ZPBEXT:.*]] = arith.extsi %[[ZPB]] : i8 to i32
  // CHECK:        %[[BSUB:.*]] = arith.subi %[[BEXT]], %[[ZPBEXT]] : i32
  // CHECK:        %[[MUL:.*]] = arith.muli %[[ASUB]], %[[BSUB]] : i32
  // CHECK:        %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : i32
  // CHECK:        linalg.yield %[[ADD]] : i32
  %0 = torch.quantized_matmul %lhs, %rhs, %lhs_zp, %rhs_zp : !torch.vtensor<[4,3],si8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[4,2],si32>
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// Mixed signedness: lhs is ui8 (extui), rhs is si8 (extsi).
// CHECK-LABEL: func.func @quantized_matmul_mixed_ui8_si8_scalar_zp
func.func @quantized_matmul_mixed_ui8_si8_scalar_zp(
    %lhs: !torch.vtensor<[4,3],ui8>,
    %rhs: !torch.vtensor<[3,2],si8>,
    %lhs_zp: !torch.vtensor<[],ui8>,
    %rhs_zp: !torch.vtensor<[],si8>) -> !torch.vtensor<[4,2],si32> {
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%[[A:.*]]: i8, %[[B:.*]]: i8, %[[ZPA:.*]]: i8, %[[ZPB:.*]]: i8, %[[ACC:.*]]: i32):
  // CHECK:        %[[AEXT:.*]] = arith.extui %[[A]] : i8 to i32
  // CHECK:        %[[BEXT:.*]] = arith.extsi %[[B]] : i8 to i32
  // CHECK:        arith.extui %[[ZPA]] : i8 to i32
  // CHECK:        arith.subi
  // CHECK:        arith.extsi %[[ZPB]] : i8 to i32
  // CHECK:        arith.subi
  // CHECK:        arith.muli
  // CHECK:        arith.addi
  // CHECK:        linalg.yield
  %0 = torch.quantized_matmul %lhs, %rhs, %lhs_zp, %rhs_zp : !torch.vtensor<[4,3],ui8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[],si8> -> !torch.vtensor<[4,2],si32>
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// Per-channel zero points: lhs_zp maps along M (d0), rhs_zp maps along N (d1).
// CHECK-LABEL: func.func @quantized_matmul_per_channel_zp
func.func @quantized_matmul_per_channel_zp(
    %lhs: !torch.vtensor<[4,3],si8>,
    %rhs: !torch.vtensor<[3,2],si8>,
    %lhs_zp: !torch.vtensor<[4],si8>,
    %rhs_zp: !torch.vtensor<[2],si8>) -> !torch.vtensor<[4,2],si32> {
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i32):
  // CHECK:        arith.extsi
  // CHECK:        arith.extsi
  // CHECK:        arith.extsi
  // CHECK:        arith.subi
  // CHECK:        arith.extsi
  // CHECK:        arith.subi
  // CHECK:        arith.muli
  // CHECK:        arith.addi
  // CHECK:        linalg.yield
  %0 = torch.quantized_matmul %lhs, %rhs, %lhs_zp, %rhs_zp : !torch.vtensor<[4,3],si8>, !torch.vtensor<[3,2],si8>, !torch.vtensor<[4],si8>, !torch.vtensor<[2],si8> -> !torch.vtensor<[4,2],si32>
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// No zero points: only lhs, rhs, and output maps. No subi ops.
// CHECK-LABEL: func.func @quantized_matmul_no_zp
func.func @quantized_matmul_no_zp(
    %lhs: !torch.vtensor<[4,3],si8>,
    %rhs: !torch.vtensor<[3,2],si8>) -> !torch.vtensor<[4,2],si32> {
  %none = torch.constant.none
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%[[A:.*]]: i8, %[[B:.*]]: i8, %[[ACC:.*]]: i32):
  // CHECK:        %[[AEXT:.*]] = arith.extsi %[[A]] : i8 to i32
  // CHECK:        %[[BEXT:.*]] = arith.extsi %[[B]] : i8 to i32
  // CHECK-NOT:    arith.subi
  // CHECK:        %[[MUL:.*]] = arith.muli %[[AEXT]], %[[BEXT]] : i32
  // CHECK:        %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : i32
  // CHECK:        linalg.yield %[[ADD]] : i32
  %0 = torch.quantized_matmul %lhs, %rhs, %none, %none : !torch.vtensor<[4,3],si8>, !torch.vtensor<[3,2],si8>, !torch.none, !torch.none -> !torch.vtensor<[4,2],si32>
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// Asymmetric zero points: only rhs has a zero point, lhs is none.
// CHECK-LABEL: func.func @quantized_matmul_rhs_zp_only
func.func @quantized_matmul_rhs_zp_only(
    %lhs: !torch.vtensor<[4,3],si8>,
    %rhs: !torch.vtensor<[3,2],si8>,
    %rhs_zp: !torch.vtensor<[],si8>) -> !torch.vtensor<[4,2],si32> {
  %none = torch.constant.none
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%[[A:.*]]: i8, %[[B:.*]]: i8, %[[ZPB:.*]]: i8, %[[ACC:.*]]: i32):
  // CHECK:        %[[AEXT:.*]] = arith.extsi %[[A]] : i8 to i32
  // CHECK:        %[[BEXT:.*]] = arith.extsi %[[B]] : i8 to i32
  // CHECK:        %[[ZPBEXT:.*]] = arith.extsi %[[ZPB]] : i8 to i32
  // CHECK:        %[[BSUB:.*]] = arith.subi %[[BEXT]], %[[ZPBEXT]] : i32
  // CHECK:        %[[MUL:.*]] = arith.muli %[[AEXT]], %[[BSUB]] : i32
  // CHECK:        %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : i32
  // CHECK:        linalg.yield %[[ADD]] : i32
  %0 = torch.quantized_matmul %lhs, %rhs, %none, %rhs_zp : !torch.vtensor<[4,3],si8>, !torch.vtensor<[3,2],si8>, !torch.none, !torch.vtensor<[],si8> -> !torch.vtensor<[4,2],si32>
  return %0 : !torch.vtensor<[4,2],si32>
}

// -----

// Batch matmul: batch dim d0 is parallel, then m=d1, n=d2, k=d3 (reduction).
// CHECK-LABEL: func.func @quantized_matmul_batch
func.func @quantized_matmul_batch(
    %lhs: !torch.vtensor<[2,4,3],si8>,
    %rhs: !torch.vtensor<[2,3,2],si8>,
    %lhs_zp: !torch.vtensor<[],si8>,
    %rhs_zp: !torch.vtensor<[],si8>) -> !torch.vtensor<[2,4,2],si32> {
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
  // CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK:      ^bb0(%{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i8, %{{.*}}: i32):
  // CHECK:        arith.extsi
  // CHECK:        arith.extsi
  // CHECK:        arith.extsi
  // CHECK:        arith.subi
  // CHECK:        arith.extsi
  // CHECK:        arith.subi
  // CHECK:        arith.muli
  // CHECK:        arith.addi
  // CHECK:        linalg.yield
  %0 = torch.quantized_matmul %lhs, %rhs, %lhs_zp, %rhs_zp : !torch.vtensor<[2,4,3],si8>, !torch.vtensor<[2,3,2],si8>, !torch.vtensor<[],si8>, !torch.vtensor<[],si8> -> !torch.vtensor<[2,4,2],si32>
  return %0 : !torch.vtensor<[2,4,2],si32>
}

// -----

// Rank-mismatch: rank-3 lhs with rank-2 rhs.
// Lhs batch dims are collapsed into M, matmul is rank-2, then result is expanded.
// CHECK-LABEL: func.func @quantized_matmul_broadcast_rhs
func.func @quantized_matmul_broadcast_rhs(
    %lhs: !torch.vtensor<[?,?,1024],ui8>,
    %rhs: !torch.vtensor<[1024,1024],si8>,
    %lhs_zp: !torch.vtensor<[],ui8>,
    %rhs_zp: !torch.vtensor<[1024],si8>) -> !torch.vtensor<[?,?,1024],si32> {
  // CHECK:      tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]]
  // CHECK:      linalg.generic
  // CHECK-SAME:   indexing_maps =
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d2)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d2, d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> ()>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d1)>
  // CHECK-SAME:     affine_map<(d0, d1, d2) -> (d0, d1)>
  // CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK:      tensor.expand_shape %{{.*}} {{\[\[}}0, 1], [2]]
  %0 = torch.quantized_matmul %lhs, %rhs, %lhs_zp, %rhs_zp : !torch.vtensor<[?,?,1024],ui8>, !torch.vtensor<[1024,1024],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[1024],si8> -> !torch.vtensor<[?,?,1024],si32>
  return %0 : !torch.vtensor<[?,?,1024],si32>
}
