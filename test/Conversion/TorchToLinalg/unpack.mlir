// RUN: torch-mlir-opt %s -convert-torch-to-linalg -split-input-file --mlir-print-local-scope | FileCheck %s

// -----

// bits=4: two 4-bit fields per byte.
// CHECK-LABEL: func.func @test_unpack_bits4_static
// CHECK:       %[[PACKED:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,2,4],ui8> -> tensor<8x2x4xi8>
// CHECK:       %[[OUT:.+]] = tensor.empty() : tensor<8x2x4x2xi8>
// CHECK:       %[[UNPACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
// CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[PACKED]] : tensor<8x2x4xi8>)
// CHECK-SAME:  outs(%[[OUT]] : tensor<8x2x4x2xi8>)
// CHECK:         ^bb0(%[[IN:[^:]+]]: i8, %[[OUT_INIT:[^:]+]]: i8):
// CHECK:         %[[IDX:.+]] = linalg.index 3 : index
// CHECK:         %[[IDX_I8:.+]] = arith.index_cast %[[IDX]] : index to i8
// CHECK:         %[[BITS:.+]] = arith.constant 4 : i8
// CHECK:         %[[MASK:.+]] = arith.constant 15 : i8
// CHECK:         %[[SHIFT:.+]] = arith.muli %[[IDX_I8]], %[[BITS]] : i8
// CHECK:         %[[SHR:.+]] = arith.shrui %[[IN]], %[[SHIFT]] : i8
// CHECK:         %[[OUT_VAL:.+]] = arith.andi %[[SHR]], %[[MASK]] : i8
// CHECK:         linalg.yield %[[OUT_VAL]] : i8
// CHECK:       } -> tensor<8x2x4x2xi8>
// CHECK:       tensor.collapse_shape %[[UNPACKED]] {{\[\[}}0], [1], [2, 3]] : tensor<8x2x4x2xi8> into tensor<8x2x8xi8>
func.func @test_unpack_bits4_static(
    %arg0: !torch.vtensor<[8,2,4],ui8>) -> !torch.vtensor<[8,2,8],ui8> {
  %result = torch.sub_byte_unpack %arg0 {bits = 4 : si64}
      : !torch.vtensor<[8,2,4],ui8> -> !torch.vtensor<[8,2,8],ui8>
  return %result : !torch.vtensor<[8,2,8],ui8>
}

// -----

// bits=2: four 2-bit fields per byte.
// CHECK-LABEL: func.func @test_unpack_bits2_static
// CHECK:       %[[PACKED:.+]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[8,2,4],ui8> -> tensor<8x2x4xi8>
// CHECK:       %[[OUT:.+]] = tensor.empty() : tensor<8x2x4x4xi8>
// CHECK:       %[[UNPACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps =
// CHECK-SAME:    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
// CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[PACKED]] : tensor<8x2x4xi8>)
// CHECK-SAME:  outs(%[[OUT]] : tensor<8x2x4x4xi8>)
// CHECK:         ^bb0(%[[IN:[^:]+]]: i8, %[[OUT_INIT:[^:]+]]: i8):
// CHECK:         %[[IDX:.+]] = linalg.index 3 : index
// CHECK:         %[[IDX_I8:.+]] = arith.index_cast %[[IDX]] : index to i8
// CHECK:         %[[BITS:.+]] = arith.constant 2 : i8
// CHECK:         %[[MASK:.+]] = arith.constant 3 : i8
// CHECK:         %[[SHIFT:.+]] = arith.muli %[[IDX_I8]], %[[BITS]] : i8
// CHECK:         %[[SHR:.+]] = arith.shrui %[[IN]], %[[SHIFT]] : i8
// CHECK:         %[[OUT_VAL:.+]] = arith.andi %[[SHR]], %[[MASK]] : i8
// CHECK:         linalg.yield %[[OUT_VAL]] : i8
// CHECK:       } -> tensor<8x2x4x4xi8>
// CHECK:       tensor.collapse_shape %[[UNPACKED]] {{\[\[}}0], [1], [2, 3]] : tensor<8x2x4x4xi8> into tensor<8x2x16xi8>
func.func @test_unpack_bits2_static(
    %arg0: !torch.vtensor<[8,2,4],ui8>) -> !torch.vtensor<[8,2,16],ui8> {
  %result = torch.sub_byte_unpack %arg0 {bits = 2 : si64}
      : !torch.vtensor<[8,2,4],ui8> -> !torch.vtensor<[8,2,16],ui8>
  return %result : !torch.vtensor<[8,2,16],ui8>
}
