// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -split-input-file -verify-diagnostics | FileCheck %s
// RUN: torch-mlir-opt <%s -convert-torch-to-linalg -canonicalize -split-input-file -verify-diagnostics | FileCheck %s --check-prefix=CANON

// A statically known, non-negative pad keeps the fast path: a single
// tensor.pad applied directly to the input, with no cropping slice or
// sign-handling arithmetic.

// CHECK-LABEL:   func.func @constant_pad_nd$static_positive(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,3],f32>
// CHECK:           %[[IN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,3],f32> -> tensor<2x3xf32>
// CHECK:           %[[FILL:.*]] = arith.truncf %{{.*}} : f64 to f32
// CHECK-NOT:       tensor.extract_slice
// CHECK-NOT:       arith.maxsi
// CHECK:           %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 0] high[0, 2]
// CHECK:             tensor.yield %[[FILL]] : f32
// CHECK:           } : tensor<2x3xf32> to tensor<2x5xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[PAD]] : tensor<2x5xf32> to tensor<2x5xf32>
// CHECK:           torch_c.from_builtin_tensor %[[CAST]] : tensor<2x5xf32> -> !torch.vtensor<[2,5],f32>

// CANON-LABEL:   func.func @constant_pad_nd$static_positive(
// CANON:           tensor.pad %{{.*}} low[0, 0] high[0, 2]
// CANON:         } : tensor<2x3xf32> to tensor<2x5xf32>
func.func @constant_pad_nd$static_positive(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,5],f32> {
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %float0 = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %int0, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[2,3],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,5],f32>
  return %0 : !torch.vtensor<[2,5],f32>
}

// -----

// Negative pads crop. Per side crop = max(-amount, 0), pad = max(amount, 0),
// and the sliced size is origDim - cropFront - cropBack. Every dim is run
// through this (the unpadded dim 0 with amount 0), extents are left dynamic,
// and a trailing cast restores the static 2x2 result. low = -1 / high = -2 ->
// cropFront = 1, cropBack = 2, size = 5 - 1 - 2 = 2.

// CHECK-LABEL:   func.func @constant_pad_nd$static_negative(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,5],f32>
// CHECK:           %[[IN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,5],f32> -> tensor<2x5xf32>
// CHECK:           %[[FILL:.*]] = arith.truncf %{{.*}} : f64 to f32
// dim 0 (not padded): amount 0 on both sides
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[L0:.*]] = arith.constant 0 : index
// CHECK:           %[[H0:.*]] = arith.constant 0 : index
// CHECK:           %[[NEGL0:.*]] = arith.subi %[[C0]], %[[L0]] : index
// CHECK:           %[[CROPF0:.*]] = arith.maxsi %[[NEGL0]], %[[C0]] : index
// CHECK:           %[[PADF0:.*]] = arith.maxsi %[[L0]], %[[C0]] : index
// CHECK:           %[[NEGH0:.*]] = arith.subi %[[C0]], %[[H0]] : index
// CHECK:           %[[CROPB0:.*]] = arith.maxsi %[[NEGH0]], %[[C0]] : index
// CHECK:           %[[PADB0:.*]] = arith.maxsi %[[H0]], %[[C0]] : index
// CHECK:           %[[DIM0:.*]] = arith.constant 2 : index
// CHECK:           %[[T0:.*]] = arith.subi %[[DIM0]], %[[CROPF0]] : index
// CHECK:           %[[SZ0:.*]] = arith.subi %[[T0]], %[[CROPB0]] : index
// dim 1: low = -1 (crop 1 front), high = -2 (crop 2 back)
// CHECK:           %[[L1:.*]] = arith.constant -1 : index
// CHECK:           %[[H1:.*]] = arith.constant -2 : index
// CHECK:           %[[NEGL1:.*]] = arith.subi %[[C0]], %[[L1]] : index
// CHECK:           %[[CROPF1:.*]] = arith.maxsi %[[NEGL1]], %[[C0]] : index
// CHECK:           %[[PADF1:.*]] = arith.maxsi %[[L1]], %[[C0]] : index
// CHECK:           %[[NEGH1:.*]] = arith.subi %[[C0]], %[[H1]] : index
// CHECK:           %[[CROPB1:.*]] = arith.maxsi %[[NEGH1]], %[[C0]] : index
// CHECK:           %[[PADB1:.*]] = arith.maxsi %[[H1]], %[[C0]] : index
// CHECK:           %[[DIM1:.*]] = arith.constant 5 : index
// CHECK:           %[[T1:.*]] = arith.subi %[[DIM1]], %[[CROPF1]] : index
// CHECK:           %[[SZ1:.*]] = arith.subi %[[T1]], %[[CROPB1]] : index
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[IN]][%[[CROPF0]], %[[CROPF1]]] [%[[SZ0]], %[[SZ1]]] [1, 1] : tensor<2x5xf32> to tensor<?x?xf32>
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SLICE]] low[%[[PADF0]], %[[PADF1]]] high[%[[PADB0]], %[[PADB1]]]
// CHECK:             tensor.yield %[[FILL]] : f32
// CHECK:           } : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[PAD]] : tensor<?x?xf32> to tensor<2x2xf32>
// CHECK:           torch_c.from_builtin_tensor %[[CAST]] : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>

// Canonicalization folds all the crop/pad arithmetic and the zero pad away;
// only a static extract_slice remains.
// CANON-LABEL:   func.func @constant_pad_nd$static_negative(
// CANON-NOT:       arith.subi
// CANON-NOT:       arith.maxsi
// CANON:           tensor.extract_slice %{{.*}}[0, 1] [2, 2] [1, 1] : tensor<2x5xf32> to tensor<2x2xf32>
// CANON-NOT:       tensor.pad
// CANON-NOT:       tensor.cast
func.func @constant_pad_nd$static_negative(%arg0: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,2],f32> {
  %intm1 = torch.constant.int -1
  %intm2 = torch.constant.int -2
  %float0 = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %intm1, %intm2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[2,5],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,2],f32>
  return %0 : !torch.vtensor<[2,2],f32>
}

// -----

// A negative low and positive high on the same dim exercise crop and pad
// together: low = -2 -> cropFront = 2 (size 5 - 2 = 3), high = 3 -> padBack = 3
// (3 + 3 = 6). cropFront feeds the slice offset, padBack feeds tensor.pad's
// high; result is cast back to 2x6.

// CHECK-LABEL:   func.func @constant_pad_nd$static_crop_and_pad(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,5],f32>
// CHECK:           %[[IN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,5],f32> -> tensor<2x5xf32>
// CHECK:           %[[FILL:.*]] = arith.truncf %{{.*}} : f64 to f32
// dim 0 (not padded): amount 0 on both sides
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[L0:.*]] = arith.constant 0 : index
// CHECK:           %[[H0:.*]] = arith.constant 0 : index
// CHECK:           %[[NEGL0:.*]] = arith.subi %[[C0]], %[[L0]] : index
// CHECK:           %[[CROPF0:.*]] = arith.maxsi %[[NEGL0]], %[[C0]] : index
// CHECK:           %[[PADF0:.*]] = arith.maxsi %[[L0]], %[[C0]] : index
// CHECK:           %[[NEGH0:.*]] = arith.subi %[[C0]], %[[H0]] : index
// CHECK:           %[[CROPB0:.*]] = arith.maxsi %[[NEGH0]], %[[C0]] : index
// CHECK:           %[[PADB0:.*]] = arith.maxsi %[[H0]], %[[C0]] : index
// CHECK:           %[[DIM0:.*]] = arith.constant 2 : index
// CHECK:           %[[T0:.*]] = arith.subi %[[DIM0]], %[[CROPF0]] : index
// CHECK:           %[[SZ0:.*]] = arith.subi %[[T0]], %[[CROPB0]] : index
// dim 1: low = -2 (crop 2 front), high = 3 (pad 3 back)
// CHECK:           %[[L1:.*]] = arith.constant -2 : index
// CHECK:           %[[H1:.*]] = arith.constant 3 : index
// CHECK:           %[[NEGL1:.*]] = arith.subi %[[C0]], %[[L1]] : index
// CHECK:           %[[CROPF1:.*]] = arith.maxsi %[[NEGL1]], %[[C0]] : index
// CHECK:           %[[PADF1:.*]] = arith.maxsi %[[L1]], %[[C0]] : index
// CHECK:           %[[NEGH1:.*]] = arith.subi %[[C0]], %[[H1]] : index
// CHECK:           %[[CROPB1:.*]] = arith.maxsi %[[NEGH1]], %[[C0]] : index
// CHECK:           %[[PADB1:.*]] = arith.maxsi %[[H1]], %[[C0]] : index
// CHECK:           %[[DIM1:.*]] = arith.constant 5 : index
// CHECK:           %[[T1:.*]] = arith.subi %[[DIM1]], %[[CROPF1]] : index
// CHECK:           %[[SZ1:.*]] = arith.subi %[[T1]], %[[CROPB1]] : index
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[IN]][%[[CROPF0]], %[[CROPF1]]] [%[[SZ0]], %[[SZ1]]] [1, 1] : tensor<2x5xf32> to tensor<?x?xf32>
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SLICE]] low[%[[PADF0]], %[[PADF1]]] high[%[[PADB0]], %[[PADB1]]]
// CHECK:             tensor.yield %[[FILL]] : f32
// CHECK:           } : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[PAD]] : tensor<?x?xf32> to tensor<2x6xf32>
// CHECK:           torch_c.from_builtin_tensor %[[CAST]] : tensor<2x6xf32> -> !torch.vtensor<[2,6],f32>

// Canonicalization folds the arithmetic into a static slice and a
// static-shaped pad.
// CANON-LABEL:   func.func @constant_pad_nd$static_crop_and_pad(
// CANON-NOT:       arith.subi
// CANON-NOT:       arith.maxsi
// CANON:           %[[SLICE:.*]] = tensor.extract_slice %{{.*}}[0, 2] [2, 3] [1, 1] : tensor<2x5xf32> to tensor<2x3xf32>
// CANON:           tensor.pad %[[SLICE]]
// CANON:         } : tensor<2x3xf32> to tensor<2x6xf32>
// CANON-NOT:       tensor.cast
func.func @constant_pad_nd$static_crop_and_pad(%arg0: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[2,6],f32> {
  %int3 = torch.constant.int 3
  %intm2 = torch.constant.int -2
  %float0 = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %intm2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[2,5],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,6],f32>
  return %0 : !torch.vtensor<[2,6],f32>
}

// -----

// A runtime pad amount is materialized and index-cast to %[[N]], then fed
// through the same crop = max(-amount, 0) / pad = max(amount, 0) split. The
// padded dim is dynamic so a tensor.dim feeds the sliced size.

// CHECK-LABEL:   func.func @constant_pad_nd$dynamic(
// CHECK-SAME:        %[[ARG0:.*]]: !torch.vtensor<[2,?],f32>,
// CHECK-SAME:        %[[ARG1:.*]]: !torch.int
// CHECK:           %[[IN:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,?],f32> -> tensor<2x?xf32>
// CHECK:           %[[N_I64:.*]] = torch_c.to_i64 %[[ARG1]]
// CHECK:           %[[N:.*]] = arith.index_cast %[[N_I64]] : i64 to index
// CHECK:           %[[FILL:.*]] = arith.truncf %{{.*}} : f64 to f32
// dim 0 (not padded): amount 0 on both sides
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[L0:.*]] = arith.constant 0 : index
// CHECK:           %[[H0:.*]] = arith.constant 0 : index
// CHECK:           %[[NEGL0:.*]] = arith.subi %[[C0]], %[[L0]] : index
// CHECK:           %[[CROPF0:.*]] = arith.maxsi %[[NEGL0]], %[[C0]] : index
// CHECK:           %[[PADF0:.*]] = arith.maxsi %[[L0]], %[[C0]] : index
// CHECK:           %[[NEGH0:.*]] = arith.subi %[[C0]], %[[H0]] : index
// CHECK:           %[[CROPB0:.*]] = arith.maxsi %[[NEGH0]], %[[C0]] : index
// CHECK:           %[[PADB0:.*]] = arith.maxsi %[[H0]], %[[C0]] : index
// CHECK:           %[[DIM0:.*]] = arith.constant 2 : index
// CHECK:           %[[T0:.*]] = arith.subi %[[DIM0]], %[[CROPF0]] : index
// CHECK:           %[[SZ0:.*]] = arith.subi %[[T0]], %[[CROPB0]] : index
// dim 1: low = %[[N]] (runtime), high = 0
// CHECK:           %[[H1:.*]] = arith.constant 0 : index
// CHECK:           %[[NEGL1:.*]] = arith.subi %[[C0]], %[[N]] : index
// CHECK:           %[[CROPF1:.*]] = arith.maxsi %[[NEGL1]], %[[C0]] : index
// CHECK:           %[[PADF1:.*]] = arith.maxsi %[[N]], %[[C0]] : index
// CHECK:           %[[NEGH1:.*]] = arith.subi %[[C0]], %[[H1]] : index
// CHECK:           %[[CROPB1:.*]] = arith.maxsi %[[NEGH1]], %[[C0]] : index
// CHECK:           %[[PADB1:.*]] = arith.maxsi %[[H1]], %[[C0]] : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<2x?xf32>
// CHECK:           %[[T1:.*]] = arith.subi %[[DIM1]], %[[CROPF1]] : index
// CHECK:           %[[SZ1:.*]] = arith.subi %[[T1]], %[[CROPB1]] : index
// CHECK:           %[[SLICE:.*]] = tensor.extract_slice %[[IN]][%[[CROPF0]], %[[CROPF1]]] [%[[SZ0]], %[[SZ1]]] [1, 1] : tensor<2x?xf32> to tensor<?x?xf32>
// CHECK:           %[[PAD:.*]] = tensor.pad %[[SLICE]] low[%[[PADF0]], %[[PADF1]]] high[%[[PADB0]], %[[PADB1]]]
// CHECK:             tensor.yield %[[FILL]] : f32
// CHECK:           } : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           %[[CAST:.*]] = tensor.cast %[[PAD]] : tensor<?x?xf32> to tensor<2x?xf32>
// CHECK:           torch_c.from_builtin_tensor %[[CAST]] : tensor<2x?xf32> -> !torch.vtensor<[2,?],f32>

// Canonicalization removes the unpadded dim 0's arithmetic entirely (constant
// zeros feed the slice and pad directly); only the runtime split for dim 1
// survives.
// CANON-LABEL:   func.func @constant_pad_nd$dynamic(
// CANON:           %[[C0:.*]] = arith.constant 0 : index
// CANON:           %[[N:.*]] = arith.index_cast %{{.*}} : i64 to index
// CANON:           %[[NEGN:.*]] = arith.subi %[[C0]], %[[N]] : index
// CANON:           %[[CROPF:.*]] = arith.maxsi %[[NEGN]], %[[C0]] : index
// CANON:           %[[PADF:.*]] = arith.maxsi %[[N]], %[[C0]] : index
// CANON:           %[[DIM:.*]] = tensor.dim
// CANON:           %[[SZ:.*]] = arith.subi %[[DIM]], %[[CROPF]] : index
// CANON-NOT:       arith.subi
// CANON-NOT:       arith.maxsi
// CANON:           tensor.extract_slice %{{.*}}[0, %[[CROPF]]] [2, %[[SZ]]] [1, 1] : tensor<2x?xf32> to tensor<2x?xf32>
// CANON:           tensor.pad %{{.*}} low[%[[C0]], %[[PADF]]] high[%[[C0]], %[[C0]]]
// CANON:         } : tensor<2x?xf32> to tensor<2x?xf32>
func.func @constant_pad_nd$dynamic(%arg0: !torch.vtensor<[2,?],f32>, %n: !torch.int) -> !torch.vtensor<[2,?],f32> {
  %int0 = torch.constant.int 0
  %float0 = torch.constant.float 0.000000e+00
  %pad = torch.prim.ListConstruct %n, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.constant_pad_nd %arg0, %pad, %float0 : !torch.vtensor<[2,?],f32>, !torch.list<int>, !torch.float -> !torch.vtensor<[2,?],f32>
  return %0 : !torch.vtensor<[2,?],f32>
}
