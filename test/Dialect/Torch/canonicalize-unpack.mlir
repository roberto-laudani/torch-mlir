// RUN: torch-mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @fold_unpack_bits8
// CHECK-SAME:  (%[[PACKED:.+]]: !torch.vtensor<[8,4,4],ui8>)
// CHECK-NOT:   torch.sub_byte_unpack
// CHECK:       return %[[PACKED]] : !torch.vtensor<[8,4,4],ui8>
func.func @fold_unpack_bits8(
    %arg0: !torch.vtensor<[8,4,4],ui8>) -> !torch.vtensor<[8,4,4],ui8> {
  %0 = torch.sub_byte_unpack %arg0 {bits = 8 : si64}
      : !torch.vtensor<[8,4,4],ui8> -> !torch.vtensor<[8,4,4],ui8>
  return %0 : !torch.vtensor<[8,4,4],ui8>
}

// -----

// CHECK-LABEL: func.func @no_fold_unpack_bits4
// CHECK-SAME:  (%[[PACKED:.+]]: !torch.vtensor<[8,2,4],ui8>)
// CHECK:       %[[UNPACKED:.+]] = torch.sub_byte_unpack %[[PACKED]]
// CHECK-SAME:    {bits = 4 : si64}
// CHECK-SAME:    : !torch.vtensor<[8,2,4],ui8> -> !torch.vtensor<[8,2,8],ui8>
// CHECK:       return %[[UNPACKED]]
func.func @no_fold_unpack_bits4(
    %arg0: !torch.vtensor<[8,2,4],ui8>) -> !torch.vtensor<[8,2,8],ui8> {
  %0 = torch.sub_byte_unpack %arg0 {bits = 4 : si64}
      : !torch.vtensor<[8,2,4],ui8> -> !torch.vtensor<[8,2,8],ui8>
  return %0 : !torch.vtensor<[8,2,8],ui8>
}
