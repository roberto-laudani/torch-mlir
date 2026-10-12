// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.softmax.int,torchao.dequantize_affine" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.softmax.int$cst_dim
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  // CHECK: torch.aten.softmax.int
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----

// CHECK-LABEL: func.func @torch.torchao.dequantize_affine
func.func @torch.torchao.dequantize_affine(%input: !torch.vtensor<[4,16],si8>, %scale: !torch.vtensor<[4,2],f32>) -> !torch.vtensor<[4,16],f32> {
  %int1 = torch.constant.int 1
  %int8 = torch.constant.int 8
  %dtype = torch.constant.int 1
  %out_dtype = torch.constant.int 6
  %none = torch.constant.none
  %block_size = torch.prim.ListConstruct %int1, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK: torch.torchao.dequantize_affine
  %ret = torch.torchao.dequantize_affine %input, %block_size, %scale, %none, %dtype, %none, %none, %out_dtype : !torch.vtensor<[4,16],si8>, !torch.list<int>, !torch.vtensor<[4,2],f32>, !torch.none, !torch.int, !torch.none, !torch.none, !torch.int -> !torch.vtensor<[4,16],f32>
  return %ret : !torch.vtensor<[4,16],f32>
}
