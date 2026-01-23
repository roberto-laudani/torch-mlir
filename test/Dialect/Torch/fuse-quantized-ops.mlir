// RUN: torch-mlir-opt %s --split-input-file --torch-fuse-quantized-ops | FileCheck %s

// CHECK-LABEL: @mm
func.func @mm(%arg0: !torch.vtensor<[4, 4],si8>, %arg1: !torch.vtensor<[4, 4],si8>) -> !torch.vtensor<[4, 4],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[4, 4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4, 4],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[4, 4],!torch.qint8> -> !torch.vtensor<[4, 4],f32>
  %16 = torch.aten.mm %7, %13 : !torch.vtensor<[4, 4],f32>, !torch.vtensor<[4, 4],f32> -> !torch.vtensor<[4, 4],f32>

  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[QUARTER:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF:.+]], %[[ONE]] : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  // CHECK-DAG: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF:.+]], %[[ZERO]] : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  // CHECK-DAG: %[[MM:.+]] = torch.aten.mm %[[QLHS]], %[[QRHS]] : !torch.vtensor<[4,4],!torch.qint8>, !torch.vtensor<[4,4],!torch.qint8> -> !torch.vtensor<[4,4],!torch.qint32>
  // CHECK-DAG: %[[INT:.+]] = torch.aten.int_repr %[[MM]] : !torch.vtensor<[4,4],!torch.qint32> -> !torch.vtensor<[4,4],si32>
  // CHECK-DAG: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[INT]], %[[QUARTER]], %[[ZERO]] : !torch.vtensor<[4,4],si32>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint32>
  // CHECK: %[[OUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[4,4],!torch.qint32> -> !torch.vtensor<[4,4],f32>
  return %16 : !torch.vtensor<[4, 4],f32>
}

// -----

// CHECK-LABEL: @matmul_commuting
func.func @matmul_commuting(%arg0: !torch.vtensor<[2,128,32,32],si8>) -> !torch.vtensor<[1,1024,1024],f32> {
  %float5.000000e-01 = torch.constant.float 5.000000e-01
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int-128 = torch.constant.int -128
  %int2 = torch.constant.int 2
  %int128 = torch.constant.int 128
  %int1024 = torch.constant.int 1024
  %int12 = torch.constant.int 12
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %float5.000000e-01, %int-128 : !torch.vtensor<[2,128,32,32],si8>, !torch.float, !torch.int -> !torch.vtensor<[2,128,32,32],!torch.qint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[2,128,32,32],!torch.qint8> -> !torch.vtensor<[2,128,32,32],f32>
  %2 = torch.aten.slice.Tensor %1, %int0, %int0, %int1, %int1 : !torch.vtensor<[2,128,32,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,32,32],f32>
  %3 = torch.aten.slice.Tensor %1, %int0, %int1, %int2, %int1 : !torch.vtensor<[2,128,32,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,32,32],f32>
  %4 = torch.prim.ListConstruct %int1, %int128, %int1024 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %5 = torch.aten.reshape %2, %4 : !torch.vtensor<[1,128,32,32],f32>, !torch.list<int> -> !torch.vtensor<[1,128,1024],f32>
  %6 = torch.aten.reshape %3, %4 : !torch.vtensor<[1,128,32,32],f32>, !torch.list<int> -> !torch.vtensor<[1,128,1024],f32>
  %7 = torch.aten.transpose.int %5, %int1, %int2 : !torch.vtensor<[1,128,1024],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1024,128],f32>
  %8 = torch.aten.quantize_per_tensor %7, %float5.000000e-01, %int0, %int12 : !torch.vtensor<[1,1024,128],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[1,1024,128],!torch.qint8>
  %9 = torch.aten.int_repr %8 : !torch.vtensor<[1,1024,128],!torch.qint8> -> !torch.vtensor<[1,1024,128],si8>
  %10 = torch.aten._make_per_tensor_quantized_tensor %9, %float5.000000e-01, %int0 : !torch.vtensor<[1,1024,128],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,1024,128],!torch.qint8>
  %11 = torch.aten.dequantize.self %10 : !torch.vtensor<[1,1024,128],!torch.qint8> -> !torch.vtensor<[1,1024,128],f32>
  %12 = torch.aten.matmul %11, %6 : !torch.vtensor<[1,1024,128],f32>, !torch.vtensor<[1,128,1024],f32> -> !torch.vtensor<[1,1024,1024],f32>

  // CHECK-DAG: %[[QUARTER:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[I0:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[I1:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[IN128:.+]] = torch.constant.int -128
  // CHECK-DAG: %[[I2:.+]] = torch.constant.int 2
  // CHECK-DAG: %[[I128:.+]] = torch.constant.int 128
  // CHECK-DAG: %[[I1024:.+]] = torch.constant.int 1024
  // CHECK-DAG: %[[I12:.+]] = torch.constant.int 12
  // CHECK-DAG: %[[MPTQT0:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF]], %[[IN128]] : !torch.vtensor<[2,128,32,32],si8>, !torch.float, !torch.int -> !torch.vtensor<[2,128,32,32],!torch.qint8>
  // CHECK-DAG: %[[DQ0:.+]] = torch.aten.dequantize.self %[[MPTQT0]] : !torch.vtensor<[2,128,32,32],!torch.qint8> -> !torch.vtensor<[2,128,32,32],f32>
  // CHECK-DAG: %[[SLICE0:.+]] = torch.aten.slice.Tensor %[[DQ0]], %[[I0]], %[[I0]], %[[I1]], %[[I1]] : !torch.vtensor<[2,128,32,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,32,32],f32>
  // CHECK-DAG: %[[LIST:.+]] = torch.prim.ListConstruct %[[I1]], %[[I128]], %[[I1024]] : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[RESHAPE0:.+]] = torch.aten.reshape %[[SLICE0]], %[[LIST]] : !torch.vtensor<[1,128,32,32],f32>, !torch.list<int> -> !torch.vtensor<[1,128,1024],f32>
  // CHECK-DAG: %[[TR0:.+]] = torch.aten.transpose.int %[[RESHAPE0]], %[[I1]], %[[I2]] : !torch.vtensor<[1,128,1024],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,1024,128],f32>
  // CHECK-DAG: %[[Q0:.+]] = torch.aten.quantize_per_tensor %[[TR0]], %[[HALF]], %[[I0]], %[[I12]] : !torch.vtensor<[1,1024,128],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[1,1024,128],!torch.qint8>
  // CHECK-DAG: %[[IR0:.+]] = torch.aten.int_repr %[[Q0]] : !torch.vtensor<[1,1024,128],!torch.qint8> -> !torch.vtensor<[1,1024,128],si8>
  // CHECK-DAG: %[[MPTQT1:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[IR0]], %[[HALF]], %[[I0]] : !torch.vtensor<[1,1024,128],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,1024,128],!torch.qint8>
  // CHECK-DAG: %[[SLICE1:.+]] = torch.aten.slice.Tensor %arg0, %[[I0]], %[[I1]], %[[I2]], %[[I1]] : !torch.vtensor<[2,128,32,32],si8>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,32,32],si8>
  // CHECK-DAG: %[[RESHAPE1:.+]] = torch.aten.reshape %[[SLICE1]], %[[LIST]] : !torch.vtensor<[1,128,32,32],si8>, !torch.list<int> -> !torch.vtensor<[1,128,1024],si8>
  // CHECK-DAG: %[[MPTQT2:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[RESHAPE1]], %[[HALF]], %[[IN128]] : !torch.vtensor<[1,128,1024],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,128,1024],!torch.qint8>
  // CHECK-DAG: %[[MATMUL:.+]] = torch.aten.matmul %[[MPTQT1]], %[[MPTQT2]] : !torch.vtensor<[1,1024,128],!torch.qint8>, !torch.vtensor<[1,128,1024],!torch.qint8> -> !torch.vtensor<[1,1024,1024],!torch.qint32>
  // CHECK-DAG: %[[IR1:.+]] = torch.aten.int_repr %[[MATMUL]] : !torch.vtensor<[1,1024,1024],!torch.qint32> -> !torch.vtensor<[1,1024,1024],si32>
  // CHECK-DAG: %[[MPTQT3:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[IR1]], %[[QUARTER]], %[[I0]] : !torch.vtensor<[1,1024,1024],si32>, !torch.float, !torch.int -> !torch.vtensor<[1,1024,1024],!torch.qint32>
  // CHECK-DAG: %[[DQ1:.+]] = torch.aten.dequantize.tensor %[[MPTQT3]] : !torch.vtensor<[1,1024,1024],!torch.qint32> -> !torch.vtensor<[1,1024,1024],f32>
  return %12 : !torch.vtensor<[1,1024,1024],f32>
}

// -----

// CHECK-LABEL: func.func @mm_pad_commute
func.func @mm_pad_commute(%arg0: !torch.vtensor<[8,8],si8>, %arg1: !torch.vtensor<[11,4],si8>) -> !torch.vtensor<[9,4],f32> {
  // CHECK-DAG: %[[cstQuart:.*]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[padVal:.*]] = torch.vtensor.literal(dense<8.000000e+00> : tensor<f64>) : !torch.vtensor<[],f64>
  // CHECK-DAG: %[[qMax:.*]] = torch.constant.float 1.270000e+02
  // CHECK-DAG: %[[qMin:.*]] = torch.constant.float -1.280000e+02
  // CHECK-DAG: %[[str:.*]] = torch.constant.str "constant"
  // CHECK-DAG: %[[cstHalf:.*]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[int0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[int1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[int2:.*]] = torch.constant.int 2
  // CHECK: %[[PadList:.*]] = torch.prim.ListConstruct %[[int1]], %[[int2]], %[[int0]], %[[int1]] : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // CHECK: %[[Clamp:.*]] = torch.aten.clamp %[[padVal]], %[[qMin]], %[[qMax]] : !torch.vtensor<[],f64>, !torch.float, !torch.float -> !torch.vtensor<[],f64>
  // CHECK: %[[Item:.*]] = torch.aten.item %[[Clamp]] : !torch.vtensor<[],f64> -> !torch.float
  // CHECK: %[[NewPad:.*]] = torch.aten.pad %arg0, %[[PadList]], %[[str]], %[[Item]] : !torch.vtensor<[8,8],si8>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[9,11],si8>
  // CHECK: %[[NewMPTQT:.*]] = torch.aten._make_per_tensor_quantized_tensor %[[NewPad]], %[[cstHalf]], %[[int1]] : !torch.vtensor<[9,11],si8>, !torch.float, !torch.int -> !torch.vtensor<[9,11],!torch.qint8>
  // CHECK: %[[OtherMPTQT:.*]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[cstHalf]], %[[int0]] : !torch.vtensor<[11,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[11,4],!torch.qint8>
  // CHECK: %[[MM:.*]] = torch.aten.mm %[[NewMPTQT]], %[[OtherMPTQT]] : !torch.vtensor<[9,11],!torch.qint8>, !torch.vtensor<[11,4],!torch.qint8> -> !torch.vtensor<[9,4],!torch.qint32>
  // CHECK: %[[IR:.*]] = torch.aten.int_repr %[[MM]] : !torch.vtensor<[9,4],!torch.qint32> -> !torch.vtensor<[9,4],si32>
  // CHECK: %[[QOUT:.*]] = torch.aten._make_per_tensor_quantized_tensor %[[IR]], %[[cstQuart]], %[[int0]] : !torch.vtensor<[9,4],si32>, !torch.float, !torch.int -> !torch.vtensor<[9,4],!torch.qint32>
  // CHECK: %[[OUT:.*]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[9,4],!torch.qint32> -> !torch.vtensor<[9,4],f32>
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %two = torch.constant.int 2
  %floatpad = torch.constant.float 3.5
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[8,8],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[8,8],!torch.qint8> -> !torch.vtensor<[8,8],f32>
  %list = torch.prim.ListConstruct %one, %two, %zero, %one : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %str = torch.constant.str "constant"
  %pad = torch.aten.pad %7, %list, %str, %floatpad : !torch.vtensor<[8,8],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.vtensor<[9,11],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[11,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[11,4],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[11,4],!torch.qint8> -> !torch.vtensor<[11,4],f32>
  %16 = torch.aten.mm %pad, %13 : !torch.vtensor<[9,11],f32>, !torch.vtensor<[11,4],f32> -> !torch.vtensor<[9,4],f32>
  return %16 : !torch.vtensor<[9,4],f32>
}

// -----

// CHECK-LABEL: @convolution_bias
func.func @convolution_bias(%arg0: !torch.vtensor<[1,3,8,8],si8>, %arg1: !torch.vtensor<[3,3,2,2],si8>, %arg2 : !torch.vtensor<[3], f32>) -> !torch.vtensor<[1,3,7,7],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[3,3,2,2],!torch.qint8> -> !torch.vtensor<[3,3,2,2],f32>
  %14 = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  %15 = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %16 = torch.aten.convolution %7, %13, %arg2, %14, %15, %14, %false, %15, %one : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3,3,2,2],f32>, !torch.vtensor<[3],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],f32>

  // CHECK-DAG: %[[DTYPE:.+]] = torch.constant.int 14
  // CHECK-DAG: %[[SCALEO:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF]], %[[ONE]] : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK-DAG: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF]], %[[ZERO]] : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  // CHECK-DAG: %[[ONES:.+]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[ZEROS:.+]] = torch.prim.ListConstruct %[[ZERO]], %[[ZERO]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[QBIAS:.+]] = torch.aten.quantize_per_tensor %arg2, %[[SCALEO]], %[[ZERO]], %[[DTYPE]] : !torch.vtensor<[3],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[3],!torch.qint32>
  // CHECK-DAG: %[[INT:.+]] = torch.aten.int_repr %[[QBIAS]] : !torch.vtensor<[3],!torch.qint32> -> !torch.vtensor<[3],si32>
  // CHECK-DAG: %[[CONV:.+]] = torch.aten.convolution %[[QLHS]], %[[QRHS]], %[[INT]], %[[ONES]], %[[ZEROS]], %[[ONES]], %[[FALSE]], %[[ZEROS]], %[[ONE]] : !torch.vtensor<[1,3,8,8],!torch.qint8>, !torch.vtensor<[3,3,2,2],!torch.qint8>, !torch.vtensor<[3],si32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],si32>
  // CHECK-DAG: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[SCALEO]], %[[ZERO]] : !torch.vtensor<[1,3,7,7],si32>, !torch.float, !torch.int -> !torch.vtensor<[1,3,7,7],!torch.qint32>
  // CHECK-DAG: %[[FOUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[1,3,7,7],!torch.qint32> -> !torch.vtensor<[1,3,7,7],f32>
  return %16 : !torch.vtensor<[1,3,7,7],f32>
}


// -----

// CHECK-LABEL: @convolution_nobias
func.func @convolution_nobias(%arg0: !torch.vtensor<[1,3,8,8],si8>, %arg1: !torch.vtensor<[3,3,2,2],si8>) -> !torch.vtensor<[1,3,7,7],f32> {
  %scale = torch.constant.float 0.5
  %false = torch.constant.bool false
  %zero = torch.constant.int 0
  %one = torch.constant.int 1
  %zp = torch.constant.int -128
  %none = torch.constant.none
  %6 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %one : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  %7 = torch.aten.dequantize.tensor %6 : !torch.vtensor<[1,3,8,8],!torch.qint8> -> !torch.vtensor<[1,3,8,8],f32>
  %12 = torch.aten._make_per_tensor_quantized_tensor %arg1, %scale, %zero : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  %13 = torch.aten.dequantize.tensor %12 : !torch.vtensor<[3,3,2,2],!torch.qint8> -> !torch.vtensor<[3,3,2,2],f32>
  %14 = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
  %15 = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
  %16 = torch.aten.convolution %7, %13, %none, %14, %15, %14, %false, %15, %one : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[3,3,2,2],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],f32>

  // CHECK-DAG: %[[SCALEO:.+]] = torch.constant.float 2.500000e-01
  // CHECK-DAG: %[[HALF:.+]] = torch.constant.float 5.000000e-01
  // CHECK-DAG: %[[FALSE:.+]] = torch.constant.bool false
  // CHECK-DAG: %[[ZERO:.+]] = torch.constant.int 0
  // CHECK-DAG: %[[ONE:.+]] = torch.constant.int 1
  // CHECK-DAG: %[[NONE:.+]] = torch.constant.none
  // CHECK-DAG: %[[QLHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg0, %[[HALF]], %[[ONE]] : !torch.vtensor<[1,3,8,8],si8>, !torch.float, !torch.int -> !torch.vtensor<[1,3,8,8],!torch.qint8>
  // CHECK-DAG: %[[QRHS:.+]] = torch.aten._make_per_tensor_quantized_tensor %arg1, %[[HALF]], %[[ZERO]] : !torch.vtensor<[3,3,2,2],si8>, !torch.float, !torch.int -> !torch.vtensor<[3,3,2,2],!torch.qint8>
  // CHECK-DAG: %[[ONES:.+]] = torch.prim.ListConstruct %[[ONE]], %[[ONE]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[ZEROS:.+]] = torch.prim.ListConstruct %[[ZERO]], %[[ZERO]] : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-DAG: %[[CONV:.+]] = torch.aten.convolution %[[QLHS]], %[[QRHS]], %[[NONE]], %[[ONES]], %[[ZEROS]], %[[ONES]], %[[FALSE]], %[[ZEROS]], %[[ONE]] : !torch.vtensor<[1,3,8,8],!torch.qint8>, !torch.vtensor<[3,3,2,2],!torch.qint8>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,3,7,7],si32>
  // CHECK-DAG: %[[QOUT:.+]] = torch.aten._make_per_tensor_quantized_tensor %[[CONV]], %[[SCALEO]], %[[ZERO]] : !torch.vtensor<[1,3,7,7],si32>, !torch.float, !torch.int -> !torch.vtensor<[1,3,7,7],!torch.qint32>
  // CHECK: %[[FOUT:.+]] = torch.aten.dequantize.tensor %[[QOUT]] : !torch.vtensor<[1,3,7,7],!torch.qint32> -> !torch.vtensor<[1,3,7,7],f32>
  return %16 : !torch.vtensor<[1,3,7,7],f32>
}

// -----

// Test fusion with per-channel quantized weight (common in ONNX models)
// This should lower to torch.quantized_matmul which handles mixed quantization properly.
// CHECK-LABEL: @mm_per_channel_weight
func.func @mm_per_channel_weight(%arg0: !torch.vtensor<[4,4],si8>, %arg1: !torch.vtensor<[4,4],si8>, %arg2: !torch.vtensor<[4],f32>, %arg3: !torch.vtensor<[4],si8>) -> !torch.vtensor<[4,4],f32> {
  %scale = torch.constant.float 0.5
  %zero = torch.constant.int 0
  %axis = torch.constant.int 1
  // Per-tensor quantized input
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %scale, %zero : !torch.vtensor<[4,4],si8>, !torch.float, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[4,4],!torch.qint8> -> !torch.vtensor<[4,4],f32>
  // Per-channel quantized weight
  %2 = torch.aten._make_per_channel_quantized_tensor %arg1, %arg2, %arg3, %axis : !torch.vtensor<[4,4],si8>, !torch.vtensor<[4],f32>, !torch.vtensor<[4],si8>, !torch.int -> !torch.vtensor<[4,4],!torch.qint8>
  %3 = torch.aten.dequantize.self %2 : !torch.vtensor<[4,4],!torch.qint8> -> !torch.vtensor<[4,4],f32>
  %4 = torch.aten.mm %1, %3 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
  // CHECK: torch.quantized_matmul %arg0, %arg1, %{{.*}}, %arg3 : !torch.vtensor<[4,4],si8>, !torch.vtensor<[4,4],si8>, !torch.vtensor<[],si8>, !torch.vtensor<[4],si8> -> !torch.vtensor<[4,4],si32>
  // CHECK: torch.aten.to.dtype
  // CHECK: torch.aten.mul.Scalar %{{.*}}, %{{.*}} : !torch.vtensor<[4,4],f32>, !torch.float
  // CHECK: torch.aten.mul.Tensor %{{.*}}, %arg2 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32>
  return %4 : !torch.vtensor<[4,4],f32>
}

// -----

// Test matmul with quint8 input (unsigned) and qint8 per-channel weight (signed).
// This is a common pattern in ONNX quantized models (e.g., Qwen).
// CHECK-LABEL: @matmul_quint8_input_qint8_per_channel_weight
func.func @matmul_quint8_input_qint8_per_channel_weight(
    %arg0: !torch.vtensor<[1,4,8],ui8>,
    %arg1: !torch.vtensor<[8,16],si8>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[16],si8>) -> !torch.vtensor<[1,4,16],f32> {
  %input_scale = torch.constant.float 0.1
  %input_zp = torch.constant.int 128
  %axis = torch.constant.int 1
  // Per-tensor quantized unsigned input
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %input_scale, %input_zp : !torch.vtensor<[1,4,8],ui8>, !torch.float, !torch.int -> !torch.vtensor<[1,4,8],!torch.quint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[1,4,8],!torch.quint8> -> !torch.vtensor<[1,4,8],f32>
  // Per-channel quantized signed weight
  %2 = torch.aten._make_per_channel_quantized_tensor %arg1, %arg2, %arg3, %axis : !torch.vtensor<[8,16],si8>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],si8>, !torch.int -> !torch.vtensor<[8,16],!torch.qint8>
  %3 = torch.aten.dequantize.self %2 : !torch.vtensor<[8,16],!torch.qint8> -> !torch.vtensor<[8,16],f32>
  %4 = torch.aten.matmul %1, %3 : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[1,4,16],f32>
  // CHECK: torch.quantized_matmul %arg0, %arg1, %{{.*}}, %arg3 : !torch.vtensor<[1,4,8],ui8>, !torch.vtensor<[8,16],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[16],si8> -> !torch.vtensor<[1,4,16],si32>
  // CHECK: torch.aten.to.dtype
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.mul.Tensor
  return %4 : !torch.vtensor<[1,4,16],f32>
}

// -----

// Test matmul with both operands being per-channel quantized.
// CHECK-LABEL: @matmul_both_per_channel
func.func @matmul_both_per_channel(
    %arg0: !torch.vtensor<[4,8],si8>,
    %arg1: !torch.vtensor<[8,16],si8>,
    %arg2: !torch.vtensor<[8],f32>,
    %arg3: !torch.vtensor<[8],si8>,
    %arg4: !torch.vtensor<[16],f32>,
    %arg5: !torch.vtensor<[16],si8>) -> !torch.vtensor<[4,16],f32> {
  %axis0 = torch.constant.int 1
  %axis1 = torch.constant.int 1
  // Per-channel quantized input
  %0 = torch.aten._make_per_channel_quantized_tensor %arg0, %arg2, %arg3, %axis0 : !torch.vtensor<[4,8],si8>, !torch.vtensor<[8],f32>, !torch.vtensor<[8],si8>, !torch.int -> !torch.vtensor<[4,8],!torch.qint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[4,8],!torch.qint8> -> !torch.vtensor<[4,8],f32>
  // Per-channel quantized weight
  %2 = torch.aten._make_per_channel_quantized_tensor %arg1, %arg4, %arg5, %axis1 : !torch.vtensor<[8,16],si8>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],si8>, !torch.int -> !torch.vtensor<[8,16],!torch.qint8>
  %3 = torch.aten.dequantize.self %2 : !torch.vtensor<[8,16],!torch.qint8> -> !torch.vtensor<[8,16],f32>
  %4 = torch.aten.matmul %1, %3 : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  // CHECK: torch.quantized_matmul %arg0, %arg1, %arg3, %arg5 : !torch.vtensor<[4,8],si8>, !torch.vtensor<[8,16],si8>, !torch.vtensor<[8],si8>, !torch.vtensor<[16],si8> -> !torch.vtensor<[4,16],si32>
  // CHECK: torch.aten.to.dtype
  // CHECK: torch.aten.mul.Tensor {{.*}}, %arg2
  // CHECK: torch.aten.mul.Tensor {{.*}}, %arg4
  return %4 : !torch.vtensor<[4,16],f32>
}

// -----

// Test mm (2D matmul) with quint8/qint8 mixed types.
// CHECK-LABEL: @mm_quint8_qint8_mixed
func.func @mm_quint8_qint8_mixed(
    %arg0: !torch.vtensor<[4,8],ui8>,
    %arg1: !torch.vtensor<[8,16],si8>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[16],si8>) -> !torch.vtensor<[4,16],f32> {
  %input_scale = torch.constant.float 0.05
  %input_zp = torch.constant.int 127
  %axis = torch.constant.int 1
  // Per-tensor unsigned input
  %0 = torch.aten._make_per_tensor_quantized_tensor %arg0, %input_scale, %input_zp : !torch.vtensor<[4,8],ui8>, !torch.float, !torch.int -> !torch.vtensor<[4,8],!torch.quint8>
  %1 = torch.aten.dequantize.self %0 : !torch.vtensor<[4,8],!torch.quint8> -> !torch.vtensor<[4,8],f32>
  // Per-channel signed weight
  %2 = torch.aten._make_per_channel_quantized_tensor %arg1, %arg2, %arg3, %axis : !torch.vtensor<[8,16],si8>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],si8>, !torch.int -> !torch.vtensor<[8,16],!torch.qint8>
  %3 = torch.aten.dequantize.self %2 : !torch.vtensor<[8,16],!torch.qint8> -> !torch.vtensor<[8,16],f32>
  %4 = torch.aten.mm %1, %3 : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,16],f32> -> !torch.vtensor<[4,16],f32>
  // CHECK: torch.quantized_matmul %arg0, %arg1, %{{.*}}, %arg3 : !torch.vtensor<[4,8],ui8>, !torch.vtensor<[8,16],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[16],si8> -> !torch.vtensor<[4,16],si32>
  // CHECK: torch.aten.to.dtype
  // CHECK: torch.aten.mul.Scalar
  // CHECK: torch.aten.mul.Tensor
  return %4 : !torch.vtensor<[4,16],f32>
}
