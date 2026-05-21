//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

namespace {
/// Result struct for decomposed RMS normalization
struct RMSNormResult {
  Value output;    // normalized * scale
  Value meanSq;    // mean(x²) - intermediate
  Value invStdVar; // rsqrt(mean(x²) + eps) - intermediate
};

/// Build decomposed RMS normalization ops.
/// Returns the output and intermediates (meanSq, invStdVar) for optional
/// outputs. RMS norm formula: y = x * rsqrt(mean(x²) + eps) * scale
static RMSNormResult buildRMSNorm(OpBinder &binder,
                                  ConversionPatternRewriter &rewriter,
                                  Value input, Value scale, Value epsilon,
                                  int64_t axis, Type outputType) {
  Location loc = binder.getLoc();
  auto inputTy = cast<Torch::ValueTensorType>(input.getType());
  ArrayRef<int64_t> inputShape = inputTy.getSizes();
  int64_t rank = inputShape.size();

  // Build reduce dims list [axis, axis+1, ..., rank-1]
  auto context = binder.op->getContext();
  auto sizeListType = Torch::ListType::get(Torch::IntType::get(context));
  SmallVector<Value> reduceDimVals;
  for (int64_t i = axis; i < rank; i++) {
    reduceDimVals.push_back(Torch::ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(i)));
  }
  Value reduceDimList = Torch::PrimListConstructOp::create(
      rewriter, loc, sizeListType, reduceDimVals);

  // Compute reduced shape for meanSq/invStdVar (keepdim=true)
  SmallVector<int64_t> reducedShape(inputShape.begin(), inputShape.end());
  for (int64_t i = axis; i < rank; i++)
    reducedShape[i] = 1;
  auto reducedType =
      Torch::ValueTensorType::get(context, reducedShape, inputTy.getDtype());

  // x²
  Value xSquared =
      Torch::AtenMulTensorOp::create(rewriter, loc, inputTy, input, input);

  // mean(x²) with keepdim=true
  Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
  Value none = Torch::ConstantNoneOp::create(rewriter, loc);
  Value meanSq = Torch::AtenMeanDimOp::create(
      rewriter, loc, reducedType, xSquared, reduceDimList, cstTrue, none);

  // mean(x²) + eps
  Value one = Torch::ConstantIntOp::create(rewriter, loc,
                                           rewriter.getI64IntegerAttr(1));
  Value meanPlusEps = Torch::AtenAddScalarOp::create(rewriter, loc, reducedType,
                                                     meanSq, epsilon, one);

  // inv_std_var = rsqrt(mean(x²) + eps)
  Value invStdVar =
      Torch::AtenRsqrtOp::create(rewriter, loc, reducedType, meanPlusEps);

  // x * inv_std_var
  Value normalized =
      Torch::AtenMulTensorOp::create(rewriter, loc, inputTy, input, invStdVar);

  // normalized * scale
  Value output = Torch::AtenMulTensorOp::create(rewriter, loc, outputType,
                                                normalized, scale);

  return {output, meanSq, invStdVar};
}

/// If input is 1D, reshape it to (n, size/n). None or already-2D inputs pass
/// through unchanged. ONNX MatMulNBits allows scales and zero_points to be
/// flattened.
static Value promote1DTo2D(ConversionPatternRewriter &rewriter, Location loc,
                           Value input, int64_t n) {
  auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
  if (!inputTy || !inputTy.hasSizes() || inputTy.getSizes().size() != 1)
    return input;
  assert(inputTy.getSizes()[0] % n == 0 &&
         "promote1DTo2D: input size must be divisible by n");
  int64_t inner = inputTy.getSizes()[0] / n;
  Value shape = Torch::toIntListConstruct(rewriter, loc, {n, inner});
  return Torch::AtenReshapeOp::create(
      rewriter, loc,
      inputTy.getWithSizesAndDtype(ArrayRef<int64_t>{n, inner},
                                   inputTy.getOptionalDtype()),
      input, shape);
}

/// When key and value are None, the query operand carries a packed QKV tensor
/// of shape [batch, seq, (num_heads + 2*kv_num_heads) * head_size].
/// Split it into separate Q, K, V tensors and reassign the references.
/// Returns failure if the input is invalid.
static LogicalResult splitPackedQKV(OpBinder &binder,
                                    ConversionPatternRewriter &rewriter,
                                    int64_t numHeads, int64_t kvNumHeads,
                                    Value &query, Value &key, Value &value) {
  Location loc = binder.getLoc();
  MLIRContext *context = binder.op->getContext();

  auto packedType = cast<Torch::ValueTensorType>(query.getType());
  if (!packedType.hasSizes() || packedType.getSizes().size() != 3)
    return rewriter.notifyMatchFailure(binder.op,
                                       "Expected packed QKV input to be rank-3 "
                                       "[batch, seq, hidden]");

  if (packedType.getSizes()[2] == ShapedType::kDynamic)
    return rewriter.notifyMatchFailure(
        binder.op,
        "Expected packed QKV input to have statically known hidden size");

  if (!packedType.hasDtype())
    return rewriter.notifyMatchFailure(
        binder.op, "Expected packed QKV input to have a known dtype");

  int64_t packedHiddenSize = packedType.getSizes()[2];
  int64_t totalHeads = numHeads + 2 * kvNumHeads;
  if (packedHiddenSize % totalHeads != 0)
    return rewriter.notifyMatchFailure(binder.op,
                                       "Packed hidden size not divisible by "
                                       "(num_heads + 2*kv_num_heads)");

  int64_t headSize = packedHiddenSize / totalHeads;
  int64_t qHiddenSize = numHeads * headSize;
  int64_t kvHiddenSize = kvNumHeads * headSize;
  SmallVector<int64_t> packedDims{packedType.getSizes()};
  Type elemDtype = packedType.getOptionalDtype();

  Value cstQHidden = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(qHiddenSize));
  Value cstKVHidden = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(kvHiddenSize));
  Value splitSizesList = Torch::PrimListConstructOp::create(
      rewriter, loc, Torch::ListType::get(Torch::IntType::get(context)),
      SmallVector<Value>{cstQHidden, cstKVHidden, cstKVHidden});
  Value cstSplitDim = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(2));

  SmallVector<int64_t> splitElemDims{packedDims[0], packedDims[1],
                                     Torch::kUnknownSize};
  auto splitListType = Torch::ListType::get(
      Torch::ValueTensorType::get(context, splitElemDims, elemDtype));
  Value splitResult = Torch::AtenSplitWithSizesOp::create(
      rewriter, loc, splitListType, query, splitSizesList, cstSplitDim);

  SmallVector<int64_t> qSplitDims{packedDims[0], packedDims[1], qHiddenSize};
  SmallVector<int64_t> kvSplitDims{packedDims[0], packedDims[1], kvHiddenSize};
  auto qSplitType = Torch::ValueTensorType::get(context, qSplitDims, elemDtype);
  auto kvSplitType =
      Torch::ValueTensorType::get(context, kvSplitDims, elemDtype);

  auto unpack = Torch::PrimListUnpackOp::create(
      rewriter, loc, TypeRange{qSplitType, kvSplitType, kvSplitType},
      splitResult);
  query = unpack.getResult(0);
  key = unpack.getResult(1);
  value = unpack.getResult(2);
  return success();
}

static Value buildActiveCacheSlice(ConversionPatternRewriter &rewriter,
                                   Location loc, Value cache, int64_t seqDimIdx,
                                   Value seqDim, Value start, Value end,
                                   Value step) {
  auto cacheType = cast<Torch::ValueTensorType>(cache.getType());
  SmallVector<int64_t> activeSizes(cacheType.getSizes());
  activeSizes[seqDimIdx] = Torch::kUnknownSize;
  Type activeType =
      cacheType.getWithSizesAndDtype(activeSizes, cacheType.getOptionalDtype());
  return Torch::AtenSliceTensorOp::create(rewriter, loc, activeType, cache,
                                          seqDim, start, end, step);
}

} // namespace

void mlir::torch::onnx_c::populateComMicrosoftDomain(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "SimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType yType;
        Value x, scale;
        int64_t axis;
        float epsilon;
        if (binder.tensorOperandAtIndex(x, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.tensorResultTypeAtIndex(yType, 0) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.f32FloatAttr(epsilon, "epsilon", 0.00001f))
          return failure();

        auto xType = dyn_cast<Torch::ValueTensorType>(x.getType());
        if (!xType || !xType.hasSizes())
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected input to have sizes");

        ArrayRef<int64_t> xShape = xType.getSizes();
        int64_t rank = static_cast<int64_t>(xShape.size());
        int64_t axisNorm = Torch::toPositiveDim(axis, rank);

        // Build epsilon constant
        Value constEpsilon = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(epsilon));

        // Use decomposed RMS norm to get intermediates
        RMSNormResult rmsResult = buildRMSNorm(binder, rewriter, x, scale,
                                               constEpsilon, axisNorm, yType);

        // Build results
        SmallVector<Value> results;
        results.push_back(rmsResult.output);

        // Output 1: inv_std_var (only if requested)
        if (binder.op->getNumResults() > 1)
          results.push_back(rmsResult.invStdVar);

        rewriter.replaceOp(binder.op, results);
        return success();
      });
  patterns.onOp(
      "SkipSimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType yType;
        Value x, skip, scale;
        float epsilon;
        if (binder.tensorOperandAtIndex(x, 0) ||
            binder.tensorOperandAtIndex(skip, 1) ||
            binder.tensorOperandAtIndex(scale, 2) ||
            binder.tensorResultTypeAtIndex(yType, 0) ||
            binder.f32FloatAttr(epsilon, "epsilon", 0.00001f))
          return failure();

        // Optional bias at operand 3.
        Value bias;
        if (failed(binder.tensorOperandAtIndex(bias, 3)))
          bias = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());

        // First compute x + skip (+ bias if present) - this is what gets
        // normalized
        Value alpha = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getF64FloatAttr(1.0));
        Value inputSkipBiasSum = Torch::AtenAddTensorOp::create(
            rewriter, binder.getLoc(), yType, x, skip, alpha);
        if (!isa<Torch::NoneType>(bias.getType()))
          inputSkipBiasSum = Torch::AtenAddTensorOp::create(
              rewriter, binder.getLoc(), yType, inputSkipBiasSum, bias, alpha);

        auto inputType =
            dyn_cast<Torch::ValueTensorType>(inputSkipBiasSum.getType());
        if (!inputType || !inputType.hasSizes())
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected input to have sizes");

        ArrayRef<int64_t> inputShape = inputType.getSizes();
        int64_t rank = static_cast<int64_t>(inputShape.size());

        // Build epsilon constant
        Value constEpsilon = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(epsilon));

        // Use decomposed RMS norm to get intermediates
        // Normalize over the last dimension (axis = rank - 1)
        int64_t axis = rank - 1;
        RMSNormResult rmsResult =
            buildRMSNorm(binder, rewriter, inputSkipBiasSum, scale,
                         constEpsilon, axis, yType);

        // Build results based on number of declared outputs
        SmallVector<Value> results;
        int64_t numResults = binder.op->getNumResults();

        results.push_back(rmsResult.output); // output 0: normalized
        if (numResults > 1)
          results.push_back(rmsResult.meanSq); // output 1: mean(x²)
        if (numResults > 2)
          results.push_back(rmsResult.invStdVar); // output 2: inv_std_var
        if (numResults > 3)
          results.push_back(inputSkipBiasSum); // output 3: input_skip_bias_sum

        rewriter.replaceOp(binder.op, results);
        return success();
      });
  patterns.onOp(
      "MatMulIntegerToFloat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // This op:
        //     onnx.MatMulIntegerToFloat(
        //         lhs, rhs, lhs_scale, rhs_scale,
        //         [lhs_zp, rhs_zp, bias])
        //     -> fpType{dtype.float32, dtype.float16}
        //
        // Lowered to:
        //   1. torch.quantized_matmul
        //   2. Convert to float
        //   3. Scale by lhs_scale * rhs_scale
        //   4. Add bias if present
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value lhs, rhs, lhsScale, rhsScale, lhsZeroPoint, rhsZeroPoint, bias;
        if (binder.tensorOperandAtIndex(lhs, 0) ||
            binder.tensorOperandAtIndex(rhs, 1) ||
            binder.anyOperandAtIndex(lhsScale, 2) ||
            binder.anyOperandAtIndex(rhsScale, 3) ||
            binder.tensorResultType(resultType))
          return failure();

        auto *ctx = binder.op->getContext();

        // Get zero points (optional)
        Value noneVal = Torch::ConstantNoneOp::create(rewriter, loc);
        if (failed(binder.tensorOperandAtIndex(lhsZeroPoint, 4)))
          lhsZeroPoint = noneVal;
        if (failed(binder.tensorOperandAtIndex(rhsZeroPoint, 5)))
          rhsZeroPoint = noneVal;

        // Create torch.quantized_matmul op
        auto matMulIntTy = Torch::ValueTensorType::get(
            ctx, resultType.getSizes(),
            mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed));

        Value matMulInt = Torch::QuantizedMatmulOp::create(
            rewriter, loc, matMulIntTy, lhs, rhs, lhsZeroPoint, rhsZeroPoint);

        // Convert to float
        Value matmulFloat = Torch::convertTensorToDtype(
            rewriter, loc, matMulInt, resultType.getDtype());

        // Compute scale product
        auto lhsScaleTy = lhsScale.getType();
        auto lhsScaleTensorTy = dyn_cast<Torch::ValueTensorType>(lhsScaleTy);
        auto rhsScaleTy = rhsScale.getType();
        auto rhsScaleTensorTy = dyn_cast<Torch::ValueTensorType>(rhsScaleTy);

        const bool isLhsTrueTensor =
            lhsScaleTensorTy && !lhsScaleTensorTy.getSizes().empty();
        const bool isRhsTrueTensor =
            rhsScaleTensorTy && !rhsScaleTensorTy.getSizes().empty();
        if (isLhsTrueTensor && isRhsTrueTensor)
          return failure();

        Value scaleProduct;
        auto scaleProdTy = lhsScaleTensorTy;
        if (isRhsTrueTensor || !lhsScaleTensorTy)
          scaleProdTy = rhsScaleTensorTy;
        Value mulLhs = lhsScaleTensorTy ? lhsScale : rhsScale;
        Value mulRhs = lhsScaleTensorTy ? rhsScale : lhsScale;
        if (lhsScaleTensorTy && rhsScaleTensorTy)
          scaleProduct = Torch::AtenMulTensorOp::create(
              rewriter, loc, scaleProdTy, mulLhs, mulRhs);
        else if (lhsScaleTensorTy || rhsScaleTensorTy)
          scaleProduct = Torch::AtenMulScalarOp::create(
              rewriter, loc, scaleProdTy, mulLhs, mulRhs);
        else
          scaleProduct = Torch::AtenMulOp::create(rewriter, loc, lhsScaleTy,
                                                  lhsScale, rhsScale);

        // Apply scale to matmul result
        Value scaledMatmul;
        if (isa<Torch::ValueTensorType>(scaleProduct.getType()))
          scaledMatmul = Torch::AtenMulTensorOp::create(
              rewriter, loc, resultType, matmulFloat, scaleProduct);
        else
          scaledMatmul = Torch::AtenMulScalarOp::create(
              rewriter, loc, resultType, matmulFloat, scaleProduct);

        // Add bias if present
        if (failed(binder.tensorOperandAtIndex(bias, 6))) {
          rewriter.replaceOp(binder.op, scaledMatmul);
        } else {
          Value alphaFP = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(1.0));
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, scaledMatmul, bias, alphaFP);
        }
        return success();
      });
  patterns.onOp(
      "QuickGelu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Value input;
        float alpha;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.f32FloatAttr(alpha, "alpha", 1.0)) {
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get required input and attribute values");
        }
        auto inputType = cast<Torch::ValueTensorType>(input.getType());
        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType) || resultType != inputType) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Input and result type mismatch");
        }
        Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)alpha));
        auto scaleMul = rewriter.create<Torch::AtenMulScalarOp>(
            loc, inputType, input, constAlpha);
        auto sigmoid =
            rewriter.create<Torch::AtenSigmoidOp>(loc, inputType, scaleMul);
        rewriter.replaceOpWithNewOp<Torch::AtenMulTensorOp>(
            binder.op, resultType, input, sigmoid);
        return success();
      });
  patterns.onOp(
      "RotaryEmbedding", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        int64_t interleaved, isPackedBatching, numHeads, rotaryEmbeddingDim;
        float scale;
        Value input, positionIds, cosCache, sinCache;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(positionIds, 1) ||
            binder.tensorOperandAtIndex(cosCache, 2) ||
            binder.tensorOperandAtIndex(sinCache, 3) ||
            binder.s64IntegerAttr(interleaved, "interleaved", 0) ||
            binder.s64IntegerAttr(isPackedBatching, "is_packed_batching", 0) ||
            binder.s64IntegerAttr(numHeads, "num_heads", 0) ||
            binder.s64IntegerAttr(rotaryEmbeddingDim, "rotary_embedding_dim",
                                  0) ||
            binder.f32FloatAttr(scale, "scale", 1.0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");
        }

        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }

        Value cstInterleaved = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(interleaved));
        Value cstIsPackedBatching = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(isPackedBatching));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstRotaryEmbeddingDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(rotaryEmbeddingDim));
        Value cstScale = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(scale));

        rewriter.replaceOpWithNewOp<Torch::OnnxVariantRotaryEmbeddingOp>(
            binder.op, resultType, input, positionIds, cosCache, sinCache,
            cstInterleaved, cstIsPackedBatching, cstNumHeads,
            cstRotaryEmbeddingDim, cstScale);
        return success();
      });
  patterns.onOp(
      "GroupQueryAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<Value> operands;
        SmallVector<Type> resultTypes;
        int64_t doRotary, kvNumHeads, localWindowSize, numHeads,
            rotaryInterleaved, smoothSoftmax;
        float scale, softcap;
        if (binder.tensorOperandsList(operands))
          return rewriter.notifyMatchFailure(binder.op,
                                             "operands bind failure");

        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        if (resultTypes.size() != 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 3 result types");

        if (binder.s64IntegerAttr(doRotary, "do_rotary") ||
            binder.s64IntegerAttr(kvNumHeads, "kv_num_heads") ||
            binder.s64IntegerAttr(localWindowSize, "local_window_size", -1) ||
            binder.s64IntegerAttr(numHeads, "num_heads") ||
            binder.s64IntegerAttr(rotaryInterleaved, "rotary_interleaved") ||
            binder.f32FloatAttr(scale, "scale") ||
            binder.s64IntegerAttr(smoothSoftmax, "smooth_softmax") ||
            binder.f32FloatAttr(softcap, "softcap"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "op attributes bind failure");

        // This lowering excepts input operands to be either 7 or 9 based on the
        // `do_rotary` attribute. If it's false, then the input operands can be
        // 7 but if it's true then the operands has to be 9 including cos_cache
        // and sin_cache for rotary_embedding.
        if (!((operands.size() == 9) || (!doRotary && operands.size() == 7)))
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented:  excepted input operands to be either "
                         "7 or 9 based on the `do_rotary` attribute");

        if (kvNumHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "kv_num_heads is a required attribute and should be non-zero");

        if (localWindowSize != -1)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: local_window_size attribute is not supported, "
              "hence it should have default value equal to -1");

        if (numHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "num_heads is a required attribute and should be non-zero");

        if (smoothSoftmax != 0 && smoothSoftmax != -1)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: smooth_softmax attribute is not supported, hence "
              "it should have default value equal to 0 or -1");

        if (softcap != 0.0f)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: softcap attribute is not supported, "
                         "hence it should have default value equal to 0.0");

        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        Value query = operands[0];
        Value key = operands[1];
        Value value = operands[2];
        Value pastKey = operands[3];
        Value pastValue = operands[4];
        Value seqlensK = operands[5];
        Value totalSequenceLength = operands[6];
        Value cosCache, sinCache;
        if (doRotary) {
          cosCache = operands[7];
          sinCache = operands[8];
        }

        // Packed QKV: when key and value are None, the query input contains
        // all three projections concatenated along the last dimension.
        // Split into separate Q, K, V tensors before proceeding.
        if (isa<Torch::NoneType>(key.getType()) &&
            isa<Torch::NoneType>(value.getType())) {
          if (failed(splitPackedQKV(binder, rewriter, numHeads, kvNumHeads,
                                    query, key, value)))
            return failure();
        }

        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!queryType.hasSizes() ||
            queryType.getSizes()[2] == ShapedType::kDynamic)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected `query` input to have "
                                             "statically known hidden size");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        int64_t batchSize = queryDims[0];
        int64_t sequenceLength = queryDims[1];
        int64_t hiddenSize = queryDims[2];
        int64_t headSize = hiddenSize / numHeads;

        // For dynamic batch/sequence dimensions, use AtenSizeIntOp to get
        // runtime values instead of using -1 (kDynamic) as a constant.
        Value cstDim0 = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstDim1 = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value cstBatchSize = Torch::AtenSizeIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            query, cstDim0);
        Value cstSequenceLength = Torch::AtenSizeIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            query, cstDim1);
        Value cstHiddenSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(hiddenSize));
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(numHeads));
        Value cstKVNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(kvNumHeads));

        // Reshape Query, Key and Value as follows:
        // Query: (batch_size, sequence_length, hidden_size)
        //     -> (batch_size, num_heads, sequence_length, head_size)
        // Key: (batch_size, kv_sequence_length, kv_hidden_size)
        //   -> (batch_size, kv_num_heads, sequence_length, head_size)
        // Value: (batch_size, kv_sequence_length, kv_hidden_size)
        //     -> (batch_size, kv_num_heads, sequence_length, head_size)

        // Reshaping query from [batch, seq, hidden] to [batch, heads, seq,
        // head_dim]. First reshape to [batch, seq, heads, head_dim] (preserves
        // data layout), then transpose to [batch, heads, seq, head_dim].
        SmallVector<int64_t> queryReshapeSizesInt{batchSize, sequenceLength,
                                                  numHeads, headSize};
        Value queryReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstNumHeads, cstHeadSize});
        Value qReshaped = Torch::AtenReshapeOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryReshapeSizesInt,
                                           queryType.getOptionalDtype()),
            query, queryReshapeSizesList);
        // Transpose from [batch, seq, heads, head_dim] to [batch, heads, seq,
        // head_dim]
        SmallVector<int64_t> queryTransposedSizesInt{batchSize, numHeads,
                                                     sequenceLength, headSize};
        Value permuteDimsList =
            createConstantIntList(binder, rewriter, {0, 2, 1, 3});
        Value qInput = Torch::AtenPermuteOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryTransposedSizesInt,
                                           queryType.getOptionalDtype()),
            qReshaped, permuteDimsList);

        // Reshaping key from [batch, seq, kv_hidden] to [batch, kv_heads, seq,
        // head_dim].
        SmallVector<int64_t> kvReshapeSizesInt{batchSize, sequenceLength,
                                               kvNumHeads, headSize};
        Value kvReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstKVNumHeads, cstHeadSize});
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kReshaped = Torch::AtenReshapeOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvReshapeSizesInt,
                                         keyType.getOptionalDtype()),
            key, kvReshapeSizesList);
        SmallVector<int64_t> kvTransposedSizesInt{batchSize, kvNumHeads,
                                                  sequenceLength, headSize};
        Value kInput = Torch::AtenPermuteOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvTransposedSizesInt,
                                         keyType.getOptionalDtype()),
            kReshaped, permuteDimsList);

        // Reshaping value from [batch, seq, kv_hidden] to [batch, kv_heads,
        // seq, head_dim].
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vReshaped = Torch::AtenReshapeOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvReshapeSizesInt,
                                           valueType.getOptionalDtype()),
            value, kvReshapeSizesList);
        Value vInput = Torch::AtenPermuteOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvTransposedSizesInt,
                                           valueType.getOptionalDtype()),
            vReshaped, permuteDimsList);

        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);

        Value qRotary = qInput, kRotary = kInput;
        if (doRotary) {
          // Generating position_ids for rotary_embedding:
          //   past_seqlens = (seqlens_k + 1) - sequence_length
          //   pos_ids = arange(sequence_length).repeat(batch_size, 1)
          //          + past_seqlens.view(-1, 1)
          //
          // This works for all cases:
          //   - Initial prompt (past=0): pos_ids = [0, 1, ..., seq-1]
          //   - Decode (seq=1): pos_ids = [past_seq_len]
          //   - Subsequent prompt: pos_ids = [past, past+1, ..., past+seq-1]
          Value cstIntOne = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));

          SmallVector<int64_t> positionIdsSizeInt{batchSize, sequenceLength};
          Torch::ValueTensorType positionIdsType = Torch::ValueTensorType::get(
              context, positionIdsSizeInt,
              IntegerType::get(context, 64, IntegerType::Signed));
          Value cstInt64Dtype = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(
                  (int)torch_upstream::ScalarType::Long));

          Value cstInterleaved = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(rotaryInterleaved));
          Value cstIntZero = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
          Value cstFloatOne = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(1.0));

          // Convert seqlens_k which is a tensor of type si32 to si64.
          Torch::ValueTensorType seqLensKType =
              cast<Torch::ValueTensorType>(seqlensK.getType());
          seqlensK = Torch::AtenToDtypeOp::create(
              rewriter, loc,
              seqLensKType.getWithSizesAndDtype(
                  seqLensKType.getSizes(),
                  rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
              seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
              /*copy=*/cstFalse, /*memory_format=*/cstNone);
          Value totalSeqLens = Torch::AtenAddScalarOp::create(
              rewriter, loc, seqlensK.getType(), /*self=*/seqlensK,
              /*other=*/cstIntOne,
              /*alpha=*/cstIntOne);
          Value pastSeqLens = Torch::AtenSubScalarOp::create(
              rewriter, loc, totalSeqLens.getType(), /*self=*/totalSeqLens,
              /*other=*/cstSequenceLength, /*alpha=*/cstIntOne);
          Torch::ValueTensorType initPosIdsType = Torch::ValueTensorType::get(
              context, {sequenceLength},
              IntegerType::get(context, 64, IntegerType::Signed));
          Value initPosIds = Torch::AtenArangeOp::create(
              rewriter, loc, initPosIdsType, cstSequenceLength, cstInt64Dtype,
              /*layout=*/cstNone,
              /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value repeatValuesList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{cstBatchSize, cstIntOne});
          Value positionIds = Torch::AtenRepeatOp::create(
              rewriter, loc, positionIdsType, initPosIds,
              /*repeats=*/repeatValuesList);

          Value viewSizeList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{
                  Torch::ConstantIntOp::create(rewriter, binder.getLoc(),
                                               rewriter.getI64IntegerAttr(-1)),
                  cstIntOne});

          Torch::ValueTensorType seqLensViewType = Torch::ValueTensorType::get(
              context, llvm::SmallVector<int64_t>{batchSize, 1},
              IntegerType::get(context, 64, IntegerType::Signed));
          pastSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, pastSeqLens, viewSizeList);

          positionIds = Torch::AtenAddTensorOp::create(
              rewriter, loc, positionIdsType, positionIds, pastSeqLens,
              /*alpha=*/cstIntOne);

          // Performing RotaryEmbedding over Query and Key.
          qRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), qInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);

          kRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, kInput.getType(), kInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);
        }

        // Update KV cache and prepare keys/values for attention.
        // For in-place KV cache:
        // 1. Write new K/V to pre-allocated cache at position
        // [past_len:past_len+seq_len]
        // 2. Slice active portion [0:total_seq_len] for attention
        // 3. Return updated cache as present_key/present_value

        // Get total sequence length from the scalar input
        // totalSequenceLength is a scalar tensor [1] containing the total seq
        // len
        Value totalSeqLen = Torch::AtenItemOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(),
            totalSequenceLength);

        // past_seq_len = total_seq_len - current_seq_len
        Value pastSeqLen = Torch::AtenSubIntOp::create(
            rewriter, loc, totalSeqLen, cstSequenceLength);
        Value cstIntOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));

        // Compute end position for slice scatter: past_seq_len +
        // current_seq_len = total_seq_len
        Value cstCacheDim = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(2));
        Value cstIntZero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));

        // Write new K/V to cache using slice_scatter
        // present_key = past_key.slice_scatter(kRotary, dim=2,
        // start=past_seq_len, end=total_seq_len)
        Value presentKey = Torch::AtenSliceScatterOp::create(
            rewriter, loc, pastKey.getType(), pastKey, kRotary, cstCacheDim,
            pastSeqLen, totalSeqLen, cstIntOne);
        Value presentValue = Torch::AtenSliceScatterOp::create(
            rewriter, loc, pastValue.getType(), pastValue, vInput, cstCacheDim,
            pastSeqLen, totalSeqLen, cstIntOne);

        // Slice active portion [0:total_seq_len] for attention.
        // KV cache layout is [batch, kv_num_heads, seq, head_size], so the
        // sequence dimension is at index 2.
        constexpr int64_t kCacheSeqDim = 2;
        Value activeKey = buildActiveCacheSlice(
            rewriter, loc, presentKey, kCacheSeqDim, cstCacheDim, cstIntZero,
            totalSeqLen, cstIntOne);
        Value activeValue = buildActiveCacheSlice(
            rewriter, loc, presentValue, kCacheSeqDim, cstCacheDim, cstIntZero,
            totalSeqLen, cstIntOne);

        // Do attention using the full active cache
        Value cstEnableGQA = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value cstScale = cstNone;
        if (scale != 0.0f)
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qRotary.getType(), qRotary, activeKey, activeValue,
            /*attn_mask=*/cstNone,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/cstTrue, cstScale,
            cstEnableGQA);
        // Reshaping the attention result from:
        //    (batch_size, num_heads, sequence_length, head_size)
        // -> (batch_size, sequence_length, hidden_size)
        // First transpose from [batch, heads, seq, head_dim] to [batch, seq,
        // heads, head_dim]
        Value outputPermuteDimsList =
            createConstantIntList(binder, rewriter, {0, 2, 1, 3});
        SmallVector<int64_t> attentionTransposedSizes{batchSize, sequenceLength,
                                                      numHeads, headSize};
        Value attentionTransposed = Torch::AtenPermuteOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(attentionTransposedSizes,
                                           queryType.getOptionalDtype()),
            attention, outputPermuteDimsList);
        // Then reshape to [batch, seq, hidden]
        Value attentionResultSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(attention.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstHiddenSize});
        attention = Torch::AtenReshapeOp::create(rewriter, loc, resultTypes[0],
                                                 attentionTransposed,
                                                 attentionResultSizesList);

        rewriter.replaceOp(binder.op, {attention, presentKey, presentValue});
        return success();
      });
  patterns.onOp(
      "QLinearAdd", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, aScale, aZp, b, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the result of "Add".
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value alpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value c = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b, alpha);

        // Quantizing the result of "Add" operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "QLinearLeakyRelu", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        float alpha;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha"))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the LeakyRelu result.
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)alpha));
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y =
            Torch::AtenLeakyReluOp::create(rewriter, loc, yTy, x, constAlpha);

        // Quantizing the result of LeakyRelu op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearConcat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t axis;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(axis, "axis") ||
            binder.tensorResultType(resultType))
          return failure();

        SmallVector<Value> inputs, inputScales, inputZeroPoints;
        for (unsigned i = 2; i < operands.size(); i = i + 3) {
          inputs.push_back(operands[i]);
          inputScales.push_back(operands[i + 1]);
          inputZeroPoints.push_back(operands[i + 2]);
        }

        unsigned numInputs = (operands.size() - 2) / 3;
        if (!(llvm::all_equal({inputs.size(), inputScales.size(),
                               inputZeroPoints.size()}) &&
              inputs.size() == numInputs))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible number of input operands, scales and/or "
                         "zero-points");

        // Preparing the dequantized inputs.
        SmallVector<Value> dequantizedInputs;
        for (unsigned i = 0; i < numInputs; i++) {
          Value scale, zeroPoint;
          if (failed(extractPerTensorQuantizationArguments(
                  rewriter, loc, /*scale=*/inputScales[i],
                  /*zero_point=*/inputZeroPoints[i], scale, zeroPoint)))
            return rewriter.notifyMatchFailure(
                binder.op, "Incompatible scale and zero-points argument for "
                           "per-tensor quantization");

          Value dequantizedInput;
          if (failed(createDequantizeTensor(rewriter, loc, inputs[i], scale,
                                            zeroPoint,
                                            /*output=*/dequantizedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to dequantize the input tensor because of "
                           "missing sizes");

          dequantizedInputs.push_back(dequantizedInput);
        }

        // Concatenating the inputs.
        Type listElemType =
            cast<Torch::BaseTensorType>(dequantizedInputs[0].getType())
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = Torch::PrimListConstructOp::create(
            rewriter, binder.op->getLoc(), listType, dequantizedInputs);
        Value cstAxis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axis));
        auto concatTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value concat = Torch::AtenCatOp::create(rewriter, loc, concatTy,
                                                tensorList, cstAxis);

        // Quantizing the result of concatenated inputs.
        Value yScale, yZp;
        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[0],
                /*zero_point=*/operands[1], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible scale and zero-points argument for "
                         "per-tensor quantization");
        Torch::ValueTensorType yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        Value result = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, concat, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          result);
        return success();
      });
  patterns.onOp(
      "QLinearGlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperands(operands, 5) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        auto xTy = dyn_cast<Torch::ValueTensorType>(operands[0].getType());
        if (!xTy || !xTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input argument `x` to have sizes");
        ArrayRef<int64_t> inputShape = xTy.getSizes();

        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the AvgPool result.
        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        unsigned inputRank = inputShape.size();
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            Value inputDimSize =
                Torch::AtenSizeIntOp::create(rewriter, loc, x, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultShape, rewriter.getF32Type());
        Value avgpool;
        if (inputRank == 3) {
          avgpool = Torch::AtenAvgPool1dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad);
        } else if (inputRank == 4) {
          avgpool = Torch::AtenAvgPool2dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else if (inputRank == 5) {
          avgpool = Torch::AtenAvgPool3dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else {
          return failure();
        }

        // Quantizing the result of AvgPool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        avgpool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, avgpool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          avgpool);
        return success();
      });
  patterns.onOp(
      "QLinearSigmoid", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the Sigmoid result.
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y = Torch::AtenSigmoidOp::create(rewriter, loc, yTy, x);

        // Quantizing the result of Sigmoid op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Creating Onnx.AveragePool op.
        llvm::SmallVector<Value> newOperands = {x};
        llvm::SmallVector<NamedAttribute> newAttributes;
        newAttributes.push_back(rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("onnx.AveragePool")));
        for (auto namedAttr : binder.op->getAttrDictionary()) {
          if (namedAttr.getName().getValue().compare("name") == 0)
            continue;
          newAttributes.push_back(namedAttr);
        }

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value averagePool = Torch::OperatorOp::create(
                                rewriter, binder.getLoc(), yTy, newOperands,
                                newAttributes, binder.op->getRegions().size())
                                .getResult(0);

        // Quantizing the result of AveragePool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        averagePool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, averagePool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          averagePool);
        return success();
      });
  patterns.onOp(
      "FusedMatMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        int64_t transA, transB, transBatchA, transBatchB;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.s64IntegerAttr(transBatchA, "transBatchA", 0) ||
            binder.s64IntegerAttr(transBatchB, "transBatchB", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        // Transposing the LHS argument.
        Value transposedLhs = lhs;
        if (transA) {
          // Determine the rank of lhs tensor.
          std::optional<unsigned> maybeRank = Torch::getTensorRank(lhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked lhs tensor");
          unsigned lhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), lhs,
                  /*dimA=*/lhsRank - 2, /*dimB=*/lhsRank - 1, transposedLhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for lhs");
        }

        // Transposing the RHS argument.
        Value transposedRhs = rhs;
        if (transB) {
          std::optional<unsigned> maybeRank = Torch::getTensorRank(rhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked rhs tensor");
          unsigned rhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), rhs,
                  /*dimA=*/rhsRank - 2, /*dimB=*/rhsRank - 1, transposedRhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for rhs");
        }

        // TODO: Add support for `transBatchA` and `transBatchB`
        // attribute.
        if (transBatchA || transBatchB)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: support not present for "
                         "transBatchA and transBatchB attribute");

        rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
            binder.op, resultType, transposedLhs, transposedRhs);
        return success();
      });
  patterns.onOp(
      "QLinearMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, b, aScale, aZp, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the Mul result.
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value c = Torch::AtenMulTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b);

        // Quantizing the result of Mul operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "MatMulNBits", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        int64_t k, n, accuracyLevel, bits, blockSize;
        Value lhs, rhs, scales, zeroPoints, bias;
        if (binder.tensorOperandAtIndex(lhs, 0) ||
            binder.tensorOperandAtIndex(rhs, 1) ||
            binder.tensorOperandAtIndex(scales, 2) ||
            binder.s64IntegerAttr(k, "K") || binder.s64IntegerAttr(n, "N") ||
            binder.s64IntegerAttr(accuracyLevel, "accuracy_level", 0) ||
            binder.s64IntegerAttr(bits, "bits") ||
            binder.s64IntegerAttr(blockSize, "block_size"))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get required input and attribute values");

        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Result type bind failure");

        if (failed(binder.tensorOperandAtIndex(zeroPoints, 3)))
          zeroPoints = Torch::ConstantNoneOp::create(rewriter, loc);

        // Index 4 (`g_idx`) is deprecated in ORT.
        Value gIdx;
        if (succeeded(binder.tensorOperandAtIndex(gIdx, 4)))
          return rewriter.notifyMatchFailure(binder.op, "Deprecated: `g_idx`");

        if (failed(binder.tensorOperandAtIndex(bias, 5)))
          bias = Torch::ConstantNoneOp::create(rewriter, loc);

        // scales and zero_points may be 1D-flattened per ONNX spec.
        // downstream lowerings expect the 2D form.
        scales = promote1DTo2D(rewriter, loc, scales, n);
        zeroPoints = promote1DTo2D(rewriter, loc, zeroPoints, n);

        MLIRContext *context = rewriter.getContext();
        auto si64Type = IntegerType::get(context, 64, IntegerType::Signed);
        rewriter.replaceOpWithNewOp<Torch::MatMulNBitsOp>(
            binder.op, resultType, lhs, rhs, scales, zeroPoints, bias,
            IntegerAttr::get(si64Type, accuracyLevel),
            IntegerAttr::get(si64Type, bits),
            IntegerAttr::get(si64Type, blockSize));
        return success();
      });
}
