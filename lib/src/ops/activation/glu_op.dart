import 'dart:math' as math;
import 'dart:typed_data';

import '../../core/dtype.dart';
import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/tensor_indexing.dart';
import '../transform_op.dart';

/// Gated Linear Unit (GLU) activation function.
///
/// Equivalent to `F.glu()` in PyTorch.
///
/// ## Formula
///
/// `GLU(a, b) = a * sigmoid(b)`
///
/// where `[a, b] = split(input, dim)`.
///
/// The specified dimension must have even size, and the output halves that
/// dimension.
///
/// ```dart
/// final result = GLUOp(dim: -1)(tensor);
/// ```
class GLUOp extends TransformOp with RequiresContiguous {
  /// The dimension along which to split. Supports negative indexing.
  final int dim;

  /// Creates a GLU operation splitting along [dim].
  GLUOp({this.dim = -1});

  @override
  String get name => dim == -1 ? 'GLU' : 'GLU(dim=$dim)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    requiresContiguous: true,
    pytorchEquivalent: 'F.glu',
    onnxOpType: 'Split+Sigmoid+Mul',
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final shape = contiguous.shape;
    final rank = shape.length;

    // Normalize dim
    final normalizedDim = dim < 0 ? rank + dim : dim;
    if (normalizedDim < 0 || normalizedDim >= rank) {
      throw IndexOutOfBoundsException(
        index: dim,
        min: -rank,
        max: rank - 1,
        dimension: 'GLUOp dim',
      );
    }

    final dimSize = shape[normalizedDim];
    if (dimSize % 2 != 0) {
      throw InvalidParameterException(
        'dim',
        dim,
        'GLUOp: dimension $normalizedDim has odd size $dimSize, must be even',
      );
    }

    final halfSize = dimSize ~/ 2;
    final outputShape = List<int>.from(shape);
    outputShape[normalizedDim] = halfSize;

    final output = TensorBuffer.uninitialized(
      outputShape,
      dtype: contiguous.dtype,
    );

    _computeGlu(contiguous, output, normalizedDim, halfSize);

    return output;
  }

  void _computeGlu(
    TensorBuffer input,
    TensorBuffer output,
    int axis,
    int halfSize,
  ) {
    final shape = input.shape;
    final rank = shape.length;
    final outShape = output.shape;

    final inStrides = TensorIndexer.computeStrides(shape);
    final outStrides = TensorIndexer.computeStrides(outShape);
    final numel = output.numel;

    switch (input.dtype) {
      case DType.float32:
        final inData = input.storage.data as Float32List;
        final outData = output.storage.data as Float32List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          // Convert output linear index to coordinates
          int remaining = outIdx;
          int aIdx = 0;
          int bIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == axis) {
              aIdx += coord * inStrides[d];
              bIdx += (coord + halfSize) * inStrides[d];
            } else {
              aIdx += coord * inStrides[d];
              bIdx += coord * inStrides[d];
            }
          }

          final a = inData[aIdx];
          final b = inData[bIdx];
          outData[outIdx] = a * (1.0 / (1.0 + math.exp(-b)));
        }

      case DType.float64:
        final inData = input.storage.data as Float64List;
        final outData = output.storage.data as Float64List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int aIdx = 0;
          int bIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == axis) {
              aIdx += coord * inStrides[d];
              bIdx += (coord + halfSize) * inStrides[d];
            } else {
              aIdx += coord * inStrides[d];
              bIdx += coord * inStrides[d];
            }
          }

          final a = inData[aIdx];
          final b = inData[bIdx];
          outData[outIdx] = a * (1.0 / (1.0 + math.exp(-b)));
        }

      default:
        final inStorage = input.storage;
        final outStorage = output.storage;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int aIdx = 0;
          int bIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == axis) {
              aIdx += coord * inStrides[d];
              bIdx += (coord + halfSize) * inStrides[d];
            } else {
              aIdx += coord * inStrides[d];
              bIdx += coord * inStrides[d];
            }
          }

          final a = inStorage.getAsDouble(aIdx);
          final b = inStorage.getAsDouble(bIdx);
          outStorage.setFromDouble(outIdx, a * (1.0 / (1.0 + math.exp(-b))));
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final rank = inputShape.length;
    final normalizedDim = dim < 0 ? rank + dim : dim;
    final result = List<int>.from(inputShape);
    result[normalizedDim] = inputShape[normalizedDim] ~/ 2;
    return result;
  }
}
