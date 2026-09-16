import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../utils/contiguous_storage.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// Gathers values along an axis specified by dim.
///
/// Equivalent to `torch.gather()` in PyTorch and ONNX `GatherElements`.
///
/// For a 3D tensor with dim=0:
/// `output[i][j][k] = input[index[i][j][k]][j][k]`
///
/// Output shape equals the index shape.
///
/// ```dart
/// final result = GatherOp(dim: 0, index: indices)(input);
/// ```
class GatherOp extends TransformOp {
  /// The dimension along which to index.
  final int dim;

  /// The index tensor containing indices to gather.
  final TensorBuffer index;

  /// Creates a Gather operation.
  GatherOp({required this.dim, required this.index});

  @override
  String get name => 'Gather';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    pytorchEquivalent: 'torch.gather',
    onnxOpType: 'GatherElements',
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final rank = input.rank;

    // Normalize dim
    final normalizedDim = dim < 0 ? rank + dim : dim;
    if (normalizedDim < 0 || normalizedDim >= rank) {
      throw IndexOutOfBoundsException(
        index: dim,
        min: -rank,
        max: rank - 1,
        dimension: 'GatherOp dim',
      );
    }

    // Validate rank match
    if (index.rank != rank) {
      throw ShapeMismatchException(
        actual: index.shape,
        message: 'Gather: index rank ${index.rank} must match input rank $rank',
      );
    }

    if (!index.dtype.isInteger) {
      throw InvalidParameterException(
        'index.dtype',
        index.dtype,
        'Gather indices must have an integer dtype',
      );
    }
    for (var d = 0; d < rank; d++) {
      if (d != normalizedDim && index.shape[d] > input.shape[d]) {
        throw ShapeMismatchException(
          actual: index.shape,
          message: 'Gather index exceeds input size on non-gather dimension $d',
        );
      }
    }

    final inputContiguous = contiguousStorageView(input);
    final indexContiguous = contiguousStorageView(index);

    final outputShape = List<int>.from(indexContiguous.shape);
    final output = TensorBuffer.uninitialized(outputShape, dtype: input.dtype);
    final numel = output.numel;

    final inputShape = inputContiguous.shape;
    final inStrides = TensorIndexer.computeStrides(inputShape);
    final outStrides = TensorIndexer.computeStrides(outputShape);

    // Read index values and validate range
    final indexData = indexContiguous.storage.data as List<int>;
    final dimSize = inputShape[normalizedDim];
    for (int i = 0; i < numel; i++) {
      final indexVal = indexData[i];
      if (indexVal < 0 || indexVal >= dimSize) {
        throw IndexOutOfBoundsException(
          index: indexVal,
          min: 0,
          max: dimSize - 1,
          dimension: 'Gather index at dim $normalizedDim',
        );
      }
    }

    switch (input.dtype) {
      case DType.float32:
        final inData = inputContiguous.storage.data as Float32List;
        final outData = output.storage.data as Float32List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int srcIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == normalizedDim) {
              // Use the index value for this dimension
              final indexVal = indexData[outIdx];
              srcIdx += indexVal * inStrides[d];
            } else {
              srcIdx += coord * inStrides[d];
            }
          }

          outData[outIdx] = inData[srcIdx];
        }

      case DType.float64:
        final inData = inputContiguous.storage.data as Float64List;
        final outData = output.storage.data as Float64List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int srcIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == normalizedDim) {
              final indexVal = indexData[outIdx];
              srcIdx += indexVal * inStrides[d];
            } else {
              srcIdx += coord * inStrides[d];
            }
          }

          outData[outIdx] = inData[srcIdx];
        }

      default:
        final inStorage = inputContiguous.storage;
        final outStorage = output.storage;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int srcIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];

            if (d == normalizedDim) {
              final indexVal = indexData[outIdx];
              srcIdx += indexVal * inStrides[d];
            } else {
              srcIdx += coord * inStrides[d];
            }
          }

          (outStorage.data as List<num>)[outIdx] =
              (inStorage.data as List<num>)[srcIdx];
        }
    }

    return output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) =>
      List<int>.from(index.shape);
}
