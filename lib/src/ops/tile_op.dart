import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// Tiles a tensor by repeating it along each dimension.
///
/// Equivalent to `Tensor.repeat()` in PyTorch and ONNX `Tile` operator.
///
/// Each dimension is repeated `reps[i]` times, so the output shape is
/// `shape[i] * reps[i]` for each dimension.
///
/// ```dart
/// final result = TileOp(reps: [2, 3])(tensor);
/// ```
class TileOp extends TransformOp with RequiresContiguous {
  /// The number of repetitions along each dimension.
  final List<int> reps;

  /// Creates a Tile operation.
  TileOp({required this.reps});

  @override
  String get name => 'Tile';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    requiresContiguous: true,
    pytorchEquivalent: 'Tensor.repeat',
    onnxOpType: 'Tile',
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final shape = contiguous.shape;
    final rank = shape.length;

    if (reps.length != rank) {
      throw InvalidParameterException(
        'reps',
        reps,
        'Tile: reps length ${reps.length} must match tensor rank $rank',
      );
    }

    final outputShape = computeOutputShape(shape);
    final output = TensorBuffer.uninitialized(
      outputShape,
      dtype: contiguous.dtype,
    );

    final inStrides = TensorIndexer.computeStrides(shape);
    final outStrides = TensorIndexer.computeStrides(outputShape);
    final numel = output.numel;

    switch (contiguous.dtype) {
      case DType.float32:
        final inData = contiguous.storage.data as Float32List;
        final outData = output.storage.data as Float32List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int inIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];
            final inCoord = coord % shape[d];
            inIdx += inCoord * inStrides[d];
          }

          outData[outIdx] = inData[inIdx];
        }

      case DType.float64:
        final inData = contiguous.storage.data as Float64List;
        final outData = output.storage.data as Float64List;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int inIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];
            final inCoord = coord % shape[d];
            inIdx += inCoord * inStrides[d];
          }

          outData[outIdx] = inData[inIdx];
        }

      default:
        final inStorage = contiguous.storage;
        final outStorage = output.storage;

        for (int outIdx = 0; outIdx < numel; outIdx++) {
          int remaining = outIdx;
          int inIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ outStrides[d];
            remaining = remaining % outStrides[d];
            final inCoord = coord % shape[d];
            inIdx += inCoord * inStrides[d];
          }

          outStorage.setFromDouble(outIdx, inStorage.getAsDouble(inIdx));
        }
    }

    return output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    return [
      for (int i = 0; i < inputShape.length; i++) inputShape[i] * reps[i],
    ];
  }
}
