import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Fills elements of the tensor where mask is non-zero with a given value.
///
/// Equivalent to `Tensor.masked_fill_()` in PyTorch.
///
/// ```dart
/// final result = MaskedFillOp(mask: mask, value: -1e9)(tensor);
/// ```
class MaskedFillOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// The mask tensor (non-zero = fill).
  final TensorBuffer mask;

  /// The value to fill at masked positions.
  final double value;

  /// Creates a MaskedFill operation.
  MaskedFillOp({required this.mask, required this.value});

  /// Creates a MaskedFill operation for attention masking.
  ///
  /// Fills masked positions with negative infinity, which effectively
  /// zeroes them out after softmax.
  factory MaskedFillOp.attentionMask(TensorBuffer mask) =>
      MaskedFillOp(mask: mask, value: double.negativeInfinity);

  @override
  String get name => 'MaskedFill';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'Tensor.masked_fill_',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShapes(input);
    final output = cloneForModification(input);
    _maskedFill(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('MaskedFillOp.applyInPlace');
    }
    _validateShapes(input);
    _maskedFill(input);
  }

  void _validateShapes(TensorBuffer input) {
    final inputShape = input.shape;
    final maskShape = mask.shape;

    if (inputShape.length != maskShape.length) {
      throw ShapeMismatchException(
        actual: maskShape,
        message:
            'MaskedFill: mask shape $maskShape does not match input shape $inputShape',
      );
    }
    for (int i = 0; i < inputShape.length; i++) {
      if (inputShape[i] != maskShape[i]) {
        throw ShapeMismatchException(
          actual: maskShape,
          message:
              'MaskedFill: mask shape $maskShape does not match input shape $inputShape',
        );
      }
    }
  }

  void _maskedFill(TensorBuffer tensor) {
    final maskContiguous = mask.isContiguous ? mask : mask.contiguous();
    final numel = tensor.numel;
    final fillVal = value;

    switch (tensor.dtype) {
      case DType.float32:
        final data = tensor.storage.data as Float32List;
        final maskData = maskContiguous.storage.data as Float32List;
        for (int i = 0; i < numel; i++) {
          if (maskData[i] != 0) data[i] = fillVal;
        }

      case DType.float64:
        final data = tensor.storage.data as Float64List;
        final maskData = maskContiguous.storage.data as Float64List;
        for (int i = 0; i < numel; i++) {
          if (maskData[i] != 0) data[i] = fillVal;
        }

      default:
        final storage = tensor.storage;
        final maskStorage = maskContiguous.storage;
        for (int i = 0; i < numel; i++) {
          if (maskStorage.getAsDouble(i) != 0) {
            storage.setFromDouble(i, fillVal);
          }
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
