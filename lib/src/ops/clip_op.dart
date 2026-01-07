import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Clips tensor values to a specified range.
///
/// Element-wise operation that clamps values to [min, max] range.
/// Values below min are set to min, values above max are set to max.
class ClipOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Minimum value to clip to.
  final double min;

  /// Maximum value to clip to.
  final double max;

  /// Creates a clip operation with the given [min] and [max] bounds.
  ClipOp({required this.min, required this.max}) {
    if (min >= max) {
      throw InvalidParameterException(
        'min/max',
        'min=$min, max=$max',
        'min must be less than max',
      );
    }
  }

  /// Creates a clip operation for unit range [0.0, 1.0].
  factory ClipOp.unit() => ClipOp(min: 0.0, max: 1.0);

  /// Creates a clip operation for symmetric range [-1.0, 1.0].
  factory ClipOp.symmetric() => ClipOp(min: -1.0, max: 1.0);

  /// Creates a clip operation for uint8 range [0.0, 255.0].
  factory ClipOp.uint8() => ClipOp(min: 0.0, max: 255.0);

  @override
  String get name => 'Clip(min=$min, max=$max)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final output = contiguous.clone();
    _clip(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('ClipOp.applyInPlace');
    }
    _clip(input);
  }

  void _clip(TensorBuffer tensor) {
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      final clampedValue = value.clamp(min, max);
      tensor.storage.setFromDouble(i, clampedValue);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
