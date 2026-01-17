import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/simd_ops.dart';
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
    // Optimization: if input is already contiguous, clone() once.
    // If not contiguous, contiguous() creates a copy, so no need to clone again.
    final TensorBuffer output;
    if (input.isContiguous) {
      output = input.clone();
    } else {
      output = input.contiguous();
    }
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
    final data = tensor.storage.data;

    // Dtype-specialized loop to avoid per-element switch overhead
    switch (tensor.dtype) {
      case DType.float32:
        // SIMD-optimized path for Float32
        SimdOps.clip(data as Float32List, min, max);
      case DType.float64:
        // SIMD-optimized path for Float64 (aligned data only, else scalar fallback)
        SimdOps.clipF64(data as Float64List, min, max);
      default:
        // Fallback for integer types
        for (int i = 0; i < numel; i++) {
          final value = tensor.storage.getAsDouble(i);
          tensor.storage.setFromDouble(i, value.clamp(min, max));
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
