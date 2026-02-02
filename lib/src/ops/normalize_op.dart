import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/dtype_dispatcher.dart';
import '../utils/simd_ops.dart';
import 'transform_op.dart';

/// Normalizes tensor values per channel using mean and standard deviation.
///
/// Applies the formula: `(value - mean) / std` for each channel.
///
/// ## Complexity
///
/// - **Time**: O(n) where n = total elements. Uses SIMD acceleration for Float32/Float64.
/// - **Space**: O(n) for output tensor (or O(1) with in-place mode).
class NormalizeOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Per-channel mean values to subtract.
  final List<double> mean;

  /// Per-channel standard deviation values to divide by.
  final List<double> std;

  /// Creates a normalize operation with the given [mean] and [std].
  NormalizeOp({required this.mean, required this.std}) {
    if (mean.length != std.length) {
      throw InvalidParameterException(
        'mean/std',
        'mean.length=${mean.length}, std.length=${std.length}',
        'Must have same length',
      );
    }
    if (mean.isEmpty) {
      throw InvalidParameterException(
        'mean/std',
        'empty',
        'Must have at least one channel',
      );
    }
    for (int i = 0; i < std.length; i++) {
      if (std[i] == 0) {
        throw InvalidParameterException(
          'std[$i]',
          '0',
          'Standard deviation cannot be zero',
        );
      }
    }
  }

  /// Creates normalization using ImageNet statistics.
  factory NormalizeOp.imagenet() {
    return NormalizeOp(
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    );
  }

  /// Creates normalization using CIFAR-10 statistics.
  factory NormalizeOp.cifar10() {
    return NormalizeOp(
      mean: [0.4914, 0.4822, 0.4465],
      std: [0.2470, 0.2435, 0.2616],
    );
  }

  /// Creates symmetric normalization mapping `[0, 1]` to `[-1, 1]`.
  factory NormalizeOp.symmetric() {
    return NormalizeOp(
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    );
  }

  @override
  String get name => 'Normalize(mean=$mean, std=$std)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);
    final output = cloneForModification(input);
    _normalize(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('NormalizeOp.applyInPlace');
    }
    _validateShape(input.shape);
    _normalize(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'NormalizeOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != mean.length) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but mean/std has ${mean.length}',
      );
    }
  }

  void _normalize(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank == 3) {
      _normalize3D(tensor);
    } else {
      _normalize4D(tensor);
    }
  }

  void _normalize3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;
    final data = tensor.storage.data;

    // Dtype-specialized loop to avoid per-element switch overhead
    switch (tensor.dtype) {
      case DType.float32:
        // SIMD-optimized path for Float32
        final list = data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final channelData =
              Float32List.sublistView(list, offset, offset + channelSize);
          SimdOps.normalize(channelData, mean[ch], std[ch]);
        }
      case DType.float64:
        // SIMD-optimized path for Float64 (aligned data only, else scalar fallback)
        final list = data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final channelData =
              Float64List.sublistView(list, offset, offset + channelSize);
          SimdOps.normalizeF64(channelData, mean[ch], std[ch]);
        }
      default:
        // Fallback for integer types (less common for normalization)
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final m = mean[ch];
          final s = std[ch];
          for (int i = 0; i < channelSize; i++) {
            final idx = offset + i;
            final value = tensor.storage.getAsDouble(idx);
            tensor.storage.setFromDouble(idx, (value - m) / s);
          }
        }
    }
  }

  void _normalize4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = c * channelSize;
    final data = tensor.storage.data;

    // Dtype-specialized loop to avoid per-element switch overhead
    switch (tensor.dtype) {
      case DType.float32:
        // SIMD-optimized path for Float32
        final list = data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final channelData =
                Float32List.sublistView(list, offset, offset + channelSize);
            SimdOps.normalize(channelData, mean[ch], std[ch]);
          }
        }
      case DType.float64:
        // SIMD-optimized path for Float64 (aligned data only, else scalar fallback)
        final list = data as Float64List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final channelData =
                Float64List.sublistView(list, offset, offset + channelSize);
            SimdOps.normalizeF64(channelData, mean[ch], std[ch]);
          }
        }
      default:
        // Fallback for integer types
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final m = mean[ch];
            final s = std[ch];
            for (int i = 0; i < channelSize; i++) {
              final idx = offset + i;
              final value = tensor.storage.getAsDouble(idx);
              tensor.storage.setFromDouble(idx, (value - m) / s);
            }
          }
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Scales tensor values by a constant factor.
///
/// Applies the formula: `(value - offset) / scale`.
class ScaleOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// The scale divisor.
  final double scale;

  /// The offset to subtract before scaling.
  final double offset;

  /// Creates a scale operation with the given [scale] and [offset].
  ScaleOp({this.scale = 255.0, this.offset = 0.0}) {
    if (scale == 0) {
      throw InvalidParameterException('scale', '0', 'Cannot be zero');
    }
  }

  /// Scales `[0, 255]` to `[0, 1]`.
  factory ScaleOp.toUnit() => ScaleOp(scale: 255.0);

  /// Scales `[0, 1]` to `[0, 255]`.
  factory ScaleOp.fromUnit() => ScaleOp(scale: 1 / 255.0);

  /// Scales `[0, 255]` to `[-1, 1]`.
  factory ScaleOp.toSymmetric() => ScaleOp(scale: 127.5, offset: 127.5);

  @override
  String get name => 'Scale(scale=$scale, offset=$offset)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _scale(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('ScaleOp.applyInPlace');
    }
    _scale(input);
  }

  void _scale(TensorBuffer tensor) {
    final s = scale;
    final o = offset;
    // (value - offset) / scale = value * invScale + bias
    // where invScale = 1/scale and bias = -offset/scale
    final invScale = 1.0 / s;
    final bias = -o / s;

    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        // Use SIMD acceleration for Float32
        SimdOps.multiplyScalar(list, invScale);
        SimdOps.addScalar(list, bias);
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          list[i] = (list[i] - o) / s;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final value = t.storage.getAsDouble(i);
          t.storage.setFromDouble(i, (value - o) / s);
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
