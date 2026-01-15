import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Group normalization operation.
///
/// Normalizes across groups of channels. Used in U-Net, newer CNNs when
/// batch size is small.
///
/// **PyTorch Reference**: `torch.nn.GroupNorm`, `F.group_norm()`
///
/// **Formula**:
/// ```
/// For each group g:
///   group_mean = mean of channels in group
///   group_var = variance of channels in group
///   y[c] = (x[c] - group_mean) / sqrt(group_var + eps) * weight[c] + bias[c]
/// ```
///
/// **Usage**:
/// ```dart
/// final groupNorm = GroupNormOp(
///   numGroups: 32,
///   numChannels: 256,
///   weight: List<double>.filled(256, 1.0),
///   bias: List<double>.filled(256, 0.0),
/// );
/// final output = groupNorm.apply(input);
/// ```
class GroupNormOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Number of groups to divide channels into.
  final int numGroups;

  /// Number of channels (must be divisible by numGroups).
  final int numChannels;

  /// Optional learned scale parameter (gamma/weight).
  /// Must have length == numChannels if provided.
  final List<double>? weight;

  /// Optional learned bias parameter (beta/bias).
  /// Must have length == numChannels if provided.
  final List<double>? bias;

  /// Small constant for numerical stability (default: 1e-5).
  final double eps;

  /// Creates a Group Normalization operation.
  ///
  /// [numGroups]: Number of groups to divide channels into.
  /// [numChannels]: Total number of channels (must be divisible by numGroups).
  /// [weight]: Optional per-channel scale parameter.
  /// [bias]: Optional per-channel bias parameter.
  /// [eps]: Numerical stability constant (default: 1e-5).
  ///
  /// Throws [InvalidParameterException] if:
  /// - numGroups <= 0
  /// - numChannels <= 0
  /// - numChannels % numGroups != 0
  /// - weight length != numChannels
  /// - bias length != numChannels
  GroupNormOp({
    required this.numGroups,
    required this.numChannels,
    this.weight,
    this.bias,
    this.eps = 1e-5,
  }) {
    if (numGroups <= 0) {
      throw InvalidParameterException(
        'numGroups',
        numGroups.toString(),
        'Must be positive',
      );
    }
    if (numChannels <= 0) {
      throw InvalidParameterException(
        'numChannels',
        numChannels.toString(),
        'Must be positive',
      );
    }
    if (numChannels % numGroups != 0) {
      throw InvalidParameterException(
        'numChannels',
        numChannels.toString(),
        'Must be divisible by numGroups ($numGroups)',
      );
    }
    if (weight != null && weight!.length != numChannels) {
      throw InvalidParameterException(
        'weight',
        'length=${weight!.length}',
        'Must have length $numChannels',
      );
    }
    if (bias != null && bias!.length != numChannels) {
      throw InvalidParameterException(
        'bias',
        'length=${bias!.length}',
        'Must have length $numChannels',
      );
    }
    if (eps <= 0) {
      throw InvalidParameterException(
        'eps',
        eps.toString(),
        'Must be positive',
      );
    }
  }

  /// Creates Group Normalization with default PyTorch-style initialization.
  ///
  /// weight is initialized to 1.0, bias to 0.0.
  factory GroupNormOp.withAffine({
    required int numGroups,
    required int numChannels,
    double eps = 1e-5,
  }) {
    return GroupNormOp(
      numGroups: numGroups,
      numChannels: numChannels,
      weight: List<double>.filled(numChannels, 1.0),
      bias: List<double>.filled(numChannels, 0.0),
      eps: eps,
    );
  }

  /// Factory from PyTorch state_dict values.
  ///
  /// Accepts Float32List weights directly from ONNX/PyTorch exports.
  factory GroupNormOp.fromStateDict({
    required int numGroups,
    required int numChannels,
    Float32List? weight,
    Float32List? bias,
    double eps = 1e-5,
  }) {
    return GroupNormOp(
      numGroups: numGroups,
      numChannels: numChannels,
      weight: weight?.toList(),
      bias: bias?.toList(),
      eps: eps,
    );
  }

  @override
  String get name => 'GroupNorm(groups=$numGroups, channels=$numChannels)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);
    final output = cloneForModification(input);
    _groupNorm(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('GroupNormOp.applyInPlace');
    }
    _validateShape(input.shape);
    _groupNorm(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'GroupNormOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != numChannels) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but GroupNormOp expects $numChannels',
      );
    }
  }

  void _groupNorm(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank == 3) {
      _groupNorm3D(tensor);
    } else {
      _groupNorm4D(tensor);
    }
  }

  void _groupNorm3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final spatialSize = h * w;
    final channelsPerGroup = c ~/ numGroups;
    final groupSize = channelsPerGroup * spatialSize;

    switch (tensor.dtype) {
      case DType.float32:
        _groupNorm3DFloat32(
          tensor.storage.data as Float32List,
          c,
          spatialSize,
          channelsPerGroup,
          groupSize,
        );
      case DType.float64:
        _groupNorm3DFloat64(
          tensor.storage.data as Float64List,
          c,
          spatialSize,
          channelsPerGroup,
          groupSize,
        );
      default:
        _groupNorm3DGeneric(tensor, c, spatialSize, channelsPerGroup, groupSize);
    }
  }

  void _groupNorm3DFloat32(
    Float32List data,
    int c,
    int spatialSize,
    int channelsPerGroup,
    int groupSize,
  ) {
    for (int g = 0; g < numGroups; g++) {
      final groupStart = g * channelsPerGroup * spatialSize;

      // Compute mean using Welford's algorithm for numerical stability
      double mean = 0;
      int count = 0;
      for (int i = 0; i < groupSize; i++) {
        count++;
        mean += (data[groupStart + i] - mean) / count;
      }

      // Compute variance
      double variance = 0;
      for (int i = 0; i < groupSize; i++) {
        final diff = data[groupStart + i] - mean;
        variance += diff * diff;
      }
      variance /= groupSize;

      // Pre-compute scale factor
      final invStd = 1.0 / math.sqrt(variance + eps);

      // Normalize and apply affine transform
      for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
        final ch = g * channelsPerGroup + chInGroup;
        final channelOffset = ch * spatialSize;
        final w = weight?[ch] ?? 1.0;
        final b = bias?[ch] ?? 0.0;

        for (int i = 0; i < spatialSize; i++) {
          final idx = channelOffset + i;
          data[idx] = ((data[idx] - mean) * invStd) * w + b;
        }
      }
    }
  }

  void _groupNorm3DFloat64(
    Float64List data,
    int c,
    int spatialSize,
    int channelsPerGroup,
    int groupSize,
  ) {
    for (int g = 0; g < numGroups; g++) {
      final groupStart = g * channelsPerGroup * spatialSize;

      // Compute mean using Welford's algorithm
      double mean = 0;
      int count = 0;
      for (int i = 0; i < groupSize; i++) {
        count++;
        mean += (data[groupStart + i] - mean) / count;
      }

      // Compute variance
      double variance = 0;
      for (int i = 0; i < groupSize; i++) {
        final diff = data[groupStart + i] - mean;
        variance += diff * diff;
      }
      variance /= groupSize;

      // Pre-compute scale factor
      final invStd = 1.0 / math.sqrt(variance + eps);

      // Normalize and apply affine transform
      for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
        final ch = g * channelsPerGroup + chInGroup;
        final channelOffset = ch * spatialSize;
        final w = weight?[ch] ?? 1.0;
        final b = bias?[ch] ?? 0.0;

        for (int i = 0; i < spatialSize; i++) {
          final idx = channelOffset + i;
          data[idx] = ((data[idx] - mean) * invStd) * w + b;
        }
      }
    }
  }

  void _groupNorm3DGeneric(
    TensorBuffer tensor,
    int c,
    int spatialSize,
    int channelsPerGroup,
    int groupSize,
  ) {
    final storage = tensor.storage;

    for (int g = 0; g < numGroups; g++) {
      final groupStart = g * channelsPerGroup * spatialSize;

      // Compute mean
      double mean = 0;
      int count = 0;
      for (int i = 0; i < groupSize; i++) {
        count++;
        mean += (storage.getAsDouble(groupStart + i) - mean) / count;
      }

      // Compute variance
      double variance = 0;
      for (int i = 0; i < groupSize; i++) {
        final diff = storage.getAsDouble(groupStart + i) - mean;
        variance += diff * diff;
      }
      variance /= groupSize;

      final invStd = 1.0 / math.sqrt(variance + eps);

      // Normalize and apply affine transform
      for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
        final ch = g * channelsPerGroup + chInGroup;
        final channelOffset = ch * spatialSize;
        final w = weight?[ch] ?? 1.0;
        final b = bias?[ch] ?? 0.0;

        for (int i = 0; i < spatialSize; i++) {
          final idx = channelOffset + i;
          final normalized = (storage.getAsDouble(idx) - mean) * invStd;
          storage.setFromDouble(idx, normalized * w + b);
        }
      }
    }
  }

  void _groupNorm4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final spatialSize = h * w;
    final batchStride = c * spatialSize;
    final channelsPerGroup = c ~/ numGroups;
    final groupSize = channelsPerGroup * spatialSize;

    switch (tensor.dtype) {
      case DType.float32:
        _groupNorm4DFloat32(
          tensor.storage.data as Float32List,
          n,
          c,
          spatialSize,
          batchStride,
          channelsPerGroup,
          groupSize,
        );
      case DType.float64:
        _groupNorm4DFloat64(
          tensor.storage.data as Float64List,
          n,
          c,
          spatialSize,
          batchStride,
          channelsPerGroup,
          groupSize,
        );
      default:
        _groupNorm4DGeneric(
          tensor,
          n,
          c,
          spatialSize,
          batchStride,
          channelsPerGroup,
          groupSize,
        );
    }
  }

  void _groupNorm4DFloat32(
    Float32List data,
    int n,
    int c,
    int spatialSize,
    int batchStride,
    int channelsPerGroup,
    int groupSize,
  ) {
    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchStride;

      for (int g = 0; g < numGroups; g++) {
        final groupStart = batchOffset + g * channelsPerGroup * spatialSize;

        // Compute mean using Welford's algorithm
        double mean = 0;
        int count = 0;
        for (int i = 0; i < groupSize; i++) {
          count++;
          mean += (data[groupStart + i] - mean) / count;
        }

        // Compute variance
        double variance = 0;
        for (int i = 0; i < groupSize; i++) {
          final diff = data[groupStart + i] - mean;
          variance += diff * diff;
        }
        variance /= groupSize;

        final invStd = 1.0 / math.sqrt(variance + eps);

        // Normalize and apply affine transform
        for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
          final ch = g * channelsPerGroup + chInGroup;
          final channelOffset = batchOffset + ch * spatialSize;
          final w = weight?[ch] ?? 1.0;
          final b = bias?[ch] ?? 0.0;

          for (int i = 0; i < spatialSize; i++) {
            final idx = channelOffset + i;
            data[idx] = ((data[idx] - mean) * invStd) * w + b;
          }
        }
      }
    }
  }

  void _groupNorm4DFloat64(
    Float64List data,
    int n,
    int c,
    int spatialSize,
    int batchStride,
    int channelsPerGroup,
    int groupSize,
  ) {
    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchStride;

      for (int g = 0; g < numGroups; g++) {
        final groupStart = batchOffset + g * channelsPerGroup * spatialSize;

        // Compute mean using Welford's algorithm
        double mean = 0;
        int count = 0;
        for (int i = 0; i < groupSize; i++) {
          count++;
          mean += (data[groupStart + i] - mean) / count;
        }

        // Compute variance
        double variance = 0;
        for (int i = 0; i < groupSize; i++) {
          final diff = data[groupStart + i] - mean;
          variance += diff * diff;
        }
        variance /= groupSize;

        final invStd = 1.0 / math.sqrt(variance + eps);

        // Normalize and apply affine transform
        for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
          final ch = g * channelsPerGroup + chInGroup;
          final channelOffset = batchOffset + ch * spatialSize;
          final w = weight?[ch] ?? 1.0;
          final b = bias?[ch] ?? 0.0;

          for (int i = 0; i < spatialSize; i++) {
            final idx = channelOffset + i;
            data[idx] = ((data[idx] - mean) * invStd) * w + b;
          }
        }
      }
    }
  }

  void _groupNorm4DGeneric(
    TensorBuffer tensor,
    int n,
    int c,
    int spatialSize,
    int batchStride,
    int channelsPerGroup,
    int groupSize,
  ) {
    final storage = tensor.storage;

    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchStride;

      for (int g = 0; g < numGroups; g++) {
        final groupStart = batchOffset + g * channelsPerGroup * spatialSize;

        // Compute mean
        double mean = 0;
        int count = 0;
        for (int i = 0; i < groupSize; i++) {
          count++;
          mean += (storage.getAsDouble(groupStart + i) - mean) / count;
        }

        // Compute variance
        double variance = 0;
        for (int i = 0; i < groupSize; i++) {
          final diff = storage.getAsDouble(groupStart + i) - mean;
          variance += diff * diff;
        }
        variance /= groupSize;

        final invStd = 1.0 / math.sqrt(variance + eps);

        // Normalize and apply affine transform
        for (int chInGroup = 0; chInGroup < channelsPerGroup; chInGroup++) {
          final ch = g * channelsPerGroup + chInGroup;
          final channelOffset = batchOffset + ch * spatialSize;
          final w = weight?[ch] ?? 1.0;
          final b = bias?[ch] ?? 0.0;

          for (int i = 0; i < spatialSize; i++) {
            final idx = channelOffset + i;
            final normalized = (storage.getAsDouble(idx) - mean) * invStd;
            storage.setFromDouble(idx, normalized * w + b);
          }
        }
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
