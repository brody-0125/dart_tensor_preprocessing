import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Batch Normalization for inference.
///
/// Applies the formula: `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`
///
/// This operation uses running statistics (computed during training) to normalize
/// activations. It is essential for running pretrained CNN models like ResNet, VGG,
/// and MobileNet.
///
/// Equivalent to `torch.nn.BatchNorm2d` in PyTorch (inference mode).
///
/// ## Example
///
/// ```dart
/// final batchNorm = BatchNormOp(
///   runningMean: [0.5, 0.5, 0.5],
///   runningVar: [0.25, 0.25, 0.25],
///   weight: [1.0, 1.0, 1.0],
///   bias: [0.0, 0.0, 0.0],
/// );
/// final result = batchNorm(tensor);
/// ```
class BatchNormOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Learned per-channel mean from training (running_mean).
  final List<double> runningMean;

  /// Learned per-channel variance from training (running_var).
  final List<double> runningVar;

  /// Optional learned scale parameter (gamma/weight).
  /// If null, defaults to 1.0 for all channels.
  final List<double>? weight;

  /// Optional learned bias parameter (beta/bias).
  /// If null, defaults to 0.0 for all channels.
  final List<double>? bias;

  /// Small constant for numerical stability.
  final double eps;

  /// Pre-computed scale: weight / sqrt(running_var + eps)
  late final List<double> _scale;

  /// Pre-computed shift: bias - running_mean * scale
  late final List<double> _shift;

  /// Creates a batch normalization operation.
  ///
  /// - [runningMean]: Per-channel mean values from training
  /// - [runningVar]: Per-channel variance values from training
  /// - [weight]: Optional per-channel scale (gamma), defaults to 1.0
  /// - [bias]: Optional per-channel bias (beta), defaults to 0.0
  /// - [eps]: Small constant to avoid division by zero (default: 1e-5)
  BatchNormOp({
    required this.runningMean,
    required this.runningVar,
    this.weight,
    this.bias,
    this.eps = 1e-5,
  }) {
    _validateParameters();
    _precomputeCoefficients();
  }

  /// Creates a batch normalization from PyTorch state_dict Float32Lists.
  factory BatchNormOp.fromStateDict({
    required Float32List runningMean,
    required Float32List runningVar,
    Float32List? weight,
    Float32List? bias,
    double eps = 1e-5,
  }) {
    return BatchNormOp(
      runningMean: runningMean.toList(),
      runningVar: runningVar.toList(),
      weight: weight?.toList(),
      bias: bias?.toList(),
      eps: eps,
    );
  }

  void _validateParameters() {
    final numChannels = runningMean.length;

    if (numChannels == 0) {
      throw InvalidParameterException(
        'runningMean',
        'empty',
        'BatchNormOp requires at least one channel',
      );
    }

    if (runningVar.length != numChannels) {
      throw InvalidParameterException(
        'runningVar',
        'length ${runningVar.length}',
        'Must match runningMean length ($numChannels)',
      );
    }

    if (weight != null && weight!.length != numChannels) {
      throw InvalidParameterException(
        'weight',
        'length ${weight!.length}',
        'Must match runningMean length ($numChannels)',
      );
    }

    if (bias != null && bias!.length != numChannels) {
      throw InvalidParameterException(
        'bias',
        'length ${bias!.length}',
        'Must match runningMean length ($numChannels)',
      );
    }

    if (eps <= 0) {
      throw InvalidParameterException(
        'eps',
        eps.toString(),
        'Must be positive',
      );
    }

    // Check for negative variance
    for (int i = 0; i < numChannels; i++) {
      if (runningVar[i] < 0) {
        throw InvalidParameterException(
          'runningVar[$i]',
          runningVar[i].toString(),
          'Variance cannot be negative',
        );
      }
    }
  }

  /// Pre-computes scale and shift for efficient inference.
  ///
  /// y = (x - mean) / sqrt(var + eps) * w + b
  ///   = x * (w / sqrt(var + eps)) + (b - mean * w / sqrt(var + eps))
  ///   = x * scale + shift
  void _precomputeCoefficients() {
    final numChannels = runningMean.length;
    _scale = List<double>.filled(numChannels, 0.0);
    _shift = List<double>.filled(numChannels, 0.0);

    for (int i = 0; i < numChannels; i++) {
      final invStd = 1.0 / math.sqrt(runningVar[i] + eps);
      final w = weight?[i] ?? 1.0;
      final b = bias?[i] ?? 0.0;

      _scale[i] = w * invStd;
      _shift[i] = b - runningMean[i] * _scale[i];
    }
  }

  /// Number of channels this batch norm expects.
  int get numChannels => runningMean.length;

  @override
  String get name => 'BatchNorm(channels=$numChannels, eps=$eps)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);

    // Optimized: single copy regardless of contiguity
    final TensorBuffer output;
    if (input.isContiguous) {
      output = input.clone();
    } else {
      output = input.contiguous();
    }

    _batchNorm(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('BatchNormOp.applyInPlace');
    }
    _validateShape(input.shape);
    _batchNorm(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'BatchNormOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != numChannels) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but BatchNormOp expects $numChannels',
      );
    }
  }

  void _batchNorm(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank == 3) {
      _batchNorm3D(tensor);
    } else {
      _batchNorm4D(tensor);
    }
  }

  void _batchNorm3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;
    final data = tensor.storage.data;

    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final s = _scale[ch];
          final sh = _shift[ch];
          for (int i = 0; i < channelSize; i++) {
            final idx = offset + i;
            list[idx] = list[idx] * s + sh;
          }
        }
      case DType.float64:
        final list = data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final s = _scale[ch];
          final sh = _shift[ch];
          for (int i = 0; i < channelSize; i++) {
            final idx = offset + i;
            list[idx] = list[idx] * s + sh;
          }
        }
      default:
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * channelSize;
          final s = _scale[ch];
          final sh = _shift[ch];
          for (int i = 0; i < channelSize; i++) {
            final idx = offset + i;
            final value = tensor.storage.getAsDouble(idx);
            tensor.storage.setFromDouble(idx, value * s + sh);
          }
        }
    }
  }

  void _batchNorm4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = c * channelSize;
    final data = tensor.storage.data;

    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final s = _scale[ch];
            final sh = _shift[ch];
            for (int i = 0; i < channelSize; i++) {
              final idx = offset + i;
              list[idx] = list[idx] * s + sh;
            }
          }
        }
      case DType.float64:
        final list = data as Float64List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final s = _scale[ch];
            final sh = _shift[ch];
            for (int i = 0; i < channelSize; i++) {
              final idx = offset + i;
              list[idx] = list[idx] * s + sh;
            }
          }
        }
      default:
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * batchSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * channelSize;
            final s = _scale[ch];
            final sh = _shift[ch];
            for (int i = 0; i < channelSize; i++) {
              final idx = offset + i;
              final value = tensor.storage.getAsDouble(idx);
              tensor.storage.setFromDouble(idx, value * s + sh);
            }
          }
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
