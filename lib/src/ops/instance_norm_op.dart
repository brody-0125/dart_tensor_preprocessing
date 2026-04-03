import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Instance Normalization for inference.
///
/// Applies the formula: `y = (x - mean) / sqrt(var + eps) * weight + bias`
/// where mean and variance are computed per sample per channel.
///
/// Unlike BatchNorm which uses running statistics and GroupNorm which groups channels,
/// InstanceNorm computes statistics independently for each (sample, channel) pair.
/// This makes it ideal for style transfer and GANs where batch statistics are not meaningful.
///
/// Equivalent to `torch.nn.InstanceNorm2d` in PyTorch.
///
/// ## Complexity
///
/// Let `n` = total elements, `HW` = spatial size per channel.
///
/// - **Time**: O(n) with 2 passes per (sample, channel): compute stats, normalize.
/// - **Space**: O(n) for output tensor (or O(1) with in-place mode).
///
/// ## Example
///
/// ```dart
/// // For style transfer with 64 channels
/// final instanceNorm = InstanceNormOp(
///   numFeatures: 64,
///   weight: gammaWeights,
///   bias: betaWeights,
/// );
/// final result = instanceNorm(tensor);
/// ```
class InstanceNormOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Number of channels (features) expected in the input.
  final int numFeatures;

  /// Optional learned scale parameter (gamma/weight).
  /// If null, no affine transformation is applied (weight=1).
  final List<double>? weight;

  /// Optional learned bias parameter (beta/bias).
  /// If null, no affine transformation is applied (bias=0).
  final List<double>? bias;

  /// Small constant for numerical stability.
  final double eps;

  /// Creates an instance normalization operation.
  ///
  /// - [numFeatures]: Number of channels expected in input
  /// - [weight]: Optional per-channel scale (gamma), defaults to 1.0
  /// - [bias]: Optional per-channel bias (beta), defaults to 0.0
  /// - [eps]: Small constant to avoid division by zero (default: 1e-5)
  InstanceNormOp({
    required this.numFeatures,
    this.weight,
    this.bias,
    this.eps = 1e-5,
  }) {
    _validateParameters();
  }

  /// Creates an instance normalization from PyTorch state_dict Float32Lists.
  factory InstanceNormOp.fromStateDict({
    required int numFeatures,
    Float32List? weight,
    Float32List? bias,
    double eps = 1e-5,
  }) {
    return InstanceNormOp(
      numFeatures: numFeatures,
      weight: weight?.toList(),
      bias: bias?.toList(),
      eps: eps,
    );
  }

  void _validateParameters() {
    if (numFeatures <= 0) {
      throw InvalidParameterException(
        'numFeatures',
        numFeatures.toString(),
        'Must be positive',
      );
    }

    if (weight != null && weight!.length != numFeatures) {
      throw InvalidParameterException(
        'weight',
        'length ${weight!.length}',
        'Must match numFeatures ($numFeatures)',
      );
    }

    if (bias != null && bias!.length != numFeatures) {
      throw InvalidParameterException(
        'bias',
        'length ${bias!.length}',
        'Must match numFeatures ($numFeatures)',
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

  @override
  String get name => 'InstanceNorm(features=$numFeatures, eps=$eps)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    supportsInPlace: true,
    requiresContiguous: true,
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);
    final output = cloneForModification(input);
    _instanceNorm(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('InstanceNormOp.applyInPlace');
    }
    _validateShape(input.shape);
    _instanceNorm(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'InstanceNormOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != numFeatures) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but InstanceNormOp expects $numFeatures',
      );
    }
  }

  void _instanceNorm(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank == 3) {
      _instanceNorm3D(tensor);
    } else {
      _instanceNorm4D(tensor);
    }
  }

  void _instanceNorm3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final spatialSize = h * w;
    final data = tensor.storage.data;

    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * spatialSize;
          _normalizeSliceFloat32(list, offset, spatialSize, ch);
        }
      case DType.float64:
        final list = data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * spatialSize;
          _normalizeSliceFloat64(list, offset, spatialSize, ch);
        }
      default:
        for (int ch = 0; ch < c; ch++) {
          final offset = ch * spatialSize;
          _normalizeSliceGeneric(tensor, offset, spatialSize, ch);
        }
    }
  }

  void _instanceNorm4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final spatialSize = h * w;
    final channelSize = c * spatialSize;
    final data = tensor.storage.data;

    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * channelSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * spatialSize;
            _normalizeSliceFloat32(list, offset, spatialSize, ch);
          }
        }
      case DType.float64:
        final list = data as Float64List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * channelSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * spatialSize;
            _normalizeSliceFloat64(list, offset, spatialSize, ch);
          }
        }
      default:
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * channelSize;
          for (int ch = 0; ch < c; ch++) {
            final offset = batchOffset + ch * spatialSize;
            _normalizeSliceGeneric(tensor, offset, spatialSize, ch);
          }
        }
    }
  }

  void _normalizeSliceFloat32(
    Float32List data,
    int offset,
    int size,
    int channel,
  ) {
    // Welford's algorithm for numerically stable mean and variance
    double mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < size; i++) {
      final x = data[offset + i];
      final delta = x - mean;
      mean += delta / (i + 1);
      m2 += delta * (x - mean);
    }
    final variance = size > 1 ? m2 / size : 0.0;

    // Normalize with optional affine parameters
    final invStd = 1.0 / math.sqrt(variance + eps);
    final w = weight?[channel] ?? 1.0;
    final b = bias?[channel] ?? 0.0;

    for (int i = 0; i < size; i++) {
      final idx = offset + i;
      final normalized = (data[idx] - mean) * invStd;
      data[idx] = normalized * w + b;
    }
  }

  void _normalizeSliceFloat64(
    Float64List data,
    int offset,
    int size,
    int channel,
  ) {
    // Welford's algorithm for numerically stable mean and variance
    double mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < size; i++) {
      final x = data[offset + i];
      final delta = x - mean;
      mean += delta / (i + 1);
      m2 += delta * (x - mean);
    }
    final variance = size > 1 ? m2 / size : 0.0;

    // Normalize with optional affine parameters
    final invStd = 1.0 / math.sqrt(variance + eps);
    final w = weight?[channel] ?? 1.0;
    final b = bias?[channel] ?? 0.0;

    for (int i = 0; i < size; i++) {
      final idx = offset + i;
      final normalized = (data[idx] - mean) * invStd;
      data[idx] = normalized * w + b;
    }
  }

  void _normalizeSliceGeneric(
    TensorBuffer tensor,
    int offset,
    int size,
    int channel,
  ) {
    // Welford's algorithm for numerically stable mean and variance
    double mean = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < size; i++) {
      final x = tensor.storage.getAsDouble(offset + i);
      final delta = x - mean;
      mean += delta / (i + 1);
      m2 += delta * (x - mean);
    }
    final variance = size > 1 ? m2 / size : 0.0;

    // Normalize with optional affine parameters
    final invStd = 1.0 / math.sqrt(variance + eps);
    final w = weight?[channel] ?? 1.0;
    final b = bias?[channel] ?? 0.0;

    for (int i = 0; i < size; i++) {
      final idx = offset + i;
      final value = tensor.storage.getAsDouble(idx);
      final normalized = (value - mean) * invStd;
      tensor.storage.setFromDouble(idx, normalized * w + b);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
