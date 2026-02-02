import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Layer Normalization for inference.
///
/// Applies the formula: `y = (x - mean) / sqrt(var + eps) * weight + bias`
/// where mean and variance are computed over the normalized dimensions.
///
/// This operation normalizes across feature dimensions (not batch), making it
/// essential for Transformer models like BERT, ViT, and GPT.
///
/// Equivalent to `torch.nn.LayerNorm` in PyTorch.
///
/// ## Complexity
///
/// Let `n` = total elements, `k` = normalizedShape product (e.g., 768 for BERT).
///
/// - **Time**: O(n) with 3 passes per normalized slice: mean, variance (Welford's), normalize.
/// - **Space**: O(n) for output tensor (or O(1) with in-place mode).
///
/// ## Example
///
/// ```dart
/// // For BERT-style transformer with hidden size 768
/// final layerNorm = LayerNormOp(
///   normalizedShape: [768],
///   weight: gammaWeights,
///   bias: betaWeights,
/// );
/// final result = layerNorm(tensor);
/// ```
class LayerNormOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Shape of the normalized dimensions (e.g., [768] for BERT).
  ///
  /// Normalization is performed over the last N dimensions where
  /// N = normalizedShape.length.
  final List<int> normalizedShape;

  /// Optional learned scale parameter (gamma/weight).
  /// If null, defaults to 1.0 for all elements.
  final List<double>? weight;

  /// Optional learned bias parameter (beta/bias).
  /// If null, defaults to 0.0 for all elements.
  final List<double>? bias;

  /// Small constant for numerical stability.
  final double eps;

  /// Number of elements in the normalized dimensions.
  late final int _normalizedSize;

  /// Creates a layer normalization operation.
  ///
  /// - [normalizedShape]: Shape of dimensions to normalize over (from right)
  /// - [weight]: Optional per-element scale (gamma), defaults to 1.0
  /// - [bias]: Optional per-element bias (beta), defaults to 0.0
  /// - [eps]: Small constant to avoid division by zero (default: 1e-5)
  LayerNormOp({
    required this.normalizedShape,
    this.weight,
    this.bias,
    this.eps = 1e-5,
  }) {
    _validateParameters();
    _normalizedSize = normalizedShape.fold(1, (a, b) => a * b);
  }

  /// Creates a layer normalization for BERT-base (hidden size 768).
  factory LayerNormOp.bert({
    required List<double> weight,
    required List<double> bias,
    double eps = 1e-5,
  }) {
    return LayerNormOp(
      normalizedShape: [768],
      weight: weight,
      bias: bias,
      eps: eps,
    );
  }

  /// Creates a layer normalization for BERT-large (hidden size 1024).
  factory LayerNormOp.bertLarge({
    required List<double> weight,
    required List<double> bias,
    double eps = 1e-5,
  }) {
    return LayerNormOp(
      normalizedShape: [1024],
      weight: weight,
      bias: bias,
      eps: eps,
    );
  }

  /// Creates a layer normalization from PyTorch state_dict Float32Lists.
  factory LayerNormOp.fromStateDict({
    required List<int> normalizedShape,
    Float32List? weight,
    Float32List? bias,
    double eps = 1e-5,
  }) {
    return LayerNormOp(
      normalizedShape: normalizedShape,
      weight: weight?.toList(),
      bias: bias?.toList(),
      eps: eps,
    );
  }

  void _validateParameters() {
    if (normalizedShape.isEmpty) {
      throw InvalidParameterException(
        'normalizedShape',
        'empty',
        'LayerNormOp requires at least one dimension',
      );
    }

    for (int i = 0; i < normalizedShape.length; i++) {
      if (normalizedShape[i] <= 0) {
        throw InvalidParameterException(
          'normalizedShape[$i]',
          normalizedShape[i].toString(),
          'Must be positive',
        );
      }
    }

    final expectedSize = normalizedShape.fold(1, (a, b) => a * b);

    if (weight != null && weight!.length != expectedSize) {
      throw InvalidParameterException(
        'weight',
        'length ${weight!.length}',
        'Must match normalizedShape product ($expectedSize)',
      );
    }

    if (bias != null && bias!.length != expectedSize) {
      throw InvalidParameterException(
        'bias',
        'length ${bias!.length}',
        'Must match normalizedShape product ($expectedSize)',
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

  /// Number of elements in the normalized shape.
  int get normalizedSize => _normalizedSize;

  @override
  String get name => 'LayerNorm(shape=$normalizedShape, eps=$eps)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);
    final output = cloneForModification(input);
    _layerNorm(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('LayerNormOp.applyInPlace');
    }
    _validateShape(input.shape);
    _layerNorm(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    final normDims = normalizedShape.length;

    if (rank < normDims) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Input rank ($rank) must be >= normalizedShape length ($normDims)',
      );
    }

    // Check that trailing dimensions match normalizedShape
    for (int i = 0; i < normDims; i++) {
      final inputDimIdx = rank - normDims + i;
      if (shape[inputDimIdx] != normalizedShape[i]) {
        throw ShapeMismatchException(
          actual: shape,
          message:
              'Input shape trailing dimensions must match normalizedShape. '
              'Expected [..., ${normalizedShape.join(', ')}], got [..., ${shape.sublist(rank - normDims).join(', ')}]',
        );
      }
    }
  }

  void _layerNorm(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;
    final normDims = normalizedShape.length;

    // Number of instances to normalize (product of leading dimensions)
    int numInstances = 1;
    for (int i = 0; i < rank - normDims; i++) {
      numInstances *= shape[i];
    }

    final data = tensor.storage.data;

    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        _layerNormFloat32(list, numInstances);
      case DType.float64:
        final list = data as Float64List;
        _layerNormFloat64(list, numInstances);
      default:
        _layerNormGeneric(tensor, numInstances);
    }
  }

  void _layerNormFloat32(Float32List data, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Welford's algorithm for numerically stable mean and variance
      double mean = 0.0;
      double m2 = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = data[offset + i];
        final delta = x - mean;
        mean += delta / (i + 1);
        m2 += delta * (x - mean);
      }
      final variance = _normalizedSize > 1 ? m2 / _normalizedSize : 0.0;

      // Normalize
      final invStd = 1.0 / math.sqrt(variance + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final normalized = (data[idx] - mean) * invStd;
        final w = weight?[i] ?? 1.0;
        final b = bias?[i] ?? 0.0;
        data[idx] = normalized * w + b;
      }
    }
  }

  void _layerNormFloat64(Float64List data, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Welford's algorithm for numerically stable mean and variance
      double mean = 0.0;
      double m2 = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = data[offset + i];
        final delta = x - mean;
        mean += delta / (i + 1);
        m2 += delta * (x - mean);
      }
      final variance = _normalizedSize > 1 ? m2 / _normalizedSize : 0.0;

      // Normalize
      final invStd = 1.0 / math.sqrt(variance + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final normalized = (data[idx] - mean) * invStd;
        final w = weight?[i] ?? 1.0;
        final b = bias?[i] ?? 0.0;
        data[idx] = normalized * w + b;
      }
    }
  }

  void _layerNormGeneric(TensorBuffer tensor, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Welford's algorithm for numerically stable mean and variance
      double mean = 0.0;
      double m2 = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = tensor.storage.getAsDouble(offset + i);
        final delta = x - mean;
        mean += delta / (i + 1);
        m2 += delta * (x - mean);
      }
      final variance = _normalizedSize > 1 ? m2 / _normalizedSize : 0.0;

      // Normalize
      final invStd = 1.0 / math.sqrt(variance + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final value = tensor.storage.getAsDouble(idx);
        final normalized = (value - mean) * invStd;
        final w = weight?[i] ?? 1.0;
        final b = bias?[i] ?? 0.0;
        tensor.storage.setFromDouble(idx, normalized * w + b);
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
