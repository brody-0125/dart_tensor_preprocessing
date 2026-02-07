import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Root Mean Square Normalization for inference.
///
/// Applies the formula: `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// RMSNorm is more efficient than LayerNorm as it does not compute or subtract the mean.
/// It is used in modern LLMs like LLaMA, Gemma, and other models based on the
/// "Root Mean Square Layer Normalization" paper.
///
/// Equivalent to `torch.nn.RMSNorm` (PyTorch 2.4+) or the custom implementation in LLaMA.
///
/// ## Complexity
///
/// Let `n` = total elements, `k` = normalizedShape product.
///
/// - **Time**: O(n) with 2 passes per normalized slice: compute RMS, scale.
/// - **Space**: O(n) for output tensor (or O(1) with in-place mode).
///
/// ## Example
///
/// ```dart
/// // For LLaMA with hidden size 4096
/// final rmsNorm = RMSNormOp(
///   normalizedShape: [4096],
///   weight: gammaWeights,
/// );
/// final result = rmsNorm(tensor);
/// ```
class RMSNormOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Shape of the normalized dimensions (e.g., [4096] for LLaMA).
  ///
  /// Normalization is performed over the last N dimensions where
  /// N = normalizedShape.length.
  final List<int> normalizedShape;

  /// Optional learned scale parameter (weight).
  /// If null, defaults to 1.0 for all elements.
  final List<double>? weight;

  /// Small constant for numerical stability.
  final double eps;

  /// Number of elements in the normalized dimensions.
  late final int _normalizedSize;

  /// Creates a RMS normalization operation.
  ///
  /// - [normalizedShape]: Shape of dimensions to normalize over (from right)
  /// - [weight]: Optional per-element scale, defaults to 1.0
  /// - [eps]: Small constant to avoid division by zero (default: 1e-6, LLaMA default)
  RMSNormOp({
    required this.normalizedShape,
    this.weight,
    this.eps = 1e-6,
  }) {
    _validateParameters();
    _normalizedSize = normalizedShape.fold(1, (a, b) => a * b);
  }

  /// Creates a RMS normalization for LLaMA 7B (hidden size 4096).
  factory RMSNormOp.llama7B({
    required List<double> weight,
    double eps = 1e-6,
  }) {
    return RMSNormOp(
      normalizedShape: [4096],
      weight: weight,
      eps: eps,
    );
  }

  /// Creates a RMS normalization for LLaMA 13B (hidden size 5120).
  factory RMSNormOp.llama13B({
    required List<double> weight,
    double eps = 1e-6,
  }) {
    return RMSNormOp(
      normalizedShape: [5120],
      weight: weight,
      eps: eps,
    );
  }

  /// Creates a RMS normalization for LLaMA 70B (hidden size 8192).
  factory RMSNormOp.llama70B({
    required List<double> weight,
    double eps = 1e-6,
  }) {
    return RMSNormOp(
      normalizedShape: [8192],
      weight: weight,
      eps: eps,
    );
  }

  /// Creates a RMS normalization for Gemma 2B (hidden size 2048).
  factory RMSNormOp.gemma2B({
    required List<double> weight,
    double eps = 1e-6,
  }) {
    return RMSNormOp(
      normalizedShape: [2048],
      weight: weight,
      eps: eps,
    );
  }

  /// Creates a RMS normalization from PyTorch state_dict Float32Lists.
  factory RMSNormOp.fromStateDict({
    required List<int> normalizedShape,
    Float32List? weight,
    double eps = 1e-6,
  }) {
    return RMSNormOp(
      normalizedShape: normalizedShape,
      weight: weight?.toList(),
      eps: eps,
    );
  }

  void _validateParameters() {
    if (normalizedShape.isEmpty) {
      throw InvalidParameterException(
        'normalizedShape',
        'empty',
        'RMSNormOp requires at least one dimension',
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
  String get name => 'RMSNorm(shape=$normalizedShape, eps=$eps)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateShape(input.shape);
    final output = cloneForModification(input);
    _rmsNorm(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('RMSNormOp.applyInPlace');
    }
    _validateShape(input.shape);
    _rmsNorm(input);
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

  void _rmsNorm(TensorBuffer tensor) {
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
        _rmsNormFloat32(list, numInstances);
      case DType.float64:
        final list = data as Float64List;
        _rmsNormFloat64(list, numInstances);
      default:
        _rmsNormGeneric(tensor, numInstances);
    }
  }

  void _rmsNormFloat32(Float32List data, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Compute mean of squares
      double sumSquares = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = data[offset + i];
        sumSquares += x * x;
      }
      final meanSquare = sumSquares / _normalizedSize;

      // Normalize: x / sqrt(mean(x^2) + eps) * weight
      final invRms = 1.0 / math.sqrt(meanSquare + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final w = weight?[i] ?? 1.0;
        data[idx] = data[idx] * invRms * w;
      }
    }
  }

  void _rmsNormFloat64(Float64List data, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Compute mean of squares
      double sumSquares = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = data[offset + i];
        sumSquares += x * x;
      }
      final meanSquare = sumSquares / _normalizedSize;

      // Normalize: x / sqrt(mean(x^2) + eps) * weight
      final invRms = 1.0 / math.sqrt(meanSquare + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final w = weight?[i] ?? 1.0;
        data[idx] = data[idx] * invRms * w;
      }
    }
  }

  void _rmsNormGeneric(TensorBuffer tensor, int numInstances) {
    for (int instance = 0; instance < numInstances; instance++) {
      final offset = instance * _normalizedSize;

      // Compute mean of squares
      double sumSquares = 0.0;
      for (int i = 0; i < _normalizedSize; i++) {
        final x = tensor.storage.getAsDouble(offset + i);
        sumSquares += x * x;
      }
      final meanSquare = sumSquares / _normalizedSize;

      // Normalize: x / sqrt(mean(x^2) + eps) * weight
      final invRms = 1.0 / math.sqrt(meanSquare + eps);
      for (int i = 0; i < _normalizedSize; i++) {
        final idx = offset + i;
        final value = tensor.storage.getAsDouble(idx);
        final w = weight?[i] ?? 1.0;
        tensor.storage.setFromDouble(idx, value * invRms * w);
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
