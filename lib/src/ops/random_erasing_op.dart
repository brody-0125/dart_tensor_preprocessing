import 'dart:math';
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Randomly erases a rectangular region of a tensor image for data augmentation.
///
/// This implements the "Random Erasing Data Augmentation" technique (Zhong et al., 2020),
/// which randomly selects a rectangle region in an image and erases its pixels with
/// a constant or random value. This serves as a regularization method that enhances
/// model robustness for object detection and recognition tasks.
///
/// Equivalent to PyTorch's `torchvision.transforms.RandomErasing`.
///
/// ## Algorithm
///
/// 1. With probability [probability], attempt to erase a region:
///    - Sample area fraction from [scaleRange]
///    - Sample aspect ratio from [ratioRange] (log-uniform)
///    - Compute rectangle dimensions and random placement
///    - Fill erased region with [value] (constant) or random values
/// 2. If no valid rectangle is found after 10 attempts, return input unchanged.
///
/// ## Example
///
/// ```dart
/// final erase = RandomErasingOp(
///   probability: 0.5,
///   scaleRange: (0.02, 0.33),
///   ratioRange: (0.3, 3.3),
///   value: 0.0,
/// );
///
/// final augmented = erase(tensor); // 3D [C,H,W] or 4D [N,C,H,W]
/// ```
class RandomErasingOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Probability that the random erasing operation will be performed.
  ///
  /// Must be in range [0.0, 1.0].
  final double probability;

  /// Range of proportion of erased area against input image area.
  ///
  /// Must satisfy `0 < min <= max <= 1`.
  final (double, double) scaleRange;

  /// Range of aspect ratio of erased area.
  ///
  /// Must satisfy `0 < min <= max`. Sampled log-uniformly.
  final (double, double) ratioRange;

  /// Fill value for erased region. If `null`, fills with random values.
  final double? value;

  /// Optional random seed for reproducibility.
  final int? seed;

  /// Maximum number of attempts to find a valid erasing rectangle.
  static const int _maxAttempts = 10;

  /// Creates a random erasing operation.
  ///
  /// - [probability]: Probability of applying the erasure (default 0.5).
  /// - [scaleRange]: Range of area fraction to erase (default (0.02, 0.33)).
  /// - [ratioRange]: Range of aspect ratio for the erased rectangle (default (0.3, 3.3)).
  /// - [value]: Constant fill value for the erased region. If `null`, fills with random values.
  /// - [seed]: Optional random seed for reproducibility.
  RandomErasingOp({
    this.probability = 0.5,
    this.scaleRange = const (0.02, 0.33),
    this.ratioRange = const (0.3, 3.3),
    this.value = 0.0,
    this.seed,
  }) {
    if (probability < 0.0 || probability > 1.0) {
      throw InvalidParameterException(
        'probability',
        probability,
        'Must be in range [0.0, 1.0]',
      );
    }
    if (scaleRange.$1 <= 0 || scaleRange.$1 > scaleRange.$2) {
      throw InvalidParameterException(
        'scaleRange',
        scaleRange,
        'Must satisfy 0 < min <= max',
      );
    }
    if (scaleRange.$2 > 1.0) {
      throw InvalidParameterException(
        'scaleRange',
        scaleRange,
        'Max scale must be <= 1.0',
      );
    }
    if (ratioRange.$1 <= 0 || ratioRange.$1 > ratioRange.$2) {
      throw InvalidParameterException(
        'ratioRange',
        ratioRange,
        'Must satisfy 0 < min <= max',
      );
    }
  }

  @override
  String get name => 'RandomErasing('
      'p=$probability, '
      'scale=$scaleRange, '
      'ratio=$ratioRange, '
      'value=$value)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torchvision.transforms.RandomErasing',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _eraseImpl(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('RandomErasingOp.applyInPlace');
    }
    _eraseImpl(input);
  }

  void _eraseImpl(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'RandomErasingOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final random = Random(seed);

    if (rank == 3) {
      _eraseImage(tensor, shape[0], shape[1], shape[2], 0, random);
    } else {
      // 4D: apply independently to each image in the batch
      final n = shape[0];
      final c = shape[1];
      final h = shape[2];
      final w = shape[3];
      final imageSize = c * h * w;
      for (int batch = 0; batch < n; batch++) {
        _eraseImage(tensor, c, h, w, batch * imageSize, random);
      }
    }
  }

  /// Attempts to erase a random rectangle from a single image.
  ///
  /// [tensor] is the tensor to modify in place.
  /// [c], [h], [w] are the channel, height, width dimensions.
  /// [baseOffset] is the flat index offset to the start of this image.
  /// [random] is the random number generator.
  void _eraseImage(
    TensorBuffer tensor,
    int c,
    int h,
    int w,
    int baseOffset,
    Random random,
  ) {
    // Check probability
    if (random.nextDouble() >= probability) {
      return;
    }

    final imageArea = h * w;
    final logRatioMin = log(ratioRange.$1);
    final logRatioMax = log(ratioRange.$2);

    for (int attempt = 0; attempt < _maxAttempts; attempt++) {
      // Sample area fraction and aspect ratio
      final scaleFraction =
          scaleRange.$1 + random.nextDouble() * (scaleRange.$2 - scaleRange.$1);
      final targetArea = (imageArea * scaleFraction).round();

      final logRatio =
          logRatioMin + random.nextDouble() * (logRatioMax - logRatioMin);
      final aspectRatio = exp(logRatio);

      final eraseH = sqrt(targetArea * aspectRatio).round();
      final eraseW = sqrt(targetArea / aspectRatio).round();

      // Check if the rectangle fits
      if (eraseH <= 0 || eraseW <= 0 || eraseH > h || eraseW > w) {
        continue;
      }

      // Random placement
      final startH = random.nextInt(h - eraseH + 1);
      final startW = random.nextInt(w - eraseW + 1);

      // Fill the erased region
      _fillRegion(
          tensor, c, h, w, baseOffset, startH, startW, eraseH, eraseW, random);
      return;
    }
    // If no valid rectangle found after maxAttempts, do nothing
  }

  /// Fills the specified rectangular region with constant or random values.
  void _fillRegion(
    TensorBuffer tensor,
    int c,
    int h,
    int w,
    int baseOffset,
    int startH,
    int startW,
    int eraseH,
    int eraseW,
    Random random,
  ) {
    final fillValue = value;

    switch (tensor.dtype) {
      case DType.float32:
        final data = tensor.storage.data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = baseOffset + ch * h * w;
          for (int row = 0; row < eraseH; row++) {
            final rowOffset = chOffset + (startH + row) * w + startW;
            for (int col = 0; col < eraseW; col++) {
              data[rowOffset + col] =
                  fillValue ?? random.nextDouble();
            }
          }
        }
      case DType.float64:
        final data = tensor.storage.data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = baseOffset + ch * h * w;
          for (int row = 0; row < eraseH; row++) {
            final rowOffset = chOffset + (startH + row) * w + startW;
            for (int col = 0; col < eraseW; col++) {
              data[rowOffset + col] =
                  fillValue ?? random.nextDouble();
            }
          }
        }
      default:
        // Generic fallback for integer types
        for (int ch = 0; ch < c; ch++) {
          final chOffset = baseOffset + ch * h * w;
          for (int row = 0; row < eraseH; row++) {
            final rowOffset = chOffset + (startH + row) * w + startW;
            for (int col = 0; col < eraseW; col++) {
              final v = fillValue ?? random.nextDouble();
              tensor.storage.setFromDouble(rowOffset + col, v);
            }
          }
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
