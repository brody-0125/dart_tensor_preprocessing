import 'dart:math';
import 'dart:typed_data';

import '../core/buffer_pool.dart';
import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/index_utils.dart';
import 'transform_op.dart';

/// Randomly crops tensor to specified dimensions for data augmentation.
///
/// Similar to CenterCropOp but selects a random crop region.
/// Uses optional seed for reproducible results.
class RandomCropOp extends TransformOp with RequiresContiguous {
  /// Target crop height.
  final int height;

  /// Target crop width.
  final int width;

  /// Optional random seed for reproducibility.
  final int? seed;

  /// Creates a random crop operation.
  RandomCropOp({
    required this.height,
    required this.width,
    this.seed,
  }) {
    if (height <= 0 || width <= 0) {
      throw InvalidParameterException(
        'height/width',
        'height=$height, width=$width',
        'Height and width must be positive',
      );
    }
  }

  @override
  String get name => 'RandomCrop(height=$height, width=$width, seed=$seed)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: false,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    final random = Random(seed);

    // Determine crop region
    final inputShape = contiguous.shape;
    final (h, w) = inputShape.length == 3
        ? (inputShape[1], inputShape[2])
        : (inputShape[2], inputShape[3]);

    if (height > h || width > w) {
      throw InvalidParameterException(
        'crop size',
        'height=$height, width=$width, input=${h}x$w',
        'Crop size cannot exceed input dimensions',
      );
    }

    final maxStartH = h - height;
    final maxStartW = w - width;
    final startH = random.nextInt(maxStartH + 1);
    final startW = random.nextInt(maxStartW + 1);

    return _cropRegion(contiguous, startH, startW);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'RandomCropOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }
  }

  TensorBuffer _cropRegion(TensorBuffer input, int startH, int startW) {
    final inputShape = input.shape;

    if (inputShape.length == 3) {
      // 3D: [C, H, W]
      final (c, h, w) = (inputShape[0], inputShape[1], inputShape[2]);
      final output =
          TensorBuffer.uninitialized([c, height, width], dtype: input.dtype);

      for (int ch = 0; ch < c; ch++) {
        for (int row = 0; row < height; row++) {
          for (int col = 0; col < width; col++) {
            final inputIdx = ch * h * w + (startH + row) * w + (startW + col);
            final outputIdx = ch * height * width + row * width + col;
            final val = input.storage.getAsDouble(inputIdx);
            output.storage.setFromDouble(outputIdx, val);
          }
        }
      }

      return output;
    } else {
      // 4D: [N, C, H, W]
      final (n, c, h, w) =
          (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
      final output =
          TensorBuffer.uninitialized([n, c, height, width], dtype: input.dtype);

      for (int batch = 0; batch < n; batch++) {
        for (int ch = 0; ch < c; ch++) {
          for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
              final inputIdx = batch * c * h * w +
                  ch * h * w +
                  (startH + row) * w +
                  (startW + col);
              final outputIdx = batch * c * height * width +
                  ch * height * width +
                  row * width +
                  col;
              final val = input.storage.getAsDouble(inputIdx);
              output.storage.setFromDouble(outputIdx, val);
            }
          }
        }
      }

      return output;
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [inputShape[0], height, width]; // [C, H, W]
    } else {
      return [inputShape[0], inputShape[1], height, width]; // [N, C, H, W]
    }
  }
}

/// Applies Gaussian blur to tensor for data augmentation.
///
/// Uses separable Gaussian convolution for efficiency.
/// Kernel size must be odd and >= 1.
///
/// ## Complexity
///
/// Let `C` = channels, `H` = height, `W` = width, `k` = kernelSize.
///
/// - **Time**: O(C × H × W × k) using separable convolution (instead of O(C × H × W × k²)).
/// - **Space**: O(C × H × W) for output + O(H × W) temporary buffer from BufferPool.
class GaussianBlurOp extends TransformOp with RequiresContiguous {
  /// Kernel size for blur (must be odd).
  final int kernelSize;

  /// Standard deviation for Gaussian distribution.
  final double sigma;

  /// Creates a Gaussian blur operation.
  GaussianBlurOp({
    this.kernelSize = 3,
    double? sigma,
  }) : sigma = sigma ?? _defaultSigma(kernelSize) {
    if (kernelSize < 1 || kernelSize % 2 == 0) {
      throw InvalidParameterException(
        'kernelSize',
        kernelSize.toString(),
        'Kernel size must be odd and >= 1',
      );
    }
    if (this.sigma <= 0) {
      throw InvalidParameterException(
        'sigma',
        this.sigma.toString(),
        'Sigma must be positive',
      );
    }
  }

  /// Creates light blur (3x3, sigma=0.5).
  factory GaussianBlurOp.light() => GaussianBlurOp(kernelSize: 3, sigma: 0.5);

  /// Creates medium blur (5x5, sigma=1.0).
  factory GaussianBlurOp.medium() => GaussianBlurOp(kernelSize: 5, sigma: 1.0);

  /// Creates strong blur (7x7, sigma=2.0).
  factory GaussianBlurOp.strong() => GaussianBlurOp(kernelSize: 7, sigma: 2.0);

  @override
  String get name => 'GaussianBlur(kernelSize=$kernelSize, sigma=$sigma)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    // Pre-compute 1D Gaussian kernel
    final kernel = _computeGaussianKernel();

    // Apply separable convolution
    return _applySeparableBlur(contiguous, kernel);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'GaussianBlurOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }
  }

  static double _defaultSigma(int kernelSize) {
    return 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8;
  }

  List<double> _computeGaussianKernel() {
    final radius = kernelSize ~/ 2;
    final kernel = List<double>.filled(kernelSize, 0.0);
    double sum = 0.0;

    // Compute 1D Gaussian
    for (int i = 0; i < kernelSize; i++) {
      final x = (i - radius).toDouble();
      kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }

    // Normalize
    for (int i = 0; i < kernelSize; i++) {
      kernel[i] /= sum;
    }

    return kernel;
  }

  TensorBuffer _applySeparableBlur(TensorBuffer input, List<double> kernel) {
    final inputShape = input.shape;
    final radius = kernelSize ~/ 2;

    if (inputShape.length == 3) {
      // 3D: [C, H, W]
      final (c, h, w) = (inputShape[0], inputShape[1], inputShape[2]);
      final output = TensorBuffer.uninitialized([c, h, w], dtype: input.dtype);

      // Acquire pooled temp buffer (reused across channels)
      final temp = BufferPool.instance.acquireFloat64(h * w);
      try {
        // Dtype-specialized for hot path optimization
        switch (input.dtype) {
          case DType.float32:
            final inList = input.storage.data as Float32List;
            final outList = output.storage.data as Float32List;
            for (int ch = 0; ch < c; ch++) {
              final chOffset = ch * h * w;
              // Horizontal pass
              for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                  double sum = 0.0;
                  for (int k = 0; k < kernelSize; k++) {
                    final xi = reflectIndex(col + k - radius, w);
                    sum += inList[chOffset + row * w + xi] * kernel[k];
                  }
                  temp[row * w + col] = sum;
                }
              }
              // Vertical pass
              for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                  double sum = 0.0;
                  for (int k = 0; k < kernelSize; k++) {
                    final yi = reflectIndex(row + k - radius, h);
                    sum += temp[yi * w + col] * kernel[k];
                  }
                  outList[chOffset + row * w + col] = sum;
                }
              }
            }
          default:
            // Generic fallback
            for (int ch = 0; ch < c; ch++) {
              // Horizontal pass
              for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                  double sum = 0.0;
                  for (int k = 0; k < kernelSize; k++) {
                    final xi = reflectIndex(col + k - radius, w);
                    final inputIdx = ch * h * w + row * w + xi;
                    sum += input.storage.getAsDouble(inputIdx) * kernel[k];
                  }
                  temp[row * w + col] = sum;
                }
              }
              // Vertical pass
              for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                  double sum = 0.0;
                  for (int k = 0; k < kernelSize; k++) {
                    final yi = reflectIndex(row + k - radius, h);
                    sum += temp[yi * w + col] * kernel[k];
                  }
                  final outputIdx = ch * h * w + row * w + col;
                  output.storage.setFromDouble(outputIdx, sum);
                }
              }
            }
        }
      } finally {
        BufferPool.instance.release(temp);
      }
      return output;
    } else {
      // 4D: [N, C, H, W]
      final (n, c, h, w) =
          (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
      final output =
          TensorBuffer.uninitialized([n, c, h, w], dtype: input.dtype);

      // Acquire pooled temp buffer (reused across batches and channels)
      final temp = BufferPool.instance.acquireFloat64(h * w);
      try {
        // Dtype-specialized for hot path optimization
        switch (input.dtype) {
          case DType.float32:
            final inList = input.storage.data as Float32List;
            final outList = output.storage.data as Float32List;
            for (int batch = 0; batch < n; batch++) {
              final batchOffset = batch * c * h * w;
              for (int ch = 0; ch < c; ch++) {
                final chOffset = batchOffset + ch * h * w;
                // Horizontal pass
                for (int row = 0; row < h; row++) {
                  for (int col = 0; col < w; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < kernelSize; k++) {
                      final xi = reflectIndex(col + k - radius, w);
                      sum += inList[chOffset + row * w + xi] * kernel[k];
                    }
                    temp[row * w + col] = sum;
                  }
                }
                // Vertical pass
                for (int row = 0; row < h; row++) {
                  for (int col = 0; col < w; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < kernelSize; k++) {
                      final yi = reflectIndex(row + k - radius, h);
                      sum += temp[yi * w + col] * kernel[k];
                    }
                    outList[chOffset + row * w + col] = sum;
                  }
                }
              }
            }
          default:
            // Generic fallback
            for (int batch = 0; batch < n; batch++) {
              for (int ch = 0; ch < c; ch++) {
                // Horizontal pass
                for (int row = 0; row < h; row++) {
                  for (int col = 0; col < w; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < kernelSize; k++) {
                      final xi = reflectIndex(col + k - radius, w);
                      final inputIdx =
                          batch * c * h * w + ch * h * w + row * w + xi;
                      sum += input.storage.getAsDouble(inputIdx) * kernel[k];
                    }
                    temp[row * w + col] = sum;
                  }
                }
                // Vertical pass
                for (int row = 0; row < h; row++) {
                  for (int col = 0; col < w; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < kernelSize; k++) {
                      final yi = reflectIndex(row + k - radius, h);
                      sum += temp[yi * w + col] * kernel[k];
                    }
                    final outputIdx =
                        batch * c * h * w + ch * h * w + row * w + col;
                    output.storage.setFromDouble(outputIdx, sum);
                  }
                }
              }
            }
        }
      } finally {
        BufferPool.instance.release(temp);
      }
      return output;
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// Flip Operations
// ============================================================================

/// Deterministically flips a tensor left-to-right (horizontal flip).
///
/// Reverses the width dimension (last axis) of the tensor.
/// Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` tensor formats.
///
/// Equivalent to PyTorch's `torch.flip(tensor, dims=[-1])` or
/// `torchvision.transforms.functional.hflip`.
///
/// ## Complexity
///
/// Let `N` = batch, `C` = channels, `H` = height, `W` = width.
///
/// - **Time**: O(N × C × H × W)
/// - **Space**: O(N × C × H × W) for output buffer.
class HorizontalFlipOp extends TransformOp with RequiresContiguous {
  /// Creates a deterministic horizontal flip operation.
  HorizontalFlipOp();

  @override
  String get name => 'HorizontalFlip';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
        pytorchEquivalent: 'torchvision.transforms.functional.hflip',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);
    return _flipHorizontal(contiguous);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Deterministically flips a tensor top-to-bottom (vertical flip).
///
/// Reverses the height dimension (second-to-last axis) of the tensor.
/// Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` tensor formats.
///
/// Equivalent to PyTorch's `torch.flip(tensor, dims=[-2])` or
/// `torchvision.transforms.functional.vflip`.
///
/// ## Complexity
///
/// Let `N` = batch, `C` = channels, `H` = height, `W` = width.
///
/// - **Time**: O(N × C × H × W)
/// - **Space**: O(N × C × H × W) for output buffer.
class VerticalFlipOp extends TransformOp with RequiresContiguous {
  /// Creates a deterministic vertical flip operation.
  VerticalFlipOp();

  @override
  String get name => 'VerticalFlip';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
        pytorchEquivalent: 'torchvision.transforms.functional.vflip',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);
    return _flipVertical(contiguous);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Randomly flips a tensor left-to-right with configurable probability.
///
/// One of the most fundamental data augmentation techniques in computer
/// vision. Each call to [apply] independently decides whether to flip
/// based on [probability].
///
/// Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` tensor formats.
///
/// Equivalent to PyTorch's `torchvision.transforms.RandomHorizontalFlip(p)`
/// or TensorFlow's `tf.image.random_flip_left_right()`.
///
/// ## Complexity
///
/// Let `N` = batch, `C` = channels, `H` = height, `W` = width.
///
/// - **Time**: O(N × C × H × W) when flip occurs, O(1) when skipped (returns clone).
/// - **Space**: O(N × C × H × W) for output buffer.
class RandomHorizontalFlipOp extends TransformOp with RequiresContiguous {
  /// Probability of flipping the tensor. Must be in [0.0, 1.0].
  final double probability;

  /// Optional random seed for reproducibility.
  final int? seed;

  late final Random _random;

  /// Creates a random horizontal flip operation.
  ///
  /// [probability] defaults to 0.5, matching PyTorch's default.
  /// [seed] can be provided for deterministic behavior.
  RandomHorizontalFlipOp({
    this.probability = 0.5,
    this.seed,
  }) {
    if (probability < 0.0 || probability > 1.0) {
      throw InvalidParameterException(
        'probability',
        probability.toString(),
        'Probability must be between 0.0 and 1.0',
      );
    }
    _random = Random(seed);
  }

  @override
  String get name =>
      'RandomHorizontalFlip(p=$probability${seed != null ? ', seed=$seed' : ''})';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
        pytorchEquivalent: 'torchvision.transforms.RandomHorizontalFlip',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    if (_random.nextDouble() < probability) {
      return _flipHorizontal(contiguous);
    }
    return contiguous.clone();
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Randomly flips a tensor top-to-bottom with configurable probability.
///
/// Useful for data augmentation in tasks where vertical orientation
/// is not semantically important (e.g., satellite imagery, medical imaging).
///
/// Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` tensor formats.
///
/// Equivalent to PyTorch's `torchvision.transforms.RandomVerticalFlip(p)`
/// or TensorFlow's `tf.image.random_flip_up_down()`.
///
/// ## Complexity
///
/// Let `N` = batch, `C` = channels, `H` = height, `W` = width.
///
/// - **Time**: O(N × C × H × W) when flip occurs, O(1) when skipped (returns clone).
/// - **Space**: O(N × C × H × W) for output buffer.
class RandomVerticalFlipOp extends TransformOp with RequiresContiguous {
  /// Probability of flipping the tensor. Must be in [0.0, 1.0].
  final double probability;

  /// Optional random seed for reproducibility.
  final int? seed;

  late final Random _random;

  /// Creates a random vertical flip operation.
  ///
  /// [probability] defaults to 0.5, matching PyTorch's default.
  /// [seed] can be provided for deterministic behavior.
  RandomVerticalFlipOp({
    this.probability = 0.5,
    this.seed,
  }) {
    if (probability < 0.0 || probability > 1.0) {
      throw InvalidParameterException(
        'probability',
        probability.toString(),
        'Probability must be between 0.0 and 1.0',
      );
    }
    _random = Random(seed);
  }

  @override
  String get name =>
      'RandomVerticalFlip(p=$probability${seed != null ? ', seed=$seed' : ''})';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
        pytorchEquivalent: 'torchvision.transforms.RandomVerticalFlip',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    if (_random.nextDouble() < probability) {
      return _flipVertical(contiguous);
    }
    return contiguous.clone();
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// Shared flip helpers (file-private)
// ============================================================================

void _validateShape(List<int> shape) {
  final rank = shape.length;
  if (rank != 3 && rank != 4) {
    throw ShapeMismatchException(
      actual: shape,
      message: 'Flip operations require 3D [C,H,W] or 4D [N,C,H,W] tensor',
    );
  }
}

/// Flips a tensor along the width axis (last dimension).
TensorBuffer _flipHorizontal(TensorBuffer input) {
  final shape = input.shape;

  if (shape.length == 3) {
    final (c, h, w) = (shape[0], shape[1], shape[2]);
    final output = TensorBuffer.uninitialized([c, h, w], dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = ch * h * w;
          for (int row = 0; row < h; row++) {
            final rowOffset = chOffset + row * w;
            for (int col = 0; col < w; col++) {
              outList[rowOffset + col] = inList[rowOffset + (w - 1 - col)];
            }
          }
        }
      case DType.float64:
        final inList = input.storage.data as Float64List;
        final outList = output.storage.data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = ch * h * w;
          for (int row = 0; row < h; row++) {
            final rowOffset = chOffset + row * w;
            for (int col = 0; col < w; col++) {
              outList[rowOffset + col] = inList[rowOffset + (w - 1 - col)];
            }
          }
        }
      default:
        for (int ch = 0; ch < c; ch++) {
          for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
              final srcIdx = ch * h * w + row * w + (w - 1 - col);
              final dstIdx = ch * h * w + row * w + col;
              output.storage
                  .setFromDouble(dstIdx, input.storage.getAsDouble(srcIdx));
            }
          }
        }
    }

    return output;
  } else {
    // 4D: [N, C, H, W]
    final (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    final output = TensorBuffer.uninitialized([n, c, h, w], dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * c * h * w;
          for (int ch = 0; ch < c; ch++) {
            final chOffset = batchOffset + ch * h * w;
            for (int row = 0; row < h; row++) {
              final rowOffset = chOffset + row * w;
              for (int col = 0; col < w; col++) {
                outList[rowOffset + col] = inList[rowOffset + (w - 1 - col)];
              }
            }
          }
        }
      case DType.float64:
        final inList = input.storage.data as Float64List;
        final outList = output.storage.data as Float64List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * c * h * w;
          for (int ch = 0; ch < c; ch++) {
            final chOffset = batchOffset + ch * h * w;
            for (int row = 0; row < h; row++) {
              final rowOffset = chOffset + row * w;
              for (int col = 0; col < w; col++) {
                outList[rowOffset + col] = inList[rowOffset + (w - 1 - col)];
              }
            }
          }
        }
      default:
        for (int batch = 0; batch < n; batch++) {
          for (int ch = 0; ch < c; ch++) {
            for (int row = 0; row < h; row++) {
              for (int col = 0; col < w; col++) {
                final srcIdx =
                    batch * c * h * w + ch * h * w + row * w + (w - 1 - col);
                final dstIdx = batch * c * h * w + ch * h * w + row * w + col;
                output.storage
                    .setFromDouble(dstIdx, input.storage.getAsDouble(srcIdx));
              }
            }
          }
        }
    }

    return output;
  }
}

/// Flips a tensor along the height axis (second-to-last dimension).
TensorBuffer _flipVertical(TensorBuffer input) {
  final shape = input.shape;

  if (shape.length == 3) {
    final (c, h, w) = (shape[0], shape[1], shape[2]);
    final output = TensorBuffer.uninitialized([c, h, w], dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = ch * h * w;
          for (int row = 0; row < h; row++) {
            final srcRowOffset = chOffset + (h - 1 - row) * w;
            final dstRowOffset = chOffset + row * w;
            for (int col = 0; col < w; col++) {
              outList[dstRowOffset + col] = inList[srcRowOffset + col];
            }
          }
        }
      case DType.float64:
        final inList = input.storage.data as Float64List;
        final outList = output.storage.data as Float64List;
        for (int ch = 0; ch < c; ch++) {
          final chOffset = ch * h * w;
          for (int row = 0; row < h; row++) {
            final srcRowOffset = chOffset + (h - 1 - row) * w;
            final dstRowOffset = chOffset + row * w;
            for (int col = 0; col < w; col++) {
              outList[dstRowOffset + col] = inList[srcRowOffset + col];
            }
          }
        }
      default:
        for (int ch = 0; ch < c; ch++) {
          for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
              final srcIdx = ch * h * w + (h - 1 - row) * w + col;
              final dstIdx = ch * h * w + row * w + col;
              output.storage
                  .setFromDouble(dstIdx, input.storage.getAsDouble(srcIdx));
            }
          }
        }
    }

    return output;
  } else {
    // 4D: [N, C, H, W]
    final (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    final output = TensorBuffer.uninitialized([n, c, h, w], dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * c * h * w;
          for (int ch = 0; ch < c; ch++) {
            final chOffset = batchOffset + ch * h * w;
            for (int row = 0; row < h; row++) {
              final srcRowOffset = chOffset + (h - 1 - row) * w;
              final dstRowOffset = chOffset + row * w;
              for (int col = 0; col < w; col++) {
                outList[dstRowOffset + col] = inList[srcRowOffset + col];
              }
            }
          }
        }
      case DType.float64:
        final inList = input.storage.data as Float64List;
        final outList = output.storage.data as Float64List;
        for (int batch = 0; batch < n; batch++) {
          final batchOffset = batch * c * h * w;
          for (int ch = 0; ch < c; ch++) {
            final chOffset = batchOffset + ch * h * w;
            for (int row = 0; row < h; row++) {
              final srcRowOffset = chOffset + (h - 1 - row) * w;
              final dstRowOffset = chOffset + row * w;
              for (int col = 0; col < w; col++) {
                outList[dstRowOffset + col] = inList[srcRowOffset + col];
              }
            }
          }
        }
      default:
        for (int batch = 0; batch < n; batch++) {
          for (int ch = 0; ch < c; ch++) {
            for (int row = 0; row < h; row++) {
              for (int col = 0; col < w; col++) {
                final srcIdx =
                    batch * c * h * w + ch * h * w + (h - 1 - row) * w + col;
                final dstIdx = batch * c * h * w + ch * h * w + row * w + col;
                output.storage
                    .setFromDouble(dstIdx, input.storage.getAsDouble(srcIdx));
              }
            }
          }
        }
    }

    return output;
  }
}
