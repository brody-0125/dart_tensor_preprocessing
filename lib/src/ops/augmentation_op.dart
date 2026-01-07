import 'dart:math';

import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
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
      final output = TensorBuffer.zeros([c, height, width], dtype: input.dtype);

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
          TensorBuffer.zeros([n, c, height, width], dtype: input.dtype);

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
      final output = TensorBuffer.zeros([c, h, w], dtype: input.dtype);

      for (int ch = 0; ch < c; ch++) {
        // Horizontal pass
        final temp = List<double>.filled(h * w, 0.0);
        for (int row = 0; row < h; row++) {
          for (int col = 0; col < w; col++) {
            double sum = 0.0;
            for (int k = 0; k < kernelSize; k++) {
              final x = col + k - radius;
              final xi = _reflectIndex(x, w);
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
              final y = row + k - radius;
              final yi = _reflectIndex(y, h);
              sum += temp[yi * w + col] * kernel[k];
            }
            final outputIdx = ch * h * w + row * w + col;
            output.storage.setFromDouble(outputIdx, sum);
          }
        }
      }

      return output;
    } else {
      // 4D: [N, C, H, W]
      final (n, c, h, w) =
          (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
      final output = TensorBuffer.zeros([n, c, h, w], dtype: input.dtype);

      for (int batch = 0; batch < n; batch++) {
        for (int ch = 0; ch < c; ch++) {
          // Horizontal pass
          final temp = List<double>.filled(h * w, 0.0);
          for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
              double sum = 0.0;
              for (int k = 0; k < kernelSize; k++) {
                final x = col + k - radius;
                final xi = _reflectIndex(x, w);
                final inputIdx = batch * c * h * w + ch * h * w + row * w + xi;
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
                final y = row + k - radius;
                final yi = _reflectIndex(y, h);
                sum += temp[yi * w + col] * kernel[k];
              }
              final outputIdx = batch * c * h * w + ch * h * w + row * w + col;
              output.storage.setFromDouble(outputIdx, sum);
            }
          }
        }
      }

      return output;
    }
  }

  int _reflectIndex(int idx, int size) {
    if (idx < 0) {
      return -idx - 1;
    } else if (idx >= size) {
      return 2 * size - idx - 1;
    }
    return idx;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
