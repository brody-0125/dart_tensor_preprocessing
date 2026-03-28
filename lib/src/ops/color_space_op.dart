import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/validation_utils.dart';
import 'color_jitter_op.dart' show rgbToHsv, hsvToRgb;
import 'transform_op.dart';

// ============================================================================
// RgbToGrayscaleOp
// ============================================================================

/// Converts an RGB tensor to grayscale using ITU-R BT.601 weighted coefficients.
///
/// Formula: `Y = 0.2989 * R + 0.5870 * G + 0.1140 * B`
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
/// Output has 1 channel: [1, H, W] or [N, 1, H, W].
///
/// ## Example
/// ```dart
/// final grayscale = RgbToGrayscaleOp();
/// final result = grayscale(rgbTensor); // [3, H, W] -> [1, H, W]
/// ```
///
/// ## References
/// - PyTorch: `torchvision.transforms.Grayscale(num_output_channels=1)`
/// - TensorFlow: `tf.image.rgb_to_grayscale`
class RgbToGrayscaleOp extends TransformOp with RequiresContiguous {
  /// ITU-R BT.601 luminance coefficients.
  static const double _rWeight = 0.2989;
  static const double _gWeight = 0.5870;
  static const double _bWeight = 0.1140;

  /// Creates an RGB to grayscale conversion operation.
  RgbToGrayscaleOp();

  @override
  String get name => 'RgbToGrayscale';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: false,
        pytorchEquivalent: 'torchvision.transforms.Grayscale',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    OpValidator.validateImageTensor(
      contiguous,
      operationName: name,
      expectedChannels: 3,
    );

    if (contiguous.shape.length == 3) {
      return _convert3D(contiguous);
    } else {
      return _convert4D(contiguous);
    }
  }

  TensorBuffer _convert3D(TensorBuffer input) {
    final h = input.shape[1];
    final w = input.shape[2];
    final hw = h * w;
    final outputShape = [1, h, w];

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(hw);
        for (int i = 0; i < hw; i++) {
          dst[i] = src[i] * _rWeight +
              src[hw + i] * _gWeight +
              src[2 * hw + i] * _bWeight;
        }
        return TensorBuffer.fromFloat32List(dst, outputShape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(hw);
        for (int i = 0; i < hw; i++) {
          dst[i] = src[i] * _rWeight +
              src[hw + i] * _gWeight +
              src[2 * hw + i] * _bWeight;
        }
        return TensorBuffer.fromFloat64List(dst, outputShape);
      default:
        final dst = Float32List(hw);
        for (int i = 0; i < hw; i++) {
          final r = input.storage.getAsDouble(i);
          final g = input.storage.getAsDouble(hw + i);
          final b = input.storage.getAsDouble(2 * hw + i);
          dst[i] = (r * _rWeight + g * _gWeight + b * _bWeight);
        }
        return TensorBuffer.fromFloat32List(dst, outputShape);
    }
  }

  TensorBuffer _convert4D(TensorBuffer input) {
    final n = input.shape[0];
    final h = input.shape[2];
    final w = input.shape[3];
    final hw = h * w;
    final chw = 3 * hw;
    final outputShape = [n, 1, h, w];

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(n * hw);
        for (int batch = 0; batch < n; batch++) {
          final srcOffset = batch * chw;
          final dstOffset = batch * hw;
          for (int i = 0; i < hw; i++) {
            dst[dstOffset + i] = src[srcOffset + i] * _rWeight +
                src[srcOffset + hw + i] * _gWeight +
                src[srcOffset + 2 * hw + i] * _bWeight;
          }
        }
        return TensorBuffer.fromFloat32List(dst, outputShape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(n * hw);
        for (int batch = 0; batch < n; batch++) {
          final srcOffset = batch * chw;
          final dstOffset = batch * hw;
          for (int i = 0; i < hw; i++) {
            dst[dstOffset + i] = src[srcOffset + i] * _rWeight +
                src[srcOffset + hw + i] * _gWeight +
                src[srcOffset + 2 * hw + i] * _bWeight;
          }
        }
        return TensorBuffer.fromFloat64List(dst, outputShape);
      default:
        final dst = Float32List(n * hw);
        for (int batch = 0; batch < n; batch++) {
          final srcOffset = batch * chw;
          final dstOffset = batch * hw;
          for (int i = 0; i < hw; i++) {
            final r = input.storage.getAsDouble(srcOffset + i);
            final g = input.storage.getAsDouble(srcOffset + hw + i);
            final b = input.storage.getAsDouble(srcOffset + 2 * hw + i);
            dst[dstOffset + i] = (r * _rWeight + g * _gWeight + b * _bWeight);
          }
        }
        return TensorBuffer.fromFloat32List(dst, outputShape);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [1, inputShape[1], inputShape[2]];
    } else if (inputShape.length == 4) {
      return [inputShape[0], 1, inputShape[2], inputShape[3]];
    }
    throw ShapeMismatchException(
      actual: inputShape,
      message: '$name requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
    );
  }
}

// ============================================================================
// RgbToHsvOp
// ============================================================================

/// Converts an RGB tensor to HSV color space.
///
/// All channels use [0, 1] range: H∈[0,1], S∈[0,1], V∈[0,1].
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
/// Output shape is the same as input.
///
/// ## Example
/// ```dart
/// final toHsv = RgbToHsvOp();
/// final hsv = toHsv(rgbTensor); // [3, H, W] -> [3, H, W]
/// ```
///
/// ## References
/// - TensorFlow: `tf.image.rgb_to_hsv`
class RgbToHsvOp extends TransformOp with RequiresContiguous {
  /// Creates an RGB to HSV conversion operation.
  RgbToHsvOp();

  @override
  String get name => 'RgbToHsv';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    OpValidator.validateImageTensor(
      contiguous,
      operationName: name,
      expectedChannels: 3,
    );

    if (contiguous.shape.length == 3) {
      return _convert3D(contiguous);
    } else {
      return _convert4D(contiguous);
    }
  }

  TensorBuffer _convert3D(TensorBuffer input) {
    final h = input.shape[1];
    final w = input.shape[2];
    final hw = h * w;

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final (hv, s, v) = rgbToHsv(src[i], src[hw + i], src[2 * hw + i]);
          dst[i] = hv;
          dst[hw + i] = s;
          dst[2 * hw + i] = v;
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final (hv, s, v) = rgbToHsv(src[i], src[hw + i], src[2 * hw + i]);
          dst[i] = hv;
          dst[hw + i] = s;
          dst[2 * hw + i] = v;
        }
        return TensorBuffer.fromFloat64List(dst, input.shape);
      default:
        final dst = Float32List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final r = input.storage.getAsDouble(i);
          final g = input.storage.getAsDouble(hw + i);
          final b = input.storage.getAsDouble(2 * hw + i);
          final (hv, s, v) = rgbToHsv(r, g, b);
          dst[i] = hv;
          dst[hw + i] = s;
          dst[2 * hw + i] = v;
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
    }
  }

  TensorBuffer _convert4D(TensorBuffer input) {
    final n = input.shape[0];
    final h = input.shape[2];
    final w = input.shape[3];
    final hw = h * w;
    final chw = 3 * hw;

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final (hv, s, v) = rgbToHsv(
                src[off + i], src[off + hw + i], src[off + 2 * hw + i]);
            dst[off + i] = hv;
            dst[off + hw + i] = s;
            dst[off + 2 * hw + i] = v;
          }
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final (hv, s, v) = rgbToHsv(
                src[off + i], src[off + hw + i], src[off + 2 * hw + i]);
            dst[off + i] = hv;
            dst[off + hw + i] = s;
            dst[off + 2 * hw + i] = v;
          }
        }
        return TensorBuffer.fromFloat64List(dst, input.shape);
      default:
        final dst = Float32List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final r = input.storage.getAsDouble(off + i);
            final g = input.storage.getAsDouble(off + hw + i);
            final b = input.storage.getAsDouble(off + 2 * hw + i);
            final (hv, s, v) = rgbToHsv(r, g, b);
            dst[off + i] = hv;
            dst[off + hw + i] = s;
            dst[off + 2 * hw + i] = v;
          }
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// HsvToRgbOp
// ============================================================================

/// Converts an HSV tensor to RGB color space.
///
/// All channels use [0, 1] range: H∈[0,1], S∈[0,1], V∈[0,1].
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (HSV).
/// Output shape is the same as input.
///
/// ## Example
/// ```dart
/// final toRgb = HsvToRgbOp();
/// final rgb = toRgb(hsvTensor); // [3, H, W] -> [3, H, W]
/// ```
///
/// ## References
/// - TensorFlow: `tf.image.hsv_to_rgb`
class HsvToRgbOp extends TransformOp with RequiresContiguous {
  /// Creates an HSV to RGB conversion operation.
  HsvToRgbOp();

  @override
  String get name => 'HsvToRgb';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    OpValidator.validateImageTensor(
      contiguous,
      operationName: name,
      expectedChannels: 3,
    );

    if (contiguous.shape.length == 3) {
      return _convert3D(contiguous);
    } else {
      return _convert4D(contiguous);
    }
  }

  TensorBuffer _convert3D(TensorBuffer input) {
    final h = input.shape[1];
    final w = input.shape[2];
    final hw = h * w;

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final (r, g, b) = hsvToRgb(src[i], src[hw + i], src[2 * hw + i]);
          dst[i] = r;
          dst[hw + i] = g;
          dst[2 * hw + i] = b;
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final (r, g, b) = hsvToRgb(src[i], src[hw + i], src[2 * hw + i]);
          dst[i] = r;
          dst[hw + i] = g;
          dst[2 * hw + i] = b;
        }
        return TensorBuffer.fromFloat64List(dst, input.shape);
      default:
        final dst = Float32List(3 * hw);
        for (int i = 0; i < hw; i++) {
          final hv = input.storage.getAsDouble(i);
          final s = input.storage.getAsDouble(hw + i);
          final v = input.storage.getAsDouble(2 * hw + i);
          final (r, g, b) = hsvToRgb(hv, s, v);
          dst[i] = r;
          dst[hw + i] = g;
          dst[2 * hw + i] = b;
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
    }
  }

  TensorBuffer _convert4D(TensorBuffer input) {
    final n = input.shape[0];
    final h = input.shape[2];
    final w = input.shape[3];
    final hw = h * w;
    final chw = 3 * hw;

    switch (input.dtype) {
      case DType.float32:
        final src = input.storage.data as Float32List;
        final dst = Float32List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final (r, g, b) = hsvToRgb(
                src[off + i], src[off + hw + i], src[off + 2 * hw + i]);
            dst[off + i] = r;
            dst[off + hw + i] = g;
            dst[off + 2 * hw + i] = b;
          }
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
      case DType.float64:
        final src = input.storage.data as Float64List;
        final dst = Float64List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final (r, g, b) = hsvToRgb(
                src[off + i], src[off + hw + i], src[off + 2 * hw + i]);
            dst[off + i] = r;
            dst[off + hw + i] = g;
            dst[off + 2 * hw + i] = b;
          }
        }
        return TensorBuffer.fromFloat64List(dst, input.shape);
      default:
        final dst = Float32List(n * chw);
        for (int batch = 0; batch < n; batch++) {
          final off = batch * chw;
          for (int i = 0; i < hw; i++) {
            final hv = input.storage.getAsDouble(off + i);
            final s = input.storage.getAsDouble(off + hw + i);
            final v = input.storage.getAsDouble(off + 2 * hw + i);
            final (r, g, b) = hsvToRgb(hv, s, v);
            dst[off + i] = r;
            dst[off + hw + i] = g;
            dst[off + 2 * hw + i] = b;
          }
        }
        return TensorBuffer.fromFloat32List(dst, input.shape);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
