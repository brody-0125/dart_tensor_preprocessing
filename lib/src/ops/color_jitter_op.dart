import 'dart:math';
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

// ============================================================================
// RGB <-> HSV Color Space Conversion Utilities
// ============================================================================

/// Converts RGB values to HSV.
///
/// Input: r, g, b in [0, 1]
/// Output: (h, s, v) where h in [0, 1], s in [0, 1], v in [0, 1]
(double, double, double) rgbToHsv(double r, double g, double b) {
  final maxVal = max(r, max(g, b));
  final minVal = min(r, min(g, b));
  final delta = maxVal - minVal;

  // Value
  final v = maxVal;

  // Saturation
  final s = maxVal == 0 ? 0.0 : delta / maxVal;

  // Hue
  double h;
  if (delta == 0) {
    h = 0.0;
  } else if (maxVal == r) {
    h = ((g - b) / delta) % 6.0;
    if (h < 0) h += 6.0;
  } else if (maxVal == g) {
    h = (b - r) / delta + 2.0;
  } else {
    h = (r - g) / delta + 4.0;
  }
  h /= 6.0; // Normalize to [0, 1]

  return (h, s, v);
}

/// Converts HSV values to RGB.
///
/// Input: h in [0, 1], s in [0, 1], v in [0, 1]
/// Output: (r, g, b) in [0, 1]
(double, double, double) hsvToRgb(double h, double s, double v) {
  // Wrap h to [0, 1)
  h = h % 1.0;
  if (h < 0) h += 1.0;

  final c = v * s;
  final hPrime = h * 6.0;
  final x = c * (1.0 - (hPrime % 2.0 - 1.0).abs());
  final m = v - c;

  double r, g, b;
  if (hPrime < 1) {
    (r, g, b) = (c, x, 0.0);
  } else if (hPrime < 2) {
    (r, g, b) = (x, c, 0.0);
  } else if (hPrime < 3) {
    (r, g, b) = (0.0, c, x);
  } else if (hPrime < 4) {
    (r, g, b) = (0.0, x, c);
  } else if (hPrime < 5) {
    (r, g, b) = (x, 0.0, c);
  } else {
    (r, g, b) = (c, 0.0, x);
  }

  return (r + m, g + m, b + m);
}

// ============================================================================
// Shape Validation Helpers
// ============================================================================

/// Validates that a tensor has 3D [C, H, W] or 4D [N, C, H, W] shape
/// with exactly 3 channels (RGB).
void _validateColorShape(List<int> shape, String opName) {
  final rank = shape.length;
  if (rank != 3 && rank != 4) {
    throw ShapeMismatchException(
      actual: shape,
      message: '$opName requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
    );
  }

  final channels = rank == 3 ? shape[0] : shape[1];
  if (channels != 3) {
    throw ShapeMismatchException(
      actual: shape,
      message: '$opName requires 3-channel (RGB) tensor, got $channels channels',
    );
  }
}

// ============================================================================
// AdjustBrightnessOp
// ============================================================================

/// Adjusts the brightness of an image tensor by an additive factor.
///
/// For each pixel value: `output = input + factor`
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
/// Values are clamped to [0, 1] range.
///
/// ## Parameters
/// - [factor]: Brightness adjustment amount. Positive values brighten,
///   negative values darken. Typically in range [-1, 1].
///
/// ## Example
/// ```dart
/// final brighten = AdjustBrightnessOp(factor: 0.2);
/// final result = brighten(tensor);
/// ```
class AdjustBrightnessOp extends TransformOp with RequiresContiguous {
  /// Brightness adjustment factor (additive).
  final double factor;

  /// Creates a brightness adjustment operation.
  AdjustBrightnessOp({required this.factor});

  @override
  String get name => 'AdjustBrightness(factor=$factor)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent: 'torchvision.transforms.functional.adjust_brightness',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateColorShape(contiguous.shape, 'AdjustBrightnessOp');

    final output = contiguous.clone();
    final numel = output.numel;

    switch (output.dtype) {
      case DType.float32:
        final list = output.storage.data as Float32List;
        for (int i = 0; i < numel; i++) {
          list[i] = (list[i] + factor).clamp(0.0, 1.0);
        }
      case DType.float64:
        final list = output.storage.data as Float64List;
        for (int i = 0; i < numel; i++) {
          list[i] = (list[i] + factor).clamp(0.0, 1.0);
        }
      default:
        for (int i = 0; i < numel; i++) {
          final val = output.storage.getAsDouble(i);
          output.storage.setFromDouble(i, (val + factor).clamp(0.0, 1.0));
        }
    }

    return output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// AdjustContrastOp
// ============================================================================

/// Adjusts the contrast of an image tensor.
///
/// For each channel: `output = (input - mean) * factor + mean`
/// where mean is the per-channel mean value.
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
/// Values are clamped to [0, 1] range.
///
/// ## Parameters
/// - [factor]: Contrast multiplier. 0 gives a solid gray image,
///   1 gives the original image, >1 increases contrast.
///
/// ## Example
/// ```dart
/// final contrast = AdjustContrastOp(factor: 1.5);
/// final result = contrast(tensor);
/// ```
class AdjustContrastOp extends TransformOp with RequiresContiguous {
  /// Contrast adjustment factor (multiplicative).
  final double factor;

  /// Creates a contrast adjustment operation.
  AdjustContrastOp({required this.factor}) {
    if (factor < 0) {
      throw InvalidParameterException(
        'factor',
        factor.toString(),
        'Contrast factor must be non-negative',
      );
    }
  }

  @override
  String get name => 'AdjustContrast(factor=$factor)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent: 'torchvision.transforms.functional.adjust_contrast',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateColorShape(contiguous.shape, 'AdjustContrastOp');

    final output = contiguous.clone();
    final shape = output.shape;

    if (shape.length == 3) {
      _adjustContrast3D(output);
    } else {
      _adjustContrast4D(output);
    }

    return output;
  }

  void _adjustContrast3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;

    for (int ch = 0; ch < c; ch++) {
      final offset = ch * channelSize;

      // Compute per-channel mean
      double sum = 0.0;
      for (int i = 0; i < channelSize; i++) {
        sum += tensor.storage.getAsDouble(offset + i);
      }
      final mean = sum / channelSize;

      // Apply contrast adjustment
      for (int i = 0; i < channelSize; i++) {
        final idx = offset + i;
        final val = tensor.storage.getAsDouble(idx);
        final adjusted = (val - mean) * factor + mean;
        tensor.storage.setFromDouble(idx, adjusted.clamp(0.0, 1.0));
      }
    }
  }

  void _adjustContrast4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = c * channelSize;

    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchSize;
      for (int ch = 0; ch < c; ch++) {
        final offset = batchOffset + ch * channelSize;

        // Compute per-channel mean
        double sum = 0.0;
        for (int i = 0; i < channelSize; i++) {
          sum += tensor.storage.getAsDouble(offset + i);
        }
        final mean = sum / channelSize;

        // Apply contrast adjustment
        for (int i = 0; i < channelSize; i++) {
          final idx = offset + i;
          final val = tensor.storage.getAsDouble(idx);
          final adjusted = (val - mean) * factor + mean;
          tensor.storage.setFromDouble(idx, adjusted.clamp(0.0, 1.0));
        }
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// AdjustSaturationOp
// ============================================================================

/// Adjusts the saturation of an image tensor using RGB-to-HSV conversion.
///
/// Converts RGB to HSV, multiplies the S channel by the given factor,
/// clamps to [0, 1], and converts back to RGB.
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
///
/// ## Parameters
/// - [factor]: Saturation multiplier. 0 gives a grayscale image,
///   1 gives the original image, >1 increases saturation.
///
/// ## Example
/// ```dart
/// final saturation = AdjustSaturationOp(factor: 1.5);
/// final result = saturation(tensor);
/// ```
class AdjustSaturationOp extends TransformOp with RequiresContiguous {
  /// Saturation adjustment factor (multiplicative).
  final double factor;

  /// Creates a saturation adjustment operation.
  AdjustSaturationOp({required this.factor}) {
    if (factor < 0) {
      throw InvalidParameterException(
        'factor',
        factor.toString(),
        'Saturation factor must be non-negative',
      );
    }
  }

  @override
  String get name => 'AdjustSaturation(factor=$factor)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent:
            'torchvision.transforms.functional.adjust_saturation',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateColorShape(contiguous.shape, 'AdjustSaturationOp');

    final output = contiguous.clone();
    final shape = output.shape;

    if (shape.length == 3) {
      _adjustSaturation3D(output);
    } else {
      _adjustSaturation4D(output);
    }

    return output;
  }

  void _adjustSaturation3D(TensorBuffer tensor) {
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;

    for (int i = 0; i < channelSize; i++) {
      final r = tensor.storage.getAsDouble(i);
      final g = tensor.storage.getAsDouble(channelSize + i);
      final b = tensor.storage.getAsDouble(2 * channelSize + i);

      var (hue, sat, val) = rgbToHsv(r, g, b);
      sat = (sat * factor).clamp(0.0, 1.0);
      final (nr, ng, nb) = hsvToRgb(hue, sat, val);

      tensor.storage.setFromDouble(i, nr.clamp(0.0, 1.0));
      tensor.storage.setFromDouble(channelSize + i, ng.clamp(0.0, 1.0));
      tensor.storage.setFromDouble(2 * channelSize + i, nb.clamp(0.0, 1.0));
    }
  }

  void _adjustSaturation4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = 3 * channelSize;

    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchSize;
      for (int i = 0; i < channelSize; i++) {
        final rIdx = batchOffset + i;
        final gIdx = batchOffset + channelSize + i;
        final bIdx = batchOffset + 2 * channelSize + i;

        final r = tensor.storage.getAsDouble(rIdx);
        final g = tensor.storage.getAsDouble(gIdx);
        final b = tensor.storage.getAsDouble(bIdx);

        var (hue, sat, val) = rgbToHsv(r, g, b);
        sat = (sat * factor).clamp(0.0, 1.0);
        final (nr, ng, nb) = hsvToRgb(hue, sat, val);

        tensor.storage.setFromDouble(rIdx, nr.clamp(0.0, 1.0));
        tensor.storage.setFromDouble(gIdx, ng.clamp(0.0, 1.0));
        tensor.storage.setFromDouble(bIdx, nb.clamp(0.0, 1.0));
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// AdjustHueOp
// ============================================================================

/// Adjusts the hue of an image tensor using RGB-to-HSV conversion.
///
/// Converts RGB to HSV, adds the given factor to the H channel
/// (wrapping around [0, 1]), and converts back to RGB.
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
///
/// ## Parameters
/// - [factor]: Hue shift amount. Must be in [-0.5, 0.5].
///   The hue channel is in [0, 1], so 0.5 corresponds to a 180-degree shift.
///
/// ## Example
/// ```dart
/// final hue = AdjustHueOp(factor: 0.1);
/// final result = hue(tensor);
/// ```
class AdjustHueOp extends TransformOp with RequiresContiguous {
  /// Hue adjustment factor (additive, in [-0.5, 0.5]).
  final double factor;

  /// Creates a hue adjustment operation.
  AdjustHueOp({required this.factor}) {
    if (factor < -0.5 || factor > 0.5) {
      throw InvalidParameterException(
        'factor',
        factor.toString(),
        'Hue factor must be in [-0.5, 0.5]',
      );
    }
  }

  @override
  String get name => 'AdjustHue(factor=$factor)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent: 'torchvision.transforms.functional.adjust_hue',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateColorShape(contiguous.shape, 'AdjustHueOp');

    final output = contiguous.clone();
    final shape = output.shape;

    if (shape.length == 3) {
      _adjustHue3D(output);
    } else {
      _adjustHue4D(output);
    }

    return output;
  }

  void _adjustHue3D(TensorBuffer tensor) {
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;

    for (int i = 0; i < channelSize; i++) {
      final r = tensor.storage.getAsDouble(i);
      final g = tensor.storage.getAsDouble(channelSize + i);
      final b = tensor.storage.getAsDouble(2 * channelSize + i);

      var (hue, sat, val) = rgbToHsv(r, g, b);
      hue = (hue + factor) % 1.0;
      if (hue < 0) hue += 1.0;
      final (nr, ng, nb) = hsvToRgb(hue, sat, val);

      tensor.storage.setFromDouble(i, nr.clamp(0.0, 1.0));
      tensor.storage.setFromDouble(channelSize + i, ng.clamp(0.0, 1.0));
      tensor.storage.setFromDouble(2 * channelSize + i, nb.clamp(0.0, 1.0));
    }
  }

  void _adjustHue4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = 3 * channelSize;

    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchSize;
      for (int i = 0; i < channelSize; i++) {
        final rIdx = batchOffset + i;
        final gIdx = batchOffset + channelSize + i;
        final bIdx = batchOffset + 2 * channelSize + i;

        final r = tensor.storage.getAsDouble(rIdx);
        final g = tensor.storage.getAsDouble(gIdx);
        final b = tensor.storage.getAsDouble(bIdx);

        var (hue, sat, val) = rgbToHsv(r, g, b);
        hue = (hue + factor) % 1.0;
        if (hue < 0) hue += 1.0;
        final (nr, ng, nb) = hsvToRgb(hue, sat, val);

        tensor.storage.setFromDouble(rIdx, nr.clamp(0.0, 1.0));
        tensor.storage.setFromDouble(gIdx, ng.clamp(0.0, 1.0));
        tensor.storage.setFromDouble(bIdx, nb.clamp(0.0, 1.0));
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

// ============================================================================
// ColorJitterOp (Combined PyTorch-style)
// ============================================================================

/// Randomly adjusts brightness, contrast, saturation, and hue of image tensors.
///
/// This is a combined color jitter operation similar to
/// `torchvision.transforms.ColorJitter`. Each parameter specifies a range
/// from which a random factor is uniformly sampled.
///
/// The transforms are applied in a random order for each invocation.
///
/// Supports 3D [C, H, W] and 4D [N, C, H, W] tensors with 3 channels (RGB).
/// Input values are expected in [0, 1] range.
///
/// ## Parameters
/// - [brightness]: Range for additive brightness. If given as a single value `v`,
///   the range is `[-v, v]`. If null, brightness is not adjusted.
/// - [contrast]: Range for multiplicative contrast. If given as a single value `v`,
///   the range is `[max(0, 1-v), 1+v]`. If null, contrast is not adjusted.
/// - [saturation]: Range for multiplicative saturation. If given as a single value `v`,
///   the range is `[max(0, 1-v), 1+v]`. If null, saturation is not adjusted.
/// - [hue]: Range for additive hue. If given as a single value `v`,
///   the range is `[-v, v]` where v must be <= 0.5. If null, hue is not adjusted.
/// - [seed]: Optional random seed for reproducibility.
///
/// ## Example
/// ```dart
/// // PyTorch-style combined color jitter
/// final jitter = ColorJitterOp(
///   brightness: 0.2,
///   contrast: 0.2,
///   saturation: 0.2,
///   hue: 0.1,
///   seed: 42,
/// );
/// final result = jitter(tensor);
/// ```
class ColorJitterOp extends TransformOp with RequiresContiguous {
  /// Brightness jitter range (additive). Sampled from [-brightness, brightness].
  final double? brightness;

  /// Contrast jitter range (multiplicative). Sampled from [max(0, 1-contrast), 1+contrast].
  final double? contrast;

  /// Saturation jitter range (multiplicative). Sampled from [max(0, 1-saturation), 1+saturation].
  final double? saturation;

  /// Hue jitter range (additive). Sampled from [-hue, hue]. Must be <= 0.5.
  final double? hue;

  /// Optional random seed for reproducibility.
  final int? seed;

  /// Creates a color jitter operation.
  ColorJitterOp({
    this.brightness,
    this.contrast,
    this.saturation,
    this.hue,
    this.seed,
  }) {
    if (brightness != null && brightness! < 0) {
      throw InvalidParameterException(
        'brightness',
        brightness.toString(),
        'Brightness must be non-negative',
      );
    }
    if (contrast != null && contrast! < 0) {
      throw InvalidParameterException(
        'contrast',
        contrast.toString(),
        'Contrast must be non-negative',
      );
    }
    if (saturation != null && saturation! < 0) {
      throw InvalidParameterException(
        'saturation',
        saturation.toString(),
        'Saturation must be non-negative',
      );
    }
    if (hue != null && (hue! < 0 || hue! > 0.5)) {
      throw InvalidParameterException(
        'hue',
        hue.toString(),
        'Hue must be in [0, 0.5]',
      );
    }
  }

  @override
  String get name =>
      'ColorJitter(brightness=$brightness, contrast=$contrast, '
      'saturation=$saturation, hue=$hue, seed=$seed)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent: 'torchvision.transforms.ColorJitter',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateColorShape(contiguous.shape, 'ColorJitterOp');

    final random = Random(seed);

    // Build list of transforms to apply
    final transforms = <TransformOp>[];

    if (brightness != null) {
      final bFactor =
          -brightness! + random.nextDouble() * (2 * brightness!);
      transforms.add(AdjustBrightnessOp(factor: bFactor));
    }

    if (contrast != null) {
      final low = max(0.0, 1.0 - contrast!);
      final high = 1.0 + contrast!;
      final cFactor = low + random.nextDouble() * (high - low);
      transforms.add(AdjustContrastOp(factor: cFactor));
    }

    if (saturation != null) {
      final low = max(0.0, 1.0 - saturation!);
      final high = 1.0 + saturation!;
      final sFactor = low + random.nextDouble() * (high - low);
      transforms.add(AdjustSaturationOp(factor: sFactor));
    }

    if (hue != null) {
      final hFactor = -hue! + random.nextDouble() * (2 * hue!);
      transforms.add(AdjustHueOp(factor: hFactor));
    }

    // Shuffle the order of transforms (PyTorch behavior)
    transforms.shuffle(random);

    // Apply transforms sequentially
    TensorBuffer result = contiguous;
    for (final transform in transforms) {
      result = transform.apply(result);
    }

    return result;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
