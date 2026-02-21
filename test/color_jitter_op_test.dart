import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/color_jitter_op.dart';

/// Creates a 3D RGB tensor [3, H, W] with known pixel values in [0, 1].
TensorBuffer _createRgbTensor3D(int h, int w, {DType dtype = DType.float32}) {
  final size = 3 * h * w;
  final data = Float32List(size);
  // Fill with gradient values in [0, 1]
  for (int ch = 0; ch < 3; ch++) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final idx = ch * h * w + row * w + col;
        data[idx] = (idx % 10) / 10.0; // 0.0, 0.1, ..., 0.9, 0.0, ...
      }
    }
  }
  return TensorBuffer.fromFloat32List(data, [3, h, w]);
}

/// Creates a 4D RGB tensor [N, 3, H, W] with known pixel values in [0, 1].
TensorBuffer _createRgbTensor4D(int n, int h, int w,
    {DType dtype = DType.float32}) {
  final size = n * 3 * h * w;
  final data = Float32List(size);
  for (int batch = 0; batch < n; batch++) {
    for (int ch = 0; ch < 3; ch++) {
      for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
          final idx = batch * 3 * h * w + ch * h * w + row * w + col;
          data[idx] = (idx % 10) / 10.0;
        }
      }
    }
  }
  return TensorBuffer.fromFloat32List(data, [n, 3, h, w]);
}

/// Creates a 3D tensor with uniform RGB values for all pixels.
TensorBuffer _createUniformRgbTensor(double r, double g, double b,
    {int h = 2, int w = 2}) {
  final size = 3 * h * w;
  final data = Float32List(size);
  for (int i = 0; i < h * w; i++) {
    data[i] = r;
    data[h * w + i] = g;
    data[2 * h * w + i] = b;
  }
  return TensorBuffer.fromFloat32List(data, [3, h, w]);
}

void main() {
  // ==========================================================================
  // RGB <-> HSV Conversion Tests
  // ==========================================================================
  group('RGB <-> HSV conversion', () {
    test('pure red converts correctly', () {
      final (h, s, v) = rgbToHsv(1.0, 0.0, 0.0);
      expect(h, closeTo(0.0, 1e-6));
      expect(s, closeTo(1.0, 1e-6));
      expect(v, closeTo(1.0, 1e-6));
    });

    test('pure green converts correctly', () {
      final (h, s, v) = rgbToHsv(0.0, 1.0, 0.0);
      expect(h, closeTo(1.0 / 3.0, 1e-6));
      expect(s, closeTo(1.0, 1e-6));
      expect(v, closeTo(1.0, 1e-6));
    });

    test('pure blue converts correctly', () {
      final (h, s, v) = rgbToHsv(0.0, 0.0, 1.0);
      expect(h, closeTo(2.0 / 3.0, 1e-6));
      expect(s, closeTo(1.0, 1e-6));
      expect(v, closeTo(1.0, 1e-6));
    });

    test('white converts correctly', () {
      final (h, s, v) = rgbToHsv(1.0, 1.0, 1.0);
      expect(h, closeTo(0.0, 1e-6));
      expect(s, closeTo(0.0, 1e-6));
      expect(v, closeTo(1.0, 1e-6));
    });

    test('black converts correctly', () {
      final (h, s, v) = rgbToHsv(0.0, 0.0, 0.0);
      expect(h, closeTo(0.0, 1e-6));
      expect(s, closeTo(0.0, 1e-6));
      expect(v, closeTo(0.0, 1e-6));
    });

    test('gray converts correctly', () {
      final (h, s, v) = rgbToHsv(0.5, 0.5, 0.5);
      expect(s, closeTo(0.0, 1e-6));
      expect(v, closeTo(0.5, 1e-6));
    });

    test('roundtrip RGB -> HSV -> RGB preserves values', () {
      final testColors = [
        (0.8, 0.2, 0.4),
        (0.1, 0.9, 0.3),
        (0.5, 0.5, 0.7),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
      ];

      for (final (r, g, b) in testColors) {
        final (h, s, v) = rgbToHsv(r, g, b);
        final (nr, ng, nb) = hsvToRgb(h, s, v);
        expect(nr, closeTo(r, 1e-6), reason: 'Red mismatch for ($r, $g, $b)');
        expect(ng, closeTo(g, 1e-6),
            reason: 'Green mismatch for ($r, $g, $b)');
        expect(nb, closeTo(b, 1e-6), reason: 'Blue mismatch for ($r, $g, $b)');
      }
    });

    test('HSV -> RGB -> HSV roundtrip preserves values for chromatic colors',
        () {
      // Only test chromatic (s > 0) colors since achromatic hue is undefined
      final testHsvs = [
        (0.0, 1.0, 1.0), // pure red
        (0.5, 0.5, 0.8),
        (0.25, 0.7, 0.6),
        (0.75, 0.9, 0.4),
      ];

      for (final (h, s, v) in testHsvs) {
        final (r, g, b) = hsvToRgb(h, s, v);
        final (nh, ns, nv) = rgbToHsv(r, g, b);
        expect(nh, closeTo(h, 1e-6), reason: 'Hue mismatch for ($h, $s, $v)');
        expect(ns, closeTo(s, 1e-6),
            reason: 'Saturation mismatch for ($h, $s, $v)');
        expect(nv, closeTo(v, 1e-6),
            reason: 'Value mismatch for ($h, $s, $v)');
      }
    });
  });

  // ==========================================================================
  // AdjustBrightnessOp Tests
  // ==========================================================================
  group('AdjustBrightnessOp', () {
    group('basic functionality', () {
      test('increases brightness on 3D tensor', () {
        final tensor = _createUniformRgbTensor(0.5, 0.5, 0.5);
        final op = AdjustBrightnessOp(factor: 0.2);
        final result = op(tensor);

        expect(result.shape, equals([3, 2, 2]));
        // All pixels should be 0.5 + 0.2 = 0.7
        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.7, 1e-5));
        }
      });

      test('decreases brightness on 3D tensor', () {
        final tensor = _createUniformRgbTensor(0.5, 0.5, 0.5);
        final op = AdjustBrightnessOp(factor: -0.3);
        final result = op(tensor);

        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.2, 1e-5));
        }
      });

      test('factor 0 returns same values', () {
        final tensor = _createUniformRgbTensor(0.5, 0.3, 0.7);
        final op = AdjustBrightnessOp(factor: 0.0);
        final result = op(tensor);

        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i),
              closeTo(tensor.storage.getAsDouble(i), 1e-5));
        }
      });

      test('clamps to [0, 1]', () {
        final tensor = _createUniformRgbTensor(0.9, 0.9, 0.9);
        final op = AdjustBrightnessOp(factor: 0.5);
        final result = op(tensor);

        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i), closeTo(1.0, 1e-5));
        }
      });

      test('clamps negative to 0', () {
        final tensor = _createUniformRgbTensor(0.1, 0.1, 0.1);
        final op = AdjustBrightnessOp(factor: -0.5);
        final result = op(tensor);

        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5));
        }
      });

      test('works on 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 3, 4);
        final op = AdjustBrightnessOp(factor: 0.1);
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 4]));
        // Verify values are within [0, 1]
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(0.0));
          expect(val, lessThanOrEqualTo(1.0));
        }
      });
    });

    group('shape validation', () {
      test('throws for 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);
        final op = AdjustBrightnessOp(factor: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for non-3-channel tensor', () {
        final data = Float32List(8);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);
        final op = AdjustBrightnessOp(factor: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('preserves shape', () {
        final op = AdjustBrightnessOp(factor: 0.1);
        expect(op.computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
        expect(op.computeOutputShape([2, 3, 4, 4]), equals([2, 3, 4, 4]));
      });

      test('name includes factor', () {
        final op = AdjustBrightnessOp(factor: 0.2);
        expect(op.name, contains('0.2'));
        expect(op.name, contains('AdjustBrightness'));
      });
    });
  });

  // ==========================================================================
  // AdjustContrastOp Tests
  // ==========================================================================
  group('AdjustContrastOp', () {
    group('constructor validation', () {
      test('throws for negative factor', () {
        expect(
          () => AdjustContrastOp(factor: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts zero factor', () {
        expect(() => AdjustContrastOp(factor: 0.0), returnsNormally);
      });

      test('accepts positive factor', () {
        expect(() => AdjustContrastOp(factor: 2.0), returnsNormally);
      });
    });

    group('basic functionality', () {
      test('factor 1 returns same values', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op = AdjustContrastOp(factor: 1.0);
        final result = op(tensor);

        expect(result.shape, equals(tensor.shape));
        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i),
              closeTo(tensor.storage.getAsDouble(i), 1e-5));
        }
      });

      test('factor 0 produces per-channel mean', () {
        final tensor = _createUniformRgbTensor(0.2, 0.6, 0.8);
        final op = AdjustContrastOp(factor: 0.0);
        final result = op(tensor);

        // When all pixels in channel have same value, mean = that value,
        // so (val - mean) * 0 + mean = mean = original value
        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.2, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(0.6, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(0.8, 1e-5));
        }
      });

      test('factor > 1 increases contrast', () {
        // Create a tensor with varying values
        final data = Float32List.fromList([
          // R channel
          0.3, 0.5, 0.7, 0.9,
          // G channel
          0.2, 0.4, 0.6, 0.8,
          // B channel
          0.1, 0.3, 0.5, 0.7,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);
        final op = AdjustContrastOp(factor: 2.0);
        final result = op(tensor);

        // R mean = (0.3+0.5+0.7+0.9)/4 = 0.6
        // New values: (0.3-0.6)*2+0.6=0.0, (0.5-0.6)*2+0.6=0.4, etc.
        expect(result.storage.getAsDouble(0), closeTo(0.0, 1e-5));
        expect(result.storage.getAsDouble(1), closeTo(0.4, 1e-5));
        expect(result.storage.getAsDouble(2), closeTo(0.8, 1e-5));
        expect(result.storage.getAsDouble(3), closeTo(1.0, 1e-5)); // clamped
      });

      test('works on 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 3, 4);
        final op = AdjustContrastOp(factor: 1.5);
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 4]));
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(0.0));
          expect(val, lessThanOrEqualTo(1.0));
        }
      });
    });

    group('shape validation', () {
      test('throws for non-3-channel tensor', () {
        final data = Float32List(16);
        final tensor = TensorBuffer.fromFloat32List(data, [4, 2, 2]);
        final op = AdjustContrastOp(factor: 1.0);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('preserves shape', () {
        final op = AdjustContrastOp(factor: 1.5);
        expect(op.computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
      });

      test('name includes factor', () {
        final op = AdjustContrastOp(factor: 1.5);
        expect(op.name, contains('1.5'));
        expect(op.name, contains('AdjustContrast'));
      });
    });
  });

  // ==========================================================================
  // AdjustSaturationOp Tests
  // ==========================================================================
  group('AdjustSaturationOp', () {
    group('constructor validation', () {
      test('throws for negative factor', () {
        expect(
          () => AdjustSaturationOp(factor: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts zero factor', () {
        expect(() => AdjustSaturationOp(factor: 0.0), returnsNormally);
      });
    });

    group('basic functionality', () {
      test('factor 1 returns same values', () {
        final tensor = _createUniformRgbTensor(0.8, 0.2, 0.4);
        final op = AdjustSaturationOp(factor: 1.0);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.8, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(0.2, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(0.4, 1e-5));
        }
      });

      test('factor 0 produces grayscale-like output', () {
        final tensor = _createUniformRgbTensor(1.0, 0.0, 0.0); // Pure red
        final op = AdjustSaturationOp(factor: 0.0);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        // With saturation=0, HSV becomes (h, 0, v) which maps to (v, v, v)
        for (int i = 0; i < h * w; i++) {
          final r = result.storage.getAsDouble(i);
          final g = result.storage.getAsDouble(h * w + i);
          final b = result.storage.getAsDouble(2 * h * w + i);
          // All channels should be equal (grayscale)
          expect(r, closeTo(g, 1e-5));
          expect(g, closeTo(b, 1e-5));
        }
      });

      test('works on 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 3, 4);
        final op = AdjustSaturationOp(factor: 1.5);
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 4]));
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(-1e-6));
          expect(val, lessThanOrEqualTo(1.0 + 1e-6));
        }
      });

      test('does not modify input tensor', () {
        final tensor = _createUniformRgbTensor(0.8, 0.2, 0.4);
        final originalR = tensor.storage.getAsDouble(0);
        final op = AdjustSaturationOp(factor: 2.0);
        op(tensor);

        expect(tensor.storage.getAsDouble(0), equals(originalR));
      });
    });

    group('shape validation', () {
      test('throws for non-3-channel tensor', () {
        final data = Float32List(8);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);
        final op = AdjustSaturationOp(factor: 1.0);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('preserves shape', () {
        final op = AdjustSaturationOp(factor: 1.5);
        expect(op.computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
      });

      test('name includes factor', () {
        final op = AdjustSaturationOp(factor: 1.5);
        expect(op.name, contains('1.5'));
        expect(op.name, contains('AdjustSaturation'));
      });
    });
  });

  // ==========================================================================
  // AdjustHueOp Tests
  // ==========================================================================
  group('AdjustHueOp', () {
    group('constructor validation', () {
      test('throws for factor > 0.5', () {
        expect(
          () => AdjustHueOp(factor: 0.6),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for factor < -0.5', () {
        expect(
          () => AdjustHueOp(factor: -0.6),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts boundary values', () {
        expect(() => AdjustHueOp(factor: 0.5), returnsNormally);
        expect(() => AdjustHueOp(factor: -0.5), returnsNormally);
        expect(() => AdjustHueOp(factor: 0.0), returnsNormally);
      });
    });

    group('basic functionality', () {
      test('factor 0 returns same values', () {
        final tensor = _createUniformRgbTensor(0.8, 0.2, 0.4);
        final op = AdjustHueOp(factor: 0.0);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.8, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(0.2, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(0.4, 1e-5));
        }
      });

      test('shifting red by 1/3 produces green', () {
        // Pure red -> shift hue by 1/3 (120 degrees) -> should become green
        final tensor = _createUniformRgbTensor(1.0, 0.0, 0.0);
        final op = AdjustHueOp(factor: 1.0 / 3.0);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(1.0, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(0.0, 1e-5));
        }
      });

      test('shifting red by -1/3 produces blue', () {
        final tensor = _createUniformRgbTensor(1.0, 0.0, 0.0);
        final op = AdjustHueOp(factor: -1.0 / 3.0);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(0.0, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(1.0, 1e-5));
        }
      });

      test('does not affect grayscale pixels', () {
        final tensor = _createUniformRgbTensor(0.5, 0.5, 0.5);
        final op = AdjustHueOp(factor: 0.3);
        final result = op(tensor);

        final h = tensor.shape[1];
        final w = tensor.shape[2];
        for (int i = 0; i < h * w; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.5, 1e-5));
          expect(
              result.storage.getAsDouble(h * w + i), closeTo(0.5, 1e-5));
          expect(
              result.storage.getAsDouble(2 * h * w + i), closeTo(0.5, 1e-5));
        }
      });

      test('works on 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 3, 4);
        final op = AdjustHueOp(factor: 0.1);
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 4]));
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(-1e-6));
          expect(val, lessThanOrEqualTo(1.0 + 1e-6));
        }
      });
    });

    group('shape validation', () {
      test('throws for non-RGB tensor', () {
        final data = Float32List(8);
        final tensor = TensorBuffer.fromFloat32List(data, [4, 2, 1]);
        final op = AdjustHueOp(factor: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 1D tensor', () {
        final data = Float32List(3);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);
        final op = AdjustHueOp(factor: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('preserves shape', () {
        final op = AdjustHueOp(factor: 0.1);
        expect(op.computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
      });

      test('name includes factor', () {
        final op = AdjustHueOp(factor: 0.1);
        expect(op.name, contains('0.1'));
        expect(op.name, contains('AdjustHue'));
      });
    });
  });

  // ==========================================================================
  // ColorJitterOp Tests
  // ==========================================================================
  group('ColorJitterOp', () {
    group('constructor validation', () {
      test('throws for negative brightness', () {
        expect(
          () => ColorJitterOp(brightness: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for negative contrast', () {
        expect(
          () => ColorJitterOp(contrast: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for negative saturation', () {
        expect(
          () => ColorJitterOp(saturation: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for hue > 0.5', () {
        expect(
          () => ColorJitterOp(hue: 0.6),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for negative hue', () {
        expect(
          () => ColorJitterOp(hue: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts all null parameters', () {
        expect(() => ColorJitterOp(), returnsNormally);
      });

      test('accepts valid parameters', () {
        expect(
          () => ColorJitterOp(
            brightness: 0.2,
            contrast: 0.3,
            saturation: 0.4,
            hue: 0.1,
          ),
          returnsNormally,
        );
      });
    });

    group('basic functionality', () {
      test('applies all transforms on 3D tensor', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op = ColorJitterOp(
          brightness: 0.2,
          contrast: 0.2,
          saturation: 0.2,
          hue: 0.1,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.shape, equals([3, 4, 4]));
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(-1e-6));
          expect(val, lessThanOrEqualTo(1.0 + 1e-6));
        }
      });

      test('applies all transforms on 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 4, 4);
        final op = ColorJitterOp(
          brightness: 0.1,
          contrast: 0.1,
          saturation: 0.1,
          hue: 0.05,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 4, 4]));
      });

      test('no-op when all parameters are null', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op = ColorJitterOp(seed: 42);
        final result = op(tensor);

        // With no adjustments, should return same values
        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i),
              closeTo(tensor.storage.getAsDouble(i), 1e-5));
        }
      });

      test('only brightness when others are null', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op = ColorJitterOp(brightness: 0.1, seed: 42);
        final result = op(tensor);

        expect(result.shape, equals([3, 4, 4]));
        // All values should still be in [0, 1]
        for (int i = 0; i < result.numel; i++) {
          final val = result.storage.getAsDouble(i);
          expect(val, greaterThanOrEqualTo(-1e-6));
          expect(val, lessThanOrEqualTo(1.0 + 1e-6));
        }
      });
    });

    group('deterministic behavior', () {
      test('same seed produces same result', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op1 = ColorJitterOp(
          brightness: 0.2,
          contrast: 0.2,
          saturation: 0.2,
          hue: 0.1,
          seed: 123,
        );
        final op2 = ColorJitterOp(
          brightness: 0.2,
          contrast: 0.2,
          saturation: 0.2,
          hue: 0.1,
          seed: 123,
        );

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        for (int i = 0; i < result1.numel; i++) {
          expect(result1.storage.getAsDouble(i),
              closeTo(result2.storage.getAsDouble(i), 1e-6));
        }
      });

      test('different seeds produce different results', () {
        final tensor = _createRgbTensor3D(4, 4);
        final op1 = ColorJitterOp(
          brightness: 0.5,
          contrast: 0.5,
          saturation: 0.5,
          hue: 0.3,
          seed: 1,
        );
        final op2 = ColorJitterOp(
          brightness: 0.5,
          contrast: 0.5,
          saturation: 0.5,
          hue: 0.3,
          seed: 999,
        );

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        // At least some values should differ
        bool anyDifferent = false;
        for (int i = 0; i < result1.numel; i++) {
          if ((result1.storage.getAsDouble(i) -
                      result2.storage.getAsDouble(i))
                  .abs() >
              1e-6) {
            anyDifferent = true;
            break;
          }
        }
        expect(anyDifferent, isTrue);
      });
    });

    group('shape validation', () {
      test('throws for non-RGB tensor', () {
        final data = Float32List(8);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);
        final op = ColorJitterOp(brightness: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 2D tensor', () {
        final data = Float32List(6);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
        final op = ColorJitterOp(brightness: 0.1);

        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('preserves shape', () {
        final op = ColorJitterOp(brightness: 0.1);
        expect(op.computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
        expect(op.computeOutputShape([2, 3, 4, 4]), equals([2, 3, 4, 4]));
      });

      test('name includes all parameters', () {
        final op = ColorJitterOp(
          brightness: 0.2,
          contrast: 0.3,
          saturation: 0.4,
          hue: 0.1,
        );
        expect(op.name, contains('ColorJitter'));
        expect(op.name, contains('brightness'));
        expect(op.name, contains('contrast'));
        expect(op.name, contains('saturation'));
        expect(op.name, contains('hue'));
      });

      test('capabilities are correct', () {
        final op = ColorJitterOp(brightness: 0.1);
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
      });
    });
  });

  // ==========================================================================
  // Different DType Tests
  // ==========================================================================
  group('different dtypes', () {
    test('AdjustBrightnessOp works with float64', () {
      final data = Float64List.fromList([
        0.5, 0.5, 0.5, 0.5, // R
        0.5, 0.5, 0.5, 0.5, // G
        0.5, 0.5, 0.5, 0.5, // B
      ]);
      final tensor = TensorBuffer.fromFloat64List(data, [3, 2, 2]);
      final op = AdjustBrightnessOp(factor: 0.2);
      final result = op(tensor);

      expect(result.dtype, equals(DType.float64));
      for (int i = 0; i < result.numel; i++) {
        expect(result.storage.getAsDouble(i), closeTo(0.7, 1e-5));
      }
    });

    test('AdjustContrastOp works with float64', () {
      final data = Float64List.fromList([
        0.3, 0.5, 0.7, 0.9, // R
        0.2, 0.4, 0.6, 0.8, // G
        0.1, 0.3, 0.5, 0.7, // B
      ]);
      final tensor = TensorBuffer.fromFloat64List(data, [3, 2, 2]);
      final op = AdjustContrastOp(factor: 1.0);
      final result = op(tensor);

      expect(result.dtype, equals(DType.float64));
      for (int i = 0; i < result.numel; i++) {
        expect(result.storage.getAsDouble(i),
            closeTo(tensor.storage.getAsDouble(i), 1e-5));
      }
    });
  });

  // ==========================================================================
  // Edge Cases
  // ==========================================================================
  group('edge cases', () {
    test('handles minimum tensor size (3, 1, 1)', () {
      final data = Float32List.fromList([0.5, 0.3, 0.7]);
      final tensor = TensorBuffer.fromFloat32List(data, [3, 1, 1]);

      final brightness = AdjustBrightnessOp(factor: 0.1);
      expect(brightness(tensor).shape, equals([3, 1, 1]));

      final contrast = AdjustContrastOp(factor: 1.5);
      expect(contrast(tensor).shape, equals([3, 1, 1]));

      final saturation = AdjustSaturationOp(factor: 1.5);
      expect(saturation(tensor).shape, equals([3, 1, 1]));

      final hue = AdjustHueOp(factor: 0.1);
      expect(hue(tensor).shape, equals([3, 1, 1]));
    });

    test('handles black image (all zeros)', () {
      final tensor = _createUniformRgbTensor(0.0, 0.0, 0.0);

      final saturation = AdjustSaturationOp(factor: 2.0);
      final result = saturation(tensor);
      for (int i = 0; i < result.numel; i++) {
        expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5));
      }
    });

    test('handles white image (all ones)', () {
      final tensor = _createUniformRgbTensor(1.0, 1.0, 1.0);

      final hue = AdjustHueOp(factor: 0.3);
      final result = hue(tensor);
      // White has no saturation, so hue shift shouldn't change anything
      for (int i = 0; i < result.numel; i++) {
        expect(result.storage.getAsDouble(i), closeTo(1.0, 1e-5));
      }
    });

    test('ColorJitterOp with zero ranges returns same values', () {
      final tensor = _createRgbTensor3D(4, 4);
      final op = ColorJitterOp(
        brightness: 0.0,
        contrast: 0.0,
        saturation: 0.0,
        hue: 0.0,
        seed: 42,
      );
      final result = op(tensor);

      for (int i = 0; i < result.numel; i++) {
        expect(result.storage.getAsDouble(i),
            closeTo(tensor.storage.getAsDouble(i), 1e-5));
      }
    });
  });
}
