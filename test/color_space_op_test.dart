import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

/// Creates a 3D RGB tensor [3, H, W] with known pixel values in [0, 1].
TensorBuffer _createRgbTensor3D(int h, int w) {
  final size = 3 * h * w;
  final data = Float32List(size);
  for (int ch = 0; ch < 3; ch++) {
    for (int row = 0; row < h; row++) {
      for (int col = 0; col < w; col++) {
        final idx = ch * h * w + row * w + col;
        data[idx] = (idx % 10) / 10.0;
      }
    }
  }
  return TensorBuffer.fromFloat32List(data, [3, h, w]);
}

/// Creates a 4D RGB tensor [N, 3, H, W] with known pixel values.
TensorBuffer _createRgbTensor4D(int n, int h, int w) {
  final size = n * 3 * h * w;
  final data = Float32List(size);
  for (int i = 0; i < size; i++) {
    data[i] = (i % 10) / 10.0;
  }
  return TensorBuffer.fromFloat32List(data, [n, 3, h, w]);
}

/// Creates a 3D tensor with uniform RGB values.
TensorBuffer _createUniformRgbTensor(
  double r,
  double g,
  double b, {
  int h = 2,
  int w = 2,
}) {
  final hw = h * w;
  final data = Float32List(3 * hw);
  for (int i = 0; i < hw; i++) {
    data[i] = r;
    data[hw + i] = g;
    data[2 * hw + i] = b;
  }
  return TensorBuffer.fromFloat32List(data, [3, h, w]);
}

void main() {
  // ========================================================================
  // RgbToGrayscaleOp
  // ========================================================================

  group('RgbToGrayscaleOp', () {
    group('basic functionality', () {
      test('converts 3D RGB tensor to grayscale', () {
        final tensor = _createUniformRgbTensor(1.0, 0.0, 0.0);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2]));
        // Pure red: Y = 0.2989 * 1.0 + 0.5870 * 0.0 + 0.1140 * 0.0 = 0.2989
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.2989, 1e-4));
        }
      });

      test('converts 4D RGB tensor to grayscale', () {
        final tensor = _createRgbTensor4D(2, 3, 3);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        expect(result.shape, equals([2, 1, 3, 3]));
      });

      test('pure white produces white grayscale', () {
        final tensor = _createUniformRgbTensor(1.0, 1.0, 1.0);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        // Y = 0.2989 + 0.5870 + 0.1140 = 0.9999 ≈ 1.0
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(1.0, 2e-4));
        }
      });

      test('pure black produces black grayscale', () {
        final tensor = _createUniformRgbTensor(0.0, 0.0, 0.0);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-6));
        }
      });

      test('pure green uses correct BT.601 weight', () {
        final tensor = _createUniformRgbTensor(0.0, 1.0, 0.0);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.5870, 1e-4));
        }
      });

      test('pure blue uses correct BT.601 weight', () {
        final tensor = _createUniformRgbTensor(0.0, 0.0, 1.0);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.1140, 1e-4));
        }
      });
    });

    group('shape validation', () {
      test('rejects 2D tensor', () {
        final tensor = TensorBuffer.zeros([4, 4]);
        final op = RgbToGrayscaleOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('rejects 5D tensor', () {
        final tensor = TensorBuffer.zeros([1, 3, 2, 2, 2]);
        final op = RgbToGrayscaleOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('rejects non-3-channel tensor', () {
        final tensor = TensorBuffer.zeros([4, 3, 3]);
        final op = RgbToGrayscaleOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('rejects 4D tensor with wrong channels', () {
        final tensor = TensorBuffer.zeros([1, 4, 3, 3]);
        final op = RgbToGrayscaleOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('metadata', () {
      test('name is correct', () {
        expect(RgbToGrayscaleOp().name, equals('RgbToGrayscale'));
      });

      test('capabilities are correct', () {
        final op = RgbToGrayscaleOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isFalse);
        expect(
          op.capabilities.pytorchEquivalent,
          equals('torchvision.transforms.Grayscale'),
        );
      });

      test('computeOutputShape for 3D', () {
        expect(
          RgbToGrayscaleOp().computeOutputShape([3, 4, 4]),
          equals([1, 4, 4]),
        );
      });

      test('computeOutputShape for 4D', () {
        expect(
          RgbToGrayscaleOp().computeOutputShape([2, 3, 4, 4]),
          equals([2, 1, 4, 4]),
        );
      });

      test('computeOutputShape rejects invalid rank', () {
        expect(
          () => RgbToGrayscaleOp().computeOutputShape([4, 4]),
          throwsA(isA<ShapeMismatchException>()),
        );
      });
    });

    group('different dtypes', () {
      test('works with float64', () {
        final data = Float64List(3 * 4);
        for (int i = 0; i < 4; i++) {
          data[i] = 1.0; // R
          data[4 + i] = 0.0; // G
          data[8 + i] = 0.0; // B
        }
        final tensor = TensorBuffer.fromFloat64List(data, [3, 2, 2]);
        final op = RgbToGrayscaleOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2]));
        expect(result.dtype, equals(DType.float64));
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.2989, 1e-4));
        }
      });
    });
  });

  // ========================================================================
  // RgbToHsvOp
  // ========================================================================

  group('RgbToHsvOp', () {
    group('basic functionality', () {
      test('converts pure red to HSV', () {
        final tensor = _createUniformRgbTensor(1.0, 0.0, 0.0);
        final op = RgbToHsvOp();
        final result = op(tensor);

        expect(result.shape, equals([3, 2, 2]));
        // Pure red: H=0, S=1, V=1
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5)); // H
          expect(result.storage.getAsDouble(4 + i), closeTo(1.0, 1e-5)); // S
          expect(result.storage.getAsDouble(8 + i), closeTo(1.0, 1e-5)); // V
        }
      });

      test('converts pure green to HSV', () {
        final tensor = _createUniformRgbTensor(0.0, 1.0, 0.0);
        final op = RgbToHsvOp();
        final result = op(tensor);

        // Pure green: H=1/3, S=1, V=1
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(1.0 / 3.0, 1e-5)); // H
          expect(result.storage.getAsDouble(4 + i), closeTo(1.0, 1e-5)); // S
          expect(result.storage.getAsDouble(8 + i), closeTo(1.0, 1e-5)); // V
        }
      });

      test('converts pure blue to HSV', () {
        final tensor = _createUniformRgbTensor(0.0, 0.0, 1.0);
        final op = RgbToHsvOp();
        final result = op(tensor);

        // Pure blue: H=2/3, S=1, V=1
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(2.0 / 3.0, 1e-5)); // H
          expect(result.storage.getAsDouble(4 + i), closeTo(1.0, 1e-5)); // S
          expect(result.storage.getAsDouble(8 + i), closeTo(1.0, 1e-5)); // V
        }
      });

      test('converts black to HSV', () {
        final tensor = _createUniformRgbTensor(0.0, 0.0, 0.0);
        final op = RgbToHsvOp();
        final result = op(tensor);

        // Black: H=0, S=0, V=0
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-6));
          expect(result.storage.getAsDouble(4 + i), closeTo(0.0, 1e-6));
          expect(result.storage.getAsDouble(8 + i), closeTo(0.0, 1e-6));
        }
      });

      test('converts white to HSV', () {
        final tensor = _createUniformRgbTensor(1.0, 1.0, 1.0);
        final op = RgbToHsvOp();
        final result = op(tensor);

        // White: H=0, S=0, V=1
        for (int i = 0; i < 4; i++) {
          expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5)); // H
          expect(result.storage.getAsDouble(4 + i), closeTo(0.0, 1e-5)); // S
          expect(result.storage.getAsDouble(8 + i), closeTo(1.0, 1e-5)); // V
        }
      });

      test('converts 4D tensor', () {
        final tensor = _createRgbTensor4D(2, 3, 3);
        final op = RgbToHsvOp();
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 3]));
      });
    });

    group('shape validation', () {
      test('rejects 2D tensor', () {
        final tensor = TensorBuffer.zeros([4, 4]);
        expect(
          () => RgbToHsvOp()(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('rejects non-3-channel tensor', () {
        final tensor = TensorBuffer.zeros([4, 3, 3]);
        expect(
          () => RgbToHsvOp()(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });
    });

    group('metadata', () {
      test('name is correct', () {
        expect(RgbToHsvOp().name, equals('RgbToHsv'));
      });

      test('capabilities are correct', () {
        final op = RgbToHsvOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
      });

      test('computeOutputShape preserves shape', () {
        expect(RgbToHsvOp().computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
        expect(
          RgbToHsvOp().computeOutputShape([2, 3, 4, 4]),
          equals([2, 3, 4, 4]),
        );
      });
    });
  });

  // ========================================================================
  // HsvToRgbOp
  // ========================================================================

  group('HsvToRgbOp', () {
    group('basic functionality', () {
      test('converts pure red HSV to RGB', () {
        // H=0, S=1, V=1 -> R=1, G=0, B=0
        const hw = 4;
        final data = Float32List(3 * hw);
        for (int i = 0; i < hw; i++) {
          data[i] = 0.0; // H
          data[hw + i] = 1.0; // S
          data[2 * hw + i] = 1.0; // V
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);
        final op = HsvToRgbOp();
        final result = op(tensor);

        expect(result.shape, equals([3, 2, 2]));
        for (int i = 0; i < hw; i++) {
          expect(result.storage.getAsDouble(i), closeTo(1.0, 1e-5)); // R
          expect(result.storage.getAsDouble(hw + i), closeTo(0.0, 1e-5)); // G
          expect(
            result.storage.getAsDouble(2 * hw + i),
            closeTo(0.0, 1e-5),
          ); // B
        }
      });

      test('converts 4D tensor', () {
        // Create a 4D HSV tensor
        const n = 2;
        const h = 3;
        const w = 3;
        final data = Float32List(n * 3 * h * w);
        for (int i = 0; i < data.length; i++) {
          data[i] = 0.5; // All 0.5 as generic HSV values
        }
        final tensor = TensorBuffer.fromFloat32List(data, [n, 3, h, w]);
        final op = HsvToRgbOp();
        final result = op(tensor);

        expect(result.shape, equals([2, 3, 3, 3]));
      });
    });

    group('shape validation', () {
      test('rejects 2D tensor', () {
        final tensor = TensorBuffer.zeros([4, 4]);
        expect(
          () => HsvToRgbOp()(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('rejects non-3-channel tensor', () {
        final tensor = TensorBuffer.zeros([1, 3, 3]);
        expect(
          () => HsvToRgbOp()(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });
    });

    group('metadata', () {
      test('name is correct', () {
        expect(HsvToRgbOp().name, equals('HsvToRgb'));
      });

      test('capabilities are correct', () {
        final op = HsvToRgbOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
      });

      test('computeOutputShape preserves shape', () {
        expect(HsvToRgbOp().computeOutputShape([3, 4, 4]), equals([3, 4, 4]));
        expect(
          HsvToRgbOp().computeOutputShape([2, 3, 4, 4]),
          equals([2, 3, 4, 4]),
        );
      });
    });
  });

  // ========================================================================
  // Round-trip tests
  // ========================================================================

  group('RGB <-> HSV round-trip', () {
    test('RGB -> HSV -> RGB preserves values (3D)', () {
      final original = _createRgbTensor3D(4, 4);
      final toHsv = RgbToHsvOp();
      final toRgb = HsvToRgbOp();

      final hsv = toHsv(original);
      final restored = toRgb(hsv);

      expect(restored.shape, equals(original.shape));
      final numel = original.numel;
      for (int i = 0; i < numel; i++) {
        expect(
          restored.storage.getAsDouble(i),
          closeTo(original.storage.getAsDouble(i), 1e-5),
        );
      }
    });

    test('RGB -> HSV -> RGB preserves values (4D)', () {
      final original = _createRgbTensor4D(2, 3, 3);
      final toHsv = RgbToHsvOp();
      final toRgb = HsvToRgbOp();

      final hsv = toHsv(original);
      final restored = toRgb(hsv);

      expect(restored.shape, equals(original.shape));
      final numel = original.numel;
      for (int i = 0; i < numel; i++) {
        expect(
          restored.storage.getAsDouble(i),
          closeTo(original.storage.getAsDouble(i), 1e-5),
        );
      }
    });

    test('round-trip with uniform color', () {
      final original = _createUniformRgbTensor(0.3, 0.6, 0.9);
      final toHsv = RgbToHsvOp();
      final toRgb = HsvToRgbOp();

      final restored = toRgb(toHsv(original));

      final numel = original.numel;
      for (int i = 0; i < numel; i++) {
        expect(
          restored.storage.getAsDouble(i),
          closeTo(original.storage.getAsDouble(i), 1e-5),
        );
      }
    });
  });

  // ========================================================================
  // Pipeline integration
  // ========================================================================

  group('pipeline integration', () {
    test('works in TensorPipeline', () {
      final pipeline = TensorPipeline([RgbToGrayscaleOp()]);

      final input = _createRgbTensor3D(4, 4);
      final result = pipeline.run(input);
      expect(result.shape, equals([1, 4, 4]));
    });

    test('RGB -> Grayscale pipeline shape computation', () {
      final pipeline = TensorPipeline([RgbToGrayscaleOp()]);

      expect(pipeline.computeOutputShape([3, 224, 224]), equals([1, 224, 224]));
    });

    test('RGB -> HSV -> RGB pipeline', () {
      final pipeline = TensorPipeline([RgbToHsvOp(), HsvToRgbOp()]);

      final input = _createRgbTensor3D(4, 4);
      final result = pipeline.run(input);
      expect(result.shape, equals([3, 4, 4]));
    });
  });

  // ========================================================================
  // Float64 dtype tests
  // ========================================================================

  group('float64 dtype', () {
    test('RgbToHsvOp with float64', () {
      final data = Float64List(3 * 4);
      for (int i = 0; i < 4; i++) {
        data[i] = 1.0; // R
        data[4 + i] = 0.0; // G
        data[8 + i] = 0.0; // B
      }
      final tensor = TensorBuffer.fromFloat64List(data, [3, 2, 2]);
      final op = RgbToHsvOp();
      final result = op(tensor);

      expect(result.dtype, equals(DType.float64));
      expect(result.shape, equals([3, 2, 2]));
      // Pure red -> H=0, S=1, V=1
      for (int i = 0; i < 4; i++) {
        expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-10));
        expect(result.storage.getAsDouble(4 + i), closeTo(1.0, 1e-10));
        expect(result.storage.getAsDouble(8 + i), closeTo(1.0, 1e-10));
      }
    });

    test('HsvToRgbOp with float64', () {
      final data = Float64List(3 * 4);
      for (int i = 0; i < 4; i++) {
        data[i] = 1.0 / 3.0; // H (green)
        data[4 + i] = 1.0; // S
        data[8 + i] = 1.0; // V
      }
      final tensor = TensorBuffer.fromFloat64List(data, [3, 2, 2]);
      final op = HsvToRgbOp();
      final result = op(tensor);

      expect(result.dtype, equals(DType.float64));
      // Green: R=0, G=1, B=0
      for (int i = 0; i < 4; i++) {
        expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-10));
        expect(result.storage.getAsDouble(4 + i), closeTo(1.0, 1e-10));
        expect(result.storage.getAsDouble(8 + i), closeTo(0.0, 1e-10));
      }
    });

    test('round-trip with float64 preserves precision', () {
      final data = Float64List(3 * 4);
      for (int i = 0; i < 4; i++) {
        data[i] = 0.123456789012345; // R
        data[4 + i] = 0.678901234567890; // G
        data[8 + i] = 0.345678901234567; // B
      }
      final original = TensorBuffer.fromFloat64List(data, [3, 2, 2]);

      final restored = HsvToRgbOp()(RgbToHsvOp()(original));

      expect(restored.dtype, equals(DType.float64));
      for (int i = 0; i < original.numel; i++) {
        expect(
          restored.storage.getAsDouble(i),
          closeTo(original.storage.getAsDouble(i), 1e-10),
        );
      }
    });
  });

  // ========================================================================
  // Edge cases
  // ========================================================================

  group('edge cases', () {
    test('grayscale with 1x1 image', () {
      final data = Float32List.fromList([0.5, 0.3, 0.7]);
      final tensor = TensorBuffer.fromFloat32List(data, [3, 1, 1]);
      final op = RgbToGrayscaleOp();
      final result = op(tensor);

      expect(result.shape, equals([1, 1, 1]));
      const expected = 0.5 * 0.2989 + 0.3 * 0.5870 + 0.7 * 0.1140;
      expect(result.storage.getAsDouble(0), closeTo(expected, 1e-4));
    });

    test('HSV conversion with gray (equal RGB)', () {
      final tensor = _createUniformRgbTensor(0.5, 0.5, 0.5);
      final op = RgbToHsvOp();
      final result = op(tensor);

      // Gray: H=0, S=0, V=0.5
      for (int i = 0; i < 4; i++) {
        expect(result.storage.getAsDouble(i), closeTo(0.0, 1e-5)); // H
        expect(result.storage.getAsDouble(4 + i), closeTo(0.0, 1e-5)); // S
        expect(result.storage.getAsDouble(8 + i), closeTo(0.5, 1e-5)); // V
      }
    });

    test('toString returns expected format', () {
      expect(
        RgbToGrayscaleOp().toString(),
        equals('TransformOp(RgbToGrayscale)'),
      );
      expect(RgbToHsvOp().toString(), equals('TransformOp(RgbToHsv)'));
      expect(HsvToRgbOp().toString(), equals('TransformOp(HsvToRgb)'));
    });
  });
}
