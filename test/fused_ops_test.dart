import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('ResizeNormalizeFusedOp', () {
    group('constructor validation', () {
      test('throws for mismatched mean/std lengths', () {
        expect(
          () => ResizeNormalizeFusedOp(
            height: 224,
            width: 224,
            mean: [0.485, 0.456],
            std: [0.229],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for empty mean/std', () {
        expect(
          () => ResizeNormalizeFusedOp(
            height: 224,
            width: 224,
            mean: [],
            std: [],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for zero std', () {
        expect(
          () => ResizeNormalizeFusedOp(
            height: 224,
            width: 224,
            mean: [0.5],
            std: [0.0],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for non-positive dimensions', () {
        expect(
          () => ResizeNormalizeFusedOp(
            height: 0,
            width: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for negative width', () {
        expect(
          () => ResizeNormalizeFusedOp(
            height: 224,
            width: -1,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });
    });

    group('imagenet factory', () {
      test('creates with ImageNet statistics', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.mean, [0.485, 0.456, 0.406]);
        expect(op.std, [0.229, 0.224, 0.225]);
        expect(op.height, 224);
        expect(op.width, 224);
      });

      test('defaults to alignCorners=false', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.alignCorners, isFalse);
      });
    });

    group('equivalence with separate ops', () {
      test('3D tensor matches ResizeOp + NormalizeOp', () {
        // Create a 3-channel 4x4 input
        final data = Float32List(3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length; // Values in [0, 1)
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 4, 4]);

        // Fused path
        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        // Separate path
        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, separateResult.shape);

        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });

      test('4D tensor matches ResizeOp + NormalizeOp', () {
        // Create a batch of 2, 3-channel 4x4 input
        final data = Float32List(2 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [2, 3, 4, 4]);

        // Fused path
        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        // Separate path
        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, separateResult.shape);

        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('shape computation', () {
      test('computeOutputShape for 3D input', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.computeOutputShape([3, 100, 100]), [3, 224, 224]);
      });

      test('computeOutputShape for 4D input', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.computeOutputShape([2, 3, 100, 100]), [2, 3, 224, 224]);
      });

      test('computeOutputShape for 2D input returns modified shape', () {
        // computeOutputShape doesn't validate rank, it just computes
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        // For 2D, it falls through to 4D branch logic
        expect(op.computeOutputShape([4, 4]), [4, 4, 224, 224]);
      });

      test('rejects 2D input', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        final input = TensorBuffer.zeros([4, 4]);
        expect(() => op.apply(input), throwsA(isA<ShapeMismatchException>()));
      });

      test('rejects channel mismatch', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final input = TensorBuffer.zeros([1, 4, 4]); // 1 channel vs 3 expected
        expect(() => op.apply(input), throwsA(isA<ShapeMismatchException>()));
      });

      test('rejects 5D input', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final input = TensorBuffer.zeros([1, 2, 3, 4, 4]);
        expect(() => op.apply(input), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isFalse);
      });
    });

    group('name', () {
      test('includes parameters', () {
        final op = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
        expect(op.name, contains('ResizeNormalize'));
        expect(op.name, contains('224'));
        expect(op.name, contains('alignCorners=false'));
      });

      test('includes alignCorners=true when set', () {
        final op = ResizeNormalizeFusedOp.imagenet(
          height: 224,
          width: 224,
          alignCorners: true,
        );
        expect(op.name, contains('alignCorners=true'));
      });
    });

    group('non-contiguous input', () {
      test('handles transposed (non-contiguous) 3D tensor', () {
        // Create a 3×4×4 tensor and transpose to make it non-contiguous
        final data = Float32List(3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 4, 4]);
        final transposed = input.transpose([0, 2, 1]); // Non-contiguous

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        // Should not throw; ensureContiguous handles it
        final result = fused.apply(transposed);
        expect(result.shape, [3, 2, 2]);
        expect(result.isContiguous, isTrue);
      });

      test('handles transposed (non-contiguous) 4D tensor', () {
        final data = Float32List(2 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [2, 3, 4, 4]);
        final transposed = input.transpose([0, 1, 3, 2]); // Non-contiguous

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final result = fused.apply(transposed);
        expect(result.shape, [2, 3, 2, 2]);
        expect(result.isContiguous, isTrue);
      });
    });

    group('upscaling equivalence', () {
      test('2x2 to 4x4 matches separate ops', () {
        final data = Float32List(3 * 2 * 2);
        for (int i = 0; i < data.length; i++) {
          data[i] = (i + 1) / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 4, width: 4);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 4, width: 4);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, separateResult.shape);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('large tensor blocking', () {
      test('3x100x100 to 3x80x80 exercises block boundaries', () {
        final data = Float32List(3 * 100 * 100);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 100, 100]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 80, width: 80);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 80, width: 80);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [3, 80, 80]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('float64 dtype', () {
      test('generic fallback path matches separate ops', () {
        final data = Float64List(3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat64List(data, [3, 4, 4]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, separateResult.shape);
        for (int i = 0; i < fusedResult.shape.reduce((a, b) => a * b); i++) {
          expect(
            fusedResult.storage.getAsDouble(i),
            closeTo(separateResult.storage.getAsDouble(i), 1e-10),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('single channel', () {
      test('1-channel tensor with custom mean/std', () {
        final data = Float32List(1 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [1, 4, 4]);

        final fused = ResizeNormalizeFusedOp(
          height: 2,
          width: 2,
          mean: [0.5],
          std: [0.25],
        );
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp(mean: [0.5], std: [0.25]);
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [1, 2, 2]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('multi-channel (4+)', () {
      test('4-channel tensor with custom mean/std', () {
        final data = Float32List(4 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [4, 4, 4]);

        final fused = ResizeNormalizeFusedOp(
          height: 2,
          width: 2,
          mean: [0.1, 0.2, 0.3, 0.4],
          std: [0.1, 0.2, 0.3, 0.4],
        );
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp(
          mean: [0.1, 0.2, 0.3, 0.4],
          std: [0.1, 0.2, 0.3, 0.4],
        );
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [4, 2, 2]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('batch=1 4D', () {
      test('single batch 4D tensor matches separate ops', () {
        final data = Float32List(1 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [1, 3, 4, 4]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [1, 3, 2, 2]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('edge cases', () {
      test('1x1 input upscales correctly', () {
        // Single pixel per channel: [3, 1, 1]
        final data = Float32List.fromList([0.2, 0.4, 0.6]);
        final input = TensorBuffer.fromFloat32List(data, [3, 1, 1]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [3, 2, 2]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });

      test('same-size resize only normalizes', () {
        final data = Float32List(3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 4, 4]);

        // Same size: 4x4 -> 4x4
        final fused = ResizeNormalizeFusedOp.imagenet(height: 4, width: 4);
        final fusedResult = fused.apply(input);

        // Only normalize (no resize needed conceptually)
        final normalize = NormalizeOp.imagenet();
        final normalizeOnly = normalize.apply(input);

        expect(fusedResult.shape, normalizeOnly.shape);
        final fusedData = fusedResult.storage.data as Float32List;
        final normalizeData = normalizeOnly.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(normalizeData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });

      test('alignCorners with height=1 (scale=0 case)', () {
        // Input [3, 1, 4] with alignCorners=true, output [3, 1, 2]
        final data = Float32List(3 * 1 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 1, 4]);

        final fused = ResizeNormalizeFusedOp.imagenet(
          height: 1,
          width: 2,
          alignCorners: true,
        );
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 1, width: 2, alignCorners: true);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [3, 1, 2]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });

      test('very large upscale (25x) preserves precision', () {
        // 2x2 -> 50x50 (25x factor)
        final data = Float32List(3 * 2 * 2);
        for (int i = 0; i < data.length; i++) {
          data[i] = (i + 1) / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 50, width: 50);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 50, width: 50);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [3, 50, 50]);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('alignCorners', () {
      test('produces different results with alignCorners', () {
        final data = Float32List(3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [3, 4, 4]);

        final opDefault = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final opAligned = ResizeNormalizeFusedOp.imagenet(
          height: 2,
          width: 2,
          alignCorners: true,
        );

        final resultDefault = opDefault.apply(input);
        final resultAligned = opAligned.apply(input);

        // Should produce different values
        final defaultData = resultDefault.storage.data as Float32List;
        final alignedData = resultAligned.storage.data as Float32List;
        var hasDifference = false;
        for (int i = 0; i < defaultData.length; i++) {
          if ((defaultData[i] - alignedData[i]).abs() > 1e-6) {
            hasDifference = true;
            break;
          }
        }
        expect(hasDifference, isTrue);
      });

      test('4D tensor with alignCorners=true matches separate ops', () {
        final data = Float32List(2 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat32List(data, [2, 3, 4, 4]);

        final fused = ResizeNormalizeFusedOp.imagenet(
          height: 2,
          width: 2,
          alignCorners: true,
        );
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2, alignCorners: true);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, separateResult.shape);
        final fusedData = fusedResult.storage.data as Float32List;
        final separateData = separateResult.storage.data as Float32List;
        for (int i = 0; i < fusedData.length; i++) {
          expect(
            fusedData[i],
            closeTo(separateData[i], 1e-5),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });

    group('float64 4D', () {
      test('4D float64 tensor uses generic fallback path', () {
        final data = Float64List(2 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / data.length;
        }
        final input = TensorBuffer.fromFloat64List(data, [2, 3, 4, 4]);

        final fused = ResizeNormalizeFusedOp.imagenet(height: 2, width: 2);
        final fusedResult = fused.apply(input);

        final resize = ResizeOp(height: 2, width: 2);
        final normalize = NormalizeOp.imagenet();
        final separateResult = normalize.apply(resize.apply(input));

        expect(fusedResult.shape, [2, 3, 2, 2]);
        for (int i = 0; i < fusedResult.shape.reduce((a, b) => a * b); i++) {
          expect(
            fusedResult.storage.getAsDouble(i),
            closeTo(separateResult.storage.getAsDouble(i), 1e-10),
            reason: 'Mismatch at index $i',
          );
        }
      });
    });
  });
}
