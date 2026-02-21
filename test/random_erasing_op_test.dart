import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/random_erasing_op.dart';

void main() {
  group('RandomErasingOp', () {
    group('constructor validation', () {
      test('accepts default parameters', () {
        expect(() => RandomErasingOp(), returnsNormally);
      });

      test('accepts valid custom parameters', () {
        expect(
          () => RandomErasingOp(
            probability: 0.3,
            scaleRange: (0.05, 0.25),
            ratioRange: (0.5, 2.0),
            value: 0.5,
            seed: 42,
          ),
          returnsNormally,
        );
      });

      test('throws for probability < 0', () {
        expect(
          () => RandomErasingOp(probability: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for probability > 1', () {
        expect(
          () => RandomErasingOp(probability: 1.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts probability = 0.0 and 1.0', () {
        expect(() => RandomErasingOp(probability: 0.0), returnsNormally);
        expect(() => RandomErasingOp(probability: 1.0), returnsNormally);
      });

      test('throws for invalid scaleRange (min <= 0)', () {
        expect(
          () => RandomErasingOp(scaleRange: (0.0, 0.5)),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => RandomErasingOp(scaleRange: (-0.1, 0.5)),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for invalid scaleRange (min > max)', () {
        expect(
          () => RandomErasingOp(scaleRange: (0.5, 0.3)),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for scaleRange max > 1.0', () {
        expect(
          () => RandomErasingOp(scaleRange: (0.02, 1.1)),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for invalid ratioRange (min <= 0)', () {
        expect(
          () => RandomErasingOp(ratioRange: (0.0, 3.0)),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => RandomErasingOp(ratioRange: (-1.0, 3.0)),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for invalid ratioRange (min > max)', () {
        expect(
          () => RandomErasingOp(ratioRange: (3.0, 0.5)),
          throwsA(isA<InvalidParameterException>()),
        );
      });
    });

    group('shape validation', () {
      test('throws for 1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 2D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4]),
          [2, 2],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 5D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(32),
          [1, 1, 2, 4, 4],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('accepts 3D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(3 * 8 * 8),
          [3, 8, 8],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), returnsNormally);
      });

      test('accepts 4D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(2 * 3 * 8 * 8),
          [2, 3, 8, 8],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), returnsNormally);
      });
    });

    group('probability behavior', () {
      test('probability=0.0 never erases', () {
        final data = Float32List(3 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 16, 16]);

        final op = RandomErasingOp(probability: 0.0, value: 0.0, seed: 42);
        final result = op(tensor);

        // All values should remain 1.0
        for (int i = 0; i < result.numel; i++) {
          expect(result.storage.getAsDouble(i), equals(1.0));
        }
      });

      test('probability=1.0 always attempts erase on sufficiently large tensor',
          () {
        // With a large enough tensor and reasonable parameters, erasure should happen
        final data = Float32List(3 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 32, 32]);

        final op = RandomErasingOp(
          probability: 1.0,
          scaleRange: (0.02, 0.33),
          ratioRange: (0.3, 3.3),
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        // At least some pixels should be erased (set to 0.0)
        bool hasErased = false;
        for (int i = 0; i < result.numel; i++) {
          if (result.storage.getAsDouble(i) == 0.0) {
            hasErased = true;
            break;
          }
        }
        expect(hasErased, isTrue,
            reason: 'Expected some pixels to be erased');
      });
    });

    group('3D tensor erasing', () {
      test('erases rectangular region with constant value', () {
        final data = Float32List(1 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [1, 16, 16]);

        final op = RandomErasingOp(
          probability: 1.0,
          scaleRange: (0.1, 0.3),
          ratioRange: (0.5, 2.0),
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.shape, equals([1, 16, 16]));

        // Count erased vs non-erased pixels
        int erasedCount = 0;
        int originalCount = 0;
        for (int i = 0; i < result.numel; i++) {
          final v = result.storage.getAsDouble(i);
          if (v == 0.0) {
            erasedCount++;
          } else if (v == 1.0) {
            originalCount++;
          }
        }
        // Should have some erased and some original pixels
        expect(erasedCount, greaterThan(0));
        expect(originalCount, greaterThan(0));
        expect(erasedCount + originalCount, equals(result.numel));
      });

      test('preserves shape', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(3 * 10 * 10),
          [3, 10, 10],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        final result = op(tensor);
        expect(result.shape, equals([3, 10, 10]));
      });

      test('erases across all channels at same position', () {
        const c = 3;
        const h = 16;
        const w = 16;
        final data = Float32List(c * h * w);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [c, h, w]);

        final op = RandomErasingOp(
          probability: 1.0,
          scaleRange: (0.1, 0.3),
          ratioRange: (0.5, 2.0),
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        // Find erased positions in channel 0
        final erasedPositions = <int>{};
        for (int i = 0; i < h * w; i++) {
          if (result.storage.getAsDouble(i) == 0.0) {
            erasedPositions.add(i);
          }
        }

        // Same positions should be erased in all channels
        for (int ch = 1; ch < c; ch++) {
          for (final pos in erasedPositions) {
            expect(
              result.storage.getAsDouble(ch * h * w + pos),
              equals(0.0),
              reason: 'Position $pos in channel $ch should also be erased',
            );
          }
        }
      });
    });

    group('4D tensor erasing', () {
      test('erases in 4D batch tensor', () {
        const n = 2;
        const c = 3;
        const h = 16;
        const w = 16;
        final data = Float32List(n * c * h * w);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [n, c, h, w]);

        final op = RandomErasingOp(
          probability: 1.0,
          scaleRange: (0.1, 0.3),
          ratioRange: (0.5, 2.0),
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.shape, equals([n, c, h, w]));

        // Both batch items should have some erasure
        const imageSize = c * h * w;
        for (int batch = 0; batch < n; batch++) {
          int erasedCount = 0;
          for (int i = 0; i < imageSize; i++) {
            if (result.storage.getAsDouble(batch * imageSize + i) == 0.0) {
              erasedCount++;
            }
          }
          expect(erasedCount, greaterThan(0),
              reason: 'Batch $batch should have erased pixels');
        }
      });

      test('preserves 4D shape', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(2 * 3 * 8 * 8),
          [2, 3, 8, 8],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        final result = op(tensor);
        expect(result.shape, equals([2, 3, 8, 8]));
      });
    });

    group('fill modes', () {
      test('fills with constant value', () {
        final data = Float32List(1 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [1, 32, 32]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: 0.5,
          seed: 42,
        );
        final result = op(tensor);

        // Erased pixels should be exactly 0.5
        bool foundFillValue = false;
        for (int i = 0; i < result.numel; i++) {
          final v = result.storage.getAsDouble(i);
          if (v == 0.5) {
            foundFillValue = true;
          }
          // Should only contain original value (1.0) or fill value (0.5)
          expect(v == 1.0 || v == 0.5, isTrue,
              reason: 'Expected 1.0 or 0.5, got $v');
        }
        expect(foundFillValue, isTrue);
      });

      test('fills with random values when value is null', () {
        final data = Float32List(1 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = 5.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [1, 32, 32]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: null,
          seed: 42,
        );
        final result = op(tensor);

        // Should have values other than the original 5.0
        final uniqueValues = <double>{};
        for (int i = 0; i < result.numel; i++) {
          uniqueValues.add(result.storage.getAsDouble(i));
        }
        // With random fill, we expect multiple different values
        expect(uniqueValues.length, greaterThan(2),
            reason: 'Random fill should produce diverse values');
      });
    });

    group('in-place operation', () {
      test('modifies tensor in place', () {
        final data = Float32List(1 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [1, 32, 32]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        op.applyInPlace(tensor);

        // Some values should be erased
        bool hasErased = false;
        for (int i = 0; i < tensor.numel; i++) {
          if (tensor.storage.getAsDouble(i) == 0.0) {
            hasErased = true;
            break;
          }
        }
        expect(hasErased, isTrue);
      });

      test('in-place matches non-in-place result', () {
        // Create two identical tensors
        final data1 = Float32List(3 * 16 * 16);
        final data2 = Float32List(3 * 16 * 16);
        for (int i = 0; i < data1.length; i++) {
          data1[i] = i.toDouble();
          data2[i] = i.toDouble();
        }
        final tensor1 = TensorBuffer.fromFloat32List(data1, [3, 16, 16]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [3, 16, 16]);

        final op1 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 99,
        );
        final op2 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 99,
        );

        final result1 = op1(tensor1);
        op2.applyInPlace(tensor2);

        // Results should match
        for (int i = 0; i < result1.numel; i++) {
          expect(
            result1.storage.getAsDouble(i),
            equals(tensor2.storage.getAsDouble(i)),
            reason: 'Mismatch at index $i',
          );
        }
      });

      test('throws for non-contiguous tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(3 * 8 * 8),
          [3, 8, 8],
        );
        // Transpose creates a non-contiguous view
        final transposed = tensor.transpose([2, 1, 0]);

        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(
          () => op.applyInPlace(transposed),
          throwsA(isA<NonContiguousException>()),
        );
      });
    });

    group('deterministic behavior', () {
      test('produces same result with same seed', () {
        final data = Float32List(3 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = i.toDouble();
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 16, 16]);

        final op1 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        final op2 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        for (int i = 0; i < result1.numel; i++) {
          expect(
            result1.storage.getAsDouble(i),
            equals(result2.storage.getAsDouble(i)),
            reason: 'Results should be identical with same seed at index $i',
          );
        }
      });

      test('produces different results with different seeds', () {
        final data = Float32List(3 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 32, 32]);

        final op1 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        final op2 = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 99,
        );

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        // At least some values should differ
        bool hasDifference = false;
        for (int i = 0; i < result1.numel; i++) {
          if (result1.storage.getAsDouble(i) !=
              result2.storage.getAsDouble(i)) {
            hasDifference = true;
            break;
          }
        }
        expect(hasDifference, isTrue,
            reason: 'Different seeds should produce different results');
      });
    });

    group('data types', () {
      test('works with float64', () {
        final data = Float64List(1 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final storage = TensorStorage(data, DType.float64);
        final tensor = TensorBuffer(storage: storage, shape: [1, 16, 16]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.dtype, equals(DType.float64));
        expect(result.shape, equals([1, 16, 16]));

        // Should have erased pixels
        bool hasErased = false;
        for (int i = 0; i < result.numel; i++) {
          if (result.storage.getAsDouble(i) == 0.0) {
            hasErased = true;
            break;
          }
        }
        expect(hasErased, isTrue);
      });

      test('works with int32', () {
        final data = Int32List(1 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = 100;
        }
        final storage = TensorStorage(data, DType.int32);
        final tensor = TensorBuffer(storage: storage, shape: [1, 16, 16]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        final result = op(tensor);

        expect(result.dtype, equals(DType.int32));
        expect(result.shape, equals([1, 16, 16]));
      });
    });

    group('does not modify input', () {
      test('apply does not modify the input tensor', () {
        final data = Float32List(3 * 16 * 16);
        for (int i = 0; i < data.length; i++) {
          data[i] = 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 16, 16]);

        final op = RandomErasingOp(
          probability: 1.0,
          value: 0.0,
          seed: 42,
        );
        op(tensor);

        // Original tensor should be untouched
        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(1.0),
              reason: 'Input tensor should not be modified by apply()');
        }
      });
    });

    group('output shape computation', () {
      test('preserves input shape for 3D', () {
        final op = RandomErasingOp();
        expect(op.computeOutputShape([3, 224, 224]), equals([3, 224, 224]));
      });

      test('preserves input shape for 4D', () {
        final op = RandomErasingOp();
        expect(
            op.computeOutputShape([1, 3, 224, 224]), equals([1, 3, 224, 224]));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = RandomErasingOp();
        expect(op.capabilities.supportsInPlace, isTrue);
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
        expect(op.capabilities.pytorchEquivalent,
            equals('torchvision.transforms.RandomErasing'));
      });
    });

    group('string representation', () {
      test('toString returns descriptive name', () {
        final op = RandomErasingOp(
          probability: 0.5,
          scaleRange: (0.02, 0.33),
          ratioRange: (0.3, 3.3),
          value: 0.0,
        );
        final str = op.toString();
        expect(str, contains('RandomErasing'));
        expect(str, contains('0.5'));
      });
    });

    group('edge cases', () {
      test('handles single-pixel tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0]),
          [1, 1, 1],
        );
        final op = RandomErasingOp(
          probability: 1.0,
          scaleRange: (0.5, 1.0),
          ratioRange: (0.5, 2.0),
          value: 0.0,
          seed: 42,
        );
        // Should not crash
        expect(() => op(tensor), returnsNormally);
      });

      test('handles tall narrow tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(1 * 32 * 4),
          [1, 32, 4],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), returnsNormally);
      });

      test('handles wide short tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(1 * 4 * 32),
          [1, 4, 32],
        );
        final op = RandomErasingOp(probability: 1.0, seed: 42);
        expect(() => op(tensor), returnsNormally);
      });

      test('handles single batch item in 4D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List(1 * 3 * 16 * 16),
          [1, 3, 16, 16],
        );
        final op = RandomErasingOp(probability: 1.0, value: 0.0, seed: 42);
        final result = op(tensor);
        expect(result.shape, equals([1, 3, 16, 16]));
      });
    });
  });
}
