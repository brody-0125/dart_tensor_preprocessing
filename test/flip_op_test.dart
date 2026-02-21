import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/augmentation_op.dart';

void main() {
  // ========================================================================
  // HorizontalFlipOp
  // ========================================================================
  group('HorizontalFlipOp', () {
    group('basic functionality', () {
      test('flips 3D tensor horizontally', () {
        // [1, 2, 3]  ->  [3, 2, 1]
        // [4, 5, 6]      [6, 5, 4]
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 3]));
        expect(result[[0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(1.0));
        expect(result[[0, 1, 0]], equals(6.0));
        expect(result[[0, 1, 1]], equals(5.0));
        expect(result[[0, 1, 2]], equals(4.0));
      });

      test('flips 4D tensor horizontally', () {
        // Batch 0, Channel 0:
        // [1, 2]  ->  [2, 1]
        // [3, 4]      [4, 3]
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2, 2]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2, 2]));
        // Channel 0
        expect(result[[0, 0, 0, 0]], equals(2.0));
        expect(result[[0, 0, 0, 1]], equals(1.0));
        expect(result[[0, 0, 1, 0]], equals(4.0));
        expect(result[[0, 0, 1, 1]], equals(3.0));
        // Channel 1
        expect(result[[0, 1, 0, 0]], equals(6.0));
        expect(result[[0, 1, 0, 1]], equals(5.0));
        expect(result[[0, 1, 1, 0]], equals(8.0));
        expect(result[[0, 1, 1, 1]], equals(7.0));
      });

      test('double flip restores original', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = HorizontalFlipOp();
        final result = op(op(tensor));

        for (int i = 0; i < 6; i++) {
          expect(result.storage.getAsDouble(i), equals(data[i].toDouble()));
        }
      });
    });

    group('shape validation', () {
      test('throws for 1D tensor', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final op = HorizontalFlipOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final op = HorizontalFlipOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for 5D tensor', () {
        final data = Float32List(32);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 2, 4, 4]);

        final op = HorizontalFlipOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('edge cases', () {
      test('handles single-element tensor', () {
        final data = Float32List.fromList([42]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 1, 1]));
        expect(result[[0, 0, 0]], equals(42.0));
      });

      test('handles width=1 tensor', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 3, 1]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 1, 0]], equals(2.0));
        expect(result[[0, 2, 0]], equals(3.0));
      });
    });

    group('different data types', () {
      test('works with float64', () {
        final data = Float64List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat64List(data, [1, 2, 3]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.dtype, equals(DType.float64));
        expect(result[[0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 2]], equals(1.0));
      });

      test('works with int32', () {
        final data = Int32List.fromList([1, 2, 3, 4, 5, 6]);
        final storage = TensorStorage(data, DType.int32);
        final tensor = TensorBuffer(storage: storage, shape: [1, 2, 3]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.dtype, equals(DType.int32));
        expect(result[[0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 2]], equals(1.0));
      });

      test('works with uint8', () {
        final data = Uint8List.fromList([10, 20, 30, 40, 50, 60]);
        final tensor = TensorBuffer.fromUint8List(data, [1, 2, 3]);

        final op = HorizontalFlipOp();
        final result = op(tensor);

        expect(result.dtype, equals(DType.uint8));
        expect(result[[0, 0, 0]], equals(30.0));
        expect(result[[0, 0, 2]], equals(10.0));
      });
    });

    group('does not mutate input', () {
      test('original tensor is unchanged', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = HorizontalFlipOp();
        op(tensor);

        expect(tensor[[0, 0, 0]], equals(1.0));
        expect(tensor[[0, 0, 1]], equals(2.0));
        expect(tensor[[0, 0, 2]], equals(3.0));
      });
    });

    group('output shape computation', () {
      test('preserves input shape', () {
        final op = HorizontalFlipOp();
        expect(op.computeOutputShape([3, 4, 5]), equals([3, 4, 5]));
        expect(op.computeOutputShape([2, 3, 4, 5]), equals([2, 3, 4, 5]));
      });
    });

    group('string representation', () {
      test('toString returns operation name', () {
        final op = HorizontalFlipOp();
        expect(op.toString(), equals('TransformOp(HorizontalFlip)'));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = HorizontalFlipOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
        expect(op.capabilities.pytorchEquivalent,
            equals('torchvision.transforms.functional.hflip'));
      });
    });
  });

  // ========================================================================
  // VerticalFlipOp
  // ========================================================================
  group('VerticalFlipOp', () {
    group('basic functionality', () {
      test('flips 3D tensor vertically', () {
        // [1, 2, 3]  ->  [4, 5, 6]
        // [4, 5, 6]      [1, 2, 3]
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 3]));
        expect(result[[0, 0, 0]], equals(4.0));
        expect(result[[0, 0, 1]], equals(5.0));
        expect(result[[0, 0, 2]], equals(6.0));
        expect(result[[0, 1, 0]], equals(1.0));
        expect(result[[0, 1, 1]], equals(2.0));
        expect(result[[0, 1, 2]], equals(3.0));
      });

      test('flips 4D tensor vertically', () {
        // Batch 0, Channel 0:
        // [1, 2]  ->  [3, 4]
        // [3, 4]      [1, 2]
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2, 2]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2, 2]));
        // Channel 0
        expect(result[[0, 0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 0, 1]], equals(4.0));
        expect(result[[0, 0, 1, 0]], equals(1.0));
        expect(result[[0, 0, 1, 1]], equals(2.0));
        // Channel 1
        expect(result[[0, 1, 0, 0]], equals(7.0));
        expect(result[[0, 1, 0, 1]], equals(8.0));
        expect(result[[0, 1, 1, 0]], equals(5.0));
        expect(result[[0, 1, 1, 1]], equals(6.0));
      });

      test('double flip restores original', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = VerticalFlipOp();
        final result = op(op(tensor));

        for (int i = 0; i < 6; i++) {
          expect(result.storage.getAsDouble(i), equals(data[i].toDouble()));
        }
      });
    });

    group('shape validation', () {
      test('throws for 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final op = VerticalFlipOp();
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('edge cases', () {
      test('handles single-element tensor', () {
        final data = Float32List.fromList([42]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result.shape, equals([1, 1, 1]));
        expect(result[[0, 0, 0]], equals(42.0));
      });

      test('handles height=1 tensor', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 3]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(3.0));
      });

      test('flips 3-row tensor correctly', () {
        // [1, 2]    [5, 6]
        // [3, 4] -> [3, 4]
        // [5, 6]    [1, 2]
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 3, 2]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result[[0, 0, 0]], equals(5.0));
        expect(result[[0, 0, 1]], equals(6.0));
        expect(result[[0, 1, 0]], equals(3.0));
        expect(result[[0, 1, 1]], equals(4.0));
        expect(result[[0, 2, 0]], equals(1.0));
        expect(result[[0, 2, 1]], equals(2.0));
      });
    });

    group('different data types', () {
      test('works with float64', () {
        final data = Float64List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat64List(data, [1, 2, 3]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result.dtype, equals(DType.float64));
        expect(result[[0, 0, 0]], equals(4.0));
        expect(result[[0, 1, 0]], equals(1.0));
      });

      test('works with uint8', () {
        final data = Uint8List.fromList([10, 20, 30, 40, 50, 60]);
        final tensor = TensorBuffer.fromUint8List(data, [1, 2, 3]);

        final op = VerticalFlipOp();
        final result = op(tensor);

        expect(result.dtype, equals(DType.uint8));
        expect(result[[0, 0, 0]], equals(40.0));
        expect(result[[0, 1, 0]], equals(10.0));
      });
    });

    group('does not mutate input', () {
      test('original tensor is unchanged', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = VerticalFlipOp();
        op(tensor);

        expect(tensor[[0, 0, 0]], equals(1.0));
        expect(tensor[[0, 1, 0]], equals(4.0));
      });
    });

    group('output shape computation', () {
      test('preserves input shape', () {
        final op = VerticalFlipOp();
        expect(op.computeOutputShape([3, 4, 5]), equals([3, 4, 5]));
        expect(op.computeOutputShape([2, 3, 4, 5]), equals([2, 3, 4, 5]));
      });
    });

    group('string representation', () {
      test('toString returns operation name', () {
        final op = VerticalFlipOp();
        expect(op.toString(), equals('TransformOp(VerticalFlip)'));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = VerticalFlipOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
        expect(op.capabilities.pytorchEquivalent,
            equals('torchvision.transforms.functional.vflip'));
      });
    });
  });

  // ========================================================================
  // RandomHorizontalFlipOp
  // ========================================================================
  group('RandomHorizontalFlipOp', () {
    group('constructor validation', () {
      test('throws for probability < 0', () {
        expect(
          () => RandomHorizontalFlipOp(probability: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for probability > 1', () {
        expect(
          () => RandomHorizontalFlipOp(probability: 1.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts probability = 0', () {
        expect(() => RandomHorizontalFlipOp(probability: 0.0), returnsNormally);
      });

      test('accepts probability = 1', () {
        expect(() => RandomHorizontalFlipOp(probability: 1.0), returnsNormally);
      });

      test('accepts default probability', () {
        final op = RandomHorizontalFlipOp();
        expect(op.probability, equals(0.5));
      });
    });

    group('deterministic behavior with probability=1.0', () {
      test('always flips with probability=1.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = RandomHorizontalFlipOp(probability: 1.0, seed: 42);
        final result = op(tensor);

        // Must be flipped
        expect(result[[0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(1.0));
        expect(result[[0, 1, 0]], equals(6.0));
        expect(result[[0, 1, 1]], equals(5.0));
        expect(result[[0, 1, 2]], equals(4.0));
      });
    });

    group('deterministic behavior with probability=0.0', () {
      test('never flips with probability=0.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = RandomHorizontalFlipOp(probability: 0.0, seed: 42);
        final result = op(tensor);

        // Must not be flipped
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(3.0));
      });
    });

    group('reproducibility with seed', () {
      test('same seed produces same results across instances', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op1 = RandomHorizontalFlipOp(probability: 0.5, seed: 123);
        final op2 = RandomHorizontalFlipOp(probability: 0.5, seed: 123);

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        for (int i = 0; i < 6; i++) {
          expect(result1.storage.getAsDouble(i),
              equals(result2.storage.getAsDouble(i)));
        }
      });
    });

    group('shape validation', () {
      test('throws for 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final op = RandomHorizontalFlipOp(probability: 1.0);
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('4D tensor support', () {
      test('flips 4D tensor with probability=1.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2, 2]);

        final op = RandomHorizontalFlipOp(probability: 1.0, seed: 42);
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2, 2]));
        expect(result[[0, 0, 0, 0]], equals(2.0));
        expect(result[[0, 0, 0, 1]], equals(1.0));
      });
    });

    group('output shape computation', () {
      test('preserves input shape', () {
        final op = RandomHorizontalFlipOp();
        expect(op.computeOutputShape([3, 4, 5]), equals([3, 4, 5]));
        expect(op.computeOutputShape([2, 3, 4, 5]), equals([2, 3, 4, 5]));
      });
    });

    group('string representation', () {
      test('toString without seed', () {
        final op = RandomHorizontalFlipOp(probability: 0.5);
        expect(op.toString(),
            equals('TransformOp(RandomHorizontalFlip(p=0.5))'));
      });

      test('toString with seed', () {
        final op = RandomHorizontalFlipOp(probability: 0.3, seed: 42);
        expect(op.toString(),
            equals('TransformOp(RandomHorizontalFlip(p=0.3, seed=42))'));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = RandomHorizontalFlipOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
        expect(op.capabilities.pytorchEquivalent,
            equals('torchvision.transforms.RandomHorizontalFlip'));
      });
    });
  });

  // ========================================================================
  // RandomVerticalFlipOp
  // ========================================================================
  group('RandomVerticalFlipOp', () {
    group('constructor validation', () {
      test('throws for probability < 0', () {
        expect(
          () => RandomVerticalFlipOp(probability: -0.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for probability > 1', () {
        expect(
          () => RandomVerticalFlipOp(probability: 1.1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts probability = 0', () {
        expect(() => RandomVerticalFlipOp(probability: 0.0), returnsNormally);
      });

      test('accepts probability = 1', () {
        expect(() => RandomVerticalFlipOp(probability: 1.0), returnsNormally);
      });

      test('accepts default probability', () {
        final op = RandomVerticalFlipOp();
        expect(op.probability, equals(0.5));
      });
    });

    group('deterministic behavior with probability=1.0', () {
      test('always flips with probability=1.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = RandomVerticalFlipOp(probability: 1.0, seed: 42);
        final result = op(tensor);

        // Must be flipped vertically
        expect(result[[0, 0, 0]], equals(4.0));
        expect(result[[0, 0, 1]], equals(5.0));
        expect(result[[0, 0, 2]], equals(6.0));
        expect(result[[0, 1, 0]], equals(1.0));
        expect(result[[0, 1, 1]], equals(2.0));
        expect(result[[0, 1, 2]], equals(3.0));
      });
    });

    group('deterministic behavior with probability=0.0', () {
      test('never flips with probability=0.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op = RandomVerticalFlipOp(probability: 0.0, seed: 42);
        final result = op(tensor);

        // Must not be flipped
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(3.0));
        expect(result[[0, 1, 0]], equals(4.0));
      });
    });

    group('reproducibility with seed', () {
      test('same seed produces same results across instances', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

        final op1 = RandomVerticalFlipOp(probability: 0.5, seed: 456);
        final op2 = RandomVerticalFlipOp(probability: 0.5, seed: 456);

        final result1 = op1(tensor);
        final result2 = op2(tensor);

        for (int i = 0; i < 6; i++) {
          expect(result1.storage.getAsDouble(i),
              equals(result2.storage.getAsDouble(i)));
        }
      });
    });

    group('shape validation', () {
      test('throws for 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final op = RandomVerticalFlipOp(probability: 1.0);
        expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('4D tensor support', () {
      test('flips 4D tensor with probability=1.0', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2, 2]);

        final op = RandomVerticalFlipOp(probability: 1.0, seed: 42);
        final result = op(tensor);

        expect(result.shape, equals([1, 2, 2, 2]));
        // Channel 0: rows swapped
        expect(result[[0, 0, 0, 0]], equals(3.0));
        expect(result[[0, 0, 0, 1]], equals(4.0));
        expect(result[[0, 0, 1, 0]], equals(1.0));
        expect(result[[0, 0, 1, 1]], equals(2.0));
      });
    });

    group('output shape computation', () {
      test('preserves input shape', () {
        final op = RandomVerticalFlipOp();
        expect(op.computeOutputShape([3, 4, 5]), equals([3, 4, 5]));
        expect(op.computeOutputShape([2, 3, 4, 5]), equals([2, 3, 4, 5]));
      });
    });

    group('string representation', () {
      test('toString without seed', () {
        final op = RandomVerticalFlipOp(probability: 0.5);
        expect(
            op.toString(), equals('TransformOp(RandomVerticalFlip(p=0.5))'));
      });

      test('toString with seed', () {
        final op = RandomVerticalFlipOp(probability: 0.7, seed: 99);
        expect(op.toString(),
            equals('TransformOp(RandomVerticalFlip(p=0.7, seed=99))'));
      });
    });

    group('capabilities', () {
      test('reports correct capabilities', () {
        final op = RandomVerticalFlipOp();
        expect(op.capabilities.requiresContiguous, isTrue);
        expect(op.capabilities.preservesShape, isTrue);
        expect(op.capabilities.pytorchEquivalent,
            equals('torchvision.transforms.RandomVerticalFlip'));
      });
    });
  });

  // ========================================================================
  // Combined flip operations
  // ========================================================================
  group('Combined flip operations', () {
    test('horizontal then vertical flip', () {
      // Original:          HFlip:          VFlip:
      // [1, 2, 3]    ->   [3, 2, 1]  ->   [6, 5, 4]
      // [4, 5, 6]         [6, 5, 4]       [3, 2, 1]
      final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]);

      final hFlip = HorizontalFlipOp();
      final vFlip = VerticalFlipOp();

      final result = vFlip(hFlip(tensor));

      expect(result[[0, 0, 0]], equals(6.0));
      expect(result[[0, 0, 1]], equals(5.0));
      expect(result[[0, 0, 2]], equals(4.0));
      expect(result[[0, 1, 0]], equals(3.0));
      expect(result[[0, 1, 1]], equals(2.0));
      expect(result[[0, 1, 2]], equals(1.0));
    });

    test('both flips are commutative', () {
      final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 3, 4]);

      final hFlip = HorizontalFlipOp();
      final vFlip = VerticalFlipOp();

      final hvResult = vFlip(hFlip(tensor));
      final vhResult = hFlip(vFlip(tensor));

      for (int i = 0; i < 12; i++) {
        expect(hvResult.storage.getAsDouble(i),
            equals(vhResult.storage.getAsDouble(i)));
      }
    });

    test('multi-channel 4D batch flip', () {
      // 2 batches, 3 channels, 2x2 spatial
      final data = Float32List(24);
      for (int i = 0; i < 24; i++) {
        data[i] = (i + 1).toDouble();
      }
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 2, 2]);

      final hFlip = HorizontalFlipOp();
      final vFlip = VerticalFlipOp();

      final hResult = hFlip(tensor);
      final vResult = vFlip(tensor);

      expect(hResult.shape, equals([2, 3, 2, 2]));
      expect(vResult.shape, equals([2, 3, 2, 2]));

      // Batch 0, Channel 0: [1,2,3,4] -> hflip: [2,1,4,3]
      expect(hResult[[0, 0, 0, 0]], equals(2.0));
      expect(hResult[[0, 0, 0, 1]], equals(1.0));
      expect(hResult[[0, 0, 1, 0]], equals(4.0));
      expect(hResult[[0, 0, 1, 1]], equals(3.0));

      // Batch 0, Channel 0: [1,2,3,4] -> vflip: [3,4,1,2]
      expect(vResult[[0, 0, 0, 0]], equals(3.0));
      expect(vResult[[0, 0, 0, 1]], equals(4.0));
      expect(vResult[[0, 0, 1, 0]], equals(1.0));
      expect(vResult[[0, 0, 1, 1]], equals(2.0));
    });
  });
}
