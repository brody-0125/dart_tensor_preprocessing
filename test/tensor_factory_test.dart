import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';

void main() {
  group('TensorBuffer.full', () {
    test('creates tensor filled with specified value', () {
      final tensor = TensorBuffer.full([2, 3], fillValue: 5.0);

      expect(tensor.shape, equals([2, 3]));
      expect(tensor.numel, equals(6));
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
          expect(tensor[[i, j]], closeTo(5.0, 1e-6));
        }
      }
    });

    test('handles negative fill value', () {
      final tensor = TensorBuffer.full([3], fillValue: -2.5);

      expect(tensor[[0]], closeTo(-2.5, 1e-6));
      expect(tensor[[1]], closeTo(-2.5, 1e-6));
      expect(tensor[[2]], closeTo(-2.5, 1e-6));
    });

    test('handles zero fill value', () {
      final tensor = TensorBuffer.full([2, 2], fillValue: 0.0);

      expect(tensor[[0, 0]], closeTo(0.0, 1e-6));
      expect(tensor[[1, 1]], closeTo(0.0, 1e-6));
    });

    test('supports different dtypes', () {
      final tensorFloat = TensorBuffer.full([2], fillValue: 3.0, dtype: DType.float32);
      final tensorInt = TensorBuffer.full([2], fillValue: 3.0, dtype: DType.int32);

      expect(tensorFloat.dtype, equals(DType.float32));
      expect(tensorInt.dtype, equals(DType.int32));
    });
  });

  group('TensorBuffer.random', () {
    test('creates tensor with values in [0, 1)', () {
      final tensor = TensorBuffer.random([10, 10]);

      expect(tensor.shape, equals([10, 10]));
      for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
          expect(tensor[[i, j]], greaterThanOrEqualTo(0.0));
          expect(tensor[[i, j]], lessThan(1.0));
        }
      }
    });

    test('produces different values each call (statistically)', () {
      final tensor1 = TensorBuffer.random([100]);
      final tensor2 = TensorBuffer.random([100]);

      // With high probability, they should be different
      int differences = 0;
      for (int i = 0; i < 100; i++) {
        if ((tensor1[[i]] - tensor2[[i]]).abs() > 1e-10) {
          differences++;
        }
      }
      expect(differences, greaterThan(90)); // Allow for rare collisions
    });

    test('supports seed for reproducibility', () {
      final tensor1 = TensorBuffer.random([5], seed: 42);
      final tensor2 = TensorBuffer.random([5], seed: 42);

      for (int i = 0; i < 5; i++) {
        expect(tensor1[[i]], closeTo(tensor2[[i]], 1e-10));
      }
    });
  });

  group('TensorBuffer.randn', () {
    test('creates tensor with approximately mean 0', () {
      final tensor = TensorBuffer.randn([1000]);

      double sum = 0;
      for (int i = 0; i < 1000; i++) {
        sum += tensor[[i]];
      }
      final mean = sum / 1000;

      // Mean should be close to 0 (within 3 standard errors)
      expect(mean.abs(), lessThan(0.15)); // ~3 * 1/sqrt(1000)
    });

    test('creates tensor with approximately std 1', () {
      final tensor = TensorBuffer.randn([1000]);

      double sum = 0;
      double sumSq = 0;
      for (int i = 0; i < 1000; i++) {
        final v = tensor[[i]];
        sum += v;
        sumSq += v * v;
      }
      final mean = sum / 1000;
      final variance = (sumSq / 1000) - mean * mean;
      final std = variance > 0 ? variance : 0;

      // Std should be close to 1
      expect(std, closeTo(1.0, 0.2));
    });

    test('supports seed for reproducibility', () {
      final tensor1 = TensorBuffer.randn([5], seed: 123);
      final tensor2 = TensorBuffer.randn([5], seed: 123);

      for (int i = 0; i < 5; i++) {
        expect(tensor1[[i]], closeTo(tensor2[[i]], 1e-10));
      }
    });
  });

  group('TensorBuffer.eye', () {
    test('creates identity matrix', () {
      final tensor = TensorBuffer.eye(3);

      expect(tensor.shape, equals([3, 3]));
      expect(tensor[[0, 0]], closeTo(1.0, 1e-6));
      expect(tensor[[1, 1]], closeTo(1.0, 1e-6));
      expect(tensor[[2, 2]], closeTo(1.0, 1e-6));
      expect(tensor[[0, 1]], closeTo(0.0, 1e-6));
      expect(tensor[[0, 2]], closeTo(0.0, 1e-6));
      expect(tensor[[1, 0]], closeTo(0.0, 1e-6));
    });

    test('handles size 1', () {
      final tensor = TensorBuffer.eye(1);

      expect(tensor.shape, equals([1, 1]));
      expect(tensor[[0, 0]], closeTo(1.0, 1e-6));
    });

    test('supports rectangular matrices', () {
      final tensor = TensorBuffer.eye(2, m: 4);

      expect(tensor.shape, equals([2, 4]));
      expect(tensor[[0, 0]], closeTo(1.0, 1e-6));
      expect(tensor[[1, 1]], closeTo(1.0, 1e-6));
      expect(tensor[[0, 2]], closeTo(0.0, 1e-6));
      expect(tensor[[1, 3]], closeTo(0.0, 1e-6));
    });

    test('throws for non-positive size', () {
      expect(
        () => TensorBuffer.eye(0),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => TensorBuffer.eye(-1),
        throwsA(isA<InvalidParameterException>()),
      );
    });
  });

  group('TensorBuffer.linspace', () {
    test('creates evenly spaced values', () {
      final tensor = TensorBuffer.linspace(0.0, 1.0, steps: 5);

      expect(tensor.shape, equals([5]));
      expect(tensor[[0]], closeTo(0.0, 1e-6));
      expect(tensor[[1]], closeTo(0.25, 1e-6));
      expect(tensor[[2]], closeTo(0.5, 1e-6));
      expect(tensor[[3]], closeTo(0.75, 1e-6));
      expect(tensor[[4]], closeTo(1.0, 1e-6));
    });

    test('handles descending range', () {
      final tensor = TensorBuffer.linspace(10.0, 0.0, steps: 3);

      expect(tensor[[0]], closeTo(10.0, 1e-6));
      expect(tensor[[1]], closeTo(5.0, 1e-6));
      expect(tensor[[2]], closeTo(0.0, 1e-6));
    });

    test('handles single step', () {
      final tensor = TensorBuffer.linspace(5.0, 10.0, steps: 1);

      expect(tensor.shape, equals([1]));
      expect(tensor[[0]], closeTo(5.0, 1e-6));
    });

    test('handles negative range', () {
      final tensor = TensorBuffer.linspace(-1.0, 1.0, steps: 3);

      expect(tensor[[0]], closeTo(-1.0, 1e-6));
      expect(tensor[[1]], closeTo(0.0, 1e-6));
      expect(tensor[[2]], closeTo(1.0, 1e-6));
    });

    test('throws for steps < 1', () {
      expect(
        () => TensorBuffer.linspace(0.0, 1.0, steps: 0),
        throwsA(isA<InvalidParameterException>()),
      );
    });
  });

  group('TensorBuffer.arange', () {
    test('creates sequence with default step', () {
      final tensor = TensorBuffer.arange(start: 0.0, end: 5.0);

      expect(tensor.shape, equals([5]));
      expect(tensor[[0]], closeTo(0.0, 1e-6));
      expect(tensor[[1]], closeTo(1.0, 1e-6));
      expect(tensor[[2]], closeTo(2.0, 1e-6));
      expect(tensor[[3]], closeTo(3.0, 1e-6));
      expect(tensor[[4]], closeTo(4.0, 1e-6));
    });

    test('creates sequence with custom step', () {
      final tensor = TensorBuffer.arange(start: 0.0, end: 10.0, step: 2.0);

      expect(tensor.shape, equals([5]));
      expect(tensor[[0]], closeTo(0.0, 1e-6));
      expect(tensor[[1]], closeTo(2.0, 1e-6));
      expect(tensor[[2]], closeTo(4.0, 1e-6));
      expect(tensor[[3]], closeTo(6.0, 1e-6));
      expect(tensor[[4]], closeTo(8.0, 1e-6));
    });

    test('handles fractional step', () {
      final tensor = TensorBuffer.arange(start: 0.0, end: 1.0, step: 0.25);

      expect(tensor.shape, equals([4]));
      expect(tensor[[0]], closeTo(0.0, 1e-6));
      expect(tensor[[1]], closeTo(0.25, 1e-6));
      expect(tensor[[2]], closeTo(0.5, 1e-6));
      expect(tensor[[3]], closeTo(0.75, 1e-6));
    });

    test('handles descending sequence', () {
      final tensor = TensorBuffer.arange(start: 5.0, end: 0.0, step: -1.0);

      expect(tensor.shape, equals([5]));
      expect(tensor[[0]], closeTo(5.0, 1e-6));
      expect(tensor[[1]], closeTo(4.0, 1e-6));
      expect(tensor[[2]], closeTo(3.0, 1e-6));
      expect(tensor[[3]], closeTo(2.0, 1e-6));
      expect(tensor[[4]], closeTo(1.0, 1e-6));
    });

    test('throws for zero step', () {
      expect(
        () => TensorBuffer.arange(start: 0.0, end: 5.0, step: 0.0),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for invalid step direction', () {
      expect(
        () => TensorBuffer.arange(start: 0.0, end: 5.0, step: -1.0),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => TensorBuffer.arange(start: 5.0, end: 0.0, step: 1.0),
        throwsA(isA<InvalidParameterException>()),
      );
    });
  });
}
