import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('SimdOps', () {
    group('multiplyScalar', () {
      test('multiplies all elements by scalar', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        SimdOps.multiplyScalar(data, 2.0);
        expect(data, equals([2, 4, 6, 8, 10, 12, 14, 16]));
      });

      test('handles non-aligned length', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5]);
        SimdOps.multiplyScalar(data, 3.0);
        expect(data, equals([3, 6, 9, 12, 15]));
      });

      test('handles empty array', () {
        final data = Float32List(0);
        SimdOps.multiplyScalar(data, 2.0);
        expect(data, isEmpty);
      });
    });

    group('addScalar', () {
      test('adds scalar to all elements', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        SimdOps.addScalar(data, 10.0);
        expect(data, equals([11, 12, 13, 14, 15, 16, 17, 18]));
      });

      test('handles negative scalar', () {
        final data = Float32List.fromList([10, 20, 30, 40]);
        SimdOps.addScalar(data, -5.0);
        expect(data, equals([5, 15, 25, 35]));
      });
    });

    group('relu', () {
      test('applies ReLU correctly', () {
        final data = Float32List.fromList([-2, -1, 0, 1, 2, 3, -4, 5]);
        SimdOps.relu(data);
        expect(data, equals([0, 0, 0, 1, 2, 3, 0, 5]));
      });

      test('handles all negative values', () {
        final data = Float32List.fromList([-1, -2, -3, -4]);
        SimdOps.relu(data);
        expect(data, equals([0, 0, 0, 0]));
      });

      test('handles all positive values', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5]);
        SimdOps.relu(data);
        expect(data, equals([1, 2, 3, 4, 5]));
      });
    });

    group('leakyRelu', () {
      test('applies Leaky ReLU with alpha=0.1', () {
        final data = Float32List.fromList([-10, -5, 0, 5, 10, 15, -20, 25]);
        SimdOps.leakyRelu(data, 0.1);
        expect(data[0], closeTo(-1.0, 1e-6));
        expect(data[1], closeTo(-0.5, 1e-6));
        expect(data[2], equals(0));
        expect(data[3], equals(5));
        expect(data[6], closeTo(-2.0, 1e-6));
      });
    });

    group('normalize', () {
      test('normalizes with mean and std', () {
        final data = Float32List.fromList([10, 20, 30, 40]);
        SimdOps.normalize(data, 25.0, 10.0);
        expect(data[0], closeTo(-1.5, 1e-6));
        expect(data[1], closeTo(-0.5, 1e-6));
        expect(data[2], closeTo(0.5, 1e-6));
        expect(data[3], closeTo(1.5, 1e-6));
      });
    });

    group('add', () {
      test('adds two arrays element-wise', () {
        final a = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final b = Float32List.fromList([8, 7, 6, 5, 4, 3, 2, 1]);
        final out = Float32List(8);
        SimdOps.add(a, b, out);
        expect(out, equals([9, 9, 9, 9, 9, 9, 9, 9]));
      });
    });

    group('subtract', () {
      test('subtracts two arrays element-wise', () {
        final a = Float32List.fromList([10, 20, 30, 40, 50, 60, 70, 80]);
        final b = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final out = Float32List(8);
        SimdOps.subtract(a, b, out);
        expect(out, equals([9, 18, 27, 36, 45, 54, 63, 72]));
      });

      test('handles non-aligned length', () {
        final a = Float32List.fromList([10, 20, 30, 40, 50]);
        final b = Float32List.fromList([1, 2, 3, 4, 5]);
        final out = Float32List(5);
        SimdOps.subtract(a, b, out);
        expect(out, equals([9, 18, 27, 36, 45]));
      });
    });

    group('divide', () {
      test('divides two arrays element-wise', () {
        final a = Float32List.fromList([10, 20, 30, 40, 50, 60, 70, 80]);
        final b = Float32List.fromList([2, 4, 5, 8, 10, 10, 7, 8]);
        final out = Float32List(8);
        SimdOps.divide(a, b, out);
        expect(out, equals([5, 5, 6, 5, 5, 6, 10, 10]));
      });

      test('handles non-aligned length', () {
        final a = Float32List.fromList([10, 20, 30]);
        final b = Float32List.fromList([2, 4, 5]);
        final out = Float32List(3);
        SimdOps.divide(a, b, out);
        expect(out, equals([5, 5, 6]));
      });
    });

    group('multiply', () {
      test('multiplies two arrays element-wise', () {
        final a = Float32List.fromList([1, 2, 3, 4]);
        final b = Float32List.fromList([4, 3, 2, 1]);
        final out = Float32List(4);
        SimdOps.multiply(a, b, out);
        expect(out, equals([4, 6, 6, 4]));
      });
    });

    group('copy', () {
      test('copies array', () {
        final src = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final dst = Float32List(8);
        SimdOps.copy(src, dst);
        expect(dst, equals(src));
      });
    });

    group('fill', () {
      test('fills array with value', () {
        final data = Float32List(8);
        SimdOps.fill(data, 42.0);
        expect(data, equals([42, 42, 42, 42, 42, 42, 42, 42]));
      });
    });

    group('sum', () {
      test('computes sum of all elements', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        expect(SimdOps.sum(data), equals(36));
      });

      test('handles empty array', () {
        final data = Float32List(0);
        expect(SimdOps.sum(data), equals(0));
      });
    });

    group('clip', () {
      test('clips values to range', () {
        final data = Float32List.fromList([-5, 0, 5, 10, 15, 20, 25, 30]);
        SimdOps.clip(data, 0, 20);
        expect(data, equals([0, 0, 5, 10, 15, 20, 20, 20]));
      });
    });
  });

  group('SIMD Integration', () {
    test('ScaleOp uses SIMD for Float32', () {
      final tensor = TensorBuffer.full([3, 4, 4], fillValue: 10.0);
      final result = ScaleOp(scale: 2.0, offset: 2.0).apply(tensor);
      // (10 - 2) / 2 = 4
      expect(result[[0, 0, 0]], equals(4.0));
    });

    test('ReLUOp uses SIMD for Float32', () {
      final data = Float32List.fromList([-1, 2, -3, 4, -5, 6]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
      final result = ReLUOp().apply(tensor);
      expect(result[[0, 0]], equals(0.0));
      expect(result[[0, 1]], equals(2.0));
      expect(result[[0, 2]], equals(0.0));
      expect(result[[1, 0]], equals(4.0));
    });

    test('LeakyReLUOp uses SIMD for Float32', () {
      final data = Float32List.fromList([-10, 20, -30, 40]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);
      final result = LeakyReLUOp(negativeSlope: 0.1).apply(tensor);
      expect(result[[0, 0]], closeTo(-1.0, 1e-6));
      expect(result[[0, 1]], equals(20.0));
      expect(result[[1, 0]], closeTo(-3.0, 1e-6));
      expect(result[[1, 1]], equals(40.0));
    });
  });
}
