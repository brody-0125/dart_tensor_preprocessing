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

    group('abs', () {
      test('handles positive values', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        SimdOps.abs(data);
        expect(data, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles negative values', () {
        final data = Float32List.fromList([-1, -2, -3, -4, -5, -6, -7, -8]);
        SimdOps.abs(data);
        expect(data, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles mixed values', () {
        final data = Float32List.fromList([-5, 3, -2, 0, 7, -1, 4, -8]);
        SimdOps.abs(data);
        expect(data, equals([5, 3, 2, 0, 7, 1, 4, 8]));
      });

      test('handles length not multiple of 4', () {
        final data = Float32List.fromList([-1, 2, -3, 4, -5]);
        SimdOps.abs(data);
        expect(data, equals([1, 2, 3, 4, 5]));
      });

      test('handles empty array', () {
        final data = Float32List(0);
        SimdOps.abs(data);
        expect(data, isEmpty);
      });
    });

    group('sqrt', () {
      test('computes correct sqrt', () {
        final data = Float32List.fromList([1, 4, 9, 16, 25, 36, 49, 64]);
        SimdOps.sqrt(data);
        expect(data, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles zero', () {
        final data = Float32List.fromList([0, 1, 0, 4]);
        SimdOps.sqrt(data);
        expect(data, equals([0, 1, 0, 2]));
      });

      test('handles length not multiple of 4', () {
        final data = Float32List.fromList([1, 4, 9, 16, 25]);
        SimdOps.sqrt(data);
        expect(data, equals([1, 2, 3, 4, 5]));
      });

      test('handles empty array', () {
        final data = Float32List(0);
        SimdOps.sqrt(data);
        expect(data, isEmpty);
      });
    });
  });

  group('SIMD alignment edge cases', () {
    test('abs handles unaligned offset correctly', () {
      // Create buffer with extra element, then get sublist view starting at offset 1
      // This creates data that is NOT 16-byte aligned (offset = 4 bytes instead of 0)
      final buffer = Float32List.fromList([0, -1, -2, -3, -4, 5, 6, 7, 8]);
      final unalignedData = Float32List.sublistView(buffer, 1);

      SimdOps.abs(unalignedData);
      expect(unalignedData, equals([1, 2, 3, 4, 5, 6, 7, 8]));
    });

    test('sqrt handles unaligned offset correctly', () {
      final buffer = Float32List.fromList([0, 1, 4, 9, 16, 25, 36, 49, 64]);
      final unalignedData = Float32List.sublistView(buffer, 1);

      SimdOps.sqrt(unalignedData);
      expect(unalignedData, equals([1, 2, 3, 4, 5, 6, 7, 8]));
    });

    test('clip handles unaligned offset correctly', () {
      final buffer = Float32List.fromList([0, -5, 0, 5, 10, 15, 20, 25, 30]);
      final unalignedData = Float32List.sublistView(buffer, 1);

      SimdOps.clip(unalignedData, 0, 20);
      expect(unalignedData, equals([0, 0, 5, 10, 15, 20, 20, 20]));
    });

    test('normalize handles unaligned offset correctly', () {
      final buffer = Float32List.fromList([0, 10, 20, 30, 40]);
      final unalignedData = Float32List.sublistView(buffer, 1);

      SimdOps.normalize(unalignedData, 25.0, 10.0);
      expect(unalignedData[0], closeTo(-1.5, 1e-6));
      expect(unalignedData[1], closeTo(-0.5, 1e-6));
      expect(unalignedData[2], closeTo(0.5, 1e-6));
      expect(unalignedData[3], closeTo(1.5, 1e-6));
    });

    test('multiplyScalar handles unaligned offset correctly', () {
      final buffer = Float32List.fromList([0, 1, 2, 3, 4, 5, 6, 7, 8]);
      final unalignedData = Float32List.sublistView(buffer, 1);

      SimdOps.multiplyScalar(unalignedData, 2.0);
      expect(unalignedData, equals([2, 4, 6, 8, 10, 12, 14, 16]));
    });

    test('sublist view with various offsets (1-4 elements)', () {
      // Test offset = 1 (4 bytes offset, not 16-byte aligned)
      final buffer1 = Float32List(9);
      for (int i = 0; i < 9; i++) {
        buffer1[i] = -(i.toDouble());
      }
      final view1 = Float32List.sublistView(buffer1, 1);
      SimdOps.abs(view1);
      expect(view1, equals([1, 2, 3, 4, 5, 6, 7, 8]));

      // Test offset = 2 (8 bytes offset, not 16-byte aligned)
      final buffer2 = Float32List(10);
      for (int i = 0; i < 10; i++) {
        buffer2[i] = -(i.toDouble());
      }
      final view2 = Float32List.sublistView(buffer2, 2);
      SimdOps.abs(view2);
      expect(view2, equals([2, 3, 4, 5, 6, 7, 8, 9]));

      // Test offset = 3 (12 bytes offset, not 16-byte aligned)
      final buffer3 = Float32List(11);
      for (int i = 0; i < 11; i++) {
        buffer3[i] = -(i.toDouble());
      }
      final view3 = Float32List.sublistView(buffer3, 3);
      SimdOps.abs(view3);
      expect(view3, equals([3, 4, 5, 6, 7, 8, 9, 10]));

      // Test offset = 4 (16 bytes offset, IS 16-byte aligned)
      final buffer4 = Float32List(12);
      for (int i = 0; i < 12; i++) {
        buffer4[i] = -(i.toDouble());
      }
      final view4 = Float32List.sublistView(buffer4, 4);
      SimdOps.abs(view4);
      expect(view4, equals([4, 5, 6, 7, 8, 9, 10, 11]));
    });

    test('handles small unaligned arrays (length < 4)', () {
      // Length 1
      final buffer1 = Float32List.fromList([0, -5]);
      final view1 = Float32List.sublistView(buffer1, 1, 2);
      SimdOps.abs(view1);
      expect(view1, equals([-5].map((e) => e.abs())));

      // Length 2
      final buffer2 = Float32List.fromList([0, -3, -7]);
      final view2 = Float32List.sublistView(buffer2, 1, 3);
      SimdOps.abs(view2);
      expect(view2, equals([3, 7]));

      // Length 3
      final buffer3 = Float32List.fromList([0, -1, -2, -3]);
      final view3 = Float32List.sublistView(buffer3, 1);
      SimdOps.abs(view3);
      expect(view3, equals([1, 2, 3]));
    });
  });

  group('SimdOps Float64', () {
    group('clipF64', () {
      test('clips values to range (aligned)', () {
        final data = Float64List.fromList([-5, 0, 5, 10, 15, 20, 25, 30]);
        SimdOps.clipF64(data, 0, 20);
        expect(data, equals([0, 0, 5, 10, 15, 20, 20, 20]));
      });

      test('clips values to range (unaligned fallback)', () {
        // Create unaligned data by using sublist view
        final buffer = Float64List.fromList([0, -5, 0, 5, 10, 15, 20, 25, 30]);
        final unalignedData = Float64List.sublistView(buffer, 1);
        SimdOps.clipF64(unalignedData, 0, 20);
        expect(unalignedData, equals([0, 0, 5, 10, 15, 20, 20, 20]));
      });

      test('handles small arrays (scalar path)', () {
        // Length < 4 should use scalar path
        final data = Float64List.fromList([-5, 25]);
        SimdOps.clipF64(data, 0, 20);
        expect(data, equals([0, 20]));
      });

      test('handles empty array', () {
        final data = Float64List(0);
        SimdOps.clipF64(data, 0, 1);
        expect(data, isEmpty);
      });

      test('handles length not multiple of 2', () {
        final data = Float64List.fromList([-5, 10, 25, 15, 30]);
        SimdOps.clipF64(data, 0, 20);
        expect(data, equals([0, 10, 20, 15, 20]));
      });
    });

    group('absF64', () {
      test('computes absolute value (aligned)', () {
        final data = Float64List.fromList([-1, -2, -3, -4, 5, 6, 7, 8]);
        SimdOps.absF64(data);
        expect(data, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles unaligned data (scalar fallback)', () {
        final buffer = Float64List.fromList([0, -1, -2, -3, -4, 5, 6, 7, 8]);
        final unalignedData = Float64List.sublistView(buffer, 1);
        SimdOps.absF64(unalignedData);
        expect(unalignedData, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles small arrays', () {
        final data = Float64List.fromList([-3, 4]);
        SimdOps.absF64(data);
        expect(data, equals([3, 4]));
      });

      test('handles empty array', () {
        final data = Float64List(0);
        SimdOps.absF64(data);
        expect(data, isEmpty);
      });

      test('handles length not multiple of 2', () {
        final data = Float64List.fromList([-1, -2, -3, -4, -5]);
        SimdOps.absF64(data);
        expect(data, equals([1, 2, 3, 4, 5]));
      });
    });

    group('sqrtF64', () {
      test('computes square root (aligned)', () {
        final data = Float64List.fromList([1, 4, 9, 16, 25, 36, 49, 64]);
        SimdOps.sqrtF64(data);
        expect(data, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles unaligned data (scalar fallback)', () {
        final buffer = Float64List.fromList([0, 1, 4, 9, 16, 25, 36, 49, 64]);
        final unalignedData = Float64List.sublistView(buffer, 1);
        SimdOps.sqrtF64(unalignedData);
        expect(unalignedData, equals([1, 2, 3, 4, 5, 6, 7, 8]));
      });

      test('handles small arrays', () {
        final data = Float64List.fromList([4, 9]);
        SimdOps.sqrtF64(data);
        expect(data, equals([2, 3]));
      });

      test('handles empty array', () {
        final data = Float64List(0);
        SimdOps.sqrtF64(data);
        expect(data, isEmpty);
      });

      test('handles length not multiple of 2', () {
        final data = Float64List.fromList([1, 4, 9, 16, 25]);
        SimdOps.sqrtF64(data);
        expect(data, equals([1, 2, 3, 4, 5]));
      });
    });

    group('normalizeF64', () {
      test('normalizes with mean and std (aligned)', () {
        final data = Float64List.fromList([10, 20, 30, 40]);
        SimdOps.normalizeF64(data, 25.0, 10.0);
        expect(data[0], closeTo(-1.5, 1e-10));
        expect(data[1], closeTo(-0.5, 1e-10));
        expect(data[2], closeTo(0.5, 1e-10));
        expect(data[3], closeTo(1.5, 1e-10));
      });

      test('handles unaligned data (scalar fallback)', () {
        final buffer = Float64List.fromList([0, 10, 20, 30, 40]);
        final unalignedData = Float64List.sublistView(buffer, 1);
        SimdOps.normalizeF64(unalignedData, 25.0, 10.0);
        expect(unalignedData[0], closeTo(-1.5, 1e-10));
        expect(unalignedData[1], closeTo(-0.5, 1e-10));
        expect(unalignedData[2], closeTo(0.5, 1e-10));
        expect(unalignedData[3], closeTo(1.5, 1e-10));
      });

      test('handles small arrays', () {
        final data = Float64List.fromList([10, 30]);
        SimdOps.normalizeF64(data, 20.0, 10.0);
        expect(data[0], closeTo(-1.0, 1e-10));
        expect(data[1], closeTo(1.0, 1e-10));
      });

      test('handles empty array', () {
        final data = Float64List(0);
        SimdOps.normalizeF64(data, 0, 1);
        expect(data, isEmpty);
      });

      test('handles length not multiple of 2', () {
        final data = Float64List.fromList([10, 20, 30, 40, 50]);
        SimdOps.normalizeF64(data, 30.0, 10.0);
        expect(data[0], closeTo(-2.0, 1e-10));
        expect(data[1], closeTo(-1.0, 1e-10));
        expect(data[2], closeTo(0.0, 1e-10));
        expect(data[3], closeTo(1.0, 1e-10));
        expect(data[4], closeTo(2.0, 1e-10));
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

  group('Float64 SIMD Integration', () {
    test('ClipOp uses SIMD for Float64', () {
      final data = Float64List.fromList([-5, 0, 5, 10, 15, 20, 25, 30]);
      final tensor = TensorBuffer.fromFloat64List(data, [2, 4]);
      final result = ClipOp(min: 0, max: 20).apply(tensor);
      expect(result[[0, 0]], equals(0.0));
      expect(result[[0, 1]], equals(0.0));
      expect(result[[0, 2]], equals(5.0));
      expect(result[[0, 3]], equals(10.0));
      expect(result[[1, 0]], equals(15.0));
      expect(result[[1, 1]], equals(20.0));
      expect(result[[1, 2]], equals(20.0));
      expect(result[[1, 3]], equals(20.0));
    });

    test('AbsOp uses SIMD for Float64', () {
      final data = Float64List.fromList([-1, 2, -3, 4, -5, 6, -7, 8]);
      final tensor = TensorBuffer.fromFloat64List(data, [2, 4]);
      final result = AbsOp().apply(tensor);
      expect(result[[0, 0]], equals(1.0));
      expect(result[[0, 2]], equals(3.0));
      expect(result[[1, 0]], equals(5.0));
      expect(result[[1, 2]], equals(7.0));
    });

    test('SqrtOp uses SIMD for Float64', () {
      final data = Float64List.fromList([1, 4, 9, 16, 25, 36, 49, 64]);
      final tensor = TensorBuffer.fromFloat64List(data, [2, 4]);
      final result = SqrtOp().apply(tensor);
      expect(result[[0, 0]], equals(1.0));
      expect(result[[0, 1]], equals(2.0));
      expect(result[[0, 2]], equals(3.0));
      expect(result[[0, 3]], equals(4.0));
      expect(result[[1, 0]], equals(5.0));
      expect(result[[1, 1]], equals(6.0));
    });

    test('NormalizeOp uses SIMD for Float64', () {
      // Create a 3D tensor [C=1, H=2, W=4] with Float64
      final data = Float64List.fromList([10, 20, 30, 40, 50, 60, 70, 80]);
      final tensor = TensorBuffer.fromFloat64List(data, [1, 2, 4]);

      // Normalize with mean=45, std=10
      final result = NormalizeOp(mean: [45.0], std: [10.0]).apply(tensor);

      expect(result[[0, 0, 0]], closeTo(-3.5, 1e-10)); // (10-45)/10 = -3.5
      expect(result[[0, 0, 1]], closeTo(-2.5, 1e-10)); // (20-45)/10 = -2.5
      expect(result[[0, 0, 2]], closeTo(-1.5, 1e-10)); // (30-45)/10 = -1.5
      expect(result[[0, 0, 3]], closeTo(-0.5, 1e-10)); // (40-45)/10 = -0.5
      expect(result[[0, 1, 0]], closeTo(0.5, 1e-10)); // (50-45)/10 = 0.5
      expect(result[[0, 1, 1]], closeTo(1.5, 1e-10)); // (60-45)/10 = 1.5
    });
  });
}
