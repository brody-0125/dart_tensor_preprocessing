import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('ResizeOp', () {
    group('InterpolationMode.nearest', () {
      test('upscales 2x2 to 4x4', () {
        final data = Float32List.fromList([
          1,
          2,
          3,
          4,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);
        final op = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.nearest,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 4, 4]));
        // Check corners
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 3]], equals(2.0));
        expect(result[[0, 3, 0]], equals(3.0));
        expect(result[[0, 3, 3]], equals(4.0));
      });

      test('downscales 4x4 to 2x2', () {
        final data = Float32List.fromList([
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 4, 4]);
        final op = ResizeOp(
          height: 2,
          width: 2,
          mode: InterpolationMode.nearest,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 2, 2]));
      });
    });

    group('InterpolationMode.bilinear', () {
      test('upscales with interpolation', () {
        final data = Float32List.fromList([
          0,
          10,
          10,
          20,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);
        final op = ResizeOp(
          height: 3,
          width: 3,
          mode: InterpolationMode.bilinear,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 3, 3]));
        // Corners should be preserved (approximately)
        expect(result[[0, 0, 0]], closeTo(0.0, 0.5));
        expect(result[[0, 0, 2]], closeTo(10.0, 0.5));
      });
    });

    group('InterpolationMode.bicubic', () {
      test('upscales with smooth interpolation', () {
        final tensor = TensorBuffer.zeros([1, 4, 4]);
        // Set center values
        (tensor.storage.data as Float32List)[5] = 100.0;
        (tensor.storage.data as Float32List)[6] = 100.0;
        (tensor.storage.data as Float32List)[9] = 100.0;
        (tensor.storage.data as Float32List)[10] = 100.0;

        final op = ResizeOp(
          height: 8,
          width: 8,
          mode: InterpolationMode.bicubic,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 8, 8]));
        // Center should have high values
        expect(result[[0, 3, 3]], greaterThan(50.0));
      });

      test('uses cache-friendly blocking for large tensors', () {
        // Create a larger tensor to trigger blocking
        final tensor = TensorBuffer.full([3, 100, 100], fillValue: 50.0);
        final op = ResizeOp(
          height: 200,
          width: 200,
          mode: InterpolationMode.bicubic,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([3, 200, 200]));
        // Values should be approximately preserved - check center pixel
        // Bicubic interpolation of constant values should preserve the value
        expect(result[[0, 100, 100]], closeTo(50.0, 5.0));
        // Also check other positions to verify blocking works
        expect(result[[1, 50, 50]], closeTo(50.0, 5.0));
        expect(result[[2, 150, 150]], closeTo(50.0, 5.0));
      });
    });

    group('InterpolationMode.area', () {
      test('downscales 4x4 to 2x2 with averaging', () {
        final data = Float32List.fromList([
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 4, 4]);
        final op = ResizeOp(
          height: 2,
          width: 2,
          mode: InterpolationMode.area,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 2, 2]));
        // Each 2x2 block should be averaged
        // Top-left: (1+2+5+6)/4 = 3.5
        expect(result[[0, 0, 0]], closeTo(3.5, 0.5));
        // Top-right: (3+4+7+8)/4 = 5.5
        expect(result[[0, 0, 1]], closeTo(5.5, 0.5));
        // Bottom-left: (9+10+13+14)/4 = 11.5
        expect(result[[0, 1, 0]], closeTo(11.5, 0.5));
        // Bottom-right: (11+12+15+16)/4 = 13.5
        expect(result[[0, 1, 1]], closeTo(13.5, 0.5));
      });

      test('handles fractional scale factors', () {
        final tensor = TensorBuffer.full([3, 10, 10], fillValue: 100.0);
        final op = ResizeOp(
          height: 3,
          width: 3,
          mode: InterpolationMode.area,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([3, 3, 3]));
        // All values should be approximately 100 (constant input)
        expect(result[[0, 1, 1]], closeTo(100.0, 1.0));
      });

      test('falls back to bilinear for upscaling', () {
        final data = Float32List.fromList([
          1,
          2,
          3,
          4,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);
        final op = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.area,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 4, 4]));
        // Should produce interpolated values
        expect(result[[0, 0, 0]], closeTo(1.0, 0.5));
      });

      test('preserves values for same-size resize', () {
        final tensor = TensorBuffer.full([3, 8, 8], fillValue: 42.0);
        final op = ResizeOp(
          height: 8,
          width: 8,
          mode: InterpolationMode.area,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([3, 8, 8]));
        expect(result[[0, 4, 4]], closeTo(42.0, 0.1));
      });
    });

    group('InterpolationMode.lanczos', () {
      test('upscales with Lanczos3 kernel', () {
        final data = Float32List.fromList([
          0,
          100,
          0,
          100,
          255,
          100,
          0,
          100,
          0,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 3, 3]);
        final op = ResizeOp(
          height: 6,
          width: 6,
          mode: InterpolationMode.lanczos,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 6, 6]));
        // Center should have high values
        expect(result[[0, 3, 3]], greaterThan(150.0));
      });

      test('downscales with anti-aliasing', () {
        final tensor = TensorBuffer.full([3, 100, 100], fillValue: 128.0);
        final op = ResizeOp(
          height: 25,
          width: 25,
          mode: InterpolationMode.lanczos,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([3, 25, 25]));
        // Values should be approximately preserved
        expect(result[[0, 12, 12]], closeTo(128.0, 10.0));
      });

      test('handles edge cases at boundaries', () {
        final tensor = TensorBuffer.full([1, 10, 10], fillValue: 50.0);
        final op = ResizeOp(
          height: 20,
          width: 20,
          mode: InterpolationMode.lanczos,
        );
        final result = op.apply(tensor);

        expect(result.shape, equals([1, 20, 20]));
        // Edges should have reasonable values
        expect(result[[0, 0, 0]], closeTo(50.0, 5.0));
        expect(result[[0, 19, 19]], closeTo(50.0, 5.0));
      });
    });

    group('4D tensor support', () {
      test('handles batch dimension with all modes', () {
        final tensor = TensorBuffer.full([2, 3, 8, 8], fillValue: 1.0);

        for (final mode in InterpolationMode.values) {
          final op = ResizeOp(height: 4, width: 4, mode: mode);
          final result = op.apply(tensor);
          expect(result.shape, equals([2, 3, 4, 4]),
              reason: 'Failed for mode: $mode');
        }
      });
    });

    group('dtype support', () {
      test('handles Float64 tensors with all modes', () {
        final tensor =
            TensorBuffer.full([1, 4, 4], fillValue: 10.0, dtype: DType.float64);

        for (final mode in InterpolationMode.values) {
          final op = ResizeOp(height: 8, width: 8, mode: mode);
          final result = op.apply(tensor);
          expect(result.dtype, equals(DType.float64),
              reason: 'Failed for mode: $mode');
        }
      });
    });

    group('computeOutputShape', () {
      test('calculates correct output shape for 3D', () {
        final op = ResizeOp(height: 224, width: 224);
        expect(op.computeOutputShape([3, 480, 640]), equals([3, 224, 224]));
      });

      test('calculates correct output shape for 4D', () {
        final op = ResizeOp(height: 224, width: 224);
        expect(
            op.computeOutputShape([1, 3, 480, 640]), equals([1, 3, 224, 224]));
      });
    });
  });
}
