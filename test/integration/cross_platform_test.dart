import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

/// Cross-platform integration tests.
///
/// These tests run on all platforms and verify that tensor operations
/// produce consistent results regardless of the host OS.
void main() {
  group('Cross-Platform Integration', () {
    group('numerical consistency', () {
      test('tensor creation produces identical results', () {
        final zeros = TensorBuffer.zeros([4, 4]);
        expect(zeros.numel, equals(16));
        for (int i = 0; i < zeros.numel; i++) {
          expect(zeros.storage.getAsDouble(i), equals(0.0));
        }

        final ones = TensorBuffer.ones([4, 4]);
        for (int i = 0; i < ones.numel; i++) {
          expect(ones.storage.getAsDouble(i), equals(1.0));
        }
      });

      test('ImageNet normalization produces consistent values', () {
        // Fixed input: 3-channel 2x2 tensor with value 0.5
        final data = Float32List.fromList(List.filled(12, 0.5));
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        // Channel 0: (0.5 - 0.485) / 0.229
        expect(result[[0, 0, 0]], closeTo(0.0655, 0.01));
        // Channel 1: (0.5 - 0.456) / 0.224
        expect(result[[1, 0, 0]], closeTo(0.1964, 0.01));
        // Channel 2: (0.5 - 0.406) / 0.225
        expect(result[[2, 0, 0]], closeTo(0.4178, 0.01));
      });

      test('activation functions produce consistent values', () {
        final data = Float32List.fromList([-2, -1, 0, 1, 2, 3, -0.5, 0.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 8]);

        // ReLU
        final reluResult = ReLUOp().apply(tensor);
        expect(reluResult[[0, 0, 0]], equals(0.0)); // -2 -> 0
        expect(reluResult[[0, 0, 3]], equals(1.0)); // 1 -> 1
        expect(reluResult[[0, 0, 5]], equals(3.0)); // 3 -> 3

        // Sigmoid
        final sigmoidResult = SigmoidOp().apply(tensor);
        expect(sigmoidResult[[0, 0, 2]], closeTo(0.5, 1e-5)); // sigmoid(0) = 0.5
      });

      test('arithmetic operations produce consistent values', () {
        final a = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4]),
          [2, 2],
        );
        final b = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 6, 7, 8]),
          [2, 2],
        );

        final addResult = AddOp.tensor(b).apply(a);
        expect(addResult[[0, 0]], closeTo(6.0, 1e-5));
        expect(addResult[[1, 1]], closeTo(12.0, 1e-5));

        final mulResult = MulOp.tensor(b).apply(a);
        expect(mulResult[[0, 0]], closeTo(5.0, 1e-5));
        expect(mulResult[[1, 1]], closeTo(32.0, 1e-5));
      });
    });

    group('pipeline consistency', () {
      test('pipeline produces same output shape on all platforms', () {
        final pipeline = TensorPipeline([
          ResizeOp(height: 64, width: 64),
          NormalizeOp.imagenet(),
          UnsqueezeOp.batch(),
        ]);

        final outputShape = pipeline.computeOutputShape([3, 128, 128]);
        expect(outputShape, equals([1, 3, 64, 64]));
      });

      test('pipeline chaining works correctly', () {
        final p1 = TensorPipeline([
          NormalizeOp.symmetric(),
        ]);
        final p2 = TensorPipeline([
          UnsqueezeOp.batch(),
        ]);
        final combined = p1 + p2;

        expect(combined.length, equals(2));

        final outputShape = combined.computeOutputShape([3, 32, 32]);
        expect(outputShape, equals([1, 3, 32, 32]));
      });
    });

    group('async execution consistency', () {
      test('sync and async produce same results', () async {
        final data = Float32List(3 * 32 * 32);
        for (int i = 0; i < data.length; i++) {
          data[i] = (i % 256) / 255.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 32, 32]);

        final pipeline = TensorPipeline([
          NormalizeOp.symmetric(),
        ]);

        final syncResult = pipeline.run(tensor);
        final asyncResult = await pipeline.runAsync(tensor);

        expect(syncResult.shape, equals(asyncResult.shape));
        expect(syncResult.dtype, equals(asyncResult.dtype));

        // Compare values
        for (int i = 0; i < syncResult.numel; i++) {
          expect(
            syncResult.storage.getAsDouble(i),
            closeTo(asyncResult.storage.getAsDouble(i), 1e-5),
          );
        }
      });
    });

    group('edge cases consistency', () {
      test('invalid shape throws exception', () {
        expect(
          () => TensorBuffer.zeros([0]),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('single element tensor', () {
        final data = Float32List.fromList([42.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [1]);

        expect(tensor.numel, equals(1));
        expect(tensor[[0]], equals(42.0));
      });

      test('high-dimensional tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4, 5, 6]);
        expect(tensor.numel, equals(720));
        expect(tensor.shape, equals([2, 3, 4, 5, 6]));
      });
    });
  });
}
