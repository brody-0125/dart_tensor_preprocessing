import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('BatchNormOp', () {
    group('constructor validation', () {
      test('throws on empty runningMean', () {
        expect(
          () => BatchNormOp(runningMean: [], runningVar: []),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched runningVar length', () {
        expect(
          () => BatchNormOp(
            runningMean: [0.5, 0.5, 0.5],
            runningVar: [0.25, 0.25],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched weight length', () {
        expect(
          () => BatchNormOp(
            runningMean: [0.5, 0.5, 0.5],
            runningVar: [0.25, 0.25, 0.25],
            weight: [1.0, 1.0],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched bias length', () {
        expect(
          () => BatchNormOp(
            runningMean: [0.5, 0.5, 0.5],
            runningVar: [0.25, 0.25, 0.25],
            bias: [0.0, 0.0],
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on eps <= 0', () {
        expect(
          () => BatchNormOp(runningMean: [0.5], runningVar: [0.25], eps: 0),
          throwsA(isA<InvalidParameterException>()),
        );

        expect(
          () => BatchNormOp(runningMean: [0.5], runningVar: [0.25], eps: -1e-5),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on negative variance', () {
        expect(
          () => BatchNormOp(runningMean: [0.5, 0.5], runningVar: [0.25, -0.1]),
          throwsA(isA<InvalidParameterException>()),
        );
      });
    });

    group('shape validation', () {
      test('throws on 2D tensor', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on 5D tensor', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );
        final tensor = TensorBuffer.zeros([1, 3, 4, 4, 4]);

        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on channel mismatch for 3D tensor', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );
        final tensor = TensorBuffer.zeros([2, 4, 4]); // 2 channels

        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on channel mismatch for 4D tensor', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );
        final tensor = TensorBuffer.zeros([1, 2, 4, 4]); // 2 channels

        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('3D tensor normalization', () {
      test('normalizes with identity parameters', () {
        // Identity: mean=0, var=1, weight=1, bias=0 => y = x
        final op = BatchNormOp(
          runningMean: [0.0, 0.0, 0.0],
          runningVar: [1.0, 1.0, 1.0],
          weight: [1.0, 1.0, 1.0],
          bias: [0.0, 0.0, 0.0],
        );

        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // Channel 0
            1.0, 2.0, 3.0, 4.0,
            // Channel 1
            5.0, 6.0, 7.0, 8.0,
            // Channel 2
            9.0, 10.0, 11.0, 12.0,
          ]),
          [3, 2, 2],
        );

        final output = op.apply(input);

        expect(output.shape, [3, 2, 2]);
        // With identity params, output should equal input
        for (int i = 0; i < input.numel; i++) {
          expect(
            output.storage.getAsDouble(i),
            closeTo(input.storage.getAsDouble(i), 1e-4),
          );
        }
      });

      test('normalizes with default weight/bias', () {
        // No weight/bias provided => defaults to weight=1, bias=0
        final op = BatchNormOp(
          runningMean: [1.0, 2.0],
          runningVar: [0.25, 0.25], // std = 0.5
        );

        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // Channel 0: values around mean=1
            0.5, 1.0, 1.5, 2.0,
            // Channel 1: values around mean=2
            1.5, 2.0, 2.5, 3.0,
          ]),
          [2, 2, 2],
        );

        final output = op.apply(input);

        // For channel 0: (x - 1) / sqrt(0.25 + 1e-5) ≈ (x - 1) / 0.5
        // For channel 1: (x - 2) / sqrt(0.25 + 1e-5) ≈ (x - 2) / 0.5
        final expected = [
          // Channel 0
          (0.5 - 1) / math.sqrt(0.25 + 1e-5),
          (1.0 - 1) / math.sqrt(0.25 + 1e-5),
          (1.5 - 1) / math.sqrt(0.25 + 1e-5),
          (2.0 - 1) / math.sqrt(0.25 + 1e-5),
          // Channel 1
          (1.5 - 2) / math.sqrt(0.25 + 1e-5),
          (2.0 - 2) / math.sqrt(0.25 + 1e-5),
          (2.5 - 2) / math.sqrt(0.25 + 1e-5),
          (3.0 - 2) / math.sqrt(0.25 + 1e-5),
        ];

        for (int i = 0; i < expected.length; i++) {
          expect(output.storage.getAsDouble(i), closeTo(expected[i], 1e-5));
        }
      });

      test('normalizes with custom weight and bias', () {
        final op = BatchNormOp(
          runningMean: [0.0],
          runningVar: [1.0],
          weight: [2.0],
          bias: [1.0],
        );

        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [1, 2, 2],
        );

        final output = op.apply(input);

        // y = x * 2 + 1
        expect(output.storage.getAsDouble(0), closeTo(3.0, 1e-4));
        expect(output.storage.getAsDouble(1), closeTo(5.0, 1e-4));
        expect(output.storage.getAsDouble(2), closeTo(7.0, 1e-4));
        expect(output.storage.getAsDouble(3), closeTo(9.0, 1e-4));
      });
    });

    group('4D tensor normalization', () {
      test('normalizes batch of images', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5],
          runningVar: [0.25, 0.25],
        );

        // 2 batches, 2 channels, 2x2 spatial
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // Batch 0
            // Channel 0
            0.0, 0.5, 1.0, 1.5,
            // Channel 1
            0.0, 0.5, 1.0, 1.5,
            // Batch 1
            // Channel 0
            0.25, 0.75, 1.25, 1.75,
            // Channel 1
            0.25, 0.75, 1.25, 1.75,
          ]),
          [2, 2, 2, 2],
        );

        final output = op.apply(input);

        expect(output.shape, [2, 2, 2, 2]);

        // Verify batch 0, channel 0, first element
        final invStd = 1.0 / math.sqrt(0.25 + 1e-5);
        expect(
          output.storage.getAsDouble(0),
          closeTo((0.0 - 0.5) * invStd, 1e-5),
        );
      });
    });

    group('fromStateDict factory', () {
      test('creates BatchNormOp from Float32Lists', () {
        final op = BatchNormOp.fromStateDict(
          runningMean: Float32List.fromList([0.5, 0.5, 0.5]),
          runningVar: Float32List.fromList([0.25, 0.25, 0.25]),
          weight: Float32List.fromList([1.0, 1.0, 1.0]),
          bias: Float32List.fromList([0.0, 0.0, 0.0]),
        );

        expect(op.numChannels, 3);
      });
    });

    group('in-place operation', () {
      test('applyInPlace modifies tensor directly', () {
        final op = BatchNormOp(runningMean: [0.5], runningVar: [0.25]);

        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([0.0, 0.5, 1.0, 1.5]),
          [1, 2, 2],
        );

        op.applyInPlace(tensor);

        final invStd = 1.0 / math.sqrt(0.25 + 1e-5);
        expect(tensor.storage.getAsDouble(0), closeTo(-0.5 * invStd, 1e-5));
        expect(tensor.storage.getAsDouble(1), closeTo(0.0, 1e-5));
      });

      test('applyInPlace throws on non-contiguous tensor', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );

        final tensor = TensorBuffer.zeros([3, 4, 4]).transpose([1, 0, 2]);

        expect(
          () => op.applyInPlace(tensor),
          throwsA(isA<NonContiguousException>()),
        );
      });
    });

    group('properties', () {
      test('name includes channel count and eps', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
          eps: 1e-6,
        );

        expect(op.name, contains('3'));
        expect(op.name, contains('0.000001'));
      });

      test('computeOutputShape returns input shape', () {
        final op = BatchNormOp(
          runningMean: [0.5, 0.5, 0.5],
          runningVar: [0.25, 0.25, 0.25],
        );

        expect(op.computeOutputShape([3, 224, 224]), [3, 224, 224]);
        expect(op.computeOutputShape([1, 3, 224, 224]), [1, 3, 224, 224]);
      });
    });

    group('non-contiguous input', () {
      test('handles transposed tensor', () {
        final op = BatchNormOp(runningMean: [0.0, 0.0], runningVar: [1.0, 1.0]);

        // Create transposed (non-contiguous) tensor
        final original = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
          [2, 2, 2],
        );
        final transposed = original.transpose([1, 0, 2]);

        // Should still work (will make contiguous internally)
        final output = op.apply(transposed);

        expect(output.shape, [2, 2, 2]);
        expect(output.isContiguous, true);
      });
    });
  });
}
