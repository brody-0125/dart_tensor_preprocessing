import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('LayerNormOp', () {
    group('constructor validation', () {
      test('throws on empty normalizedShape', () {
        expect(
          () => LayerNormOp(normalizedShape: []),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on zero in normalizedShape', () {
        expect(
          () => LayerNormOp(normalizedShape: [768, 0]),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on negative in normalizedShape', () {
        expect(
          () => LayerNormOp(normalizedShape: [-1]),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched weight length', () {
        expect(
          () => LayerNormOp(
            normalizedShape: [4],
            weight: [1.0, 1.0, 1.0], // 3 != 4
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched bias length', () {
        expect(
          () => LayerNormOp(
            normalizedShape: [4],
            bias: [0.0, 0.0], // 2 != 4
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on eps <= 0', () {
        expect(
          () => LayerNormOp(normalizedShape: [4], eps: 0),
          throwsA(isA<InvalidParameterException>()),
        );

        expect(
          () => LayerNormOp(normalizedShape: [4], eps: -1e-5),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts multi-dimensional normalizedShape', () {
        final op = LayerNormOp(
          normalizedShape: [2, 3],
          weight: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          bias: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        expect(op.normalizedSize, 6);
      });
    });

    group('shape validation', () {
      test('throws on insufficient rank', () {
        final op = LayerNormOp(normalizedShape: [4, 4]);
        final tensor = TensorBuffer.zeros([4]); // rank 1 < 2

        expect(
          () => op.apply(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws on trailing dimension mismatch', () {
        final op = LayerNormOp(normalizedShape: [768]);
        final tensor = TensorBuffer.zeros([2, 512]); // 512 != 768

        expect(
          () => op.apply(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws on multi-dim shape mismatch', () {
        final op = LayerNormOp(normalizedShape: [4, 4]);
        final tensor = TensorBuffer.zeros([2, 4, 3]); // [4, 3] != [4, 4]

        expect(
          () => op.apply(tensor),
          throwsA(isA<ShapeMismatchException>()),
        );
      });
    });

    group('2D tensor normalization', () {
      test('normalizes with identity parameters', () {
        // Identity: weight=1, bias=0
        final op = LayerNormOp(
          normalizedShape: [4],
          weight: [1.0, 1.0, 1.0, 1.0],
          bias: [0.0, 0.0, 0.0, 0.0],
        );

        // Values with known mean=2.5, var=1.25
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [1, 4],
        );

        final output = op.apply(input);

        expect(output.shape, [1, 4]);

        // Expected: (x - 2.5) / sqrt(1.25 + 1e-5)
        const mean = 2.5;
        const variance = 1.25;
        final invStd = 1.0 / math.sqrt(variance + 1e-5);

        for (int i = 0; i < 4; i++) {
          final expected = (input.storage.getAsDouble(i) - mean) * invStd;
          expect(output.storage.getAsDouble(i), closeTo(expected, 1e-4));
        }
      });

      test('normalizes batch of vectors', () {
        final op = LayerNormOp(normalizedShape: [3]);

        // 2 samples, 3 features each
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            1.0, 2.0, 3.0, // Sample 0: mean=2, var=2/3
            4.0, 5.0, 6.0, // Sample 1: mean=5, var=2/3
          ]),
          [2, 3],
        );

        final output = op.apply(input);

        expect(output.shape, [2, 3]);

        // Sample 0
        const mean0 = 2.0;
        const var0 = 2.0 / 3.0;
        final invStd0 = 1.0 / math.sqrt(var0 + 1e-5);
        expect(output.storage.getAsDouble(0), closeTo((1.0 - mean0) * invStd0, 1e-4));
        expect(output.storage.getAsDouble(1), closeTo((2.0 - mean0) * invStd0, 1e-4));
        expect(output.storage.getAsDouble(2), closeTo((3.0 - mean0) * invStd0, 1e-4));

        // Sample 1
        const mean1 = 5.0;
        const var1 = 2.0 / 3.0;
        final invStd1 = 1.0 / math.sqrt(var1 + 1e-5);
        expect(output.storage.getAsDouble(3), closeTo((4.0 - mean1) * invStd1, 1e-4));
        expect(output.storage.getAsDouble(4), closeTo((5.0 - mean1) * invStd1, 1e-4));
        expect(output.storage.getAsDouble(5), closeTo((6.0 - mean1) * invStd1, 1e-4));
      });

      test('normalizes with custom weight and bias', () {
        final op = LayerNormOp(
          normalizedShape: [2],
          weight: [2.0, 0.5],
          bias: [1.0, -1.0],
        );

        // Values: [0.0, 2.0] -> mean=1, var=1
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([0.0, 2.0]),
          [1, 2],
        );

        final output = op.apply(input);

        const mean = 1.0;
        const variance = 1.0;
        final invStd = 1.0 / math.sqrt(variance + 1e-5);

        // First element: (0 - 1) / sqrt(1.00001) * 2 + 1
        final norm0 = (0.0 - mean) * invStd;
        expect(output.storage.getAsDouble(0), closeTo(norm0 * 2.0 + 1.0, 1e-4));

        // Second element: (2 - 1) / sqrt(1.00001) * 0.5 - 1
        final norm1 = (2.0 - mean) * invStd;
        expect(output.storage.getAsDouble(1), closeTo(norm1 * 0.5 - 1.0, 1e-4));
      });
    });

    group('3D tensor normalization', () {
      test('normalizes [batch, seq, features] tensor', () {
        final op = LayerNormOp(normalizedShape: [4]);

        // [batch=2, seq=2, features=4]
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // Batch 0
            1.0, 2.0, 3.0, 4.0, // seq 0
            5.0, 6.0, 7.0, 8.0, // seq 1
            // Batch 1
            0.0, 1.0, 2.0, 3.0, // seq 0
            4.0, 5.0, 6.0, 7.0, // seq 1
          ]),
          [2, 2, 4],
        );

        final output = op.apply(input);

        expect(output.shape, [2, 2, 4]);

        // Each sequence position is normalized independently
        // Batch 0, seq 0: [1,2,3,4] -> mean=2.5, var=1.25
        const mean = 2.5;
        const variance = 1.25;
        final invStd = 1.0 / math.sqrt(variance + 1e-5);
        expect(output.storage.getAsDouble(0), closeTo((1.0 - mean) * invStd, 1e-4));
      });

      test('normalizes over last 2 dimensions', () {
        // Normalize over [2, 3] = 6 elements
        final op = LayerNormOp(normalizedShape: [2, 3]);

        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // Instance 0 (6 elements)
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            // Instance 1 (6 elements)
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
          ]),
          [2, 2, 3],
        );

        final output = op.apply(input);

        expect(output.shape, [2, 2, 3]);

        // Instance 0: mean=3.5
        const mean0 = 3.5;
        var var0 = 0.0;
        for (int i = 0; i < 6; i++) {
          final diff = (i + 1).toDouble() - mean0;
          var0 += diff * diff;
        }
        var0 /= 6;
        final invStd0 = 1.0 / math.sqrt(var0 + 1e-5);
        expect(output.storage.getAsDouble(0), closeTo((1.0 - mean0) * invStd0, 1e-4));
      });
    });

    group('factory constructors', () {
      test('bert() creates 768-dim normalization', () {
        final weight = List.filled(768, 1.0);
        final bias = List.filled(768, 0.0);
        final op = LayerNormOp.bert(weight: weight, bias: bias);

        expect(op.normalizedSize, 768);
        expect(op.normalizedShape, [768]);
      });

      test('bertLarge() creates 1024-dim normalization', () {
        final weight = List.filled(1024, 1.0);
        final bias = List.filled(1024, 0.0);
        final op = LayerNormOp.bertLarge(weight: weight, bias: bias);

        expect(op.normalizedSize, 1024);
        expect(op.normalizedShape, [1024]);
      });

      test('fromStateDict() creates from Float32Lists', () {
        final op = LayerNormOp.fromStateDict(
          normalizedShape: [4],
          weight: Float32List.fromList([1.0, 1.0, 1.0, 1.0]),
          bias: Float32List.fromList([0.0, 0.0, 0.0, 0.0]),
        );

        expect(op.normalizedSize, 4);
      });
    });

    group('in-place operation', () {
      test('applyInPlace modifies tensor directly', () {
        final op = LayerNormOp(normalizedShape: [4]);

        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [1, 4],
        );

        op.applyInPlace(tensor);

        // Should be normalized now
        const mean = 2.5;
        const variance = 1.25;
        final invStd = 1.0 / math.sqrt(variance + 1e-5);
        expect(tensor.storage.getAsDouble(0), closeTo((1.0 - mean) * invStd, 1e-4));
      });

      test('applyInPlace throws on non-contiguous tensor', () {
        final op = LayerNormOp(normalizedShape: [4]);

        final tensor = TensorBuffer.zeros([4, 4]).transpose([1, 0]);

        expect(
          () => op.applyInPlace(tensor),
          throwsA(isA<NonContiguousException>()),
        );
      });
    });

    group('properties', () {
      test('name includes shape and eps', () {
        final op = LayerNormOp(
          normalizedShape: [768],
          eps: 1e-6,
        );

        expect(op.name, contains('[768]'));
        expect(op.name, contains('0.000001'));
      });

      test('computeOutputShape returns input shape', () {
        final op = LayerNormOp(normalizedShape: [768]);

        expect(op.computeOutputShape([2, 10, 768]), [2, 10, 768]);
        expect(op.computeOutputShape([768]), [768]);
      });
    });

    group('non-contiguous input', () {
      test('handles transposed tensor', () {
        final op = LayerNormOp(normalizedShape: [2]);

        // Create transposed (non-contiguous) tensor
        final original = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4]),
          [2, 2],
        );
        final transposed = original.transpose([1, 0]);

        // Should still work (will make contiguous internally)
        final output = op.apply(transposed);

        expect(output.shape, [2, 2]);
        expect(output.isContiguous, true);
      });
    });

    group('edge cases', () {
      test('handles single element normalization', () {
        final op = LayerNormOp(
          normalizedShape: [1],
          weight: [2.0],
          bias: [1.0],
        );

        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([5.0]),
          [1, 1],
        );

        final output = op.apply(input);

        // Single element: mean=5, var=0, (x-mean)/sqrt(0+eps) * 2 + 1
        // With var=0, the normalized value approaches 0
        expect(output.storage.getAsDouble(0), closeTo(1.0, 1e-4)); // 0 * 2 + 1 = 1
      });

      test('handles constant input (zero variance)', () {
        final op = LayerNormOp(normalizedShape: [4]);

        // All same values -> mean=5, var=0
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([5.0, 5.0, 5.0, 5.0]),
          [1, 4],
        );

        final output = op.apply(input);

        // With zero variance, normalized values are all 0 (before affine)
        for (int i = 0; i < 4; i++) {
          expect(output.storage.getAsDouble(i), closeTo(0.0, 1e-4));
        }
      });
    });
  });
}
