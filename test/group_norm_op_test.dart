import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('GroupNormOp', () {
    group('constructor validation', () {
      test('throws on numGroups <= 0', () {
        expect(
          () => GroupNormOp(numGroups: 0, numChannels: 4),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => GroupNormOp(numGroups: -1, numChannels: 4),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on numChannels <= 0', () {
        expect(
          () => GroupNormOp(numGroups: 2, numChannels: 0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => GroupNormOp(numGroups: 2, numChannels: -4),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws when numChannels not divisible by numGroups', () {
        expect(
          () => GroupNormOp(numGroups: 3, numChannels: 4),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched weight length', () {
        expect(
          () => GroupNormOp(
            numGroups: 2,
            numChannels: 4,
            weight: [1.0, 1.0], // should be 4
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on mismatched bias length', () {
        expect(
          () => GroupNormOp(
            numGroups: 2,
            numChannels: 4,
            bias: [0.0, 0.0, 0.0], // should be 4
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on eps <= 0', () {
        expect(
          () => GroupNormOp(numGroups: 2, numChannels: 4, eps: 0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => GroupNormOp(numGroups: 2, numChannels: 4, eps: -1e-5),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts valid parameters', () {
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        expect(op.numGroups, equals(2));
        expect(op.numChannels, equals(4));
        expect(op.eps, equals(1e-5));
      });
    });

    group('shape validation', () {
      test('throws on 2D tensor', () {
        final tensor = TensorBuffer.zeros([4, 16]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on 5D tensor', () {
        final tensor = TensorBuffer.zeros([1, 4, 2, 2, 2]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on channel mismatch for 3D', () {
        final tensor = TensorBuffer.zeros([8, 2, 2]); // 8 channels
        final op = GroupNormOp(numGroups: 2, numChannels: 4); // expects 4
        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws on channel mismatch for 4D', () {
        final tensor = TensorBuffer.zeros([1, 8, 2, 2]); // 8 channels
        final op = GroupNormOp(numGroups: 2, numChannels: 4); // expects 4
        expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('accepts valid 3D tensor', () {
        final tensor = TensorBuffer.zeros([4, 2, 2]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);
        expect(result.shape, equals([4, 2, 2]));
      });

      test('accepts valid 4D tensor', () {
        final tensor = TensorBuffer.zeros([2, 4, 2, 2]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);
        expect(result.shape, equals([2, 4, 2, 2]));
      });
    });

    group('3D normalization', () {
      test('normalizes constant input to zeros (without affine)', () {
        // All values are 5.0, so after normalization they should be 0
        final tensor = TensorBuffer.full([4, 2, 2], fillValue: 5.0);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);

        for (int c = 0; c < 4; c++) {
          for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
              expect(result[[c, h, w]], closeTo(0.0, 1e-5));
            }
          }
        }
      });

      test('normalizes with varying values', () {
        // Create tensor with known mean and variance within each group
        // Group 0: channels 0-1, Group 1: channels 2-3
        final data = Float32List(4 * 2 * 2);
        // Group 0 values: all 1.0 in channel 0, all 3.0 in channel 1
        // Mean = 2.0, Std = 1.0
        for (int i = 0; i < 4; i++) {
          data[0 * 4 + i] = 1.0; // channel 0
          data[1 * 4 + i] = 3.0; // channel 1
        }
        // Group 1 values: all 0.0 in channel 2, all 4.0 in channel 3
        // Mean = 2.0, Std = 2.0
        for (int i = 0; i < 4; i++) {
          data[2 * 4 + i] = 0.0; // channel 2
          data[3 * 4 + i] = 4.0; // channel 3
        }

        final tensor = TensorBuffer.fromFloat32List(data, [4, 2, 2]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);

        // Group 0: (1-2)/1 = -1, (3-2)/1 = 1
        expect(result[[0, 0, 0]], closeTo(-1.0, 1e-5));
        expect(result[[1, 0, 0]], closeTo(1.0, 1e-5));

        // Group 1: (0-2)/2 = -1, (4-2)/2 = 1
        expect(result[[2, 0, 0]], closeTo(-1.0, 1e-5));
        expect(result[[3, 0, 0]], closeTo(1.0, 1e-5));
      });

      test('applies weight and bias correctly', () {
        final tensor = TensorBuffer.full([4, 2, 2], fillValue: 0.0);
        // Set channel 0 to -1.0, channel 1 to 1.0 (so mean=0, std=1)
        for (int h = 0; h < 2; h++) {
          for (int w = 0; w < 2; w++) {
            // Manually set values - we need to use storage
            final idx0 = 0 * 4 + h * 2 + w;
            final idx1 = 1 * 4 + h * 2 + w;
            (tensor.storage.data as Float32List)[idx0] = -1.0;
            (tensor.storage.data as Float32List)[idx1] = 1.0;
          }
        }

        final op = GroupNormOp(
          numGroups: 2,
          numChannels: 4,
          weight: [2.0, 2.0, 1.0, 1.0],
          bias: [1.0, 1.0, 0.0, 0.0],
        );
        final result = op.apply(tensor);

        // Channel 0: normalized = -1, with weight=2, bias=1: -1*2+1 = -1
        expect(result[[0, 0, 0]], closeTo(-1.0, 1e-4));
        // Channel 1: normalized = 1, with weight=2, bias=1: 1*2+1 = 3
        expect(result[[1, 0, 0]], closeTo(3.0, 1e-4));
      });
    });

    group('4D normalization', () {
      test('normalizes each batch independently', () {
        // Batch 0: all 1s, Batch 1: all 2s
        final data = Float32List(2 * 4 * 2 * 2);
        const batchStride = 4 * 2 * 2;
        for (int i = 0; i < batchStride; i++) {
          data[i] = 1.0;
          data[batchStride + i] = 2.0;
        }

        final tensor = TensorBuffer.fromFloat32List(data, [2, 4, 2, 2]);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);

        // Both batches should normalize to 0 (constant input)
        for (int n = 0; n < 2; n++) {
          for (int c = 0; c < 4; c++) {
            expect(result[[n, c, 0, 0]], closeTo(0.0, 1e-5));
          }
        }
      });

      test('preserves output shape', () {
        final tensor = TensorBuffer.zeros([3, 8, 4, 4]);
        final op = GroupNormOp(numGroups: 4, numChannels: 8);
        final result = op.apply(tensor);
        expect(result.shape, equals([3, 8, 4, 4]));
      });
    });

    group('factory methods', () {
      test('withAffine creates correct weights and biases', () {
        final op = GroupNormOp.withAffine(numGroups: 2, numChannels: 4);
        expect(op.weight, equals([1.0, 1.0, 1.0, 1.0]));
        expect(op.bias, equals([0.0, 0.0, 0.0, 0.0]));
      });

      test('fromStateDict accepts Float32List', () {
        final weight = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
        final bias = Float32List.fromList([0.1, 0.2, 0.3, 0.4]);

        final op = GroupNormOp.fromStateDict(
          numGroups: 2,
          numChannels: 4,
          weight: weight,
          bias: bias,
        );

        expect(op.weight, equals([1.0, 2.0, 3.0, 4.0]));
        // Use closeTo for Float32 precision
        expect(op.bias![0], closeTo(0.1, 1e-6));
        expect(op.bias![1], closeTo(0.2, 1e-6));
        expect(op.bias![2], closeTo(0.3, 1e-6));
        expect(op.bias![3], closeTo(0.4, 1e-6));
      });
    });

    group('dtype support', () {
      test('handles Float64 tensors', () {
        final tensor = TensorBuffer.zeros([4, 2, 2], dtype: DType.float64);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        final result = op.apply(tensor);
        expect(result.dtype, equals(DType.float64));
        expect(result.shape, equals([4, 2, 2]));
      });
    });

    group('in-place operation', () {
      test('applyInPlace modifies tensor', () {
        final tensor = TensorBuffer.full([4, 2, 2], fillValue: 5.0);
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        op.applyInPlace(tensor);

        // After normalization, constant input becomes 0
        expect(tensor[[0, 0, 0]], closeTo(0.0, 1e-5));
      });

      test('applyInPlace throws on non-contiguous', () {
        final tensor = TensorBuffer.zeros([4, 4, 4]);
        final transposed = tensor.transpose([1, 0, 2]); // Non-contiguous
        final op = GroupNormOp(numGroups: 2, numChannels: 4);

        expect(
          () => op.applyInPlace(transposed),
          throwsA(isA<NonContiguousException>()),
        );
      });
    });

    group('name', () {
      test('returns descriptive name', () {
        final op = GroupNormOp(numGroups: 8, numChannels: 32);
        expect(op.name, equals('GroupNorm(groups=8, channels=32)'));
      });
    });

    group('computeOutputShape', () {
      test('preserves input shape', () {
        final op = GroupNormOp(numGroups: 2, numChannels: 4);
        expect(op.computeOutputShape([4, 8, 8]), equals([4, 8, 8]));
        expect(op.computeOutputShape([2, 4, 16, 16]), equals([2, 4, 16, 16]));
      });
    });
  });
}
