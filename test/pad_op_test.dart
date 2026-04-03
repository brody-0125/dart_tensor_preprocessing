import 'dart:typed_data';

import 'package:test/test.dart';

// Import all required classes directly since PadOp is not exported yet
import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/pad_op.dart';

void main() {
  group('PadOp', () {
    group('constructor validation', () {
      test('throws for negative padding sizes', () {
        expect(
          () => PadOp(top: -1, bottom: 0, left: 0, right: 0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => PadOp(top: 0, bottom: -1, left: 0, right: 0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => PadOp(top: 0, bottom: 0, left: -1, right: 0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => PadOp(top: 0, bottom: 0, left: 0, right: -1),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for invalid constant value', () {
        expect(
          () => PadOp(top: 1, bottom: 1, left: 1, right: 1, value: double.nan),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => PadOp(
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
            value: double.infinity,
          ),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts valid parameters', () {
        expect(
          () => PadOp(top: 1, bottom: 2, left: 3, right: 4),
          returnsNormally,
        );
        expect(
          () => PadOp(top: 0, bottom: 0, left: 0, right: 0),
          returnsNormally,
        );
      });
    });

    group('factory constructors', () {
      test('symmetric creates equal padding', () {
        final pad = PadOp.symmetric(2, 3);
        expect(pad.top, equals(2));
        expect(pad.bottom, equals(2));
        expect(pad.left, equals(3));
        expect(pad.right, equals(3));
        expect(pad.mode, equals(PadMode.constant));
        expect(pad.value, equals(0.0));
      });

      test('all creates square padding', () {
        final pad = PadOp.all(1, mode: PadMode.reflect, value: 5.0);
        expect(pad.top, equals(1));
        expect(pad.bottom, equals(1));
        expect(pad.left, equals(1));
        expect(pad.right, equals(1));
        expect(pad.mode, equals(PadMode.reflect));
        expect(pad.value, equals(5.0));
      });
    });

    group('shape validation', () {
      test('throws for invalid tensor ranks', () {
        final data1d = Float32List.fromList([1, 2, 3]);
        final tensor1d = TensorBuffer.fromFloat32List(data1d, [3]);

        final data2d = Float32List.fromList([1, 2, 3, 4]);
        final tensor2d = TensorBuffer.fromFloat32List(data2d, [2, 2]);

        final data5d = Float32List(16)..fillRange(0, 16, 1.0);
        final tensor5d = TensorBuffer.fromFloat32List(data5d, [2, 2, 2, 2, 1]);

        final pad = PadOp(top: 1, bottom: 1, left: 1, right: 1);

        expect(() => pad(tensor1d), throwsA(isA<ShapeMismatchException>()));
        expect(() => pad(tensor2d), throwsA(isA<ShapeMismatchException>()));
        expect(() => pad(tensor5d), throwsA(isA<ShapeMismatchException>()));
      });
    });

    group('output shape computation', () {
      test('computes correct shape for 3D tensors', () {
        final pad = PadOp(top: 2, bottom: 3, left: 1, right: 4);
        final inputShape = [3, 4, 5]; // [C, H, W]
        final outputShape = pad.computeOutputShape(inputShape);

        expect(
          outputShape,
          equals([3, 4 + 2 + 3, 5 + 1 + 4]),
        ); // [C, H+top+bottom, W+left+right]
      });

      test('computes correct shape for 4D tensors', () {
        final pad = PadOp(top: 1, bottom: 2, left: 3, right: 4);
        final inputShape = [2, 3, 4, 5]; // [N, C, H, W]
        final outputShape = pad.computeOutputShape(inputShape);

        expect(
          outputShape,
          equals([2, 3, 4 + 1 + 2, 5 + 3 + 4]),
        ); // [N, C, H+top+bottom, W+left+right]
      });
    });

    group('constant padding', () {
      test('pads 3D tensor with constant value', () {
        // Create a simple 2x2 tensor with 1 channel
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final pad = PadOp(top: 1, bottom: 1, left: 1, right: 1, value: 0.0);
        final result = pad(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check corners (should be padding)
        expect(result[[0, 0, 0]], equals(0.0)); // top-left
        expect(result[[0, 0, 3]], equals(0.0)); // top-right
        expect(result[[0, 3, 0]], equals(0.0)); // bottom-left
        expect(result[[0, 3, 3]], equals(0.0)); // bottom-right

        // Check center (should be original data)
        expect(result[[0, 1, 1]], equals(1.0)); // original [0,0]
        expect(result[[0, 1, 2]], equals(2.0)); // original [0,1]
        expect(result[[0, 2, 1]], equals(3.0)); // original [1,0]
        expect(result[[0, 2, 2]], equals(4.0)); // original [1,1]
      });

      test('pads 4D tensor with constant value', () {
        // Create a simple 1x1x2x2 tensor
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 2, 2]);

        final pad = PadOp(top: 1, bottom: 1, left: 1, right: 1, value: 5.0);
        final result = pad(tensor);

        expect(result.shape, equals([1, 1, 4, 4]));

        // Check corners (should be padding with value 5.0)
        expect(result[[0, 0, 0, 0]], equals(5.0));
        expect(result[[0, 0, 1, 1]], equals(1.0)); // original data
      });
    });

    group('reflect padding', () {
      test('reflects 3D tensor boundaries', () {
        // Create a 1x2x2 tensor
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final pad = PadOp(
          top: 1,
          bottom: 1,
          left: 1,
          right: 1,
          mode: PadMode.reflect,
        );
        final result = pad(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check reflection pattern
        // Original tensor:
        // [1, 2]
        // [3, 4]
        // With reflection, corners should mirror edge values

        // Top-left corner reflects to [-1,-1] which reflects to [0,0] -> should be 1
        expect(result[[0, 0, 0]], equals(1.0));

        // Center should be original data
        expect(result[[0, 1, 1]], equals(1.0));
        expect(result[[0, 1, 2]], equals(2.0));
        expect(result[[0, 2, 1]], equals(3.0));
        expect(result[[0, 2, 2]], equals(4.0));
      });

      test('handles larger reflection', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3]); // 1x2x3

        final pad = PadOp.symmetric(2, 1, mode: PadMode.reflect);
        final result = pad(tensor);

        expect(result.shape, equals([1, 6, 5]));

        // Just verify it runs without error and produces expected shape
        expect(result.numel, equals(30));
      });
    });

    group('replicate padding', () {
      test('replicates 3D tensor edges', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final pad = PadOp(
          top: 1,
          bottom: 1,
          left: 1,
          right: 1,
          mode: PadMode.replicate,
        );
        final result = pad(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check that edges replicate
        // Top-left should copy from [0,0] = 1
        expect(result[[0, 0, 0]], equals(1.0));

        // Top-right should copy from [0,1] = 2
        expect(result[[0, 0, 3]], equals(2.0));

        // Bottom-left should copy from [1,0] = 3
        expect(result[[0, 3, 0]], equals(3.0));

        // Bottom-right should copy from [1,1] = 4
        expect(result[[0, 3, 3]], equals(4.0));

        // Center should be original data
        expect(result[[0, 1, 1]], equals(1.0));
        expect(result[[0, 2, 2]], equals(4.0));
      });
    });

    group('circular padding', () {
      test('wraps around 3D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final pad = PadOp(
          top: 1,
          bottom: 1,
          left: 1,
          right: 1,
          mode: PadMode.circular,
        );
        final result = pad(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check circular wrapping
        // Top-left should wrap to bottom-right = 4
        expect(result[[0, 0, 0]], equals(4.0));

        // Top-right should wrap to bottom-left = 3
        expect(result[[0, 0, 3]], equals(3.0));

        // Bottom-left should wrap to top-right = 2
        expect(result[[0, 3, 0]], equals(2.0));

        // Bottom-right should wrap to top-left = 1
        expect(result[[0, 3, 3]], equals(1.0));

        // Center should be original data
        expect(result[[0, 1, 1]], equals(1.0));
        expect(result[[0, 2, 2]], equals(4.0));
      });
    });

    group('edge cases', () {
      test('handles zero padding', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final pad = PadOp(
          top: 0,
          bottom: 0,
          left: 0,
          right: 0,
          mode: PadMode.constant,
          value: 9.0,
        );
        final result = pad(tensor);

        expect(result.shape, equals([1, 2, 2]));
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 1, 0]], equals(3.0));
        expect(result[[0, 1, 1]], equals(4.0));
      });

      test('handles asymmetric padding', () {
        final data = Float32List.fromList([1]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1]);

        final pad = PadOp(
          top: 3,
          bottom: 1,
          left: 2,
          right: 4,
          mode: PadMode.constant,
          value: 7.0,
        );
        final result = pad(tensor);

        expect(result.shape, equals([1, 5, 7]));

        // Center should be original
        expect(result[[0, 3, 2]], equals(1.0));

        // Corners should be padding
        expect(result[[0, 0, 0]], equals(7.0));
        expect(result[[0, 4, 6]], equals(7.0));
      });
    });

    group('different data types', () {
      test('works with different dtypes', () {
        // Test with int32
        final intData = Int32List.fromList([1, 2, 3, 4]);
        final intStorage = TensorStorage(intData, DType.int32);
        final intTensor = TensorBuffer(storage: intStorage, shape: [1, 2, 2]);

        final pad = PadOp(top: 1, bottom: 1, left: 1, right: 1, value: 0.0);
        final intResult = pad(intTensor);

        expect(intResult.dtype, equals(DType.int32));
        expect(intResult.shape, equals([1, 4, 4]));
        expect(intResult[[0, 1, 1]], equals(1.0));
      });
    });

    group('string representation', () {
      test('toString returns operation name', () {
        final pad = PadOp(
          top: 1,
          bottom: 2,
          left: 3,
          right: 4,
          mode: PadMode.reflect,
        );
        expect(
          pad.toString(),
          equals(
            'TransformOp(Pad(top=1, bottom=2, left=3, right=4, mode=PadMode.reflect))',
          ),
        );
      });
    });
  });
}
