import 'dart:typed_data';

import 'package:test/test.dart';

// Import all required classes directly since ClipOp is not exported yet
import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/clip_op.dart';

void main() {
  group('ClipOp', () {
    group('constructor validation', () {
      test('throws when min >= max', () {
        expect(
          () => ClipOp(min: 1.0, max: 1.0),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => ClipOp(min: 2.0, max: 1.0),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts valid min < max', () {
        expect(() => ClipOp(min: -1.0, max: 1.0), returnsNormally);
        expect(() => ClipOp(min: 0.0, max: 0.1), returnsNormally);
      });
    });

    group('factory constructors', () {
      test('unit() creates [0,1] range', () {
        final clip = ClipOp.unit();
        expect(clip.min, equals(0.0));
        expect(clip.max, equals(1.0));
      });

      test('symmetric() creates [-1,1] range', () {
        final clip = ClipOp.symmetric();
        expect(clip.min, equals(-1.0));
        expect(clip.max, equals(1.0));
      });

      test('uint8() creates [0,255] range', () {
        final clip = ClipOp.uint8();
        expect(clip.min, equals(0.0));
        expect(clip.max, equals(255.0));
      });
    });

    group('basic clipping', () {
      test('clips values below min', () {
        final data = Float32List.fromList([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        final clip = ClipOp(min: -1.0, max: 1.0);
        final result = clip(tensor);

        expect(result[[0, 0]], equals(-1.0)); // -2.0 clipped to -1.0
        expect(result[[0, 1]], equals(-1.0)); // -1.0 unchanged
        expect(result[[0, 2]], equals(0.0)); // 0.0 unchanged
        expect(result[[1, 0]], equals(0.5)); // 0.5 unchanged
        expect(result[[1, 1]], equals(1.0)); // 1.0 unchanged
        expect(result[[1, 2]], equals(1.0)); // 2.0 clipped to 1.0
      });

      test('clips values above max', () {
        final data = Float32List.fromList([0.0, 0.5, 1.0, 1.5, 2.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final clip = ClipOp(min: 0.0, max: 1.0);
        final result = clip(tensor);

        expect(result[[0]], equals(0.0));
        expect(result[[1]], equals(0.5));
        expect(result[[2]], equals(1.0));
        expect(result[[3]], equals(1.0)); // 1.5 clipped to 1.0
        expect(result[[4]], equals(1.0)); // 2.0 clipped to 1.0
      });

      test('preserves values within range', () {
        final data = Float32List.fromList([0.2, 0.4, 0.6, 0.8]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final clip = ClipOp(min: 0.0, max: 1.0);
        final result = clip(tensor);

        expect(result[[0, 0]], closeTo(0.2, 1e-6));
        expect(result[[0, 1]], closeTo(0.4, 1e-6));
        expect(result[[1, 0]], closeTo(0.6, 1e-6));
        expect(result[[1, 1]], closeTo(0.8, 1e-6));
      });
    });

    group('edge cases', () {
      test('handles single element', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([-5.0]),
          [1],
        );
        final clip = ClipOp(min: -1.0, max: 1.0);
        final result = clip(tensor);

        expect(result[[0]], equals(-1.0));
      });

      test('handles very small range', () {
        final data = Float32List.fromList([0.999, 1.0, 1.001]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);
        final clip = ClipOp(min: 0.999, max: 1.001);
        final result = clip(tensor);

        expect(result[[0]], closeTo(0.999, 1e-6));
        expect(result[[1]], closeTo(1.0, 1e-6));
        expect(result[[2]], closeTo(1.001, 1e-6));
      });
    });

    group('in-place operation', () {
      test('applyInPlace modifies tensor in place', () {
        final data = Float32List.fromList([-2.0, 0.0, 2.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final clip = ClipOp(min: -1.0, max: 1.0);
        clip.applyInPlace(tensor);

        expect(tensor[[0]], equals(-1.0)); // -2.0 clipped to -1.0
        expect(tensor[[1]], equals(0.0)); // 0.0 unchanged
        expect(tensor[[2]], equals(1.0)); // 2.0 clipped to 1.0
      });

      test('applyInPlace throws for non-contiguous tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);
        final transposed = tensor.transpose([1, 0]); // Non-contiguous

        final clip = ClipOp(min: 0.0, max: 1.0);

        expect(
          () => clip.applyInPlace(transposed),
          throwsA(isA<NonContiguousException>()),
        );
      });
    });

    group('different data types', () {
      test('works with float32', () {
        final data = Float32List.fromList([-1.5, 0.0, 1.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final clip = ClipOp(min: -1.0, max: 1.0);
        final result = clip(tensor);

        expect(result.dtype, equals(DType.float32));
        expect(result[[0]], equals(-1.0));
        expect(result[[1]], equals(0.0));
        expect(result[[2]], equals(1.0));
      });

      test('works with int64', () {
        // Create int64 tensor manually since no fromInt64List method exists
        final storage = TensorStorage(
          Int64List.fromList([-5, 0, 5]),
          DType.int64,
        );
        final tensor = TensorBuffer(storage: storage, shape: [3]);

        final clip = ClipOp(min: -2.0, max: 2.0);
        final result = clip(tensor);

        expect(result.dtype, equals(DType.int64));
        expect(result[[0]], equals(-2.0));
        expect(result[[1]], equals(0.0));
        expect(result[[2]], equals(2.0));
      });
    });

    group('factory presets', () {
      test('unit() clips to [0,1]', () {
        final data = Float32List.fromList([-0.5, 0.0, 0.5, 1.0, 1.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final clip = ClipOp.unit();
        final result = clip(tensor);

        expect(result[[0]], equals(0.0)); // -0.5 clipped to 0.0
        expect(result[[1]], equals(0.0)); // 0.0 unchanged
        expect(result[[2]], equals(0.5)); // 0.5 unchanged
        expect(result[[3]], equals(1.0)); // 1.0 unchanged
        expect(result[[4]], equals(1.0)); // 1.5 clipped to 1.0
      });

      test('symmetric() clips to [-1,1]', () {
        final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final clip = ClipOp.symmetric();
        final result = clip(tensor);

        expect(result[[0]], equals(-1.0)); // -2.0 clipped to -1.0
        expect(result[[1]], equals(-1.0)); // -1.0 unchanged
        expect(result[[2]], equals(0.0)); // 0.0 unchanged
        expect(result[[3]], equals(1.0)); // 1.0 unchanged
        expect(result[[4]], equals(1.0)); // 2.0 clipped to 1.0
      });

      test('uint8() clips to [0,255]', () {
        final data = Float32List.fromList([-10, 0, 100, 255, 300]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final clip = ClipOp.uint8();
        final result = clip(tensor);

        expect(result[[0]], equals(0.0)); // -10 clipped to 0.0
        expect(result[[1]], equals(0.0)); // 0 unchanged
        expect(result[[2]], equals(100.0)); // 100 unchanged
        expect(result[[3]], equals(255.0)); // 255 unchanged
        expect(result[[4]], equals(255.0)); // 300 clipped to 255.0
      });
    });

    group('output shape computation', () {
      test('preserves input shape', () {
        final shapes = [
          [3],
          [2, 3],
          [1, 3, 224, 224],
          [4, 1, 1],
        ];

        final clip = ClipOp(min: 0.0, max: 1.0);

        for (final shape in shapes) {
          expect(clip.computeOutputShape(shape), equals(shape));
        }
      });
    });

    group('string representation', () {
      test('toString returns operation name', () {
        final clip = ClipOp(min: -1.0, max: 2.0);
        expect(clip.toString(), equals('TransformOp(Clip(min=-1.0, max=2.0))'));
      });
    });
  });
}
