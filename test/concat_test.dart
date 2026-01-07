import 'dart:typed_data';

import 'package:test/test.dart';

// Import all required classes directly since concat is not exported yet
import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/concat_op.dart';

void main() {
  group('concat', () {
    group('parameter validation', () {
      test('throws for empty tensor list', () {
        expect(
          () => concat([]),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws for single tensor list', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        // Should return the tensor unchanged
        final result = concat([tensor]);
        expect(result.shape, equals([3]));
        expect(result.numel, equals(3));
        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(3.0));
      });
    });

    group('axis validation', () {
      test('throws for axis out of bounds', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        expect(
          () => concat([tensor, tensor], axis: -2),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => concat([tensor, tensor], axis: 3),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('accepts valid axis', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        // For 1D tensor [3], valid axis is only 0 or -1
        expect(() => concat([tensor, tensor], axis: 0), returnsNormally);
        expect(() => concat([tensor, tensor], axis: -1), returnsNormally);
      });
    });

    group('shape compatibility validation', () {
      test('throws for different ranks', () {
        final data1d = Float32List.fromList([1, 2]);
        final tensor1d = TensorBuffer.fromFloat32List(data1d, [2]);

        final data2d = Float32List.fromList([3, 4]);
        final tensor2d = TensorBuffer.fromFloat32List(data2d, [1, 2]);

        expect(
          () => concat([tensor1d, tensor2d]),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws for incompatible shapes', () {
        final data1 = Float32List.fromList([1, 2, 3, 4]);
        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2]); // 2x2

        final data2 = Float32List.fromList([5, 6, 7, 8]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [1, 4]); // 1x4

        expect(
          () =>
              concat([tensor1, tensor2], axis: 1), // Channel dimension mismatch
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws for different dtypes', () {
        final data1 = Float32List.fromList([1, 2]);
        final tensor1 = TensorBuffer.fromFloat32List(data1, [2]);

        final data2 = Int32List.fromList([3, 4]);
        final storage2 = TensorStorage(data2, DType.int32);
        final tensor2 = TensorBuffer(storage: storage2, shape: [2]);

        expect(
          () => concat([tensor1, tensor2]),
          throwsA(isA<InvalidParameterException>()),
        );
      });
    });

    group('basic concatenation', () {
      test('concatenates 1D tensors along axis 0', () {
        final data1 = Float32List.fromList([1, 2]);
        final data2 = Float32List.fromList([3, 4]);
        final data3 = Float32List.fromList([5, 6]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2]);
        final tensor3 = TensorBuffer.fromFloat32List(data3, [2]);

        final result = concat([tensor1, tensor2, tensor3], axis: 0);

        expect(result.shape, equals([6]));
        expect(result.dtype, equals(DType.float32));

        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(3.0));
        expect(result[[3]], equals(4.0));
        expect(result[[4]], equals(5.0));
        expect(result[[5]], equals(6.0));
      });

      test('concatenates 2D tensors along axis 0', () {
        final data1 = Float32List.fromList([1, 2, 3, 4]);
        final data2 = Float32List.fromList([5, 6, 7, 8]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2, 2]);

        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.shape, equals([4, 2]));
        expect(result.dtype, equals(DType.float32));

        expect(result[[0, 0]], equals(1.0));
        expect(result[[0, 1]], equals(2.0));
        expect(result[[1, 0]], equals(3.0));
        expect(result[[1, 1]], equals(4.0));
        expect(result[[2, 0]], equals(5.0));
        expect(result[[2, 1]], equals(6.0));
        expect(result[[3, 0]], equals(7.0));
        expect(result[[3, 1]], equals(8.0));
      });

      test('concatenates 2D tensors along axis 1', () {
        final data1 = Float32List.fromList([1, 2, 3, 4]);
        final data2 = Float32List.fromList([5, 6, 7, 8]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2, 2]);

        // tensor1 = [[1, 2], [3, 4]]
        // tensor2 = [[5, 6], [7, 8]]
        // axis=1 concat -> [[1, 2, 5, 6], [3, 4, 7, 8]]
        final result = concat([tensor1, tensor2], axis: 1);

        expect(result.shape, equals([2, 4]));
        expect(result.dtype, equals(DType.float32));

        expect(result[[0, 0]], equals(1.0));
        expect(result[[0, 1]], equals(2.0));
        expect(result[[0, 2]], equals(5.0));
        expect(result[[0, 3]], equals(6.0));
        expect(result[[1, 0]], equals(3.0));
        expect(result[[1, 1]], equals(4.0));
        expect(result[[1, 2]], equals(7.0));
        expect(result[[1, 3]], equals(8.0));
      });

      test('concatenates 3D tensors', () {
        final data1 = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final data2 = Float32List.fromList([7, 8, 9, 10, 11, 12]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [1, 2, 3]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [1, 2, 3]);

        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.shape, equals([2, 2, 3]));
        expect(result.numel, equals(12));
        expect(result.dtype, equals(DType.float32));

        // Verify some key values
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 0, 2]], equals(3.0));
        expect(result[[1, 0, 0]], equals(7.0));
        expect(result[[1, 0, 1]], equals(8.0));
        expect(result[[1, 0, 2]], equals(9.0));
      });

      test('concatenates 4D tensors', () {
        // Create actual 4D tensors
        final data1 = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final data2 = Float32List.fromList([9, 10, 11, 12, 13, 14, 15, 16]);

        // shape [1, 2, 2, 2] = batch x channel x height x width
        final tensor1 = TensorBuffer.fromFloat32List(data1, [1, 2, 2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [1, 2, 2, 2]);

        // concat along batch axis (axis=0)
        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.shape, equals([2, 2, 2, 2]));
        expect(result.numel, equals(16));
        expect(result.dtype, equals(DType.float32));

        // Verify some key values
        expect(result[[0, 0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 0, 1]], equals(2.0));
        expect(result[[1, 0, 0, 0]], equals(9.0));
        expect(result[[1, 1, 1, 1]], equals(16.0));
      });
    });

    group('negative axis handling', () {
      test('handles negative axis properly', () {
        final data1 = Float32List.fromList([1, 2, 3, 4]);
        final data2 = Float32List.fromList([5, 6, 7, 8]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2, 2]);

        // axis = -1 should be same as axis = 1 (last dimension)
        // tensor1 = [[1, 2], [3, 4]]
        // tensor2 = [[5, 6], [7, 8]]
        // axis=-1 concat -> [[1, 2, 5, 6], [3, 4, 7, 8]]
        final result = concat([tensor1, tensor2], axis: -1);

        expect(result.shape, equals([2, 4]));
        expect(result[[0, 0]], equals(1.0));
        expect(result[[0, 1]], equals(2.0));
        expect(result[[0, 2]], equals(5.0));
        expect(result[[0, 3]], equals(6.0));
        expect(result[[1, 0]], equals(3.0));
        expect(result[[1, 1]], equals(4.0));
        expect(result[[1, 2]], equals(7.0));
        expect(result[[1, 3]], equals(8.0));
      });
    });

    group('multiple tensors', () {
      test('concatenates more than two tensors', () {
        final data1 = Float32List.fromList([1, 2]);
        final data2 = Float32List.fromList([3, 4]);
        final data3 = Float32List.fromList([5, 6]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2]);
        final tensor3 = TensorBuffer.fromFloat32List(data3, [2]);

        final result = concat([tensor1, tensor2, tensor3], axis: 0);

        expect(result.shape, equals([6]));
        expect(result.dtype, equals(DType.float32));

        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(3.0));
        expect(result[[3]], equals(4.0));
        expect(result[[4]], equals(5.0));
        expect(result[[5]], equals(6.0));
      });
    });

    group('different data types', () {
      test('works with int32 tensors', () {
        final data1 = Int32List.fromList([1, 2]);
        final data2 = Int32List.fromList([3, 4]);

        final storage1 = TensorStorage(data1, DType.int32);
        final storage2 = TensorStorage(data2, DType.int32);

        final tensor1 = TensorBuffer(storage: storage1, shape: [2]);
        final tensor2 = TensorBuffer(storage: storage2, shape: [2]);

        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.dtype, equals(DType.int32));
        expect(result.shape, equals([4]));
        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(3.0));
        expect(result[[3]], equals(4.0));
      });
    });

    group('edge cases', () {
      test('handles single element tensors', () {
        final data1 = Float32List.fromList([1]);
        final data2 = Float32List.fromList([2]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [1]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [1]);

        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.shape, equals([2]));
        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
      });

      test('handles large tensors', () {
        final data1 = Float32List(100)..fillRange(0, 100, 1.0);
        final data2 = Float32List(100)..fillRange(0, 100, 2.0);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [100]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [100]);

        final result = concat([tensor1, tensor2], axis: 0);

        expect(result.shape, equals([200]));
        expect(result.numel, equals(200));
        expect(result[[0]], equals(1.0));
        expect(result[[99]], equals(1.0)); // Last element of first tensor
        expect(result[[100]], equals(2.0)); // First element of second tensor
        expect(result[[199]], equals(2.0));
      });
    });

    group('comprehensive test', () {
      test('concatenates complex tensor shapes', () {
        // Create tensors with different shapes but compatible along concat axis
        // Each tensor has shape [2, 2, 2]
        final data1 = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
        final data2 = Float32List.fromList([9, 10, 11, 12, 13, 14, 15, 16]);
        final data3 = Float32List.fromList([17, 18, 19, 20, 21, 22, 23, 24]);

        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2, 2, 2]);
        final tensor3 = TensorBuffer.fromFloat32List(data3, [2, 2, 2]);

        // Concat along first dimension (axis 0)
        // Result shape: [6, 2, 2]
        final result = concat([tensor1, tensor2, tensor3], axis: 0);

        expect(result.shape, equals([6, 2, 2]));
        expect(result.numel, equals(24));

        // tensor1 layout in [2,2,2]:
        // [0,0,0]=1, [0,0,1]=2, [0,1,0]=3, [0,1,1]=4
        // [1,0,0]=5, [1,0,1]=6, [1,1,0]=7, [1,1,1]=8
        // After concat, tensor1 occupies result[0:2,:,:]
        expect(result[[0, 0, 0]], equals(1.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 1, 0]], equals(3.0));
        expect(result[[0, 1, 1]], equals(4.0));
        expect(result[[1, 0, 0]], equals(5.0));
        expect(result[[1, 1, 1]], equals(8.0));

        // tensor2 occupies result[2:4,:,:]
        expect(result[[2, 0, 0]], equals(9.0));
        expect(result[[3, 1, 1]], equals(16.0));

        // tensor3 occupies result[4:6,:,:]
        expect(result[[4, 0, 0]], equals(17.0));
        expect(result[[5, 1, 1]], equals(24.0));
      });
    });
  });
}
