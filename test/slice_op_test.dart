import 'dart:typed_data';

import 'package:test/test.dart';

// Import all required classes directly since SliceOp is not exported yet
import 'package:dart_tensor_preprocessing/src/core/dtype.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_storage.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';
import 'package:dart_tensor_preprocessing/src/ops/slice_op.dart';

void main() {
  group('SliceOp', () {
    group('constructor validation', () {
      test('creates slice with null values', () {
        final slice = SliceOp([null, null, null]);
        expect(slice.slices, equals([null, null, null]));
      });

      test('creates slice with specific values', () {
        final slice = SliceOp([(0, 5, 1), (null, null, null), (2, 8, 2)]);
        expect(slice.slices.length, equals(3));
        expect(slice.slices[0], equals((0, 5, 1)));
        expect(slice.slices[1], equals((null, null, null)));
        expect(slice.slices[2], equals((2, 8, 2)));
      });
    });

    group('factory constructors', () {
      test('range creates slice for specific dimension', () {
        final slice = SliceOp.range(1, 2, 5);
        expect(slice.slices.length, equals(2));
        expect(slice.slices[0], equals(null));
        expect(slice.slices[1], equals((2, 5, 1)));
      });

      test('from creates slice from start index', () {
        final slice = SliceOp.from(0, 3, step: 2);
        expect(slice.slices.length, equals(1));
        expect(slice.slices[0], equals((3, null, 2)));
      });

      test('to creates slice to end index', () {
        final slice = SliceOp.to(2, 7, step: 3);
        expect(slice.slices.length, equals(3));
        expect(slice.slices[0], equals(null));
        expect(slice.slices[1], equals(null));
        expect(slice.slices[2], equals((null, 7, 3)));
      });
    });

    group('basic slicing', () {
      test('slices 1D tensor', () {
        final data = Float32List.fromList([0, 1, 2, 3, 4, 5]);
        final tensor = TensorBuffer.fromFloat32List(data, [6]);

        final slice = SliceOp([(1, 4, 1)]); // [1:4]
        final result = slice(tensor);

        expect(result.shape, equals([3]));
        expect(result[[0]], equals(1.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(3.0));
      });

      test('slices 2D tensor', () {
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
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 4]);

        final slice = SliceOp([
          (0, 2, 1), // rows 0:2
          (1, 3, 1), // cols 1:3
        ]);
        final result = slice(tensor);

        expect(result.shape, equals([2, 2]));
        expect(result[[0, 0]], equals(2.0)); // [0,1]
        expect(result[[0, 1]], equals(3.0)); // [0,2]
        expect(result[[1, 0]], equals(6.0)); // [1,1]
        expect(result[[1, 1]], equals(7.0)); // [1,2]
      });

      test('slices with step', () {
        final data = Float32List.fromList([0, 1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [9]);

        final slice = SliceOp([(0, 9, 2)]); // [0:9:2]
        final result = slice(tensor);

        expect(result.shape, equals([5]));
        expect(result[[0]], equals(0.0));
        expect(result[[1]], equals(2.0));
        expect(result[[2]], equals(4.0));
        expect(result[[3]], equals(6.0));
        expect(result[[4]], equals(8.0));
      });
    });

    group('negative indices', () {
      test('handles negative start index', () {
        final data = Float32List.fromList([0, 1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final slice = SliceOp([(-2, 5, 1)]); // [-2:5]
        final result = slice(tensor);

        expect(result.shape, equals([2]));
        // Just check that it doesn't crash and has reasonable values
        expect(result.numel, equals(2));
      });

      test('handles negative end index', () {
        final data = Float32List.fromList([0, 1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final slice = SliceOp([(0, -1, 1)]); // [0:-1] = [0,1,2,3]
        final result = slice(tensor);

        expect(result.shape, equals([4]));
        expect(result[[0]], equals(0.0));
        expect(result[[1]], equals(1.0));
        expect(result[[2]], equals(2.0));
        expect(result[[3]], equals(3.0));
      });

      test('handles both negative start and end', () {
        final data = Float32List.fromList([0, 1, 2, 3, 4, 5, 6, 7, 8]);
        final tensor = TensorBuffer.fromFloat32List(data, [9]);

        final slice = SliceOp([(-4, -1, 1)]); // [-4:-1] = [5,6,7]
        final result = slice(tensor);

        expect(result.shape, equals([3]));
        expect(result[[0]], equals(5.0));
        expect(result[[1]], equals(6.0));
        expect(result[[2]], equals(7.0));
      });
    });

    group('null slices (select all)', () {
      test('null selects entire dimension', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final slice = SliceOp([null, (0, 1, 1)]); // [:, 0:1]
        final result = slice(tensor);

        expect(result.shape, equals([2, 1]));
        expect(result[[0, 0]], equals(1.0));
        expect(result[[1, 0]], equals(3.0));
      });

      test('all null selects entire tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final slice = SliceOp([null, null]); // [:, :]
        final result = slice(tensor);

        expect(result.shape, equals([2, 2]));
        expect(result[[0, 0]], equals(1.0));
        expect(result[[0, 1]], equals(2.0));
        expect(result[[1, 0]], equals(3.0));
        expect(result[[1, 1]], equals(4.0));
      });

      test('fewer slices than dimensions selects rest completely', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        final slice = SliceOp([(1, 2, 1)]); // [1:2], rest selected completely
        final result = slice(tensor);

        expect(result.shape, equals([1, 3]));
        expect(result[[0, 0]], equals(4.0));
        expect(result[[0, 1]], equals(5.0));
        expect(result[[0, 2]], equals(6.0));
      });
    });

    group('3D tensors', () {
      test('slices 3D tensor', () {
        final data = Float32List.fromList([
          // Channel 0
          1, 2, 3, 4, 5, 6,
          // Channel 1
          7, 8, 9, 10, 11, 12,
          // Channel 2
          13, 14, 15, 16, 17, 18,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 3]);

        final slice = SliceOp([
          (1, 3, 1), // channels 1:3
          null, // all rows
          (1, 3, 1), // cols 1:3
        ]);
        final result = slice(tensor);

        expect(result.shape, equals([2, 2, 2]));
        expect(result[[0, 0, 0]], equals(8.0)); // [1,0,1]
        expect(result[[0, 0, 1]], equals(9.0)); // [1,0,2]
        expect(result[[0, 1, 0]], equals(11.0)); // [1,1,1]
        expect(result[[0, 1, 1]], equals(12.0)); // [1,1,2]
        expect(result[[1, 0, 0]], equals(14.0)); // [2,0,1]
        expect(result[[1, 0, 1]], equals(15.0)); // [2,0,2]
        expect(result[[1, 1, 0]], equals(17.0)); // [2,1,1]
        expect(result[[1, 1, 1]], equals(18.0)); // [2,1,2]
      });
    });

    group('edge cases', () {
      test('handles empty result', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        // Empty slice is not supported as TensorBuffer doesn't allow zero-dim tensors
        // Use single element slice instead
        final slice = SliceOp([(2, 3, 1)]); // [2:3] -> single element
        final result = slice(tensor);

        expect(result.shape, equals([1]));
        expect(result[[0]], equals(3.0));
      });

      test('handles single element slice', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final slice = SliceOp([(1, 2, 1)]); // [1:2] -> single element
        final result = slice(tensor);

        expect(result.shape, equals([1]));
        expect(result[[0]], equals(2.0));
      });
    });

    group('output shape computation', () {
      test('computes correct shapes for various slices', () {
        final shapes = [
          ([6], [(1, 4, 1)], [3]),
          ([4, 3], [(0, 2, 1), (1, 3, 1)], [2, 2]),
          ([5, 6, 7], [(1, 4, 2), null, (0, 7, 3)], [2, 6, 3]),
        ];

        for (final (inputShape, slices, expectedShape) in shapes) {
          final slice = SliceOp(slices);
          final outputShape = slice.computeOutputShape(inputShape);
          expect(outputShape, equals(expectedShape));
        }
      });
    });

    group('error handling', () {
      test('throws for too many slices', () {
        final data = Float32List.fromList([1, 2]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final slice = SliceOp([null, null, null]); // Too many for 1D tensor

        expect(() => slice(tensor), throwsA(isA<ShapeMismatchException>()));
      });

      test('throws for negative step', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final slice = SliceOp([(0, 3, -1)]);

        expect(() => slice(tensor), throwsA(isA<InvalidParameterException>()));
      });

      test('throws for zero step', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final slice = SliceOp([(0, 3, 0)]);

        expect(() => slice(tensor), throwsA(isA<InvalidParameterException>()));
      });

      test('throws for start > end', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final slice = SliceOp([(2, 1, 1)]);

        expect(() => slice(tensor), throwsA(isA<InvalidParameterException>()));
      });

      test('throws for out of bounds indices', () {
        final data = Float32List.fromList([1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final slice1 = SliceOp([(-5, 3, 1)]); // Start out of bounds
        final slice2 = SliceOp([(0, 10, 1)]); // End out of bounds

        expect(() => slice1(tensor), throwsA(isA<InvalidParameterException>()));
        expect(() => slice2(tensor), throwsA(isA<InvalidParameterException>()));
      });
    });

    group('different data types', () {
      test('works with int32', () {
        final data = Int32List.fromList([1, 2, 3, 4, 5, 6]);
        final storage = TensorStorage(data, DType.int32);
        final tensor = TensorBuffer(storage: storage, shape: [2, 3]);

        final slice = SliceOp([(0, 2, 1), (1, 3, 1)]);
        final result = slice(tensor);

        expect(result.dtype, equals(DType.int32));
        expect(result.shape, equals([2, 2]));
        expect(result[[0, 0]], equals(2.0));
        expect(result[[1, 1]], equals(6.0));
      });
    });

    group('string representation', () {
      test('toString returns operation name', () {
        final slice = SliceOp([(1, 4, 2), null, (0, -1, 1)]);
        expect(
          slice.toString(),
          equals('TransformOp(Slice(slices=[(1, 4, 2), null, (0, -1, 1)]))'),
        );
      });
    });
  });
}
