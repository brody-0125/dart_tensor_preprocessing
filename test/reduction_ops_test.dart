// Reduction operations tests
//
// Tests cover:
// - sum(), mean(), min(), max() for various tensor shapes
// - Different data types (float32, int32, uint8)
// - Non-contiguous tensors (transposed)
// - Edge cases (single element, large tensors)
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('Reduction Operations', () {
    group('sum', () {
      test('1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5]),
          [5],
        );
        expect(tensor.sum(), equals(15.0));
      });

      test('2D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        expect(tensor.sum(), equals(21.0));
      });

      test('3D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
          [2, 2, 2],
        );
        expect(tensor.sum(), equals(36.0));
      });

      test('single element', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([42]),
          [1],
        );
        expect(tensor.sum(), equals(42.0));
      });

      test('zeros tensor', () {
        final tensor = TensorBuffer.zeros([3, 3]);
        expect(tensor.sum(), equals(0.0));
      });

      test('ones tensor', () {
        final tensor = TensorBuffer.ones([2, 3, 4]);
        expect(tensor.sum(), equals(24.0));
      });

      test('non-contiguous (transposed) tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        final transposed = tensor.transpose([1, 0]);
        expect(transposed.sum(), equals(21.0));
      });

      test('uint8 tensor', () {
        final tensor = TensorBuffer.fromUint8List(
          Uint8List.fromList([10, 20, 30, 40]),
          [4],
        );
        expect(tensor.sum(), equals(100.0));
      });
    });

    group('mean', () {
      test('1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5]),
          [5],
        );
        expect(tensor.mean(), equals(3.0));
      });

      test('2D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        expect(tensor.mean(), equals(3.5));
      });

      test('single element', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([42]),
          [1],
        );
        expect(tensor.mean(), equals(42.0));
      });

      test('non-contiguous (transposed) tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        final transposed = tensor.transpose([1, 0]);
        expect(transposed.mean(), equals(3.5));
      });
    });

    group('min', () {
      test('1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9, 2, 6]),
          [8],
        );
        expect(tensor.min(), equals(1.0));
      });

      test('2D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [2, 3],
        );
        expect(tensor.min(), equals(1.0));
      });

      test('single element', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([42]),
          [1],
        );
        expect(tensor.min(), equals(42.0));
      });

      test('negative values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, -5, 3, -2, 4]),
          [5],
        );
        expect(tensor.min(), equals(-5.0));
      });

      test('non-contiguous (transposed) tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [2, 3],
        );
        final transposed = tensor.transpose([1, 0]);
        expect(transposed.min(), equals(1.0));
      });

      test('uint8 tensor', () {
        final tensor = TensorBuffer.fromUint8List(
          Uint8List.fromList([50, 10, 30, 20]),
          [4],
        );
        expect(tensor.min(), equals(10.0));
      });
    });

    group('max', () {
      test('1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9, 2, 6]),
          [8],
        );
        expect(tensor.max(), equals(9.0));
      });

      test('2D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [2, 3],
        );
        expect(tensor.max(), equals(9.0));
      });

      test('single element', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([42]),
          [1],
        );
        expect(tensor.max(), equals(42.0));
      });

      test('negative values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([-1, -5, -3, -2, -4]),
          [5],
        );
        expect(tensor.max(), equals(-1.0));
      });

      test('non-contiguous (transposed) tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [2, 3],
        );
        final transposed = tensor.transpose([1, 0]);
        expect(transposed.max(), equals(9.0));
      });

      test('uint8 tensor', () {
        final tensor = TensorBuffer.fromUint8List(
          Uint8List.fromList([50, 10, 30, 200]),
          [4],
        );
        expect(tensor.max(), equals(200.0));
      });
    });

    group('sumAxis', () {
      test('2D tensor along axis 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        // [[1, 2, 3], [4, 5, 6]] -> [5, 7, 9]
        final result = tensor.sumAxis(0);
        expect(result.shape, equals([3]));
        expect(result.toList(), equals([5.0, 7.0, 9.0]));
      });

      test('2D tensor along axis 1', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
        final result = tensor.sumAxis(1);
        expect(result.shape, equals([2]));
        expect(result.toList(), equals([6.0, 15.0]));
      });

      test('3D tensor along axis 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
          [2, 2, 2],
        );
        // [[[1,2],[3,4]], [[5,6],[7,8]]] -> [[6,8],[10,12]]
        final result = tensor.sumAxis(0);
        expect(result.shape, equals([2, 2]));
        expect(result.toList(), equals([6.0, 8.0, 10.0, 12.0]));
      });

      test('3D tensor along axis 1', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
          [2, 2, 2],
        );
        // [[[1,2],[3,4]], [[5,6],[7,8]]] -> [[4,6],[12,14]]
        final result = tensor.sumAxis(1);
        expect(result.shape, equals([2, 2]));
        expect(result.toList(), equals([4.0, 6.0, 12.0, 14.0]));
      });

      test('3D tensor along axis 2', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
          [2, 2, 2],
        );
        // [[[1,2],[3,4]], [[5,6],[7,8]]] -> [[3,7],[11,15]]
        final result = tensor.sumAxis(2);
        expect(result.shape, equals([2, 2]));
        expect(result.toList(), equals([3.0, 7.0, 11.0, 15.0]));
      });

      test('keepDims=true preserves dimension', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        final result = tensor.sumAxis(0, keepDims: true);
        expect(result.shape, equals([1, 3]));
        expect(result.toList(), equals([5.0, 7.0, 9.0]));
      });

      test('negative axis', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        // axis=-1 is same as axis=1
        final result = tensor.sumAxis(-1);
        expect(result.shape, equals([2]));
        expect(result.toList(), equals([6.0, 15.0]));
      });

      test('1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5]),
          [5],
        );
        final result = tensor.sumAxis(0);
        expect(result.shape, equals([1]));
        expect(result.toList(), equals([15.0]));
      });

      test('non-contiguous (transposed) tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        final transposed = tensor.transpose([1, 0]); // shape [3, 2]
        // [[1, 4], [2, 5], [3, 6]] -> [5, 7, 9]
        final result = transposed.sumAxis(1);
        expect(result.shape, equals([3]));
        expect(result.toList(), equals([5.0, 7.0, 9.0]));
      });
    });

    group('meanAxis', () {
      test('2D tensor along axis 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        // [[1, 2, 3], [4, 5, 6]] -> [2.5, 3.5, 4.5]
        final result = tensor.meanAxis(0);
        expect(result.shape, equals([3]));
        expect(result.toList(), equals([2.5, 3.5, 4.5]));
      });

      test('2D tensor along axis 1', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        // [[1, 2, 3], [4, 5, 6]] -> [2.0, 5.0]
        final result = tensor.meanAxis(1);
        expect(result.shape, equals([2]));
        expect(result.toList(), equals([2.0, 5.0]));
      });

      test('keepDims=true', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6]),
          [2, 3],
        );
        final result = tensor.meanAxis(1, keepDims: true);
        expect(result.shape, equals([2, 1]));
        expect(result.toList(), equals([2.0, 5.0]));
      });
    });

    group('minAxis', () {
      test('2D tensor along axis 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        // [[3, 1, 4], [1, 5, 9]] -> [1, 1, 4]
        final result = tensor.minAxis(0);
        expect(result.shape, equals([3]));
        expect(result.toList(), equals([1.0, 1.0, 4.0]));
      });

      test('2D tensor along axis 1', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        // [[3, 1, 4], [1, 5, 9]] -> [1, 1]
        final result = tensor.minAxis(1);
        expect(result.shape, equals([2]));
        expect(result.toList(), equals([1.0, 1.0]));
      });

      test('with negative values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, -2, 3, -4, 5, -6]),
          [2, 3],
        );
        final result = tensor.minAxis(0);
        expect(result.toList(), equals([-4.0, -2.0, -6.0]));
      });

      test('keepDims=true', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        final result = tensor.minAxis(0, keepDims: true);
        expect(result.shape, equals([1, 3]));
      });
    });

    group('maxAxis', () {
      test('2D tensor along axis 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        // [[3, 1, 4], [1, 5, 9]] -> [3, 5, 9]
        final result = tensor.maxAxis(0);
        expect(result.shape, equals([3]));
        expect(result.toList(), equals([3.0, 5.0, 9.0]));
      });

      test('2D tensor along axis 1', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        // [[3, 1, 4], [1, 5, 9]] -> [4, 9]
        final result = tensor.maxAxis(1);
        expect(result.shape, equals([2]));
        expect(result.toList(), equals([4.0, 9.0]));
      });

      test('with negative values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([-1, -2, -3, -4, -5, -6]),
          [2, 3],
        );
        // [[-1, -2, -3], [-4, -5, -6]] -> [-1, -2, -3]
        final result = tensor.maxAxis(0);
        expect(result.toList(), equals([-1.0, -2.0, -3.0]));
      });

      test('keepDims=true', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9]),
          [2, 3],
        );
        final result = tensor.maxAxis(1, keepDims: true);
        expect(result.shape, equals([2, 1]));
        expect(result.toList(), equals([4.0, 9.0]));
      });
    });

    group('axis reduction edge cases', () {
      test('invalid axis throws RangeError', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4]),
          [2, 2],
        );
        expect(() => tensor.sumAxis(2), throwsRangeError);
        expect(() => tensor.sumAxis(-3), throwsRangeError);
      });

      test('4D tensor reduction', () {
        // Shape: [2, 2, 2, 2]
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList(List.generate(16, (i) => (i + 1).toDouble())),
          [2, 2, 2, 2],
        );

        // Sum along axis 0
        final result0 = tensor.sumAxis(0);
        expect(result0.shape, equals([2, 2, 2]));

        // Sum along last axis
        final result3 = tensor.sumAxis(3);
        expect(result3.shape, equals([2, 2, 2]));
      });
    });

    group('combined operations', () {
      test('sum, mean, min, max on same tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
          [2, 5],
        );

        expect(tensor.sum(), equals(55.0));
        expect(tensor.mean(), equals(5.5));
        expect(tensor.min(), equals(1.0));
        expect(tensor.max(), equals(10.0));
      });

      test('operations on squeezed tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [1, 3, 1],
        );
        final squeezed = tensor.squeeze();

        expect(squeezed.shape, equals([3]));
        expect(squeezed.sum(), equals(6.0));
        expect(squeezed.mean(), equals(2.0));
        expect(squeezed.min(), equals(1.0));
        expect(squeezed.max(), equals(3.0));
      });

      test('operations on unsqueezed tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        final unsqueezed = tensor.unsqueeze(0);

        expect(unsqueezed.shape, equals([1, 3]));
        expect(unsqueezed.sum(), equals(6.0));
        expect(unsqueezed.mean(), equals(2.0));
        expect(unsqueezed.min(), equals(1.0));
        expect(unsqueezed.max(), equals(3.0));
      });
    });
  });
}
