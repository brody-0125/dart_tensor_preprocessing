import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('TileOp', () {
    test('tiles 1D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );

      final result = TileOp(reps: [3])(tensor);

      expect(result.shape, equals([9]));
      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(3.0, 1e-6));
      expect(result[[3]], closeTo(1.0, 1e-6));
      expect(result[[6]], closeTo(1.0, 1e-6));
    });

    test('tiles 2D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );

      final result = TileOp(reps: [2, 3])(tensor);

      expect(result.shape, equals([4, 6]));
      // Row 0: [1,2,1,2,1,2]
      expect(result[[0, 0]], closeTo(1.0, 1e-6));
      expect(result[[0, 2]], closeTo(1.0, 1e-6));
      expect(result[[0, 4]], closeTo(1.0, 1e-6));
      // Row 2 (tiled from row 0): same pattern
      expect(result[[2, 0]], closeTo(1.0, 1e-6));
    });

    test('tiles with rep=1 returns copy', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0]),
        [2],
      );

      final result = TileOp(reps: [1])(tensor);

      expect(result.shape, equals([2]));
      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
    });

    test('computeOutputShape returns correct shape', () {
      final op = TileOp(reps: [2, 3]);
      expect(op.computeOutputShape([4, 5]), equals([8, 15]));
    });

    test('throws for reps length mismatch', () {
      final tensor = TensorBuffer.zeros([2, 3]);

      expect(
        () => TileOp(reps: [2, 3, 4])(tensor),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('does not modify input', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0]),
        [2],
      );

      TileOp(reps: [3])(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[1]], closeTo(2.0, 1e-6));
    });

    test('works with float64', () {
      final tensor = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 2.0]),
        [2],
      );

      final result = TileOp(reps: [2])(tensor);

      expect(result[[0]], closeTo(1.0, 1e-10));
      expect(result[[2]], closeTo(1.0, 1e-10));
    });

    test('name is Tile', () {
      expect(TileOp(reps: [2]).name, equals('Tile'));
    });

    test('capabilities show shape changes', () {
      final caps = TileOp(reps: [2]).capabilities;
      expect(caps.preservesShape, isFalse);
      expect(caps.pytorchEquivalent, equals('Tensor.repeat'));
      expect(caps.onnxOpType, equals('Tile'));
    });
  });

  group('RepeatOp', () {
    test('repeats 1D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0]),
        [2],
      );

      final result = RepeatOp(sizes: [3])(tensor);

      expect(result.shape, equals([6]));
      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(1.0, 1e-6));
    });

    test('repeats 2D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );

      final result = RepeatOp(sizes: [2, 3])(tensor);

      expect(result.shape, equals([4, 6]));
    });

    test('name is Repeat', () {
      expect(RepeatOp(sizes: [2]).name, equals('Repeat'));
    });

    test('computeOutputShape returns correct shape', () {
      final op = RepeatOp(sizes: [2, 3]);
      expect(op.computeOutputShape([4, 5]), equals([8, 15]));
    });

    test('works with float64', () {
      final tensor = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 2.0]),
        [2],
      );

      final result = RepeatOp(sizes: [3])(tensor);

      expect(result.shape, equals([6]));
      expect(result[[0]], closeTo(1.0, 1e-10));
      expect(result[[2]], closeTo(1.0, 1e-10));
      expect(result[[5]], closeTo(2.0, 1e-10));
    });

    test('throws for sizes length mismatch', () {
      final tensor = TensorBuffer.zeros([2, 3]);

      expect(
        () => RepeatOp(sizes: [2, 3, 4])(tensor),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('does not modify input', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0]),
        [2],
      );

      RepeatOp(sizes: [3])(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[1]], closeTo(2.0, 1e-6));
    });

    test('capabilities show shape changes', () {
      final caps = RepeatOp(sizes: [2]).capabilities;
      expect(caps.preservesShape, isFalse);
      expect(caps.pytorchEquivalent, equals('Tensor.repeat'));
      expect(caps.onnxOpType, equals('Tile'));
    });
  });

  group('RollOp', () {
    test('rolls 1D tensor forward', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [4],
      );

      final result = RollOp(shifts: [1], dims: [0])(tensor);

      expect(result.shape, equals([4]));
      // [4, 1, 2, 3] — shift by 1 means each element moves forward
      expect(result[[0]], closeTo(4.0, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(2.0, 1e-6));
      expect(result[[3]], closeTo(3.0, 1e-6));
    });

    test('rolls 1D tensor backward', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [4],
      );

      final result = RollOp(shifts: [-1], dims: [0])(tensor);

      // [2, 3, 4, 1]
      expect(result[[0]], closeTo(2.0, 1e-6));
      expect(result[[1]], closeTo(3.0, 1e-6));
      expect(result[[2]], closeTo(4.0, 1e-6));
      expect(result[[3]], closeTo(1.0, 1e-6));
    });

    test('rolls 2D tensor along dim=0', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([
          1.0, 2.0,
          3.0, 4.0,
          5.0, 6.0,
        ]),
        [3, 2],
      );

      final result = RollOp(shifts: [1], dims: [0])(tensor);

      expect(result.shape, equals([3, 2]));
      // Row 0 should be old row 2
      expect(result[[0, 0]], closeTo(5.0, 1e-6));
      expect(result[[0, 1]], closeTo(6.0, 1e-6));
      // Row 1 should be old row 0
      expect(result[[1, 0]], closeTo(1.0, 1e-6));
    });

    test('rolls 2D tensor along dim=1', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
        ]),
        [2, 3],
      );

      final result = RollOp(shifts: [1], dims: [1])(tensor);

      // Each row shifted right by 1
      expect(result[[0, 0]], closeTo(3.0, 1e-6));
      expect(result[[0, 1]], closeTo(1.0, 1e-6));
      expect(result[[0, 2]], closeTo(2.0, 1e-6));
    });

    test('roll with shift=0 returns copy', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );

      final result = RollOp(shifts: [0], dims: [0])(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(3.0, 1e-6));
    });

    test('roll without dims flattens', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );

      // Without dims, treat as flat
      final result = RollOp(shifts: [1])(tensor);

      expect(result.shape, equals([2, 2]));
      // Flat: [1,2,3,4] -> roll 1 -> [4,1,2,3]
      expect(result[[0, 0]], closeTo(4.0, 1e-6));
      expect(result[[0, 1]], closeTo(1.0, 1e-6));
      expect(result[[1, 0]], closeTo(2.0, 1e-6));
      expect(result[[1, 1]], closeTo(3.0, 1e-6));
    });

    test('does not modify input', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );

      RollOp(shifts: [1], dims: [0])(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[2]], closeTo(3.0, 1e-6));
    });

    test('works with float64', () {
      final tensor = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 2.0, 3.0]),
        [3],
      );

      final result = RollOp(shifts: [1], dims: [0])(tensor);

      expect(result[[0]], closeTo(3.0, 1e-10));
      expect(result[[1]], closeTo(1.0, 1e-10));
    });

    test('name is Roll', () {
      expect(RollOp(shifts: [1]).name, equals('Roll'));
    });

    test('preserves shape', () {
      final op = RollOp(shifts: [1], dims: [0]);
      expect(op.computeOutputShape([2, 3]), equals([2, 3]));
    });

    test('capabilities include pytorch equivalent', () {
      final caps = RollOp(shifts: [1]).capabilities;
      expect(caps.preservesShape, isTrue);
      expect(caps.pytorchEquivalent, equals('torch.roll'));
    });

    test('rolls 2D tensor along multiple dims simultaneously', () {
      // [[1, 2, 3],
      //  [4, 5, 6]]
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [2, 3],
      );

      // shifts: [1, 1], dims: [0, 1]
      // Roll dim=0 by 1: row swap -> [[4,5,6],[1,2,3]]
      // Roll dim=1 by 1: each row shift right -> [[6,4,5],[3,1,2]]
      final result = RollOp(shifts: [1, 1], dims: [0, 1])(tensor);

      expect(result.shape, equals([2, 3]));
      expect(result[[0, 0]], closeTo(6.0, 1e-6));
      expect(result[[0, 1]], closeTo(4.0, 1e-6));
      expect(result[[0, 2]], closeTo(5.0, 1e-6));
      expect(result[[1, 0]], closeTo(3.0, 1e-6));
      expect(result[[1, 1]], closeTo(1.0, 1e-6));
      expect(result[[1, 2]], closeTo(2.0, 1e-6));
    });

    test('throws for shifts/dims length mismatch', () {
      final tensor = TensorBuffer.zeros([2, 3]);

      expect(
        () => RollOp(shifts: [1, 2], dims: [0])(tensor),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for out-of-bounds dim', () {
      final tensor = TensorBuffer.zeros([2, 3]);

      expect(
        () => RollOp(shifts: [1], dims: [5])(tensor),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });
  });
}
