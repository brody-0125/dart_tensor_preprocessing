import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

/// Helper to create an int32 index tensor.
TensorBuffer intTensor(List<int> values, List<int> shape) {
  return TensorBuffer(
    storage: TensorStorage(Int32List.fromList(values), DType.int32),
    shape: shape,
  );
}

void main() {
  group('GatherOp', () {
    test('gathers along dim=0 for 2D tensor', () {
      // input shape [3, 2], index shape [2, 2], dim=0
      // output[i][j] = input[index[i][j]][j]
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([
          1.0, 2.0, // row 0
          3.0, 4.0, // row 1
          5.0, 6.0, // row 2
        ]),
        [3, 2],
      );
      final index = intTensor([2, 0, 1, 1], [2, 2]);

      final result = GatherOp(dim: 0, index: index)(input);

      expect(result.shape, equals([2, 2]));
      // output[0][0] = input[2][0] = 5.0
      expect(result[[0, 0]], closeTo(5.0, 1e-6));
      // output[0][1] = input[0][1] = 2.0
      expect(result[[0, 1]], closeTo(2.0, 1e-6));
      // output[1][0] = input[1][0] = 3.0
      expect(result[[1, 0]], closeTo(3.0, 1e-6));
      // output[1][1] = input[1][1] = 4.0
      expect(result[[1, 1]], closeTo(4.0, 1e-6));
    });

    test('gathers along dim=1 for 2D tensor', () {
      // input shape [2, 3], index shape [2, 2], dim=1
      // output[i][j] = input[i][index[i][j]]
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([
          1.0, 2.0, 3.0, // row 0
          4.0, 5.0, 6.0, // row 1
        ]),
        [2, 3],
      );
      final index = intTensor([2, 0, 1, 2], [2, 2]);

      final result = GatherOp(dim: 1, index: index)(input);

      expect(result.shape, equals([2, 2]));
      // output[0][0] = input[0][2] = 3.0
      expect(result[[0, 0]], closeTo(3.0, 1e-6));
      // output[0][1] = input[0][0] = 1.0
      expect(result[[0, 1]], closeTo(1.0, 1e-6));
      // output[1][0] = input[1][1] = 5.0
      expect(result[[1, 0]], closeTo(5.0, 1e-6));
      // output[1][1] = input[1][2] = 6.0
      expect(result[[1, 1]], closeTo(6.0, 1e-6));
    });

    test('gathers 1D tensor', () {
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0, 30.0, 40.0]),
        [4],
      );
      final index = intTensor([3, 0, 1], [3]);

      final result = GatherOp(dim: 0, index: index)(input);

      expect(result.shape, equals([3]));
      expect(result[[0]], closeTo(40.0, 1e-6));
      expect(result[[1]], closeTo(10.0, 1e-6));
      expect(result[[2]], closeTo(20.0, 1e-6));
    });

    test('negative dim indexes from end', () {
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [2, 3],
      );
      final index = intTensor([2, 0, 1, 2], [2, 2]);

      final result = GatherOp(dim: -1, index: index)(input);

      expect(result.shape, equals([2, 2]));
      // Same as dim=1
      expect(result[[0, 0]], closeTo(3.0, 1e-6));
      expect(result[[0, 1]], closeTo(1.0, 1e-6));
    });

    test('throws for out-of-bounds dim', () {
      final input = TensorBuffer.zeros([3, 4]);
      final index = intTensor([0, 1, 0, 1], [2, 2]);

      expect(
        () => GatherOp(dim: 3, index: index)(input),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });

    test('throws for rank mismatch between input and index', () {
      final input = TensorBuffer.zeros([3, 4]);
      final index = intTensor([0, 1, 2], [3]);

      expect(
        () => GatherOp(dim: 0, index: index)(input),
        throwsA(isA<ShapeMismatchException>()),
      );
    });

    test('output shape equals index shape', () {
      final index = intTensor([0, 0, 0, 0, 0, 0], [2, 3]);
      final op = GatherOp(dim: 0, index: index);

      expect(op.computeOutputShape([4, 3]), equals([2, 3]));
    });

    test('does not modify input', () {
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );
      final index = intTensor([2, 0], [2]);

      GatherOp(dim: 0, index: index)(input);

      expect(input[[0]], closeTo(1.0, 1e-6));
      expect(input[[2]], closeTo(3.0, 1e-6));
    });

    test('works with float64', () {
      final input = TensorBuffer.fromFloat64List(
        Float64List.fromList([10.0, 20.0, 30.0]),
        [3],
      );
      final index = intTensor([2, 0], [2]);

      final result = GatherOp(dim: 0, index: index)(input);

      expect(result[[0]], closeTo(30.0, 1e-10));
      expect(result[[1]], closeTo(10.0, 1e-10));
    });

    test('name is Gather', () {
      final index = intTensor([0], [1]);
      expect(GatherOp(dim: 0, index: index).name, equals('Gather'));
    });

    test('capabilities include pytorch and onnx equivalents', () {
      final index = intTensor([0], [1]);
      final caps = GatherOp(dim: 0, index: index).capabilities;
      expect(caps.preservesShape, isFalse);
      expect(caps.pytorchEquivalent, equals('torch.gather'));
      expect(caps.onnxOpType, equals('GatherElements'));
    });

    test('throws for index value out of bounds', () {
      final input = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );
      // Index value 5 is out of bounds for dim 0 with size 3
      final index = intTensor([0, 5], [2]);

      expect(
        () => GatherOp(dim: 0, index: index)(input),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });
  });
}
