import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('tensorWhere', () {
    test('selects from x when condition is non-zero', () {
      final condition = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 1.0, 0.0]),
        [4],
      );
      final x = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0, 30.0, 40.0]),
        [4],
      );
      final y = TensorBuffer.fromFloat32List(
        Float32List.fromList([100.0, 200.0, 300.0, 400.0]),
        [4],
      );

      final result = tensorWhere(condition, x, y);

      expect(result[[0]], closeTo(10.0, 1e-6));
      expect(result[[1]], closeTo(200.0, 1e-6));
      expect(result[[2]], closeTo(30.0, 1e-6));
      expect(result[[3]], closeTo(400.0, 1e-6));
    });

    test('works with 2D tensors', () {
      final condition = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 0.0, 1.0]),
        [2, 2],
      );
      final x = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );
      final y = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0, 30.0, 40.0]),
        [2, 2],
      );

      final result = tensorWhere(condition, x, y);

      expect(result.shape, equals([2, 2]));
      expect(result[[0, 0]], closeTo(1.0, 1e-6));
      expect(result[[0, 1]], closeTo(20.0, 1e-6));
      expect(result[[1, 0]], closeTo(30.0, 1e-6));
      expect(result[[1, 1]], closeTo(4.0, 1e-6));
    });

    test('negative values are truthy', () {
      final condition = TensorBuffer.fromFloat32List(
        Float32List.fromList([-1.0, 0.0]),
        [2],
      );
      final x = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0]),
        [2],
      );
      final y = TensorBuffer.fromFloat32List(
        Float32List.fromList([100.0, 200.0]),
        [2],
      );

      final result = tensorWhere(condition, x, y);

      expect(result[[0]], closeTo(10.0, 1e-6)); // -1 is truthy
      expect(result[[1]], closeTo(200.0, 1e-6)); // 0 is falsy
    });

    test('throws for shape mismatch between condition and x', () {
      final condition = TensorBuffer.zeros([3]);
      final x = TensorBuffer.zeros([4]);
      final y = TensorBuffer.zeros([4]);

      expect(
        () => tensorWhere(condition, x, y),
        throwsA(isA<ShapeMismatchException>()),
      );
    });

    test('throws for shape mismatch between x and y', () {
      final condition = TensorBuffer.zeros([4]);
      final x = TensorBuffer.zeros([4]);
      final y = TensorBuffer.zeros([3]);

      expect(
        () => tensorWhere(condition, x, y),
        throwsA(isA<ShapeMismatchException>()),
      );
    });

    test('works with float64', () {
      final condition = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 0.0]),
        [2],
      );
      final x = TensorBuffer.fromFloat64List(
        Float64List.fromList([10.0, 20.0]),
        [2],
      );
      final y = TensorBuffer.fromFloat64List(
        Float64List.fromList([100.0, 200.0]),
        [2],
      );

      final result = tensorWhere(condition, x, y);

      expect(result[[0]], closeTo(10.0, 1e-10));
      expect(result[[1]], closeTo(200.0, 1e-10));
    });
  });

  group('WhereOp', () {
    test('applies as TransformOp with input as x', () {
      final condition = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 1.0]),
        [3],
      );
      final y = TensorBuffer.fromFloat32List(
        Float32List.fromList([100.0, 200.0, 300.0]),
        [3],
      );

      final op = WhereOp(condition: condition, y: y);
      final x = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0, 30.0]),
        [3],
      );

      final result = op(x);

      expect(result[[0]], closeTo(10.0, 1e-6));
      expect(result[[1]], closeTo(200.0, 1e-6));
      expect(result[[2]], closeTo(30.0, 1e-6));
    });

    test('preserves shape', () {
      final condition = TensorBuffer.zeros([2, 3]);
      final y = TensorBuffer.zeros([2, 3]);
      final op = WhereOp(condition: condition, y: y);

      expect(op.computeOutputShape([2, 3]), equals([2, 3]));
    });

    test('name is Where', () {
      final condition = TensorBuffer.zeros([2]);
      final y = TensorBuffer.zeros([2]);
      expect(WhereOp(condition: condition, y: y).name, equals('Where'));
    });

    test('capabilities include pytorch and onnx equivalents', () {
      final condition = TensorBuffer.zeros([2]);
      final y = TensorBuffer.zeros([2]);
      final caps = WhereOp(condition: condition, y: y).capabilities;
      expect(caps.preservesShape, isTrue);
      expect(caps.pytorchEquivalent, equals('torch.where'));
      expect(caps.onnxOpType, equals('Where'));
    });

    test('does not modify input tensors', () {
      final condition = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0]),
        [2],
      );
      final x = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0]),
        [2],
      );
      final y = TensorBuffer.fromFloat32List(
        Float32List.fromList([100.0, 200.0]),
        [2],
      );

      WhereOp(condition: condition, y: y)(x);

      expect(x[[0]], closeTo(10.0, 1e-6));
      expect(x[[1]], closeTo(20.0, 1e-6));
    });
  });
}
