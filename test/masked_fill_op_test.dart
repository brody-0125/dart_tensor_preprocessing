import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('MaskedFillOp', () {
    test('fills masked positions with value', () {
      final mask = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 1.0, 0.0]),
        [4],
      );
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0, 30.0, 40.0]),
        [4],
      );

      final op = MaskedFillOp(mask: mask, value: -1.0);
      final result = op(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-6)); // masked
      expect(result[[1]], closeTo(20.0, 1e-6)); // not masked
      expect(result[[2]], closeTo(-1.0, 1e-6)); // masked
      expect(result[[3]], closeTo(40.0, 1e-6)); // not masked
    });

    test('works with 2D tensor', () {
      final mask = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 0.0, 1.0]),
        [2, 2],
      );
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );

      final result = MaskedFillOp(mask: mask, value: 0.0)(tensor);

      expect(result.shape, equals([2, 2]));
      expect(result[[0, 0]], closeTo(0.0, 1e-6));
      expect(result[[0, 1]], closeTo(2.0, 1e-6));
      expect(result[[1, 0]], closeTo(3.0, 1e-6));
      expect(result[[1, 1]], closeTo(0.0, 1e-6));
    });

    test('attention mask fills with negative infinity', () {
      final mask = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0, 1.0]),
        [3],
      );
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0]),
        [3],
      );

      final result = MaskedFillOp.attentionMask(mask)(tensor);

      expect(result[[0]], equals(double.negativeInfinity));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], equals(double.negativeInfinity));
    });

    test('does not modify input tensor', () {
      final mask = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0]),
        [2],
      );
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0]),
        [2],
      );

      MaskedFillOp(mask: mask, value: 0.0)(tensor);

      expect(tensor[[0]], closeTo(10.0, 1e-6));
      expect(tensor[[1]], closeTo(20.0, 1e-6));
    });

    test('applyInPlace modifies tensor', () {
      final mask = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 0.0]),
        [2],
      );
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([10.0, 20.0]),
        [2],
      );

      MaskedFillOp(mask: mask, value: -1.0).applyInPlace(tensor);

      expect(tensor[[0]], closeTo(-1.0, 1e-6));
      expect(tensor[[1]], closeTo(20.0, 1e-6));
    });

    test('applyInPlace throws for non-contiguous tensor', () {
      final mask = TensorBuffer.zeros([4, 4]);
      final tensor = TensorBuffer.zeros([4, 4]);
      final transposed = tensor.transpose([1, 0]);

      expect(
        () => MaskedFillOp(mask: mask, value: 0.0).applyInPlace(transposed),
        throwsA(isA<NonContiguousException>()),
      );
    });

    test('throws for shape mismatch', () {
      final mask = TensorBuffer.zeros([3]);
      final tensor = TensorBuffer.zeros([4]);

      expect(
        () => MaskedFillOp(mask: mask, value: 0.0)(tensor),
        throwsA(isA<ShapeMismatchException>()),
      );
    });

    test('works with float64', () {
      final mask = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 0.0]),
        [2],
      );
      final tensor = TensorBuffer.fromFloat64List(
        Float64List.fromList([10.0, 20.0]),
        [2],
      );

      final result = MaskedFillOp(mask: mask, value: -1.0)(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-10));
      expect(result[[1]], closeTo(20.0, 1e-10));
    });

    test('preserves shape', () {
      final mask = TensorBuffer.zeros([2, 3]);
      final op = MaskedFillOp(mask: mask, value: 0.0);
      expect(op.computeOutputShape([2, 3]), equals([2, 3]));
    });

    test('name is MaskedFill', () {
      final mask = TensorBuffer.zeros([2]);
      expect(MaskedFillOp(mask: mask, value: 0.0).name, equals('MaskedFill'));
    });

    test('capabilities include pytorch equivalent', () {
      final mask = TensorBuffer.zeros([2]);
      final caps = MaskedFillOp(mask: mask, value: 0.0).capabilities;
      expect(caps.supportsInPlace, isTrue);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.pytorchEquivalent, equals('Tensor.masked_fill_'));
    });
  });
}
