import 'dart:math' as math;
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('LpNormalizeOp', () {
    group('L2 normalization', () {
      test('normalizes 1D vector', () {
        final data = Float32List.fromList([3.0, 4.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = LpNormalizeOp.l2()(tensor);

        // norm = sqrt(9 + 16) = 5
        expect(result[[0]], closeTo(3.0 / 5.0, 1e-5));
        expect(result[[1]], closeTo(4.0 / 5.0, 1e-5));
      });

      test('normalizes along last dim for 2D tensor', () {
        final data = Float32List.fromList([
          3.0, 4.0, // row 0: norm = 5
          1.0, 0.0, // row 1: norm = 1
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final result = LpNormalizeOp.l2(dim: -1)(tensor);

        expect(result[[0, 0]], closeTo(3.0 / 5.0, 1e-5));
        expect(result[[0, 1]], closeTo(4.0 / 5.0, 1e-5));
        expect(result[[1, 0]], closeTo(1.0, 1e-5));
        expect(result[[1, 1]], closeTo(0.0, 1e-5));
      });

      test('normalizes along dim=0 for 2D tensor', () {
        final data = Float32List.fromList([3.0, 1.0, 4.0, 0.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final result = LpNormalizeOp.l2(dim: 0)(tensor);

        // col 0: norm = sqrt(9+16) = 5
        expect(result[[0, 0]], closeTo(3.0 / 5.0, 1e-5));
        expect(result[[1, 0]], closeTo(4.0 / 5.0, 1e-5));
        // col 1: norm = 1
        expect(result[[0, 1]], closeTo(1.0, 1e-5));
        expect(result[[1, 1]], closeTo(0.0, 1e-5));
      });

      test('handles zero vector with eps', () {
        final data = Float32List.fromList([0.0, 0.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = LpNormalizeOp.l2()(tensor);

        // Should not throw or produce NaN
        expect(result[[0]], isNot(isNaN));
        expect(result[[1]], isNot(isNaN));
      });

      test('unit vector remains unchanged', () {
        final norm = math.sqrt(2.0);
        final data = Float32List.fromList([1.0 / norm, 1.0 / norm]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = LpNormalizeOp.l2()(tensor);

        expect(result[[0]], closeTo(1.0 / norm, 1e-5));
        expect(result[[1]], closeTo(1.0 / norm, 1e-5));
      });
    });

    group('L1 normalization', () {
      test('normalizes 1D vector', () {
        final data = Float32List.fromList([1.0, -2.0, 3.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final result = LpNormalizeOp.l1()(tensor);

        // norm = |1| + |2| + |3| = 6
        expect(result[[0]], closeTo(1.0 / 6.0, 1e-5));
        expect(result[[1]], closeTo(-2.0 / 6.0, 1e-5));
        expect(result[[2]], closeTo(3.0 / 6.0, 1e-5));
      });

      test('handles zero vector with eps', () {
        final data = Float32List.fromList([0.0, 0.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = LpNormalizeOp.l1()(tensor);

        expect(result[[0]], isNot(isNaN));
      });
    });

    group('Linf normalization', () {
      test('normalizes 1D vector', () {
        final data = Float32List.fromList([1.0, -3.0, 2.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final result = LpNormalizeOp.linf()(tensor);

        // norm = max(|1|, |3|, |2|) = 3
        expect(result[[0]], closeTo(1.0 / 3.0, 1e-5));
        expect(result[[1]], closeTo(-3.0 / 3.0, 1e-5));
        expect(result[[2]], closeTo(2.0 / 3.0, 1e-5));
      });

      test('handles zero vector with eps', () {
        final data = Float32List.fromList([0.0, 0.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = LpNormalizeOp.linf()(tensor);

        expect(result[[0]], isNot(isNaN));
      });
    });

    group('general p-norm', () {
      test('p=3 normalizes correctly', () {
        final data = Float32List.fromList([1.0, 2.0, 3.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final result = LpNormalizeOp(p: 3.0)(tensor);

        // norm = (1^3 + 2^3 + 3^3)^(1/3) = (1+8+27)^(1/3) = 36^(1/3)
        final norm = math.pow(36.0, 1.0 / 3.0);
        expect(result[[0]], closeTo(1.0 / norm, 1e-5));
        expect(result[[1]], closeTo(2.0 / norm, 1e-5));
        expect(result[[2]], closeTo(3.0 / norm, 1e-5));
      });
    });

    group('edge cases and errors', () {
      test('preserves shape', () {
        expect(
          LpNormalizeOp.l2().computeOutputShape([2, 3, 4]),
          equals([2, 3, 4]),
        );
      });

      test('does not modify input', () {
        final data = Float32List.fromList([3.0, 4.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        LpNormalizeOp.l2()(tensor);

        expect(tensor[[0]], closeTo(3.0, 1e-6));
        expect(tensor[[1]], closeTo(4.0, 1e-6));
      });

      test('in-place modifies tensor', () {
        final data = Float32List.fromList([3.0, 4.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        LpNormalizeOp.l2().applyInPlace(tensor);

        expect(tensor[[0]], closeTo(3.0 / 5.0, 1e-5));
        expect(tensor[[1]], closeTo(4.0 / 5.0, 1e-5));
      });

      test('applyInPlace throws for non-contiguous', () {
        final tensor = TensorBuffer.zeros([4, 4]);
        final transposed = tensor.transpose([1, 0]);

        expect(
          () => LpNormalizeOp.l2().applyInPlace(transposed),
          throwsA(isA<NonContiguousException>()),
        );
      });

      test('works with float64', () {
        final data = Float64List.fromList([3.0, 4.0]);
        final tensor = TensorBuffer.fromFloat64List(data, [2]);

        final result = LpNormalizeOp.l2()(tensor);

        expect(result[[0]], closeTo(3.0 / 5.0, 1e-10));
        expect(result[[1]], closeTo(4.0 / 5.0, 1e-10));
      });

      test('name reflects parameters', () {
        expect(LpNormalizeOp.l2().name, equals('LpNormalize(p=2.0)'));
        expect(LpNormalizeOp.l1().name, equals('LpNormalize(p=1.0)'));
        expect(LpNormalizeOp.linf().name, equals('LpNormalize(p=Infinity)'));
      });

      test('capabilities include pytorch and onnx equivalents', () {
        final caps = LpNormalizeOp.l2().capabilities;
        expect(caps.supportsInPlace, isTrue);
        expect(caps.requiresContiguous, isTrue);
        expect(caps.pytorchEquivalent, equals('F.normalize'));
        expect(caps.onnxOpType, equals('LpNormalization'));
      });
    });
  });
}
