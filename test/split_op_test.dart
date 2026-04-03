import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('split', () {
    test('splits 1D tensor into equal parts', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [6],
      );

      final result = split(tensor, [2, 2, 2]);

      expect(result.length, equals(3));
      expect(result[0].shape, equals([2]));
      expect(result[0][[0]], closeTo(1.0, 1e-6));
      expect(result[0][[1]], closeTo(2.0, 1e-6));
      expect(result[1][[0]], closeTo(3.0, 1e-6));
      expect(result[1][[1]], closeTo(4.0, 1e-6));
      expect(result[2][[0]], closeTo(5.0, 1e-6));
      expect(result[2][[1]], closeTo(6.0, 1e-6));
    });

    test('splits 1D tensor into unequal parts', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
        [5],
      );

      final result = split(tensor, [2, 3]);

      expect(result.length, equals(2));
      expect(result[0].shape, equals([2]));
      expect(result[1].shape, equals([3]));
      expect(result[0][[0]], closeTo(1.0, 1e-6));
      expect(result[1][[0]], closeTo(3.0, 1e-6));
    });

    test('splits 2D tensor along dim=0', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        [4, 2],
      );

      final result = split(tensor, [2, 2], dim: 0);

      expect(result.length, equals(2));
      expect(result[0].shape, equals([2, 2]));
      expect(result[1].shape, equals([2, 2]));
      expect(result[0][[0, 0]], closeTo(1.0, 1e-6));
      expect(result[0][[1, 1]], closeTo(4.0, 1e-6));
      expect(result[1][[0, 0]], closeTo(5.0, 1e-6));
      expect(result[1][[1, 1]], closeTo(8.0, 1e-6));
    });

    test('splits 2D tensor along dim=1', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        [2, 4],
      );

      final result = split(tensor, [1, 3], dim: 1);

      expect(result.length, equals(2));
      expect(result[0].shape, equals([2, 1]));
      expect(result[1].shape, equals([2, 3]));
      expect(result[0][[0, 0]], closeTo(1.0, 1e-6));
      expect(result[0][[1, 0]], closeTo(5.0, 1e-6));
      expect(result[1][[0, 0]], closeTo(2.0, 1e-6));
      expect(result[1][[0, 2]], closeTo(4.0, 1e-6));
    });

    test('supports negative dim', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
      );

      final result = split(tensor, [1, 1], dim: -1);

      expect(result.length, equals(2));
      expect(result[0].shape, equals([2, 1]));
      expect(result[1].shape, equals([2, 1]));
    });

    test('throws for size sum mismatch', () {
      final tensor = TensorBuffer.zeros([6]);

      expect(
        () => split(tensor, [2, 2]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for empty splitSizes', () {
      final tensor = TensorBuffer.zeros([6]);

      expect(
        () => split(tensor, []),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('works with float64', () {
      final tensor = TensorBuffer.fromFloat64List(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
        [4],
      );

      final result = split(tensor, [2, 2]);

      expect(result[0][[0]], closeTo(1.0, 1e-10));
      expect(result[1][[0]], closeTo(3.0, 1e-10));
    });

    test('does not modify input', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
        [4],
      );

      split(tensor, [2, 2]);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[3]], closeTo(4.0, 1e-6));
    });
  });

  group('chunk', () {
    test('splits into equal chunks', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [6],
      );

      final result = chunk(tensor, 3);

      expect(result.length, equals(3));
      expect(result[0].shape, equals([2]));
      expect(result[1].shape, equals([2]));
      expect(result[2].shape, equals([2]));
    });

    test('last chunk is smaller when not evenly divisible', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
        [5],
      );

      final result = chunk(tensor, 3);

      expect(result.length, equals(3));
      expect(result[0].shape, equals([2]));
      expect(result[1].shape, equals([2]));
      expect(result[2].shape, equals([1]));
    });

    test('chunks 2D tensor along dim=0', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [3, 2],
      );

      final result = chunk(tensor, 2, dim: 0);

      expect(result.length, equals(2));
      expect(result[0].shape, equals([2, 2]));
      expect(result[1].shape, equals([1, 2]));
    });

    test('throws for zero chunks', () {
      final tensor = TensorBuffer.zeros([6]);

      expect(() => chunk(tensor, 0), throwsA(isA<InvalidParameterException>()));
    });

    test('throws for negative chunks', () {
      final tensor = TensorBuffer.zeros([6]);

      expect(
        () => chunk(tensor, -1),
        throwsA(isA<InvalidParameterException>()),
      );
    });
  });
}
