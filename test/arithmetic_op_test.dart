import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/ops/arithmetic_op.dart';

void main() {
  group('AddOp', () {
    group('scalar addition', () {
      test('adds scalar to all elements', () {
        final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

        final add = AddOp(scalar: 10.0);
        final result = add(tensor);

        expect(result[[0, 0]], closeTo(11.0, 1e-6));
        expect(result[[0, 1]], closeTo(12.0, 1e-6));
        expect(result[[1, 0]], closeTo(13.0, 1e-6));
        expect(result[[1, 1]], closeTo(14.0, 1e-6));
      });

      test('adds negative scalar', () {
        final data = Float32List.fromList([5.0, 10.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final add = AddOp(scalar: -3.0);
        final result = add(tensor);

        expect(result[[0]], closeTo(2.0, 1e-6));
        expect(result[[1]], closeTo(7.0, 1e-6));
      });
    });

    group('tensor addition', () {
      test('adds two tensors element-wise', () {
        final data1 = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
        final data2 = Float32List.fromList([10.0, 20.0, 30.0, 40.0]);
        final tensor1 = TensorBuffer.fromFloat32List(data1, [2, 2]);
        final tensor2 = TensorBuffer.fromFloat32List(data2, [2, 2]);

        final add = AddOp.tensor(tensor2);
        final result = add(tensor1);

        expect(result[[0, 0]], closeTo(11.0, 1e-6));
        expect(result[[0, 1]], closeTo(22.0, 1e-6));
        expect(result[[1, 0]], closeTo(33.0, 1e-6));
        expect(result[[1, 1]], closeTo(44.0, 1e-6));
      });
    });

    test('preserves shape', () {
      final add = AddOp(scalar: 1.0);
      expect(add.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('SubOp', () {
    test('subtracts scalar from all elements', () {
      final data = Float32List.fromList([10.0, 20.0, 30.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final sub = SubOp(scalar: 5.0);
      final result = sub(tensor);

      expect(result[[0]], closeTo(5.0, 1e-6));
      expect(result[[1]], closeTo(15.0, 1e-6));
      expect(result[[2]], closeTo(25.0, 1e-6));
    });

    test('subtracts two tensors element-wise', () {
      final data1 = Float32List.fromList([10.0, 20.0]);
      final data2 = Float32List.fromList([3.0, 8.0]);
      final tensor1 = TensorBuffer.fromFloat32List(data1, [2]);
      final tensor2 = TensorBuffer.fromFloat32List(data2, [2]);

      final sub = SubOp.tensor(tensor2);
      final result = sub(tensor1);

      expect(result[[0]], closeTo(7.0, 1e-6));
      expect(result[[1]], closeTo(12.0, 1e-6));
    });
  });

  group('MulOp', () {
    test('multiplies all elements by scalar', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final mul = MulOp(scalar: 2.5);
      final result = mul(tensor);

      expect(result[[0]], closeTo(2.5, 1e-6));
      expect(result[[1]], closeTo(5.0, 1e-6));
      expect(result[[2]], closeTo(7.5, 1e-6));
    });

    test('multiplies two tensors element-wise', () {
      final data1 = Float32List.fromList([2.0, 3.0, 4.0]);
      final data2 = Float32List.fromList([10.0, 20.0, 30.0]);
      final tensor1 = TensorBuffer.fromFloat32List(data1, [3]);
      final tensor2 = TensorBuffer.fromFloat32List(data2, [3]);

      final mul = MulOp.tensor(tensor2);
      final result = mul(tensor1);

      expect(result[[0]], closeTo(20.0, 1e-6));
      expect(result[[1]], closeTo(60.0, 1e-6));
      expect(result[[2]], closeTo(120.0, 1e-6));
    });

    test('multiplies by zero', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final mul = MulOp(scalar: 0.0);
      final result = mul(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6));
      expect(result[[1]], closeTo(0.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
    });
  });

  group('DivOp', () {
    test('divides all elements by scalar', () {
      final data = Float32List.fromList([10.0, 20.0, 30.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final div = DivOp(scalar: 2.0);
      final result = div(tensor);

      expect(result[[0]], closeTo(5.0, 1e-6));
      expect(result[[1]], closeTo(10.0, 1e-6));
      expect(result[[2]], closeTo(15.0, 1e-6));
    });

    test('divides two tensors element-wise', () {
      final data1 = Float32List.fromList([10.0, 20.0, 30.0]);
      final data2 = Float32List.fromList([2.0, 4.0, 5.0]);
      final tensor1 = TensorBuffer.fromFloat32List(data1, [3]);
      final tensor2 = TensorBuffer.fromFloat32List(data2, [3]);

      final div = DivOp.tensor(tensor2);
      final result = div(tensor1);

      expect(result[[0]], closeTo(5.0, 1e-6));
      expect(result[[1]], closeTo(5.0, 1e-6));
      expect(result[[2]], closeTo(6.0, 1e-6));
    });

    test('handles division by zero (returns infinity)', () {
      final data = Float32List.fromList([1.0, -1.0, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final div = DivOp(scalar: 0.0);
      final result = div(tensor);

      expect(result[[0]], equals(double.infinity));
      expect(result[[1]], equals(double.negativeInfinity));
      expect(result[[2]].isNaN, isTrue);
    });
  });

  group('PowOp', () {
    test('raises all elements to power', () {
      final data = Float32List.fromList([2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final pow = PowOp(exponent: 2.0);
      final result = pow(tensor);

      expect(result[[0]], closeTo(4.0, 1e-6));
      expect(result[[1]], closeTo(9.0, 1e-6));
      expect(result[[2]], closeTo(16.0, 1e-6));
    });

    test('handles square root (power 0.5)', () {
      final data = Float32List.fromList([4.0, 9.0, 16.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final pow = PowOp(exponent: 0.5);
      final result = pow(tensor);

      expect(result[[0]], closeTo(2.0, 1e-6));
      expect(result[[1]], closeTo(3.0, 1e-6));
      expect(result[[2]], closeTo(4.0, 1e-6));
    });

    test('handles power 0 (all ones)', () {
      final data = Float32List.fromList([2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final pow = PowOp(exponent: 0.0);
      final result = pow(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(1.0, 1e-6));
    });

    test('handles negative exponent', () {
      final data = Float32List.fromList([2.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final pow = PowOp(exponent: -1.0);
      final result = pow(tensor);

      expect(result[[0]], closeTo(0.5, 1e-6));
      expect(result[[1]], closeTo(0.25, 1e-6));
    });
  });

  group('in-place operations', () {
    test('AddOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final add = AddOp(scalar: 10.0);
      add.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(11.0, 1e-6));
      expect(tensor[[1]], closeTo(12.0, 1e-6));
      expect(tensor[[2]], closeTo(13.0, 1e-6));
    });

    test('MulOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final mul = MulOp(scalar: 2.0);
      mul.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(2.0, 1e-6));
      expect(tensor[[1]], closeTo(4.0, 1e-6));
      expect(tensor[[2]], closeTo(6.0, 1e-6));
    });
  });

  group('name property', () {
    test('AddOp scalar name', () {
      final add = AddOp(scalar: 5.0);
      expect(add.name, equals('Add(scalar=5.0)'));
    });

    test('AddOp tensor name', () {
      final other = TensorBuffer.zeros([2, 2]);
      final add = AddOp.tensor(other);
      expect(add.name, contains('tensor'));
    });

    test('MulOp name', () {
      final mul = MulOp(scalar: 2.0);
      expect(mul.name, equals('Mul(scalar=2.0)'));
    });

    test('PowOp name', () {
      final pow = PowOp(exponent: 2.0);
      expect(pow.name, equals('Pow(exponent=2.0)'));
    });
  });
}
