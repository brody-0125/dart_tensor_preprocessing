import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/ops/activation_op.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/tensor_exceptions.dart';

void main() {
  group('ReLUOp', () {
    test('sets negative values to zero', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final relu = ReLUOp();
      final result = relu(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6));
      expect(result[[1]], closeTo(0.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
      expect(result[[3]], closeTo(1.0, 1e-6));
      expect(result[[4]], closeTo(2.0, 1e-6));
    });

    test('preserves positive values', () {
      final data = Float32List.fromList([0.5, 1.0, 2.0, 10.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final relu = ReLUOp();
      final result = relu(tensor);

      expect(result[[0]], closeTo(0.5, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(2.0, 1e-6));
      expect(result[[3]], closeTo(10.0, 1e-6));
    });

    test('preserves shape', () {
      final relu = ReLUOp();
      expect(relu.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('LeakyReLUOp', () {
    test('applies negative slope to negative values', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final leakyRelu = LeakyReLUOp(negativeSlope: 0.1);
      final result = leakyRelu(tensor);

      expect(result[[0]], closeTo(-0.2, 1e-6)); // -2.0 * 0.1
      expect(result[[1]], closeTo(-0.1, 1e-6)); // -1.0 * 0.1
      expect(result[[2]], closeTo(0.0, 1e-6));
      expect(result[[3]], closeTo(1.0, 1e-6));
      expect(result[[4]], closeTo(2.0, 1e-6));
    });

    test('default negative slope is 0.01', () {
      final data = Float32List.fromList([-100.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final leakyRelu = LeakyReLUOp();
      final result = leakyRelu(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-6)); // -100.0 * 0.01
    });
  });

  group('SigmoidOp', () {
    test('computes sigmoid of all elements', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final sigmoid = SigmoidOp();
      final result = sigmoid(tensor);

      expect(result[[0]], closeTo(0.5, 1e-6)); // sigmoid(0) = 0.5
    });

    test('large positive values approach 1', () {
      final data = Float32List.fromList([10.0, 20.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final sigmoid = SigmoidOp();
      final result = sigmoid(tensor);

      expect(result[[0]], closeTo(1.0, 1e-4));
      expect(result[[1]], closeTo(1.0, 1e-8));
    });

    test('large negative values approach 0', () {
      final data = Float32List.fromList([-10.0, -20.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final sigmoid = SigmoidOp();
      final result = sigmoid(tensor);

      expect(result[[0]], closeTo(0.0, 1e-4));
      expect(result[[1]], closeTo(0.0, 1e-8));
    });

    test('output range is (0, 1)', () {
      final data = Float32List.fromList([-5.0, -2.0, 0.0, 2.0, 5.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final sigmoid = SigmoidOp();
      final result = sigmoid(tensor);

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], greaterThan(0.0));
        expect(result[[i]], lessThan(1.0));
      }
    });
  });

  group('TanhOp', () {
    test('computes tanh of all elements', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final tanh = TanhOp();
      final result = tanh(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6)); // tanh(0) = 0
    });

    test('large positive values approach 1', () {
      final data = Float32List.fromList([5.0, 10.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final tanh = TanhOp();
      final result = tanh(tensor);

      expect(result[[0]], closeTo(1.0, 1e-4));
      expect(result[[1]], closeTo(1.0, 1e-8));
    });

    test('large negative values approach -1', () {
      final data = Float32List.fromList([-5.0, -10.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final tanh = TanhOp();
      final result = tanh(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-4));
      expect(result[[1]], closeTo(-1.0, 1e-8));
    });

    test('output range is (-1, 1)', () {
      final data = Float32List.fromList([-5.0, -2.0, 0.0, 2.0, 5.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final tanh = TanhOp();
      final result = tanh(tensor);

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], greaterThan(-1.0));
        expect(result[[i]], lessThan(1.0));
      }
    });

    test('is symmetric: tanh(-x) = -tanh(x)', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final negData = Float32List.fromList([-1.0, -2.0, -3.0]);
      final negTensor = TensorBuffer.fromFloat32List(negData, [3]);

      final tanh = TanhOp();
      final result = tanh(tensor);
      final negResult = tanh(negTensor);

      for (int i = 0; i < 3; i++) {
        expect(result[[i]], closeTo(-negResult[[i]], 1e-6));
      }
    });
  });

  group('SoftmaxOp', () {
    test('output sums to 1 along axis', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

      final softmax = SoftmaxOp(axis: 1);
      final result = softmax(tensor);

      // Sum along axis 1 for each row should be 1
      final sum0 = result[[0, 0]] + result[[0, 1]] + result[[0, 2]];
      final sum1 = result[[1, 0]] + result[[1, 1]] + result[[1, 2]];

      expect(sum0, closeTo(1.0, 1e-5));
      expect(sum1, closeTo(1.0, 1e-5));
    });

    test('all outputs are positive', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final softmax = SoftmaxOp(axis: 0);
      final result = softmax(tensor);

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], greaterThan(0.0));
      }
    });

    test('largest input has largest output', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final softmax = SoftmaxOp(axis: 0);
      final result = softmax(tensor);

      expect(result[[2]], greaterThan(result[[1]]));
      expect(result[[1]], greaterThan(result[[0]]));
    });

    test('equal inputs give equal outputs', () {
      final data = Float32List.fromList([2.0, 2.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final softmax = SoftmaxOp(axis: 0);
      final result = softmax(tensor);

      expect(result[[0]], closeTo(1.0 / 3.0, 1e-5));
      expect(result[[1]], closeTo(1.0 / 3.0, 1e-5));
      expect(result[[2]], closeTo(1.0 / 3.0, 1e-5));
    });

    test('handles negative axis', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

      final softmax = SoftmaxOp(axis: -1); // Same as axis=1
      final result = softmax(tensor);

      final sum0 = result[[0, 0]] + result[[0, 1]] + result[[0, 2]];
      expect(sum0, closeTo(1.0, 1e-5));
    });

    test('throws for invalid axis', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final softmax = SoftmaxOp(axis: 5);

      expect(
        () => softmax(tensor),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });
  });

  group('in-place operations', () {
    test('ReLUOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([-2.0, 0.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final relu = ReLUOp();
      relu.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(0.0, 1e-6));
      expect(tensor[[1]], closeTo(0.0, 1e-6));
      expect(tensor[[2]], closeTo(2.0, 1e-6));
    });

    test('SigmoidOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final sigmoid = SigmoidOp();
      sigmoid.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(0.5, 1e-6));
    });
  });

  group('name property', () {
    test('ReLUOp name', () {
      expect(ReLUOp().name, equals('ReLU'));
    });

    test('LeakyReLUOp name', () {
      expect(LeakyReLUOp(negativeSlope: 0.2).name, equals('LeakyReLU(slope=0.2)'));
    });

    test('SigmoidOp name', () {
      expect(SigmoidOp().name, equals('Sigmoid'));
    });

    test('TanhOp name', () {
      expect(TanhOp().name, equals('Tanh'));
    });

    test('SoftmaxOp name', () {
      expect(SoftmaxOp(axis: 1).name, equals('Softmax(axis=1)'));
    });
  });
}
