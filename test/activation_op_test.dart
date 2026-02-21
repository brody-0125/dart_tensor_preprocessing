import 'dart:math' as math;
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
      expect(
          LeakyReLUOp(negativeSlope: 0.2).name, equals('LeakyReLU(slope=0.2)'));
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

  group('GELUOp', () {
    test('GELU(0) = 0', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final gelu = GELUOp();
      final result = gelu(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
    });

    test('GELU is monotonically increasing for x > -0.5', () {
      final data = Float32List.fromList([0.0, 1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final gelu = GELUOp();
      final result = gelu(tensor);

      for (int i = 1; i < 4; i++) {
        expect(result[[i]], greaterThan(result[[i - 1]]));
      }
    });

    test('exact and tanh approximation give similar results', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final geluExact = GELUOp(approximate: 'none');
      final geluTanh = GELUOp(approximate: 'tanh');

      final exactResult = geluExact(tensor);
      final tanhResult = geluTanh(tensor);

      for (int i = 0; i < 5; i++) {
        expect(exactResult[[i]], closeTo(tanhResult[[i]], 0.01));
      }
    });

    test('preserves shape', () {
      expect(GELUOp().computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });

    test('throws for invalid approximate parameter', () {
      expect(
        () => GELUOp(approximate: 'invalid'),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('name reflects approximation mode', () {
      expect(GELUOp().name, equals('GELU'));
      expect(GELUOp(approximate: 'tanh').name, equals('GELU(approximate=tanh)'));
    });
  });

  group('SiLUOp', () {
    test('SiLU(0) = 0', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final silu = SiLUOp();
      final result = silu(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6));
    });

    test('SiLU(x) = x * sigmoid(x)', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final silu = SiLUOp();
      final result = silu(tensor);

      // x / (1 + exp(-x))
      expect(result[[0]], closeTo(-2.0 / (1 + 7.389), 0.01)); // e^2 ≈ 7.389
      expect(result[[2]], closeTo(0.0, 1e-6));
      expect(result[[3]], closeTo(1.0 / (1 + 0.368), 0.01)); // e^-1 ≈ 0.368
    });

    test('large positive values approach x', () {
      final data = Float32List.fromList([10.0, 20.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final silu = SiLUOp();
      final result = silu(tensor);

      expect(result[[0]], closeTo(10.0, 0.1));
      expect(result[[1]], closeTo(20.0, 0.1));
    });

    test('name is SiLU', () {
      expect(SiLUOp().name, equals('SiLU'));
    });
  });

  group('HardsigmoidOp', () {
    test('hardsigmoid(0) = 0.5', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final op = HardsigmoidOp();
      final result = op(tensor);

      expect(result[[0]], closeTo(0.5, 1e-6)); // (0+3)/6 = 0.5
    });

    test('hardsigmoid clamps to [0, 1]', () {
      final data = Float32List.fromList([-10.0, -3.0, 3.0, 10.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final op = HardsigmoidOp();
      final result = op(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6)); // clamped
      expect(result[[1]], closeTo(0.0, 1e-6)); // (-3+3)/6 = 0
      expect(result[[2]], closeTo(1.0, 1e-6)); // (3+3)/6 = 1
      expect(result[[3]], closeTo(1.0, 1e-6)); // clamped
    });

    test('name is Hardsigmoid', () {
      expect(HardsigmoidOp().name, equals('Hardsigmoid'));
    });
  });

  group('HardswishOp', () {
    test('hardswish(0) = 0', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final op = HardswishOp();
      final result = op(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6)); // 0 * 0.5 = 0
    });

    test('hardswish(x) = x * hardsigmoid(x)', () {
      final data = Float32List.fromList([-3.0, 0.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final op = HardswishOp();
      final result = op(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6)); // -3 * 0 = 0
      expect(result[[1]], closeTo(0.0, 1e-6)); // 0 * 0.5 = 0
      expect(result[[2]], closeTo(3.0, 1e-6)); // 3 * 1 = 3
    });

    test('name is Hardswish', () {
      expect(HardswishOp().name, equals('Hardswish'));
    });
  });

  group('MishOp', () {
    test('Mish(0) ≈ 0', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final op = MishOp();
      final result = op(tensor);

      // mish(0) = 0 * tanh(ln(2)) ≈ 0
      expect(result[[0]], closeTo(0.0, 1e-5));
    });

    test('Mish is continuous and smooth', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final op = MishOp();
      final result = op(tensor);

      // Just verify no NaN/Inf
      for (int i = 0; i < 5; i++) {
        expect(result[[i]].isNaN, isFalse);
        expect(result[[i]].isInfinite, isFalse);
      }
    });

    test('name is Mish', () {
      expect(MishOp().name, equals('Mish'));
    });
  });

  group('ELUOp', () {
    test('ELU(x) = x for x > 0', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final op = ELUOp();
      final result = op(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(3.0, 1e-6));
    });

    test('ELU(x) = alpha * (exp(x) - 1) for x < 0', () {
      final data = Float32List.fromList([-1.0, -2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final op = ELUOp(alpha: 1.0);
      final result = op(tensor);

      // e^-1 - 1 ≈ -0.632
      expect(result[[0]], closeTo(-0.632, 0.01));
      // e^-2 - 1 ≈ -0.865
      expect(result[[1]], closeTo(-0.865, 0.01));
    });

    test('custom alpha', () {
      final data = Float32List.fromList([-1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final op = ELUOp(alpha: 2.0);
      final result = op(tensor);

      // 2 * (e^-1 - 1) ≈ -1.264
      expect(result[[0]], closeTo(-1.264, 0.01));
    });

    test('name reflects alpha', () {
      expect(ELUOp().name, equals('ELU'));
      expect(ELUOp(alpha: 2.0).name, equals('ELU(alpha=2.0)'));
    });
  });

  group('SELUOp', () {
    const alpha = 1.6732632423543772;
    const scale = 1.0507009873554805;

    test('applies SELU formula for positive values', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final selu = SELUOp();
      final result = selu(tensor);

      // For x > 0: selu(x) = scale * x
      expect(result[[0]], closeTo(scale * 1.0, 1e-5));
      expect(result[[1]], closeTo(scale * 2.0, 1e-5));
      expect(result[[2]], closeTo(scale * 3.0, 1e-5));
    });

    test('applies SELU formula for negative values', () {
      final data = Float32List.fromList([-1.0, -2.0, -0.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final selu = SELUOp();
      final result = selu(tensor);

      // For x < 0: selu(x) = scale * alpha * (exp(x) - 1)
      for (int i = 0; i < 3; i++) {
        final x = data[i];
        final expected = scale * alpha * (math.exp(x) - 1);
        expect(result[[i]], closeTo(expected, 1e-5));
      }
    });

    test('selu(0) equals 0', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = SELUOp()(tensor);
      expect(result[[0]], closeTo(0.0, 1e-6));
    });

    test('preserves shape', () {
      expect(SELUOp().computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });

    test('does not modify input tensor', () {
      final data = Float32List.fromList([-1.0, 0.0, 1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      SELUOp()(tensor);

      expect(tensor[[0]], closeTo(-1.0, 1e-6));
      expect(tensor[[1]], closeTo(0.0, 1e-6));
      expect(tensor[[2]], closeTo(1.0, 1e-6));
    });

    test('applyInPlace modifies tensor', () {
      final data = Float32List.fromList([-1.0, 0.0, 1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      SELUOp().applyInPlace(tensor);

      expect(tensor[[0]], closeTo(scale * alpha * (math.exp(-1.0) - 1), 1e-5));
      expect(tensor[[1]], closeTo(0.0, 1e-6));
      expect(tensor[[2]], closeTo(scale * 1.0, 1e-5));
    });

    test('applyInPlace throws for non-contiguous tensor', () {
      final tensor = TensorBuffer.zeros([4, 4]);
      final transposed = tensor.transpose([1, 0]);

      expect(
        () => SELUOp().applyInPlace(transposed),
        throwsA(isA<NonContiguousException>()),
      );
    });

    test('works with float64', () {
      final data = Float64List.fromList([-1.0, 0.0, 1.0]);
      final tensor = TensorBuffer.fromFloat64List(data, [3]);

      final result = SELUOp()(tensor);

      expect(result[[0]], closeTo(scale * alpha * (math.exp(-1.0) - 1), 1e-10));
      expect(result[[1]], closeTo(0.0, 1e-10));
      expect(result[[2]], closeTo(scale * 1.0, 1e-10));
    });

    test('name is SELU', () {
      expect(SELUOp().name, equals('SELU'));
    });

    test('capabilities indicate in-place and contiguous', () {
      final caps = SELUOp().capabilities;
      expect(caps.supportsInPlace, isTrue);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.pytorchEquivalent, equals('F.selu'));
      expect(caps.onnxOpType, equals('Selu'));
    });
  });

  group('GLUOp', () {
    double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

    test('splits and applies GLU along last dim', () {
      // [a1, a2, b1, b2] -> GLU -> [a1*sigmoid(b1), a2*sigmoid(b2)]
      final data = Float32List.fromList([1.0, 2.0, 0.5, -0.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final result = GLUOp()(tensor);

      expect(result.shape, equals([2]));
      expect(result[[0]], closeTo(1.0 * sigmoid(0.5), 1e-5));
      expect(result[[1]], closeTo(2.0 * sigmoid(-0.5), 1e-5));
    });

    test('splits along specified dim for 2D tensor', () {
      // Shape [2, 4] with dim=1 -> split along dim 1 -> [2, 2]
      final data = Float32List.fromList([
        1.0, 2.0, 0.5, -0.5, // row 0
        3.0, 4.0, 1.0, -1.0, // row 1
      ]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 4]);

      final result = GLUOp(dim: 1)(tensor);

      expect(result.shape, equals([2, 2]));
      expect(result[[0, 0]], closeTo(1.0 * sigmoid(0.5), 1e-5));
      expect(result[[0, 1]], closeTo(2.0 * sigmoid(-0.5), 1e-5));
      expect(result[[1, 0]], closeTo(3.0 * sigmoid(1.0), 1e-5));
      expect(result[[1, 1]], closeTo(4.0 * sigmoid(-1.0), 1e-5));
    });

    test('splits along dim=0 for 2D tensor', () {
      // Shape [4, 2] with dim=0 -> split along dim 0 -> [2, 2]
      final data = Float32List.fromList([
        1.0, 2.0, // row 0 -> a
        3.0, 4.0, // row 1 -> a
        0.5, -0.5, // row 2 -> b
        1.0, -1.0, // row 3 -> b
      ]);
      final tensor = TensorBuffer.fromFloat32List(data, [4, 2]);

      final result = GLUOp(dim: 0)(tensor);

      expect(result.shape, equals([2, 2]));
      expect(result[[0, 0]], closeTo(1.0 * sigmoid(0.5), 1e-5));
      expect(result[[0, 1]], closeTo(2.0 * sigmoid(-0.5), 1e-5));
      expect(result[[1, 0]], closeTo(3.0 * sigmoid(1.0), 1e-5));
      expect(result[[1, 1]], closeTo(4.0 * sigmoid(-1.0), 1e-5));
    });

    test('negative dim indexes from end', () {
      final data = Float32List.fromList([1.0, 2.0, 0.5, -0.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final result = GLUOp(dim: -1)(tensor);

      expect(result.shape, equals([2]));
      expect(result[[0]], closeTo(1.0 * sigmoid(0.5), 1e-5));
    });

    test('throws for odd-sized dimension', () {
      final tensor = TensorBuffer.zeros([3]);

      expect(
        () => GLUOp()(tensor),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for out-of-bounds dim', () {
      final tensor = TensorBuffer.zeros([4]);

      expect(
        () => GLUOp(dim: 2)(tensor),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });

    test('computeOutputShape halves the dim', () {
      final glu = GLUOp(dim: 1);
      expect(glu.computeOutputShape([2, 6, 4]), equals([2, 3, 4]));
    });

    test('does not modify input tensor', () {
      final data = Float32List.fromList([1.0, 2.0, 0.5, -0.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      GLUOp()(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[3]], closeTo(-0.5, 1e-6));
    });

    test('works with float64', () {
      final data = Float64List.fromList([1.0, 2.0, 0.5, -0.5]);
      final tensor = TensorBuffer.fromFloat64List(data, [4]);

      final result = GLUOp()(tensor);

      expect(result.shape, equals([2]));
      expect(result[[0]], closeTo(1.0 * sigmoid(0.5), 1e-10));
    });

    test('name is GLU', () {
      expect(GLUOp().name, equals('GLU'));
      expect(GLUOp(dim: 1).name, equals('GLU(dim=1)'));
    });

    test('capabilities indicate shape change', () {
      final caps = GLUOp().capabilities;
      expect(caps.preservesShape, isFalse);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.supportsInPlace, isFalse);
      expect(caps.pytorchEquivalent, equals('F.glu'));
      expect(caps.onnxOpType, equals('Split+Sigmoid+Mul'));
    });

    test('works with 3D tensor', () {
      // Shape [2, 2, 4], dim=-1 -> [2, 2, 2]
      final data = Float32List.fromList([
        1.0, 2.0, 0.5, -0.5,
        3.0, 4.0, 1.0, -1.0,
        5.0, 6.0, 0.0, 0.0,
        7.0, 8.0, 2.0, -2.0,
      ]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 4]);

      final result = GLUOp(dim: -1)(tensor);

      expect(result.shape, equals([2, 2, 2]));
      expect(result[[0, 0, 0]], closeTo(1.0 * sigmoid(0.5), 1e-5));
      expect(result[[0, 0, 1]], closeTo(2.0 * sigmoid(-0.5), 1e-5));
    });
  });
}
