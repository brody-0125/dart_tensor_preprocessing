import 'dart:typed_data';
import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/ops/math_op.dart';

void main() {
  group('AbsOp', () {
    test('computes absolute value of all elements', () {
      final data = Float32List.fromList([-3.0, -1.0, 0.0, 1.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final abs = AbsOp();
      final result = abs(tensor);

      expect(result[[0]], closeTo(3.0, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
      expect(result[[3]], closeTo(1.0, 1e-6));
      expect(result[[4]], closeTo(3.0, 1e-6));
    });

    test('handles already positive values', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final abs = AbsOp();
      final result = abs(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(3.0, 1e-6));
    });

    test('preserves shape', () {
      final abs = AbsOp();
      expect(abs.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('NegOp', () {
    test('negates all elements', () {
      final data = Float32List.fromList([-2.0, -1.0, 0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final neg = NegOp();
      final result = neg(tensor);

      expect(result[[0]], closeTo(2.0, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
      expect(result[[3]], closeTo(-1.0, 1e-6));
      expect(result[[4]], closeTo(-2.0, 1e-6));
    });

    test('double negation returns original', () {
      final data = Float32List.fromList([1.0, -2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final neg = NegOp();
      final result = neg(neg(tensor));

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(-2.0, 1e-6));
      expect(result[[2]], closeTo(3.0, 1e-6));
    });
  });

  group('SqrtOp', () {
    test('computes square root of all elements', () {
      final data = Float32List.fromList([0.0, 1.0, 4.0, 9.0, 16.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final sqrt = SqrtOp();
      final result = sqrt(tensor);

      expect(result[[0]], closeTo(0.0, 1e-6));
      expect(result[[1]], closeTo(1.0, 1e-6));
      expect(result[[2]], closeTo(2.0, 1e-6));
      expect(result[[3]], closeTo(3.0, 1e-6));
      expect(result[[4]], closeTo(4.0, 1e-6));
    });

    test('handles negative values (returns NaN)', () {
      final data = Float32List.fromList([-1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final sqrt = SqrtOp();
      final result = sqrt(tensor);

      expect(result[[0]].isNaN, isTrue);
    });
  });

  group('ExpOp', () {
    test('computes exponential of all elements', () {
      final data = Float32List.fromList([0.0, 1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final exp = ExpOp();
      final result = exp(tensor);

      expect(result[[0]], closeTo(1.0, 1e-5));
      expect(result[[1]], closeTo(math.e, 1e-5));
      expect(result[[2]], closeTo(math.e * math.e, 1e-4));
    });

    test('handles negative exponents', () {
      final data = Float32List.fromList([-1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final exp = ExpOp();
      final result = exp(tensor);

      expect(result[[0]], closeTo(1.0 / math.e, 1e-6));
    });
  });

  group('LogOp', () {
    test('computes natural logarithm of all elements', () {
      final data = Float32List.fromList([1.0, math.e, math.e * math.e]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final log = LogOp();
      final result = log(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(1.0, 1e-5));
      expect(result[[2]], closeTo(2.0, 1e-5));
    });

    test('handles zero (returns -infinity)', () {
      final data = Float32List.fromList([0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final log = LogOp();
      final result = log(tensor);

      expect(result[[0]], equals(double.negativeInfinity));
    });

    test('handles negative values (returns NaN)', () {
      final data = Float32List.fromList([-1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final log = LogOp();
      final result = log(tensor);

      expect(result[[0]].isNaN, isTrue);
    });
  });

  group('FloorOp', () {
    test('rounds all elements down', () {
      final data = Float32List.fromList([1.7, 2.3, -1.2, -2.8, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final floor = FloorOp();
      final result = floor(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(-2.0, 1e-6));
      expect(result[[3]], closeTo(-3.0, 1e-6));
      expect(result[[4]], closeTo(0.0, 1e-6));
    });

    test('handles already integer values', () {
      final data = Float32List.fromList([1.0, -2.0, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final floor = FloorOp();
      final result = floor(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(-2.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
    });

    test('preserves shape', () {
      final floor = FloorOp();
      expect(floor.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('CeilOp', () {
    test('rounds all elements up', () {
      final data = Float32List.fromList([1.2, 2.8, -1.7, -2.3, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final ceil = CeilOp();
      final result = ceil(tensor);

      expect(result[[0]], closeTo(2.0, 1e-6));
      expect(result[[1]], closeTo(3.0, 1e-6));
      expect(result[[2]], closeTo(-1.0, 1e-6));
      expect(result[[3]], closeTo(-2.0, 1e-6));
      expect(result[[4]], closeTo(0.0, 1e-6));
    });

    test('handles already integer values', () {
      final data = Float32List.fromList([1.0, -2.0, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final ceil = CeilOp();
      final result = ceil(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(-2.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
    });

    test('preserves shape', () {
      final ceil = CeilOp();
      expect(ceil.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('RoundOp', () {
    test('rounds all elements to nearest integer', () {
      final data = Float32List.fromList([1.3, 2.7, -1.3, -2.7, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final round = RoundOp();
      final result = round(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(3.0, 1e-6));
      expect(result[[2]], closeTo(-1.0, 1e-6));
      expect(result[[3]], closeTo(-3.0, 1e-6));
      expect(result[[4]], closeTo(0.0, 1e-6));
    });

    test('handles half values', () {
      final data = Float32List.fromList([0.5, 1.5, -0.5, -1.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final round = RoundOp();
      final result = round(tensor);

      // Dart's roundToDouble uses half-to-even (banker's rounding)
      // 0.5 -> 1.0 (rounds up), 1.5 -> 2.0 (rounds up)
      // -0.5 -> -1.0 (rounds down), -1.5 -> -2.0 (rounds down)
      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(2.0, 1e-6));
      expect(result[[2]], closeTo(-1.0, 1e-6));
      expect(result[[3]], closeTo(-2.0, 1e-6));
    });

    test('handles already integer values', () {
      final data = Float32List.fromList([1.0, -2.0, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final round = RoundOp();
      final result = round(tensor);

      expect(result[[0]], closeTo(1.0, 1e-6));
      expect(result[[1]], closeTo(-2.0, 1e-6));
      expect(result[[2]], closeTo(0.0, 1e-6));
    });

    test('preserves shape', () {
      final round = RoundOp();
      expect(round.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('floor and ceil relationship', () {
    test('floor(x) <= x <= ceil(x)', () {
      final data = Float32List.fromList([1.5, -1.5, 2.3, -2.7]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final floor = FloorOp();
      final ceil = CeilOp();
      final floorResult = floor(tensor);
      final ceilResult = ceil(tensor);

      for (int i = 0; i < 4; i++) {
        expect(floorResult[[i]], lessThanOrEqualTo(data[i]));
        expect(ceilResult[[i]], greaterThanOrEqualTo(data[i]));
      }
    });
  });

  group('exp and log are inverses', () {
    test('log(exp(x)) = x', () {
      final data = Float32List.fromList([0.0, 1.0, 2.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final exp = ExpOp();
      final log = LogOp();
      final result = log(exp(tensor));

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(1.0, 1e-5));
      expect(result[[2]], closeTo(2.0, 1e-5));
      expect(result[[3]], closeTo(-1.0, 1e-5));
    });
  });

  group('in-place operations', () {
    test('AbsOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([-3.0, -1.0, 2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final abs = AbsOp();
      abs.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(3.0, 1e-6));
      expect(tensor[[1]], closeTo(1.0, 1e-6));
      expect(tensor[[2]], closeTo(2.0, 1e-6));
    });

    test('FloorOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.7, -2.3, 3.9]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final floor = FloorOp();
      floor.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[1]], closeTo(-3.0, 1e-6));
      expect(tensor[[2]], closeTo(3.0, 1e-6));
    });

    test('CeilOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.2, -2.8, 3.1]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final ceil = CeilOp();
      ceil.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(2.0, 1e-6));
      expect(tensor[[1]], closeTo(-2.0, 1e-6));
      expect(tensor[[2]], closeTo(4.0, 1e-6));
    });

    test('RoundOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.3, -2.7, 3.5]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final round = RoundOp();
      round.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-6));
      expect(tensor[[1]], closeTo(-3.0, 1e-6));
      expect(tensor[[2]], closeTo(4.0, 1e-6));
    });

    test('SqrtOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([4.0, 9.0, 16.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final sqrt = SqrtOp();
      sqrt.applyInPlace(tensor);

      expect(tensor[[0]], closeTo(2.0, 1e-6));
      expect(tensor[[1]], closeTo(3.0, 1e-6));
      expect(tensor[[2]], closeTo(4.0, 1e-6));
    });
  });

  group('name property', () {
    test('AbsOp name', () {
      expect(AbsOp().name, equals('Abs'));
    });

    test('NegOp name', () {
      expect(NegOp().name, equals('Neg'));
    });

    test('SqrtOp name', () {
      expect(SqrtOp().name, equals('Sqrt'));
    });

    test('ExpOp name', () {
      expect(ExpOp().name, equals('Exp'));
    });

    test('LogOp name', () {
      expect(LogOp().name, equals('Log'));
    });

    test('FloorOp name', () {
      expect(FloorOp().name, equals('Floor'));
    });

    test('CeilOp name', () {
      expect(CeilOp().name, equals('Ceil'));
    });

    test('RoundOp name', () {
      expect(RoundOp().name, equals('Round'));
    });
  });
}
