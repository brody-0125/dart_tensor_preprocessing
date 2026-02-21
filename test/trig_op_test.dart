import 'dart:typed_data';
import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/ops/trig_op.dart';

void main() {
  // =========================================================================
  // SinOp
  // =========================================================================
  group('SinOp', () {
    test('computes sine of all elements', () {
      final data = Float32List.fromList([0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final sinOp = SinOp();
      final result = sinOp(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(0.5, 1e-5));
      expect(result[[2]], closeTo(math.sqrt2 / 2, 1e-5));
      expect(result[[3]], closeTo(1.0, 1e-5));
      expect(result[[4]], closeTo(0.0, 1e-5));
    });

    test('handles negative angles', () {
      final data = Float32List.fromList([-math.pi / 2]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = SinOp()(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-5));
    });

    test('preserves shape', () {
      expect(SinOp().computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });

    test('does not modify original tensor', () {
      final data = Float32List.fromList([math.pi / 2]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      SinOp()(tensor);

      expect(tensor[[0]], closeTo(math.pi / 2, 1e-6));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = SinOp().capabilities;
      expect(caps.supportsInPlace, isTrue);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.pytorchEquivalent, equals('torch.sin'));
      expect(caps.onnxOpType, equals('Sin'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Sin', () {
      expect(SinOp().name, equals('Sin'));
    });
  });

  // =========================================================================
  // CosOp
  // =========================================================================
  group('CosOp', () {
    test('computes cosine of all elements', () {
      final data = Float32List.fromList([0.0, math.pi / 3, math.pi / 2, math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final cosOp = CosOp();
      final result = cosOp(tensor);

      expect(result[[0]], closeTo(1.0, 1e-5));
      expect(result[[1]], closeTo(0.5, 1e-5));
      expect(result[[2]], closeTo(0.0, 1e-5));
      expect(result[[3]], closeTo(-1.0, 1e-5));
    });

    test('handles negative angles', () {
      final data = Float32List.fromList([-math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = CosOp()(tensor);

      expect(result[[0]], closeTo(-1.0, 1e-5));
    });

    test('preserves shape', () {
      expect(CosOp().computeOutputShape([3, 5]), equals([3, 5]));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = CosOp().capabilities;
      expect(caps.pytorchEquivalent, equals('torch.cos'));
      expect(caps.onnxOpType, equals('Cos'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Cos', () {
      expect(CosOp().name, equals('Cos'));
    });
  });

  // =========================================================================
  // TanOp
  // =========================================================================
  group('TanOp', () {
    test('computes tangent of all elements', () {
      final data = Float32List.fromList([0.0, math.pi / 4, -math.pi / 4]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final tanOp = TanOp();
      final result = tanOp(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(1.0, 1e-5));
      expect(result[[2]], closeTo(-1.0, 1e-5));
    });

    test('handles values near pi/2 (large tangent)', () {
      final data = Float32List.fromList([1.5]); // close to pi/2
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = TanOp()(tensor);

      expect(result[[0]], closeTo(math.tan(1.5), 1e-3));
    });

    test('preserves shape', () {
      expect(TanOp().computeOutputShape([4, 4]), equals([4, 4]));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = TanOp().capabilities;
      expect(caps.pytorchEquivalent, equals('torch.tan'));
      expect(caps.onnxOpType, equals('Tan'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Tan', () {
      expect(TanOp().name, equals('Tan'));
    });
  });

  // =========================================================================
  // AsinOp
  // =========================================================================
  group('AsinOp', () {
    test('computes arcsine of all elements', () {
      final data = Float32List.fromList([0.0, 0.5, 1.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final asinOp = AsinOp();
      final result = asinOp(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(math.pi / 6, 1e-5));
      expect(result[[2]], closeTo(math.pi / 2, 1e-5));
      expect(result[[3]], closeTo(-math.pi / 2, 1e-5));
    });

    test('returns NaN for values outside [-1, 1]', () {
      final data = Float32List.fromList([2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = AsinOp()(tensor);

      expect(result[[0]].isNaN, isTrue);
    });

    test('preserves shape', () {
      expect(AsinOp().computeOutputShape([2, 3]), equals([2, 3]));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = AsinOp().capabilities;
      expect(caps.pytorchEquivalent, equals('torch.asin'));
      expect(caps.onnxOpType, equals('Asin'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Asin', () {
      expect(AsinOp().name, equals('Asin'));
    });
  });

  // =========================================================================
  // AcosOp
  // =========================================================================
  group('AcosOp', () {
    test('computes arccosine of all elements', () {
      final data = Float32List.fromList([1.0, 0.5, 0.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final acosOp = AcosOp();
      final result = acosOp(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(math.pi / 3, 1e-5));
      expect(result[[2]], closeTo(math.pi / 2, 1e-5));
      expect(result[[3]], closeTo(math.pi, 1e-5));
    });

    test('returns NaN for values outside [-1, 1]', () {
      final data = Float32List.fromList([-2.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      final result = AcosOp()(tensor);

      expect(result[[0]].isNaN, isTrue);
    });

    test('preserves shape', () {
      expect(AcosOp().computeOutputShape([5]), equals([5]));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = AcosOp().capabilities;
      expect(caps.pytorchEquivalent, equals('torch.acos'));
      expect(caps.onnxOpType, equals('Acos'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Acos', () {
      expect(AcosOp().name, equals('Acos'));
    });
  });

  // =========================================================================
  // AtanOp
  // =========================================================================
  group('AtanOp', () {
    test('computes arctangent of all elements', () {
      final data = Float32List.fromList([0.0, 1.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      final atanOp = AtanOp();
      final result = atanOp(tensor);

      expect(result[[0]], closeTo(0.0, 1e-5));
      expect(result[[1]], closeTo(math.pi / 4, 1e-5));
      expect(result[[2]], closeTo(-math.pi / 4, 1e-5));
    });

    test('handles large values', () {
      final data = Float32List.fromList([1000.0, -1000.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      final result = AtanOp()(tensor);

      expect(result[[0]], closeTo(math.pi / 2, 2e-3));
      expect(result[[1]], closeTo(-math.pi / 2, 2e-3));
    });

    test('preserves shape', () {
      expect(AtanOp().computeOutputShape([2, 2]), equals([2, 2]));
    });

    test('capabilities include ONNX and PyTorch metadata', () {
      final caps = AtanOp().capabilities;
      expect(caps.pytorchEquivalent, equals('torch.atan'));
      expect(caps.onnxOpType, equals('Atan'));
      expect(caps.onnxOpsetVersion, equals(7));
    });

    test('name returns Atan', () {
      expect(AtanOp().name, equals('Atan'));
    });
  });

  // =========================================================================
  // Atan2Op
  // =========================================================================
  group('Atan2Op', () {
    group('scalar mode', () {
      test('computes atan2(y, scalar) for all elements', () {
        final data = Float32List.fromList([1.0, -1.0, 0.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final atan2Op = Atan2Op(scalar: 1.0);
        final result = atan2Op(tensor);

        expect(result[[0]], closeTo(math.pi / 4, 1e-5));
        expect(result[[1]], closeTo(-math.pi / 4, 1e-5));
        expect(result[[2]], closeTo(0.0, 1e-5));
      });

      test('handles x=0 (y-axis)', () {
        final data = Float32List.fromList([1.0, -1.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [2]);

        final result = Atan2Op(scalar: 0.0)(tensor);

        expect(result[[0]], closeTo(math.pi / 2, 1e-5));
        expect(result[[1]], closeTo(-math.pi / 2, 1e-5));
      });
    });

    group('tensor mode', () {
      test('computes atan2(y, x) element-wise', () {
        final yData = Float32List.fromList([1.0, 1.0, -1.0, -1.0]);
        final xData = Float32List.fromList([1.0, -1.0, 1.0, -1.0]);
        final yTensor = TensorBuffer.fromFloat32List(yData, [4]);
        final xTensor = TensorBuffer.fromFloat32List(xData, [4]);

        final atan2Op = Atan2Op.tensor(xTensor);
        final result = atan2Op(yTensor);

        // Q1: atan2(1,1) = pi/4
        expect(result[[0]], closeTo(math.pi / 4, 1e-5));
        // Q2: atan2(1,-1) = 3pi/4
        expect(result[[1]], closeTo(3 * math.pi / 4, 1e-5));
        // Q4: atan2(-1,1) = -pi/4
        expect(result[[2]], closeTo(-math.pi / 4, 1e-5));
        // Q3: atan2(-1,-1) = -3pi/4
        expect(result[[3]], closeTo(-3 * math.pi / 4, 1e-5));
      });

      test('throws on shape mismatch', () {
        final yData = Float32List.fromList([1.0, 2.0, 3.0]);
        final xData = Float32List.fromList([1.0, 2.0]);
        final yTensor = TensorBuffer.fromFloat32List(yData, [3]);
        final xTensor = TensorBuffer.fromFloat32List(xData, [2]);

        final atan2Op = Atan2Op.tensor(xTensor);

        expect(() => atan2Op(yTensor), throwsA(isA<Exception>()));
      });
    });

    test('preserves shape', () {
      expect(Atan2Op(scalar: 1.0).computeOutputShape([3, 4]), equals([3, 4]));
    });

    test('capabilities include PyTorch metadata', () {
      final caps = Atan2Op(scalar: 1.0).capabilities;
      expect(caps.pytorchEquivalent, equals('torch.atan2'));
      expect(caps.onnxOpType, isNull);
    });

    test('name includes operand info', () {
      expect(Atan2Op(scalar: 1.0).name, equals('Atan2(scalar=1.0)'));
    });
  });

  // =========================================================================
  // In-place operations
  // =========================================================================
  group('in-place operations', () {
    test('SinOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([0.0, math.pi / 2, math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [3]);

      SinOp().applyInPlace(tensor);

      expect(tensor[[0]], closeTo(0.0, 1e-5));
      expect(tensor[[1]], closeTo(1.0, 1e-5));
      expect(tensor[[2]], closeTo(0.0, 1e-5));
    });

    test('CosOp applyInPlace modifies tensor', () {
      final data = Float32List.fromList([0.0, math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      CosOp().applyInPlace(tensor);

      expect(tensor[[0]], closeTo(1.0, 1e-5));
      expect(tensor[[1]], closeTo(-1.0, 1e-5));
    });

    test('Atan2Op applyInPlace modifies tensor', () {
      final data = Float32List.fromList([1.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2]);

      Atan2Op(scalar: 1.0).applyInPlace(tensor);

      expect(tensor[[0]], closeTo(math.pi / 4, 1e-5));
      expect(tensor[[1]], closeTo(-math.pi / 4, 1e-5));
    });
  });

  // =========================================================================
  // Float64 support
  // =========================================================================
  group('float64 support', () {
    test('SinOp works with float64', () {
      final data = Float64List.fromList([0.0, math.pi / 2, math.pi]);
      final tensor = TensorBuffer.fromFloat64List(data, [3]);

      final result = SinOp()(tensor);

      expect(result[[0]], closeTo(0.0, 1e-10));
      expect(result[[1]], closeTo(1.0, 1e-10));
      expect(result[[2]], closeTo(0.0, 1e-10));
    });

    test('Atan2Op works with float64', () {
      final yData = Float64List.fromList([1.0, -1.0]);
      final xData = Float64List.fromList([1.0, -1.0]);
      final yTensor = TensorBuffer.fromFloat64List(yData, [2]);
      final xTensor = TensorBuffer.fromFloat64List(xData, [2]);

      final result = Atan2Op.tensor(xTensor)(yTensor);

      expect(result[[0]], closeTo(math.pi / 4, 1e-10));
      expect(result[[1]], closeTo(-3 * math.pi / 4, 1e-10));
    });
  });

  // =========================================================================
  // Inverse function identities
  // =========================================================================
  group('inverse identities', () {
    test('asin(sin(x)) = x for x in [-pi/2, pi/2]', () {
      final data = Float32List.fromList([-1.0, -0.5, 0.0, 0.5, 1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final result = AsinOp()(SinOp()(tensor));

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], closeTo(data[i], 1e-5));
      }
    });

    test('acos(cos(x)) = x for x in [0, pi]', () {
      final data = Float32List.fromList([0.1, 0.5, 1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final result = AcosOp()(CosOp()(tensor));

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], closeTo(data[i], 1e-5));
      }
    });

    test('atan(tan(x)) = x for x in (-pi/2, pi/2)', () {
      final data = Float32List.fromList([-1.0, -0.5, 0.0, 0.5, 1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [5]);

      final result = AtanOp()(TanOp()(tensor));

      for (int i = 0; i < 5; i++) {
        expect(result[[i]], closeTo(data[i], 1e-5));
      }
    });

    test('sin^2(x) + cos^2(x) = 1', () {
      final data = Float32List.fromList([0.0, 0.5, 1.0, 2.0, 3.0, -1.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [6]);

      final sinResult = SinOp()(tensor);
      final cosResult = CosOp()(tensor);

      for (int i = 0; i < 6; i++) {
        final sinVal = sinResult[[i]];
        final cosVal = cosResult[[i]];
        expect(sinVal * sinVal + cosVal * cosVal, closeTo(1.0, 1e-5));
      }
    });
  });

  // =========================================================================
  // Multi-dimensional tensor support
  // =========================================================================
  group('multi-dimensional tensors', () {
    test('SinOp works on 2D tensors', () {
      final data = Float32List.fromList([
        0.0, math.pi / 2,
        math.pi, 3 * math.pi / 2,
      ]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2]);

      final result = SinOp()(tensor);

      expect(result[[0, 0]], closeTo(0.0, 1e-5));
      expect(result[[0, 1]], closeTo(1.0, 1e-5));
      expect(result[[1, 0]], closeTo(0.0, 1e-5));
      expect(result[[1, 1]], closeTo(-1.0, 1e-5));
    });

    test('CosOp works on 3D tensors', () {
      final data = Float32List.fromList([0.0, math.pi, 0.0, math.pi]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

      final result = CosOp()(tensor);

      expect(result[[0, 0, 0]], closeTo(1.0, 1e-5));
      expect(result[[0, 0, 1]], closeTo(-1.0, 1e-5));
      expect(result[[0, 1, 0]], closeTo(1.0, 1e-5));
      expect(result[[0, 1, 1]], closeTo(-1.0, 1e-5));
    });
  });
}
