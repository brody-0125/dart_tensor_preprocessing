import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('RMSNormOp', () {
    test('computes RMS normalization correctly', () {
      // Simple case: [1, 2, 3, 4] with normalized shape [4]
      // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.7386
      // Normalized: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final op = RMSNormOp(normalizedShape: [4]);
      final result = op.apply(tensor);

      final expectedRms = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6);
      expect(result.storage.getAsDouble(0), closeTo(1.0 / expectedRms, 1e-5));
      expect(result.storage.getAsDouble(1), closeTo(2.0 / expectedRms, 1e-5));
      expect(result.storage.getAsDouble(2), closeTo(3.0 / expectedRms, 1e-5));
      expect(result.storage.getAsDouble(3), closeTo(4.0 / expectedRms, 1e-5));
    });

    test('normalizes 2D tensor along last dimension', () {
      // [2, 4] tensor - normalize last 4 elements independently
      final data = Float32List.fromList([
        1.0, 2.0, 3.0, 4.0, // row 0
        2.0, 4.0, 6.0, 8.0, // row 1
      ]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, 4]);

      final op = RMSNormOp(normalizedShape: [4]);
      final result = op.apply(tensor);

      // Row 0: RMS = sqrt(7.5 + eps)
      final rms0 = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6);
      expect(result.storage.getAsDouble(0), closeTo(1.0 / rms0, 1e-5));

      // Row 1: RMS = sqrt(30 + eps) = sqrt((4+16+36+64)/4)
      final rms1 = math.sqrt((4 + 16 + 36 + 64) / 4 + 1e-6);
      expect(result.storage.getAsDouble(4), closeTo(2.0 / rms1, 1e-5));
    });

    test('applies weight scaling', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final op = RMSNormOp(
        normalizedShape: [4],
        weight: [2.0, 2.0, 2.0, 2.0],
      );
      final result = op.apply(tensor);

      final expectedRms = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6);
      // With weight=2.0, result should be doubled
      expect(result.storage.getAsDouble(0), closeTo(2.0 * 1.0 / expectedRms, 1e-5));
      expect(result.storage.getAsDouble(1), closeTo(2.0 * 2.0 / expectedRms, 1e-5));
    });

    test('works with 3D tensor (batch, seq, hidden)', () {
      // [2, 3, 4] - batch=2, seq_len=3, hidden_size=4
      final data = Float32List.fromList(List.generate(24, (i) => (i + 1).toDouble()));
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4]);

      final op = RMSNormOp(normalizedShape: [4]);
      final result = op.apply(tensor);

      expect(result.shape, equals([2, 3, 4]));

      // First hidden vector [1,2,3,4]
      final rms0 = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6);
      expect(result.storage.getAsDouble(0), closeTo(1.0 / rms0, 1e-5));
    });

    test('in-place modification works', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);
      final original = tensor.storage.getAsDouble(0);

      final op = RMSNormOp(normalizedShape: [4]);
      op.applyInPlace(tensor);

      expect(tensor.storage.getAsDouble(0), isNot(equals(original)));
    });

    test('throws on shape mismatch', () {
      final tensor = TensorBuffer.zeros([3]);
      final op = RMSNormOp(normalizedShape: [4]);

      expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
    });

    test('throws on insufficient rank', () {
      final tensor = TensorBuffer.zeros([4]);
      final op = RMSNormOp(normalizedShape: [2, 4]); // requires 2D normalized shape

      expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
    });

    test('throws on empty normalizedShape', () {
      expect(
        () => RMSNormOp(normalizedShape: []),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws on invalid normalizedShape dimension', () {
      expect(
        () => RMSNormOp(normalizedShape: [0]),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => RMSNormOp(normalizedShape: [-1]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws on weight length mismatch', () {
      expect(
        () => RMSNormOp(normalizedShape: [4], weight: [1.0, 2.0]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws on invalid eps', () {
      expect(
        () => RMSNormOp(normalizedShape: [4], eps: 0),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => RMSNormOp(normalizedShape: [4], eps: -1),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('llama7B factory creates correct op', () {
      final weight = List.generate(4096, (i) => 1.0);
      final op = RMSNormOp.llama7B(weight: weight);

      expect(op.normalizedShape, equals([4096]));
      expect(op.eps, equals(1e-6));
    });

    test('llama13B factory creates correct op', () {
      final weight = List.generate(5120, (i) => 1.0);
      final op = RMSNormOp.llama13B(weight: weight);

      expect(op.normalizedShape, equals([5120]));
    });

    test('llama70B factory creates correct op', () {
      final weight = List.generate(8192, (i) => 1.0);
      final op = RMSNormOp.llama70B(weight: weight);

      expect(op.normalizedShape, equals([8192]));
    });

    test('gemma2B factory creates correct op', () {
      final weight = List.generate(2048, (i) => 1.0);
      final op = RMSNormOp.gemma2B(weight: weight);

      expect(op.normalizedShape, equals([2048]));
    });

    test('fromStateDict factory works correctly', () {
      final weight = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);

      final op = RMSNormOp.fromStateDict(
        normalizedShape: [4],
        weight: weight,
        eps: 1e-5,
      );

      expect(op.normalizedShape, equals([4]));
      expect(op.weight, equals([1.0, 2.0, 3.0, 4.0]));
      expect(op.eps, equals(1e-5));
    });

    test('works with Float64 dtype', () {
      final data = Float64List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat64List(data, [4]);

      final op = RMSNormOp(normalizedShape: [4]);
      final result = op.apply(tensor);

      expect(result.dtype, equals(DType.float64));

      final expectedRms = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6);
      expect(result.storage.getAsDouble(0), closeTo(1.0 / expectedRms, 1e-10));
    });

    test('normalizedSize getter returns correct value', () {
      final op = RMSNormOp(normalizedShape: [2, 3, 4]);

      expect(op.normalizedSize, equals(24));
    });

    test('capabilities are correct', () {
      final op = RMSNormOp(normalizedShape: [4]);

      expect(op.capabilities.supportsInPlace, isTrue);
      expect(op.capabilities.requiresContiguous, isTrue);
    });

    test('name is descriptive', () {
      final op = RMSNormOp(normalizedShape: [4096], eps: 1e-6);

      expect(op.name, contains('RMSNorm'));
      expect(op.name, contains('4096'));
    });

    test('difference from LayerNorm: no mean subtraction', () {
      // RMSNorm differs from LayerNorm in that it doesn't subtract mean
      // For data with non-zero mean, results should differ
      final data = Float32List.fromList([10.0, 11.0, 12.0, 13.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final rmsOp = RMSNormOp(normalizedShape: [4]);
      final layerOp = LayerNormOp(normalizedShape: [4]);

      final rmsResult = rmsOp.apply(tensor);
      final layerResult = layerOp.apply(tensor);

      // Results should be different
      expect(
        rmsResult.storage.getAsDouble(0),
        isNot(closeTo(layerResult.storage.getAsDouble(0), 0.01)),
      );
    });
  });
}
