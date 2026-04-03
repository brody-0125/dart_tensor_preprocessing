import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('InstanceNormOp', () {
    test('normalizes 3D tensor per channel', () {
      // Create [2, 2, 2] tensor with known values per channel
      // Channel 0: [0, 2, 4, 6] -> mean=3, var=5
      // Channel 1: [1, 3, 5, 7] -> mean=4, var=5
      final data = Float32List.fromList(
        [0, 2, 4, 6, 1, 3, 5, 7].map((e) => e.toDouble()).toList(),
      );
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);

      final op = InstanceNormOp(numFeatures: 2);
      final result = op.apply(tensor);

      // After normalization, each channel should have mean≈0, std≈1
      expect(result.shape, equals([2, 2, 2]));

      // Check channel 0 is normalized
      final ch0Values = [
        result.storage.getAsDouble(0),
        result.storage.getAsDouble(1),
        result.storage.getAsDouble(2),
        result.storage.getAsDouble(3),
      ];
      final ch0Mean = ch0Values.reduce((a, b) => a + b) / ch0Values.length;
      expect(ch0Mean, closeTo(0.0, 1e-5));

      // Check channel 1 is normalized
      final ch1Values = [
        result.storage.getAsDouble(4),
        result.storage.getAsDouble(5),
        result.storage.getAsDouble(6),
        result.storage.getAsDouble(7),
      ];
      final ch1Mean = ch1Values.reduce((a, b) => a + b) / ch1Values.length;
      expect(ch1Mean, closeTo(0.0, 1e-5));
    });

    test('normalizes 4D tensor per sample per channel', () {
      // Create [2, 2, 2, 2] tensor (batch=2, channels=2, H=2, W=2)
      final data = Float32List.fromList(List.generate(16, (i) => i.toDouble()));
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2, 2]);

      final op = InstanceNormOp(numFeatures: 2);
      final result = op.apply(tensor);

      expect(result.shape, equals([2, 2, 2, 2]));

      // Each (batch, channel) group should have mean≈0
      // Batch 0, Channel 0: indices 0-3
      final b0c0Values = [0, 1, 2, 3].map((i) => result.storage.getAsDouble(i));
      final b0c0Mean = b0c0Values.reduce((a, b) => a + b) / 4;
      expect(b0c0Mean, closeTo(0.0, 1e-5));

      // Batch 1, Channel 1: indices 12-15
      final b1c1Values = [
        12,
        13,
        14,
        15,
      ].map((i) => result.storage.getAsDouble(i));
      final b1c1Mean = b1c1Values.reduce((a, b) => a + b) / 4;
      expect(b1c1Mean, closeTo(0.0, 1e-5));
    });

    test('applies affine transformation with weight and bias', () {
      final data = Float32List.fromList(
        [0, 2, 4, 6, 1, 3, 5, 7].map((e) => e.toDouble()).toList(),
      );
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);

      final op = InstanceNormOp(
        numFeatures: 2,
        weight: [2.0, 0.5],
        bias: [1.0, -1.0],
      );
      final result = op.apply(tensor);

      // Channel 0 should be scaled by 2.0 and shifted by 1.0
      // Channel 1 should be scaled by 0.5 and shifted by -1.0
      expect(result.shape, equals([2, 2, 2]));

      // Mean of channel 0 should be close to bias (1.0)
      final ch0Values = [0, 1, 2, 3].map((i) => result.storage.getAsDouble(i));
      final ch0Mean = ch0Values.reduce((a, b) => a + b) / 4;
      expect(ch0Mean, closeTo(1.0, 1e-5));

      // Mean of channel 1 should be close to bias (-1.0)
      final ch1Values = [4, 5, 6, 7].map((i) => result.storage.getAsDouble(i));
      final ch1Mean = ch1Values.reduce((a, b) => a + b) / 4;
      expect(ch1Mean, closeTo(-1.0, 1e-5));
    });

    test('applies in-place modification', () {
      final data = Float32List.fromList(
        [0, 2, 4, 6, 1, 3, 5, 7].map((e) => e.toDouble()).toList(),
      );
      final tensor = TensorBuffer.fromFloat32List(data, [2, 2, 2]);
      final original = tensor.storage.getAsDouble(0);

      final op = InstanceNormOp(numFeatures: 2);
      op.applyInPlace(tensor);

      // Value should have changed
      expect(tensor.storage.getAsDouble(0), isNot(equals(original)));
    });

    test('throws on invalid rank', () {
      final tensor = TensorBuffer.zeros([10]);

      final op = InstanceNormOp(numFeatures: 10);

      expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
    });

    test('throws on channel mismatch', () {
      final tensor = TensorBuffer.zeros([4, 8, 8]);

      final op = InstanceNormOp(numFeatures: 3);

      expect(() => op.apply(tensor), throwsA(isA<ShapeMismatchException>()));
    });

    test('throws on invalid numFeatures', () {
      expect(
        () => InstanceNormOp(numFeatures: 0),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => InstanceNormOp(numFeatures: -1),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws on weight length mismatch', () {
      expect(
        () => InstanceNormOp(numFeatures: 3, weight: [1.0, 2.0]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws on bias length mismatch', () {
      expect(
        () => InstanceNormOp(numFeatures: 3, bias: [1.0, 2.0]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('fromStateDict factory works correctly', () {
      final weight = Float32List.fromList([1.0, 2.0, 3.0]);
      final bias = Float32List.fromList([0.1, 0.2, 0.3]);

      final op = InstanceNormOp.fromStateDict(
        numFeatures: 3,
        weight: weight,
        bias: bias,
        eps: 1e-6,
      );

      expect(op.numFeatures, equals(3));
      expect(op.weight!.length, equals(3));
      expect(op.bias!.length, equals(3));
      expect(op.eps, equals(1e-6));
    });

    test('works with Float64 dtype', () {
      final data = Float64List.fromList(
        [0, 2, 4, 6, 1, 3, 5, 7].map((e) => e.toDouble()).toList(),
      );
      final tensor = TensorBuffer.fromFloat64List(data, [2, 2, 2]);

      final op = InstanceNormOp(numFeatures: 2);
      final result = op.apply(tensor);

      expect(result.dtype, equals(DType.float64));

      // Check normalization
      final ch0Values = [0, 1, 2, 3].map((i) => result.storage.getAsDouble(i));
      final ch0Mean = ch0Values.reduce((a, b) => a + b) / 4;
      expect(ch0Mean, closeTo(0.0, 1e-10));
    });

    test('capabilities are correct', () {
      final op = InstanceNormOp(numFeatures: 3);

      expect(op.capabilities.supportsInPlace, isTrue);
      expect(op.capabilities.requiresContiguous, isTrue);
    });

    test('name is descriptive', () {
      final op = InstanceNormOp(numFeatures: 64, eps: 1e-6);

      expect(op.name, contains('InstanceNorm'));
      expect(op.name, contains('64'));
    });
  });
}
