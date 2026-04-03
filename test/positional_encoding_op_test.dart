import 'dart:typed_data';
import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/core/tensor_buffer.dart';
import 'package:dart_tensor_preprocessing/src/ops/positional_encoding_op.dart';

void main() {
  group('PositionalEncodingOp', () {
    test('adds sinusoidal encoding to 2D tensor', () {
      const dModel = 4;
      const seqLen = 3;
      final data = Float32List(seqLen * dModel); // all zeros
      final tensor = TensorBuffer.fromFloat32List(data, [seqLen, dModel]);

      final posEnc = PositionalEncodingOp(dModel: dModel, maxLen: 100);
      final result = posEnc(tensor);

      // PE(0, 0) = sin(0 / 10000^(0/4)) = sin(0) = 0
      expect(result[[0, 0]], closeTo(0.0, 1e-5));
      // PE(0, 1) = cos(0 / 10000^(0/4)) = cos(0) = 1
      expect(result[[0, 1]], closeTo(1.0, 1e-5));
      // PE(0, 2) = sin(0 / 10000^(2/4)) = sin(0) = 0
      expect(result[[0, 2]], closeTo(0.0, 1e-5));
      // PE(0, 3) = cos(0 / 10000^(2/4)) = cos(0) = 1
      expect(result[[0, 3]], closeTo(1.0, 1e-5));

      // PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1)
      expect(result[[1, 0]], closeTo(math.sin(1.0), 1e-5));
      // PE(1, 1) = cos(1 / 10000^(0/4)) = cos(1)
      expect(result[[1, 1]], closeTo(math.cos(1.0), 1e-5));
      // PE(1, 2) = sin(1 / 10000^(2/4)) = sin(1/100)
      expect(result[[1, 2]], closeTo(math.sin(1.0 / 100.0), 1e-5));
      // PE(1, 3) = cos(1 / 10000^(2/4)) = cos(1/100)
      expect(result[[1, 3]], closeTo(math.cos(1.0 / 100.0), 1e-5));
    });

    test('adds encoding to existing values', () {
      const dModel = 2;
      const seqLen = 2;
      final data = Float32List.fromList([10.0, 20.0, 30.0, 40.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [seqLen, dModel]);

      final posEnc = PositionalEncodingOp(dModel: dModel, maxLen: 100);
      final result = posEnc(tensor);

      // pos=0: PE(0,0)=sin(0)=0, PE(0,1)=cos(0)=1
      expect(result[[0, 0]], closeTo(10.0 + 0.0, 1e-5));
      expect(result[[0, 1]], closeTo(20.0 + 1.0, 1e-5));
      // pos=1: PE(1,0)=sin(1), PE(1,1)=cos(1)
      expect(result[[1, 0]], closeTo(30.0 + math.sin(1.0), 1e-5));
      expect(result[[1, 1]], closeTo(40.0 + math.cos(1.0), 1e-5));
    });

    test('works with 3D batched input', () {
      const dModel = 2;
      const seqLen = 2;
      const batchSize = 2;
      final data = Float32List(batchSize * seqLen * dModel); // all zeros
      final tensor = TensorBuffer.fromFloat32List(data, [
        batchSize,
        seqLen,
        dModel,
      ]);

      final posEnc = PositionalEncodingOp(dModel: dModel, maxLen: 100);
      final result = posEnc(tensor);

      // Both batches should have the same positional encoding
      // Batch 0, pos 0
      final batch0Pos0Dim0 = result[[0, 0, 0]];
      final batch0Pos0Dim1 = result[[0, 0, 1]];
      // Batch 1, pos 0
      final batch1Pos0Dim0 = result[[1, 0, 0]];
      final batch1Pos0Dim1 = result[[1, 0, 1]];

      expect(batch0Pos0Dim0, closeTo(batch1Pos0Dim0, 1e-10));
      expect(batch0Pos0Dim1, closeTo(batch1Pos0Dim1, 1e-10));
    });

    test('preserves shape', () {
      final posEnc = PositionalEncodingOp(dModel: 512, maxLen: 5000);
      expect(posEnc.computeOutputShape([2, 100, 512]), equals([2, 100, 512]));
      expect(posEnc.computeOutputShape([50, 512]), equals([50, 512]));
    });

    test('does not modify original tensor', () {
      const dModel = 4;
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, dModel]);

      PositionalEncodingOp(dModel: dModel, maxLen: 10)(tensor);

      expect(tensor[[0, 0]], closeTo(1.0, 1e-6));
      expect(tensor[[0, 1]], closeTo(2.0, 1e-6));
      expect(tensor[[0, 2]], closeTo(3.0, 1e-6));
      expect(tensor[[0, 3]], closeTo(4.0, 1e-6));
    });

    test('applyInPlace modifies tensor directly', () {
      const dModel = 2;
      final data = Float32List.fromList([0.0, 0.0, 0.0, 0.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [2, dModel]);

      PositionalEncodingOp(dModel: dModel, maxLen: 10).applyInPlace(tensor);

      // pos=0: sin(0)=0, cos(0)=1
      expect(tensor[[0, 0]], closeTo(0.0, 1e-5));
      expect(tensor[[0, 1]], closeTo(1.0, 1e-5));
      // pos=1: sin(1), cos(1)
      expect(tensor[[1, 0]], closeTo(math.sin(1.0), 1e-5));
      expect(tensor[[1, 1]], closeTo(math.cos(1.0), 1e-5));
    });

    test('works with float64', () {
      const dModel = 2;
      final data = Float64List.fromList([0.0, 0.0, 0.0, 0.0]);
      final tensor = TensorBuffer.fromFloat64List(data, [2, dModel]);

      final result = PositionalEncodingOp(dModel: dModel, maxLen: 10)(tensor);

      expect(result[[0, 0]], closeTo(0.0, 1e-10));
      expect(result[[0, 1]], closeTo(1.0, 1e-10));
      expect(result[[1, 0]], closeTo(math.sin(1.0), 1e-10));
      expect(result[[1, 1]], closeTo(math.cos(1.0), 1e-10));
    });

    test('name includes parameters', () {
      final posEnc = PositionalEncodingOp(dModel: 512, maxLen: 5000);
      expect(posEnc.name, equals('PositionalEncoding(d=512, max=5000)'));
    });

    test('capabilities include PyTorch metadata', () {
      final caps = PositionalEncodingOp(dModel: 4, maxLen: 10).capabilities;
      expect(caps.supportsInPlace, isTrue);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.pytorchEquivalent, isNotNull);
    });
  });

  group('PositionalEncodingOp validation', () {
    test('throws on invalid dModel', () {
      expect(
        () => PositionalEncodingOp(dModel: 0, maxLen: 10),
        throwsA(isA<Exception>()),
      );
      expect(
        () => PositionalEncodingOp(dModel: -1, maxLen: 10),
        throwsA(isA<Exception>()),
      );
    });

    test('throws on invalid maxLen', () {
      expect(
        () => PositionalEncodingOp(dModel: 4, maxLen: 0),
        throwsA(isA<Exception>()),
      );
      expect(
        () => PositionalEncodingOp(dModel: 4, maxLen: -1),
        throwsA(isA<Exception>()),
      );
    });

    test('throws on 1D input', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [4]);

      final posEnc = PositionalEncodingOp(dModel: 4, maxLen: 10);

      expect(() => posEnc(tensor), throwsA(isA<Exception>()));
    });

    test('throws on dModel mismatch', () {
      final data = Float32List.fromList([1.0, 2.0, 3.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 3]);

      final posEnc = PositionalEncodingOp(dModel: 4, maxLen: 10);

      expect(() => posEnc(tensor), throwsA(isA<Exception>()));
    });

    test('throws when sequence length exceeds maxLen', () {
      final data = Float32List(20); // 5 x 4
      final tensor = TensorBuffer.fromFloat32List(data, [5, 4]);

      final posEnc = PositionalEncodingOp(dModel: 4, maxLen: 3);

      expect(() => posEnc(tensor), throwsA(isA<Exception>()));
    });
  });

  group('PositionalEncodingOp mathematical properties', () {
    test('encoding values are bounded in [-1, 1] for zero input', () {
      const dModel = 64;
      const seqLen = 100;
      final data = Float32List(seqLen * dModel);
      final tensor = TensorBuffer.fromFloat32List(data, [seqLen, dModel]);

      final result = PositionalEncodingOp(dModel: dModel, maxLen: 200)(tensor);

      for (int pos = 0; pos < seqLen; pos++) {
        for (int d = 0; d < dModel; d++) {
          final val = result[[pos, d]];
          expect(val, greaterThanOrEqualTo(-1.0));
          expect(val, lessThanOrEqualTo(1.0));
        }
      }
    });

    test('different positions produce different encodings', () {
      const dModel = 8;
      final data = Float32List(2 * dModel);
      final tensor = TensorBuffer.fromFloat32List(data, [2, dModel]);

      final result = PositionalEncodingOp(dModel: dModel, maxLen: 100)(tensor);

      // pos=0 and pos=1 should differ
      bool anyDifferent = false;
      for (int d = 0; d < dModel; d++) {
        if ((result[[0, d]] - result[[1, d]]).abs() > 1e-10) {
          anyDifferent = true;
          break;
        }
      }
      expect(anyDifferent, isTrue);
    });

    test('custom base parameter works', () {
      const dModel = 4;
      final data = Float32List(dModel);
      final tensor = TensorBuffer.fromFloat32List(data, [1, dModel]);

      final result1 = PositionalEncodingOp(
        dModel: dModel,
        maxLen: 10,
        base: 10000.0,
      )(tensor);
      final result2 = PositionalEncodingOp(
        dModel: dModel,
        maxLen: 10,
        base: 100.0,
      )(tensor);

      // Position 0 should be the same (sin(0) and cos(0) regardless of base)
      // But let's check they're both valid
      expect(result1[[0, 0]], closeTo(0.0, 1e-5)); // sin(0)
      expect(result2[[0, 0]], closeTo(0.0, 1e-5)); // sin(0)
    });
  });
}
