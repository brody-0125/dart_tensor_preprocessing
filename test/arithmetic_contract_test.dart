import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('atan2 validates shape and snapshots overlapping input', () {
    final raw = Float64List.fromList([1, 1, 1, 1]);
    final storage = TensorStorage(raw, DType.float64);
    final x = TensorBuffer(storage: storage, shape: [3], storageOffset: 1);
    Atan2Op.tensor(TensorBuffer(storage: storage, shape: [3])).applyInPlace(x);
    for (final v in raw.skip(1)) {
      expect(v, closeTo(0.7853981633974483, 1e-15));
    }
    final op = Atan2Op.tensor(TensorBuffer.ones([3, 2]));
    expect(
      () => op(TensorBuffer.ones([2, 3])),
      throwsA(isA<ShapeMismatchException>()),
    );
    expect(
      () => op.computeOutputShape([2, 3]),
      throwsA(isA<ShapeMismatchException>()),
    );
  });

  test('clip rejects NaN bounds', () {
    expect(
      () => ClipOp(min: double.nan, max: 1),
      throwsA(isA<InvalidParameterException>()),
    );
    expect(
      () => ClipOp(min: 0, max: double.nan),
      throwsA(isA<InvalidParameterException>()),
    );
  });

  test('integer zero divisors are rejected before in-place mutation', () {
    final x = TensorBuffer(
      storage: TensorStorage(Int64List.fromList([9, 7, 5]), DType.int64),
      shape: [3],
    );
    final other = TensorBuffer(
      storage: TensorStorage(Int64List.fromList([3, 0, 1]), DType.int64),
      shape: [3],
    );
    expect(
      () => DivOp.tensor(other).applyInPlace(x),
      throwsA(isA<InvalidParameterException>()),
    );
    expect(x.storage.data, [9, 7, 5]);
    expect(
      () => DivOp(scalar: 0).applyInPlace(x),
      throwsA(isA<InvalidParameterException>()),
    );
    expect(x.storage.data, [9, 7, 5]);
  });

  test('binary tensor shapes must match, not just their element counts', () {
    final x = TensorBuffer.ones([2, 3]);
    final other = TensorBuffer.ones([3, 2]);
    for (final op in [
      AddOp.tensor(other),
      SubOp.tensor(other),
      MulOp.tensor(other),
      DivOp.tensor(other),
    ]) {
      expect(() => op(x), throwsA(isA<ShapeMismatchException>()));
      expect(() => op.applyInPlace(x), throwsA(isA<ShapeMismatchException>()));
      expect(
        () => op.computeOutputShape(x.shape),
        throwsA(isA<ShapeMismatchException>()),
      );
      expect(x.toList(), [1, 1, 1, 1, 1, 1]);
    }
  });

  test('in-place arithmetic snapshots overlapping tensor operands', () {
    final raw = Float64List.fromList([1, 2, 3, 4]);
    final storage = TensorStorage(raw, DType.float64);
    final x = TensorBuffer(storage: storage, shape: [3], storageOffset: 1);
    final other = TensorBuffer(storage: storage, shape: [3]);
    AddOp.tensor(other).applyInPlace(x);
    expect(raw, [1, 3, 5, 7]);
  });
}
