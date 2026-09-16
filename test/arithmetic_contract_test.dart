import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
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
