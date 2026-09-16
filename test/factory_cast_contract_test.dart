import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('sequence factories reject empty and non-finite ranges', () {
    for (final create in <TensorBuffer Function()>[
      () => TensorBuffer.arange(start: 1, end: 1),
      () => TensorBuffer.arange(start: 0, end: 1, step: 0),
      () => TensorBuffer.arange(start: 0, end: 1, step: -1),
      () => TensorBuffer.arange(start: double.nan, end: 1),
      () => TensorBuffer.arange(start: 0, end: double.infinity),
      () => TensorBuffer.arange(start: 0, end: 1, step: double.infinity),
      () => TensorBuffer.linspace(0, 1, steps: 0),
      () => TensorBuffer.linspace(0, double.infinity, steps: 2),
      () => TensorBuffer.eye(0),
      () => TensorBuffer.eye(2, m: 0),
    ]) {
      expect(create, throwsA(isA<InvalidParameterException>()));
    }
  });

  test('same dtype cast preserves view identity and storage alias', () {
    final base = TensorBuffer.fromFloat64List(
      Float64List.fromList([1, 2, 3, 4]),
      [2, 2],
    );
    final view = base.transpose([1, 0]);
    final result = TypeCastOp.toFloat64()(view);
    expect(identical(result, view), isTrue);
    result.storage.setFromDouble(result.strides[1], 9);
    expect(base[[1, 0]], 9);
  });

  test('integer cast uses all 64 bits before destination wrapping', () {
    final input = TensorBuffer(
      storage: TensorStorage(
        Int64List.fromList([9007199254740993, -9007199254740993]),
        DType.int64,
      ),
      shape: [2],
    );
    expect(TypeCastOp.toInt32()(input).storage.data, [1, -1]);
    final unsigned = TypeCastOp(DType.uint64)(input);
    // Dart native integers are signed 64-bit; unsigned storage retains the bits.
    expect(TypeCastOp.toInt64()(unsigned).storage.data, input.storage.data);
  });
}
