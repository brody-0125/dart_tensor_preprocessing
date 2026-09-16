import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  for (final dtype in [DType.int64, DType.uint64]) {
    test('$dtype clone and strided copies preserve exact integers', () {
      const values = [
        9007199254740993,
        9007199254740995,
        9223372036854775807,
        9007199254740997,
      ];
      final data = dtype.createBuffer(6) as List<int>;
      data.setAll(0, [17, ...values, 19]);
      final input = TensorBuffer(
        storage: TensorStorage(data as TypedData, dtype),
        shape: [2, 2],
        storageOffset: 1,
      );
      final cloned = input.clone();
      expect(cloned.storage.data as List<int>, values);
      final transposed = input.transpose([1, 0]);
      for (final copied in [transposed.clone(), transposed.contiguous()]) {
        expect(copied.storage.data as List<int>, [
          values[0],
          values[2],
          values[1],
          values[3],
        ]);
        expect(copied.shape, [2, 2]);
        expect(copied.dtype, dtype);
      }
      (cloned.storage.data as List<int>)[0] = 0;
      expect(data, [17, ...values, 19]);
    });
  }
}
