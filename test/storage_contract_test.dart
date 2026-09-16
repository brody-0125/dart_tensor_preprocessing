import 'dart:typed_data';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('storage conversion has explicit clamp, wrap and truncation rules', () {
    final expected = <DType, List<num>>{
      DType.int8: [1, -1, 44, 0],
      DType.int16: [1, -1, 300, 0],
      DType.int32: [1, -1, 300, 65536],
      DType.int64: [1, -1, 300, 65536],
      DType.uint8: [1, 0, 255, 255],
      DType.uint16: [1, 0, 300, 65535],
      DType.uint32: [1, 4294967295, 300, 65536],
      DType.uint64: [1, -1, 300, 65536],
    };
    for (final entry in expected.entries) {
      final storage = TensorStorage(entry.key.createBuffer(4), entry.key);
      final values = [1.9, -1.9, 300.9, 65536.9];
      for (var i = 0; i < values.length; i++) {
        storage.setFromDouble(i, values[i]);
      }
      expect(storage.data, entry.value, reason: '${entry.key}');
      for (final value in [
        double.nan,
        double.infinity,
        double.negativeInfinity,
      ]) {
        expect(() => storage.setFromDouble(0, value), throwsUnsupportedError);
        expect(storage.data, entry.value, reason: 'failed write is unchanged');
      }
    }
  });
  test('storage clones bounded typed views without integer precision loss', () {
    for (final dtype in DType.values) {
      final data = dtype.createBuffer(5);
      final values = data as List<num>;
      final initial = [11.0, 21.0, 22.0, 23.0, 99.0];
      final full = TensorStorage(data, dtype);
      for (var i = 0; i < initial.length; i++) {
        full.setFromDouble(i, initial[i]);
      }
      if (dtype == DType.int64 || dtype == DType.uint64) {
        values[2] = 9007199254740993;
      }
      final bounded = switch (dtype) {
        DType.float32 => Float32List.sublistView(data, 1, 4),
        DType.float64 => Float64List.sublistView(data, 1, 4),
        DType.int8 => Int8List.sublistView(data, 1, 4),
        DType.int16 => Int16List.sublistView(data, 1, 4),
        DType.int32 => Int32List.sublistView(data, 1, 4),
        DType.int64 => Int64List.sublistView(data, 1, 4),
        DType.uint8 => Uint8List.sublistView(data, 1, 4),
        DType.uint16 => Uint16List.sublistView(data, 1, 4),
        DType.uint32 => Uint32List.sublistView(data, 1, 4),
        DType.uint64 => Uint64List.sublistView(data, 1, 4),
      };
      final storage = TensorStorage(bounded, dtype);
      final copy = storage.clone();
      expect(copy.data, values.sublist(1, 4));
      expect(copy.sizeInBytes, 3 * dtype.byteSize);
      copy.setFromDouble(0, 7);
      expect(values, [11, 21, values[2], 23, 99]);
      expect(
        () => storage.getAsDouble(-1),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
      expect(
        () => storage.setFromDouble(3, 0),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    }
  });
}
