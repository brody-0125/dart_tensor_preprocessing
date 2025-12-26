import 'dart:typed_data';

import 'dtype.dart';

class TensorStorage {
  final TypedData _data;
  final DType dtype;

  TensorStorage(this._data, this.dtype) {
    final inferredDtype = DType.fromTypedData(_data);
    if (inferredDtype != dtype) {
      throw ArgumentError(
        'DType mismatch: expected $dtype, but data is $inferredDtype',
      );
    }
  }

  factory TensorStorage.fromFloat32List(Float32List data) {
    return TensorStorage(data, DType.float32);
  }

  factory TensorStorage.fromUint8List(Uint8List data) {
    return TensorStorage(data, DType.uint8);
  }

  int get length {
    return switch (_data) {
      final Float32List list => list.length,
      final Float64List list => list.length,
      final Int8List list => list.length,
      final Int16List list => list.length,
      final Int32List list => list.length,
      final Int64List list => list.length,
      final Uint8List list => list.length,
      final Uint16List list => list.length,
      final Uint32List list => list.length,
      final Uint64List list => list.length,
      _ => throw StateError('Unknown TypedData type'),
    };
  }

  int get sizeInBytes => length * dtype.byteSize;

  TypedData get data => _data;

  double getAsDouble(int index) {
    _checkBounds(index);
    return switch (_data) {
      final Float32List list => list[index],
      final Float64List list => list[index],
      final Int8List list => list[index].toDouble(),
      final Int16List list => list[index].toDouble(),
      final Int32List list => list[index].toDouble(),
      final Int64List list => list[index].toDouble(),
      final Uint8List list => list[index].toDouble(),
      final Uint16List list => list[index].toDouble(),
      final Uint32List list => list[index].toDouble(),
      final Uint64List list => list[index].toDouble(),
      _ => throw StateError('Unknown TypedData type'),
    };
  }

  void setFromDouble(int index, double value) {
    _checkBounds(index);
    switch (_data) {
      case final Float32List list:
        list[index] = value;
      case final Float64List list:
        list[index] = value;
      case final Int8List list:
        list[index] = value.toInt();
      case final Int16List list:
        list[index] = value.toInt();
      case final Int32List list:
        list[index] = value.toInt();
      case final Int64List list:
        list[index] = value.toInt();
      case final Uint8List list:
        list[index] = value.toInt().clamp(0, 255);
      case final Uint16List list:
        list[index] = value.toInt().clamp(0, 65535);
      case final Uint32List list:
        list[index] = value.toInt();
      case final Uint64List list:
        list[index] = value.toInt();
      default:
        throw StateError('Unknown TypedData type');
    }
  }

  void _checkBounds(int index) {
    if (index < 0 || index >= length) {
      throw RangeError.index(index, this, 'index', null, length);
    }
  }

  TensorStorage clone() {
    final newData = dtype.createBuffer(length);
    _copyData(newData);
    return TensorStorage(newData, dtype);
  }

  void _copyData(TypedData dest) {
    switch (_data) {
      case final Float32List src:
        (dest as Float32List).setAll(0, src);
      case final Float64List src:
        (dest as Float64List).setAll(0, src);
      case final Int8List src:
        (dest as Int8List).setAll(0, src);
      case final Int16List src:
        (dest as Int16List).setAll(0, src);
      case final Int32List src:
        (dest as Int32List).setAll(0, src);
      case final Int64List src:
        (dest as Int64List).setAll(0, src);
      case final Uint8List src:
        (dest as Uint8List).setAll(0, src);
      case final Uint16List src:
        (dest as Uint16List).setAll(0, src);
      case final Uint32List src:
        (dest as Uint32List).setAll(0, src);
      case final Uint64List src:
        (dest as Uint64List).setAll(0, src);
    }
  }

  @override
  String toString() => 'TensorStorage(dtype: $dtype, length: $length)';
}
