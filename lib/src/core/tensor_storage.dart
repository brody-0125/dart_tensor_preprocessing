import 'dart:typed_data';

import 'dtype.dart';

/// Immutable wrapper around typed data that provides the physical storage
/// for tensor elements.
///
/// [TensorStorage] holds the raw data buffer and its [DType], ensuring
/// type safety when accessing elements. Multiple [TensorBuffer] instances
/// can share the same storage for zero-copy operations like transpose.
///
/// ## Example
///
/// ```dart
/// final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
/// final storage = TensorStorage.fromFloat32List(data);
///
/// print(storage.dtype);   // DType.float32
/// print(storage.length);  // 4
/// print(storage.getAsDouble(0));  // 1.0
/// ```
class TensorStorage {
  final TypedData _data;

  /// The data type of elements in this storage.
  final DType dtype;

  /// Creates a [TensorStorage] from raw [TypedData] and its [dtype].
  ///
  /// Throws [ArgumentError] if [dtype] does not match the actual type
  /// of the data.
  TensorStorage(this._data, this.dtype) {
    final inferredDtype = DType.fromTypedData(_data);
    if (inferredDtype != dtype) {
      throw ArgumentError(
        'DType mismatch: expected $dtype, but data is $inferredDtype',
      );
    }
  }

  /// Creates a [TensorStorage] from a [Float32List].
  factory TensorStorage.fromFloat32List(Float32List data) {
    return TensorStorage(data, DType.float32);
  }

  /// Creates a [TensorStorage] from a [Uint8List].
  factory TensorStorage.fromUint8List(Uint8List data) {
    return TensorStorage(data, DType.uint8);
  }

  /// The number of elements in this storage.
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

  /// The total size in bytes of the data in this storage.
  int get sizeInBytes => length * dtype.byteSize;

  /// The underlying typed data buffer.
  TypedData get data => _data;

  /// Returns the element at [index] as a double.
  ///
  /// Integer values are converted to double. Throws [RangeError] if [index]
  /// is out of bounds.
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

  /// Sets the element at [index] from a double [value].
  ///
  /// The value is converted to the storage's dtype. For unsigned types,
  /// values are clamped to valid ranges. Throws [RangeError] if [index]
  /// is out of bounds.
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

  /// Creates a deep copy of this storage with its own data buffer.
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
