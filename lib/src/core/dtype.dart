import 'dart:typed_data';

enum DType {
  float32(1, 4, 'float32'),
  float64(11, 8, 'float64'),
  int8(3, 1, 'int8'),
  int16(5, 2, 'int16'),
  int32(6, 4, 'int32'),
  int64(7, 8, 'int64'),
  uint8(2, 1, 'uint8'),
  uint16(4, 2, 'uint16'),
  uint32(12, 4, 'uint32'),
  uint64(13, 8, 'uint64');

  final int onnxId;
  final int byteSize;
  final String typeName;

  const DType(this.onnxId, this.byteSize, this.typeName);

  TypedData createBuffer(int length) {
    return switch (this) {
      DType.float32 => Float32List(length),
      DType.float64 => Float64List(length),
      DType.int8 => Int8List(length),
      DType.int16 => Int16List(length),
      DType.int32 => Int32List(length),
      DType.int64 => Int64List(length),
      DType.uint8 => Uint8List(length),
      DType.uint16 => Uint16List(length),
      DType.uint32 => Uint32List(length),
      DType.uint64 => Uint64List(length),
    };
  }

  static DType fromTypedData(TypedData data) {
    return switch (data) {
      Float32List() => DType.float32,
      Float64List() => DType.float64,
      Int8List() => DType.int8,
      Int16List() => DType.int16,
      Int32List() => DType.int32,
      Int64List() => DType.int64,
      Uint8List() => DType.uint8,
      Uint16List() => DType.uint16,
      Uint32List() => DType.uint32,
      Uint64List() => DType.uint64,
      _ =>
        throw ArgumentError('Unsupported TypedData type: ${data.runtimeType}'),
    };
  }

  bool get isFloatingPoint => this == DType.float32 || this == DType.float64;

  bool get isInteger => !isFloatingPoint;

  bool get isSigned =>
      this == DType.float32 ||
      this == DType.float64 ||
      this == DType.int8 ||
      this == DType.int16 ||
      this == DType.int32 ||
      this == DType.int64;

  @override
  String toString() => typeName;
}
