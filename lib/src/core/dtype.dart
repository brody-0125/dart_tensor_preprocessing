import 'dart:typed_data';

/// Represents the data type of tensor elements.
///
/// Each [DType] maps directly to an ONNX TensorProto.DataType, making it
/// suitable for interoperability with ONNX Runtime.
///
/// ```dart
/// final dtype = DType.float32;
/// print(dtype.onnxId);    // 1
/// print(dtype.byteSize);  // 4
/// ```
enum DType {
  /// 32-bit floating point (ONNX type 1).
  float32(1, 4, 'float32'),

  /// 64-bit floating point (ONNX type 11).
  float64(11, 8, 'float64'),

  /// 8-bit signed integer (ONNX type 3).
  int8(3, 1, 'int8'),

  /// 16-bit signed integer (ONNX type 5).
  int16(5, 2, 'int16'),

  /// 32-bit signed integer (ONNX type 6).
  int32(6, 4, 'int32'),

  /// 64-bit signed integer (ONNX type 7).
  int64(7, 8, 'int64'),

  /// 8-bit unsigned integer (ONNX type 2).
  uint8(2, 1, 'uint8'),

  /// 16-bit unsigned integer (ONNX type 4).
  uint16(4, 2, 'uint16'),

  /// 32-bit unsigned integer (ONNX type 12).
  uint32(12, 4, 'uint32'),

  /// 64-bit unsigned integer (ONNX type 13).
  uint64(13, 8, 'uint64');

  /// The ONNX TensorProto.DataType value.
  final int onnxId;

  /// The size of one element in bytes.
  final int byteSize;

  /// The human-readable name of this type.
  final String typeName;

  const DType(this.onnxId, this.byteSize, this.typeName);

  /// Creates a new typed buffer with the given [length] for this dtype.
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

  /// Infers the [DType] from a [TypedData] instance.
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

  /// Whether this is a floating-point type.
  bool get isFloatingPoint => this == DType.float32 || this == DType.float64;

  /// Whether this is an integer type.
  bool get isInteger => !isFloatingPoint;

  /// Whether this type supports negative values.
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
