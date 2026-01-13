import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/memory_format.dart';
import '../core/tensor_buffer.dart';

/// Utilities for working with TypedData views and zero-copy operations.
///
/// These utilities help determine when tensor operations can use views
/// instead of copying data, and provide efficient view creation methods.
class TypedDataViews {
  TypedDataViews._();

  /// Creates a sublist view of a Float32List without copying.
  ///
  /// This is a zero-copy operation that returns a view into the original data.
  static Float32List float32SublistView(Float32List data, int start,
      [int? end]) {
    return Float32List.sublistView(data, start, end);
  }

  /// Creates a sublist view of a Float64List without copying.
  static Float64List float64SublistView(Float64List data, int start,
      [int? end]) {
    return Float64List.sublistView(data, start, end);
  }

  /// Creates a sublist view of an Int32List without copying.
  static Int32List int32SublistView(Int32List data, int start, [int? end]) {
    return Int32List.sublistView(data, start, end);
  }

  /// Creates a sublist view of a Uint8List without copying.
  static Uint8List uint8SublistView(Uint8List data, int start, [int? end]) {
    return Uint8List.sublistView(data, start, end);
  }

  /// Creates a typed view from a ByteBuffer at a specific offset.
  ///
  /// This allows reinterpreting bytes as different types without copying.
  static TypedData viewAs(
    ByteBuffer buffer,
    DType dtype, {
    int offsetInBytes = 0,
    int? length,
  }) {
    switch (dtype) {
      case DType.float32:
        return buffer.asFloat32List(offsetInBytes, length);
      case DType.float64:
        return buffer.asFloat64List(offsetInBytes, length);
      case DType.int8:
        return buffer.asInt8List(offsetInBytes, length);
      case DType.int16:
        return buffer.asInt16List(offsetInBytes, length);
      case DType.int32:
        return buffer.asInt32List(offsetInBytes, length);
      case DType.int64:
        return buffer.asInt64List(offsetInBytes, length);
      case DType.uint8:
        return buffer.asUint8List(offsetInBytes, length);
      case DType.uint16:
        return buffer.asUint16List(offsetInBytes, length);
      case DType.uint32:
        return buffer.asUint32List(offsetInBytes, length);
      case DType.uint64:
        return buffer.asUint64List(offsetInBytes, length);
    }
  }
}

/// Extension methods for creating tensor views efficiently.
extension TensorViewExtension on TensorBuffer {
  /// Creates a view of a contiguous slice along the first dimension.
  ///
  /// This is a zero-copy operation when possible. Returns a view into
  /// the same underlying storage with adjusted offset and shape.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([10, 224, 224], dtype: DType.float32);
  /// final batch = tensor.sliceFirst(2, 5); // Elements 2..4 along dim 0
  /// // batch.shape is [3, 224, 224], shares storage with tensor
  /// ```
  ///
  /// Throws if the tensor is not contiguous.
  TensorBuffer sliceFirst(int start, int end) {
    if (!isContiguous) {
      throw StateError('sliceFirst requires contiguous tensor');
    }
    if (start < 0 || end > shape[0] || start >= end) {
      throw RangeError('Invalid slice range: [$start, $end) for shape $shape');
    }

    final newShape = [end - start, ...shape.skip(1)];
    final elementsPerSlice = shape.skip(1).fold(1, (a, b) => a * b);
    final newOffset = storageOffset + start * elementsPerSlice;

    return TensorBuffer(
      storage: storage,
      shape: newShape,
      storageOffset: newOffset,
      memoryFormat: memoryFormat,
    );
  }

  /// Checks if this tensor can be represented as a contiguous view
  /// of another tensor's storage.
  ///
  /// This is useful for determining whether operations can avoid copying.
  bool get isViewable => isContiguous && storageOffset == 0;

  /// Returns a view with channels moved to a different position.
  ///
  /// For a 4D NCHW tensor, converts to NHWC format without copying
  /// by adjusting strides.
  TensorBuffer toChannelsLast() {
    if (rank != 4) {
      throw StateError('toChannelsLast only supports 4D tensors');
    }
    // NCHW -> NHWC via transpose
    return transpose([0, 2, 3, 1]);
  }

  /// For a 4D NHWC tensor, converts to NCHW format without copying.
  TensorBuffer toChannelsFirst() {
    if (rank != 4) {
      throw StateError('toChannelsFirst only supports 4D tensors');
    }
    // NHWC -> NCHW via transpose
    return transpose([0, 3, 1, 2]);
  }

  /// Creates a flat view of this tensor.
  ///
  /// Returns a 1D tensor sharing the same storage.
  /// Throws if the tensor is not contiguous.
  TensorBuffer flatten() {
    if (!isContiguous) {
      throw StateError('flatten requires contiguous tensor');
    }
    return reshape([numel]);
  }

  /// Splits the tensor along a dimension into multiple views.
  ///
  /// Returns a list of tensors, each sharing storage with the original.
  /// This is useful for batch processing.
  ///
  /// ```dart
  /// final batch = TensorBuffer.zeros([8, 3, 224, 224]);
  /// final items = batch.unbind(0); // 8 tensors of shape [3, 224, 224]
  /// ```
  List<TensorBuffer> unbind(int dim) {
    if (dim < 0 || dim >= rank) {
      throw RangeError('dim $dim out of range for rank $rank');
    }

    final results = <TensorBuffer>[];
    final dimSize = shape[dim];

    for (int i = 0; i < dimSize; i++) {
      // Create a view for each index along dim
      final newShape = [...shape]..removeAt(dim);
      final newStrides = [...strides]..removeAt(dim);

      // Calculate offset for this slice
      final sliceOffset = storageOffset + i * strides[dim];

      results.add(TensorBuffer(
        storage: storage,
        shape: newShape,
        strides: newStrides,
        storageOffset: sliceOffset,
        memoryFormat: MemoryFormat.contiguous,
      ));
    }

    return results;
  }

  /// Selects a single index along a dimension, returning a view with
  /// reduced rank.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([10, 3, 224, 224]);
  /// final first = tensor.select(0, 0); // shape [3, 224, 224]
  /// ```
  TensorBuffer select(int dim, int index) {
    if (dim < 0 || dim >= rank) {
      throw RangeError('dim $dim out of range for rank $rank');
    }
    if (index < 0 || index >= shape[dim]) {
      throw RangeError('index $index out of range for dim $dim size ${shape[dim]}');
    }

    final newShape = [...shape]..removeAt(dim);
    final newStrides = [...strides]..removeAt(dim);
    final newOffset = storageOffset + index * strides[dim];

    return TensorBuffer(
      storage: storage,
      shape: newShape,
      strides: newStrides,
      storageOffset: newOffset,
      memoryFormat: memoryFormat,
    );
  }

  /// Narrows the tensor along a dimension.
  ///
  /// Creates a view that only includes elements from [start] to [start + length)
  /// along the specified dimension.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([10, 224, 224]);
  /// final narrowed = tensor.narrow(0, 2, 5); // shape [5, 224, 224]
  /// ```
  TensorBuffer narrow(int dim, int start, int length) {
    if (dim < 0 || dim >= rank) {
      throw RangeError('dim $dim out of range for rank $rank');
    }
    if (start < 0 || start + length > shape[dim]) {
      throw RangeError('Invalid narrow range: start=$start, length=$length, dim size=${shape[dim]}');
    }

    final newShape = [...shape];
    newShape[dim] = length;

    final newOffset = storageOffset + start * strides[dim];

    return TensorBuffer(
      storage: storage,
      shape: newShape,
      strides: strides,
      storageOffset: newOffset,
      memoryFormat: memoryFormat,
    );
  }
}
