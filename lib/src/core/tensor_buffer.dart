import 'dart:typed_data';

import 'dtype.dart';
import 'memory_format.dart';
import 'tensor_storage.dart';

/// A multi-dimensional tensor with shape, strides, and memory format metadata.
///
/// [TensorBuffer] is the primary data structure for tensor operations. It
/// provides a view over [TensorStorage] with shape and stride information,
/// enabling zero-copy operations like [transpose] and [squeeze].
///
/// ## Zero-copy operations
///
/// Operations that only modify metadata (not data) are O(1):
/// - [transpose]: Reorders dimensions by permuting strides
/// - [squeeze]: Removes dimensions of size 1
/// - [unsqueeze]: Adds a dimension of size 1
///
/// ## Copy operations
///
/// Operations that require a contiguous layout copy data:
/// - [reshape]: Requires contiguous input, call [contiguous] first if needed
/// - [clone]: Always creates a new copy
/// - [contiguous]: Copies only if not already contiguous
///
/// ## Example
///
/// ```dart
/// // Create a 2x3 tensor of zeros
/// final tensor = TensorBuffer.zeros([2, 3]);
///
/// // Access element at position [1, 2]
/// print(tensor[[1, 2]]);  // 0.0
///
/// // Transpose (zero-copy)
/// final transposed = tensor.transpose([1, 0]);
/// print(transposed.shape);  // [3, 2]
/// ```
class TensorBuffer {
  /// The underlying storage containing the tensor data.
  final TensorStorage storage;

  /// The dimensions of this tensor.
  final List<int> shape;

  /// The stride for each dimension.
  ///
  /// The stride indicates how many elements to skip in the storage
  /// to move one position along that dimension.
  final List<int> strides;

  /// The offset into storage where this tensor's data begins.
  final int storageOffset;

  /// The memory layout format of this tensor.
  final MemoryFormat memoryFormat;

  bool? _isContiguousCache;

  /// Creates a [TensorBuffer] with the given [storage] and [shape].
  ///
  /// If [strides] is not provided, it is computed based on [shape] and
  /// [memoryFormat]. Throws [ArgumentError] if [shape] is invalid.
  TensorBuffer({
    required this.storage,
    required this.shape,
    List<int>? strides,
    this.storageOffset = 0,
    this.memoryFormat = MemoryFormat.contiguous,
  }) : strides = strides ?? computeStrides(shape, memoryFormat) {
    _validateShape();
  }

  TensorBuffer._view({
    required this.storage,
    required this.shape,
    required this.strides,
    required this.storageOffset,
    required this.memoryFormat,
  });

  /// The data type of elements in this tensor.
  DType get dtype => storage.dtype;

  /// The number of dimensions (axes) in this tensor.
  int get rank => shape.length;

  /// The total number of elements in this tensor.
  int get numel => shape.fold(1, (a, b) => a * b);

  /// The total size in bytes of this tensor's data.
  int get sizeInBytes => numel * dtype.byteSize;

  /// Whether this tensor's data is stored contiguously in memory.
  ///
  /// A tensor is contiguous if its elements are stored in row-major order
  /// without gaps. Non-contiguous tensors result from operations like
  /// [transpose].
  bool get isContiguous {
    _isContiguousCache ??= _checkContiguity();
    return _isContiguousCache!;
  }

  bool _checkContiguity() {
    int expectedStride = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      if (shape[i] == 1) continue;
      if (strides[i] != expectedStride) return false;
      expectedStride *= shape[i];
    }
    return true;
  }

  /// Returns a view of this tensor with dimensions permuted according to [axes].
  ///
  /// This is a zero-copy operation that only changes strides. The [axes]
  /// list must be a permutation of `[0, 1, ..., rank-1]`.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3, 4]);
  /// final transposed = tensor.transpose([2, 0, 1]);
  /// print(transposed.shape);  // [4, 2, 3]
  /// ```
  ///
  /// Throws [ArgumentError] if [axes] length doesn't match [rank] or
  /// contains invalid or duplicate axis indices.
  TensorBuffer transpose(List<int> axes) {
    if (axes.length != rank) {
      throw ArgumentError(
        'axes length (${axes.length}) must match rank ($rank)',
      );
    }

    final seen = <int>{};
    for (final axis in axes) {
      if (axis < 0 || axis >= rank) {
        throw RangeError.range(axis, 0, rank - 1, 'axis');
      }
      if (!seen.add(axis)) {
        throw ArgumentError('Duplicate axis: $axis');
      }
    }

    return TensorBuffer._view(
      storage: storage,
      shape: [for (final a in axes) shape[a]],
      strides: [for (final a in axes) strides[a]],
      storageOffset: storageOffset,
      memoryFormat: memoryFormat,
    );
  }

  /// Returns a view of this tensor with a new shape.
  ///
  /// The tensor must be contiguous. The total number of elements must
  /// remain the same.
  ///
  /// Throws [StateError] if the tensor is not contiguous.
  /// Throws [ArgumentError] if [newShape] has a different total element count.
  TensorBuffer reshape(List<int> newShape) {
    final newNumel = newShape.fold(1, (a, b) => a * b);
    if (newNumel != numel) {
      throw ArgumentError(
        'Cannot reshape tensor of size $numel to $newShape (size $newNumel)',
      );
    }

    if (!isContiguous) {
      throw StateError(
        'Cannot reshape non-contiguous tensor. Call contiguous() first.',
      );
    }

    return TensorBuffer._view(
      storage: storage,
      shape: List.unmodifiable(newShape),
      strides: computeStrides(newShape, MemoryFormat.contiguous),
      storageOffset: storageOffset,
      memoryFormat: MemoryFormat.contiguous,
    );
  }

  /// Returns a view with dimensions of size 1 removed.
  ///
  /// If [dim] is provided, only that dimension is squeezed (if it has size 1).
  /// This is a zero-copy operation.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([1, 3, 1, 4]);
  /// print(tensor.squeeze().shape);     // [3, 4]
  /// print(tensor.squeeze(0).shape);    // [3, 1, 4]
  /// ```
  TensorBuffer squeeze([int? dim]) {
    final newShape = <int>[];
    final newStrides = <int>[];

    for (int i = 0; i < rank; i++) {
      if (dim != null) {
        if (i == dim && shape[i] == 1) continue;
      } else {
        if (shape[i] == 1) continue;
      }
      newShape.add(shape[i]);
      newStrides.add(strides[i]);
    }

    return TensorBuffer._view(
      storage: storage,
      shape: newShape,
      strides: newStrides,
      storageOffset: storageOffset,
      memoryFormat: memoryFormat,
    );
  }

  /// Returns a view with a dimension of size 1 inserted at [dim].
  ///
  /// This is a zero-copy operation.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([3, 4]);
  /// print(tensor.unsqueeze(0).shape);  // [1, 3, 4]
  /// print(tensor.unsqueeze(2).shape);  // [3, 4, 1]
  /// ```
  TensorBuffer unsqueeze(int dim) {
    if (dim < 0 || dim > rank) {
      throw RangeError.range(dim, 0, rank, 'dim');
    }

    final newShape = List<int>.from(shape);
    final newStrides = List<int>.from(strides);

    newShape.insert(dim, 1);
    final strideValue = dim < rank ? strides[dim] * shape[dim] : 1;
    newStrides.insert(dim, strideValue);

    return TensorBuffer._view(
      storage: storage,
      shape: newShape,
      strides: newStrides,
      storageOffset: storageOffset,
      memoryFormat: memoryFormat,
    );
  }

  /// Returns a contiguous copy of this tensor.
  ///
  /// If the tensor is already contiguous, returns itself without copying.
  TensorBuffer contiguous() {
    if (isContiguous) return this;

    final newData = dtype.createBuffer(numel);
    _copyToContiguous(newData);

    return TensorBuffer(
      storage: TensorStorage(newData, dtype),
      shape: shape.toList(),
      memoryFormat: MemoryFormat.contiguous,
    );
  }

  /// Creates a deep copy of this tensor.
  ///
  /// Always creates a new contiguous tensor, even if this tensor is
  /// already contiguous.
  TensorBuffer clone() {
    final newData = dtype.createBuffer(numel);
    _copyToContiguous(newData);

    return TensorBuffer(
      storage: TensorStorage(newData, dtype),
      shape: shape.toList(),
      memoryFormat: MemoryFormat.contiguous,
    );
  }

  void _copyToContiguous(TypedData dest) {
    final indices = List<int>.filled(rank, 0);
    for (int i = 0; i < numel; i++) {
      int srcOffset = storageOffset;
      for (int d = 0; d < rank; d++) {
        srcOffset += indices[d] * strides[d];
      }

      final value = storage.getAsDouble(srcOffset);
      _setTypedDataValue(dest, i, value);

      for (int d = rank - 1; d >= 0; d--) {
        indices[d]++;
        if (indices[d] < shape[d]) break;
        indices[d] = 0;
      }
    }
  }

  void _setTypedDataValue(TypedData data, int index, double value) {
    switch (data) {
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
    }
  }

  /// The underlying typed data buffer.
  ///
  /// Throws [StateError] if the tensor is not contiguous or has a
  /// non-zero storage offset.
  TypedData get data {
    if (!isContiguous) {
      throw StateError(
        'Tensor must be contiguous for direct data access. Call contiguous() first.',
      );
    }
    if (storageOffset != 0) {
      throw StateError(
        'Tensor with non-zero offset cannot provide direct data access.',
      );
    }
    return storage.data;
  }

  /// The underlying data as a [Float32List].
  ///
  /// Throws [StateError] if [dtype] is not [DType.float32] or if the
  /// tensor is not contiguous.
  Float32List get dataAsFloat32List {
    if (dtype != DType.float32) {
      throw StateError('Tensor dtype is $dtype, not float32');
    }
    return data as Float32List;
  }

  /// Returns the element at the given [indices].
  ///
  /// The [indices] list must have exactly [rank] elements.
  double operator [](List<int> indices) {
    if (indices.length != rank) {
      throw ArgumentError(
        'indices length (${indices.length}) must match rank ($rank)',
      );
    }

    int offset = storageOffset;
    for (int d = 0; d < rank; d++) {
      if (indices[d] < 0 || indices[d] >= shape[d]) {
        throw RangeError.range(indices[d], 0, shape[d] - 1, 'indices[$d]');
      }
      offset += indices[d] * strides[d];
    }

    return storage.getAsDouble(offset);
  }

  /// Creates a tensor filled with zeros.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3], dtype: DType.float32);
  /// ```
  static TensorBuffer zeros(
    List<int> shape, {
    DType dtype = DType.float32,
    MemoryFormat memoryFormat = MemoryFormat.contiguous,
  }) {
    final numel = shape.fold(1, (a, b) => a * b);
    final data = dtype.createBuffer(numel);
    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable(shape),
      memoryFormat: memoryFormat,
    );
  }

  /// Creates a tensor filled with ones.
  ///
  /// ## Example
  ///
  /// ```dart
  /// final tensor = TensorBuffer.ones([2, 3], dtype: DType.int32);
  /// ```
  static TensorBuffer ones(
    List<int> shape, {
    DType dtype = DType.float32,
  }) {
    final numel = shape.fold(1, (a, b) => a * b);
    final data = dtype.createBuffer(numel);

    for (int i = 0; i < numel; i++) {
      switch (data) {
        case final Float32List list:
          list[i] = 1.0;
        case final Float64List list:
          list[i] = 1.0;
        case final Int8List list:
          list[i] = 1;
        case final Int16List list:
          list[i] = 1;
        case final Int32List list:
          list[i] = 1;
        case final Int64List list:
          list[i] = 1;
        case final Uint8List list:
          list[i] = 1;
        case final Uint16List list:
          list[i] = 1;
        case final Uint32List list:
          list[i] = 1;
        case final Uint64List list:
          list[i] = 1;
      }
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable(shape),
    );
  }

  /// Creates a tensor from a [Float32List] with the given [shape].
  ///
  /// Throws [ArgumentError] if [data] length doesn't match the product of
  /// [shape] dimensions.
  static TensorBuffer fromFloat32List(Float32List data, List<int> shape) {
    final expectedNumel = shape.fold(1, (a, b) => a * b);
    if (data.length != expectedNumel) {
      throw ArgumentError(
        'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
      );
    }
    return TensorBuffer(
      storage: TensorStorage.fromFloat32List(data),
      shape: List.unmodifiable(shape),
    );
  }

  /// Creates a tensor from a [Uint8List] with the given [shape].
  ///
  /// Throws [ArgumentError] if [data] length doesn't match the product of
  /// [shape] dimensions.
  static TensorBuffer fromUint8List(Uint8List data, List<int> shape) {
    final expectedNumel = shape.fold(1, (a, b) => a * b);
    if (data.length != expectedNumel) {
      throw ArgumentError(
        'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
      );
    }
    return TensorBuffer(
      storage: TensorStorage.fromUint8List(data),
      shape: List.unmodifiable(shape),
    );
  }

  /// Computes strides for a tensor with the given [shape] and [format].
  static List<int> computeStrides(List<int> shape, MemoryFormat format) {
    final rank = shape.length;
    final strides = List<int>.filled(rank, 0);

    if (format == MemoryFormat.contiguous) {
      int stride = 1;
      for (int i = rank - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
      }
    } else {
      if (rank == 4) {
        final (_, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        strides[0] = h * w * c;
        strides[1] = 1;
        strides[2] = w * c;
        strides[3] = c;
      } else if (rank == 3) {
        final (c, _, w) = (shape[0], shape[1], shape[2]);
        strides[0] = 1;
        strides[1] = w * c;
        strides[2] = c;
      } else {
        throw UnsupportedError(
          'channelsLast format only supports 3D or 4D tensors, got ${rank}D',
        );
      }
    }

    return strides;
  }

  void _validateShape() {
    if (shape.isEmpty) {
      throw ArgumentError('Shape cannot be empty');
    }
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] <= 0) {
        throw ArgumentError(
            'Shape dimension must be positive, got ${shape[i]} at index $i');
      }
    }
  }

  @override
  String toString() {
    return 'TensorBuffer(shape: $shape, dtype: $dtype, '
        'strides: $strides, contiguous: $isContiguous)';
  }
}
