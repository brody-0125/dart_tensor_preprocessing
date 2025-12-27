import 'dart:typed_data';

import 'dtype.dart';
import 'memory_format.dart';
import 'tensor_storage.dart';

/// A multi-dimensional array view over typed data with shape and stride metadata.
///
/// [TensorBuffer] provides a NumPy-like interface for tensor operations. It
/// uses a view/storage separation pattern where [TensorStorage] holds the
/// physical data and [TensorBuffer] defines how to interpret it through
/// shape and stride information.
///
/// This design enables O(1) operations like [transpose] by manipulating
/// strides rather than copying data.
///
/// ```dart
/// // Create a 3x4 tensor filled with zeros
/// final tensor = TensorBuffer.zeros([3, 4], dtype: DType.float32);
///
/// // Transpose without copying data
/// final transposed = tensor.transpose([1, 0]); // Now 4x3
///
/// // Reshape (requires contiguous memory)
/// final reshaped = tensor.reshape([2, 6]);
/// ```
class TensorBuffer {
  /// The underlying storage containing the physical data.
  final TensorStorage storage;

  /// The dimensions of this tensor.
  final List<int> shape;

  /// The number of elements to skip in storage for each dimension.
  final List<int> strides;

  /// The offset into [storage] where this tensor's data begins.
  final int storageOffset;

  /// The memory layout format of this tensor.
  final MemoryFormat memoryFormat;

  bool? _isContiguousCache;

  /// Creates a new tensor buffer with the given [storage], [shape], and optional
  /// [strides].
  ///
  /// If [strides] is not provided, they are computed based on [shape] and
  /// [memoryFormat].
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

  /// The number of dimensions in this tensor.
  int get rank => shape.length;

  /// The total number of elements in this tensor.
  int get numel => shape.fold(1, (a, b) => a * b);

  /// The total size of this tensor's data in bytes.
  int get sizeInBytes => numel * dtype.byteSize;

  /// Whether this tensor's data is stored contiguously in memory.
  ///
  /// Contiguous tensors have elements stored in row-major order without gaps.
  /// Some operations like [reshape] require contiguous tensors.
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
  /// This is a zero-copy operation that only changes the strides.
  ///
  /// Throws [ArgumentError] if [axes] length does not match [rank] or contains
  /// invalid or duplicate axis indices.
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

  /// Returns a view of this tensor with a new [newShape].
  ///
  /// The total number of elements must remain the same. This tensor must be
  /// contiguous; call [contiguous] first if needed.
  ///
  /// Throws [StateError] if this tensor is not contiguous.
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

  /// Returns a view with all size-1 dimensions removed.
  ///
  /// If [dim] is specified, only that dimension is squeezed (if it has size 1).
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

  /// Returns a view with a size-1 dimension inserted at position [dim].
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

  /// Returns a contiguous copy of this tensor if not already contiguous.
  ///
  /// If [isContiguous] is true, returns this tensor unchanged.
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

  /// Creates a deep copy of this tensor with its own storage.
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

  /// The underlying typed data for direct access.
  ///
  /// Throws [StateError] if this tensor is not contiguous or has a non-zero
  /// storage offset.
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
  /// Throws [StateError] if [dtype] is not [DType.float32].
  Float32List get dataAsFloat32List {
    if (dtype != DType.float32) {
      throw StateError('Tensor dtype is $dtype, not float32');
    }
    return data as Float32List;
  }

  /// Returns the element at the given multi-dimensional [indices].
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

  /// Creates a tensor from an existing [Float32List] with the given [shape].
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

  /// Creates a tensor from an existing [Uint8List] with the given [shape].
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
