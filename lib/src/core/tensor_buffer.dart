import 'dart:typed_data';

import '../exceptions/tensor_exceptions.dart';
import 'dtype.dart';
import 'memory_format.dart';
import 'tensor_storage.dart';

part 'tensor_buffer_factory.dart';

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
  /// Throws [ShapeMismatchException] if [axes] length does not match [rank].
  /// Throws [IndexOutOfBoundsException] if axis is out of range.
  /// Throws [InvalidParameterException] if axes contain duplicates.
  TensorBuffer transpose(List<int> axes) {
    if (axes.length != rank) {
      throw ShapeMismatchException.rank(rank, axes.length);
    }

    final seen = <int>{};
    for (final axis in axes) {
      if (axis < 0 || axis >= rank) {
        throw IndexOutOfBoundsException(
          index: axis,
          min: 0,
          max: rank - 1,
          dimension: 'axis',
        );
      }
      if (!seen.add(axis)) {
        throw InvalidParameterException('axes', axes, 'Duplicate axis: $axis');
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
  /// Throws [ShapeMismatchException] if numel doesn't match.
  /// Throws [NonContiguousException] if this tensor is not contiguous.
  TensorBuffer reshape(List<int> newShape) {
    final newNumel = newShape.fold(1, (a, b) => a * b);
    if (newNumel != numel) {
      throw ShapeMismatchException(
        actual: newShape,
        message: 'Cannot reshape tensor of size $numel to $newShape (size $newNumel)',
      );
    }

    if (!isContiguous) {
      throw const NonContiguousException('reshape');
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
  ///
  /// Throws [IndexOutOfBoundsException] if [dim] is out of range.
  TensorBuffer unsqueeze(int dim) {
    if (dim < 0 || dim > rank) {
      throw IndexOutOfBoundsException(
        index: dim,
        min: 0,
        max: rank,
        dimension: 'dim',
      );
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
  /// Throws [NonContiguousException] if this tensor is not contiguous.
  /// Throws [InvalidParameterException] if tensor has non-zero storage offset.
  TypedData get data {
    if (!isContiguous) {
      throw const NonContiguousException('data access');
    }
    if (storageOffset != 0) {
      throw InvalidParameterException(
        'storageOffset',
        storageOffset,
        'Tensor with non-zero offset cannot provide direct data access',
      );
    }
    return storage.data;
  }

  /// The underlying data as a [Float32List].
  ///
  /// Throws [DTypeMismatchException] if [dtype] is not [DType.float32].
  Float32List get dataAsFloat32List {
    if (dtype != DType.float32) {
      throw DTypeMismatchException(expected: DType.float32, actual: dtype);
    }
    return data as Float32List;
  }

  /// Returns the element at the given multi-dimensional [indices].
  ///
  /// Throws [ShapeMismatchException] if indices length doesn't match rank.
  /// Throws [IndexOutOfBoundsException] if any index is out of bounds.
  double operator [](List<int> indices) {
    if (indices.length != rank) {
      throw ShapeMismatchException.rank(rank, indices.length);
    }

    int offset = storageOffset;
    for (int d = 0; d < rank; d++) {
      if (indices[d] < 0 || indices[d] >= shape[d]) {
        throw IndexOutOfBoundsException(
          index: indices[d],
          min: 0,
          max: shape[d] - 1,
          dimension: 'indices[$d]',
        );
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
  }) =>
      _zerosImpl(shape, dtype: dtype, memoryFormat: memoryFormat);

  /// Creates a tensor filled with ones.
  static TensorBuffer ones(
    List<int> shape, {
    DType dtype = DType.float32,
  }) =>
      _onesImpl(shape, dtype: dtype);

  /// Creates a tensor filled with a specific value.
  ///
  /// Equivalent to `torch.full()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.full([3, 3], fillValue: 5.0);
  /// ```
  static TensorBuffer full(
    List<int> shape, {
    required double fillValue,
    DType dtype = DType.float32,
  }) =>
      _fullImpl(shape, fillValue: fillValue, dtype: dtype);

  /// Creates a tensor with random values uniformly distributed in [0, 1).
  ///
  /// Equivalent to `torch.rand()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.random([3, 224, 224]);
  /// final seeded = TensorBuffer.random([3, 224, 224], seed: 42);
  /// ```
  static TensorBuffer random(
    List<int> shape, {
    DType dtype = DType.float32,
    int? seed,
  }) =>
      _randomImpl(shape, dtype: dtype, seed: seed);

  /// Creates a tensor with random values from a standard normal distribution N(0, 1).
  ///
  /// Equivalent to `torch.randn()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.randn([3, 224, 224]);
  /// final seeded = TensorBuffer.randn([3, 224, 224], seed: 42);
  /// ```
  static TensorBuffer randn(
    List<int> shape, {
    DType dtype = DType.float32,
    int? seed,
  }) =>
      _randnImpl(shape, dtype: dtype, seed: seed);

  /// Creates a 2D identity matrix.
  ///
  /// Equivalent to `torch.eye()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.eye(3); // 3x3 identity matrix
  /// final rect = TensorBuffer.eye(2, m: 4); // 2x4 matrix with 1s on diagonal
  /// ```
  static TensorBuffer eye(
    int n, {
    int? m,
    DType dtype = DType.float32,
  }) =>
      _eyeImpl(n, m: m, dtype: dtype);

  /// Creates a 1D tensor with evenly spaced values.
  ///
  /// Equivalent to `torch.linspace()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.linspace(0.0, 1.0, steps: 5);
  /// // [0.0, 0.25, 0.5, 0.75, 1.0]
  /// ```
  static TensorBuffer linspace(
    double start,
    double end, {
    required int steps,
    DType dtype = DType.float32,
  }) =>
      _linspaceImpl(start, end, steps: steps, dtype: dtype);

  /// Creates a 1D tensor with values in a range with a given step.
  ///
  /// Equivalent to `torch.arange()` in PyTorch.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.arange(start: 0.0, end: 5.0);
  /// // [0.0, 1.0, 2.0, 3.0, 4.0]
  ///
  /// final stepped = TensorBuffer.arange(start: 0.0, end: 10.0, step: 2.0);
  /// // [0.0, 2.0, 4.0, 6.0, 8.0]
  /// ```
  static TensorBuffer arange({
    required double start,
    required double end,
    double step = 1.0,
    DType dtype = DType.float32,
  }) =>
      _arangeImpl(start: start, end: end, step: step, dtype: dtype);

  /// Creates a tensor from an existing [Float32List] with the given [shape].
  ///
  /// Throws [ShapeMismatchException] if data length doesn't match shape.
  static TensorBuffer fromFloat32List(Float32List data, List<int> shape) =>
      _fromFloat32ListImpl(data, shape);

  /// Creates a tensor from an existing [Float64List] with the given [shape].
  ///
  /// Throws [ShapeMismatchException] if data length doesn't match shape.
  static TensorBuffer fromFloat64List(Float64List data, List<int> shape) =>
      _fromFloat64ListImpl(data, shape);

  /// Creates a tensor from an existing [Uint8List] with the given [shape].
  ///
  /// Throws [ShapeMismatchException] if data length doesn't match shape.
  static TensorBuffer fromUint8List(Uint8List data, List<int> shape) =>
      _fromUint8ListImpl(data, shape);

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
    _validateShapeStatic(shape);
  }

  /// Static helper to validate shape before tensor creation.
  static void _validateShapeStatic(List<int> shape) {
    if (shape.isEmpty) {
      throw InvalidParameterException('shape', shape, 'Shape cannot be empty');
    }
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] <= 0) {
        throw InvalidParameterException(
          'shape[$i]',
          shape[i],
          'Shape dimension must be positive',
        );
      }
    }
  }

  @override
  String toString() {
    return 'TensorBuffer(shape: $shape, dtype: $dtype, '
        'strides: $strides, contiguous: $isContiguous)';
  }
}
