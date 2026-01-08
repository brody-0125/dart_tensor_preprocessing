import 'dart:typed_data';

import '../exceptions/tensor_exceptions.dart';
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
  }) {
    _validateShapeStatic(shape);
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
    _validateShapeStatic(shape);
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
  }) {
    _validateShapeStatic(shape);
    final numel = shape.fold(1, (a, b) => a * b);
    final data = dtype.createBuffer(numel);

    for (int i = 0; i < numel; i++) {
      switch (data) {
        case final Float32List list:
          list[i] = fillValue;
        case final Float64List list:
          list[i] = fillValue;
        case final Int8List list:
          list[i] = fillValue.toInt();
        case final Int16List list:
          list[i] = fillValue.toInt();
        case final Int32List list:
          list[i] = fillValue.toInt();
        case final Int64List list:
          list[i] = fillValue.toInt();
        case final Uint8List list:
          list[i] = fillValue.toInt().clamp(0, 255);
        case final Uint16List list:
          list[i] = fillValue.toInt().clamp(0, 65535);
        case final Uint32List list:
          list[i] = fillValue.toInt();
        case final Uint64List list:
          list[i] = fillValue.toInt();
      }
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable(shape),
    );
  }

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
  }) {
    _validateShapeStatic(shape);
    final numel = shape.fold(1, (a, b) => a * b);
    final data = Float32List(numel);
    final rng = seed != null ? _SeededRandom(seed) : _SeededRandom.system();

    for (int i = 0; i < numel; i++) {
      data[i] = rng.nextDouble();
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable(shape),
    );
  }

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
  }) {
    _validateShapeStatic(shape);
    final numel = shape.fold(1, (a, b) => a * b);
    final data = Float32List(numel);
    final rng = seed != null ? _SeededRandom(seed) : _SeededRandom.system();

    // Box-Muller transform for normal distribution
    for (int i = 0; i < numel; i += 2) {
      final u1 = rng.nextDouble();
      final u2 = rng.nextDouble();
      // Avoid log(0)
      final safeU1 = u1 < 1e-10 ? 1e-10 : u1;
      final r = _sqrt(-2.0 * _log(safeU1));
      final theta = 2.0 * _pi * u2;
      data[i] = r * _cos(theta);
      if (i + 1 < numel) {
        data[i + 1] = r * _sin(theta);
      }
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable(shape),
    );
  }

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
  }) {
    if (n <= 0) {
      throw InvalidParameterException('n', n, 'n must be positive');
    }
    final cols = m ?? n;
    if (cols <= 0) {
      throw InvalidParameterException('m', cols, 'm must be positive');
    }

    final data = Float32List(n * cols);
    final diagSize = n < cols ? n : cols;
    for (int i = 0; i < diagSize; i++) {
      data[i * cols + i] = 1.0;
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable([n, cols]),
    );
  }

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
  }) {
    if (steps < 1) {
      throw InvalidParameterException('steps', steps, 'steps must be >= 1');
    }

    final data = Float32List(steps);

    if (steps == 1) {
      data[0] = start;
    } else {
      final step = (end - start) / (steps - 1);
      for (int i = 0; i < steps; i++) {
        data[i] = start + i * step;
      }
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable([steps]),
    );
  }

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
  }) {
    if (step == 0) {
      throw InvalidParameterException('step', step, 'step cannot be zero');
    }
    if ((end > start && step < 0) || (end < start && step > 0)) {
      throw InvalidParameterException(
        'step',
        step,
        'step direction does not match range direction',
      );
    }

    final numSteps = ((end - start) / step).ceil();
    if (numSteps <= 0) {
      return TensorBuffer(
        storage: TensorStorage(Float32List(0), dtype),
        shape: List.unmodifiable([0]),
      );
    }

    final data = Float32List(numSteps);
    for (int i = 0; i < numSteps; i++) {
      data[i] = start + i * step;
    }

    return TensorBuffer(
      storage: TensorStorage(data, dtype),
      shape: List.unmodifiable([numSteps]),
    );
  }

  /// Creates a tensor from an existing [Float32List] with the given [shape].
  ///
  /// Throws [ShapeMismatchException] if data length doesn't match shape.
  static TensorBuffer fromFloat32List(Float32List data, List<int> shape) {
    final expectedNumel = shape.fold(1, (a, b) => a * b);
    if (data.length != expectedNumel) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
      );
    }
    return TensorBuffer(
      storage: TensorStorage.fromFloat32List(data),
      shape: List.unmodifiable(shape),
    );
  }

  /// Creates a tensor from an existing [Uint8List] with the given [shape].
  ///
  /// Throws [ShapeMismatchException] if data length doesn't match shape.
  static TensorBuffer fromUint8List(Uint8List data, List<int> shape) {
    final expectedNumel = shape.fold(1, (a, b) => a * b);
    if (data.length != expectedNumel) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
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

  // ============================================================
  // Reduction Operations
  // ============================================================

  /// Returns the sum of all elements in this tensor.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([1, 2, 3, 4]),
  ///   [2, 2],
  /// );
  /// print(tensor.sum()); // 10.0
  /// ```
  double sum() {
    double result = 0;
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      result += storage.getAsDouble(offset);
      _incrementIndices(indices);
    }

    return result;
  }

  /// Returns the arithmetic mean of all elements in this tensor.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([1, 2, 3, 4]),
  ///   [2, 2],
  /// );
  /// print(tensor.mean()); // 2.5
  /// ```
  double mean() => sum() / numel;

  /// Returns the minimum value among all elements in this tensor.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5]),
  ///   [5],
  /// );
  /// print(tensor.min()); // 1.0
  /// ```
  double min() {
    double result = double.infinity;
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      final value = storage.getAsDouble(offset);
      if (value < result) result = value;
      _incrementIndices(indices);
    }

    return result;
  }

  /// Returns the maximum value among all elements in this tensor.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5]),
  ///   [5],
  /// );
  /// print(tensor.max()); // 5.0
  /// ```
  double max() {
    double result = double.negativeInfinity;
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      final value = storage.getAsDouble(offset);
      if (value > result) result = value;
      _incrementIndices(indices);
    }

    return result;
  }

  /// Increments multi-dimensional indices in row-major order.
  void _incrementIndices(List<int> indices) {
    for (int d = rank - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < shape[d]) break;
      indices[d] = 0;
    }
  }

  // ============================================================
  // Axis-wise Reduction Operations
  // ============================================================

  /// Returns a tensor with the sum of elements along the specified [axis].
  ///
  /// If [keepDims] is true, the reduced dimension is retained with size 1.
  /// Otherwise, the reduced dimension is removed from the output shape.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([1, 2, 3, 4, 5, 6]),
  ///   [2, 3],
  /// );
  /// // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
  /// final result = tensor.sumAxis(0);
  /// print(result.shape); // [3]
  /// print(result.toList()); // [5.0, 7.0, 9.0]
  /// ```
  TensorBuffer sumAxis(int axis, {bool keepDims = false}) {
    return _reduceAxis(axis, keepDims: keepDims, reduce: _sumReduce);
  }

  /// Returns a tensor with the mean of elements along the specified [axis].
  ///
  /// If [keepDims] is true, the reduced dimension is retained with size 1.
  /// Otherwise, the reduced dimension is removed from the output shape.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([1, 2, 3, 4, 5, 6]),
  ///   [2, 3],
  /// );
  /// // Mean along axis 0: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
  /// final result = tensor.meanAxis(0);
  /// print(result.toList()); // [2.5, 3.5, 4.5]
  /// ```
  TensorBuffer meanAxis(int axis, {bool keepDims = false}) {
    return _reduceAxis(axis, keepDims: keepDims, reduce: _meanReduce);
  }

  /// Returns a tensor with the minimum value along the specified [axis].
  ///
  /// If [keepDims] is true, the reduced dimension is retained with size 1.
  /// Otherwise, the reduced dimension is removed from the output shape.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5, 9]),
  ///   [2, 3],
  /// );
  /// // Min along axis 1: [min(3,1,4), min(1,5,9)] = [1, 1]
  /// final result = tensor.minAxis(1);
  /// print(result.toList()); // [1.0, 1.0]
  /// ```
  TensorBuffer minAxis(int axis, {bool keepDims = false}) {
    return _reduceAxis(axis, keepDims: keepDims, reduce: _minReduce);
  }

  /// Returns a tensor with the maximum value along the specified [axis].
  ///
  /// If [keepDims] is true, the reduced dimension is retained with size 1.
  /// Otherwise, the reduced dimension is removed from the output shape.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5, 9]),
  ///   [2, 3],
  /// );
  /// // Max along axis 1: [max(3,1,4), max(1,5,9)] = [4, 9]
  /// final result = tensor.maxAxis(1);
  /// print(result.toList()); // [4.0, 9.0]
  /// ```
  TensorBuffer maxAxis(int axis, {bool keepDims = false}) {
    return _reduceAxis(axis, keepDims: keepDims, reduce: _maxReduce);
  }

  /// Generic axis reduction implementation.
  TensorBuffer _reduceAxis(
    int axis, {
    required bool keepDims,
    required double Function(List<double>) reduce,
  }) {
    // Normalize negative axis
    final normalizedAxis = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -rank,
        max: rank - 1,
        dimension: 'axis',
      );
    }

    // Compute output shape
    final outputShape = <int>[];
    for (int d = 0; d < rank; d++) {
      if (d == normalizedAxis) {
        if (keepDims) outputShape.add(1);
      } else {
        outputShape.add(shape[d]);
      }
    }

    // Handle scalar result (1D tensor reduced without keepDims)
    if (outputShape.isEmpty) {
      final values = <double>[];
      final indices = List<int>.filled(rank, 0);
      for (int i = 0; i < numel; i++) {
        int offset = storageOffset;
        for (int d = 0; d < rank; d++) {
          offset += indices[d] * strides[d];
        }
        values.add(storage.getAsDouble(offset));
        _incrementIndices(indices);
      }
      final resultValue = reduce(values);
      return TensorBuffer.fromFloat32List(
        Float32List.fromList([resultValue]),
        [1],
      );
    }

    // Create output buffer
    final outputNumel = outputShape.fold(1, (a, b) => a * b);
    final outputData = Float32List(outputNumel);

    // Compute reductions
    final axisSize = shape[normalizedAxis];
    final outputIndices = List<int>.filled(outputShape.length, 0);

    for (int outIdx = 0; outIdx < outputNumel; outIdx++) {
      // Collect values along the reduction axis
      final values = <double>[];

      for (int axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        // Build input indices from output indices
        final inputIndices = <int>[];
        int outDim = 0;
        for (int d = 0; d < rank; d++) {
          if (d == normalizedAxis) {
            inputIndices.add(axisIdx);
            if (keepDims) outDim++; // Skip the size-1 dimension in output
          } else {
            inputIndices.add(outputIndices[outDim]);
            outDim++;
          }
        }

        // Compute offset and get value
        int offset = storageOffset;
        for (int d = 0; d < rank; d++) {
          offset += inputIndices[d] * strides[d];
        }
        values.add(storage.getAsDouble(offset));
      }

      // Apply reduction and store result
      outputData[outIdx] = reduce(values);

      // Increment output indices
      for (int d = outputShape.length - 1; d >= 0; d--) {
        outputIndices[d]++;
        if (outputIndices[d] < outputShape[d]) break;
        outputIndices[d] = 0;
      }
    }

    return TensorBuffer.fromFloat32List(outputData, outputShape);
  }

  static double _sumReduce(List<double> values) {
    double result = 0;
    for (final v in values) {
      result += v;
    }
    return result;
  }

  static double _meanReduce(List<double> values) {
    return _sumReduce(values) / values.length;
  }

  static double _minReduce(List<double> values) {
    double result = double.infinity;
    for (final v in values) {
      if (v < result) result = v;
    }
    return result;
  }

  static double _maxReduce(List<double> values) {
    double result = double.negativeInfinity;
    for (final v in values) {
      if (v > result) result = v;
    }
    return result;
  }

  // ============================================================
  // Data Extraction
  // ============================================================

  /// Returns all elements as a [List<double>].
  ///
  /// This method iterates over all elements in logical order and returns them
  /// as a new list. For large tensors, consider using [data] for direct access
  /// to the underlying typed data instead.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([1, 2, 3, 4]),
  ///   [2, 2],
  /// );
  /// print(tensor.toList()); // [1.0, 2.0, 3.0, 4.0]
  /// ```
  List<double> toList() {
    final result = <double>[];
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      result.add(storage.getAsDouble(offset));
      _incrementIndices(indices);
    }

    return result;
  }

  @override
  String toString() {
    return 'TensorBuffer(shape: $shape, dtype: $dtype, '
        'strides: $strides, contiguous: $isContiguous)';
  }
}

// Math function aliases for random number generation
double _sqrt(double x) => x >= 0 ? _power(x, 0.5) : double.nan;
double _log(double x) {
  if (x <= 0) return double.negativeInfinity;
  // Natural log approximation using Taylor series or built-in
  double result = 0;
  double term = (x - 1) / (x + 1);
  final termSq = term * term;
  for (int i = 1; i <= 100; i += 2) {
    result += term / i;
    term *= termSq;
  }
  return 2 * result;
}

double _power(double base, double exp) {
  if (exp == 0.5) {
    // Newton's method for square root
    if (base < 0) return double.nan;
    if (base == 0) return 0;
    double guess = base / 2;
    for (int i = 0; i < 20; i++) {
      guess = (guess + base / guess) / 2;
    }
    return guess;
  }
  // For other cases, use exp(exp * ln(base))
  return _exp(exp * _log(base));
}

double _exp(double x) {
  double result = 1;
  double term = 1;
  for (int i = 1; i <= 30; i++) {
    term *= x / i;
    result += term;
    if (term.abs() < 1e-15) break;
  }
  return result;
}

const double _pi = 3.14159265358979323846;

double _cos(double x) {
  // Normalize to [-pi, pi]
  while (x > _pi) {
    x -= 2 * _pi;
  }
  while (x < -_pi) {
    x += 2 * _pi;
  }

  double result = 1;
  double term = 1;
  final xSq = x * x;
  for (int i = 1; i <= 15; i++) {
    term *= -xSq / ((2 * i - 1) * (2 * i));
    result += term;
  }
  return result;
}

double _sin(double x) {
  // Normalize to [-pi, pi]
  while (x > _pi) {
    x -= 2 * _pi;
  }
  while (x < -_pi) {
    x += 2 * _pi;
  }

  double result = x;
  double term = x;
  final xSq = x * x;
  for (int i = 1; i <= 15; i++) {
    term *= -xSq / ((2 * i) * (2 * i + 1));
    result += term;
  }
  return result;
}

/// Simple seeded random number generator (Linear Congruential Generator).
class _SeededRandom {
  int _state;

  _SeededRandom(int seed) : _state = seed & 0xFFFFFFFF;

  factory _SeededRandom.system() {
    return _SeededRandom(DateTime.now().microsecondsSinceEpoch);
  }

  double nextDouble() {
    // LCG parameters (same as glibc)
    _state = ((_state * 1103515245) + 12345) & 0x7FFFFFFF;
    return _state / 0x7FFFFFFF;
  }
}
