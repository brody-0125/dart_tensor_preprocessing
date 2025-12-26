import 'dart:typed_data';

import 'dtype.dart';
import 'memory_format.dart';
import 'tensor_storage.dart';

class TensorBuffer {
  final TensorStorage storage;
  final List<int> shape;
  final List<int> strides;
  final int storageOffset;
  final MemoryFormat memoryFormat;
  bool? _isContiguousCache;

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

  DType get dtype => storage.dtype;
  int get rank => shape.length;
  int get numel => shape.fold(1, (a, b) => a * b);
  int get sizeInBytes => numel * dtype.byteSize;

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

  Float32List get dataAsFloat32List {
    if (dtype != DType.float32) {
      throw StateError('Tensor dtype is $dtype, not float32');
    }
    return data as Float32List;
  }

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
