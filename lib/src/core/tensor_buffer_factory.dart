part of 'tensor_buffer.dart';

// ============================================================================
// TensorBuffer Factory Methods
// ============================================================================

/// Extension on TensorBuffer for factory methods.
///
/// This file contains all static factory constructors for TensorBuffer:
/// - zeros, ones, full
/// - random, randn
/// - eye, linspace, arange
/// - fromFloat32List, fromFloat64List, fromUint8List

extension TensorBufferFactory on TensorBuffer {
  // This extension is a placeholder for documentation purposes.
  // The actual factory methods are static methods on TensorBuffer,
  // which work through the `part of` directive.
}

// ============================================================================
// Factory Methods (as static extensions via part)
// ============================================================================

// Note: The following are implemented as static methods on TensorBuffer
// in the main tensor_buffer.dart file, but their implementations are here.

/// Creates a tensor buffer without initializing values.
///
/// In Dart, typed data buffers are zero-initialized by the VM,
/// so this is functionally equivalent to [_zerosImpl]. The semantic
/// distinction signals intent: the caller will overwrite all elements.
TensorBuffer _uninitializedImpl(
  List<int> shape, {
  DType dtype = DType.float32,
  MemoryFormat memoryFormat = MemoryFormat.contiguous,
}) {
  TensorBuffer._validateShapeStatic(shape);
  final numel = shape.fold(1, (a, b) => a * b);
  final data = dtype.createBuffer(numel);
  return TensorBuffer(
    storage: TensorStorage(data, dtype),
    shape: List.unmodifiable(shape),
    memoryFormat: memoryFormat,
  );
}

/// Creates a tensor filled with zeros.
TensorBuffer _zerosImpl(
  List<int> shape, {
  DType dtype = DType.float32,
  MemoryFormat memoryFormat = MemoryFormat.contiguous,
}) {
  TensorBuffer._validateShapeStatic(shape);
  final numel = shape.fold(1, (a, b) => a * b);
  final data = dtype.createBuffer(numel);
  return TensorBuffer(
    storage: TensorStorage(data, dtype),
    shape: List.unmodifiable(shape),
    memoryFormat: memoryFormat,
  );
}

/// Creates a tensor filled with ones.
TensorBuffer _onesImpl(List<int> shape, {DType dtype = DType.float32}) {
  TensorBuffer._validateShapeStatic(shape);
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
TensorBuffer _fullImpl(
  List<int> shape, {
  required double fillValue,
  DType dtype = DType.float32,
}) {
  TensorBuffer._validateShapeStatic(shape);
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
/// Uniform float32/float64 values. Seeds are reproducible within this package,
/// but do not reproduce PyTorch random sequences.
///
/// ```dart
/// final tensor = TensorBuffer.random([3, 224, 224]);
/// final seeded = TensorBuffer.random([3, 224, 224], seed: 42);
/// ```
TensorBuffer _randomImpl(
  List<int> shape, {
  DType dtype = DType.float32,
  int? seed,
}) {
  TensorBuffer._validateShapeStatic(shape);
  final numel = shape.fold(1, (a, b) => a * b);
  if (!dtype.isFloatingPoint) {
    throw InvalidParameterException(
      'dtype',
      dtype,
      'Random values require float32 or float64',
    );
  }
  final data = dtype.createBuffer(numel) as List<double>;
  final rng = seed != null ? _SeededRandom(seed) : _SeededRandom.system();

  for (int i = 0; i < numel; i++) {
    // A valid double below one can round to one in Float32 storage.
    do {
      data[i] = rng.nextDouble();
    } while (data[i] == 1.0);
  }

  return TensorBuffer(
    storage: TensorStorage(data as TypedData, dtype),
    shape: List.unmodifiable(shape),
  );
}

/// Creates a tensor with random values from a standard normal distribution N(0, 1).
///
/// Normal float32/float64 values. Seeds are reproducible within this package,
/// but do not reproduce PyTorch random sequences.
///
/// ```dart
/// final tensor = TensorBuffer.randn([3, 224, 224]);
/// final seeded = TensorBuffer.randn([3, 224, 224], seed: 42);
/// ```
TensorBuffer _randnImpl(
  List<int> shape, {
  DType dtype = DType.float32,
  int? seed,
}) {
  TensorBuffer._validateShapeStatic(shape);
  final numel = shape.fold(1, (a, b) => a * b);
  if (!dtype.isFloatingPoint) {
    throw InvalidParameterException(
      'dtype',
      dtype,
      'Random values require float32 or float64',
    );
  }
  final data = dtype.createBuffer(numel) as List<double>;
  final rng = seed != null ? _SeededRandom(seed) : _SeededRandom.system();

  // Box-Muller transform for normal distribution
  for (int i = 0; i < numel; i += 2) {
    var u1 = rng.nextDouble();
    while (u1 == 0) {
      u1 = rng.nextDouble();
    }
    final u2 = rng.nextDouble();
    final r = math.sqrt(-2.0 * math.log(u1));
    final theta = 2.0 * math.pi * u2;
    data[i] = r * math.cos(theta);
    if (i + 1 < numel) {
      data[i + 1] = r * math.sin(theta);
    }
  }

  return TensorBuffer(
    storage: TensorStorage(data as TypedData, dtype),
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
TensorBuffer _eyeImpl(int n, {int? m, DType dtype = DType.float32}) {
  if (n <= 0) {
    throw InvalidParameterException('n', n, 'n must be positive');
  }
  final cols = m ?? n;
  if (cols <= 0) {
    throw InvalidParameterException('m', cols, 'm must be positive');
  }

  final data = dtype.createBuffer(n * cols);
  final values = data as List<num>;
  final diagSize = n < cols ? n : cols;
  for (int i = 0; i < diagSize; i++) {
    values[i * cols + i] = dtype.isInteger ? 1 : 1.0;
  }

  return TensorBuffer(
    storage: TensorStorage(data, dtype),
    shape: List.unmodifiable([n, cols]),
  );
}

/// Creates a 1D tensor with evenly spaced values.
///
/// Computes a double sequence, then truncates integer outputs toward zero.
/// Finite endpoints and positive steps are required.
///
/// ```dart
/// final tensor = TensorBuffer.linspace(0.0, 1.0, steps: 5);
/// // [0.0, 0.25, 0.5, 0.75, 1.0]
/// ```
TensorBuffer _linspaceImpl(
  double start,
  double end, {
  required int steps,
  DType dtype = DType.float32,
}) {
  if (steps < 1) {
    throw InvalidParameterException('steps', steps, 'steps must be >= 1');
  }

  if (!start.isFinite || !end.isFinite) {
    throw InvalidParameterException('range', [
      start,
      end,
    ], 'endpoints must be finite');
  }
  final data = dtype.createBuffer(steps);
  final values = data as List<num>;

  if (steps == 1) {
    values[0] = dtype.isInteger ? start.toInt() : start;
  } else {
    final step = (end - start) / (steps - 1);
    for (int i = 0; i < steps; i++) {
      final value = i < steps ~/ 2
          ? start + i * step
          : end - (steps - i - 1) * step;
      values[i] = dtype.isInteger ? value.toInt() : value;
    }
  }

  return TensorBuffer(
    storage: TensorStorage(data, dtype),
    shape: List.unmodifiable([steps]),
  );
}

/// Creates a 1D tensor with values in a range with a given step.
///
/// Computes a double sequence excluding the end, then truncates integer
/// outputs toward zero. Finite values and a nonempty range are required.
///
/// ```dart
/// final tensor = TensorBuffer.arange(start: 0.0, end: 5.0);
/// // [0.0, 1.0, 2.0, 3.0, 4.0]
///
/// final stepped = TensorBuffer.arange(start: 0.0, end: 10.0, step: 2.0);
/// // [0.0, 2.0, 4.0, 6.0, 8.0]
/// ```
TensorBuffer _arangeImpl({
  required double start,
  required double end,
  double step = 1.0,
  DType dtype = DType.float32,
}) {
  if (!start.isFinite || !end.isFinite || !step.isFinite) {
    throw InvalidParameterException('range', [
      start,
      end,
      step,
    ], 'range values must be finite');
  }
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
    throw InvalidParameterException('range', [
      start,
      end,
    ], 'empty tensors are not supported');
  }

  final data = dtype.createBuffer(numSteps);
  final values = data as List<num>;
  for (int i = 0; i < numSteps; i++) {
    final value = start + i * step;
    values[i] = dtype.isInteger ? value.toInt() : value;
  }

  return TensorBuffer(
    storage: TensorStorage(data, dtype),
    shape: List.unmodifiable([numSteps]),
  );
}

/// Creates a tensor from an existing [Float32List] with the given [shape].
///
/// Throws [ShapeMismatchException] if data length doesn't match shape.
TensorBuffer _fromFloat32ListImpl(Float32List data, List<int> shape) {
  final expectedNumel = shape.fold(1, (a, b) => a * b);
  if (data.length != expectedNumel) {
    throw ShapeMismatchException(
      actual: shape,
      message:
          'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
    );
  }
  return TensorBuffer(
    storage: TensorStorage.fromFloat32List(data),
    shape: List.unmodifiable(shape),
  );
}

/// Creates a tensor from an existing [Float64List] with the given [shape].
///
/// Throws [ShapeMismatchException] if data length doesn't match shape.
TensorBuffer _fromFloat64ListImpl(Float64List data, List<int> shape) {
  final expectedNumel = shape.fold(1, (a, b) => a * b);
  if (data.length != expectedNumel) {
    throw ShapeMismatchException(
      actual: shape,
      message:
          'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
    );
  }
  return TensorBuffer(
    storage: TensorStorage.fromFloat64List(data),
    shape: List.unmodifiable(shape),
  );
}

/// Creates a tensor from an existing [Uint8List] with the given [shape].
///
/// Throws [ShapeMismatchException] if data length doesn't match shape.
TensorBuffer _fromUint8ListImpl(Uint8List data, List<int> shape) {
  final expectedNumel = shape.fold(1, (a, b) => a * b);
  if (data.length != expectedNumel) {
    throw ShapeMismatchException(
      actual: shape,
      message:
          'Data length (${data.length}) does not match shape $shape (numel: $expectedNumel)',
    );
  }
  return TensorBuffer(
    storage: TensorStorage.fromUint8List(data),
    shape: List.unmodifiable(shape),
  );
}

// ============================================================================
// Seeded Random Number Generator
// ============================================================================

/// Simple seeded random number generator (Linear Congruential Generator).
class _SeededRandom {
  static final _systemRandom = math.Random();
  int _state;

  _SeededRandom(int seed) : _state = seed & 0xFFFFFFFF;

  factory _SeededRandom.system() {
    return _SeededRandom(_systemRandom.nextInt(0x80000000));
  }

  double nextDouble() {
    // LCG parameters (same as glibc)
    _state = ((_state * 1103515245) + 12345) & 0x7FFFFFFF;
    return _state / 0x80000000;
  }
}
