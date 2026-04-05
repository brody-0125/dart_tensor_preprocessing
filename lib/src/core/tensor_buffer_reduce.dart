import 'dart:typed_data';

import '../exceptions/tensor_exceptions.dart';
import 'dtype.dart';
import 'tensor_buffer.dart';
import 'tensor_storage.dart';

/// Extension providing reduction operations for [TensorBuffer].
///
/// Includes both full-tensor reductions (sum, mean, min, max) and
/// axis-wise reductions (sumAxis, meanAxis, minAxis, maxAxis).
extension TensorBufferReduce on TensorBuffer {
  // ============================================================
  // Full Tensor Reductions
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

  /// Returns the flat index of the maximum value in this tensor.
  ///
  /// If multiple elements have the maximum value, the index of the first
  /// occurrence (in row-major order) is returned.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5]),
  ///   [5],
  /// );
  /// print(tensor.argmax()); // 4
  /// ```
  int argmax() {
    double maxVal = double.negativeInfinity;
    int maxIdx = 0;
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      final value = storage.getAsDouble(offset);
      if (value > maxVal) {
        maxVal = value;
        maxIdx = i;
      }
      _incrementIndices(indices);
    }
    return maxIdx;
  }

  /// Returns the flat index of the minimum value in this tensor.
  ///
  /// If multiple elements have the minimum value, the index of the first
  /// occurrence (in row-major order) is returned.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5]),
  ///   [5],
  /// );
  /// print(tensor.argmin()); // 1
  /// ```
  int argmin() {
    double minVal = double.infinity;
    int minIdx = 0;
    final indices = List<int>.filled(rank, 0);

    for (int i = 0; i < numel; i++) {
      int offset = storageOffset;
      for (int d = 0; d < rank; d++) {
        offset += indices[d] * strides[d];
      }
      final value = storage.getAsDouble(offset);
      if (value < minVal) {
        minVal = value;
        minIdx = i;
      }
      _incrementIndices(indices);
    }
    return minIdx;
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
  // Single-Axis Reductions
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

  // ============================================================
  // Argmax / Argmin Axis Reductions
  // ============================================================

  /// Returns a tensor of indices of maximum values along the specified [axis].
  ///
  /// The output dtype is always [DType.int64]. If [keepDims] is true, the
  /// reduced dimension is retained with size 1.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5, 9]),
  ///   [2, 3],
  /// );
  /// // Argmax along axis 1: [argmax(3,1,4), argmax(1,5,9)] = [2, 2]
  /// final result = tensor.argmaxAxis(1);
  /// print(result.toList()); // [2, 2]
  /// print(result.dtype); // DType.int64
  /// ```
  TensorBuffer argmaxAxis(int axis, {bool keepDims = false}) {
    return _reduceAxisToIndex(
      axis,
      keepDims: keepDims,
      indexReduce: _argmaxReduce,
    );
  }

  /// Returns a tensor of indices of minimum values along the specified [axis].
  ///
  /// The output dtype is always [DType.int64]. If [keepDims] is true, the
  /// reduced dimension is retained with size 1.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.fromFloat32List(
  ///   Float32List.fromList([3, 1, 4, 1, 5, 9]),
  ///   [2, 3],
  /// );
  /// // Argmin along axis 1: [argmin(3,1,4), argmin(1,5,9)] = [1, 0]
  /// final result = tensor.argminAxis(1);
  /// print(result.toList()); // [1, 0]
  /// print(result.dtype); // DType.int64
  /// ```
  TensorBuffer argminAxis(int axis, {bool keepDims = false}) {
    return _reduceAxisToIndex(
      axis,
      keepDims: keepDims,
      indexReduce: _argminReduce,
    );
  }

  // ============================================================
  // Multi-Axis Reductions
  // ============================================================

  /// Returns a tensor with the sum of elements along multiple [axes].
  ///
  /// If [keepDims] is true, the reduced dimensions are retained with size 1.
  /// Otherwise, the reduced dimensions are removed from the output shape.
  /// Supports negative axis indexing.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3, 4]);
  /// tensor.sumAxes([0, 2]);           // shape: [3]
  /// tensor.sumAxes([0, 2], keepDims: true); // shape: [1, 3, 1]
  /// tensor.sumAxes([-1, -3]);         // same as [0, 2]
  /// ```
  TensorBuffer sumAxes(List<int> axes, {bool keepDims = false}) {
    return _reduceAxes(axes, keepDims: keepDims, reduce: _sumReduce);
  }

  /// Returns a tensor with the mean of elements along multiple [axes].
  ///
  /// If [keepDims] is true, the reduced dimensions are retained with size 1.
  /// Otherwise, the reduced dimensions are removed from the output shape.
  /// Supports negative axis indexing.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3, 4]);
  /// tensor.meanAxes([0, 2]);           // shape: [3]
  /// tensor.meanAxes([0, 2], keepDims: true); // shape: [1, 3, 1]
  /// ```
  TensorBuffer meanAxes(List<int> axes, {bool keepDims = false}) {
    return _reduceAxes(axes, keepDims: keepDims, reduce: _meanReduce);
  }

  /// Returns a tensor with the minimum value along multiple [axes].
  ///
  /// If [keepDims] is true, the reduced dimensions are retained with size 1.
  /// Otherwise, the reduced dimensions are removed from the output shape.
  /// Supports negative axis indexing.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3, 4]);
  /// tensor.minAxes([0, 2]);           // shape: [3]
  /// tensor.minAxes([0, 2], keepDims: true); // shape: [1, 3, 1]
  /// ```
  TensorBuffer minAxes(List<int> axes, {bool keepDims = false}) {
    return _reduceAxes(axes, keepDims: keepDims, reduce: _minReduce);
  }

  /// Returns a tensor with the maximum value along multiple [axes].
  ///
  /// If [keepDims] is true, the reduced dimensions are retained with size 1.
  /// Otherwise, the reduced dimensions are removed from the output shape.
  /// Supports negative axis indexing.
  ///
  /// ```dart
  /// final tensor = TensorBuffer.zeros([2, 3, 4]);
  /// tensor.maxAxes([0, 2]);           // shape: [3]
  /// tensor.maxAxes([0, 2], keepDims: true); // shape: [1, 3, 1]
  /// ```
  TensorBuffer maxAxes(List<int> axes, {bool keepDims = false}) {
    return _reduceAxes(axes, keepDims: keepDims, reduce: _maxReduce);
  }

  /// Generic multi-axis reduction implementation.
  ///
  /// Normalizes and sorts axes (largest first) to prevent index shifting,
  /// then applies sequential single-axis reductions.
  TensorBuffer _reduceAxes(
    List<int> axes, {
    required bool keepDims,
    required double Function(List<double>) reduce,
  }) {
    if (axes.isEmpty) {
      return this;
    }

    // Normalize negative axes and validate
    final normalizedAxes = <int>[];
    for (final axis in axes) {
      final normalized = axis < 0 ? rank + axis : axis;
      if (normalized < 0 || normalized >= rank) {
        throw IndexOutOfBoundsException(
          index: axis,
          min: -rank,
          max: rank - 1,
          dimension: 'axis',
        );
      }
      normalizedAxes.add(normalized);
    }

    // Check for duplicates
    final uniqueAxes = normalizedAxes.toSet();
    if (uniqueAxes.length != normalizedAxes.length) {
      throw InvalidParameterException(
        'axes',
        axes,
        'Duplicate axes are not allowed',
      );
    }

    // Sort in descending order (largest first) to prevent index shifting
    final sortedAxes = normalizedAxes.toList()..sort((a, b) => b.compareTo(a));

    // Apply sequential reductions
    var result = this;
    for (final axis in sortedAxes) {
      result = result._reduceAxis(axis, keepDims: keepDims, reduce: reduce);
    }
    return result;
  }

  /// Generic single-axis reduction implementation.
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
      return TensorBuffer.fromFloat32List(Float32List.fromList([resultValue]), [
        1,
      ]);
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

  /// Generic single-axis index-reduction implementation.
  ///
  /// Similar to [_reduceAxis] but returns the index of the selected element
  /// rather than the value. Output dtype is always [DType.int64].
  TensorBuffer _reduceAxisToIndex(
    int axis, {
    required bool keepDims,
    required int Function(List<double>) indexReduce,
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
      final resultIndex = indexReduce(values);
      final outputData = Int64List.fromList([resultIndex]);
      return TensorBuffer(
        storage: TensorStorage(outputData, DType.int64),
        shape: [1],
      );
    }

    // Create output buffer
    final outputNumel = outputShape.fold(1, (a, b) => a * b);
    final outputData = Int64List(outputNumel);

    // Compute reductions
    final axisSize = shape[normalizedAxis];
    final outputIndices = List<int>.filled(outputShape.length, 0);
    final inputIndices = List<int>.filled(rank, 0);

    for (int outIdx = 0; outIdx < outputNumel; outIdx++) {
      // Collect values along the reduction axis
      final values = <double>[];

      for (int axisIdx = 0; axisIdx < axisSize; axisIdx++) {
        // Build input indices from output indices
        int outDim = 0;
        for (int d = 0; d < rank; d++) {
          if (d == normalizedAxis) {
            inputIndices[d] = axisIdx;
            if (keepDims) outDim++;
          } else {
            inputIndices[d] = outputIndices[outDim];
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

      // Apply index reduction and store result
      outputData[outIdx] = indexReduce(values);

      // Increment output indices
      for (int d = outputShape.length - 1; d >= 0; d--) {
        outputIndices[d]++;
        if (outputIndices[d] < outputShape[d]) break;
        outputIndices[d] = 0;
      }
    }

    return TensorBuffer(
      storage: TensorStorage(outputData, DType.int64),
      shape: outputShape,
    );
  }

  // ============================================================
  // Data Extraction
  // ============================================================

  /// Returns all elements as a [List<double>].
  ///
  /// This method iterates over all elements in logical order and returns them
  /// as a new list. For large tensors, consider using [TensorBuffer.data] for
  /// direct access to the underlying typed data instead.
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
}

// ============================================================
// Private Helper Functions
// ============================================================

double _sumReduce(List<double> values) {
  double result = 0;
  for (final v in values) {
    result += v;
  }
  return result;
}

double _meanReduce(List<double> values) {
  return _sumReduce(values) / values.length;
}

double _minReduce(List<double> values) {
  double result = double.infinity;
  for (final v in values) {
    if (v < result) result = v;
  }
  return result;
}

double _maxReduce(List<double> values) {
  double result = double.negativeInfinity;
  for (final v in values) {
    if (v > result) result = v;
  }
  return result;
}

int _argmaxReduce(List<double> values) {
  double maxVal = double.negativeInfinity;
  int maxIdx = 0;
  for (int i = 0; i < values.length; i++) {
    if (values[i] > maxVal) {
      maxVal = values[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

int _argminReduce(List<double> values) {
  double minVal = double.infinity;
  int minIdx = 0;
  for (int i = 0; i < values.length; i++) {
    if (values[i] < minVal) {
      minVal = values[i];
      minIdx = i;
    }
  }
  return minIdx;
}
