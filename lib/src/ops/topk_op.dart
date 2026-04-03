import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// The result of a TopK operation: (values, indices).
///
/// - `values`: A [TensorBuffer] containing the k largest or smallest values.
/// - `indices`: A [TensorBuffer] (DType.int64) containing the original indices
///   of the selected values along the specified axis.
typedef TopKResult = (TensorBuffer values, TensorBuffer indices);

/// Selects the k largest or smallest values and their indices along an axis.
///
/// Equivalent to `torch.topk()` in PyTorch and the ONNX `TopK` operator.
///
/// The output shape is the same as the input shape, except that the specified
/// axis dimension is replaced with `k`.
///
/// ```dart
/// // Select top-5 values along last axis
/// final op = TopKOp(k: 5);
/// final (values, indices) = op.applyTopK(tensor);
///
/// // Select 3 smallest values along axis 1
/// final op2 = TopKOp(k: 3, axis: 1, largest: false);
/// final (values2, indices2) = op2.applyTopK(tensor);
/// ```
///
/// When used in a [TensorPipeline], [apply] returns only the values tensor.
/// Use [applyTopK] to get both values and indices.
class TopKOp extends TransformOp {
  /// The number of top elements to select.
  final int k;

  /// The axis along which to select. Supports negative indexing.
  final int axis;

  /// If `true`, selects the k largest values. If `false`, selects the k
  /// smallest values.
  final bool largest;

  /// If `true`, the selected k values are returned in sorted order
  /// (descending for largest, ascending for smallest).
  final bool sorted;

  /// Creates a TopK operation.
  ///
  /// - [k]: Number of top elements to select (must be > 0).
  /// - [axis]: Axis along which to operate (default: -1, the last axis).
  /// - [largest]: Whether to select largest (`true`) or smallest (`false`).
  /// - [sorted]: Whether the output should be sorted (default: `true`).
  TopKOp({
    required this.k,
    this.axis = -1,
    this.largest = true,
    this.sorted = true,
  });

  @override
  String get name => 'TopK';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    pytorchEquivalent: 'torch.topk',
    onnxOpType: 'TopK',
    onnxOpsetVersion: 1,
  );

  /// Applies the TopK operation and returns only the values tensor.
  ///
  /// This is compatible with [TensorPipeline]. Use [applyTopK] to also
  /// retrieve the indices tensor.
  @override
  TensorBuffer apply(TensorBuffer input) => applyTopK(input).$1;

  /// Applies the TopK operation and returns both values and indices.
  ///
  /// Returns a [TopKResult] record of `(values, indices)` where:
  /// - `values` has the same dtype as [input] and shape equal to the input
  ///   shape with the axis dimension replaced by [k].
  /// - `indices` has dtype [DType.int64] and the same shape as `values`.
  TopKResult applyTopK(TensorBuffer input) {
    final rank = input.rank;

    final normalizedAxis = axis < 0 ? rank + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -rank,
        max: rank - 1,
        dimension: 'TopKOp axis',
      );
    }

    final axisSize = input.shape[normalizedAxis];

    if (k <= 0) {
      throw InvalidParameterException('k', k, 'k must be positive');
    }
    if (k > axisSize) {
      throw InvalidParameterException(
        'k',
        k,
        'k ($k) cannot exceed axis size ($axisSize)',
      );
    }

    final src = input.isContiguous ? input : input.contiguous();

    final outputShape = List<int>.from(src.shape);
    outputShape[normalizedAxis] = k;

    final valuesOut = TensorBuffer.uninitialized(outputShape, dtype: src.dtype);
    final indicesOut = TensorBuffer.uninitialized(
      outputShape,
      dtype: DType.int64,
    );

    final inStrides = TensorIndexer.computeStrides(src.shape);
    final outStrides = TensorIndexer.computeStrides(outputShape);

    final workValues = Float64List(axisSize);
    final workIndices = Int32List(axisSize);

    // Shape with the target axis removed, for iterating over orthogonal slices
    final outerShape = <int>[];
    for (int d = 0; d < rank; d++) {
      if (d != normalizedAxis) outerShape.add(src.shape[d]);
    }
    final outerStrides = TensorIndexer.computeStrides(outerShape);

    final idxData = indicesOut.storage.data as Int64List;

    final double Function(int) inDataGet;
    final void Function(int, double) outDataSet;
    switch (src.dtype) {
      case DType.float32:
        final inData = src.storage.data as Float32List;
        final outData = valuesOut.storage.data as Float32List;
        inDataGet = (int idx) => inData[idx].toDouble();
        outDataSet = (int idx, double val) => outData[idx] = val;
      case DType.float64:
        final inData = src.storage.data as Float64List;
        final outData = valuesOut.storage.data as Float64List;
        inDataGet = (int idx) => inData[idx];
        outDataSet = (int idx, double val) => outData[idx] = val;
      default:
        final inStorage = src.storage;
        final outStorage = valuesOut.storage;
        inDataGet = (int idx) => inStorage.getAsDouble(idx);
        outDataSet = (int idx, double val) =>
            outStorage.setFromDouble(idx, val);
    }

    _topkLoop(
      outDataSet: outDataSet,
      inDataGet: inDataGet,
      idxData: idxData,
      inStrides: inStrides,
      outStrides: outStrides,
      outerShape: outerShape,
      outerStrides: outerStrides,
      normalizedAxis: normalizedAxis,
      workValues: workValues,
      workIndices: workIndices,
    );

    return (valuesOut, indicesOut);
  }

  void _topkLoop({
    required void Function(int idx, double val) outDataSet,
    required double Function(int idx) inDataGet,
    required Int64List idxData,
    required List<int> inStrides,
    required List<int> outStrides,
    required List<int> outerShape,
    required List<int> outerStrides,
    required int normalizedAxis,
    required Float64List workValues,
    required Int32List workIndices,
  }) {
    final outerRank = outerShape.length;
    final rank = inStrides.length;
    final axisSize = workValues.length;
    final outerNumel = outerShape.isEmpty
        ? 1
        : outerShape.fold(1, (a, b) => a * b);
    final axisInStride = inStrides[normalizedAxis];
    final axisOutStride = outStrides[normalizedAxis];
    final outerCoords = List<int>.filled(outerRank, 0);

    for (int outerIdx = 0; outerIdx < outerNumel; outerIdx++) {
      // Decompose outerIdx into coordinates for non-axis dimensions
      int remaining = outerIdx;
      for (int d = 0; d < outerRank; d++) {
        outerCoords[d] = remaining ~/ outerStrides[d];
        remaining = remaining % outerStrides[d];
      }

      int baseInOffset = 0;
      int baseOutOffset = 0;
      int outerDim = 0;
      for (int d = 0; d < rank; d++) {
        if (d == normalizedAxis) continue;
        baseInOffset += outerCoords[outerDim] * inStrides[d];
        baseOutOffset += outerCoords[outerDim] * outStrides[d];
        outerDim++;
      }

      for (int a = 0; a < axisSize; a++) {
        workValues[a] = inDataGet(baseInOffset + a * axisInStride);
        workIndices[a] = a;
      }

      _introSelect(workValues, workIndices, 0, axisSize - 1, k, largest);

      if (sorted && k > 1) {
        _insertionSort(workValues, workIndices, 0, k - 1, largest);
      }

      for (int i = 0; i < k; i++) {
        final outOffset = baseOutOffset + i * axisOutStride;
        outDataSet(outOffset, workValues[i]);
        idxData[outOffset] = workIndices[i];
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final normalizedAxis = axis < 0 ? inputShape.length + axis : axis;
    final out = List<int>.from(inputShape);
    out[normalizedAxis] = k;
    return out;
  }
}

// ============================================================================
// Quickselect / Introselect
// ============================================================================

/// Partially sorts so that the top-k elements are in positions [0, k).
///
/// Uses quickselect with median-of-three pivot. Falls back to insertion sort
/// for small partitions.
void _introSelect(
  Float64List values,
  Int32List indices,
  int left,
  int right,
  int k,
  bool largest,
) {
  while (left < right) {
    if (right - left < 16) {
      _insertionSort(values, indices, left, right, largest);
      return;
    }

    final mid = left + (right - left) ~/ 2;
    _sortThree(values, indices, left, mid, right, largest);
    // Pivot at mid; move to right - 1 so left acts as sentinel
    _swap(values, indices, mid, right - 1);
    final pivot = values[right - 1];

    int i = left;
    int j = right - 1;

    while (true) {
      if (largest) {
        while (values[++i] > pivot) {}
        while (j > left && values[--j] < pivot) {}
      } else {
        while (values[++i] < pivot) {}
        while (j > left && values[--j] > pivot) {}
      }
      if (i >= j) break;
      _swap(values, indices, i, j);
    }

    _swap(values, indices, i, right - 1);

    if (i >= k) {
      right = i - 1;
    } else {
      left = i + 1;
    }
  }
}

/// Insertion sort for small partitions. Sorts in descending order if
/// [largest] is true, ascending if false.
void _insertionSort(
  Float64List values,
  Int32List indices,
  int left,
  int right,
  bool largest,
) {
  for (int i = left + 1; i <= right; i++) {
    final val = values[i];
    final idx = indices[i];
    int j = i - 1;
    if (largest) {
      while (j >= left && values[j] < val) {
        values[j + 1] = values[j];
        indices[j + 1] = indices[j];
        j--;
      }
    } else {
      while (j >= left && values[j] > val) {
        values[j + 1] = values[j];
        indices[j + 1] = indices[j];
        j--;
      }
    }
    values[j + 1] = val;
    indices[j + 1] = idx;
  }
}

void _sortThree(
  Float64List values,
  Int32List indices,
  int a,
  int b,
  int c,
  bool largest,
) {
  if (largest) {
    if (values[a] < values[b]) _swap(values, indices, a, b);
    if (values[a] < values[c]) _swap(values, indices, a, c);
    if (values[b] < values[c]) _swap(values, indices, b, c);
  } else {
    if (values[a] > values[b]) _swap(values, indices, a, b);
    if (values[a] > values[c]) _swap(values, indices, a, c);
    if (values[b] > values[c]) _swap(values, indices, b, c);
  }
}

void _swap(Float64List values, Int32List indices, int i, int j) {
  final tmpVal = values[i];
  values[i] = values[j];
  values[j] = tmpVal;
  final tmpIdx = indices[i];
  indices[i] = indices[j];
  indices[j] = tmpIdx;
}

// ============================================================================
// Extension Method
// ============================================================================

/// Provides a convenient `topk` method on [TensorBuffer].
extension TopKExtension on TensorBuffer {
  /// Returns the [k] largest or smallest values and their indices along [axis].
  ///
  /// Equivalent to `torch.topk()` in PyTorch.
  ///
  /// ```dart
  /// final (values, indices) = tensor.topk(5);
  /// final (values, indices) = tensor.topk(3, axis: 1, largest: true, sorted: true);
  /// ```
  TopKResult topk(
    int k, {
    int axis = -1,
    bool largest = true,
    bool sorted = true,
  }) {
    return TopKOp(
      k: k,
      axis: axis,
      largest: largest,
      sorted: sorted,
    ).applyTopK(this);
  }
}
