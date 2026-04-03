import 'dart:math' as math;
import 'dart:typed_data';

import '../../core/dtype.dart';
import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/tensor_indexing.dart';
import '../transform_op.dart';

/// Softmax activation function.
///
/// Applies softmax along the specified axis, normalizing values to sum to 1.
/// Equivalent to `F.softmax()` in PyTorch.
///
/// ## Complexity
///
/// Let `n` = total elements, `k` = size of softmax axis.
///
/// - **Time**: O(n) with 3 passes per softmax slice: find max, compute exp(x-max), normalize.
/// - **Space**: O(n) for output tensor.
///
/// ```dart
/// final result = SoftmaxOp(axis: -1)(tensor);
/// ```
class SoftmaxOp extends TransformOp with RequiresContiguous {
  /// The axis along which to compute softmax.
  final int axis;

  /// Creates a Softmax operation along the given [axis].
  SoftmaxOp({required this.axis});

  @override
  String get name => 'Softmax(axis=$axis)';

  @override
  OperationCapabilities get capabilities =>
      const OperationCapabilities(requiresContiguous: true);

  @override
  TensorBuffer apply(TensorBuffer input) {
    // Normalize axis before validation
    final normalizedAxis = axis < 0 ? input.rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= input.rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -input.rank,
        max: input.rank - 1,
        dimension: 'axis',
      );
    }

    // Use single-copy pattern
    final output = cloneForModification(input);

    // Compute softmax in-place on output
    _computeSoftmax(output, normalizedAxis);

    return output;
  }

  void _computeSoftmax(TensorBuffer tensor, int axis) {
    final shape = tensor.shape;

    // Dtype-specialized implementation
    if (tensor.dtype == DType.float32) {
      _computeSoftmaxFloat32(tensor.storage.data as Float32List, shape, axis);
    } else if (tensor.dtype == DType.float64) {
      _computeSoftmaxFloat64(tensor.storage.data as Float64List, shape, axis);
    } else {
      // Fallback for other types (convert, compute, store back)
      _computeSoftmaxGeneric(tensor, shape, axis);
    }
  }

  void _computeSoftmaxFloat32(Float32List data, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];

    // Compute strides for iteration
    final strides = TensorIndexer.computeStrides(shape);

    // Number of softmax operations to perform
    int numSoftmax = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSoftmax *= shape[i];
    }

    // Iterate over all positions except the softmax axis
    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSoftmax; s++) {
      // Find max for numerical stability
      double maxVal = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        if (data[idx] > maxVal) maxVal = data[idx];
      }

      // Compute exp(x - max) and sum
      double sumExp = 0;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        final expVal = math.exp(data[idx] - maxVal);
        data[idx] = expVal;
        sumExp += expVal;
      }

      // Normalize
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        data[idx] /= sumExp;
      }

      // Increment indices for non-axis dimensions
      for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        indices[d]++;
        if (indices[d] < shape[d]) break;
        indices[d] = 0;
      }
    }
  }

  void _computeSoftmaxFloat64(Float64List data, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];

    // Compute strides for iteration
    final strides = TensorIndexer.computeStrides(shape);

    // Number of softmax operations to perform
    int numSoftmax = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSoftmax *= shape[i];
    }

    // Iterate over all positions except the softmax axis
    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSoftmax; s++) {
      // Find max for numerical stability
      double maxVal = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        if (data[idx] > maxVal) maxVal = data[idx];
      }

      // Compute exp(x - max) and sum
      double sumExp = 0;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        final expVal = math.exp(data[idx] - maxVal);
        data[idx] = expVal;
        sumExp += expVal;
      }

      // Normalize
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        data[idx] /= sumExp;
      }

      // Increment indices for non-axis dimensions
      for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        indices[d]++;
        if (indices[d] < shape[d]) break;
        indices[d] = 0;
      }
    }
  }

  void _computeSoftmaxGeneric(TensorBuffer tensor, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];
    final storage = tensor.storage;

    // Compute strides for iteration
    final strides = TensorIndexer.computeStrides(shape);

    // Number of softmax operations to perform
    int numSoftmax = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSoftmax *= shape[i];
    }

    // Iterate over all positions except the softmax axis
    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSoftmax; s++) {
      // Find max for numerical stability
      double maxVal = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        final val = storage.getAsDouble(idx);
        if (val > maxVal) maxVal = val;
      }

      // Compute exp(x - max) and sum
      double sumExp = 0;
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        final expVal = math.exp(storage.getAsDouble(idx) - maxVal);
        storage.setFromDouble(idx, expVal);
        sumExp += expVal;
      }

      // Normalize
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        storage.setFromDouble(idx, storage.getAsDouble(idx) / sumExp);
      }

      // Increment indices for non-axis dimensions
      for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        indices[d]++;
        if (indices[d] < shape[d]) break;
        indices[d] = 0;
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
