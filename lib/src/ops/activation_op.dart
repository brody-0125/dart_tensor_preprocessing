import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/dtype_dispatcher.dart';
import '../utils/simd_ops.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// Rectified Linear Unit (ReLU) activation function.
///
/// Applies max(0, x) element-wise.
/// Equivalent to `F.relu()` in PyTorch.
///
/// ```dart
/// final result = ReLUOp()(tensor);
/// ```
class ReLUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a ReLU operation.
  ReLUOp();

  @override
  String get name => 'ReLU';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _relu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('ReLUOp.applyInPlace');
    }
    _relu(input);
  }

  void _relu(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        // Use SIMD acceleration for Float32
        SimdOps.relu(list);
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          if (list[i] < 0) list[i] = 0;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final value = t.storage.getAsDouble(i);
          if (value < 0) t.storage.setFromDouble(i, 0.0);
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Leaky Rectified Linear Unit activation function.
///
/// Applies x if x > 0, else negativeSlope * x.
/// Equivalent to `F.leaky_relu()` in PyTorch.
///
/// ```dart
/// final result = LeakyReLUOp(negativeSlope: 0.1)(tensor);
/// ```
class LeakyReLUOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// The slope for negative values. Default is 0.01.
  final double negativeSlope;

  /// Creates a Leaky ReLU operation with the given [negativeSlope].
  LeakyReLUOp({this.negativeSlope = 0.01});

  @override
  String get name => 'LeakyReLU(slope=$negativeSlope)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _leakyRelu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('LeakyReLUOp.applyInPlace');
    }
    _leakyRelu(input);
  }

  void _leakyRelu(TensorBuffer tensor) {
    final slope = negativeSlope;
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        // Use SIMD acceleration for Float32
        SimdOps.leakyRelu(list, slope);
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          if (list[i] < 0) list[i] *= slope;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final value = t.storage.getAsDouble(i);
          if (value < 0) t.storage.setFromDouble(i, value * slope);
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Sigmoid activation function.
///
/// Applies 1 / (1 + exp(-x)) element-wise.
/// Equivalent to `torch.sigmoid()` in PyTorch.
///
/// ```dart
/// final result = SigmoidOp()(tensor);
/// ```
class SigmoidOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a Sigmoid operation.
  SigmoidOp();

  @override
  String get name => 'Sigmoid';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _sigmoid(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('SigmoidOp.applyInPlace');
    }
    _sigmoid(input);
  }

  void _sigmoid(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          list[i] = 1.0 / (1.0 + math.exp(-list[i]));
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          list[i] = 1.0 / (1.0 + math.exp(-list[i]));
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final value = t.storage.getAsDouble(i);
          t.storage.setFromDouble(i, 1.0 / (1.0 + math.exp(-value)));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Hyperbolic tangent (Tanh) activation function.
///
/// Applies tanh(x) element-wise.
/// Equivalent to `torch.tanh()` in PyTorch.
///
/// ```dart
/// final result = TanhOp()(tensor);
/// ```
class TanhOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a Tanh operation.
  TanhOp();

  @override
  String get name => 'Tanh';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _tanh(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('TanhOp.applyInPlace');
    }
    _tanh(input);
  }

  void _tanh(TensorBuffer tensor) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final exp2x = math.exp(2 * list[i]);
          list[i] = (exp2x - 1) / (exp2x + 1);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final exp2x = math.exp(2 * list[i]);
          list[i] = (exp2x - 1) / (exp2x + 1);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final value = t.storage.getAsDouble(i);
          final exp2x = math.exp(2 * value);
          t.storage.setFromDouble(i, (exp2x - 1) / (exp2x + 1));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

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
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
      );

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
      _computeSoftmaxFloat32(
        tensor.storage.data as Float32List,
        shape,
        axis,
      );
    } else if (tensor.dtype == DType.float64) {
      _computeSoftmaxFloat64(
        tensor.storage.data as Float64List,
        shape,
        axis,
      );
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
