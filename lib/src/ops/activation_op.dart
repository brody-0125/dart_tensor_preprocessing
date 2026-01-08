import 'dart:math' as math;
import 'dart:typed_data';

import '../core/tensor_buffer.dart';
import '../core/tensor_storage.dart';
import '../exceptions/tensor_exceptions.dart';
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
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final output = contiguous.clone();
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
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      if (value < 0) {
        tensor.storage.setFromDouble(i, 0.0);
      }
    }
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
class LeakyReLUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// The slope for negative values. Default is 0.01.
  final double negativeSlope;

  /// Creates a Leaky ReLU operation with the given [negativeSlope].
  LeakyReLUOp({this.negativeSlope = 0.01});

  @override
  String get name => 'LeakyReLU(slope=$negativeSlope)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final output = contiguous.clone();
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
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      if (value < 0) {
        tensor.storage.setFromDouble(i, value * negativeSlope);
      }
    }
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
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final output = contiguous.clone();
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
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      final result = 1.0 / (1.0 + math.exp(-value));
      tensor.storage.setFromDouble(i, result);
    }
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
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final output = contiguous.clone();
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
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
      // Using the formula that avoids overflow for large x
      final exp2x = math.exp(2 * value);
      final result = (exp2x - 1) / (exp2x + 1);
      tensor.storage.setFromDouble(i, result);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Softmax activation function.
///
/// Applies softmax along the specified axis, normalizing values to sum to 1.
/// Equivalent to `F.softmax()` in PyTorch.
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
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);

    // Normalize axis
    final normalizedAxis = axis < 0 ? input.rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= input.rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -input.rank,
        max: input.rank - 1,
        dimension: 'axis',
      );
    }

    final shape = contiguous.shape;
    final numel = contiguous.numel;
    final outputData = Float32List(numel);

    // Copy input data
    for (int i = 0; i < numel; i++) {
      outputData[i] = contiguous.storage.getAsDouble(i).toDouble();
    }

    // Compute softmax
    _computeSoftmax(outputData, shape, normalizedAxis);

    return TensorBuffer(
      storage: TensorStorage.fromFloat32List(outputData),
      shape: shape.toList(),
    );
  }

  void _computeSoftmax(Float32List data, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];

    // Compute strides for iteration
    final strides = List<int>.filled(rank, 0);
    int stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }

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

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
