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

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// Standard activation in Transformers (BERT, GPT, ViT).
/// Equivalent to `F.gelu()` in PyTorch.
///
/// Supports two modes:
/// - `approximate: 'none'` (default): Exact GELU using error function
/// - `approximate: 'tanh'`: Fast approximation (PyTorch default)
///
/// ## Formulas
///
/// - **Exact**: `x * Φ(x)` where Φ is the standard normal CDF
/// - **Tanh approx**: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// ```dart
/// final result = GELUOp()(tensor);  // Exact
/// final result = GELUOp(approximate: 'tanh')(tensor);  // Fast
/// ```
class GELUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Approximation method: 'none' (exact) or 'tanh' (fast).
  final String approximate;

  /// Creates a GELU operation.
  ///
  /// [approximate] can be 'none' for exact computation or 'tanh' for
  /// the fast approximation used by PyTorch.
  GELUOp({this.approximate = 'none'}) {
    if (approximate != 'none' && approximate != 'tanh') {
      throw InvalidParameterException(
        'approximate',
        approximate,
        "must be 'none' or 'tanh'",
      );
    }
  }

  @override
  String get name =>
      approximate == 'none' ? 'GELU' : 'GELU(approximate=$approximate)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _gelu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('GELUOp.applyInPlace');
    }
    _gelu(input);
  }

  // Constants for GELU computation
  static const double _sqrt2 = 1.4142135623730951;
  static const double _sqrt2OverPi = 0.7978845608028654; // sqrt(2/π)
  static const double _tanhCoeff = 0.044715;

  void _gelu(TensorBuffer tensor) {
    final useTanh = approximate == 'tanh';

    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        if (useTanh) {
          _geluTanhFloat32(list, numel);
        } else {
          _geluExactFloat32(list, numel);
        }
      },
      onFloat64: (list, numel) {
        if (useTanh) {
          _geluTanhFloat64(list, numel);
        } else {
          _geluExactFloat64(list, numel);
        }
      },
      fallback: (t) {
        final n = t.numel;
        if (useTanh) {
          for (int i = 0; i < n; i++) {
            final x = t.storage.getAsDouble(i);
            final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
            final tanhVal = _tanh(inner);
            t.storage.setFromDouble(i, 0.5 * x * (1.0 + tanhVal));
          }
        } else {
          for (int i = 0; i < n; i++) {
            final x = t.storage.getAsDouble(i);
            t.storage.setFromDouble(i, x * 0.5 * (1.0 + _erf(x / _sqrt2)));
          }
        }
      },
    );
  }

  void _geluExactFloat32(Float32List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      list[i] = x * 0.5 * (1.0 + _erf(x / _sqrt2));
    }
  }

  void _geluExactFloat64(Float64List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      list[i] = x * 0.5 * (1.0 + _erf(x / _sqrt2));
    }
  }

  void _geluTanhFloat32(Float32List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
      list[i] = 0.5 * x * (1.0 + _tanh(inner));
    }
  }

  void _geluTanhFloat64(Float64List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
      list[i] = 0.5 * x * (1.0 + _tanh(inner));
    }
  }

  /// Approximation of the error function using Abramowitz and Stegun formula.
  static double _erf(double x) {
    // Constants for approximation
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    final sign = x < 0 ? -1.0 : 1.0;
    final absX = x.abs();

    final t = 1.0 / (1.0 + p * absX);
    final t2 = t * t;
    final t3 = t2 * t;
    final t4 = t3 * t;
    final t5 = t4 * t;

    final y =
        1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * math.exp(-absX * absX);

    return sign * y;
  }

  /// Fast tanh implementation.
  static double _tanh(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return -1.0;
    final exp2x = math.exp(2 * x);
    return (exp2x - 1) / (exp2x + 1);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Sigmoid Linear Unit (SiLU / Swish) activation function.
///
/// Efficient modern activation used in EfficientNet, YOLOv5, and other models.
/// Equivalent to `F.silu()` in PyTorch.
///
/// ## Formula
///
/// `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
///
/// ```dart
/// final result = SiLUOp()(tensor);
/// ```
class SiLUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a SiLU (Swish) operation.
  SiLUOp();

  @override
  String get name => 'SiLU';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _silu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('SiLUOp.applyInPlace');
    }
    _silu(input);
  }

  void _silu(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x / (1.0 + math.exp(-x));
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x / (1.0 + math.exp(-x));
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          t.storage.setFromDouble(i, x / (1.0 + math.exp(-x)));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Alias for SiLUOp - Swish activation function.
///
/// Swish was the original name for this activation function.
/// `swish(x) = x * sigmoid(x)`
typedef SwishOp = SiLUOp;

/// Hard Sigmoid activation function.
///
/// Piecewise linear approximation of sigmoid, used in MobileNetV3.
/// Equivalent to `F.hardsigmoid()` in PyTorch.
///
/// ## Formula
///
/// `hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)`
///
/// ```dart
/// final result = HardsigmoidOp()(tensor);
/// ```
class HardsigmoidOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Creates a Hard Sigmoid operation.
  HardsigmoidOp();

  @override
  String get name => 'Hardsigmoid';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _hardsigmoid(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('HardsigmoidOp.applyInPlace');
    }
    _hardsigmoid(input);
  }

  void _hardsigmoid(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = ((x + 3.0) / 6.0).clamp(0.0, 1.0);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = ((x + 3.0) / 6.0).clamp(0.0, 1.0);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          t.storage.setFromDouble(i, ((x + 3.0) / 6.0).clamp(0.0, 1.0));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Hard Swish activation function.
///
/// Piecewise linear approximation of swish, used in MobileNetV3.
/// Equivalent to `F.hardswish()` in PyTorch.
///
/// ## Formula
///
/// `hardswish(x) = x * hardsigmoid(x) = x * clamp((x + 3) / 6, 0, 1)`
///
/// ```dart
/// final result = HardswishOp()(tensor);
/// ```
class HardswishOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a Hard Swish operation.
  HardswishOp();

  @override
  String get name => 'Hardswish';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _hardswish(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('HardswishOp.applyInPlace');
    }
    _hardswish(input);
  }

  void _hardswish(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x * ((x + 3.0) / 6.0).clamp(0.0, 1.0);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x * ((x + 3.0) / 6.0).clamp(0.0, 1.0);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          t.storage.setFromDouble(i, x * ((x + 3.0) / 6.0).clamp(0.0, 1.0));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Mish activation function.
///
/// Self-regularizing non-monotonic activation used in YOLOv4+.
/// Equivalent to `F.mish()` in PyTorch.
///
/// ## Formula
///
/// `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
///
/// ```dart
/// final result = MishOp()(tensor);
/// ```
class MishOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Creates a Mish operation.
  MishOp();

  @override
  String get name => 'Mish';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _mish(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('MishOp.applyInPlace');
    }
    _mish(input);
  }

  void _mish(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          // softplus(x) = ln(1 + exp(x)), with numerical stability
          final softplus = x > 20 ? x : math.log(1 + math.exp(x));
          list[i] = x * _tanh(softplus);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          final softplus = x > 20 ? x : math.log(1 + math.exp(x));
          list[i] = x * _tanh(softplus);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          final softplus = x > 20 ? x : math.log(1 + math.exp(x));
          t.storage.setFromDouble(i, x * _tanh(softplus));
        }
      },
    );
  }

  static double _tanh(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return -1.0;
    final exp2x = math.exp(2 * x);
    return (exp2x - 1) / (exp2x + 1);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Exponential Linear Unit (ELU) activation function.
///
/// Equivalent to `F.elu()` in PyTorch.
///
/// ## Formula
///
/// `elu(x) = x if x > 0 else alpha * (exp(x) - 1)`
///
/// ```dart
/// final result = ELUOp(alpha: 1.0)(tensor);
/// ```
class ELUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// The alpha value for negative inputs. Default is 1.0.
  final double alpha;

  /// Creates an ELU operation with the given [alpha].
  ELUOp({this.alpha = 1.0});

  @override
  String get name => alpha == 1.0 ? 'ELU' : 'ELU(alpha=$alpha)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _elu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('ELUOp.applyInPlace');
    }
    _elu(input);
  }

  void _elu(TensorBuffer tensor) {
    final a = alpha;
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          if (x < 0) list[i] = a * (math.exp(x) - 1);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          if (x < 0) list[i] = a * (math.exp(x) - 1);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          if (x < 0) t.storage.setFromDouble(i, a * (math.exp(x) - 1));
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
