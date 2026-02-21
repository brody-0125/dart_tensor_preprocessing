import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// Lp normalization along a specified dimension.
///
/// Equivalent to `F.normalize()` in PyTorch and ONNX `LpNormalization`.
///
/// ## Formula
///
/// - L2: `x / sqrt(sum(x^2) + eps)`
/// - L1: `x / (sum(|x|) + eps)`
/// - Linf: `x / (max(|x|) + eps)`
/// - General p: `x / (sum(|x|^p)^(1/p) + eps)`
///
/// ```dart
/// final result = LpNormalizeOp.l2(dim: -1)(tensor);
/// ```
class LpNormalizeOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// The norm order (p value).
  final double p;

  /// The dimension along which to normalize.
  final int dim;

  /// Small value to avoid division by zero.
  final double eps;

  /// Creates an Lp normalization operation.
  LpNormalizeOp({this.p = 2.0, this.dim = -1, this.eps = 1e-12});

  /// Creates an L2 normalization operation.
  factory LpNormalizeOp.l2({int dim = -1, double eps = 1e-12}) =>
      LpNormalizeOp(p: 2.0, dim: dim, eps: eps);

  /// Creates an L1 normalization operation.
  factory LpNormalizeOp.l1({int dim = -1, double eps = 1e-12}) =>
      LpNormalizeOp(p: 1.0, dim: dim, eps: eps);

  /// Creates an Linf (max) normalization operation.
  factory LpNormalizeOp.linf({int dim = -1, double eps = 1e-12}) =>
      LpNormalizeOp(p: double.infinity, dim: dim, eps: eps);

  @override
  String get name => 'LpNormalize(p=$p)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'F.normalize',
        onnxOpType: 'LpNormalization',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _normalize(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('LpNormalizeOp.applyInPlace');
    }
    _normalize(input);
  }

  void _normalize(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    final normalizedDim = dim < 0 ? rank + dim : dim;
    if (normalizedDim < 0 || normalizedDim >= rank) {
      throw IndexOutOfBoundsException(
        index: dim,
        min: -rank,
        max: rank - 1,
        dimension: 'LpNormalizeOp dim',
      );
    }

    if (tensor.dtype == DType.float32) {
      _normalizeFloat32(
        tensor.storage.data as Float32List,
        shape,
        normalizedDim,
      );
    } else if (tensor.dtype == DType.float64) {
      _normalizeFloat64(
        tensor.storage.data as Float64List,
        shape,
        normalizedDim,
      );
    } else {
      _normalizeGeneric(tensor, shape, normalizedDim);
    }
  }

  void _normalizeFloat32(Float32List data, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];
    final strides = TensorIndexer.computeStrides(shape);

    int numSlices = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSlices *= shape[i];
    }

    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSlices; s++) {
      // Compute norm
      double norm = 0;
      if (p == double.infinity) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final absVal = data[idx].abs();
          if (absVal > norm) norm = absVal;
        }
      } else if (p == 2.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final x = data[idx];
          norm += x * x;
        }
        norm = math.sqrt(norm);
      } else if (p == 1.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += data[idx].abs();
        }
      } else {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += math.pow(data[idx].abs(), p).toDouble();
        }
        norm = math.pow(norm, 1.0 / p).toDouble();
      }

      // Normalize
      final invNorm = 1.0 / (norm + eps);
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        data[idx] *= invNorm;
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

  void _normalizeFloat64(Float64List data, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];
    final strides = TensorIndexer.computeStrides(shape);

    int numSlices = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSlices *= shape[i];
    }

    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSlices; s++) {
      double norm = 0;
      if (p == double.infinity) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final absVal = data[idx].abs();
          if (absVal > norm) norm = absVal;
        }
      } else if (p == 2.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final x = data[idx];
          norm += x * x;
        }
        norm = math.sqrt(norm);
      } else if (p == 1.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += data[idx].abs();
        }
      } else {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += math.pow(data[idx].abs(), p).toDouble();
        }
        norm = math.pow(norm, 1.0 / p).toDouble();
      }

      final invNorm = 1.0 / (norm + eps);
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        data[idx] *= invNorm;
      }

      for (int d = rank - 1; d >= 0; d--) {
        if (d == axis) continue;
        indices[d]++;
        if (indices[d] < shape[d]) break;
        indices[d] = 0;
      }
    }
  }

  void _normalizeGeneric(TensorBuffer tensor, List<int> shape, int axis) {
    final rank = shape.length;
    final axisSize = shape[axis];
    final storage = tensor.storage;
    final strides = TensorIndexer.computeStrides(shape);

    int numSlices = 1;
    for (int i = 0; i < rank; i++) {
      if (i != axis) numSlices *= shape[i];
    }

    final indices = List<int>.filled(rank, 0);
    for (int s = 0; s < numSlices; s++) {
      double norm = 0;
      if (p == double.infinity) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final absVal = storage.getAsDouble(idx).abs();
          if (absVal > norm) norm = absVal;
        }
      } else if (p == 2.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          final x = storage.getAsDouble(idx);
          norm += x * x;
        }
        norm = math.sqrt(norm);
      } else if (p == 1.0) {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += storage.getAsDouble(idx).abs();
        }
      } else {
        for (int a = 0; a < axisSize; a++) {
          indices[axis] = a;
          int idx = 0;
          for (int d = 0; d < rank; d++) {
            idx += indices[d] * strides[d];
          }
          norm += math.pow(storage.getAsDouble(idx).abs(), p).toDouble();
        }
        norm = math.pow(norm, 1.0 / p).toDouble();
      }

      final invNorm = 1.0 / (norm + eps);
      for (int a = 0; a < axisSize; a++) {
        indices[axis] = a;
        int idx = 0;
        for (int d = 0; d < rank; d++) {
          idx += indices[d] * strides[d];
        }
        storage.setFromDouble(idx, storage.getAsDouble(idx) * invNorm);
      }

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
