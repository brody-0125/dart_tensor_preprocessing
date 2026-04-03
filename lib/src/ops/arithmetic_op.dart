import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/dtype_dispatcher.dart';
import '../utils/simd_ops.dart';
import 'transform_op.dart';

/// Base class for binary arithmetic operations.
///
/// Supports both scalar and tensor operands for element-wise operations.
abstract class ArithmeticOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Scalar operand (if using scalar mode).
  final double? scalar;

  /// Tensor operand (if using tensor mode).
  final TensorBuffer? other;

  ArithmeticOp._({this.scalar, this.other}) {
    if (scalar == null && other == null) {
      throw InvalidParameterException(
        'operand',
        null,
        'Either scalar or tensor operand must be provided',
      );
    }
  }

  /// The operation to apply element-wise.
  double operation(double a, double b);

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    supportsInPlace: true,
    requiresContiguous: true,
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _apply(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw NonContiguousException('$runtimeType.applyInPlace');
    }
    _apply(input);
  }

  void _apply(TensorBuffer tensor) {
    if (scalar != null) {
      _applyScalar(tensor, scalar!);
    } else {
      final otherContiguous = ensureContiguous(other!);
      if (tensor.numel != otherContiguous.numel) {
        throw ShapeMismatchException(
          actual: otherContiguous.shape,
          message:
              'Tensor shapes must match for element-wise operation: ${tensor.shape} vs ${otherContiguous.shape}',
        );
      }
      _applyTensor(tensor, otherContiguous);
    }
  }

  /// Applies scalar operation with dtype-specialized implementation.
  void _applyScalar(TensorBuffer tensor, double s);

  /// Applies tensor operation with dtype-specialized implementation.
  void _applyTensor(TensorBuffer tensor, TensorBuffer other);

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Adds a scalar or tensor to the input element-wise.
///
/// Equivalent to `torch.add()` in PyTorch.
///
/// ```dart
/// final result = AddOp(scalar: 1.0)(tensor);  // tensor + 1.0
/// final result = AddOp.tensor(other)(tensor); // tensor + other
/// ```
class AddOp extends ArithmeticOp {
  /// Creates an addition operation with a scalar operand.
  AddOp({required double scalar}) : super._(scalar: scalar);

  /// Creates an addition operation with a tensor operand.
  AddOp.tensor(TensorBuffer other) : super._(other: other);

  @override
  String get name =>
      scalar != null ? 'Add(scalar=$scalar)' : 'Add(tensor=${other!.shape})';

  @override
  double operation(double a, double b) => a + b;

  @override
  void _applyScalar(TensorBuffer tensor, double s) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) => SimdOps.addScalar(list, s),
      onFloat64: (list, numel) {
        for (var i = 0; i < numel; i++) {
          list[i] += s;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (var i = 0; i < n; i++) {
          t.storage.setFromDouble(i, t.storage.getAsDouble(i) + s);
        }
      },
    );
  }

  @override
  void _applyTensor(TensorBuffer tensor, TensorBuffer other) {
    if (tensor.dtype == DType.float32 && other.dtype == DType.float32) {
      final a = tensor.storage.data as Float32List;
      final b = other.storage.data as Float32List;
      SimdOps.add(a, b, a);
    } else if (tensor.dtype == DType.float64 && other.dtype == DType.float64) {
      final a = tensor.storage.data as Float64List;
      final b = other.storage.data as Float64List;
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        a[i] += b[i];
      }
    } else {
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        final a = tensor.storage.getAsDouble(i);
        final b = other.storage.getAsDouble(i);
        tensor.storage.setFromDouble(i, a + b);
      }
    }
  }
}

/// Subtracts a scalar or tensor from the input element-wise.
///
/// Equivalent to `torch.sub()` in PyTorch.
///
/// ```dart
/// final result = SubOp(scalar: 1.0)(tensor);  // tensor - 1.0
/// final result = SubOp.tensor(other)(tensor); // tensor - other
/// ```
class SubOp extends ArithmeticOp {
  /// Creates a subtraction operation with a scalar operand.
  SubOp({required double scalar}) : super._(scalar: scalar);

  /// Creates a subtraction operation with a tensor operand.
  SubOp.tensor(TensorBuffer other) : super._(other: other);

  @override
  String get name =>
      scalar != null ? 'Sub(scalar=$scalar)' : 'Sub(tensor=${other!.shape})';

  @override
  double operation(double a, double b) => a - b;

  @override
  void _applyScalar(TensorBuffer tensor, double s) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) => SimdOps.subtractScalar(list, s),
      onFloat64: (list, numel) {
        for (var i = 0; i < numel; i++) {
          list[i] -= s;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (var i = 0; i < n; i++) {
          t.storage.setFromDouble(i, t.storage.getAsDouble(i) - s);
        }
      },
    );
  }

  @override
  void _applyTensor(TensorBuffer tensor, TensorBuffer other) {
    if (tensor.dtype == DType.float32 && other.dtype == DType.float32) {
      final a = tensor.storage.data as Float32List;
      final b = other.storage.data as Float32List;
      SimdOps.subtract(a, b, a);
    } else if (tensor.dtype == DType.float64 && other.dtype == DType.float64) {
      final a = tensor.storage.data as Float64List;
      final b = other.storage.data as Float64List;
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        a[i] -= b[i];
      }
    } else {
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        final a = tensor.storage.getAsDouble(i);
        final b = other.storage.getAsDouble(i);
        tensor.storage.setFromDouble(i, a - b);
      }
    }
  }
}

/// Multiplies the input by a scalar or tensor element-wise.
///
/// Equivalent to `torch.mul()` in PyTorch.
///
/// ```dart
/// final result = MulOp(scalar: 2.0)(tensor);  // tensor * 2.0
/// final result = MulOp.tensor(other)(tensor); // tensor * other
/// ```
class MulOp extends ArithmeticOp {
  /// Creates a multiplication operation with a scalar operand.
  MulOp({required double scalar}) : super._(scalar: scalar);

  /// Creates a multiplication operation with a tensor operand.
  MulOp.tensor(TensorBuffer other) : super._(other: other);

  @override
  String get name =>
      scalar != null ? 'Mul(scalar=$scalar)' : 'Mul(tensor=${other!.shape})';

  @override
  double operation(double a, double b) => a * b;

  @override
  void _applyScalar(TensorBuffer tensor, double s) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) => SimdOps.multiplyScalar(list, s),
      onFloat64: (list, numel) {
        for (var i = 0; i < numel; i++) {
          list[i] *= s;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (var i = 0; i < n; i++) {
          t.storage.setFromDouble(i, t.storage.getAsDouble(i) * s);
        }
      },
    );
  }

  @override
  void _applyTensor(TensorBuffer tensor, TensorBuffer other) {
    if (tensor.dtype == DType.float32 && other.dtype == DType.float32) {
      final a = tensor.storage.data as Float32List;
      final b = other.storage.data as Float32List;
      SimdOps.multiply(a, b, a);
    } else if (tensor.dtype == DType.float64 && other.dtype == DType.float64) {
      final a = tensor.storage.data as Float64List;
      final b = other.storage.data as Float64List;
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        a[i] *= b[i];
      }
    } else {
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        final a = tensor.storage.getAsDouble(i);
        final b = other.storage.getAsDouble(i);
        tensor.storage.setFromDouble(i, a * b);
      }
    }
  }
}

/// Divides the input by a scalar or tensor element-wise.
///
/// Equivalent to `torch.div()` in PyTorch.
///
/// ```dart
/// final result = DivOp(scalar: 2.0)(tensor);  // tensor / 2.0
/// final result = DivOp.tensor(other)(tensor); // tensor / other
/// ```
class DivOp extends ArithmeticOp {
  /// Creates a division operation with a scalar operand.
  DivOp({required double scalar}) : super._(scalar: scalar);

  /// Creates a division operation with a tensor operand.
  DivOp.tensor(TensorBuffer other) : super._(other: other);

  @override
  String get name =>
      scalar != null ? 'Div(scalar=$scalar)' : 'Div(tensor=${other!.shape})';

  @override
  double operation(double a, double b) => a / b;

  @override
  void _applyScalar(TensorBuffer tensor, double s) {
    // Division by scalar is multiplication by reciprocal
    final invS = 1.0 / s;
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) => SimdOps.multiplyScalar(list, invS),
      onFloat64: (list, numel) {
        for (var i = 0; i < numel; i++) {
          list[i] *= invS;
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (var i = 0; i < n; i++) {
          t.storage.setFromDouble(i, t.storage.getAsDouble(i) * invS);
        }
      },
    );
  }

  @override
  void _applyTensor(TensorBuffer tensor, TensorBuffer other) {
    if (tensor.dtype == DType.float32 && other.dtype == DType.float32) {
      final a = tensor.storage.data as Float32List;
      final b = other.storage.data as Float32List;
      SimdOps.divide(a, b, a);
    } else if (tensor.dtype == DType.float64 && other.dtype == DType.float64) {
      final a = tensor.storage.data as Float64List;
      final b = other.storage.data as Float64List;
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        a[i] /= b[i];
      }
    } else {
      final n = tensor.numel;
      for (var i = 0; i < n; i++) {
        final a = tensor.storage.getAsDouble(i);
        final b = other.storage.getAsDouble(i);
        tensor.storage.setFromDouble(i, a / b);
      }
    }
  }
}

/// Raises each element to the given power.
///
/// Equivalent to `torch.pow()` in PyTorch.
///
/// ```dart
/// final result = PowOp(exponent: 2.0)(tensor);  // tensor ^ 2.0
/// ```
class PowOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// The exponent to raise each element to.
  final double exponent;

  /// Creates a power operation with the given [exponent].
  PowOp({required this.exponent});

  @override
  String get name => 'Pow(exponent=$exponent)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    supportsInPlace: true,
    requiresContiguous: true,
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _pow(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('PowOp.applyInPlace');
    }
    _pow(input);
  }

  void _pow(TensorBuffer tensor) {
    final numel = tensor.numel;
    final exp = exponent;
    final data = tensor.storage.data;

    // Dtype-specialized loops for better performance
    switch (tensor.dtype) {
      case DType.float32:
        final list = data as Float32List;
        for (int i = 0; i < numel; i++) {
          list[i] = math.pow(list[i], exp).toDouble();
        }
      case DType.float64:
        final list = data as Float64List;
        for (int i = 0; i < numel; i++) {
          list[i] = math.pow(list[i], exp).toDouble();
        }
      default:
        for (int i = 0; i < numel; i++) {
          final value = tensor.storage.getAsDouble(i);
          tensor.storage.setFromDouble(i, math.pow(value, exp).toDouble());
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
