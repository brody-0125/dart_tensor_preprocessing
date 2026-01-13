import 'dart:math' as math;

import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Base class for unary math operations.
///
/// Applies a mathematical function element-wise to all tensor values.
abstract class UnaryMathOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// Applies the mathematical function to a single value.
  double operation(double value);

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
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      tensor.storage.setFromDouble(i, operation(value));
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Computes the absolute value of each element.
///
/// Equivalent to `torch.abs()` in PyTorch.
///
/// ```dart
/// final result = AbsOp()(tensor);  // |tensor|
/// ```
class AbsOp extends UnaryMathOp {
  /// Creates an absolute value operation.
  AbsOp();

  @override
  String get name => 'Abs';

  @override
  double operation(double value) => value.abs();
}

/// Negates each element.
///
/// Equivalent to `torch.neg()` or `-tensor` in PyTorch.
///
/// ```dart
/// final result = NegOp()(tensor);  // -tensor
/// ```
class NegOp extends UnaryMathOp {
  /// Creates a negation operation.
  NegOp();

  @override
  String get name => 'Neg';

  @override
  double operation(double value) => -value;
}

/// Computes the square root of each element.
///
/// Equivalent to `torch.sqrt()` in PyTorch.
///
/// Note: Negative inputs will produce NaN values.
///
/// ```dart
/// final result = SqrtOp()(tensor);  // sqrt(tensor)
/// ```
class SqrtOp extends UnaryMathOp {
  /// Creates a square root operation.
  SqrtOp();

  @override
  String get name => 'Sqrt';

  @override
  double operation(double value) => math.sqrt(value);
}

/// Computes the exponential (e^x) of each element.
///
/// Equivalent to `torch.exp()` in PyTorch.
///
/// ```dart
/// final result = ExpOp()(tensor);  // e^tensor
/// ```
class ExpOp extends UnaryMathOp {
  /// Creates an exponential operation.
  ExpOp();

  @override
  String get name => 'Exp';

  @override
  double operation(double value) => math.exp(value);
}

/// Computes the natural logarithm of each element.
///
/// Equivalent to `torch.log()` in PyTorch.
///
/// Note: Zero inputs will produce -infinity, negative inputs will produce NaN.
///
/// ```dart
/// final result = LogOp()(tensor);  // ln(tensor)
/// ```
class LogOp extends UnaryMathOp {
  /// Creates a natural logarithm operation.
  LogOp();

  @override
  String get name => 'Log';

  @override
  double operation(double value) => math.log(value);
}
