import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'math_op.dart';
import 'transform_op.dart';

// ============================================================================
// Unary Trigonometric Operations
// ============================================================================

/// Computes the sine of each element (in radians).
///
/// Equivalent to `torch.sin()` in PyTorch and the ONNX `Sin` operator.
///
/// ```dart
/// final result = SinOp()(tensor);  // sin(tensor)
/// ```
class SinOp extends UnaryMathOp {
  /// Creates a sine operation.
  SinOp();

  @override
  String get name => 'Sin';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.sin',
        onnxOpType: 'Sin',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.sin(value);
}

/// Computes the cosine of each element (in radians).
///
/// Equivalent to `torch.cos()` in PyTorch and the ONNX `Cos` operator.
///
/// ```dart
/// final result = CosOp()(tensor);  // cos(tensor)
/// ```
class CosOp extends UnaryMathOp {
  /// Creates a cosine operation.
  CosOp();

  @override
  String get name => 'Cos';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.cos',
        onnxOpType: 'Cos',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.cos(value);
}

/// Computes the tangent of each element (in radians).
///
/// Equivalent to `torch.tan()` in PyTorch and the ONNX `Tan` operator.
///
/// ```dart
/// final result = TanOp()(tensor);  // tan(tensor)
/// ```
class TanOp extends UnaryMathOp {
  /// Creates a tangent operation.
  TanOp();

  @override
  String get name => 'Tan';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.tan',
        onnxOpType: 'Tan',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.tan(value);
}

/// Computes the arcsine (inverse sine) of each element.
///
/// Input values should be in the range [-1, 1]. Values outside this range
/// will produce NaN.
///
/// Equivalent to `torch.asin()` in PyTorch and the ONNX `Asin` operator.
///
/// ```dart
/// final result = AsinOp()(tensor);  // asin(tensor)
/// ```
class AsinOp extends UnaryMathOp {
  /// Creates an arcsine operation.
  AsinOp();

  @override
  String get name => 'Asin';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.asin',
        onnxOpType: 'Asin',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.asin(value);
}

/// Computes the arccosine (inverse cosine) of each element.
///
/// Input values should be in the range [-1, 1]. Values outside this range
/// will produce NaN.
///
/// Equivalent to `torch.acos()` in PyTorch and the ONNX `Acos` operator.
///
/// ```dart
/// final result = AcosOp()(tensor);  // acos(tensor)
/// ```
class AcosOp extends UnaryMathOp {
  /// Creates an arccosine operation.
  AcosOp();

  @override
  String get name => 'Acos';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.acos',
        onnxOpType: 'Acos',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.acos(value);
}

/// Computes the arctangent (inverse tangent) of each element.
///
/// Equivalent to `torch.atan()` in PyTorch and the ONNX `Atan` operator.
///
/// ```dart
/// final result = AtanOp()(tensor);  // atan(tensor)
/// ```
class AtanOp extends UnaryMathOp {
  /// Creates an arctangent operation.
  AtanOp();

  @override
  String get name => 'Atan';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.atan',
        onnxOpType: 'Atan',
        onnxOpsetVersion: 7,
      );

  @override
  double operation(double value) => math.atan(value);
}

// ============================================================================
// Binary Trigonometric Operation: Atan2
// ============================================================================

/// Computes the element-wise arctangent of y/x, considering the signs of
/// both arguments to determine the correct quadrant.
///
/// Takes a second operand (scalar or tensor) representing the x-coordinate,
/// while the input tensor represents the y-coordinate.
///
/// Equivalent to `torch.atan2()` in PyTorch.
///
/// ```dart
/// final result = Atan2Op(scalar: 1.0)(yTensor);     // atan2(y, 1.0)
/// final result = Atan2Op.tensor(xTensor)(yTensor);   // atan2(y, x)
/// ```
class Atan2Op extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Scalar x-operand (if using scalar mode).
  final double? scalar;

  /// Tensor x-operand (if using tensor mode).
  final TensorBuffer? other;

  /// Creates an atan2 operation with a scalar x-operand.
  Atan2Op({required double this.scalar}) : other = null;

  /// Creates an atan2 operation with a tensor x-operand.
  Atan2Op.tensor(TensorBuffer this.other) : scalar = null;

  @override
  String get name => scalar != null
      ? 'Atan2(scalar=$scalar)'
      : 'Atan2(tensor=${other!.shape})';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.atan2',
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
              'Tensor shapes must match for atan2: ${tensor.shape} vs ${otherContiguous.shape}',
        );
      }
      _applyTensor(tensor, otherContiguous);
    }
  }

  void _applyScalar(TensorBuffer tensor, double x) {
    final numel = tensor.numel;
    switch (tensor.dtype) {
      case DType.float32:
        final list = tensor.storage.data as Float32List;
        for (int i = 0; i < numel; i++) {
          list[i] = math.atan2(list[i], x);
        }
      case DType.float64:
        final list = tensor.storage.data as Float64List;
        for (int i = 0; i < numel; i++) {
          list[i] = math.atan2(list[i], x);
        }
      default:
        for (int i = 0; i < numel; i++) {
          final y = tensor.storage.getAsDouble(i);
          tensor.storage.setFromDouble(i, math.atan2(y, x));
        }
    }
  }

  void _applyTensor(TensorBuffer tensor, TensorBuffer other) {
    final numel = tensor.numel;
    if (tensor.dtype == DType.float32 && other.dtype == DType.float32) {
      final yList = tensor.storage.data as Float32List;
      final xList = other.storage.data as Float32List;
      for (int i = 0; i < numel; i++) {
        yList[i] = math.atan2(yList[i], xList[i]);
      }
    } else if (tensor.dtype == DType.float64 && other.dtype == DType.float64) {
      final yList = tensor.storage.data as Float64List;
      final xList = other.storage.data as Float64List;
      for (int i = 0; i < numel; i++) {
        yList[i] = math.atan2(yList[i], xList[i]);
      }
    } else {
      for (int i = 0; i < numel; i++) {
        final y = tensor.storage.getAsDouble(i);
        final x = other.storage.getAsDouble(i);
        tensor.storage.setFromDouble(i, math.atan2(y, x));
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
