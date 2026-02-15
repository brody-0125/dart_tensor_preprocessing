import 'dart:math' as math;

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

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
