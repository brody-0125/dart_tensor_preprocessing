import 'dart:math' as math;

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

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
