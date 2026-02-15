import 'dart:math' as math;

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

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
