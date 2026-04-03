import 'dart:math' as math;

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

/// Scaled Exponential Linear Unit (SELU) activation function.
///
/// Equivalent to `F.selu()` in PyTorch and ONNX `Selu` operator.
///
/// ## Formula
///
/// `selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))`
///
/// Where:
/// - `alpha = 1.6732632423543772`
/// - `scale = 1.0507009873554805`
///
/// ```dart
/// final result = SELUOp()(tensor);
/// ```
class SELUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Fixed alpha constant for SELU.
  static const double alpha = 1.6732632423543772;

  /// Fixed scale constant for SELU.
  static const double scale = 1.0507009873554805;

  /// Creates a SELU operation.
  SELUOp();

  @override
  String get name => 'SELU';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    supportsInPlace: true,
    requiresContiguous: true,
    pytorchEquivalent: 'F.selu',
    onnxOpType: 'Selu',
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _selu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('SELUOp.applyInPlace');
    }
    _selu(input);
  }

  void _selu(TensorBuffer tensor) {
    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x > 0 ? scale * x : scale * alpha * (math.exp(x) - 1);
        }
      },
      onFloat64: (list, numel) {
        for (int i = 0; i < numel; i++) {
          final x = list[i];
          list[i] = x > 0 ? scale * x : scale * alpha * (math.exp(x) - 1);
        }
      },
      fallback: (t) {
        final n = t.numel;
        for (int i = 0; i < n; i++) {
          final x = t.storage.getAsDouble(i);
          t.storage.setFromDouble(
            i,
            x > 0 ? scale * x : scale * alpha * (math.exp(x) - 1),
          );
        }
      },
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
