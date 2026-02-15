import 'dart:math' as math;

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

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
