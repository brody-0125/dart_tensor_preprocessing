import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../../utils/simd_ops.dart';
import '../transform_op.dart';

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
