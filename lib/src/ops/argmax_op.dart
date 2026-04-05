import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../core/tensor_buffer_reduce.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Base class for argmax/argmin operations that share axis reduction logic.
abstract class _ArgReduceOp extends TransformOp {
  /// The axis along which to find the extreme value. Supports negative indexing.
  final int axis;

  /// If `true`, the reduced dimension is retained with size 1.
  final bool keepDims;

  _ArgReduceOp({required this.axis, this.keepDims = false});

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final rank = inputShape.length;
    final normalizedAxis = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -rank,
        max: rank - 1,
        dimension: '$name axis',
      );
    }

    final outputShape = <int>[];
    for (int d = 0; d < rank; d++) {
      if (d == normalizedAxis) {
        if (keepDims) outputShape.add(1);
      } else {
        outputShape.add(inputShape[d]);
      }
    }
    return outputShape.isEmpty ? [1] : outputShape;
  }
}

/// Selects the index of the maximum value along an axis.
///
/// Equivalent to `torch.argmax()` in PyTorch and the ONNX `ArgMax` operator.
///
/// The output shape is the same as the input shape with the specified axis
/// removed (or retained with size 1 if [keepDims] is true). Output dtype
/// is always [DType.int64].
///
/// ```dart
/// final op = ArgMaxOp(axis: 1);
/// final indices = op.apply(tensor); // DType.int64
/// ```
class ArgMaxOp extends _ArgReduceOp {
  /// Creates an ArgMax operation.
  ///
  /// - [axis]: Axis along which to operate.
  /// - [keepDims]: Whether to retain the reduced dimension (default: `false`).
  ArgMaxOp({required super.axis, super.keepDims});

  @override
  String get name => 'ArgMax';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    modifiesDType: true,
    pytorchEquivalent: 'torch.argmax',
    onnxOpType: 'ArgMax',
    onnxOpsetVersion: 13,
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    return input.argmaxAxis(axis, keepDims: keepDims);
  }
}

/// Selects the index of the minimum value along an axis.
///
/// Equivalent to `torch.argmin()` in PyTorch and the ONNX `ArgMin` operator.
///
/// The output shape is the same as the input shape with the specified axis
/// removed (or retained with size 1 if [keepDims] is true). Output dtype
/// is always [DType.int64].
///
/// ```dart
/// final op = ArgMinOp(axis: 1);
/// final indices = op.apply(tensor); // DType.int64
/// ```
class ArgMinOp extends _ArgReduceOp {
  /// Creates an ArgMin operation.
  ///
  /// - [axis]: Axis along which to operate.
  /// - [keepDims]: Whether to retain the reduced dimension (default: `false`).
  ArgMinOp({required super.axis, super.keepDims});

  @override
  String get name => 'ArgMin';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    modifiesDType: true,
    pytorchEquivalent: 'torch.argmin',
    onnxOpType: 'ArgMin',
    onnxOpsetVersion: 13,
  );

  @override
  TensorBuffer apply(TensorBuffer input) {
    return input.argminAxis(axis, keepDims: keepDims);
  }
}
