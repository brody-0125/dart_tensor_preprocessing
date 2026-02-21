import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Element-wise conditional selection between two tensors.
///
/// Equivalent to `torch.where()` in PyTorch and ONNX `Where` operator.
///
/// For each element, selects from [x] where [condition] is non-zero,
/// and from [y] where [condition] is zero.
///
/// ```dart
/// final result = tensorWhere(condition, x, y);
/// ```
TensorBuffer tensorWhere(TensorBuffer condition, TensorBuffer x, TensorBuffer y) {
  // Validate shapes match (v0.8.1: no broadcasting)
  if (condition.shape.length != x.shape.length) {
    throw ShapeMismatchException(
      actual: condition.shape,
      message: 'Where: condition shape ${condition.shape} does not match x shape ${x.shape}',
    );
  }
  for (int i = 0; i < condition.shape.length; i++) {
    if (condition.shape[i] != x.shape[i]) {
      throw ShapeMismatchException(
        actual: condition.shape,
        message: 'Where: condition shape ${condition.shape} does not match x shape ${x.shape}',
      );
    }
  }
  if (x.shape.length != y.shape.length) {
    throw ShapeMismatchException(
      actual: y.shape,
      message: 'Where: x shape ${x.shape} does not match y shape ${y.shape}',
    );
  }
  for (int i = 0; i < x.shape.length; i++) {
    if (x.shape[i] != y.shape[i]) {
      throw ShapeMismatchException(
        actual: y.shape,
        message: 'Where: x shape ${x.shape} does not match y shape ${y.shape}',
      );
    }
  }

  final condContiguous =
      condition.isContiguous ? condition : condition.contiguous();
  final xContiguous = x.isContiguous ? x : x.contiguous();
  final yContiguous = y.isContiguous ? y : y.contiguous();

  final output =
      TensorBuffer.uninitialized(List<int>.from(x.shape), dtype: x.dtype);
  final numel = output.numel;

  switch (x.dtype) {
    case DType.float32:
      final condData = condContiguous.storage.data as Float32List;
      final xData = xContiguous.storage.data as Float32List;
      final yData = yContiguous.storage.data as Float32List;
      final outData = output.storage.data as Float32List;

      for (int i = 0; i < numel; i++) {
        outData[i] = condData[i] != 0 ? xData[i] : yData[i];
      }

    case DType.float64:
      final condData = condContiguous.storage.data as Float64List;
      final xData = xContiguous.storage.data as Float64List;
      final yData = yContiguous.storage.data as Float64List;
      final outData = output.storage.data as Float64List;

      for (int i = 0; i < numel; i++) {
        outData[i] = condData[i] != 0 ? xData[i] : yData[i];
      }

    default:
      final condStorage = condContiguous.storage;
      final xStorage = xContiguous.storage;
      final yStorage = yContiguous.storage;
      final outStorage = output.storage;

      for (int i = 0; i < numel; i++) {
        final cond = condStorage.getAsDouble(i) != 0;
        outStorage.setFromDouble(
          i,
          cond ? xStorage.getAsDouble(i) : yStorage.getAsDouble(i),
        );
      }
  }

  return output;
}

/// TransformOp form of element-wise where.
///
/// The input tensor acts as `x`. The [condition] and [y] tensors are
/// provided at construction time.
///
/// ```dart
/// final result = WhereOp(condition: mask, y: fallback)(input);
/// ```
class WhereOp extends TransformOp {
  /// The condition tensor (non-zero = true).
  final TensorBuffer condition;

  /// The tensor to select from when condition is zero.
  final TensorBuffer y;

  /// Creates a Where operation.
  WhereOp({required this.condition, required this.y});

  @override
  String get name => 'Where';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        preservesShape: true,
        pytorchEquivalent: 'torch.where',
        onnxOpType: 'Where',
      );

  @override
  TensorBuffer apply(TensorBuffer input) => tensorWhere(condition, input, y);

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
