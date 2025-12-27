import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../core/tensor_storage.dart';
import '../exceptions/tensor_exceptions.dart';
import '../ops/transform_op.dart';

/// A chain of tensor transformation operations that can be executed
/// synchronously or asynchronously.
///
/// [TensorPipeline] provides a declarative way to compose preprocessing
/// operations. Pipelines are immutable; methods like [append] return
/// new pipeline instances.
///
/// ## Example
///
/// ```dart
/// // Create a preprocessing pipeline
/// final pipeline = TensorPipeline([
///   ResizeOp(height: 224, width: 224),
///   ToTensorOp(normalize: true),
///   NormalizeOp.imagenet(),
///   UnsqueezeOp.batch(),
/// ], name: 'ImageNet Preprocessing');
///
/// // Run synchronously
/// final result = pipeline.run(input);
///
/// // Run asynchronously in an isolate
/// final result = await pipeline.runAsync(input);
///
/// // Use as a function
/// final result = pipeline(input);
/// ```
class TensorPipeline {
  final List<TransformOp> _operations;

  /// Optional name for this pipeline (for debugging).
  final String? name;

  /// Creates a [TensorPipeline] with the given [operations].
  ///
  /// Throws [EmptyPipelineException] if [operations] is empty.
  TensorPipeline(List<TransformOp> operations, {this.name})
      : _operations = List.unmodifiable(operations) {
    if (operations.isEmpty) {
      throw const EmptyPipelineException();
    }
  }

  /// The number of operations in this pipeline.
  int get length => _operations.length;

  /// The list of operations in this pipeline.
  List<TransformOp> get operations => _operations;

  /// Executes the pipeline synchronously on [input].
  ///
  /// Each operation is applied in sequence, with the output of one
  /// becoming the input of the next.
  TensorBuffer run(TensorBuffer input) {
    var result = input;
    for (final op in _operations) {
      result = op.apply(result);
    }
    return result;
  }

  /// Executes the pipeline asynchronously in a separate isolate.
  ///
  /// This is useful for avoiding UI jank when processing large tensors.
  /// The tensor is serialized, sent to the isolate, processed, and
  /// the result is deserialized back.
  Future<TensorBuffer> runAsync(TensorBuffer input) async {
    final inputData = _serializeTensor(input);

    final resultData = await Isolate.run(() {
      final tensor = _deserializeTensor(inputData);
      var result = tensor;
      for (final op in _operations) {
        result = op.apply(result);
      }
      return _serializeTensor(result);
    });

    return _deserializeTensor(resultData);
  }

  /// Computes the output shape for a given [inputShape].
  ///
  /// This can be used to validate pipeline compatibility or pre-allocate
  /// output buffers without actually running the pipeline.
  List<int> computeOutputShape(List<int> inputShape) {
    var shape = inputShape;
    for (final op in _operations) {
      shape = op.computeOutputShape(shape);
    }
    return shape;
  }

  /// Returns true if this pipeline is valid for the given [inputShape].
  ///
  /// Validation checks that all operations are compatible with their
  /// respective input shapes.
  bool validate(List<int> inputShape) {
    try {
      computeOutputShape(inputShape);
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Returns a new pipeline with [op] appended to the end.
  TensorPipeline append(TransformOp op) {
    return TensorPipeline([..._operations, op], name: name);
  }

  /// Returns a new pipeline with [op] prepended to the beginning.
  TensorPipeline prepend(TransformOp op) {
    return TensorPipeline([op, ..._operations], name: name);
  }

  /// Returns a new pipeline combining this pipeline with [other].
  TensorPipeline concat(TensorPipeline other) {
    return TensorPipeline([..._operations, ...other._operations]);
  }

  /// Calls [run] on [input]. Enables using the pipeline as a function.
  TensorBuffer call(TensorBuffer input) => run(input);

  /// Concatenates two pipelines.
  TensorPipeline operator +(TensorPipeline other) => concat(other);

  /// Appends an operation to the pipeline.
  TensorPipeline operator >>(TransformOp op) => append(op);

  @override
  String toString() {
    final opNames = _operations.map((op) => op.name).join(' -> ');
    return name != null ? 'Pipeline($name): $opNames' : 'Pipeline: $opNames';
  }
}

class _SerializedTensor {
  final List<int> dataBytes;
  final List<int> shape;
  final int dtypeIndex;

  _SerializedTensor(this.dataBytes, this.shape, this.dtypeIndex);
}

_SerializedTensor _serializeTensor(TensorBuffer tensor) {
  final contiguous = tensor.contiguous();
  final data = contiguous.data;

  final buffer = data.buffer;
  final bytes = buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

  return _SerializedTensor(
    bytes.toList(),
    contiguous.shape.toList(),
    contiguous.dtype.index,
  );
}

TensorBuffer _deserializeTensor(_SerializedTensor data) {
  final dtype = DType.values[data.dtypeIndex];
  final bytes = Uint8List.fromList(data.dataBytes);
  final typedData = _bytesToTypedData(bytes, dtype);

  return TensorBuffer(
    storage: TensorStorage(typedData, dtype),
    shape: data.shape,
  );
}

TypedData _bytesToTypedData(Uint8List bytes, DType dtype) {
  final buffer = bytes.buffer;
  return switch (dtype) {
    DType.float32 => buffer.asFloat32List(),
    DType.float64 => buffer.asFloat64List(),
    DType.int8 => buffer.asInt8List(),
    DType.int16 => buffer.asInt16List(),
    DType.int32 => buffer.asInt32List(),
    DType.int64 => buffer.asInt64List(),
    DType.uint8 => bytes,
    DType.uint16 => buffer.asUint16List(),
    DType.uint32 => buffer.asUint32List(),
    DType.uint64 => buffer.asUint64List(),
  };
}
