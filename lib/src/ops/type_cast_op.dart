import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../core/tensor_storage.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Casts tensor element values to a different data type.
class TypeCastOp extends TransformOp with RequiresContiguous {
  /// The target data type to cast to.
  final DType targetDtype;

  /// Creates a type cast operation to [targetDtype].
  TypeCastOp(this.targetDtype);

  /// Casts to 32-bit float.
  factory TypeCastOp.toFloat32() => TypeCastOp(DType.float32);

  /// Casts to 64-bit float.
  factory TypeCastOp.toFloat64() => TypeCastOp(DType.float64);

  /// Casts to 8-bit unsigned integer.
  factory TypeCastOp.toUint8() => TypeCastOp(DType.uint8);

  /// Casts to 32-bit signed integer.
  factory TypeCastOp.toInt32() => TypeCastOp(DType.int32);

  /// Casts to 64-bit signed integer.
  factory TypeCastOp.toInt64() => TypeCastOp(DType.int64);

  @override
  String get name => 'TypeCast($targetDtype)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    if (input.dtype == targetDtype) {
      return input;
    }

    final contiguous = ensureContiguous(input);
    final numel = contiguous.numel;

    final newData = targetDtype.createBuffer(numel);

    for (int i = 0; i < numel; i++) {
      final value = contiguous.storage.getAsDouble(i);
      _setTypedDataValue(newData, i, value);
    }

    return TensorBuffer(
      storage: TensorStorage(newData, targetDtype),
      shape: contiguous.shape.toList(),
    );
  }

  void _setTypedDataValue(TypedData data, int index, double value) {
    switch (data) {
      case final Float32List list:
        list[index] = value;
      case final Float64List list:
        list[index] = value;
      case final Int8List list:
        list[index] = value.round().clamp(-128, 127);
      case final Int16List list:
        list[index] = value.round().clamp(-32768, 32767);
      case final Int32List list:
        list[index] = value.round();
      case final Int64List list:
        list[index] = value.round();
      case final Uint8List list:
        list[index] = value.round().clamp(0, 255);
      case final Uint16List list:
        list[index] = value.round().clamp(0, 65535);
      case final Uint32List list:
        list[index] = value.round().clamp(0, 4294967295);
      case final Uint64List list:
        list[index] = value.round();
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

/// Converts an image tensor from HWC/NHWC to CHW/NCHW format.
///
/// Optionally normalizes pixel values from `[0, 255]` to `[0, 1]`.
class ToTensorOp extends TransformOp with RequiresContiguous {
  /// Whether to normalize values to `[0, 1]`.
  final bool normalize;

  /// Creates a ToTensor operation.
  ToTensorOp({this.normalize = true});

  @override
  String get name => 'ToTensor(normalize=$normalize)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final shape = contiguous.shape;

    if (shape.length != 3 && shape.length != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'ToTensorOp expects HWC or NHWC input',
      );
    }

    return shape.length == 3
        ? _convertHwcToChw(contiguous)
        : _convertNhwcToNchw(contiguous);
  }

  TensorBuffer _convertHwcToChw(TensorBuffer input) {
    final h = input.shape[0];
    final w = input.shape[1];
    final c = input.shape[2];

    final numel = h * w * c;
    final outputData = Float32List(numel);
    final channelSize = h * w;
    final scale = normalize ? 1.0 / 255.0 : 1.0;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final hwcBase = (y * w + x) * c;
        for (int ch = 0; ch < c; ch++) {
          final srcIdx = hwcBase + ch;
          final dstIdx = ch * channelSize + y * w + x;
          final value = input.storage.getAsDouble(srcIdx);
          outputData[dstIdx] = value * scale;
        }
      }
    }

    return TensorBuffer(
      storage: TensorStorage(outputData, DType.float32),
      shape: [c, h, w],
    );
  }

  TensorBuffer _convertNhwcToNchw(TensorBuffer input) {
    final n = input.shape[0];
    final h = input.shape[1];
    final w = input.shape[2];
    final c = input.shape[3];

    final batchSize = h * w * c;
    final numel = n * batchSize;
    final outputData = Float32List(numel);
    final channelSize = h * w;
    final scale = normalize ? 1.0 / 255.0 : 1.0;

    for (int batch = 0; batch < n; batch++) {
      final srcBatchOffset = batch * batchSize;
      final dstBatchOffset = batch * batchSize;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final hwcBase = srcBatchOffset + (y * w + x) * c;
          for (int ch = 0; ch < c; ch++) {
            final srcIdx = hwcBase + ch;
            final dstIdx = dstBatchOffset + ch * channelSize + y * w + x;
            final value = input.storage.getAsDouble(srcIdx);
            outputData[dstIdx] = value * scale;
          }
        }
      }
    }

    return TensorBuffer(
      storage: TensorStorage(outputData, DType.float32),
      shape: [n, c, h, w],
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [inputShape[2], inputShape[0], inputShape[1]];
    } else if (inputShape.length == 4) {
      return [inputShape[0], inputShape[3], inputShape[1], inputShape[2]];
    }
    throw ShapeMismatchException(
      actual: inputShape,
      message: 'ToTensorOp expects HWC or NHWC input',
    );
  }
}

/// Converts a tensor from CHW/NCHW to HWC/NHWC image format.
///
/// Optionally denormalizes values from `[0, 1]` to `[0, 255]`.
class ToImageOp extends TransformOp with RequiresContiguous {
  /// Whether to denormalize values to `[0, 255]`.
  final bool denormalize;

  /// Creates a ToImage operation.
  ToImageOp({this.denormalize = true});

  @override
  String get name => 'ToImage(denormalize=$denormalize)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final shape = contiguous.shape;

    if (shape.length != 3 && shape.length != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'ToImageOp expects CHW or NCHW input',
      );
    }

    return shape.length == 3
        ? _convertChwToHwc(contiguous)
        : _convertNchwToNhwc(contiguous);
  }

  TensorBuffer _convertChwToHwc(TensorBuffer input) {
    final c = input.shape[0];
    final h = input.shape[1];
    final w = input.shape[2];

    final numel = c * h * w;
    final outputData = Uint8List(numel);
    final channelSize = h * w;
    final scale = denormalize ? 255.0 : 1.0;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final hwcBase = (y * w + x) * c;
        for (int ch = 0; ch < c; ch++) {
          final srcIdx = ch * channelSize + y * w + x;
          final dstIdx = hwcBase + ch;
          final value = input.storage.getAsDouble(srcIdx);
          outputData[dstIdx] = (value * scale).round().clamp(0, 255);
        }
      }
    }

    return TensorBuffer(
      storage: TensorStorage(outputData, DType.uint8),
      shape: [h, w, c],
    );
  }

  TensorBuffer _convertNchwToNhwc(TensorBuffer input) {
    final n = input.shape[0];
    final c = input.shape[1];
    final h = input.shape[2];
    final w = input.shape[3];

    final batchSize = c * h * w;
    final numel = n * batchSize;
    final outputData = Uint8List(numel);
    final channelSize = h * w;
    final scale = denormalize ? 255.0 : 1.0;

    for (int batch = 0; batch < n; batch++) {
      final srcBatchOffset = batch * batchSize;
      final dstBatchOffset = batch * batchSize;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final hwcBase = dstBatchOffset + (y * w + x) * c;
          for (int ch = 0; ch < c; ch++) {
            final srcIdx = srcBatchOffset + ch * channelSize + y * w + x;
            final dstIdx = hwcBase + ch;
            final value = input.storage.getAsDouble(srcIdx);
            outputData[dstIdx] = (value * scale).round().clamp(0, 255);
          }
        }
      }
    }

    return TensorBuffer(
      storage: TensorStorage(outputData, DType.uint8),
      shape: [n, h, w, c],
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [inputShape[1], inputShape[2], inputShape[0]];
    } else if (inputShape.length == 4) {
      return [inputShape[0], inputShape[2], inputShape[3], inputShape[1]];
    }
    throw ShapeMismatchException(
      actual: inputShape,
      message: 'ToImageOp expects CHW or NCHW input',
    );
  }
}
