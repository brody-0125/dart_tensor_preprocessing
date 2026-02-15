import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Crops a tensor from the center to the specified dimensions.
class CenterCropOp extends TransformOp with RequiresContiguous {
  /// The target crop height.
  final int height;

  /// The target crop width.
  final int width;

  /// Creates a center crop operation with the specified dimensions.
  CenterCropOp({required this.height, required this.width}) {
    if (height <= 0 || width <= 0) {
      throw InvalidParameterException(
        'height/width',
        '$height x $width',
        'Must be positive',
      );
    }
  }

  @override
  String get name => 'CenterCrop(${height}x$width)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: false,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final shape = contiguous.shape;
    final rank = shape.length;

    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'CenterCropOp requires 3D or 4D tensor',
      );
    }

    final srcH = rank == 3 ? shape[1] : shape[2];
    final srcW = rank == 3 ? shape[2] : shape[3];

    if (height > srcH || width > srcW) {
      throw InvalidParameterException(
        'crop size',
        '$height x $width',
        'Cannot be larger than input size $srcH x $srcW',
      );
    }

    final startY = (srcH - height) ~/ 2;
    final startX = (srcW - width) ~/ 2;

    return rank == 3
        ? _crop3D(contiguous, startY, startX)
        : _crop4D(contiguous, startY, startX);
  }

  TensorBuffer _crop3D(TensorBuffer input, int startY, int startX) {
    final c = input.shape[0];
    final srcH = input.shape[1];
    final srcW = input.shape[2];

    final output =
        TensorBuffer.uninitialized([c, height, width], dtype: input.dtype);

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        final srcChannelStride = srcH * srcW;
        final dstChannelStride = height * width;
        for (int ch = 0; ch < c; ch++) {
          final srcChOffset = ch * srcChannelStride;
          final dstChOffset = ch * dstChannelStride;
          for (int y = 0; y < height; y++) {
            final srcRowOffset = srcChOffset + (startY + y) * srcW + startX;
            final dstRowOffset = dstChOffset + y * width;
            // Row-wise copy for better cache performance
            for (int x = 0; x < width; x++) {
              outList[dstRowOffset + x] = inList[srcRowOffset + x];
            }
          }
        }
      default:
        // Generic fallback
        for (int ch = 0; ch < c; ch++) {
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              final srcIdx =
                  ch * srcH * srcW + (startY + y) * srcW + (startX + x);
              final dstIdx = ch * height * width + y * width + x;
              final value = input.storage.getAsDouble(srcIdx);
              output.storage.setFromDouble(dstIdx, value);
            }
          }
        }
    }

    return output;
  }

  TensorBuffer _crop4D(TensorBuffer input, int startY, int startX) {
    final n = input.shape[0];
    final c = input.shape[1];
    final srcH = input.shape[2];
    final srcW = input.shape[3];

    final output =
        TensorBuffer.uninitialized([n, c, height, width], dtype: input.dtype);

    final srcBatchStride = c * srcH * srcW;
    final dstBatchStride = c * height * width;
    final srcChannelStride = srcH * srcW;
    final dstChannelStride = height * width;

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int batch = 0; batch < n; batch++) {
          final srcBatchOffset = batch * srcBatchStride;
          final dstBatchOffset = batch * dstBatchStride;
          for (int ch = 0; ch < c; ch++) {
            final srcChOffset = srcBatchOffset + ch * srcChannelStride;
            final dstChOffset = dstBatchOffset + ch * dstChannelStride;
            for (int y = 0; y < height; y++) {
              final srcRowOffset = srcChOffset + (startY + y) * srcW + startX;
              final dstRowOffset = dstChOffset + y * width;
              // Row-wise copy for better cache performance
              for (int x = 0; x < width; x++) {
                outList[dstRowOffset + x] = inList[srcRowOffset + x];
              }
            }
          }
        }
      default:
        // Generic fallback
        for (int batch = 0; batch < n; batch++) {
          for (int ch = 0; ch < c; ch++) {
            for (int y = 0; y < height; y++) {
              for (int x = 0; x < width; x++) {
                final srcIdx = batch * srcBatchStride +
                    ch * srcChannelStride +
                    (startY + y) * srcW +
                    (startX + x);
                final dstIdx = batch * dstBatchStride +
                    ch * dstChannelStride +
                    y * width +
                    x;
                final value = input.storage.getAsDouble(srcIdx);
                output.storage.setFromDouble(dstIdx, value);
              }
            }
          }
        }
    }

    return output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [inputShape[0], height, width];
    } else if (inputShape.length == 4) {
      return [inputShape[0], inputShape[1], height, width];
    }
    throw ShapeMismatchException(
      actual: inputShape,
      message: 'CenterCropOp requires 3D or 4D tensor',
    );
  }
}
