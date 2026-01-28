import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Fused resize + normalize operation that eliminates the intermediate tensor.
///
/// Combines bilinear resize and per-channel normalization into a single pass,
/// applying `(resized_value - mean) / std` directly during interpolation.
/// This avoids allocating a full intermediate tensor between resize and
/// normalize steps.
///
/// Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` inputs.
class ResizeNormalizeFusedOp extends TransformOp with RequiresContiguous {
  /// The target height.
  final int height;

  /// The target width.
  final int width;

  /// Per-channel mean values to subtract.
  final List<double> mean;

  /// Per-channel standard deviation values to divide by.
  final List<double> std;

  /// Whether to align corners during interpolation.
  final bool alignCorners;

  /// Creates a fused resize + normalize operation.
  ResizeNormalizeFusedOp({
    required this.height,
    required this.width,
    required this.mean,
    required this.std,
    this.alignCorners = false,
  }) {
    if (height <= 0 || width <= 0) {
      throw InvalidParameterException(
        'height/width',
        '$height x $width',
        'Must be positive',
      );
    }
    if (mean.length != std.length) {
      throw InvalidParameterException(
        'mean/std',
        'mean.length=${mean.length}, std.length=${std.length}',
        'Must have same length',
      );
    }
    if (mean.isEmpty) {
      throw InvalidParameterException(
        'mean/std',
        'empty',
        'Must have at least one channel',
      );
    }
    for (int i = 0; i < std.length; i++) {
      if (std[i] == 0) {
        throw InvalidParameterException(
          'std[$i]',
          '0',
          'Standard deviation cannot be zero',
        );
      }
    }
  }

  /// Creates a fused resize + normalize using ImageNet statistics.
  factory ResizeNormalizeFusedOp.imagenet({
    required int height,
    required int width,
    bool alignCorners = false,
  }) {
    return ResizeNormalizeFusedOp(
      height: height,
      width: width,
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
      alignCorners: alignCorners,
    );
  }

  @override
  String get name =>
      'ResizeNormalize(${height}x$width, mean=$mean, std=$std, alignCorners=$alignCorners)';

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
        message:
            'ResizeNormalizeFusedOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != mean.length) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but mean/std has ${mean.length}',
      );
    }

    final outputShape = computeOutputShape(shape);
    final output = TensorBuffer.uninitialized(outputShape, dtype: input.dtype);

    if (rank == 3) {
      _fusedResizeNormalize3D(contiguous, output);
    } else {
      _fusedResizeNormalize4D(contiguous, output);
    }

    return output;
  }

  void _fusedResizeNormalize3D(TensorBuffer input, TensorBuffer output) {
    final c = input.shape[0];
    final srcH = input.shape[1];
    final srcW = input.shape[2];
    final channelSize = srcH * srcW;
    final outChannelSize = height * width;

    final scaleY =
        alignCorners && height > 1 ? (srcH - 1) / (height - 1) : srcH / height;
    final scaleX =
        alignCorners && width > 1 ? (srcW - 1) / (width - 1) : srcW / width;

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;

        for (int ch = 0; ch < c; ch++) {
          final srcOffset = ch * channelSize;
          final dstOffset = ch * outChannelSize;
          final m = mean[ch];
          final invStd = 1.0 / std[ch];

          _bilinearNormalizeFloat32(
            inList,
            outList,
            srcH,
            srcW,
            srcOffset,
            dstOffset,
            scaleY,
            scaleX,
            m,
            invStd,
          );
        }
      default:
        for (int ch = 0; ch < c; ch++) {
          final srcOffset = ch * channelSize;
          final dstOffset = ch * outChannelSize;
          final m = mean[ch];
          final invStd = 1.0 / std[ch];

          _bilinearNormalizeGeneric(
            input,
            output,
            srcH,
            srcW,
            srcOffset,
            dstOffset,
            scaleY,
            scaleX,
            m,
            invStd,
          );
        }
    }
  }

  void _fusedResizeNormalize4D(TensorBuffer input, TensorBuffer output) {
    final n = input.shape[0];
    final c = input.shape[1];
    final srcH = input.shape[2];
    final srcW = input.shape[3];
    final channelSize = srcH * srcW;
    final batchSize = c * channelSize;
    final outChannelSize = height * width;
    final outBatchSize = c * outChannelSize;

    final scaleY =
        alignCorners && height > 1 ? (srcH - 1) / (height - 1) : srcH / height;
    final scaleX =
        alignCorners && width > 1 ? (srcW - 1) / (width - 1) : srcW / width;

    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;

        for (int batch = 0; batch < n; batch++) {
          for (int ch = 0; ch < c; ch++) {
            final srcOffset = batch * batchSize + ch * channelSize;
            final dstOffset = batch * outBatchSize + ch * outChannelSize;
            final m = mean[ch];
            final invStd = 1.0 / std[ch];

            _bilinearNormalizeFloat32(
              inList,
              outList,
              srcH,
              srcW,
              srcOffset,
              dstOffset,
              scaleY,
              scaleX,
              m,
              invStd,
            );
          }
        }
      default:
        for (int batch = 0; batch < n; batch++) {
          for (int ch = 0; ch < c; ch++) {
            final srcOffset = batch * batchSize + ch * channelSize;
            final dstOffset = batch * outBatchSize + ch * outChannelSize;
            final m = mean[ch];
            final invStd = 1.0 / std[ch];

            _bilinearNormalizeGeneric(
              input,
              output,
              srcH,
              srcW,
              srcOffset,
              dstOffset,
              scaleY,
              scaleX,
              m,
              invStd,
            );
          }
        }
    }
  }

  void _bilinearNormalizeFloat32(
    Float32List inList,
    Float32List outList,
    int srcH,
    int srcW,
    int srcOffset,
    int dstOffset,
    double scaleY,
    double scaleX,
    double mean,
    double invStd,
  ) {
    const blockSize = 64;

    for (int by = 0; by < height; by += blockSize) {
      final endY = math.min(by + blockSize, height);

      for (int bx = 0; bx < width; bx += blockSize) {
        final endX = math.min(bx + blockSize, width);

        for (int y = by; y < endY; y++) {
          final rawSrcY = alignCorners ? y * scaleY : (y + 0.5) * scaleY - 0.5;
          final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
          final y0 = srcY.floor().clamp(0, srcH - 1);
          final y1 = (y0 + 1).clamp(0, srcH - 1);
          final fy = srcY - y0;
          final oneMinusFy = 1 - fy;

          for (int x = bx; x < endX; x++) {
            final rawSrcX =
                alignCorners ? x * scaleX : (x + 0.5) * scaleX - 0.5;
            final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
            final x0 = srcX.floor().clamp(0, srcW - 1);
            final x1 = (x0 + 1).clamp(0, srcW - 1);
            final fx = srcX - x0;
            final oneMinusFx = 1 - fx;

            final v00 = inList[srcOffset + y0 * srcW + x0];
            final v01 = inList[srcOffset + y0 * srcW + x1];
            final v10 = inList[srcOffset + y1 * srcW + x0];
            final v11 = inList[srcOffset + y1 * srcW + x1];

            final resized = v00 * oneMinusFx * oneMinusFy +
                v01 * fx * oneMinusFy +
                v10 * oneMinusFx * fy +
                v11 * fx * fy;

            // Fused normalize: (resized - mean) / std = (resized - mean) * invStd
            outList[dstOffset + y * width + x] = (resized - mean) * invStd;
          }
        }
      }
    }
  }

  void _bilinearNormalizeGeneric(
    TensorBuffer input,
    TensorBuffer output,
    int srcH,
    int srcW,
    int srcOffset,
    int dstOffset,
    double scaleY,
    double scaleX,
    double mean,
    double invStd,
  ) {
    const blockSize = 64;

    for (int by = 0; by < height; by += blockSize) {
      final endY = math.min(by + blockSize, height);

      for (int bx = 0; bx < width; bx += blockSize) {
        final endX = math.min(bx + blockSize, width);

        for (int y = by; y < endY; y++) {
          final rawSrcY = alignCorners ? y * scaleY : (y + 0.5) * scaleY - 0.5;
          final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
          final y0 = srcY.floor().clamp(0, srcH - 1);
          final y1 = (y0 + 1).clamp(0, srcH - 1);
          final fy = srcY - y0;

          final oneMinusFy = 1 - fy;

          for (int x = bx; x < endX; x++) {
            final rawSrcX =
                alignCorners ? x * scaleX : (x + 0.5) * scaleX - 0.5;
            final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
            final x0 = srcX.floor().clamp(0, srcW - 1);
            final x1 = (x0 + 1).clamp(0, srcW - 1);
            final fx = srcX - x0;
            final oneMinusFx = 1 - fx;

            final v00 = input.storage.getAsDouble(srcOffset + y0 * srcW + x0);
            final v01 = input.storage.getAsDouble(srcOffset + y0 * srcW + x1);
            final v10 = input.storage.getAsDouble(srcOffset + y1 * srcW + x0);
            final v11 = input.storage.getAsDouble(srcOffset + y1 * srcW + x1);

            final resized = v00 * oneMinusFx * oneMinusFy +
                v01 * fx * oneMinusFy +
                v10 * oneMinusFx * fy +
                v11 * fx * fy;

            output.storage.setFromDouble(
                dstOffset + y * width + x, (resized - mean) * invStd);
          }
        }
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      return [inputShape[0], height, width];
    }
    return [inputShape[0], inputShape[1], height, width];
  }
}
