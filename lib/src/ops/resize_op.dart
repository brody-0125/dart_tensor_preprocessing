import 'dart:math' as math;

import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Interpolation algorithm used for image resizing.
enum InterpolationMode {
  /// Nearest-neighbor interpolation (fastest, lowest quality).
  nearest,

  /// Bilinear interpolation (balanced speed and quality).
  bilinear,

  /// Bicubic interpolation (slowest, highest quality).
  bicubic,
}

/// Resizes a tensor to a fixed [height] and [width].
///
/// Supports 3D tensors `[C, H, W]` and 4D tensors `[N, C, H, W]`.
class ResizeOp extends TransformOp with RequiresContiguous {
  /// The target height.
  final int height;

  /// The target width.
  final int width;

  /// The interpolation mode.
  final InterpolationMode mode;

  /// Whether to align corners during interpolation.
  final bool alignCorners;

  /// Creates a resize operation with the specified dimensions.
  ResizeOp({
    required this.height,
    required this.width,
    this.mode = InterpolationMode.bilinear,
    this.alignCorners = false,
  }) {
    if (height <= 0 || width <= 0) {
      throw InvalidParameterException(
        'height/width',
        '$height x $width',
        'Must be positive',
      );
    }
  }

  @override
  String get name => 'Resize(${height}x$width, $mode)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    final rank = contiguous.rank;

    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: contiguous.shape,
        message: 'ResizeOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    return rank == 3 ? _resize3D(contiguous) : _resize4D(contiguous);
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
      message: 'ResizeOp requires 3D or 4D tensor',
    );
  }

  TensorBuffer _resize3D(TensorBuffer input) {
    final c = input.shape[0];
    final srcH = input.shape[1];
    final srcW = input.shape[2];

    final outputShape = [c, height, width];
    final output = TensorBuffer.zeros(outputShape, dtype: input.dtype);

    _resizeChannels(
      input: input,
      output: output,
      channels: c,
      srcH: srcH,
      srcW: srcW,
      batchOffset: 0,
      outputBatchOffset: 0,
    );

    return output;
  }

  TensorBuffer _resize4D(TensorBuffer input) {
    final n = input.shape[0];
    final c = input.shape[1];
    final srcH = input.shape[2];
    final srcW = input.shape[3];

    final outputShape = [n, c, height, width];
    final output = TensorBuffer.zeros(outputShape, dtype: input.dtype);

    final srcBatchStride = c * srcH * srcW;
    final dstBatchStride = c * height * width;

    for (int batch = 0; batch < n; batch++) {
      _resizeChannels(
        input: input,
        output: output,
        channels: c,
        srcH: srcH,
        srcW: srcW,
        batchOffset: batch * srcBatchStride,
        outputBatchOffset: batch * dstBatchStride,
      );
    }

    return output;
  }

  void _resizeChannels({
    required TensorBuffer input,
    required TensorBuffer output,
    required int channels,
    required int srcH,
    required int srcW,
    required int batchOffset,
    required int outputBatchOffset,
  }) {
    final srcChannelStride = srcH * srcW;
    final dstChannelStride = height * width;

    for (int ch = 0; ch < channels; ch++) {
      final srcOffset = batchOffset + ch * srcChannelStride;
      final dstOffset = outputBatchOffset + ch * dstChannelStride;

      switch (mode) {
        case InterpolationMode.nearest:
          _resizeNearest(
            input: input,
            output: output,
            srcH: srcH,
            srcW: srcW,
            srcOffset: srcOffset,
            dstOffset: dstOffset,
          );
        case InterpolationMode.bilinear:
          _resizeBilinear(
            input: input,
            output: output,
            srcH: srcH,
            srcW: srcW,
            srcOffset: srcOffset,
            dstOffset: dstOffset,
          );
        case InterpolationMode.bicubic:
          _resizeBicubic(
            input: input,
            output: output,
            srcH: srcH,
            srcW: srcW,
            srcOffset: srcOffset,
            dstOffset: dstOffset,
          );
      }
    }
  }

  void _resizeNearest({
    required TensorBuffer input,
    required TensorBuffer output,
    required int srcH,
    required int srcW,
    required int srcOffset,
    required int dstOffset,
  }) {
    final scaleY = srcH / height;
    final scaleX = srcW / width;

    for (int y = 0; y < height; y++) {
      final srcY = (y * scaleY).floor().clamp(0, srcH - 1);
      for (int x = 0; x < width; x++) {
        final srcX = (x * scaleX).floor().clamp(0, srcW - 1);
        final value = input.storage.getAsDouble(srcOffset + srcY * srcW + srcX);
        output.storage.setFromDouble(dstOffset + y * width + x, value);
      }
    }
  }

  void _resizeBilinear({
    required TensorBuffer input,
    required TensorBuffer output,
    required int srcH,
    required int srcW,
    required int srcOffset,
    required int dstOffset,
  }) {
    final scaleY =
        alignCorners && height > 1 ? (srcH - 1) / (height - 1) : srcH / height;
    final scaleX =
        alignCorners && width > 1 ? (srcW - 1) / (width - 1) : srcW / width;

    for (int y = 0; y < height; y++) {
      final rawSrcY = alignCorners ? y * scaleY : (y + 0.5) * scaleY - 0.5;
      final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
      final y0 = srcY.floor().clamp(0, srcH - 1);
      final y1 = (y0 + 1).clamp(0, srcH - 1);
      final fy = srcY - y0;

      for (int x = 0; x < width; x++) {
        final rawSrcX = alignCorners ? x * scaleX : (x + 0.5) * scaleX - 0.5;
        final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
        final x0 = srcX.floor().clamp(0, srcW - 1);
        final x1 = (x0 + 1).clamp(0, srcW - 1);
        final fx = srcX - x0;

        final v00 = input.storage.getAsDouble(srcOffset + y0 * srcW + x0);
        final v01 = input.storage.getAsDouble(srcOffset + y0 * srcW + x1);
        final v10 = input.storage.getAsDouble(srcOffset + y1 * srcW + x0);
        final v11 = input.storage.getAsDouble(srcOffset + y1 * srcW + x1);

        final value = v00 * (1 - fx) * (1 - fy) +
            v01 * fx * (1 - fy) +
            v10 * (1 - fx) * fy +
            v11 * fx * fy;

        output.storage.setFromDouble(dstOffset + y * width + x, value);
      }
    }
  }

  void _resizeBicubic({
    required TensorBuffer input,
    required TensorBuffer output,
    required int srcH,
    required int srcW,
    required int srcOffset,
    required int dstOffset,
  }) {
    final scaleY =
        alignCorners && height > 1 ? (srcH - 1) / (height - 1) : srcH / height;
    final scaleX =
        alignCorners && width > 1 ? (srcW - 1) / (width - 1) : srcW / width;

    for (int y = 0; y < height; y++) {
      final rawSrcY = alignCorners ? y * scaleY : (y + 0.5) * scaleY - 0.5;
      final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
      final y0 = srcY.floor();
      final fy = srcY - y0;

      for (int x = 0; x < width; x++) {
        final rawSrcX = alignCorners ? x * scaleX : (x + 0.5) * scaleX - 0.5;
        final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
        final x0 = srcX.floor();
        final fx = srcX - x0;

        double value = 0.0;
        for (int j = -1; j <= 2; j++) {
          final yj = (y0 + j).clamp(0, srcH - 1);
          final wy = _cubicWeight(j - fy);
          for (int i = -1; i <= 2; i++) {
            final xi = (x0 + i).clamp(0, srcW - 1);
            final wx = _cubicWeight(i - fx);
            final v = input.storage.getAsDouble(srcOffset + yj * srcW + xi);
            value += v * wx * wy;
          }
        }

        output.storage.setFromDouble(dstOffset + y * width + x, value);
      }
    }
  }

  double _cubicWeight(double t) {
    final at = t.abs();
    if (at <= 1) {
      return 1.5 * at * at * at - 2.5 * at * at + 1;
    } else if (at < 2) {
      return -0.5 * at * at * at + 2.5 * at * at - 4 * at + 2;
    }
    return 0;
  }
}

/// Resizes a tensor so that the shortest edge matches [shortestEdge].
///
/// Maintains aspect ratio. Optionally limits the longest edge to [maxSize].
class ResizeShortestOp extends TransformOp {
  /// The target length for the shortest edge.
  final int shortestEdge;

  /// The interpolation mode.
  final InterpolationMode mode;

  /// Optional maximum size for the longest edge.
  final int? maxSize;

  /// Creates a resize operation targeting the shortest edge.
  ResizeShortestOp({
    required this.shortestEdge,
    this.mode = InterpolationMode.bilinear,
    this.maxSize,
  });

  @override
  String get name => 'ResizeShortest($shortestEdge)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final shape = input.shape;
    final rank = shape.length;

    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'ResizeShortestOp requires 3D or 4D tensor',
      );
    }

    final h = rank == 3 ? shape[1] : shape[2];
    final w = rank == 3 ? shape[2] : shape[3];

    final (newH, newW) = _computeNewSize(h, w);

    final resizeOp = ResizeOp(
      height: newH,
      width: newW,
      mode: mode,
    );

    return resizeOp.apply(input);
  }

  (int, int) _computeNewSize(int h, int w) {
    final scale = h < w ? shortestEdge / h : shortestEdge / w;

    var newH = (h * scale).round();
    var newW = (w * scale).round();

    if (maxSize != null) {
      final maxScale = maxSize! / math.max(newH, newW);
      if (maxScale < 1.0) {
        newH = (newH * maxScale).round();
        newW = (newW * maxScale).round();
      }
    }

    return (newH, newW);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final rank = inputShape.length;
    final h = rank == 3 ? inputShape[1] : inputShape[2];
    final w = rank == 3 ? inputShape[2] : inputShape[3];

    final (newH, newW) = _computeNewSize(h, w);

    if (rank == 3) {
      return [inputShape[0], newH, newW];
    }
    return [inputShape[0], inputShape[1], newH, newW];
  }
}

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

    final output = TensorBuffer.zeros([c, height, width], dtype: input.dtype);

    for (int ch = 0; ch < c; ch++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final srcIdx = ch * srcH * srcW + (startY + y) * srcW + (startX + x);
          final dstIdx = ch * height * width + y * width + x;
          final value = input.storage.getAsDouble(srcIdx);
          output.storage.setFromDouble(dstIdx, value);
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
        TensorBuffer.zeros([n, c, height, width], dtype: input.dtype);

    final srcBatchStride = c * srcH * srcW;
    final dstBatchStride = c * height * width;

    for (int batch = 0; batch < n; batch++) {
      for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final srcIdx = batch * srcBatchStride +
                ch * srcH * srcW +
                (startY + y) * srcW +
                (startX + x);
            final dstIdx =
                batch * dstBatchStride + ch * height * width + y * width + x;
            final value = input.storage.getAsDouble(srcIdx);
            output.storage.setFromDouble(dstIdx, value);
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
