import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

// Re-export CenterCropOp for backward compatibility.
export 'crop_op.dart';

/// Interpolation algorithm used for image resizing.
enum InterpolationMode {
  /// Nearest-neighbor interpolation (fastest, lowest quality).
  nearest,

  /// Bilinear interpolation (balanced speed and quality).
  bilinear,

  /// Bicubic interpolation (slowest, highest quality).
  bicubic,

  /// Area interpolation (best for downsampling, anti-aliasing).
  ///
  /// Computes output by averaging source pixels within each output pixel's area.
  /// Avoids aliasing artifacts during downsampling.
  /// Equivalent to `F.interpolate(mode='area')` in PyTorch.
  area,

  /// Lanczos interpolation (high quality, uses Lanczos3 kernel).
  ///
  /// Uses sinc-based kernel with a=3, sampling 6x6 source pixels.
  /// Best for high-quality upscaling and downscaling.
  /// Equivalent to `InterpolationMode.LANCZOS` in torchvision.
  lanczos,
}

/// ONNX-compatible coordinate transformation modes for resize operations.
///
/// Controls how output pixel coordinates are mapped to input pixel coordinates.
/// Different frameworks use different conventions:
/// - PyTorch default: [halfPixel]
/// - TensorFlow default: [asymmetric]
/// - PyTorch `align_corners=True`: [alignCorners]
enum CoordinateTransformMode {
  /// Half-pixel offset: `srcCoord = (dstCoord + 0.5) * scale - 0.5`
  ///
  /// PyTorch default when `align_corners=False`.
  /// Centers pixels within their cells for consistent interpolation.
  halfPixel,

  /// Align corners: `srcCoord = dstCoord * (inSize - 1) / (outSize - 1)`
  ///
  /// Maps corner pixels exactly. PyTorch `align_corners=True`.
  alignCorners,

  /// Asymmetric: `srcCoord = dstCoord * inSize / outSize`
  ///
  /// TensorFlow default. Simple ratio mapping without offset.
  asymmetric,

  /// PyTorch half-pixel: same as [halfPixel] but maps to 0 when `outSize == 1`.
  ///
  /// Handles the edge case where output has single pixel.
  pytorchHalfPixel,
}

/// Resizes a tensor to a fixed [height] and [width].
///
/// Supports 3D tensors `[C, H, W]` and 4D tensors `[N, C, H, W]`.
///
/// ## Complexity
///
/// Let `C` = channels, `H_out` = output height, `W_out` = output width.
///
/// | Mode | Time | Space | Notes |
/// |------|------|-------|-------|
/// | nearest | O(C × H_out × W_out) | O(C × H_out × W_out) | Fastest |
/// | bilinear | O(C × H_out × W_out) | O(C × H_out × W_out) | 2×2 interpolation |
/// | bicubic | O(C × H_out × W_out) | O(C × H_out × W_out) | 4×4 kernel |
/// | area | O(C × H_out × W_out × scale²) | O(C × H_out × W_out) | Best for downsampling |
/// | lanczos | O(C × H_out × W_out × 36) | O(C × H_out × W_out) | 6×6 kernel |
///
/// All modes use 64×64 cache-friendly blocking for improved L1 cache utilization.
class ResizeOp extends TransformOp with RequiresContiguous {
  /// The target height.
  final int height;

  /// The target width.
  final int width;

  /// The interpolation mode.
  final InterpolationMode mode;

  /// Whether to align corners during interpolation.
  ///
  /// When [coordinateMode] is explicitly set, this parameter is ignored.
  /// Kept for backward compatibility.
  final bool alignCorners;

  /// The coordinate transformation mode.
  ///
  /// When `null`, behavior is determined by [alignCorners]:
  /// - `alignCorners == false` → [CoordinateTransformMode.halfPixel]
  /// - `alignCorners == true` → [CoordinateTransformMode.alignCorners]
  final CoordinateTransformMode? coordinateMode;

  /// The effective coordinate mode resolved from [coordinateMode] and [alignCorners].
  late final CoordinateTransformMode _effectiveMode;

  /// Creates a resize operation with the specified dimensions.
  ResizeOp({
    required this.height,
    required this.width,
    this.mode = InterpolationMode.bilinear,
    this.alignCorners = false,
    this.coordinateMode,
  }) {
    if (height <= 0 || width <= 0) {
      throw InvalidParameterException(
        'height/width',
        '$height x $width',
        'Must be positive',
      );
    }
    _effectiveMode = coordinateMode ??
        (alignCorners
            ? CoordinateTransformMode.alignCorners
            : CoordinateTransformMode.halfPixel);
  }

  @override
  String get name => 'Resize(${height}x$width, $mode)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: false,
      );

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
    final output = TensorBuffer.uninitialized(outputShape, dtype: input.dtype);

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
    final output = TensorBuffer.uninitialized(outputShape, dtype: input.dtype);

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
        case InterpolationMode.area:
          _resizeArea(
            input: input,
            output: output,
            srcH: srcH,
            srcW: srcW,
            srcOffset: srcOffset,
            dstOffset: dstOffset,
          );
        case InterpolationMode.lanczos:
          _resizeLanczos(
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

  /// Computes scale factor for coordinate transformation.
  double _computeScale(int inSize, int outSize) {
    switch (_effectiveMode) {
      case CoordinateTransformMode.alignCorners:
        return outSize > 1 ? (inSize - 1) / (outSize - 1) : 0.0;
      case CoordinateTransformMode.halfPixel:
      case CoordinateTransformMode.pytorchHalfPixel:
      case CoordinateTransformMode.asymmetric:
        return inSize / outSize;
    }
  }

  /// Maps output coordinate to source coordinate.
  double _mapCoordinate(int outCoord, int inSize, int outSize, double scale) {
    switch (_effectiveMode) {
      case CoordinateTransformMode.halfPixel:
        return (outCoord + 0.5) * scale - 0.5;
      case CoordinateTransformMode.alignCorners:
        return outCoord * scale;
      case CoordinateTransformMode.asymmetric:
        return outCoord * scale;
      case CoordinateTransformMode.pytorchHalfPixel:
        if (outSize > 1) {
          return (outCoord + 0.5) * scale - 0.5;
        }
        return 0.0;
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
    final scaleY = _computeScale(srcH, height);
    final scaleX = _computeScale(srcW, width);

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int y = 0; y < height; y++) {
          final srcY =
              _mapCoordinate(y, srcH, height, scaleY).floor().clamp(0, srcH - 1);
          final srcRowOffset = srcOffset + srcY * srcW;
          final dstRowOffset = dstOffset + y * width;
          for (int x = 0; x < width; x++) {
            final srcX =
                _mapCoordinate(x, srcW, width, scaleX).floor().clamp(0, srcW - 1);
            outList[dstRowOffset + x] = inList[srcRowOffset + srcX];
          }
        }
      default:
        // Generic fallback
        for (int y = 0; y < height; y++) {
          final srcY =
              _mapCoordinate(y, srcH, height, scaleY).floor().clamp(0, srcH - 1);
          for (int x = 0; x < width; x++) {
            final srcX =
                _mapCoordinate(x, srcW, width, scaleX).floor().clamp(0, srcW - 1);
            final value =
                input.storage.getAsDouble(srcOffset + srcY * srcW + srcX);
            output.storage.setFromDouble(dstOffset + y * width + x, value);
          }
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
    final scaleY = _computeScale(srcH, height);
    final scaleX = _computeScale(srcW, width);

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;

        // Cache-friendly blocking: process output in 64x64 blocks
        // This ensures source pixels stay in L1 cache during processing
        const blockSize = 64;

        for (int by = 0; by < height; by += blockSize) {
          final endY = math.min(by + blockSize, height);

          for (int bx = 0; bx < width; bx += blockSize) {
            final endX = math.min(bx + blockSize, width);

            // Process block
            for (int y = by; y < endY; y++) {
              final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
              final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
              final y0 = srcY.floor().clamp(0, srcH - 1);
              final y1 = (y0 + 1).clamp(0, srcH - 1);
              final fy = srcY - y0;
              final oneMinusFy = 1 - fy;

              for (int x = bx; x < endX; x++) {
                final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
                final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
                final x0 = srcX.floor().clamp(0, srcW - 1);
                final x1 = (x0 + 1).clamp(0, srcW - 1);
                final fx = srcX - x0;
                final oneMinusFx = 1 - fx;

                final v00 = inList[srcOffset + y0 * srcW + x0];
                final v01 = inList[srcOffset + y0 * srcW + x1];
                final v10 = inList[srcOffset + y1 * srcW + x0];
                final v11 = inList[srcOffset + y1 * srcW + x1];

                final value = v00 * oneMinusFx * oneMinusFy +
                    v01 * fx * oneMinusFy +
                    v10 * oneMinusFx * fy +
                    v11 * fx * fy;

                outList[dstOffset + y * width + x] = value;
              }
            }
          }
        }
      default:
        // Generic fallback with cache-friendly blocking
        const blockSize = 64;

        for (int by = 0; by < height; by += blockSize) {
          final endY = math.min(by + blockSize, height);

          for (int bx = 0; bx < width; bx += blockSize) {
            final endX = math.min(bx + blockSize, width);

            for (int y = by; y < endY; y++) {
              final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
              final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
              final y0 = srcY.floor().clamp(0, srcH - 1);
              final y1 = (y0 + 1).clamp(0, srcH - 1);
              final fy = srcY - y0;

              for (int x = bx; x < endX; x++) {
                final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
                final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
                final x0 = srcX.floor().clamp(0, srcW - 1);
                final x1 = (x0 + 1).clamp(0, srcW - 1);
                final fx = srcX - x0;

                final v00 =
                    input.storage.getAsDouble(srcOffset + y0 * srcW + x0);
                final v01 =
                    input.storage.getAsDouble(srcOffset + y0 * srcW + x1);
                final v10 =
                    input.storage.getAsDouble(srcOffset + y1 * srcW + x0);
                final v11 =
                    input.storage.getAsDouble(srcOffset + y1 * srcW + x1);

                final value = v00 * (1 - fx) * (1 - fy) +
                    v01 * fx * (1 - fy) +
                    v10 * (1 - fx) * fy +
                    v11 * fx * fy;

                output.storage.setFromDouble(dstOffset + y * width + x, value);
              }
            }
          }
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
    final scaleY = _computeScale(srcH, height);
    final scaleX = _computeScale(srcW, width);

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;

        // Cache-friendly blocking: process output in 64x64 blocks
        // This ensures source pixels stay in L1 cache during processing
        const blockSize = 64;

        for (int by = 0; by < height; by += blockSize) {
          final endY = math.min(by + blockSize, height);

          for (int bx = 0; bx < width; bx += blockSize) {
            final endX = math.min(bx + blockSize, width);

            // Process block
            for (int y = by; y < endY; y++) {
              final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
              final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
              final y0 = srcY.floor();
              final fy = srcY - y0;

              // Pre-compute y weights
              final wy0 = _cubicWeight(-1 - fy);
              final wy1 = _cubicWeight(-fy);
              final wy2 = _cubicWeight(1 - fy);
              final wy3 = _cubicWeight(2 - fy);

              // Pre-compute y indices
              final yj0 = (y0 - 1).clamp(0, srcH - 1);
              final yj1 = y0.clamp(0, srcH - 1);
              final yj2 = (y0 + 1).clamp(0, srcH - 1);
              final yj3 = (y0 + 2).clamp(0, srcH - 1);

              for (int x = bx; x < endX; x++) {
                final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
                final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
                final x0 = srcX.floor();
                final fx = srcX - x0;

                // Pre-compute x weights
                final wx0 = _cubicWeight(-1 - fx);
                final wx1 = _cubicWeight(-fx);
                final wx2 = _cubicWeight(1 - fx);
                final wx3 = _cubicWeight(2 - fx);

                // Pre-compute x indices
                final xi0 = (x0 - 1).clamp(0, srcW - 1);
                final xi1 = x0.clamp(0, srcW - 1);
                final xi2 = (x0 + 1).clamp(0, srcW - 1);
                final xi3 = (x0 + 2).clamp(0, srcW - 1);

                // Unrolled 4x4 kernel for performance
                final value = wy0 *
                        (wx0 * inList[srcOffset + yj0 * srcW + xi0] +
                            wx1 * inList[srcOffset + yj0 * srcW + xi1] +
                            wx2 * inList[srcOffset + yj0 * srcW + xi2] +
                            wx3 * inList[srcOffset + yj0 * srcW + xi3]) +
                    wy1 *
                        (wx0 * inList[srcOffset + yj1 * srcW + xi0] +
                            wx1 * inList[srcOffset + yj1 * srcW + xi1] +
                            wx2 * inList[srcOffset + yj1 * srcW + xi2] +
                            wx3 * inList[srcOffset + yj1 * srcW + xi3]) +
                    wy2 *
                        (wx0 * inList[srcOffset + yj2 * srcW + xi0] +
                            wx1 * inList[srcOffset + yj2 * srcW + xi1] +
                            wx2 * inList[srcOffset + yj2 * srcW + xi2] +
                            wx3 * inList[srcOffset + yj2 * srcW + xi3]) +
                    wy3 *
                        (wx0 * inList[srcOffset + yj3 * srcW + xi0] +
                            wx1 * inList[srcOffset + yj3 * srcW + xi1] +
                            wx2 * inList[srcOffset + yj3 * srcW + xi2] +
                            wx3 * inList[srcOffset + yj3 * srcW + xi3]);

                outList[dstOffset + y * width + x] = value;
              }
            }
          }
        }
      default:
        // Generic fallback
        for (int y = 0; y < height; y++) {
          final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
          final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
          final y0 = srcY.floor();
          final fy = srcY - y0;

          for (int x = 0; x < width; x++) {
            final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
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

  /// Area interpolation for high-quality downsampling.
  ///
  /// Computes each output pixel as the weighted average of all source pixels
  /// that overlap with the output pixel's area. Handles fractional coverage
  /// at boundaries for anti-aliasing.
  void _resizeArea({
    required TensorBuffer input,
    required TensorBuffer output,
    required int srcH,
    required int srcW,
    required int srcOffset,
    required int dstOffset,
  }) {
    final scaleY = srcH / height;
    final scaleX = srcW / width;

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;

        // Cache-friendly blocking: process output in 64x64 blocks
        const blockSize = 64;

        for (int by = 0; by < height; by += blockSize) {
          final endY = math.min(by + blockSize, height);

          for (int bx = 0; bx < width; bx += blockSize) {
            final endX = math.min(bx + blockSize, width);

            for (int y = by; y < endY; y++) {
              // Source region covered by output pixel [y]
              final srcY0 = y * scaleY;
              final srcY1 = (y + 1) * scaleY;

              for (int x = bx; x < endX; x++) {
                // Source region covered by output pixel [x]
                final srcX0 = x * scaleX;
                final srcX1 = (x + 1) * scaleX;

                // Accumulate weighted sum
                double sum = 0;
                double totalWeight = 0;

                final y0Floor = srcY0.floor().clamp(0, srcH - 1);
                final y1Ceil = srcY1.ceil().clamp(0, srcH);
                final x0Floor = srcX0.floor().clamp(0, srcW - 1);
                final x1Ceil = srcX1.ceil().clamp(0, srcW);

                for (int sy = y0Floor; sy < y1Ceil; sy++) {
                  // Compute vertical overlap
                  final overlapTop = math.max(srcY0, sy.toDouble());
                  final overlapBottom = math.min(srcY1, (sy + 1).toDouble());
                  final overlapY = math.max(0.0, overlapBottom - overlapTop);

                  for (int sx = x0Floor; sx < x1Ceil; sx++) {
                    // Compute horizontal overlap
                    final overlapLeft = math.max(srcX0, sx.toDouble());
                    final overlapRight = math.min(srcX1, (sx + 1).toDouble());
                    final overlapX = math.max(0.0, overlapRight - overlapLeft);

                    final weight = overlapX * overlapY;
                    if (weight > 0) {
                      sum += inList[srcOffset + sy * srcW + sx] * weight;
                      totalWeight += weight;
                    }
                  }
                }

                outList[dstOffset + y * width + x] =
                    totalWeight > 0 ? sum / totalWeight : 0;
              }
            }
          }
        }
      default:
        // Generic fallback with cache-friendly blocking
        const blockSize = 64;

        for (int by = 0; by < height; by += blockSize) {
          final endY = math.min(by + blockSize, height);

          for (int bx = 0; bx < width; bx += blockSize) {
            final endX = math.min(bx + blockSize, width);

            for (int y = by; y < endY; y++) {
              final srcY0 = y * scaleY;
              final srcY1 = (y + 1) * scaleY;

              for (int x = bx; x < endX; x++) {
                final srcX0 = x * scaleX;
                final srcX1 = (x + 1) * scaleX;

                double sum = 0;
                double totalWeight = 0;

                final y0Floor = srcY0.floor().clamp(0, srcH - 1);
                final y1Ceil = srcY1.ceil().clamp(0, srcH);
                final x0Floor = srcX0.floor().clamp(0, srcW - 1);
                final x1Ceil = srcX1.ceil().clamp(0, srcW);

                for (int sy = y0Floor; sy < y1Ceil; sy++) {
                  final overlapTop = math.max(srcY0, sy.toDouble());
                  final overlapBottom = math.min(srcY1, (sy + 1).toDouble());
                  final overlapY = math.max(0.0, overlapBottom - overlapTop);

                  for (int sx = x0Floor; sx < x1Ceil; sx++) {
                    final overlapLeft = math.max(srcX0, sx.toDouble());
                    final overlapRight = math.min(srcX1, (sx + 1).toDouble());
                    final overlapX = math.max(0.0, overlapRight - overlapLeft);

                    final weight = overlapX * overlapY;
                    if (weight > 0) {
                      final v =
                          input.storage.getAsDouble(srcOffset + sy * srcW + sx);
                      sum += v * weight;
                      totalWeight += weight;
                    }
                  }
                }

                output.storage.setFromDouble(
                  dstOffset + y * width + x,
                  totalWeight > 0 ? sum / totalWeight : 0,
                );
              }
            }
          }
        }
    }
  }

  /// Lanczos interpolation using Lanczos3 kernel (a=3).
  ///
  /// Uses sinc-based kernel for high-quality interpolation.
  /// Samples 6x6 source pixels for each output pixel.
  void _resizeLanczos({
    required TensorBuffer input,
    required TensorBuffer output,
    required int srcH,
    required int srcW,
    required int srcOffset,
    required int dstOffset,
  }) {
    final scaleY = _computeScale(srcH, height);
    final scaleX = _computeScale(srcW, width);

    // Lanczos window size (a=3 means 6x6 kernel)
    const a = 3;

    // Dtype-specialized for hot path optimization
    switch (input.dtype) {
      case DType.float32:
        final inList = input.storage.data as Float32List;
        final outList = output.storage.data as Float32List;
        for (int y = 0; y < height; y++) {
          final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
          final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
          final y0 = srcY.floor();
          final fy = srcY - y0;

          for (int x = 0; x < width; x++) {
            final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
            final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
            final x0 = srcX.floor();
            final fx = srcX - x0;

            double value = 0.0;
            double weightSum = 0.0;

            // 6x6 kernel (a=3: indices from -2 to 3)
            for (int j = -(a - 1); j <= a; j++) {
              final yj = (y0 + j).clamp(0, srcH - 1);
              final wy = _lanczosKernel(j - fy, a);

              for (int i = -(a - 1); i <= a; i++) {
                final xi = (x0 + i).clamp(0, srcW - 1);
                final wx = _lanczosKernel(i - fx, a);
                final weight = wx * wy;

                value += inList[srcOffset + yj * srcW + xi] * weight;
                weightSum += weight;
              }
            }

            // Normalize by weight sum to handle edge effects
            outList[dstOffset + y * width + x] =
                weightSum > 0 ? value / weightSum : 0;
          }
        }
      default:
        // Generic fallback
        for (int y = 0; y < height; y++) {
          final rawSrcY = _mapCoordinate(y, srcH, height, scaleY);
          final srcY = rawSrcY.clamp(0.0, srcH - 1.0);
          final y0 = srcY.floor();
          final fy = srcY - y0;

          for (int x = 0; x < width; x++) {
            final rawSrcX = _mapCoordinate(x, srcW, width, scaleX);
            final srcX = rawSrcX.clamp(0.0, srcW - 1.0);
            final x0 = srcX.floor();
            final fx = srcX - x0;

            double value = 0.0;
            double weightSum = 0.0;

            for (int j = -(a - 1); j <= a; j++) {
              final yj = (y0 + j).clamp(0, srcH - 1);
              final wy = _lanczosKernel(j - fy, a);

              for (int i = -(a - 1); i <= a; i++) {
                final xi = (x0 + i).clamp(0, srcW - 1);
                final wx = _lanczosKernel(i - fx, a);
                final weight = wx * wy;

                final v = input.storage.getAsDouble(srcOffset + yj * srcW + xi);
                value += v * weight;
                weightSum += weight;
              }
            }

            output.storage.setFromDouble(
              dstOffset + y * width + x,
              weightSum > 0 ? value / weightSum : 0,
            );
          }
        }
    }
  }

  /// Lanczos kernel: sinc(x) * sinc(x/a)
  ///
  /// L(x) = sinc(x) * sinc(x/a) for |x| < a
  ///      = 0                   for |x| >= a
  ///
  /// where sinc(x) = sin(pi*x) / (pi*x)
  double _lanczosKernel(double x, int a) {
    if (x == 0) return 1.0;
    final ax = x.abs();
    if (ax >= a) return 0.0;
    final px = math.pi * x;
    final pxa = px / a;
    return (math.sin(px) / px) * (math.sin(pxa) / pxa);
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
