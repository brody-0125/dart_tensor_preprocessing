import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/index_utils.dart';
import 'transform_op.dart';

/// Padding modes for [PadOp].
enum PadMode {
  /// Pads with a constant value.
  constant,

  /// Pads with reflection of tensor boundary.
  reflect,

  /// Pads with replication of edge values.
  replicate,

  /// Pads with wrap around (circular).
  circular,
}

/// Pads tensor with specified padding on spatial dimensions.
///
/// Supports 3D [C,H,W] and 4D [N,C,H,W] tensors with multiple padding modes.
class PadOp extends TransformOp with RequiresContiguous {
  /// Padding size for top (H-).
  final int top;

  /// Padding size for bottom (H+).
  final int bottom;

  /// Padding size for left (W-).
  final int left;

  /// Padding size for right (W+).
  final int right;

  /// Padding mode to use.
  final PadMode mode;

  /// Constant value for constant padding mode.
  final double value;

  /// Creates a pad operation with specified padding sizes.
  PadOp({
    required this.top,
    required this.bottom,
    required this.left,
    required this.right,
    this.mode = PadMode.constant,
    this.value = 0.0,
  }) {
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
      throw InvalidParameterException(
        'padding sizes',
        'top=$top, bottom=$bottom, left=$left, right=$right',
        'All padding sizes must be non-negative',
      );
    }
    if (mode == PadMode.constant && (value.isNaN || value.isInfinite)) {
      throw InvalidParameterException(
        'value',
        value.toString(),
        'Value must be a finite number for constant padding',
      );
    }
  }

  /// Creates symmetric padding with same size on all sides.
  factory PadOp.symmetric(
    int padH,
    int padW, {
    PadMode mode = PadMode.constant,
    double value = 0.0,
  }) {
    return PadOp(
      top: padH,
      bottom: padH,
      left: padW,
      right: padW,
      mode: mode,
      value: value,
    );
  }

  /// Creates padding with same size on all sides.
  factory PadOp.all(
    int pad, {
    PadMode mode = PadMode.constant,
    double value = 0.0,
  }) {
    return PadOp.symmetric(pad, pad, mode: mode, value: value);
  }

  @override
  String get name =>
      'Pad(top=$top, bottom=$bottom, left=$left, right=$right, mode=$mode)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        preservesShape: false,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    final outputShape = computeOutputShape(contiguous.shape);
    final output =
        TensorBuffer.uninitialized(outputShape, dtype: contiguous.dtype);

    switch (mode) {
      case PadMode.constant:
        _padConstant(contiguous, output);
        break;
      case PadMode.reflect:
        _padReflect(contiguous, output);
        break;
      case PadMode.replicate:
        _padReplicate(contiguous, output);
        break;
      case PadMode.circular:
        _padCircular(contiguous, output);
        break;
    }

    return output;
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'PadOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }
  }

  void _padConstant(TensorBuffer input, TensorBuffer output) {
    // Fill output with constant value
    final numel = output.numel;
    for (int i = 0; i < numel; i++) {
      output.storage.setFromDouble(i, value);
    }

    // Copy input data to the center region
    final inputShape = input.shape;

    if (inputShape.length == 3) {
      _copy3DInput(input, output, top, left);
    } else {
      _copy4DInput(input, output, top, left);
    }
  }

  void _padReflect(TensorBuffer input, TensorBuffer output) {
    final inputShape = input.shape;

    if (inputShape.length == 3) {
      _padReflect3D(input, output);
    } else {
      _padReflect4D(input, output);
    }
  }

  void _padReplicate(TensorBuffer input, TensorBuffer output) {
    final inputShape = input.shape;

    if (inputShape.length == 3) {
      _padReplicate3D(input, output);
    } else {
      _padReplicate4D(input, output);
    }
  }

  void _padCircular(TensorBuffer input, TensorBuffer output) {
    final inputShape = input.shape;

    if (inputShape.length == 3) {
      _padCircular3D(input, output);
    } else {
      _padCircular4D(input, output);
    }
  }

  void _copy3DInput(
      TensorBuffer input, TensorBuffer output, int padH, int padW) {
    final (c, h, w) = (input.shape[0], input.shape[1], input.shape[2]);
    final inputChannelSize = h * w;
    final outputChannelSize = output.shape[1] * output.shape[2];

    for (int ch = 0; ch < c; ch++) {
      final inputOffset = ch * inputChannelSize;
      final outputOffset =
          ch * outputChannelSize + padH * output.shape[2] + padW;

      for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
          final inputIdx = inputOffset + row * w + col;
          final outputIdx = outputOffset + row * output.shape[2] + col;
          final val = input.storage.getAsDouble(inputIdx);
          output.storage.setFromDouble(outputIdx, val);
        }
      }
    }
  }

  void _copy4DInput(
      TensorBuffer input, TensorBuffer output, int padH, int padW) {
    final (n, c, h, w) =
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    final inputChannelSize = h * w;
    final outputChannelSize = output.shape[2] * output.shape[3];

    for (int batch = 0; batch < n; batch++) {
      for (int ch = 0; ch < c; ch++) {
        final inputOffset =
            batch * c * inputChannelSize + ch * inputChannelSize;
        final outputOffset = batch * c * outputChannelSize +
            ch * outputChannelSize +
            padH * output.shape[3] +
            padW;

        for (int row = 0; row < h; row++) {
          for (int col = 0; col < w; col++) {
            final inputIdx = inputOffset + row * w + col;
            final outputIdx = outputOffset + row * output.shape[3] + col;
            final val = input.storage.getAsDouble(inputIdx);
            output.storage.setFromDouble(outputIdx, val);
          }
        }
      }
    }
  }

  void _padReflect3D(TensorBuffer input, TensorBuffer output) {
    final (c, h, w) = (input.shape[0], input.shape[1], input.shape[2]);
    final (outH, outW) = (output.shape[1], output.shape[2]);

    for (int ch = 0; ch < c; ch++) {
      for (int outRow = 0; outRow < outH; outRow++) {
        for (int outCol = 0; outCol < outW; outCol++) {
          final inRow = reflectIndex(outRow - top, h);
          final inCol = reflectIndex(outCol - left, w);

          if (inRow >= 0 && inRow < h && inCol >= 0 && inCol < w) {
            final inputIdx = ch * h * w + inRow * w + inCol;
            final outputIdx = ch * outH * outW + outRow * outW + outCol;
            final val = input.storage.getAsDouble(inputIdx);
            output.storage.setFromDouble(outputIdx, val);
          }
        }
      }
    }
  }

  void _padReflect4D(TensorBuffer input, TensorBuffer output) {
    final (n, c, h, w) =
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    final (outH, outW) = (output.shape[2], output.shape[3]);

    for (int batch = 0; batch < n; batch++) {
      for (int ch = 0; ch < c; ch++) {
        for (int outRow = 0; outRow < outH; outRow++) {
          for (int outCol = 0; outCol < outW; outCol++) {
            final inRow = reflectIndex(outRow - top, h);
            final inCol = reflectIndex(outCol - left, w);

            if (inRow >= 0 && inRow < h && inCol >= 0 && inCol < w) {
              final inputIdx =
                  batch * c * h * w + ch * h * w + inRow * w + inCol;
              final outputIdx = batch * c * outH * outW +
                  ch * outH * outW +
                  outRow * outW +
                  outCol;
              final val = input.storage.getAsDouble(inputIdx);
              output.storage.setFromDouble(outputIdx, val);
            }
          }
        }
      }
    }
  }

  void _padReplicate3D(TensorBuffer input, TensorBuffer output) {
    final (c, h, w) = (input.shape[0], input.shape[1], input.shape[2]);
    final (outH, outW) = (output.shape[1], output.shape[2]);

    for (int ch = 0; ch < c; ch++) {
      for (int outRow = 0; outRow < outH; outRow++) {
        for (int outCol = 0; outCol < outW; outCol++) {
          final inRow = replicateIndex(outRow - top, h);
          final inCol = replicateIndex(outCol - left, w);

          final inputIdx = ch * h * w + inRow * w + inCol;
          final outputIdx = ch * outH * outW + outRow * outW + outCol;
          final val = input.storage.getAsDouble(inputIdx);
          output.storage.setFromDouble(outputIdx, val);
        }
      }
    }
  }

  void _padReplicate4D(TensorBuffer input, TensorBuffer output) {
    final (n, c, h, w) =
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    final (outH, outW) = (output.shape[2], output.shape[3]);

    for (int batch = 0; batch < n; batch++) {
      for (int ch = 0; ch < c; ch++) {
        for (int outRow = 0; outRow < outH; outRow++) {
          for (int outCol = 0; outCol < outW; outCol++) {
            final inRow = replicateIndex(outRow - top, h);
            final inCol = replicateIndex(outCol - left, w);

            final inputIdx = batch * c * h * w + ch * h * w + inRow * w + inCol;
            final outputIdx = batch * c * outH * outW +
                ch * outH * outW +
                outRow * outW +
                outCol;
            final val = input.storage.getAsDouble(inputIdx);
            output.storage.setFromDouble(outputIdx, val);
          }
        }
      }
    }
  }

  void _padCircular3D(TensorBuffer input, TensorBuffer output) {
    final (c, h, w) = (input.shape[0], input.shape[1], input.shape[2]);
    final (outH, outW) = (output.shape[1], output.shape[2]);

    for (int ch = 0; ch < c; ch++) {
      for (int outRow = 0; outRow < outH; outRow++) {
        for (int outCol = 0; outCol < outW; outCol++) {
          final inRow = circularIndex(outRow - top, h);
          final inCol = circularIndex(outCol - left, w);

          final inputIdx = ch * h * w + inRow * w + inCol;
          final outputIdx = ch * outH * outW + outRow * outW + outCol;
          final val = input.storage.getAsDouble(inputIdx);
          output.storage.setFromDouble(outputIdx, val);
        }
      }
    }
  }

  void _padCircular4D(TensorBuffer input, TensorBuffer output) {
    final (n, c, h, w) =
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    final (outH, outW) = (output.shape[2], output.shape[3]);

    for (int batch = 0; batch < n; batch++) {
      for (int ch = 0; ch < c; ch++) {
        for (int outRow = 0; outRow < outH; outRow++) {
          for (int outCol = 0; outCol < outW; outCol++) {
            final inRow = circularIndex(outRow - top, h);
            final inCol = circularIndex(outCol - left, w);

            final inputIdx = batch * c * h * w + ch * h * w + inRow * w + inCol;
            final outputIdx = batch * c * outH * outW +
                ch * outH * outW +
                outRow * outW +
                outCol;
            final val = input.storage.getAsDouble(inputIdx);
            output.storage.setFromDouble(outputIdx, val);
          }
        }
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length == 3) {
      // [C, H, W] -> [C, H + top + bottom, W + left + right]
      final c = inputShape[0];
      final h = inputShape[1];
      final w = inputShape[2];
      return [c, h + top + bottom, w + left + right];
    } else {
      // [N, C, H, W] -> [N, C, H + top + bottom, W + left + right]
      final n = inputShape[0];
      final c = inputShape[1];
      final h = inputShape[2];
      final w = inputShape[3];
      return [n, c, h + top + bottom, w + left + right];
    }
  }
}
