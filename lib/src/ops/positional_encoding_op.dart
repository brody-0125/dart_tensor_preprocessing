import 'dart:math' as math;
import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Adds sinusoidal positional encodings to the input tensor.
///
/// Implements the positional encoding from "Attention Is All You Need"
/// (Vaswani et al., 2017):
///
/// ```
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
///
/// This is used in Transformer models to inject position information into
/// the input embeddings. Also used in Rotary Position Embedding (RoPE)
/// for modern LLMs like LLaMA and Gemma.
///
/// The input tensor must have at least 2 dimensions, with the last two
/// dimensions interpreted as [seq_len, d_model]. The operation adds
/// precomputed positional encodings to the input.
///
/// ```dart
/// final posEnc = PositionalEncodingOp(dModel: 512, maxLen: 5000);
/// final result = posEnc(inputTensor);  // input + PE
/// ```
class PositionalEncodingOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  /// The model dimension (number of features).
  final int dModel;

  /// The maximum sequence length supported.
  final int maxLen;

  /// The base used for the frequency computation.
  final double base;

  /// Precomputed positional encoding table.
  late final Float64List _peTable;

  /// Creates a positional encoding operation.
  ///
  /// [dModel] is the embedding dimension (must be positive).
  /// [maxLen] is the maximum sequence length (must be positive).
  /// [base] is the base for frequency computation (default 10000.0).
  PositionalEncodingOp({
    required this.dModel,
    required this.maxLen,
    this.base = 10000.0,
  }) {
    if (dModel <= 0) {
      throw InvalidParameterException('dModel', dModel, 'Must be positive');
    }
    if (maxLen <= 0) {
      throw InvalidParameterException('maxLen', maxLen, 'Must be positive');
    }
    _precompute();
  }

  void _precompute() {
    _peTable = Float64List(maxLen * dModel);
    for (int pos = 0; pos < maxLen; pos++) {
      for (int i = 0; i < dModel; i++) {
        final dimIndex = i ~/ 2;
        final angle = pos / math.pow(base, 2 * dimIndex / dModel);
        _peTable[pos * dModel + i] = i.isEven ? math.sin(angle) : math.cos(angle);
      }
    }
  }

  @override
  String get name => 'PositionalEncoding(d=$dModel, max=$maxLen)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        pytorchEquivalent: 'torch.nn.Embedding (positional)',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _addEncoding(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException(
          'PositionalEncodingOp.applyInPlace');
    }
    _addEncoding(input);
  }

  void _addEncoding(TensorBuffer tensor) {
    if (tensor.rank < 2) {
      throw ShapeMismatchException(
        actual: tensor.shape,
        message:
            'PositionalEncodingOp requires at least 2D input [... , seq_len, d_model], '
            'got shape ${tensor.shape}',
      );
    }

    final seqLen = tensor.shape[tensor.rank - 2];
    final dim = tensor.shape[tensor.rank - 1];

    if (dim != dModel) {
      throw ShapeMismatchException(
        actual: tensor.shape,
        message:
            'Last dimension ($dim) must match dModel ($dModel)',
      );
    }

    if (seqLen > maxLen) {
      throw InvalidParameterException(
        'seqLen',
        seqLen,
        'Sequence length exceeds maxLen ($maxLen)',
      );
    }

    // Compute number of batches (all dimensions except last two)
    final batchSize = tensor.rank > 2
        ? tensor.shape
            .sublist(0, tensor.rank - 2)
            .fold(1, (a, b) => a * b)
        : 1;
    final sliceSize = seqLen * dModel;

    switch (tensor.dtype) {
      case DType.float32:
        final data = tensor.storage.data as Float32List;
        for (int b = 0; b < batchSize; b++) {
          final offset = b * sliceSize;
          for (int i = 0; i < sliceSize; i++) {
            data[offset + i] += _peTable[i];
          }
        }
      case DType.float64:
        final data = tensor.storage.data as Float64List;
        for (int b = 0; b < batchSize; b++) {
          final offset = b * sliceSize;
          for (int i = 0; i < sliceSize; i++) {
            data[offset + i] += _peTable[i];
          }
        }
      default:
        for (int b = 0; b < batchSize; b++) {
          final offset = b * sliceSize;
          for (int i = 0; i < sliceSize; i++) {
            final idx = offset + i;
            final value = tensor.storage.getAsDouble(idx);
            tensor.storage.setFromDouble(idx, value + _peTable[i]);
          }
        }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
