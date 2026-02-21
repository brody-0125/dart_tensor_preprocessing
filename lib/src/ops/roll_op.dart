import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';
import 'transform_op.dart';

/// Rolls a tensor along the given dimensions.
///
/// Equivalent to `torch.roll()` in PyTorch.
///
/// Elements shifted past the last position wrap around to the beginning.
/// If [dims] is null, the tensor is flattened, rolled, then reshaped.
///
/// ```dart
/// final result = RollOp(shifts: [1], dims: [0])(tensor);
/// ```
class RollOp extends TransformOp with RequiresContiguous {
  /// The number of positions to shift along each dim.
  final List<int> shifts;

  /// The dimensions to roll. If null, rolls the flattened tensor.
  final List<int>? dims;

  /// Creates a Roll operation.
  RollOp({required this.shifts, this.dims});

  @override
  String get name => 'Roll';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        requiresContiguous: true,
        pytorchEquivalent: 'torch.roll',
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);

    if (dims == null) {
      // Flat roll
      return _rollFlat(contiguous);
    }

    if (shifts.length != dims!.length) {
      throw InvalidParameterException(
        'dims',
        dims,
        'RollOp: shifts length ${shifts.length} must match dims length ${dims!.length}',
      );
    }

    return _rollAlongDims(contiguous);
  }

  TensorBuffer _rollFlat(TensorBuffer input) {
    final numel = input.numel;
    final shift = ((shifts[0] % numel) + numel) % numel;

    final output =
        TensorBuffer.uninitialized(List<int>.from(input.shape), dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inData = input.storage.data as Float32List;
        final outData = output.storage.data as Float32List;
        for (int i = 0; i < numel; i++) {
          outData[(i + shift) % numel] = inData[i];
        }

      case DType.float64:
        final inData = input.storage.data as Float64List;
        final outData = output.storage.data as Float64List;
        for (int i = 0; i < numel; i++) {
          outData[(i + shift) % numel] = inData[i];
        }

      default:
        final inStorage = input.storage;
        final outStorage = output.storage;
        for (int i = 0; i < numel; i++) {
          outStorage.setFromDouble(
            (i + shift) % numel,
            inStorage.getAsDouble(i),
          );
        }
    }

    return output;
  }

  TensorBuffer _rollAlongDims(TensorBuffer input) {
    final shape = input.shape;
    final rank = shape.length;

    // Validate dim ranges
    for (int s = 0; s < dims!.length; s++) {
      final d = dims![s];
      final normalizedDim = d < 0 ? rank + d : d;
      if (normalizedDim < 0 || normalizedDim >= rank) {
        throw IndexOutOfBoundsException(
          index: d,
          min: -rank,
          max: rank - 1,
          dimension: 'RollOp dim',
        );
      }
    }

    final strides = TensorIndexer.computeStrides(shape);
    final numel = input.numel;

    final output =
        TensorBuffer.uninitialized(List<int>.from(shape), dtype: input.dtype);

    switch (input.dtype) {
      case DType.float32:
        final inData = input.storage.data as Float32List;
        final outData = output.storage.data as Float32List;

        for (int inIdx = 0; inIdx < numel; inIdx++) {
          int remaining = inIdx;
          int outIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ strides[d];
            remaining = remaining % strides[d];

            int newCoord = coord;
            // Apply shifts for relevant dims
            for (int s = 0; s < dims!.length; s++) {
              final rollDim = dims![s] < 0 ? rank + dims![s] : dims![s];
              if (d == rollDim) {
                newCoord = ((coord + shifts[s]) % shape[d] + shape[d]) % shape[d];
              }
            }

            outIdx += newCoord * strides[d];
          }

          outData[outIdx] = inData[inIdx];
        }

      case DType.float64:
        final inData = input.storage.data as Float64List;
        final outData = output.storage.data as Float64List;

        for (int inIdx = 0; inIdx < numel; inIdx++) {
          int remaining = inIdx;
          int outIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ strides[d];
            remaining = remaining % strides[d];

            int newCoord = coord;
            for (int s = 0; s < dims!.length; s++) {
              final rollDim = dims![s] < 0 ? rank + dims![s] : dims![s];
              if (d == rollDim) {
                newCoord = ((coord + shifts[s]) % shape[d] + shape[d]) % shape[d];
              }
            }

            outIdx += newCoord * strides[d];
          }

          outData[outIdx] = inData[inIdx];
        }

      default:
        final inStorage = input.storage;
        final outStorage = output.storage;

        for (int inIdx = 0; inIdx < numel; inIdx++) {
          int remaining = inIdx;
          int outIdx = 0;

          for (int d = 0; d < rank; d++) {
            final coord = remaining ~/ strides[d];
            remaining = remaining % strides[d];

            int newCoord = coord;
            for (int s = 0; s < dims!.length; s++) {
              final rollDim = dims![s] < 0 ? rank + dims![s] : dims![s];
              if (d == rollDim) {
                newCoord = ((coord + shifts[s]) % shape[d] + shape[d]) % shape[d];
              }
            }

            outIdx += newCoord * strides[d];
          }

          outStorage.setFromDouble(outIdx, inStorage.getAsDouble(inIdx));
        }
    }

    return output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
