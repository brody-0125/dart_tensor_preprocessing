import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../utils/tensor_indexing.dart';

/// Splits a tensor into multiple sub-tensors along a dimension.
///
/// Equivalent to `torch.split()` in PyTorch.
///
/// [splitSizes] specifies the size of each split along [dim].
/// The sum of [splitSizes] must equal the tensor size along [dim].
///
/// ```dart
/// final parts = split(tensor, [2, 3], dim: 0);
/// ```
List<TensorBuffer> split(
  TensorBuffer tensor,
  List<int> splitSizes, {
  int dim = 0,
}) {
  if (splitSizes.isEmpty) {
    throw InvalidParameterException(
      'splitSizes',
      'empty list',
      'split: splitSizes cannot be empty',
    );
  }

  final rank = tensor.rank;
  final normalizedDim = dim < 0 ? rank + dim : dim;
  if (normalizedDim < 0 || normalizedDim >= rank) {
    throw IndexOutOfBoundsException(
      index: dim,
      min: -rank,
      max: rank - 1,
      dimension: 'split dim',
    );
  }

  final dimSize = tensor.shape[normalizedDim];
  final totalSplitSize = splitSizes.fold(0, (a, b) => a + b);
  if (totalSplitSize != dimSize) {
    throw InvalidParameterException(
      'splitSizes',
      splitSizes,
      'split: splitSizes sum $totalSplitSize does not match dimension size $dimSize',
    );
  }

  final contiguous = tensor.isContiguous ? tensor : tensor.contiguous();
  final shape = contiguous.shape;
  final result = <TensorBuffer>[];

  int axisOffset = 0;
  for (final size in splitSizes) {
    final outShape = List<int>.from(shape);
    outShape[normalizedDim] = size;

    final output = TensorBuffer.uninitialized(
      outShape,
      dtype: contiguous.dtype,
    );

    _copySlice(contiguous, output, normalizedDim, axisOffset);
    result.add(output);
    axisOffset += size;
  }

  return result;
}

/// Splits a tensor into a specified number of chunks.
///
/// Equivalent to `torch.chunk()` in PyTorch.
///
/// If the tensor size along [dim] is not evenly divisible by [chunks],
/// the last chunk will be smaller.
///
/// ```dart
/// final parts = chunk(tensor, 3, dim: 0);
/// ```
List<TensorBuffer> chunk(TensorBuffer tensor, int chunks, {int dim = 0}) {
  if (chunks <= 0) {
    throw InvalidParameterException(
      'chunks',
      chunks,
      'chunk: chunks must be positive',
    );
  }

  final rank = tensor.rank;
  final normalizedDim = dim < 0 ? rank + dim : dim;
  if (normalizedDim < 0 || normalizedDim >= rank) {
    throw IndexOutOfBoundsException(
      index: dim,
      min: -rank,
      max: rank - 1,
      dimension: 'chunk dim',
    );
  }

  final dimSize = tensor.shape[normalizedDim];
  final chunkSize = (dimSize + chunks - 1) ~/ chunks;

  final splitSizes = <int>[];
  int remaining = dimSize;
  for (int i = 0; i < chunks && remaining > 0; i++) {
    final size = remaining < chunkSize ? remaining : chunkSize;
    splitSizes.add(size);
    remaining -= size;
  }

  return split(tensor, splitSizes, dim: normalizedDim);
}

/// Copy a slice from source to destination along a specific axis with offset.
void _copySlice(
  TensorBuffer source,
  TensorBuffer destination,
  int axis,
  int axisOffset,
) {
  final srcShape = source.shape;
  final dstShape = destination.shape;
  final rank = srcShape.length;

  final srcStrides = TensorIndexer.computeStrides(srcShape);
  final dstStrides = TensorIndexer.computeStrides(dstShape);
  final numel = destination.numel;

  switch (source.dtype) {
    case DType.float32:
      final srcData = source.storage.data as Float32List;
      final dstData = destination.storage.data as Float32List;

      for (int dstIdx = 0; dstIdx < numel; dstIdx++) {
        int remaining = dstIdx;
        int srcIdx = 0;

        for (int d = 0; d < rank; d++) {
          final coord = remaining ~/ dstStrides[d];
          remaining = remaining % dstStrides[d];
          final srcCoord = d == axis ? coord + axisOffset : coord;
          srcIdx += srcCoord * srcStrides[d];
        }

        dstData[dstIdx] = srcData[srcIdx];
      }

    case DType.float64:
      final srcData = source.storage.data as Float64List;
      final dstData = destination.storage.data as Float64List;

      for (int dstIdx = 0; dstIdx < numel; dstIdx++) {
        int remaining = dstIdx;
        int srcIdx = 0;

        for (int d = 0; d < rank; d++) {
          final coord = remaining ~/ dstStrides[d];
          remaining = remaining % dstStrides[d];
          final srcCoord = d == axis ? coord + axisOffset : coord;
          srcIdx += srcCoord * srcStrides[d];
        }

        dstData[dstIdx] = srcData[srcIdx];
      }

    default:
      final srcStorage = source.storage;
      final dstStorage = destination.storage;

      for (int dstIdx = 0; dstIdx < numel; dstIdx++) {
        int remaining = dstIdx;
        int srcIdx = 0;

        for (int d = 0; d < rank; d++) {
          final coord = remaining ~/ dstStrides[d];
          remaining = remaining % dstStrides[d];
          final srcCoord = d == axis ? coord + axisOffset : coord;
          srcIdx += srcCoord * srcStrides[d];
        }

        dstStorage.setFromDouble(dstIdx, srcStorage.getAsDouble(srcIdx));
      }
  }
}
