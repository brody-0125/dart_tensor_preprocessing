import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';

/// Concatenates multiple tensors along a specified axis.
///
/// All tensors must have the same shape except along the concatenation axis.
/// The result tensor has the combined dimension along the specified axis.
///
/// This is a standalone utility, not a TransformOp.
TensorBuffer concat(List<TensorBuffer> tensors, {int axis = 0}) {
  if (tensors.isEmpty) {
    throw InvalidParameterException(
      'tensors',
      'empty list',
      'Cannot concat empty list of tensors',
    );
  }

  if (tensors.length == 1) {
    return tensors.first;
  }

  final firstTensor = tensors.first;
  final firstShape = firstTensor.shape;
  final rank = firstShape.length;

  // Validate axis
  if (axis < -rank || axis >= rank) {
    throw InvalidParameterException(
      'axis',
      axis.toString(),
      'Axis $axis is out of bounds for tensor rank $rank',
    );
  }

  final normalizedAxis = axis < 0 ? axis + rank : axis;

  // Validate tensor compatibility
  for (int i = 0; i < tensors.length; i++) {
    final tensor = tensors[i];
    final shape = tensor.shape;

    if (shape.length != rank) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'Tensor $i has rank ${shape.length}, expected $rank',
      );
    }

    if (tensor.dtype != firstTensor.dtype) {
      throw InvalidParameterException(
        'dtype',
        'tensor $i has dtype ${tensor.dtype}, expected ${firstTensor.dtype}',
        'All tensors must have the same dtype',
      );
    }

    // Check all dimensions except the concat axis
    for (int dim = 0; dim < rank; dim++) {
      if (dim != normalizedAxis && shape[dim] != firstShape[dim]) {
        throw ShapeMismatchException(
          actual: shape,
          message:
              'Tensor $i has shape $shape, expected dimensions except axis $axis to match $firstShape',
        );
      }
    }
  }

  // Compute output shape
  final outputShape = List<int>.from(firstShape);
  int totalAxisSize = 0;
  for (final tensor in tensors) {
    totalAxisSize += tensor.shape[normalizedAxis];
  }
  outputShape[normalizedAxis] = totalAxisSize;

  // Create output tensor
  final output =
      TensorBuffer.uninitialized(outputShape, dtype: firstTensor.dtype);

  // Optimized copy using linear indexing
  // For axis=0 and contiguous tensors, use bulk copy
  if (normalizedAxis == 0 && tensors.every((t) => t.isContiguous)) {
    _copyContiguousAxis0(tensors, output);
  } else {
    // General case: use strided copy
    int axisOffset = 0;
    for (final tensor in tensors) {
      final contiguous = tensor.isContiguous ? tensor : tensor.contiguous();
      final tensorAxisSize = contiguous.shape[normalizedAxis];
      _copyTensorToConcat(contiguous, output, axisOffset, normalizedAxis);
      axisOffset += tensorAxisSize;
    }
  }

  return output;
}

/// Optimized bulk copy for axis=0 concatenation of contiguous tensors.
void _copyContiguousAxis0(List<TensorBuffer> tensors, TensorBuffer output) {
  switch (output.dtype) {
    case DType.float32:
      final outList = output.storage.data as Float32List;
      int offset = 0;
      for (final tensor in tensors) {
        final inList = tensor.storage.data as Float32List;
        outList.setRange(offset, offset + tensor.numel, inList);
        offset += tensor.numel;
      }
    case DType.float64:
      final outList = output.storage.data as Float64List;
      int offset = 0;
      for (final tensor in tensors) {
        final inList = tensor.storage.data as Float64List;
        outList.setRange(offset, offset + tensor.numel, inList);
        offset += tensor.numel;
      }
    default:
      // Generic fallback using setRange for typed lists
      int offset = 0;
      for (final tensor in tensors) {
        for (int i = 0; i < tensor.numel; i++) {
          final value = tensor.storage.getAsDouble(i);
          output.storage.setFromDouble(offset + i, value);
        }
        offset += tensor.numel;
      }
  }
}

/// Copy tensor data using linear stride calculation instead of recursion.
void _copyTensorToConcat(
  TensorBuffer source,
  TensorBuffer destination,
  int axisOffset,
  int concatAxis,
) {
  final sourceShape = source.shape;
  final destShape = destination.shape;
  final rank = sourceShape.length;

  // Pre-compute strides for both tensors
  final srcStrides = List<int>.filled(rank, 1);
  final destStrides = List<int>.filled(rank, 1);
  for (int i = rank - 2; i >= 0; i--) {
    srcStrides[i] = srcStrides[i + 1] * sourceShape[i + 1];
    destStrides[i] = destStrides[i + 1] * destShape[i + 1];
  }

  // Compute total number of elements and iterate linearly
  final numel = source.numel;

  // Dtype-specialized for hot path optimization
  switch (source.dtype) {
    case DType.float32:
      final srcList = source.storage.data as Float32List;
      final destList = destination.storage.data as Float32List;

      for (int srcIdx = 0; srcIdx < numel; srcIdx++) {
        // Convert source linear index to multi-dimensional index
        // then to destination linear index
        int remaining = srcIdx;
        int destIdx = 0;
        for (int dim = 0; dim < rank; dim++) {
          final coord = remaining ~/ srcStrides[dim];
          remaining = remaining % srcStrides[dim];
          // Adjust coordinate for concat axis
          final destCoord = (dim == concatAxis) ? coord + axisOffset : coord;
          destIdx += destCoord * destStrides[dim];
        }
        destList[destIdx] = srcList[srcIdx];
      }

    default:
      // Generic fallback
      for (int srcIdx = 0; srcIdx < numel; srcIdx++) {
        int remaining = srcIdx;
        int destIdx = 0;
        for (int dim = 0; dim < rank; dim++) {
          final coord = remaining ~/ srcStrides[dim];
          remaining = remaining % srcStrides[dim];
          final destCoord = (dim == concatAxis) ? coord + axisOffset : coord;
          destIdx += destCoord * destStrides[dim];
        }
        final value = source.storage.getAsDouble(srcIdx);
        destination.storage.setFromDouble(destIdx, value);
      }
  }
}
