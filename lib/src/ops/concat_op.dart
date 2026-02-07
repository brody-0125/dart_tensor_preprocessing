import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';

/// Stacks a sequence of tensors along a new dimension.
///
/// Unlike [concat], which joins tensors along an existing dimension,
/// [stack] creates a new dimension and stacks tensors along it.
///
/// All input tensors must have the same shape.
///
/// Equivalent to `torch.stack()` in PyTorch.
///
/// ## Parameters
///
/// - [tensors]: List of tensors to stack. Must all have the same shape.
/// - [dim]: The dimension to insert. Default is 0 (prepend dimension).
///   Supports negative indexing (e.g., -1 means last position).
///
/// ## Examples
///
/// ```dart
/// // Stack [2,3] tensors along dim 0 -> [3,2,3]
/// final stacked = stack([a, b, c], dim: 0);
///
/// // Stack [2,3] tensors along dim 1 -> [2,3,3]
/// final stacked = stack([a, b, c], dim: 1);
///
/// // Stack [2,3] tensors along dim -1 -> [2,3,3]
/// final stacked = stack([a, b, c], dim: -1);
/// ```
TensorBuffer stack(List<TensorBuffer> tensors, {int dim = 0}) {
  if (tensors.isEmpty) {
    throw InvalidParameterException(
      'tensors',
      'empty list',
      'Cannot stack empty list of tensors',
    );
  }

  if (tensors.length == 1) {
    // Stack single tensor by adding a dimension
    return tensors.first.unsqueeze(dim);
  }

  final firstTensor = tensors.first;
  final firstShape = firstTensor.shape;
  final rank = firstShape.length;

  // Normalize dimension (supports negative indexing)
  // For stack, valid range is [-(rank+1), rank] since we're adding a new dim
  final outputRank = rank + 1;
  final normalizedDim = dim < 0 ? dim + outputRank : dim;

  if (normalizedDim < 0 || normalizedDim > rank) {
    throw InvalidParameterException(
      'dim',
      dim.toString(),
      'Dimension $dim is out of bounds for stack with tensor rank $rank',
    );
  }

  // Validate all tensors have the same shape and dtype
  for (int i = 1; i < tensors.length; i++) {
    final tensor = tensors[i];
    final shape = tensor.shape;

    if (tensor.dtype != firstTensor.dtype) {
      throw InvalidParameterException(
        'dtype',
        'tensor $i has dtype ${tensor.dtype}, expected ${firstTensor.dtype}',
        'All tensors must have the same dtype',
      );
    }

    if (shape.length != rank) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'Tensor $i has rank ${shape.length}, expected $rank',
      );
    }

    for (int d = 0; d < rank; d++) {
      if (shape[d] != firstShape[d]) {
        throw ShapeMismatchException(
          actual: shape,
          message:
              'Tensor $i has shape $shape, expected $firstShape for stack',
        );
      }
    }
  }

  // Compute output shape: insert new dimension at normalizedDim
  final outputShape = <int>[];
  for (int d = 0; d < rank; d++) {
    if (d == normalizedDim) {
      outputShape.add(tensors.length);
    }
    outputShape.add(firstShape[d]);
  }
  if (normalizedDim == rank) {
    outputShape.add(tensors.length);
  }

  // Create output tensor
  final output =
      TensorBuffer.uninitialized(outputShape, dtype: firstTensor.dtype);

  // Copy data from each tensor
  _stackTensors(tensors, output, normalizedDim);

  return output;
}

/// Internal function to copy tensor data into stacked output.
void _stackTensors(
  List<TensorBuffer> tensors,
  TensorBuffer output,
  int stackDim,
) {
  final outputShape = output.shape;
  final outputRank = outputShape.length;
  final numelPerTensor = tensors.first.numel;

  // Pre-compute output strides
  final outStrides = List<int>.filled(outputRank, 1);
  for (int i = outputRank - 2; i >= 0; i--) {
    outStrides[i] = outStrides[i + 1] * outputShape[i + 1];
  }

  // Pre-compute input strides (same for all tensors)
  final inputShape = tensors.first.shape;
  final inputRank = inputShape.length;
  final inStrides = List<int>.filled(inputRank, 1);
  for (int i = inputRank - 2; i >= 0; i--) {
    inStrides[i] = inStrides[i + 1] * inputShape[i + 1];
  }

  // Dtype-specialized copy
  switch (output.dtype) {
    case DType.float32:
      final outList = output.storage.data as Float32List;
      for (int tensorIdx = 0; tensorIdx < tensors.length; tensorIdx++) {
        final tensor = tensors[tensorIdx];
        final contiguous = tensor.isContiguous ? tensor : tensor.contiguous();
        final inList = contiguous.storage.data as Float32List;

        for (int srcIdx = 0; srcIdx < numelPerTensor; srcIdx++) {
          // Convert source index to output index
          final destIdx = _computeStackDestIndex(
            srcIdx,
            tensorIdx,
            stackDim,
            inStrides,
            outStrides,
            inputRank,
          );
          outList[destIdx] = inList[srcIdx];
        }
      }

    case DType.float64:
      final outList = output.storage.data as Float64List;
      for (int tensorIdx = 0; tensorIdx < tensors.length; tensorIdx++) {
        final tensor = tensors[tensorIdx];
        final contiguous = tensor.isContiguous ? tensor : tensor.contiguous();
        final inList = contiguous.storage.data as Float64List;

        for (int srcIdx = 0; srcIdx < numelPerTensor; srcIdx++) {
          final destIdx = _computeStackDestIndex(
            srcIdx,
            tensorIdx,
            stackDim,
            inStrides,
            outStrides,
            inputRank,
          );
          outList[destIdx] = inList[srcIdx];
        }
      }

    default:
      // Generic fallback
      for (int tensorIdx = 0; tensorIdx < tensors.length; tensorIdx++) {
        final tensor = tensors[tensorIdx];
        final contiguous = tensor.isContiguous ? tensor : tensor.contiguous();

        for (int srcIdx = 0; srcIdx < numelPerTensor; srcIdx++) {
          final destIdx = _computeStackDestIndex(
            srcIdx,
            tensorIdx,
            stackDim,
            inStrides,
            outStrides,
            inputRank,
          );
          final value = contiguous.storage.getAsDouble(srcIdx);
          output.storage.setFromDouble(destIdx, value);
        }
      }
  }
}

/// Compute destination index for stack operation.
int _computeStackDestIndex(
  int srcIdx,
  int tensorIdx,
  int stackDim,
  List<int> inStrides,
  List<int> outStrides,
  int inputRank,
) {
  int remaining = srcIdx;
  int destIdx = 0;
  int outDim = 0;

  for (int inDim = 0; inDim < inputRank; inDim++) {
    // Insert stack dimension position
    if (outDim == stackDim) {
      destIdx += tensorIdx * outStrides[outDim];
      outDim++;
    }

    // Copy coordinate from input to output
    final coord = remaining ~/ inStrides[inDim];
    remaining = remaining % inStrides[inDim];
    destIdx += coord * outStrides[outDim];
    outDim++;
  }

  // Handle case where stack dimension is at the end
  if (outDim == stackDim) {
    destIdx += tensorIdx * outStrides[outDim];
  }

  return destIdx;
}

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
