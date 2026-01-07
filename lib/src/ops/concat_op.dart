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
  final output = TensorBuffer.zeros(outputShape, dtype: firstTensor.dtype);

  // Copy data from each tensor
  int outputOffset = 0;
  for (final tensor in tensors) {
    final tensorAxisSize = tensor.shape[normalizedAxis];
    _copyTensorToConcat(tensor, output, outputOffset, normalizedAxis);
    outputOffset += tensorAxisSize;
  }

  return output;
}

void _copyTensorToConcat(
  TensorBuffer source,
  TensorBuffer destination,
  int axisOffset,
  int concatAxis,
) {
  final sourceShape = source.shape;
  final destShape = destination.shape;
  final rank = sourceShape.length;

  // Recursive copy with proper axis handling
  void copyRecursive(List<int> sourceIndices, List<int> destIndices, int dim) {
    if (dim == rank) {
      // Calculate linear indices
      int srcIdx = 0;
      int srcStride = 1;
      for (int i = rank - 1; i >= 0; i--) {
        srcIdx += sourceIndices[i] * srcStride;
        srcStride *= sourceShape[i];
      }

      int destIdx = 0;
      int destStride = 1;
      for (int i = rank - 1; i >= 0; i--) {
        destIdx += destIndices[i] * destStride;
        destStride *= destShape[i];
      }

      final value = source.storage.getAsDouble(srcIdx);
      destination.storage.setFromDouble(destIdx, value);
      return;
    }

    final dimSize = sourceShape[dim];
    for (int i = 0; i < dimSize; i++) {
      sourceIndices[dim] = i;
      if (dim == concatAxis) {
        destIndices[dim] = axisOffset + i;
      } else {
        destIndices[dim] = i;
      }
      copyRecursive(sourceIndices, destIndices, dim + 1);
    }
  }

  final sourceIndices = List<int>.filled(rank, 0);
  final destIndices = List<int>.filled(rank, 0);
  copyRecursive(sourceIndices, destIndices, 0);
}
