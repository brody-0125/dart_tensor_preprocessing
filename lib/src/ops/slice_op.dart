import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Slices tensor with Python-like syntax.
///
/// Each dimension can have an optional slice tuple (start, end, step).
/// null values select the full range, similar to ":" in Python.
/// Negative indices are supported, but step must be positive.
class SliceOp extends TransformOp {
  /// List of slice specifications for each dimension.
  /// null slice means select all elements in that dimension.
  final List<(int? start, int? end, int? step)?> slices;

  /// Creates a slice operation with the given [slices].
  SliceOp(this.slices);

  /// Creates a slice for a range in a specific dimension.
  factory SliceOp.range(int dim, int start, int end, {int step = 1}) {
    final sliceList = List<(int? start, int? end, int? step)?>.filled(
      dim + 1,
      null,
    );
    sliceList[dim] = (start, end, step);
    return SliceOp(sliceList);
  }

  /// Creates a slice from a specific start index to the end.
  factory SliceOp.from(int dim, int start, {int step = 1}) {
    final sliceList = List<(int? start, int? end, int? step)?>.filled(
      dim + 1,
      null,
    );
    sliceList[dim] = (start, null, step);
    return SliceOp(sliceList);
  }

  /// Creates a slice from the beginning to a specific end index.
  factory SliceOp.to(int dim, int end, {int step = 1}) {
    final sliceList = List<(int? start, int? end, int? step)?>.filled(
      dim + 1,
      null,
    );
    sliceList[dim] = (null, end, step);
    return SliceOp(sliceList);
  }

  @override
  String get name => 'Slice(slices=$slices)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    _validateSlices(input.rank);

    // Convert slices to concrete ranges
    final ranges = List.generate(input.rank, (dim) {
      final slice = dim < slices.length ? slices[dim] : null;
      if (slice == null) {
        return (0, input.shape[dim], 1); // Select all
      }

      final (start, end, step) = slice;
      final concreteStart = _normalizeIndex(start ?? 0, input.shape[dim]);
      final concreteEnd = _normalizeIndex(
        end ?? input.shape[dim],
        input.shape[dim],
      );
      final concreteStep = step ?? 1;

      return (concreteStart, concreteEnd, concreteStep);
    });

    // Validate ranges
    for (int i = 0; i < ranges.length; i++) {
      final (start, end, step) = ranges[i];
      if (step <= 0) {
        throw InvalidParameterException(
          'step',
          step.toString(),
          'Step must be positive',
        );
      }
      if (start < 0 || start > input.shape[i]) {
        throw InvalidParameterException(
          'start',
          start.toString(),
          'Start index out of bounds for dimension $i',
        );
      }
      if (end < 0 || end > input.shape[i]) {
        throw InvalidParameterException(
          'end',
          end.toString(),
          'End index out of bounds for dimension $i',
        );
      }
      if (start > end) {
        throw InvalidParameterException(
          'start/end',
          '$start/$end',
          'Start must be <= end for dimension $i',
        );
      }
    }

    // Compute output shape
    final outputShape = ranges.map((range) {
      final (start, end, step) = range;
      return ((end - start) / step).ceil();
    }).toList();

    // Create output tensor
    final output = TensorBuffer.uninitialized(outputShape, dtype: input.dtype);

    // Copy data using the slice ranges
    _copySlicedData(input, output, ranges);

    return output;
  }

  void _validateSlices(int rank) {
    if (slices.length > rank) {
      throw ShapeMismatchException(
        actual: List.filled(slices.length, 0),
        message: 'Too many slices (${slices.length}) for tensor rank $rank',
      );
    }
  }

  int _normalizeIndex(int index, int dimSize) {
    if (index < 0) {
      return index + dimSize;
    }
    return index;
  }

  void _copySlicedData(
    TensorBuffer input,
    TensorBuffer output,
    List<(int, int, int)> ranges,
  ) {
    final inputIndices = List<int>.filled(input.rank, 0);
    final outputIndices = List<int>.filled(output.rank, 0);

    int outputOffset = 0;

    // Recursive copy function
    void copyDimension(int dim) {
      if (dim == input.rank) {
        // Copy single element
        final inputOffset = _computeLinearIndex(input, inputIndices);
        final val = input.storage.getAsDouble(inputOffset);
        output.storage.setFromDouble(outputOffset, val);
        outputOffset++;
        return;
      }

      final (start, end, step) = ranges[dim];
      for (int i = start; i < end; i += step) {
        inputIndices[dim] = i;
        if (dim < output.rank) {
          outputIndices[dim] = (i - start) ~/ step;
        }
        copyDimension(dim + 1);
      }
    }

    copyDimension(0);
  }

  int _computeLinearIndex(TensorBuffer tensor, List<int> indices) {
    int offset = tensor.storageOffset;
    for (int i = 0; i < indices.length; i++) {
      offset += indices[i] * tensor.strides[i];
    }
    return offset;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    _validateSlices(inputShape.length);

    final outputShape = <int>[];
    for (int dim = 0; dim < inputShape.length; dim++) {
      final slice = dim < slices.length ? slices[dim] : null;

      if (slice == null) {
        outputShape.add(inputShape[dim]);
      } else {
        final (start, end, step) = slice;
        final dimSize = inputShape[dim];
        final concreteStart = start != null
            ? _normalizeIndex(start, dimSize)
            : 0;
        final concreteEnd = end != null
            ? _normalizeIndex(end, dimSize)
            : dimSize;
        final concreteStep = step ?? 1;

        if (concreteStep <= 0) {
          throw InvalidParameterException(
            'step',
            concreteStep.toString(),
            'Step must be positive',
          );
        }

        final slicedSize = ((concreteEnd - concreteStart) / concreteStep)
            .ceil();
        outputShape.add(slicedSize);
      }
    }

    return outputShape;
  }
}
