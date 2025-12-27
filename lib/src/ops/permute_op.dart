import '../core/memory_format.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

/// Permutes (reorders) tensor dimensions according to a specified order.
///
/// This is a zero-copy operation that reorders dimensions by permuting strides.
///
/// ## Example
///
/// ```dart
/// // Convert NCHW to NHWC
/// final permute = PermuteOp.nchwToNhwc();
/// final nhwc = permute.apply(nchw);
/// ```
class PermuteOp extends TransformOp {
  /// The new order of dimensions.
  final List<int> dims;

  /// Creates a [PermuteOp] with the specified dimension order.
  ///
  /// The [dims] list must be a permutation of `[0, 1, ..., rank-1]`.
  PermuteOp(this.dims) {
    if (dims.isEmpty) {
      throw InvalidParameterException(
          'dims', dims.toString(), 'Cannot be empty');
    }
  }

  /// Creates a permutation from NCHW to NHWC format.
  factory PermuteOp.nchwToNhwc() => PermuteOp([0, 2, 3, 1]);

  /// Creates a permutation from NHWC to NCHW format.
  factory PermuteOp.nhwcToNchw() => PermuteOp([0, 3, 1, 2]);

  /// Creates a permutation from CHW to HWC format.
  factory PermuteOp.chwToHwc() => PermuteOp([1, 2, 0]);

  /// Creates a permutation from HWC to CHW format.
  factory PermuteOp.hwcToChw() => PermuteOp([2, 0, 1]);

  @override
  String get name => 'Permute($dims)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    if (dims.length != input.rank) {
      throw ShapeMismatchException(
        actual: input.shape,
        message:
            'Permute dims length (${dims.length}) must match tensor rank (${input.rank})',
      );
    }

    return input.transpose(dims);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (dims.length != inputShape.length) {
      throw ShapeMismatchException(
        actual: inputShape,
        message:
            'Permute dims length (${dims.length}) must match tensor rank (${inputShape.length})',
      );
    }

    return [for (final d in dims) inputShape[d]];
  }
}

/// Converts a tensor between memory layout formats (NCHW/NHWC).
///
/// This operation handles the permutation needed to convert between
/// [MemoryFormat.contiguous] (NCHW) and [MemoryFormat.channelsLast] (NHWC).
///
/// ## Example
///
/// ```dart
/// // Convert to NHWC format
/// final convert = LayoutConvertOp.toNhwc();
/// final nhwc = convert.apply(nchw);
/// ```
class LayoutConvertOp extends TransformOp {
  /// The target memory format.
  final MemoryFormat targetFormat;

  /// Whether to force the output to be contiguous.
  final bool forceContiguous;

  /// Creates a [LayoutConvertOp] to convert to [targetFormat].
  LayoutConvertOp(this.targetFormat, {this.forceContiguous = true});

  /// Creates an operation to convert to NCHW format.
  factory LayoutConvertOp.toNchw({bool forceContiguous = true}) =>
      LayoutConvertOp(MemoryFormat.contiguous,
          forceContiguous: forceContiguous);

  /// Creates an operation to convert to NHWC format.
  factory LayoutConvertOp.toNhwc({bool forceContiguous = true}) =>
      LayoutConvertOp(MemoryFormat.channelsLast,
          forceContiguous: forceContiguous);

  @override
  String get name => 'LayoutConvert(${targetFormat.layoutName})';

  @override
  TensorBuffer apply(TensorBuffer input) {
    if (input.rank != 4) {
      throw ShapeMismatchException(
        actual: input.shape,
        message: 'LayoutConvertOp requires 4D tensor [N,C,H,W] or [N,H,W,C]',
      );
    }

    if (input.memoryFormat == targetFormat) {
      return forceContiguous ? input.contiguous() : input;
    }

    final permutation = input.memoryFormat.permuteToOther;
    var result = input.transpose(permutation);

    if (forceContiguous) {
      result = result.contiguous();
    }

    return result;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if (inputShape.length != 4) {
      throw ShapeMismatchException(
        actual: inputShape,
        message: 'LayoutConvertOp requires 4D tensor',
      );
    }

    final sourceFormat = targetFormat == MemoryFormat.channelsLast
        ? MemoryFormat.contiguous
        : MemoryFormat.channelsLast;
    final permutation = sourceFormat.permuteToOther;
    return [for (final d in permutation) inputShape[d]];
  }
}

/// Adds a dimension of size 1 at the specified position.
///
/// This is a zero-copy operation commonly used to add a batch dimension.
///
/// ## Example
///
/// ```dart
/// // Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
/// final unsqueeze = UnsqueezeOp.batch();
/// ```
class UnsqueezeOp extends TransformOp {
  /// The dimension index where to insert the new dimension.
  final int dim;

  /// Creates an [UnsqueezeOp] that inserts a dimension at [dim].
  UnsqueezeOp(this.dim);

  /// Creates an operation that adds a batch dimension at position 0.
  factory UnsqueezeOp.batch() => UnsqueezeOp(0);

  @override
  String get name => 'Unsqueeze(dim=$dim)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    return input.unsqueeze(dim);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final normalizedDim = dim < 0 ? inputShape.length + dim + 1 : dim;
    if (normalizedDim < 0 || normalizedDim > inputShape.length) {
      throw RangeError.range(
          dim, -inputShape.length - 1, inputShape.length, 'dim');
    }
    return [
      ...inputShape.sublist(0, normalizedDim),
      1,
      ...inputShape.sublist(normalizedDim)
    ];
  }
}

/// Removes dimensions of size 1 from the tensor shape.
///
/// This is a zero-copy operation.
///
/// ## Example
///
/// ```dart
/// // Remove all size-1 dimensions
/// final squeeze = SqueezeOp.all();
///
/// // Remove only the batch dimension
/// final squeezeBatch = SqueezeOp.batch();
/// ```
class SqueezeOp extends TransformOp {
  /// The dimension to squeeze, or null to squeeze all size-1 dimensions.
  final int? dim;

  /// Creates a [SqueezeOp] that removes the dimension at [dim].
  ///
  /// If [dim] is null, all dimensions of size 1 are removed.
  SqueezeOp([this.dim]);

  /// Creates an operation that removes the batch dimension at position 0.
  factory SqueezeOp.batch() => SqueezeOp(0);

  /// Creates an operation that removes all dimensions of size 1.
  factory SqueezeOp.all() => SqueezeOp();

  @override
  String get name => dim != null ? 'Squeeze(dim=$dim)' : 'Squeeze(all)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    return input.squeeze(dim);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final d = dim;
    if (d != null) {
      if (d < 0 || d >= inputShape.length) {
        throw RangeError.range(d, 0, inputShape.length - 1, 'dim');
      }
      if (inputShape[d] != 1) {
        return inputShape;
      }
      return [...inputShape.sublist(0, d), ...inputShape.sublist(d + 1)];
    } else {
      return inputShape.where((dim) => dim != 1).toList();
    }
  }
}

/// Reshapes a tensor to a new shape with the same total number of elements.
///
/// Requires the tensor to be contiguous. Use `-1` for one dimension to have
/// it inferred automatically.
///
/// ## Example
///
/// ```dart
/// // Reshape to [batch, -1] where -1 is inferred
/// final reshape = ReshapeOp([1, -1]);
/// ```
class ReshapeOp extends TransformOp {
  /// The target shape, with optional `-1` for one inferred dimension.
  final List<int> targetShape;

  /// Creates a [ReshapeOp] with the specified [targetShape].
  ///
  /// At most one dimension can be `-1`, which will be inferred from
  /// the total element count.
  ReshapeOp(this.targetShape) {
    int negativeCount = 0;
    for (final dim in targetShape) {
      if (dim == -1) {
        negativeCount++;
      } else if (dim <= 0) {
        throw InvalidParameterException(
          'targetShape',
          targetShape.toString(),
          'Dimensions must be positive or -1',
        );
      }
    }
    if (negativeCount > 1) {
      throw InvalidParameterException(
        'targetShape',
        targetShape.toString(),
        'Only one dimension can be -1',
      );
    }
  }

  @override
  String get name => 'Reshape($targetShape)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final resolvedShape = _resolveShape(input.numel);
    return input.reshape(resolvedShape);
  }

  List<int> _resolveShape(int numel) {
    int product = 1;
    int negativeIdx = -1;

    for (int i = 0; i < targetShape.length; i++) {
      if (targetShape[i] == -1) {
        negativeIdx = i;
      } else {
        product *= targetShape[i];
      }
    }

    if (negativeIdx == -1) {
      return targetShape;
    }

    if (numel % product != 0) {
      throw InvalidParameterException(
        'targetShape',
        targetShape.toString(),
        'Cannot resolve -1: $numel is not divisible by $product',
      );
    }

    final resolved = List<int>.from(targetShape);
    resolved[negativeIdx] = numel ~/ product;
    return resolved;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final numel = inputShape.fold(1, (a, b) => a * b);
    return _resolveShape(numel);
  }
}

/// Flattens a range of dimensions into a single dimension.
///
/// ## Example
///
/// ```dart
/// // Flatten all dimensions except batch: [2, 3, 4, 5] -> [2, 60]
/// final flatten = FlattenOp(startDim: 1);
/// ```
class FlattenOp extends TransformOp {
  /// The first dimension to flatten (inclusive).
  final int startDim;

  /// The last dimension to flatten (inclusive). Use -1 for the last dimension.
  final int endDim;

  /// Creates a [FlattenOp] that flattens dimensions from [startDim] to [endDim].
  FlattenOp({this.startDim = 0, this.endDim = -1});

  @override
  String get name => 'Flatten(start=$startDim, end=$endDim)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final newShape = computeOutputShape(input.shape);
    return input.reshape(newShape);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    final rank = inputShape.length;
    final normalizedEnd = endDim < 0 ? rank + endDim : endDim;

    if (startDim < 0 || startDim >= rank) {
      throw RangeError.range(startDim, 0, rank - 1, 'startDim');
    }
    if (normalizedEnd < startDim || normalizedEnd >= rank) {
      throw RangeError.range(endDim, startDim, rank - 1, 'endDim');
    }

    int flattenedSize = 1;
    for (int i = startDim; i <= normalizedEnd; i++) {
      flattenedSize *= inputShape[i];
    }

    final newShape = <int>[];
    for (int i = 0; i < startDim; i++) {
      newShape.add(inputShape[i]);
    }
    newShape.add(flattenedSize);
    for (int i = normalizedEnd + 1; i < rank; i++) {
      newShape.add(inputShape[i]);
    }

    return newShape;
  }
}

/// Ensures a tensor is stored contiguously in memory.
///
/// If the tensor is already contiguous, returns it unchanged. Otherwise,
/// creates a contiguous copy.
class ContiguousOp extends TransformOp {
  /// Creates a [ContiguousOp].
  ContiguousOp();

  @override
  String get name => 'Contiguous';

  @override
  TensorBuffer apply(TensorBuffer input) {
    return input.contiguous();
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
