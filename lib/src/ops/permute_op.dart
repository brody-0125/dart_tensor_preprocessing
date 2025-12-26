import '../core/memory_format.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

class PermuteOp extends TransformOp {
  final List<int> dims;

  PermuteOp(this.dims) {
    if (dims.isEmpty) {
      throw InvalidParameterException(
          'dims', dims.toString(), 'Cannot be empty');
    }
  }

  factory PermuteOp.nchwToNhwc() => PermuteOp([0, 2, 3, 1]);
  factory PermuteOp.nhwcToNchw() => PermuteOp([0, 3, 1, 2]);
  factory PermuteOp.chwToHwc() => PermuteOp([1, 2, 0]);
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

class LayoutConvertOp extends TransformOp {
  final MemoryFormat targetFormat;
  final bool forceContiguous;

  LayoutConvertOp(this.targetFormat, {this.forceContiguous = true});

  factory LayoutConvertOp.toNchw({bool forceContiguous = true}) =>
      LayoutConvertOp(MemoryFormat.contiguous,
          forceContiguous: forceContiguous);

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

class UnsqueezeOp extends TransformOp {
  final int dim;

  UnsqueezeOp(this.dim);

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

class SqueezeOp extends TransformOp {
  final int? dim;

  SqueezeOp([this.dim]);

  factory SqueezeOp.batch() => SqueezeOp(0);
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

class ReshapeOp extends TransformOp {
  final List<int> targetShape;

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

class FlattenOp extends TransformOp {
  final int startDim;
  final int endDim;

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

class ContiguousOp extends TransformOp {
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
