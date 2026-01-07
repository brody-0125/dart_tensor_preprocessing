import '../core/dtype.dart';

/// Base class for all tensor-related exceptions.
sealed class TensorException implements Exception {
  /// The error message.
  final String message;

  /// Creates a tensor exception with the given [message].
  const TensorException(this.message);

  @override
  String toString() => '$runtimeType: $message';
}

/// Thrown when tensor shapes do not match expected dimensions.
class ShapeMismatchException extends TensorException {
  /// The expected shape, if known.
  final List<int>? expected;

  /// The actual shape that was provided.
  final List<int> actual;

  /// Creates a shape mismatch exception.
  ShapeMismatchException({this.expected, required this.actual, String? message})
      : super(message ?? 'Shape mismatch: expected $expected, got $actual');

  /// Creates an exception for rank mismatch.
  factory ShapeMismatchException.rank(int expectedRank, int actualRank) {
    return ShapeMismatchException(
      actual: [],
      message: 'Expected rank $expectedRank, got $actualRank',
    );
  }
}

/// Thrown when an operation requires a contiguous tensor but receives a
/// non-contiguous one.
class NonContiguousException extends TensorException {
  /// Creates a non-contiguous exception, optionally specifying the [operation].
  const NonContiguousException([String? operation])
      : super(
          operation != null
              ? '$operation requires contiguous tensor. Call contiguous() first.'
              : 'Operation requires contiguous tensor. Call contiguous() first.',
        );
}

/// Thrown when attempting to create an empty pipeline.
class EmptyPipelineException extends TensorException {
  /// Creates an empty pipeline exception.
  const EmptyPipelineException()
      : super('Pipeline must contain at least one operation');
}

/// Thrown when a transform operation receives an invalid parameter value.
class InvalidParameterException extends TensorException {
  /// The name of the invalid parameter.
  final String parameterName;

  /// The invalid value that was provided.
  final dynamic value;

  /// Creates an invalid parameter exception.
  InvalidParameterException(this.parameterName, this.value, [String? reason])
      : super(
          reason != null
              ? 'Invalid parameter "$parameterName": $value. $reason'
              : 'Invalid parameter "$parameterName": $value',
        );
}

/// Thrown when an index or axis is out of valid range.
class IndexOutOfBoundsException extends TensorException {
  /// The invalid index value.
  final int index;

  /// The minimum valid value (inclusive).
  final int min;

  /// The maximum valid value (inclusive).
  final int max;

  /// The dimension name (e.g., 'axis', 'dim').
  final String? dimension;

  /// Creates an index out of bounds exception.
  IndexOutOfBoundsException({
    required this.index,
    required this.min,
    required this.max,
    this.dimension,
  }) : super(
          dimension != null
              ? '$dimension $index is out of range [$min, $max]'
              : 'Index $index is out of range [$min, $max]',
        );
}

/// Thrown when tensor data types do not match.
class DTypeMismatchException extends TensorException {
  /// The expected data type.
  final DType expected;

  /// The actual data type.
  final DType actual;

  /// Creates a dtype mismatch exception.
  DTypeMismatchException({
    required this.expected,
    required this.actual,
  }) : super('Expected dtype ${expected.name}, got ${actual.name}');
}
