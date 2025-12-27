/// Base class for all tensor-related exceptions.
///
/// This is a sealed class, meaning all tensor exceptions must be one of:
/// - [ShapeMismatchException]: Tensor shape is incompatible with operation
/// - [NonContiguousException]: Operation requires contiguous memory layout
/// - [EmptyPipelineException]: Pipeline has no operations
/// - [InvalidParameterException]: Invalid parameter value for operation
sealed class TensorException implements Exception {
  /// The error message describing what went wrong.
  final String message;

  /// Creates a [TensorException] with the given [message].
  const TensorException(this.message);

  @override
  String toString() => '$runtimeType: $message';
}

/// Exception thrown when tensor shapes are incompatible with an operation.
///
/// This exception is thrown when:
/// - Tensor rank doesn't match expected dimensions
/// - Tensor shape doesn't match required input shape
/// - Channel count doesn't match normalization parameters
class ShapeMismatchException extends TensorException {
  /// The expected shape, if applicable.
  final List<int>? expected;

  /// The actual shape that was provided.
  final List<int> actual;

  /// Creates a [ShapeMismatchException].
  ///
  /// The [actual] shape is required. The [expected] shape and [message]
  /// are optional; if [message] is not provided, a default message is
  /// generated from [expected] and [actual].
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

/// Exception thrown when an operation requires contiguous memory layout.
///
/// Some operations like [TensorBuffer.reshape] require the tensor data to be
/// stored contiguously in memory. Call [TensorBuffer.contiguous] first to
/// create a contiguous copy.
class NonContiguousException extends TensorException {
  /// Creates a [NonContiguousException].
  ///
  /// If [operation] is provided, it is included in the error message.
  const NonContiguousException([String? operation])
      : super(
          operation != null
              ? '$operation requires contiguous tensor. Call contiguous() first.'
              : 'Operation requires contiguous tensor. Call contiguous() first.',
        );
}

/// Exception thrown when a [TensorPipeline] has no operations.
///
/// A pipeline must contain at least one [TransformOp] to be valid.
class EmptyPipelineException extends TensorException {
  /// Creates an [EmptyPipelineException].
  const EmptyPipelineException()
      : super('Pipeline must contain at least one operation');
}

/// Exception thrown when an operation receives an invalid parameter value.
///
/// Examples include negative dimensions, empty mean/std arrays, or
/// incompatible crop sizes.
class InvalidParameterException extends TensorException {
  /// The name of the invalid parameter.
  final String parameterName;

  /// The invalid value that was provided.
  final dynamic value;

  /// Creates an [InvalidParameterException].
  ///
  /// The [parameterName] and [value] are required. An optional [reason]
  /// can provide additional context about why the value is invalid.
  InvalidParameterException(this.parameterName, this.value, [String? reason])
      : super(
          reason != null
              ? 'Invalid parameter "$parameterName": $value. $reason'
              : 'Invalid parameter "$parameterName": $value',
        );
}
