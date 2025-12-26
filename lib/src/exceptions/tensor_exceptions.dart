sealed class TensorException implements Exception {
  final String message;
  const TensorException(this.message);

  @override
  String toString() => '$runtimeType: $message';
}

class ShapeMismatchException extends TensorException {
  final List<int>? expected;
  final List<int> actual;

  ShapeMismatchException({this.expected, required this.actual, String? message})
      : super(message ?? 'Shape mismatch: expected $expected, got $actual');

  factory ShapeMismatchException.rank(int expectedRank, int actualRank) {
    return ShapeMismatchException(
      actual: [],
      message: 'Expected rank $expectedRank, got $actualRank',
    );
  }
}

class NonContiguousException extends TensorException {
  const NonContiguousException([String? operation])
      : super(
          operation != null
              ? '$operation requires contiguous tensor. Call contiguous() first.'
              : 'Operation requires contiguous tensor. Call contiguous() first.',
        );
}

class EmptyPipelineException extends TensorException {
  const EmptyPipelineException()
      : super('Pipeline must contain at least one operation');
}

class InvalidParameterException extends TensorException {
  final String parameterName;
  final dynamic value;

  InvalidParameterException(this.parameterName, this.value, [String? reason])
      : super(
          reason != null
              ? 'Invalid parameter "$parameterName": $value. $reason'
              : 'Invalid parameter "$parameterName": $value',
        );
}
