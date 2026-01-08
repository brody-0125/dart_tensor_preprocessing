/// Utility functions for tensor validation.
///
/// These functions provide common validation patterns used across
/// tensor operations.
library;

import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';

/// Extension methods for common tensor validation patterns.
extension TensorValidation on TensorBuffer {
  /// Validates that tensor is 3D [C,H,W] or 4D [N,C,H,W].
  ///
  /// Throws [ShapeMismatchException] if tensor has different rank.
  void requireRank3Or4(String operationName) {
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            '$operationName requires 3D [C,H,W] or 4D [N,C,H,W] tensor, got ${rank}D',
      );
    }
  }

  /// Validates that tensor has the expected rank.
  ///
  /// Throws [ShapeMismatchException] if tensor has different rank.
  void requireExactRank(int expectedRank, String operationName) {
    if (rank != expectedRank) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            '$operationName requires ${expectedRank}D tensor, got ${rank}D',
      );
    }
  }

  /// Validates that tensor is at least the specified rank.
  ///
  /// Throws [ShapeMismatchException] if tensor has lower rank.
  void requireMinRank(int minRank, String operationName) {
    if (rank < minRank) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            '$operationName requires at least ${minRank}D tensor, got ${rank}D',
      );
    }
  }
}

/// Validates that a value is positive (> 0).
///
/// Throws [InvalidParameterException] if value is not positive.
void requirePositive(num value, String paramName) {
  if (value <= 0) {
    throw InvalidParameterException(
      paramName,
      value,
      'must be positive',
    );
  }
}

/// Validates that a value is non-negative (>= 0).
///
/// Throws [InvalidParameterException] if value is negative.
void requireNonNegative(num value, String paramName) {
  if (value < 0) {
    throw InvalidParameterException(
      paramName,
      value,
      'must be non-negative',
    );
  }
}
