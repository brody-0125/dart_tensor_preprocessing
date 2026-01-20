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

// ============================================================================
// OpValidator - Centralized operation validation
// ============================================================================

/// Centralized validation utilities for tensor operations.
///
/// Provides static methods for common validation patterns with consistent
/// error messages that include the operation name for better debugging.
///
/// Example:
/// ```dart
/// class MyOp extends TransformOp {
///   @override
///   TensorBuffer apply(TensorBuffer input) {
///     OpValidator.validateRank(
///       input.shape,
///       minRank: 3,
///       maxRank: 4,
///       operationName: name,
///     );
///     // ...
///   }
/// }
/// ```
class OpValidator {
  OpValidator._();

  /// Validates that tensor rank is within the specified range.
  ///
  /// Throws [ShapeMismatchException] if rank is outside [minRank, maxRank].
  ///
  /// Example:
  /// ```dart
  /// OpValidator.validateRank(shape, minRank: 3, maxRank: 4, operationName: 'PadOp');
  /// ```
  static void validateRank(
    List<int> shape, {
    required int minRank,
    required int maxRank,
    required String operationName,
  }) {
    final rank = shape.length;
    if (rank < minRank || rank > maxRank) {
      final rankRange = minRank == maxRank ? '$minRank' : '$minRank-$maxRank';
      throw ShapeMismatchException(
        actual: shape,
        message: '$operationName requires ${rankRange}D tensor, got ${rank}D',
      );
    }
  }

  /// Validates and normalizes an axis value (supports negative indexing).
  ///
  /// Returns the normalized (positive) axis value.
  /// Throws [IndexOutOfBoundsException] if axis is out of range.
  ///
  /// Example:
  /// ```dart
  /// final normalizedAxis = OpValidator.validateAxis(-1, rank: 4, operationName: 'SoftmaxOp');
  /// // Returns 3 for a 4D tensor
  /// ```
  static int validateAxis(
    int axis, {
    required int rank,
    required String operationName,
  }) {
    final normalizedAxis = axis < 0 ? rank + axis : axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw IndexOutOfBoundsException(
        index: axis,
        min: -rank,
        max: rank - 1,
        dimension: '$operationName axis',
      );
    }
    return normalizedAxis;
  }

  /// Validates that the channel count matches the expected value.
  ///
  /// Throws [ShapeMismatchException] if channels don't match.
  ///
  /// Example:
  /// ```dart
  /// OpValidator.validateChannels(
  ///   shape[channelDim],
  ///   expected: mean.length,
  ///   operationName: 'NormalizeOp',
  /// );
  /// ```
  static void validateChannels(
    int actual, {
    required int expected,
    required String operationName,
  }) {
    if (actual != expected) {
      throw ShapeMismatchException(
        actual: [actual],
        message:
            '$operationName: channel count mismatch. Expected $expected, got $actual',
      );
    }
  }

  /// Validates that a dimension is positive.
  ///
  /// Throws [InvalidParameterException] if dimension is not positive.
  static void validatePositiveDimension(
    int value, {
    required String paramName,
    required String operationName,
  }) {
    if (value <= 0) {
      throw InvalidParameterException(
        paramName,
        value,
        '$operationName: $paramName must be positive',
      );
    }
  }

  /// Validates that a list has the expected length.
  ///
  /// Throws [InvalidParameterException] if lengths don't match.
  static void validateListLength(
    List<dynamic> list, {
    required int expectedLength,
    required String paramName,
    required String operationName,
  }) {
    if (list.length != expectedLength) {
      throw InvalidParameterException(
        paramName,
        list.length,
        '$operationName: $paramName must have length $expectedLength, got ${list.length}',
      );
    }
  }
}
