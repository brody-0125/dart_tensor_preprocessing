/// Utility functions for tensor validation.
///
/// These functions provide common validation patterns used across
/// tensor operations.
library;

import '../core/dtype.dart';
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
    throw InvalidParameterException(paramName, value, 'must be positive');
  }
}

/// Validates that a value is non-negative (>= 0).
///
/// Throws [InvalidParameterException] if value is negative.
void requireNonNegative(num value, String paramName) {
  if (value < 0) {
    throw InvalidParameterException(paramName, value, 'must be non-negative');
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

  /// Validates that a tensor is a 3D [C,H,W] or 4D [N,C,H,W] image tensor.
  ///
  /// Optionally validates that the channel dimension matches [expectedChannels].
  /// For 3D tensors, channel is dim 0. For 4D tensors, channel is dim 1.
  ///
  /// Throws [ShapeMismatchException] if validation fails.
  static void validateImageTensor(
    TensorBuffer tensor, {
    required String operationName,
    int? expectedChannels,
  }) {
    final rank = tensor.rank;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: tensor.shape,
        message:
            '$operationName requires 3D [C,H,W] or 4D [N,C,H,W] tensor, got ${rank}D',
      );
    }
    if (expectedChannels != null) {
      final channelDim = rank == 3 ? 0 : 1;
      final actualChannels = tensor.shape[channelDim];
      if (actualChannels != expectedChannels) {
        throw ShapeMismatchException(
          actual: tensor.shape,
          message:
              '$operationName: expected $expectedChannels channels, got $actualChannels',
        );
      }
    }
  }

  /// Validates that two shapes are broadcast-compatible and returns the
  /// broadcast output shape.
  ///
  /// Broadcasting follows NumPy/PyTorch rules:
  /// - Shapes are compared from trailing dimensions
  /// - Dimensions are compatible if equal or one of them is 1
  /// - Missing dimensions are treated as 1
  ///
  /// Throws [ShapeMismatchException] if shapes are not broadcast-compatible.
  static List<int> validateBroadcast(
    List<int> shape1,
    List<int> shape2, {
    required String operationName,
  }) {
    final maxRank = shape1.length > shape2.length
        ? shape1.length
        : shape2.length;
    final result = List<int>.filled(maxRank, 0);

    for (int i = 0; i < maxRank; i++) {
      final d1 = i < shape1.length ? shape1[shape1.length - 1 - i] : 1;
      final d2 = i < shape2.length ? shape2[shape2.length - 1 - i] : 1;

      if (d1 == d2) {
        result[maxRank - 1 - i] = d1;
      } else if (d1 == 1) {
        result[maxRank - 1 - i] = d2;
      } else if (d2 == 1) {
        result[maxRank - 1 - i] = d1;
      } else {
        throw ShapeMismatchException(
          actual: shape2,
          expected: shape1,
          message:
              '$operationName: shapes $shape1 and $shape2 are not broadcast-compatible',
        );
      }
    }

    return result;
  }

  /// Validates that a dtype is a floating-point type (float32 or float64).
  ///
  /// Throws [DTypeMismatchException] if dtype is not floating-point.
  static void validateFloatDType(DType dtype, {required String operationName}) {
    if (!dtype.isFloatingPoint) {
      throw DTypeMismatchException(expected: DType.float32, actual: dtype);
    }
  }
}
