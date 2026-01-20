import '../core/tensor_buffer.dart';

// ============================================================================
// Operation Capabilities
// ============================================================================

/// Describes the capabilities and characteristics of a tensor operation.
///
/// This metadata helps pipeline optimizers understand operation behavior
/// without executing them. Use this for:
/// - In-place operation chaining
/// - Memory allocation planning
/// - Pipeline fusion opportunities
///
/// Example:
/// ```dart
/// class MyOp extends TransformOp {
///   @override
///   OperationCapabilities get capabilities => const OperationCapabilities(
///     supportsInPlace: true,
///     preservesShape: true,
///   );
/// }
/// ```
class OperationCapabilities {
  /// Whether the operation can modify the input tensor in place.
  ///
  /// Operations with `supportsInPlace: true` typically also implement
  /// the [InPlaceTransform] mixin.
  final bool supportsInPlace;

  /// Whether the operation requires contiguous memory layout.
  ///
  /// Operations with `requiresContiguous: true` typically also implement
  /// the [RequiresContiguous] mixin.
  final bool requiresContiguous;

  /// Whether the operation preserves the input shape.
  ///
  /// `true` for element-wise operations (normalize, relu, etc.)
  /// `false` for shape-changing operations (resize, reshape, etc.)
  final bool preservesShape;

  /// Whether the operation may change the data type.
  ///
  /// `true` for type casting operations
  /// `false` for most operations that preserve dtype
  final bool modifiesDType;

  /// Creates operation capabilities with the specified characteristics.
  const OperationCapabilities({
    this.supportsInPlace = false,
    this.requiresContiguous = false,
    this.preservesShape = true,
    this.modifiesDType = false,
  });

  /// Default capabilities for operations that don't specify their own.
  static const defaultCapabilities = OperationCapabilities();

  @override
  String toString() =>
      'OperationCapabilities(inPlace: $supportsInPlace, contiguous: $requiresContiguous, '
      'preservesShape: $preservesShape, modifiesDType: $modifiesDType)';
}

// ============================================================================
// Transform Operation Base Class
// ============================================================================

/// Base class for all tensor transform operations.
///
/// Subclasses implement [apply] to perform the actual transformation and
/// [computeOutputShape] to calculate the output shape without running the
/// transform.
abstract class TransformOp {
  /// The human-readable name of this operation.
  String get name;

  /// The capabilities of this operation.
  ///
  /// Override this getter to specify operation-specific capabilities.
  /// Default returns [OperationCapabilities.defaultCapabilities].
  OperationCapabilities get capabilities =>
      OperationCapabilities.defaultCapabilities;

  /// Applies this transform to [input] and returns the result.
  TensorBuffer apply(TensorBuffer input);

  /// Computes the output shape for a given [inputShape].
  List<int> computeOutputShape(List<int> inputShape);

  /// Alias for [apply].
  TensorBuffer call(TensorBuffer input) => apply(input);

  @override
  String toString() => 'TransformOp($name)';
}

/// Mixin for transforms that can modify tensors in place.
mixin InPlaceTransform on TransformOp {
  /// Applies this transform to [input] in place.
  void applyInPlace(TensorBuffer input);
}

/// Mixin for transforms that require contiguous tensor input.
mixin RequiresContiguous on TransformOp {
  /// Whether this operation requires contiguous input.
  bool get requiresContiguous => true;

  /// Returns a contiguous version of [input] if needed.
  TensorBuffer ensureContiguous(TensorBuffer input) {
    if (!input.isContiguous) {
      return input.contiguous();
    }
    return input;
  }

  /// Creates an output buffer from input, ensuring contiguity with single copy.
  ///
  /// If input is contiguous, clones it. Otherwise, creates contiguous copy.
  /// This avoids the double-copy pattern of `ensureContiguous() + clone()`.
  ///
  /// Use this when you need a contiguous output buffer that will be modified
  /// in place, without affecting the original input.
  ///
  /// Example:
  /// ```dart
  /// @override
  /// TensorBuffer apply(TensorBuffer input) {
  ///   final output = cloneForModification(input);
  ///   _modifyInPlace(output);
  ///   return output;
  /// }
  /// ```
  TensorBuffer cloneForModification(TensorBuffer input) {
    return input.isContiguous ? input.clone() : input.contiguous();
  }
}

/// A no-op transform that returns the input unchanged.
class IdentityOp extends TransformOp {
  /// Creates an identity operation.
  IdentityOp();

  @override
  String get name => 'Identity';

  @override
  TensorBuffer apply(TensorBuffer input) => input;

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
