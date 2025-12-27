import '../core/tensor_buffer.dart';

/// Base class for all tensor transform operations.
///
/// Subclasses implement [apply] to perform the actual transformation and
/// [computeOutputShape] to calculate the output shape without running the
/// transform.
abstract class TransformOp {
  /// The human-readable name of this operation.
  String get name;

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
