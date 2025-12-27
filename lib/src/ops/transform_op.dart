import '../core/tensor_buffer.dart';

/// Base class for all tensor transformation operations.
///
/// A [TransformOp] takes a [TensorBuffer] input and produces a transformed
/// [TensorBuffer] output. Operations can be chained together in a
/// [TensorPipeline] for preprocessing workflows.
///
/// ## Implementing custom operations
///
/// To create a custom operation, extend this class and implement:
/// - [name]: A human-readable name for debugging
/// - [apply]: The transformation logic
/// - [computeOutputShape]: Shape calculation for validation
///
/// ## Example
///
/// ```dart
/// class MyOp extends TransformOp {
///   @override
///   String get name => 'MyOp';
///
///   @override
///   TensorBuffer apply(TensorBuffer input) {
///     // Transform the tensor
///     return input;
///   }
///
///   @override
///   List<int> computeOutputShape(List<int> inputShape) => inputShape;
/// }
/// ```
abstract class TransformOp {
  /// A human-readable name for this operation.
  String get name;

  /// Applies this transformation to the [input] tensor.
  ///
  /// Returns a new [TensorBuffer] with the transformation applied.
  TensorBuffer apply(TensorBuffer input);

  /// Computes the output shape given an [inputShape].
  ///
  /// This method is used for pipeline validation and pre-allocation.
  List<int> computeOutputShape(List<int> inputShape);

  /// Calls [apply] on the [input] tensor.
  ///
  /// This enables using the operation as a function: `op(tensor)`.
  TensorBuffer call(TensorBuffer input) => apply(input);

  @override
  String toString() => 'TransformOp($name)';
}

/// Mixin for operations that can modify tensors in place.
///
/// Operations with this mixin provide an [applyInPlace] method that modifies
/// the input tensor directly instead of creating a new one.
mixin InPlaceTransform on TransformOp {
  /// Applies this transformation in place, modifying [input] directly.
  void applyInPlace(TensorBuffer input);
}

/// Mixin for operations that require contiguous memory layout.
///
/// Operations with this mixin will automatically convert non-contiguous
/// tensors to contiguous format before processing.
mixin RequiresContiguous on TransformOp {
  /// Whether this operation requires contiguous input.
  bool get requiresContiguous => true;

  /// Returns [input] if contiguous, otherwise creates a contiguous copy.
  TensorBuffer ensureContiguous(TensorBuffer input) {
    if (!input.isContiguous) {
      return input.contiguous();
    }
    return input;
  }
}

/// A no-op transformation that returns the input unchanged.
///
/// Useful as a placeholder or for conditional pipeline construction.
class IdentityOp extends TransformOp {
  /// Creates an [IdentityOp].
  IdentityOp();

  @override
  String get name => 'Identity';

  @override
  TensorBuffer apply(TensorBuffer input) => input;

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
