import '../core/tensor_buffer.dart';

abstract class TransformOp {
  String get name;
  TensorBuffer apply(TensorBuffer input);
  List<int> computeOutputShape(List<int> inputShape);
  TensorBuffer call(TensorBuffer input) => apply(input);

  @override
  String toString() => 'TransformOp($name)';
}

mixin InPlaceTransform on TransformOp {
  void applyInPlace(TensorBuffer input);
}

mixin RequiresContiguous on TransformOp {
  bool get requiresContiguous => true;

  TensorBuffer ensureContiguous(TensorBuffer input) {
    if (!input.isContiguous) {
      return input.contiguous();
    }
    return input;
  }
}

class IdentityOp extends TransformOp {
  IdentityOp();

  @override
  String get name => 'Identity';

  @override
  TensorBuffer apply(TensorBuffer input) => input;

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
