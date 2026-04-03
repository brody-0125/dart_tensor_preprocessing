import '../core/tensor_buffer.dart';
import 'tile_op.dart';
import 'transform_op.dart';

/// Repeats a tensor along each dimension.
///
/// Equivalent to `Tensor.repeat()` in PyTorch. This has the same semantics
/// as NumPy's `tile` — each dimension is repeated `sizes[i]` times.
///
/// Delegates to [TileOp] internally.
///
/// ```dart
/// final result = RepeatOp(sizes: [2, 3])(tensor);
/// ```
class RepeatOp extends TransformOp {
  /// The number of repetitions along each dimension.
  final List<int> sizes;

  /// Creates a Repeat operation.
  RepeatOp({required this.sizes});

  @override
  String get name => 'Repeat';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
    preservesShape: false,
    requiresContiguous: true,
    pytorchEquivalent: 'Tensor.repeat',
    onnxOpType: 'Tile',
  );

  @override
  TensorBuffer apply(TensorBuffer input) => TileOp(reps: sizes).apply(input);

  @override
  List<int> computeOutputShape(List<int> inputShape) =>
      TileOp(reps: sizes).computeOutputShape(inputShape);
}
