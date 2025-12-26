import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import 'transform_op.dart';

class NormalizeOp extends TransformOp
    with InPlaceTransform, RequiresContiguous {
  final List<double> mean;
  final List<double> std;

  NormalizeOp({required this.mean, required this.std}) {
    if (mean.length != std.length) {
      throw InvalidParameterException(
        'mean/std',
        'mean.length=${mean.length}, std.length=${std.length}',
        'Must have same length',
      );
    }
    if (mean.isEmpty) {
      throw InvalidParameterException(
        'mean/std',
        'empty',
        'Must have at least one channel',
      );
    }
    for (int i = 0; i < std.length; i++) {
      if (std[i] == 0) {
        throw InvalidParameterException(
          'std[$i]',
          '0',
          'Standard deviation cannot be zero',
        );
      }
    }
  }

  factory NormalizeOp.imagenet() {
    return NormalizeOp(
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    );
  }

  factory NormalizeOp.cifar10() {
    return NormalizeOp(
      mean: [0.4914, 0.4822, 0.4465],
      std: [0.2470, 0.2435, 0.2616],
    );
  }

  factory NormalizeOp.symmetric() {
    return NormalizeOp(
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
    );
  }

  @override
  String get name => 'Normalize(mean=$mean, std=$std)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);
    _validateShape(contiguous.shape);

    final output = contiguous.clone();
    _normalize(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('NormalizeOp.applyInPlace');
    }
    _validateShape(input.shape);
    _normalize(input);
  }

  void _validateShape(List<int> shape) {
    final rank = shape.length;
    if (rank != 3 && rank != 4) {
      throw ShapeMismatchException(
        actual: shape,
        message: 'NormalizeOp requires 3D [C,H,W] or 4D [N,C,H,W] tensor',
      );
    }

    final channels = rank == 3 ? shape[0] : shape[1];
    if (channels != mean.length) {
      throw ShapeMismatchException(
        actual: shape,
        message:
            'Tensor has $channels channels, but mean/std has ${mean.length}',
      );
    }
  }

  void _normalize(TensorBuffer tensor) {
    final shape = tensor.shape;
    final rank = shape.length;

    if (rank == 3) {
      _normalize3D(tensor);
    } else {
      _normalize4D(tensor);
    }
  }

  void _normalize3D(TensorBuffer tensor) {
    final c = tensor.shape[0];
    final h = tensor.shape[1];
    final w = tensor.shape[2];
    final channelSize = h * w;

    for (int ch = 0; ch < c; ch++) {
      final offset = ch * channelSize;
      final m = mean[ch];
      final s = std[ch];

      for (int i = 0; i < channelSize; i++) {
        final idx = offset + i;
        final value = tensor.storage.getAsDouble(idx);
        tensor.storage.setFromDouble(idx, (value - m) / s);
      }
    }
  }

  void _normalize4D(TensorBuffer tensor) {
    final n = tensor.shape[0];
    final c = tensor.shape[1];
    final h = tensor.shape[2];
    final w = tensor.shape[3];
    final channelSize = h * w;
    final batchSize = c * channelSize;

    for (int batch = 0; batch < n; batch++) {
      final batchOffset = batch * batchSize;
      for (int ch = 0; ch < c; ch++) {
        final offset = batchOffset + ch * channelSize;
        final m = mean[ch];
        final s = std[ch];

        for (int i = 0; i < channelSize; i++) {
          final idx = offset + i;
          final value = tensor.storage.getAsDouble(idx);
          tensor.storage.setFromDouble(idx, (value - m) / s);
        }
      }
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}

class ScaleOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  final double scale;
  final double offset;

  ScaleOp({this.scale = 255.0, this.offset = 0.0}) {
    if (scale == 0) {
      throw InvalidParameterException('scale', '0', 'Cannot be zero');
    }
  }

  factory ScaleOp.toUnit() => ScaleOp(scale: 255.0);
  factory ScaleOp.fromUnit() => ScaleOp(scale: 1 / 255.0);
  factory ScaleOp.toSymmetric() => ScaleOp(scale: 127.5, offset: 127.5);

  @override
  String get name => 'Scale(scale=$scale, offset=$offset)';

  @override
  TensorBuffer apply(TensorBuffer input) {
    final contiguous = ensureContiguous(input);

    final output = contiguous.clone();
    _scale(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('ScaleOp.applyInPlace');
    }
    _scale(input);
  }

  void _scale(TensorBuffer tensor) {
    final numel = tensor.numel;
    for (int i = 0; i < numel; i++) {
      final value = tensor.storage.getAsDouble(i);
      tensor.storage.setFromDouble(i, (value - offset) / scale);
    }
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
