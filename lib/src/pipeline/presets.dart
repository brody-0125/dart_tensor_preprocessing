import '../core/dtype.dart';
import '../core/tensor_buffer.dart';
import '../exceptions/tensor_exceptions.dart';
import '../ops/transform_op.dart';
import '../ops/normalize_op.dart';
import '../ops/resize_op.dart';
import '../ops/type_cast_op.dart';
import 'tensor_pipeline.dart';

/// Pre-configured tensor preprocessing recipes for common ML models.
///
/// Inputs are RGB HWC/NHWC: uint8 [0,255] or float32/64 [0,1].
/// Float input is not divided by 255 again. Outputs are float32 and retain
/// an existing batch dimension. These tensor recipes do not imply exact
/// equivalence to every pretrained model's PIL-based processor.
///
/// Use these factory methods to quickly create pipelines that match the
/// preprocessing requirements of popular model architectures.
///
/// ```dart
/// // Get an ImageNet classification pipeline
/// final pipeline = PipelinePresets.imagenetClassification();
///
/// // Or create a custom pipeline
/// final custom = PipelinePresets.custom(
///   height: 256,
///   width: 256,
///   mean: [0.5, 0.5, 0.5],
///   std: [0.5, 0.5, 0.5],
/// );
/// ```
abstract class PipelinePresets {
  /// Creates a pipeline for ImageNet classification models.
  ///
  /// Resizes the shortest edge, center crops, normalizes with ImageNet stats,
  /// and adds a batch dimension.
  static TensorPipeline imagenetClassification({
    int shortestEdge = 256,
    int cropSize = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeShortestOp(
        shortestEdge: shortestEdge,
        mode: interpolation,
        antialias: true,
      ),
      CenterCropOp(height: cropSize, width: cropSize),
      NormalizeOp.imagenet(),
      _PresetOutput(),
    ], name: 'ImageNet Classification');
  }

  /// Creates a pipeline for ResNet classification models.
  static TensorPipeline resnetClassification({
    int height = 224,
    int width = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      NormalizeOp.imagenet(),
      _PresetOutput(),
    ], name: 'ResNet Classification');
  }

  /// Creates a pipeline for object detection models (e.g., YOLO).
  static TensorPipeline objectDetection({
    int height = 640,
    int width = 640,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      _PresetOutput(),
    ], name: 'Object Detection');
  }

  /// Creates a pipeline for semantic segmentation models.
  static TensorPipeline segmentation({
    int height = 512,
    int width = 512,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      NormalizeOp.imagenet(),
      _PresetOutput(),
    ], name: 'Segmentation');
  }

  /// Creates a pipeline for face recognition models (e.g., ArcFace).
  static TensorPipeline faceRecognition({
    int height = 112,
    int width = 112,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      NormalizeOp.symmetric(),
      _PresetOutput(),
    ], name: 'Face Recognition');
  }

  /// Creates a pipeline for MobileNet models.
  static TensorPipeline mobileNet({
    int height = 224,
    int width = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      NormalizeOp.symmetric(),
      _PresetOutput(),
    ], name: 'MobileNet');
  }

  /// Creates a pipeline for CLIP vision encoder.
  static TensorPipeline clip({
    int size = 224,
    InterpolationMode interpolation = InterpolationMode.bicubic,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeShortestOp(
        shortestEdge: size,
        mode: interpolation,
        antialias: true,
      ),
      CenterCropOp(height: size, width: size),
      NormalizeOp(
        mean: [0.48145466, 0.4578275, 0.40821073],
        std: [0.26862954, 0.26130258, 0.27577711],
      ),
      _PresetOutput(),
    ], name: 'CLIP');
  }

  /// Creates a pipeline for Vision Transformer (ViT) models.
  static TensorPipeline vit({
    int size = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(height: size, width: size, mode: interpolation, antialias: true),
      NormalizeOp(mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]),
      _PresetOutput(),
    ], name: 'ViT');
  }

  /// Creates a pipeline for TensorFlow Lite models.
  static TensorPipeline tflite({
    int height = 224,
    int width = 224,
    bool normalize = true,
  }) {
    return TensorPipeline([
      _PresetInput(normalize: normalize),
      ResizeOp(height: height, width: width, antialias: true),
      _PresetOutput(toChw: false),
    ], name: 'TFLite');
  }

  /// Creates a minimal preprocessing pipeline with just resize and normalize.
  static TensorPipeline minimal({int height = 224, int width = 224}) {
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(height: height, width: width, antialias: true),
      _PresetOutput(),
    ], name: 'Minimal');
  }

  /// Creates a fully customizable preprocessing pipeline.
  static TensorPipeline custom({
    required int height,
    required int width,
    InterpolationMode interpolation = InterpolationMode.bilinear,
    List<double>? mean,
    List<double>? std,
    bool addBatchDim = true,
    bool toChw = true,
  }) {
    if ((mean == null) != (std == null)) {
      throw InvalidParameterException(
        'mean/std',
        '$mean/$std',
        'Supply both mean and std',
      );
    }
    return TensorPipeline([
      _PresetInput(),
      ResizeOp(
        height: height,
        width: width,
        mode: interpolation,
        antialias: true,
      ),
      if (mean != null && std != null) NormalizeOp(mean: mean, std: std),
      _PresetOutput(toChw: toChw, addBatchDim: addBatchDim),
    ], name: 'Custom');
  }
}

class _PresetInput extends TransformOp {
  final bool normalize;
  _PresetInput({this.normalize = true});

  @override
  String get name => 'RGB input';

  @override
  TensorBuffer apply(TensorBuffer input) {
    computeOutputShape(input.shape);
    if (input.dtype != DType.uint8 &&
        input.dtype != DType.float32 &&
        input.dtype != DType.float64) {
      throw InvalidParameterException(
        'dtype',
        input.dtype,
        'Presets accept uint8 or floating-point RGB',
      );
    }
    return ToTensorOp(normalize: normalize && input.dtype == DType.uint8)(
      input,
    );
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    if ((inputShape.length != 3 && inputShape.length != 4) ||
        inputShape.last != 3) {
      throw ShapeMismatchException(
        actual: inputShape,
        message: 'Presets require RGB HWC or NHWC input',
      );
    }
    return ToTensorOp().computeOutputShape(inputShape);
  }
}

class _PresetOutput extends TransformOp {
  final bool toChw;
  final bool addBatchDim;
  _PresetOutput({this.toChw = true, this.addBatchDim = true});

  @override
  String get name => 'Image output';

  @override
  TensorBuffer apply(TensorBuffer input) {
    var output = input;
    if (!toChw) {
      output = output
          .transpose(output.rank == 3 ? [1, 2, 0] : [0, 2, 3, 1])
          .contiguous();
    }
    return addBatchDim && output.rank == 3 ? output.unsqueeze(0) : output;
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) {
    var shape = inputShape;
    if (!toChw) {
      shape = shape.length == 3
          ? [shape[1], shape[2], shape[0]]
          : [shape[0], shape[2], shape[3], shape[1]];
    }
    return addBatchDim && shape.length == 3 ? [1, ...shape] : shape;
  }
}
