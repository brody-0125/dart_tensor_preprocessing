import '../ops/normalize_op.dart';
import '../ops/permute_op.dart';
import '../ops/resize_op.dart';
import '../ops/type_cast_op.dart';
import 'tensor_pipeline.dart';

/// Pre-configured preprocessing pipelines for common ML models.
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
    return TensorPipeline(
      [
        ResizeShortestOp(shortestEdge: shortestEdge, mode: interpolation),
        CenterCropOp(height: cropSize, width: cropSize),
        ToTensorOp(normalize: true),
        NormalizeOp.imagenet(),
        UnsqueezeOp.batch(),
      ],
      name: 'ImageNet Classification',
    );
  }

  /// Creates a pipeline for ResNet classification models.
  static TensorPipeline resnetClassification({
    int height = 224,
    int width = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width, mode: interpolation),
        ToTensorOp(normalize: true),
        NormalizeOp.imagenet(),
        UnsqueezeOp.batch(),
      ],
      name: 'ResNet Classification',
    );
  }

  /// Creates a pipeline for object detection models (e.g., YOLO).
  static TensorPipeline objectDetection({
    int height = 640,
    int width = 640,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width, mode: interpolation),
        ToTensorOp(normalize: true),
        UnsqueezeOp.batch(),
      ],
      name: 'Object Detection',
    );
  }

  /// Creates a pipeline for semantic segmentation models.
  static TensorPipeline segmentation({
    int height = 512,
    int width = 512,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width, mode: interpolation),
        ToTensorOp(normalize: true),
        NormalizeOp.imagenet(),
        UnsqueezeOp.batch(),
      ],
      name: 'Segmentation',
    );
  }

  /// Creates a pipeline for face recognition models (e.g., ArcFace).
  static TensorPipeline faceRecognition({
    int height = 112,
    int width = 112,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width, mode: interpolation),
        ToTensorOp(normalize: true),
        NormalizeOp.symmetric(),
        UnsqueezeOp.batch(),
      ],
      name: 'Face Recognition',
    );
  }

  /// Creates a pipeline for MobileNet models.
  static TensorPipeline mobileNet({
    int height = 224,
    int width = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width, mode: interpolation),
        ToTensorOp(normalize: true),
        NormalizeOp.symmetric(),
        UnsqueezeOp.batch(),
      ],
      name: 'MobileNet',
    );
  }

  /// Creates a pipeline for CLIP vision encoder.
  static TensorPipeline clip({
    int size = 224,
    InterpolationMode interpolation = InterpolationMode.bicubic,
  }) {
    return TensorPipeline(
      [
        ResizeShortestOp(shortestEdge: size, mode: interpolation),
        CenterCropOp(height: size, width: size),
        ToTensorOp(normalize: true),
        NormalizeOp(
          mean: [0.48145466, 0.4578275, 0.40821073],
          std: [0.26862954, 0.26130258, 0.27577711],
        ),
        UnsqueezeOp.batch(),
      ],
      name: 'CLIP',
    );
  }

  /// Creates a pipeline for Vision Transformer (ViT) models.
  static TensorPipeline vit({
    int size = 224,
    InterpolationMode interpolation = InterpolationMode.bilinear,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: size, width: size, mode: interpolation),
        ToTensorOp(normalize: true),
        NormalizeOp(
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
        ),
        UnsqueezeOp.batch(),
      ],
      name: 'ViT',
    );
  }

  /// Creates a pipeline for TensorFlow Lite models.
  static TensorPipeline tflite({
    int height = 224,
    int width = 224,
    bool normalize = true,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width),
        TypeCastOp.toFloat32(),
        if (normalize) ScaleOp.toUnit(),
        UnsqueezeOp.batch(),
      ],
      name: 'TFLite',
    );
  }

  /// Creates a minimal preprocessing pipeline with just resize and normalize.
  static TensorPipeline minimal({
    int height = 224,
    int width = 224,
  }) {
    return TensorPipeline(
      [
        ResizeOp(height: height, width: width),
        ToTensorOp(normalize: true),
        UnsqueezeOp.batch(),
      ],
      name: 'Minimal',
    );
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
    final ops = <dynamic>[
      ResizeOp(height: height, width: width, mode: interpolation),
    ];

    if (toChw) {
      ops.add(ToTensorOp(normalize: true));
    } else {
      ops.add(TypeCastOp.toFloat32());
      ops.add(ScaleOp.toUnit());
    }

    if (mean != null && std != null) {
      ops.add(NormalizeOp(mean: mean, std: std));
    }

    if (addBatchDim) {
      ops.add(UnsqueezeOp.batch());
    }

    return TensorPipeline(ops.cast(), name: 'Custom');
  }
}
