import '../ops/normalize_op.dart';
import '../ops/permute_op.dart';
import '../ops/resize_op.dart';
import '../ops/type_cast_op.dart';
import 'tensor_pipeline.dart';

/// Pre-configured pipelines for common model preprocessing tasks.
///
/// These presets provide ready-to-use pipelines for popular model
/// architectures and use cases. All presets expect HWC (height, width,
/// channels) uint8 input and produce NCHW float32 output with a batch
/// dimension.
///
/// ## Example
///
/// ```dart
/// // Get a pipeline for ImageNet classification models
/// final pipeline = PipelinePresets.imagenetClassification();
///
/// // Process an image
/// final result = pipeline.run(imageAsTensor);
/// ```
abstract class PipelinePresets {
  /// Creates a pipeline for ImageNet classification models.
  ///
  /// Steps: resize shortest edge -> center crop -> to tensor -> normalize -> add batch
  ///
  /// Uses standard ImageNet normalization with mean `[0.485, 0.456, 0.406]`
  /// and std `[0.229, 0.224, 0.225]`.
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

  /// Creates a pipeline for ResNet and similar models.
  ///
  /// Steps: resize -> to tensor -> normalize -> add batch
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

  /// Creates a pipeline for object detection models (YOLO, SSD, etc.).
  ///
  /// Steps: resize -> to tensor -> add batch
  ///
  /// No normalization is applied (values remain in `[0, 1]`).
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
  ///
  /// Steps: resize -> to tensor -> normalize -> add batch
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

  /// Creates a pipeline for face recognition models (ArcFace, etc.).
  ///
  /// Steps: resize -> to tensor -> symmetric normalize -> add batch
  ///
  /// Uses symmetric normalization mapping `[0, 1]` to `[-1, 1]`.
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
  ///
  /// Steps: resize -> to tensor -> symmetric normalize -> add batch
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
  ///
  /// Steps: resize shortest -> center crop -> to tensor -> normalize -> add batch
  ///
  /// Uses CLIP-specific normalization values and bicubic interpolation.
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
  ///
  /// Steps: resize -> to tensor -> symmetric normalize -> add batch
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
  ///
  /// Steps: resize -> to float32 -> scale to [0,1] -> add batch
  ///
  /// Keeps NHWC layout as expected by TFLite.
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

  /// Creates a minimal pipeline with just resize and format conversion.
  ///
  /// Steps: resize -> to tensor -> add batch
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

  /// Creates a custom pipeline with configurable options.
  ///
  /// This is useful when you need a non-standard preprocessing flow.
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
