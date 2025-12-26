import '../ops/normalize_op.dart';
import '../ops/permute_op.dart';
import '../ops/resize_op.dart';
import '../ops/type_cast_op.dart';
import 'tensor_pipeline.dart';

abstract class PipelinePresets {
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
