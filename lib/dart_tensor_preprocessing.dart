/// A high-performance tensor preprocessing library for Flutter/Dart.
///
/// This library provides NumPy-like tensor transforms optimized for ONNX
/// Runtime inference. Key features include:
///
/// - Non-blocking isolate-based execution via [TensorPipeline.runAsync]
/// - Type-safe ONNX tensor types through the [DType] enum
/// - Zero-copy view/stride manipulation for [TensorBuffer.reshape] and
///   [TensorBuffer.transpose]
/// - Declarative preprocessing pipelines with [TensorPipeline]
///
/// ## Getting Started
///
/// ```dart
/// import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
///
/// // Create a preprocessing pipeline
/// final pipeline = TensorPipeline([
///   ResizeOp(height: 224, width: 224),
///   NormalizeOp.imagenet(),
///   PermuteOp.hwcToChw(),
/// ]);
///
/// // Run synchronously
/// final result = pipeline.run(inputTensor);
///
/// // Or run asynchronously in an isolate
/// final result = await pipeline.runAsync(inputTensor);
/// ```
///
/// See [PipelinePresets] for pre-configured pipelines for common models.
library;

export 'src/core/buffer_pool.dart';
export 'src/core/dtype.dart';
export 'src/core/memory_format.dart';
export 'src/core/tensor_buffer.dart';
export 'src/core/tensor_buffer_reduce.dart';
export 'src/core/tensor_storage.dart';
export 'src/exceptions/tensor_exceptions.dart';
export 'src/ops/transform_op.dart';
export 'src/ops/clip_op.dart';
export 'src/ops/pad_op.dart';
export 'src/ops/slice_op.dart';
export 'src/ops/resize_op.dart';
export 'src/ops/normalize_op.dart';
export 'src/ops/permute_op.dart';
export 'src/ops/type_cast_op.dart';
export 'src/ops/augmentation_op.dart';
export 'src/ops/concat_op.dart';
export 'src/ops/arithmetic_op.dart';
export 'src/ops/math_op.dart';
export 'src/ops/activation_op.dart';
export 'src/ops/fused_ops.dart';
export 'src/ops/batch_norm_op.dart';
export 'src/ops/group_norm_op.dart';
export 'src/ops/instance_norm_op.dart';
export 'src/ops/layer_norm_op.dart';
export 'src/ops/rms_norm_op.dart';
export 'src/pipeline/tensor_pipeline.dart';
export 'src/pipeline/presets.dart';
export 'src/service/service_config.dart';
export 'src/service/service_manager.dart';
export 'src/service/launchd/launchd_service_manager.dart';
export 'src/service/systemd/systemd_service_manager.dart';
export 'src/utils/dtype_dispatcher.dart';
export 'src/utils/simd_ops.dart';
export 'src/utils/tensor_indexing.dart';
export 'src/utils/typed_data_views.dart';
