/// A high-performance tensor preprocessing library for Flutter/Dart.
///
/// This library enables NumPy-like tensor transforms for ONNX Runtime inference
/// with non-blocking isolate-based execution, type-safe ONNX tensor types,
/// and zero-copy view/stride manipulation.
///
/// ## Features
///
/// - **Type-safe tensors**: Supports all ONNX tensor types ([DType.float32],
///   [DType.int64], [DType.uint8], etc.)
/// - **Zero-copy operations**: [TensorBuffer.transpose] and [TensorBuffer.reshape]
///   use stride manipulation without copying data
/// - **Declarative pipelines**: Chain operations using [TensorPipeline]
/// - **Async execution**: Non-blocking processing via [TensorPipeline.runAsync]
///
/// ## Quick Start
///
/// ```dart
/// import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
///
/// // Create a preprocessing pipeline
/// final pipeline = TensorPipeline([
///   ResizeOp(height: 224, width: 224),
///   ToTensorOp(normalize: true),
///   NormalizeOp.imagenet(),
///   UnsqueezeOp.batch(),
/// ]);
///
/// // Process an image tensor
/// final result = pipeline.run(inputTensor);
///
/// // Or process asynchronously in an isolate
/// final result = await pipeline.runAsync(inputTensor);
/// ```
///
/// ## Memory Formats
///
/// The library supports two memory layouts:
/// - [MemoryFormat.contiguous] (NCHW): PyTorch/ONNX standard
/// - [MemoryFormat.channelsLast] (NHWC): TensorFlow/Dart image lib standard
///
/// ## Presets
///
/// Use [PipelinePresets] for common model preprocessing:
///
/// ```dart
/// final pipeline = PipelinePresets.imagenetClassification();
/// ```
library;

export 'src/core/dtype.dart';
export 'src/core/memory_format.dart';
export 'src/core/tensor_buffer.dart';
export 'src/core/tensor_storage.dart';
export 'src/exceptions/tensor_exceptions.dart';
export 'src/ops/transform_op.dart';
export 'src/ops/resize_op.dart';
export 'src/ops/normalize_op.dart';
export 'src/ops/permute_op.dart';
export 'src/ops/type_cast_op.dart';
export 'src/pipeline/tensor_pipeline.dart';
export 'src/pipeline/presets.dart';
