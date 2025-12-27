// ignore_for_file: avoid_print

import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() async {
  // Example 1: Create tensors
  print('=== Creating Tensors ===');
  final zeros = TensorBuffer.zeros([2, 3]);
  print('Zeros tensor: shape=${zeros.shape}, dtype=${zeros.dtype}');

  final ones = TensorBuffer.ones([2, 3], dtype: DType.float32);
  print('Ones tensor: shape=${ones.shape}, dtype=${ones.dtype}');

  final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
  final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
  print('Custom tensor: shape=${tensor.shape}, data=${tensor.toList()}');

  // Example 2: Zero-copy operations
  print('\n=== Zero-Copy Operations ===');
  final transposed = tensor.transpose([1, 0]);
  print('Transposed: shape=${transposed.shape}');

  final unsqueezed = tensor.unsqueeze(0);
  print('Unsqueezed: shape=${unsqueezed.shape}');

  final squeezed = unsqueezed.squeeze();
  print('Squeezed: shape=${squeezed.shape}');

  // Example 3: Using preset pipelines
  print('\n=== Pipeline Presets ===');

  // Simulate image data (224x224 RGB image as HWC uint8)
  final imageData = Uint8List(224 * 224 * 3);
  for (var i = 0; i < imageData.length; i++) {
    imageData[i] = i % 256;
  }
  final imageTensor = TensorBuffer.fromUint8List(imageData, [224, 224, 3]);
  print('Input image: shape=${imageTensor.shape}, dtype=${imageTensor.dtype}');

  // Use ImageNet classification preset
  final pipeline = PipelinePresets.imagenetClassification();
  final result = pipeline.run(imageTensor);
  print('Output tensor: shape=${result.shape}, dtype=${result.dtype}');

  // Example 4: Custom pipeline
  print('\n=== Custom Pipeline ===');
  final customPipeline = TensorPipeline([
    ResizeOp(height: 224, width: 224),
    ToTensorOp(normalize: true),
    NormalizeOp.imagenet(),
    UnsqueezeOp.batch(),
  ]);

  final customResult = customPipeline.run(imageTensor);
  print('Custom pipeline output: shape=${customResult.shape}');

  // Example 5: Async execution (runs in isolate)
  print('\n=== Async Execution ===');
  final asyncResult = await pipeline.runAsync(imageTensor);
  print('Async result: shape=${asyncResult.shape}');

  print('\nDone!');
}
