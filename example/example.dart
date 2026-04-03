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

  // Example 3: Pipeline for HWC input (common image format)
  print('\n=== Pipeline for HWC Input ===');

  // Create HWC tensor (height x width x channels) - typical image library format
  final hwcImageData = Uint8List(256 * 256 * 3);
  for (var i = 0; i < hwcImageData.length; i++) {
    hwcImageData[i] = i % 256;
  }
  final hwcTensor = TensorBuffer.fromUint8List(hwcImageData, [256, 256, 3]);
  print('Input (HWC): shape=${hwcTensor.shape}, dtype=${hwcTensor.dtype}');

  // For HWC Uint8 input, use ToTensorOp FIRST to convert to CHW Float32
  // Then apply resize and normalization on CHW format
  final hwcPipeline = TensorPipeline([
    ToTensorOp(normalize: true), // HWC uint8 -> CHW float32 [0,1]
    ResizeOp(height: 224, width: 224), // Resize CHW tensor
    NormalizeOp.imagenet(), // ImageNet normalization
    UnsqueezeOp.batch(), // Add batch dimension
  ]);

  final hwcResult = hwcPipeline.run(hwcTensor);
  print('Output: shape=${hwcResult.shape}, dtype=${hwcResult.dtype}');

  // Example 4: Pipeline for CHW input (already processed tensor)
  print('\n=== Pipeline for CHW Input ===');

  // Create CHW tensor (channels x height x width) - model input format
  final chwImageData = Float32List(3 * 256 * 256);
  for (var i = 0; i < chwImageData.length; i++) {
    chwImageData[i] = (i % 256) / 255.0; // Already normalized [0,1] values
  }
  final chwTensor = TensorBuffer.fromFloat32List(chwImageData, [3, 256, 256]);
  print('Input (CHW): shape=${chwTensor.shape}, dtype=${chwTensor.dtype}');

  // For CHW Float32 input, skip ToTensorOp (it's for HWC->CHW conversion)
  final chwPipeline = TensorPipeline([
    ResizeOp(height: 224, width: 224), // Resize CHW tensor
    NormalizeOp.imagenet(), // ImageNet normalization
    UnsqueezeOp.batch(), // Add batch dimension
  ]);

  final chwResult = chwPipeline.run(chwTensor);
  print('Output: shape=${chwResult.shape}, dtype=${chwResult.dtype}');

  // Example 5: Async execution (runs in isolate)
  print('\n=== Async Execution ===');
  final asyncResult = await chwPipeline.runAsync(chwTensor);
  print('Async result: shape=${asyncResult.shape}');

  // Example 6: In-place operations (memory efficient)
  print('\n=== In-Place Operations ===');
  final inplaceTensor = TensorBuffer.full([3, 4, 4], fillValue: 0.5);
  print('Before ReLU: first value = ${inplaceTensor[[0, 0, 0]]}');

  // Check operation capabilities
  final reluOp = ReLUOp();
  print('ReLU supports in-place: ${reluOp.capabilities.supportsInPlace}');

  // Apply in-place (no allocation)
  reluOp.applyInPlace(inplaceTensor);
  print('After ReLU: first value = ${inplaceTensor[[0, 0, 0]]}');

  // Example 7: SIMD-accelerated operations
  print('\n=== SIMD-Accelerated Operations ===');
  final largeTensor = TensorBuffer.random([3, 224, 224]);
  print('Input tensor: ${largeTensor.numel} elements');

  // These operations use SIMD for Float32 tensors
  final clipped = ClipOp(min: 0.2, max: 0.8).apply(largeTensor);
  print(
    'Clipped: min=${clipped.min().toStringAsFixed(2)}, max=${clipped.max().toStringAsFixed(2)}',
  );

  final normalized = NormalizeOp.imagenet().apply(largeTensor);
  print('Normalized: mean≈${normalized.mean().toStringAsFixed(4)}');

  // Example 8: Activation functions
  print('\n=== Activation Functions ===');
  final activationInput = TensorBuffer.fromFloat32List(
    Float32List.fromList([-2, -1, 0, 1, 2]),
    [5],
  );
  print('Input: ${activationInput.toList()}');

  final reluResult = ReLUOp().apply(activationInput);
  print('ReLU: ${reluResult.toList()}');

  final leakyResult = LeakyReLUOp(negativeSlope: 0.1).apply(activationInput);
  print('LeakyReLU(0.1): ${leakyResult.toList()}');

  final sigmoidResult = SigmoidOp().apply(activationInput);
  print(
    'Sigmoid: ${sigmoidResult.toList().map((e) => e.toStringAsFixed(3)).toList()}',
  );

  // Example 9: Math operations
  print('\n=== Math Operations ===');
  final mathInput = TensorBuffer.fromFloat32List(
    Float32List.fromList([-4, -1, 0, 1, 4]),
    [5],
  );
  print('Input: ${mathInput.toList()}');

  final absResult = AbsOp().apply(mathInput);
  print('Abs: ${absResult.toList()}');

  final sqrtInput = TensorBuffer.fromFloat32List(
    Float32List.fromList([1, 4, 9, 16, 25]),
    [5],
  );
  final sqrtResult = SqrtOp().apply(sqrtInput);
  print('Sqrt([1,4,9,16,25]): ${sqrtResult.toList()}');

  // Example 10: Fused operations (eliminates intermediate tensor)
  print('\n=== Fused Operations ===');
  final fusedOp = ResizeNormalizeFusedOp.imagenet(height: 224, width: 224);
  print('Fused op preserves shape: ${fusedOp.capabilities.preservesShape}');

  // Input: CHW format tensor
  final fusedInput = TensorBuffer.random([3, 480, 640]);
  print('Input: shape=${fusedInput.shape}');

  final fusedResult = fusedOp.apply(fusedInput);
  print('Output: shape=${fusedResult.shape} (resized + ImageNet normalized)');

  // Example 11: Normalization layers (PyTorch compatible)
  print('\n=== Normalization Layers ===');

  // BatchNorm for CNN inference
  final batchNormOp = BatchNormOp(
    runningMean: [0.5, 0.5, 0.5],
    runningVar: [0.25, 0.25, 0.25],
    weight: [1.0, 1.0, 1.0],
    bias: [0.0, 0.0, 0.0],
  );
  final bnInput = TensorBuffer.full([3, 4, 4], fillValue: 0.5);
  final bnResult = batchNormOp.apply(bnInput);
  print('BatchNorm: input mean=0.5, output first=${bnResult[[0, 0, 0]]}');

  // LayerNorm for Transformer inference
  final layerNormOp = LayerNormOp(
    normalizedShape: [4],
    weight: [1.0, 1.0, 1.0, 1.0],
    bias: [0.0, 0.0, 0.0, 0.0],
  );
  final lnInput = TensorBuffer.fromFloat32List(
    Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]),
    [2, 4],
  );
  final lnResult = layerNormOp.apply(lnInput);
  print(
    'LayerNorm: normalized over last dimension, output shape=${lnResult.shape}',
  );

  // Example 12: Buffer pooling for memory efficiency
  print('\n=== Buffer Pooling ===');
  final pool = BufferPool.instance;

  // Acquire buffer (reuses from pool if available)
  final buffer = pool.acquireFloat32(1000);
  print('Acquired buffer: ${buffer.length} elements');

  // Use buffer...
  for (var i = 0; i < buffer.length; i++) {
    buffer[i] = i.toDouble();
  }

  // Release back to pool for reuse
  pool.release(buffer);
  print(
    'Released buffer. Pool: ${pool.pooledCount} buffers, ${pool.pooledBytes} bytes',
  );

  print('\nDone!');
}
