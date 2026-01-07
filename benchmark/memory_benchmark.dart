/// Memory usage benchmarks.
///
/// Measures memory consumption for tensor operations.
// ignore_for_file: avoid_print
library;

import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

/// Runs all memory benchmarks.
Future<List<BenchmarkResult>> runMemoryBenchmarks() async {
  final results = <BenchmarkResult>[];

  printHeader('Memory Usage Benchmark');

  // ===== Tensor Size vs Memory =====
  print('\nTensor Size vs Expected Memory:');
  print('${'Size'.padLeft(15)} | ${'Elements'.padLeft(12)} | ${'Expected'.padLeft(12)} | ${'Actual RSS'.padLeft(12)}');
  print('${'-' * 15}-+-${'-' * 12}-+-${'-' * 12}-+-${'-' * 12}');

  for (final size in [1000, 100000, 1000000, 10000000]) {
    // Force GC
    await Future.delayed(const Duration(milliseconds: 50));
    final before = getCurrentMemory();

    // Create tensor
    final tensor = TensorBuffer.zeros([size], dtype: DType.float32);

    await Future.delayed(const Duration(milliseconds: 50));
    final after = getCurrentMemory();

    final expectedBytes = size * 4; // float32 = 4 bytes
    final actualDelta = before != null && after != null ? after - before : null;

    print(
      '${formatBytes(size * 4).padLeft(15)} | '
      '${size.toString().padLeft(12)} | '
      '${formatBytes(expectedBytes).padLeft(12)} | '
      '${(actualDelta != null ? formatBytes(actualDelta) : 'N/A').padLeft(12)}'
    );

    // Keep tensor alive
    if (tensor.numel > 0) {}
  }

  // ===== View Operations (should be ~0 memory) =====
  print('\nView Operations Memory Impact:');
  print('(Views should not allocate new data)');
  print('');

  // Create base tensor
  final baseTensor = TensorBuffer.zeros([1000, 1000], dtype: DType.float32);
  await Future.delayed(const Duration(milliseconds: 50));
  final baseMemory = getCurrentMemory();

  // Transpose (view)
  final transposed = baseTensor.transpose([1, 0]);
  await Future.delayed(const Duration(milliseconds: 50));
  final afterTranspose = getCurrentMemory();

  // Reshape (view)
  final reshaped = baseTensor.reshape([1000000]);
  await Future.delayed(const Duration(milliseconds: 50));
  final afterReshape = getCurrentMemory();

  // Unsqueeze (view)
  final unsqueezed = baseTensor.unsqueeze(0);
  await Future.delayed(const Duration(milliseconds: 50));
  final afterUnsqueeze = getCurrentMemory();

  print('Operation        | Memory Delta');
  print('${'-' * 16}-+-${'-' * 15}');
  
  if (baseMemory != null && afterTranspose != null) {
    final delta = afterTranspose - baseMemory;
    final status = delta < 1024 ? '✓ O(1)' : '⚠️ Unexpected';
    print('transpose()      | ${formatBytes(delta)} $status');
  } else {
    print('transpose()      | N/A');
  }

  if (afterTranspose != null && afterReshape != null) {
    final delta = afterReshape - afterTranspose;
    final status = delta < 1024 ? '✓ O(1)' : '⚠️ Unexpected';
    print('reshape()        | ${formatBytes(delta)} $status');
  } else {
    print('reshape()        | N/A');
  }

  if (afterReshape != null && afterUnsqueeze != null) {
    final delta = afterUnsqueeze - afterReshape;
    final status = delta < 1024 ? '✓ O(1)' : '⚠️ Unexpected';
    print('unsqueeze()      | ${formatBytes(delta)} $status');
  } else {
    print('unsqueeze()      | N/A');
  }

  // Keep views alive
  if (transposed.numel > 0 && reshaped.numel > 0 && unsqueezed.numel > 0) {}

  // ===== Copy Operations Memory =====
  print('\nCopy Operations Memory Impact:');
  
  await Future.delayed(const Duration(milliseconds: 50));
  final beforeClone = getCurrentMemory();

  final cloned = baseTensor.clone();

  await Future.delayed(const Duration(milliseconds: 50));
  final afterClone = getCurrentMemory();

  if (beforeClone != null && afterClone != null) {
    final delta = afterClone - beforeClone;
    const expected = 1000 * 1000 * 4; // 4MB for float32
    print('clone([1000,1000]): ${formatBytes(delta)} (expected ~${formatBytes(expected)})');
  } else {
    print('clone([1000,1000]): N/A');
  }

  // Keep clone alive
  if (cloned.numel > 0) {}

  // ===== Pipeline Memory =====
  printHeader('Pipeline Memory Usage');

  final image = _createTestImage();
  final pipeline = PipelinePresets.resnetClassification();

  await Future.delayed(const Duration(milliseconds: 100));
  final beforePipeline = getCurrentMemory();

  final result = pipeline.run(image);

  await Future.delayed(const Duration(milliseconds: 100));
  final afterPipeline = getCurrentMemory();

  if (beforePipeline != null && afterPipeline != null) {
    final delta = afterPipeline - beforePipeline;
    print('ImageNet Pipeline Peak: ${formatBytes(delta)}');
    print('  Input:  ${image.shape} (${formatBytes(image.numel * 4)})');
    print('  Output: ${result.shape} (${formatBytes(result.numel * 4)})');
  } else {
    print('Pipeline memory: N/A (ProcessInfo not available)');
  }

  return results;
}

/// Creates a test image tensor.
TensorBuffer _createTestImage() {
  final data = Float32List(3 * 224 * 224);
  for (int i = 0; i < data.length; i++) {
    data[i] = i % 256 / 255.0;
  }
  return TensorBuffer.fromFloat32List(data, [3, 224, 224]);
}

/// Runs benchmarks if executed directly.
void main() async {
  await runMemoryBenchmarks();
}
