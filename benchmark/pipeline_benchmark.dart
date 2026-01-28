/// Pipeline performance benchmarks.
///
/// Compares sync vs async execution and measures real-world preprocessing.
// ignore_for_file: avoid_print
library;

import 'dart:typed_data';
import 'dart:math' as math;

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

/// Runs all pipeline benchmarks.
Future<List<BenchmarkResult>> runPipelineBenchmarks() async {
  final results = <BenchmarkResult>[];

  // ===== Individual Operations =====
  printHeader('Individual Operation Performance');

  // Create test tensors in CHW format for individual ops
  final tensorCHW224 = _createRandomTensor([3, 224, 224]);
  final tensorCHW640 = _createRandomTensor([3, 640, 640]);

  // ResizeOp (works on CHW format)
  final resizeOp = ResizeOp(height: 224, width: 224);
  var result = await benchmark(
    'ResizeOp(224x224)',
    () => resizeOp.apply(tensorCHW224),
    iterations: 50,
  );
  results.add(result);
  print(result);

  // NormalizeOp (works on CHW format)
  final normalizeOp = NormalizeOp.imagenet();
  final normalized224 = TensorBuffer.zeros([3, 224, 224], dtype: DType.float32);
  result = await benchmark(
    'NormalizeOp(ImageNet)',
    () => normalizeOp.apply(normalized224),
    iterations: 100,
  );
  results.add(result);
  print(result);

  // PermuteOp
  final permuteOp = PermuteOp([2, 0, 1]); // HWC -> CHW
  final tensorHWC = TensorBuffer.zeros([224, 224, 3]);
  result = await benchmark(
    'PermuteOp(HWC->CHW)',
    () => permuteOp.apply(tensorHWC),
    iterations: 100,
  );
  results.add(result);
  print(result);

  // ===== Full Pipeline =====
  printHeader('Full Pipeline Performance');

  // Create a simple custom pipeline that works with CHW input
  // (Since resnetClassification expects HWC input with ToTensorOp)
  final simplePipeline = TensorPipeline([
    NormalizeOp.imagenet(),
    UnsqueezeOp.batch(),
  ], name: 'Simple CHW Pipeline');

  result = await benchmark(
    'Simple Pipeline (CHW 224x224)',
    () => simplePipeline.run(tensorCHW224),
    iterations: 50,
  );
  results.add(result);
  print(result);

  // Minimal pipeline with resize
  final minimalPipeline = PipelinePresets.minimal();
  result = await benchmark(
    'Minimal Pipeline (CHW 224x224)',
    () => minimalPipeline.run(tensorCHW224),
    iterations: 30,
  );
  results.add(result);
  print(result);

  // Object detection style pipeline
  final detectionPipeline = PipelinePresets.objectDetection();
  result = await benchmark(
    'ObjectDetection Pipeline (CHW 640x640)',
    () => detectionPipeline.run(tensorCHW640),
    iterations: 20,
  );
  results.add(result);
  print(result);

  // ===== Sync vs Async =====
  printHeader('Sync vs Async Comparison');

  // Sync execution
  final syncResult = await benchmark(
    'Simple run() [sync]',
    () => simplePipeline.run(tensorCHW224),
    iterations: 30,
  );
  results.add(syncResult);
  print(syncResult);

  // Async execution
  final asyncResult = await benchmarkAsync(
    'Simple runAsync() [async]',
    () => simplePipeline.runAsync(tensorCHW224),
    iterations: 20,
  );
  results.add(asyncResult);
  print(asyncResult);

  // Calculate overhead
  final overhead = asyncResult.avgMilliseconds - syncResult.avgMilliseconds;
  print('  → Isolate overhead: ${overhead.toStringAsFixed(2)}ms');

  print('');

  // Larger tensor async comparison
  final syncLarge = await benchmark(
    'Detection run() [sync]',
    () => detectionPipeline.run(tensorCHW640),
    iterations: 15,
  );
  results.add(syncLarge);
  print(syncLarge);

  final asyncLarge = await benchmarkAsync(
    'Detection runAsync() [async]',
    () => detectionPipeline.runAsync(tensorCHW640),
    iterations: 10,
  );
  results.add(asyncLarge);
  print(asyncLarge);

  final overheadLarge = asyncLarge.avgMilliseconds - syncLarge.avgMilliseconds;
  print('  → Isolate overhead: ${overheadLarge.toStringAsFixed(2)}ms');

  // ===== Batch Processing =====
  printHeader('Batch Processing Simulation');

  // Process 10 images sequentially (sync)
  final images = List.generate(10, (_) => _createRandomTensor([3, 224, 224]));
  result = await benchmark(
    'Batch 10 images (sync)',
    () {
      for (final img in images) {
        simplePipeline.run(img);
      }
    },
    iterations: 5,
  );
  results.add(result);
  print(
      '${result.name}: ${(result.avgMilliseconds / 10).toStringAsFixed(2)}ms per image');

  return results;
}

/// Creates a random tensor for testing.
TensorBuffer _createRandomTensor(List<int> shape) {
  final numel = shape.reduce((a, b) => a * b);
  final random = math.Random(42);
  final data = Float32List(numel);
  for (int i = 0; i < numel; i++) {
    data[i] = random.nextDouble();
  }
  return TensorBuffer.fromFloat32List(data, shape);
}

/// Runs benchmarks if executed directly.
void main() async {
  await runPipelineBenchmarks();
}
