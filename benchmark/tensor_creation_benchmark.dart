/// Tensor creation performance benchmarks.
// ignore_for_file: avoid_print
library;

import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

/// Runs all tensor creation benchmarks.
Future<List<BenchmarkResult>> runTensorCreationBenchmarks() async {
  final results = <BenchmarkResult>[];

  printHeader('Tensor Creation Benchmark');

  // zeros - various sizes
  for (final size in [1000, 100000, 1000000, 10000000]) {
    final result = await benchmark(
      'zeros([$size])',
      () => TensorBuffer.zeros([size]),
      iterations: size > 1000000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  print('');

  // ones - various sizes
  for (final size in [1000, 100000, 1000000, 10000000]) {
    final result = await benchmark(
      'ones([$size])',
      () => TensorBuffer.ones([size]),
      iterations: size > 1000000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  print('');

  // 2D tensors
  for (final dims in [
    [100, 100],
    [1000, 1000],
    [3, 224, 224],
    [3, 640, 640]
  ]) {
    final name = 'zeros(${dims.toString()})';
    final result = await benchmark(
      name,
      () => TensorBuffer.zeros(dims),
      iterations: dims.reduce((a, b) => a * b) > 1000000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  print('');

  // fromFloat32List - pre-allocated data
  for (final size in [1000, 100000, 1000000]) {
    final data = Float32List(size);
    final result = await benchmark(
      'fromFloat32List($size)',
      () => TensorBuffer.fromFloat32List(data, [size]),
      iterations: size > 100000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  return results;
}

/// Runs benchmarks if executed directly.
void main() async {
  await runTensorCreationBenchmarks();
}
