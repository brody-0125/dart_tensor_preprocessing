/// Tensor operations performance benchmarks.
///
/// Tests both zero-copy (O(1)) operations and copy operations.
// ignore_for_file: avoid_print
library;

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

// Shared tensors for benchmarking
late TensorBuffer _tensor1D;
late TensorBuffer _tensor2D;
late TensorBuffer _tensor3D;
late TensorBuffer _transposedTensor;

/// Runs all tensor operations benchmarks.
Future<List<BenchmarkResult>> runTensorOpsBenchmarks() async {
  final results = <BenchmarkResult>[];

  // ===== Zero-Copy Operations =====
  printHeader('Zero-Copy Operations (should be O(1))');

  // Transpose O(1) verification
  var result = await benchmarkO1(
    'transpose([N,N])',
    (size) {
      final adjustedSize = (size * size > 10000000) ? 3162 : size;
      _tensor2D = TensorBuffer.zeros([adjustedSize, adjustedSize]);
    },
    () => _tensor2D.transpose([1, 0]),
    sizes: [32, 100, 316, 1000],
  );
  results.add(result);
  print(result);

  // Reshape O(1) verification
  result = await benchmarkO1(
    'reshape(contiguous)',
    (size) {
      _tensor1D = TensorBuffer.zeros([size]);
    },
    () => _tensor1D.reshape([_tensor1D.numel]),
    sizes: [1000, 10000, 100000, 1000000],
  );
  results.add(result);
  print(result);

  // Squeeze O(1) verification
  result = await benchmarkO1(
    'squeeze()',
    (size) {
      _tensor3D = TensorBuffer.zeros([1, size, 1]);
    },
    () => _tensor3D.squeeze(),
    sizes: [1000, 10000, 100000, 1000000],
  );
  results.add(result);
  print(result);

  // Unsqueeze O(1) verification
  result = await benchmarkO1(
    'unsqueeze(0)',
    (size) {
      _tensor1D = TensorBuffer.zeros([size]);
    },
    () => _tensor1D.unsqueeze(0),
    sizes: [1000, 10000, 100000, 1000000],
  );
  results.add(result);
  print(result);

  // ===== Copy Operations =====
  printHeader('Copy Operations (scales with size)');

  // Clone - various sizes
  for (final size in [1000, 100000, 1000000, 10000000]) {
    _tensor1D = TensorBuffer.zeros([size]);
    result = await benchmark(
      'clone([$size])',
      () => _tensor1D.clone(),
      iterations: size > 1000000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  print('');

  // Contiguous on transposed tensor
  for (final dim in [100, 316, 1000]) {
    _tensor2D = TensorBuffer.zeros([dim, dim]);
    _transposedTensor = _tensor2D.transpose([1, 0]);
    final size = dim * dim;
    result = await benchmark(
      'contiguous($size from transposed)',
      () => _transposedTensor.contiguous(),
      iterations: size > 100000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  print('');

  // toList conversion
  for (final size in [1000, 100000, 1000000]) {
    _tensor1D = TensorBuffer.zeros([size]);
    result = await benchmark(
      'toList([$size])',
      () => _tensor1D.toList(),
      iterations: size > 100000 ? 10 : 100,
    );
    results.add(result);
    print(result);
  }

  return results;
}

/// Runs benchmarks if executed directly.
void main() async {
  await runTensorOpsBenchmarks();
}
