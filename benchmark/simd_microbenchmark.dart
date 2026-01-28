/// SIMD operations microbenchmark.
///
/// Measures performance of SIMD-optimized operations vs scalar fallback.
/// Tests: clip, abs, sqrt, normalize with aligned/unaligned data.
// ignore_for_file: avoid_print
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

/// Runs all SIMD microbenchmarks.
Future<List<BenchmarkResult>> runSimdMicrobenchmarks() async {
  final results = <BenchmarkResult>[];

  // ===== SimdOps Direct Benchmarks =====
  printHeader('SimdOps Direct Performance');

  // Test sizes: small, medium, large
  final sizes = [1024, 65536, 1048576]; // 1K, 64K, 1M elements

  for (final size in sizes) {
    final sizeLabel = _formatSize(size);

    // === clip ===
    var data = _createFloat32Data(size);
    var result = await benchmark(
      'SimdOps.clip($sizeLabel)',
      () => SimdOps.clip(data, 0.0, 1.0),
      setup: () => data = _createFloat32Data(size, range: 2.0),
      iterations: size > 100000 ? 50 : 200,
    );
    results.add(result);
    print('$result  [${_formatThroughput(size, result.opsPerSecond)}]');

    // === abs ===
    data = _createFloat32Data(size);
    result = await benchmark(
      'SimdOps.abs($sizeLabel)',
      () => SimdOps.abs(data),
      setup: () => data = _createFloat32Data(size, range: 2.0, signed: true),
      iterations: size > 100000 ? 50 : 200,
    );
    results.add(result);
    print('$result  [${_formatThroughput(size, result.opsPerSecond)}]');

    // === sqrt ===
    data = _createFloat32Data(size);
    result = await benchmark(
      'SimdOps.sqrt($sizeLabel)',
      () => SimdOps.sqrt(data),
      setup: () => data = _createFloat32Data(size, range: 100.0),
      iterations: size > 100000 ? 50 : 200,
    );
    results.add(result);
    print('$result  [${_formatThroughput(size, result.opsPerSecond)}]');

    // === normalize ===
    data = _createFloat32Data(size);
    result = await benchmark(
      'SimdOps.normalize($sizeLabel)',
      () => SimdOps.normalize(data, 0.5, 0.25),
      setup: () => data = _createFloat32Data(size),
      iterations: size > 100000 ? 50 : 200,
    );
    results.add(result);
    print('$result  [${_formatThroughput(size, result.opsPerSecond)}]');

    print('');
  }

  // ===== Aligned vs Unaligned Comparison =====
  printHeader('Aligned vs Unaligned Performance');

  const alignmentTestSize = 65536;

  // Aligned data (offset = 0)
  final alignedData = Float32List(alignmentTestSize);
  _fillData(alignedData, range: 2.0, signed: true);

  var result = await benchmark(
    'abs(aligned, offset=0)',
    () => SimdOps.abs(alignedData),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(alignmentTestSize, result.opsPerSecond)}]');

  // Unaligned data (offset = 1 element = 4 bytes, not 16-byte aligned)
  final buffer = Float32List(alignmentTestSize + 1);
  _fillData(buffer, range: 2.0, signed: true);
  final unalignedData = Float32List.sublistView(buffer, 1);

  result = await benchmark(
    'abs(unaligned, offset=4)',
    () => SimdOps.abs(unalignedData),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(alignmentTestSize, result.opsPerSecond)}]');

  // ===== Float32 SIMD vs Float64 SIMD Comparison =====
  printHeader('Float32 SIMD vs Float64 SIMD');

  // Float32 (SIMD path)
  final float32Tensor =
      TensorBuffer.random([1, 3, 147, 147], dtype: DType.float32);

  result = await benchmark(
    'ClipOp(Float32, SIMD)',
    () => ClipOp(min: 0.0, max: 1.0).apply(float32Tensor),
    iterations: 100,
  );
  results.add(result);
  print(result);

  // Float64 (SIMD path - uses Float64x2)
  // Note: TensorBuffer.random has a bug with float64, use fromTypedDataList instead
  final float64Tensor = _createFloat64Tensor([1, 3, 147, 147]);

  result = await benchmark(
    'ClipOp(Float64, SIMD)',
    () => ClipOp(min: 0.0, max: 1.0).apply(float64Tensor),
    iterations: 100,
  );
  results.add(result);
  print(result);

  print('');

  // AbsOp comparison
  result = await benchmark(
    'AbsOp(Float32, SIMD)',
    () => AbsOp().apply(float32Tensor),
    iterations: 100,
  );
  results.add(result);
  print(result);

  result = await benchmark(
    'AbsOp(Float64, SIMD)',
    () => AbsOp().apply(float64Tensor),
    iterations: 100,
  );
  results.add(result);
  print(result);

  print('');

  // SqrtOp comparison
  final float32Positive =
      TensorBuffer.random([1, 3, 147, 147], dtype: DType.float32);
  final float64Positive = _createFloat64Tensor([1, 3, 147, 147]);

  result = await benchmark(
    'SqrtOp(Float32, SIMD)',
    () => SqrtOp().apply(float32Positive),
    iterations: 100,
  );
  results.add(result);
  print(result);

  result = await benchmark(
    'SqrtOp(Float64, SIMD)',
    () => SqrtOp().apply(float64Positive),
    iterations: 100,
  );
  results.add(result);
  print(result);

  print('');

  // NormalizeOp comparison
  final float32Image = TensorBuffer.random([3, 224, 224], dtype: DType.float32);
  final float64Image = _createFloat64Tensor([3, 224, 224]);

  result = await benchmark(
    'NormalizeOp(Float32, SIMD)',
    () => NormalizeOp.imagenet().apply(float32Image),
    iterations: 100,
  );
  results.add(result);
  print(result);

  result = await benchmark(
    'NormalizeOp(Float64, SIMD)',
    () => NormalizeOp.imagenet().apply(float64Image),
    iterations: 100,
  );
  results.add(result);
  print(result);

  // ===== Float64 SIMD: Aligned vs Unaligned =====
  printHeader('Float64 SIMD: Aligned vs Unaligned');

  const float64TestSize = 65536;

  // Aligned Float64 data (offset = 0)
  final alignedF64Data = Float64List(float64TestSize);
  _fillDataF64(alignedF64Data, range: 2.0, signed: true);

  result = await benchmark(
    'absF64(aligned)',
    () => SimdOps.absF64(alignedF64Data),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(float64TestSize, result.opsPerSecond)}]');

  // Unaligned Float64 data (offset = 1 element = 8 bytes, not 16-byte aligned)
  final bufferF64 = Float64List(float64TestSize + 1);
  _fillDataF64(bufferF64, range: 2.0, signed: true);
  final unalignedF64Data = Float64List.sublistView(bufferF64, 1);

  result = await benchmark(
    'absF64(unaligned)',
    () => SimdOps.absF64(unalignedF64Data),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(float64TestSize, result.opsPerSecond)}]');

  print('');

  // clipF64 aligned vs unaligned
  final alignedClipF64 = Float64List(float64TestSize);
  _fillDataF64(alignedClipF64, range: 2.0, signed: true);

  result = await benchmark(
    'clipF64(aligned)',
    () => SimdOps.clipF64(alignedClipF64, 0.0, 1.0),
    setup: () => _fillDataF64(alignedClipF64, range: 2.0, signed: true),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(float64TestSize, result.opsPerSecond)}]');

  final bufferClipF64 = Float64List(float64TestSize + 1);
  _fillDataF64(bufferClipF64, range: 2.0, signed: true);
  final unalignedClipF64 = Float64List.sublistView(bufferClipF64, 1);

  result = await benchmark(
    'clipF64(unaligned)',
    () => SimdOps.clipF64(unalignedClipF64, 0.0, 1.0),
    setup: () => _fillDataF64(bufferClipF64, range: 2.0, signed: true),
    iterations: 200,
  );
  results.add(result);
  print(
      '$result  [${_formatThroughput(float64TestSize, result.opsPerSecond)}]');

  // ===== Edge case: Non-multiple-of-4 lengths =====
  printHeader('Edge Case: Non-Multiple-of-4 Lengths');

  for (final len in [1, 3, 5, 7, 15, 17, 33, 65]) {
    final smallData = _createFloat32Data(len, range: 2.0, signed: true);
    result = await benchmark(
      'abs(len=$len)',
      () => SimdOps.abs(smallData),
      iterations: 1000,
    );
    results.add(result);
    print(result);
  }

  return results;
}

/// Creates Float32List with random data.
Float32List _createFloat32Data(
  int size, {
  double range = 1.0,
  bool signed = false,
}) {
  final data = Float32List(size);
  _fillData(data, range: range, signed: signed);
  return data;
}

/// Fills data with random values.
void _fillData(Float32List data, {double range = 1.0, bool signed = false}) {
  final random = math.Random(42);
  for (int i = 0; i < data.length; i++) {
    if (signed) {
      data[i] = (random.nextDouble() * 2 - 1) * range;
    } else {
      data[i] = random.nextDouble() * range;
    }
  }
}

/// Fills Float64 data with random values.
void _fillDataF64(Float64List data, {double range = 1.0, bool signed = false}) {
  final random = math.Random(42);
  for (int i = 0; i < data.length; i++) {
    if (signed) {
      data[i] = (random.nextDouble() * 2 - 1) * range;
    } else {
      data[i] = random.nextDouble() * range;
    }
  }
}

/// Formats size for display.
String _formatSize(int size) {
  if (size >= 1048576) {
    return '${size ~/ 1048576}M';
  } else if (size >= 1024) {
    return '${size ~/ 1024}K';
  } else {
    return '$size';
  }
}

/// Creates a Float64 tensor with random values.
/// Workaround for TensorBuffer.random bug with float64.
TensorBuffer _createFloat64Tensor(List<int> shape) {
  final numel = shape.fold(1, (a, b) => a * b);
  final data = Float64List(numel);
  final random = math.Random(42);
  for (int i = 0; i < numel; i++) {
    data[i] = random.nextDouble();
  }
  return TensorBuffer.fromFloat64List(data, shape);
}

/// Formats throughput in elements/second.
String _formatThroughput(int elements, double opsPerSec) {
  final throughput = elements * opsPerSec;
  if (throughput >= 1e9) {
    return '${(throughput / 1e9).toStringAsFixed(1)} GE/s';
  } else if (throughput >= 1e6) {
    return '${(throughput / 1e6).toStringAsFixed(1)} ME/s';
  } else if (throughput >= 1e3) {
    return '${(throughput / 1e3).toStringAsFixed(1)} KE/s';
  } else {
    return '${throughput.toStringAsFixed(0)} E/s';
  }
}

/// Runs benchmarks if executed directly.
void main() async {
  print('SIMD Microbenchmark');
  print('==================');
  print('');
  await runSimdMicrobenchmarks();
}
