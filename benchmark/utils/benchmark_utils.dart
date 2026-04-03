/// Benchmark utilities for measuring tensor operation performance.
// ignore_for_file: avoid_print
library;

import 'dart:io';

/// Result from a single benchmark run.
class BenchmarkResult {
  /// Human-readable name of the benchmark.
  final String name;

  /// Total elapsed time for all iterations.
  final Duration elapsed;

  /// Number of iterations performed.
  final int iterations;

  /// Operations per second.
  final double opsPerSecond;

  /// Optional memory usage in bytes.
  final int? memoryBytes;

  /// Whether this is a zero-copy O(1) operation.
  final bool? isConstantTime;

  BenchmarkResult({
    required this.name,
    required this.elapsed,
    required this.iterations,
    required this.opsPerSecond,
    this.memoryBytes,
    this.isConstantTime,
  });

  /// Average time per operation in milliseconds.
  double get avgMilliseconds => elapsed.inMicroseconds / iterations / 1000;

  @override
  String toString() {
    final ms = formatDuration(
      Duration(microseconds: (elapsed.inMicroseconds / iterations).round()),
    );
    final ops = formatOps(opsPerSecond);
    final o1 = isConstantTime == true ? ' ✓ O(1)' : '';
    return '$name: $ms ($ops)$o1';
  }
}

/// Runs a benchmark with warmup and multiple iterations.
///
/// [name] - Human-readable name for the benchmark
/// [setup] - Optional setup function called once before warmup
/// [action] - The operation to benchmark
/// [warmup] - Number of warmup iterations (not measured)
/// [iterations] - Number of measured iterations
Future<BenchmarkResult> benchmark(
  String name,
  void Function() action, {
  void Function()? setup,
  int warmup = 10,
  int iterations = 100,
}) async {
  // Setup
  setup?.call();

  // Warmup
  for (int i = 0; i < warmup; i++) {
    action();
  }

  // Measure
  final stopwatch = Stopwatch()..start();
  for (int i = 0; i < iterations; i++) {
    action();
  }
  stopwatch.stop();

  final elapsed = stopwatch.elapsed;
  final opsPerSecond = iterations / (elapsed.inMicroseconds / 1000000);

  return BenchmarkResult(
    name: name,
    elapsed: elapsed,
    iterations: iterations,
    opsPerSecond: opsPerSecond,
  );
}

/// Runs an async benchmark with warmup and multiple iterations.
Future<BenchmarkResult> benchmarkAsync(
  String name,
  Future<void> Function() action, {
  void Function()? setup,
  int warmup = 5,
  int iterations = 50,
}) async {
  // Setup
  setup?.call();

  // Warmup
  for (int i = 0; i < warmup; i++) {
    await action();
  }

  // Measure
  final stopwatch = Stopwatch()..start();
  for (int i = 0; i < iterations; i++) {
    await action();
  }
  stopwatch.stop();

  final elapsed = stopwatch.elapsed;
  final opsPerSecond = iterations / (elapsed.inMicroseconds / 1000000);

  return BenchmarkResult(
    name: name,
    elapsed: elapsed,
    iterations: iterations,
    opsPerSecond: opsPerSecond,
  );
}

/// Verifies that an operation runs in O(1) time.
///
/// Runs the operation at multiple sizes and checks that timing doesn't
/// scale with size.
Future<BenchmarkResult> benchmarkO1(
  String name,
  void Function(int size) createTensor,
  void Function() action, {
  List<int> sizes = const [1000, 10000, 100000, 1000000],
  int iterations = 100,
  double toleranceFactor = 2.0,
}) async {
  final timings = <int, double>{};

  for (final size in sizes) {
    createTensor(size);

    // Warmup
    for (int i = 0; i < 10; i++) {
      action();
    }

    // Measure
    final stopwatch = Stopwatch()..start();
    for (int i = 0; i < iterations; i++) {
      action();
    }
    stopwatch.stop();

    timings[size] = stopwatch.elapsedMicroseconds / iterations;
  }

  // Check if O(1) - timing should not scale significantly with size
  final minTime = timings.values.reduce((a, b) => a < b ? a : b);
  final maxTime = timings.values.reduce((a, b) => a > b ? a : b);
  final isConstantTime = maxTime < minTime * toleranceFactor;

  final avgMicros = timings.values.reduce((a, b) => a + b) / timings.length;
  final elapsed = Duration(microseconds: (avgMicros * iterations).round());
  final opsPerSecond = 1000000 / avgMicros;

  return BenchmarkResult(
    name: name,
    elapsed: elapsed,
    iterations: iterations,
    opsPerSecond: opsPerSecond,
    isConstantTime: isConstantTime,
  );
}

/// Formats a duration for display.
String formatDuration(Duration d) {
  if (d.inMicroseconds < 1000) {
    return '${d.inMicroseconds}µs';
  } else if (d.inMilliseconds < 1000) {
    final ms = d.inMicroseconds / 1000;
    return '${ms.toStringAsFixed(2)}ms';
  } else {
    final s = d.inMilliseconds / 1000;
    return '${s.toStringAsFixed(2)}s';
  }
}

/// Formats operations per second for display.
String formatOps(double ops) {
  if (ops >= 1000000) {
    return '${(ops / 1000000).toStringAsFixed(1)}M ops/sec';
  } else if (ops >= 1000) {
    return '${(ops / 1000).toStringAsFixed(1)}K ops/sec';
  } else {
    return '${ops.toStringAsFixed(0)} ops/sec';
  }
}

/// Formats bytes for display.
String formatBytes(int bytes) {
  if (bytes >= 1024 * 1024 * 1024) {
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
  } else if (bytes >= 1024 * 1024) {
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  } else if (bytes >= 1024) {
    return '${(bytes / 1024).toStringAsFixed(1)} KB';
  } else {
    return '$bytes B';
  }
}

/// Prints a benchmark section header.
void printHeader(String title) {
  print('');
  print('=' * 50);
  print(' $title');
  print('=' * 50);
}

/// Gets current memory usage (RSS) if available.
int? getCurrentMemory() {
  try {
    return ProcessInfo.currentRss;
  } catch (_) {
    return null;
  }
}

/// Measures memory delta during an operation.
Future<int?> measureMemory(void Function() action) async {
  final before = getCurrentMemory();
  if (before == null) return null;

  // Force GC if possible
  await Future.delayed(const Duration(milliseconds: 10));

  action();

  final after = getCurrentMemory();
  if (after == null) return null;

  return after - before;
}
