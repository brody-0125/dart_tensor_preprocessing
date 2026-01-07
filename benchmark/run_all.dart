/// Main benchmark runner.
///
/// Runs all benchmarks and outputs formatted results.
// ignore_for_file: avoid_print
library;

import 'tensor_creation_benchmark.dart' as creation;
import 'tensor_ops_benchmark.dart' as ops;
import 'pipeline_benchmark.dart' as pipeline;
import 'memory_benchmark.dart' as memory;
import 'utils/benchmark_utils.dart';

void main() async {
  print('');
  print('╔═══════════════════════════════════════════════════════════════╗');
  print('║        dart_tensor_preprocessing Performance Benchmark        ║');
  print('╚═══════════════════════════════════════════════════════════════╝');

  final stopwatch = Stopwatch()..start();

  // Run all benchmarks
  await creation.runTensorCreationBenchmarks();
  await ops.runTensorOpsBenchmarks();
  await pipeline.runPipelineBenchmarks();
  await memory.runMemoryBenchmarks();

  stopwatch.stop();

  // Summary
  printHeader('Benchmark Complete');
  print('Total time: ${formatDuration(stopwatch.elapsed)}');
  print('');
}
