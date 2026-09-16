// ignore_for_file: avoid_print
import 'dart:convert';
import 'dart:io';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

/// Local timing probe, not a CI performance threshold or cross-device claim.
Future<void> main() async {
  print(Platform.version);
  for (final dtype in [DType.float32, DType.float64]) {
    for (final (shape, size, mode) in [
      ([3, 480, 640], 224, InterpolationMode.bilinear),
      ([3, 224, 224], 640, InterpolationMode.bicubic),
      ([4, 3, 480, 640], 224, InterpolationMode.bilinear),
    ]) {
      final input = TensorBuffer.random(shape, dtype: dtype, seed: 42);
      final op = ResizeOp(
        height: size,
        width: size,
        mode: mode,
        antialias: true,
      );
      final samples = <double>[];
      var checksum = 0.0;
      for (var trial = 0; trial < 5; trial++) {
        final result = await benchmark(
          'antialias',
          () {
            checksum += op(input).storage.getAsDouble(0);
          },
          warmup: 5,
          iterations: 10,
        );
        samples.add(result.avgMilliseconds);
      }
      final sorted = [...samples]..sort();
      print(
        jsonEncode({
          'dtype': dtype.name,
          'shape': shape,
          'size': size,
          'mode': mode.name,
          'samples_ms': samples,
          'median_ms': sorted[2],
          'checksum': checksum,
        }),
      );
    }
  }
}
