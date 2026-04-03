/// Normalization operations performance benchmarks.
///
/// Benchmarks BatchNormOp and LayerNormOp for various tensor sizes
/// to measure CNN and Transformer inference performance.
// ignore_for_file: avoid_print
library;

import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

import 'utils/benchmark_utils.dart';

// Shared tensors and operations for benchmarking
late TensorBuffer _tensor;
late BatchNormOp _batchNormOp;
late LayerNormOp _layerNormOp;

/// Runs all normalization benchmarks.
Future<List<BenchmarkResult>> runNormalizationBenchmarks() async {
  final results = <BenchmarkResult>[];

  // ===== BatchNormOp Benchmarks =====
  printHeader('BatchNormOp (CNN Inference)');

  // Common CNN input sizes (NCHW format)
  // Sources:
  // - MobileNetV2: https://arxiv.org/abs/1801.04381 (Sandler et al., 2018)
  // - ResNet-50: https://arxiv.org/abs/1512.03385 (He et al., 2015)
  //   - Bottleneck structure: [1x1, c] -> [3x3, c] -> [1x1, 4c]
  //   - Stage outputs: conv2_x=256, conv3_x=512, conv4_x=1024, conv5_x=2048
  // - YOLOv5: https://github.com/ultralytics/yolov5 (detection head at 640x640 input)
  final batchNormConfigs = [
    // (name, shape, channels)
    // MobileNetV2 first conv: 224->112, 32 channels (verified)
    ('MobileNet [1,32,112,112]', [1, 32, 112, 112], 32),
    // ResNet-50 bottleneck inner layer: 56x56, 64 channels (conv2_x 1x1 conv)
    ('ResNet early [1,64,56,56]', [1, 64, 56, 56], 64),
    // ResNet-50 bottleneck inner layer: 28x28, 256 channels (conv3_x 3x3 conv)
    ('ResNet mid [1,256,28,28]', [1, 256, 28, 28], 256),
    // ResNet-50 bottleneck inner layer: 14x14, 512 channels (conv4_x 3x3 conv)
    ('ResNet late [1,512,14,14]', [1, 512, 14, 14], 512),
    // ResNet-50 conv5_x output: 7x7, 2048 channels (stage output, verified)
    ('ResNet final [1,2048,7,7]', [1, 2048, 7, 7], 2048),
    // YOLOv5 P3 feature map: 80x80 at 640 input, 256 channels
    ('YOLO [1,256,80,80]', [1, 256, 80, 80], 256),
  ];

  for (final (name, shape, channels) in batchNormConfigs) {
    // Create BatchNormOp with realistic parameters
    _batchNormOp = BatchNormOp(
      runningMean: List.filled(channels, 0.0),
      runningVar: List.filled(channels, 1.0),
      weight: List.filled(channels, 1.0),
      bias: List.filled(channels, 0.0),
    );

    // Create input tensor
    final numel = shape.reduce((a, b) => a * b);
    _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);

    final result = await benchmark(
      'BatchNorm $name',
      () => _batchNormOp.apply(_tensor),
      iterations: numel > 500000 ? 20 : 50,
    );
    results.add(result);
    print(result);
  }

  // BatchNorm in-place benchmark
  print('');
  print('--- In-place Operations ---');

  for (final (name, shape, channels) in batchNormConfigs.take(3)) {
    _batchNormOp = BatchNormOp(
      runningMean: List.filled(channels, 0.0),
      runningVar: List.filled(channels, 1.0),
      weight: List.filled(channels, 1.0),
      bias: List.filled(channels, 0.0),
    );

    final numel = shape.reduce((a, b) => a * b);

    final result = await benchmark('BatchNorm.inPlace $name', () {
      // Create fresh tensor each iteration for in-place
      _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);
      _batchNormOp.applyInPlace(_tensor);
    }, iterations: numel > 500000 ? 20 : 50);
    results.add(result);
    print(result);
  }

  // ===== LayerNormOp Benchmarks =====
  printHeader('LayerNormOp (Transformer Inference)');

  // Common Transformer input sizes [batch, seq_len, hidden_dim]
  // Sources:
  // - BERT: https://arxiv.org/abs/1810.04805 (Devlin et al., 2018)
  //   - base: hidden_size=768, max_position=512
  //   - large: hidden_size=1024, max_position=512
  // - GPT-2: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  //   - small: hidden_size=768, context=1024
  //   - medium: hidden_size=1024, context=1024
  //   - large: hidden_size=1280, context=1024
  // - LLaMA: https://arxiv.org/abs/2302.13971 (Touvron et al., 2023)
  //   - 7B: hidden_size=4096, context=2048
  final layerNormConfigs = [
    // (name, shape, normalizedShape)
    // BERT-base: 768 hidden, 128 tokens (common inference length)
    ('BERT-base [1,128,768]', [1, 128, 768], [768]),
    // BERT-base: 768 hidden, 512 tokens (max length)
    ('BERT-base [1,512,768]', [1, 512, 768], [768]),
    // BERT-large: 1024 hidden, 128 tokens
    ('BERT-large [1,128,1024]', [1, 128, 1024], [1024]),
    // BERT-large: 1024 hidden, 512 tokens (max length)
    ('BERT-large [1,512,1024]', [1, 512, 1024], [1024]),
    // GPT-2 small: 768 hidden, 1024 context
    ('GPT-2 [1,1024,768]', [1, 1024, 768], [768]),
    // GPT-2 large: 1280 hidden, 1024 context
    ('GPT-2 large [1,1024,1280]', [1, 1024, 1280], [1280]),
    // LLaMA-7B: 4096 hidden, 2048 context
    ('LLaMA-7B style [1,2048,4096]', [1, 2048, 4096], [4096]),
  ];

  for (final (name, shape, normalizedShape) in layerNormConfigs) {
    final normalizedSize = normalizedShape.reduce((a, b) => a * b);

    _layerNormOp = LayerNormOp(
      normalizedShape: normalizedShape,
      weight: List.filled(normalizedSize, 1.0),
      bias: List.filled(normalizedSize, 0.0),
    );

    final numel = shape.reduce((a, b) => a * b);
    _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);

    final result = await benchmark(
      'LayerNorm $name',
      () => _layerNormOp.apply(_tensor),
      iterations: numel > 500000 ? 10 : 50,
    );
    results.add(result);
    print(result);
  }

  // LayerNorm in-place benchmark
  print('');
  print('--- In-place Operations ---');

  for (final (name, shape, normalizedShape) in layerNormConfigs.take(4)) {
    final normalizedSize = normalizedShape.reduce((a, b) => a * b);

    _layerNormOp = LayerNormOp(
      normalizedShape: normalizedShape,
      weight: List.filled(normalizedSize, 1.0),
      bias: List.filled(normalizedSize, 0.0),
    );

    final numel = shape.reduce((a, b) => a * b);

    final result = await benchmark('LayerNorm.inPlace $name', () {
      _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);
      _layerNormOp.applyInPlace(_tensor);
    }, iterations: numel > 500000 ? 10 : 50);
    results.add(result);
    print(result);
  }

  // ===== Multi-dimensional LayerNorm =====
  printHeader('LayerNormOp Multi-Dimensional');

  // Vision Transformer style: normalize over trailing dimensions
  // Sources:
  // - ViT: https://arxiv.org/abs/2010.11929 (Dosovitskiy et al., 2020)
  //   - ViT-B/16: 196 patches (224/16)^2, 768 hidden
  final multiDimConfigs = [
    // ViT-Base: 196 patches (14x14 grid from 224x224 image), 768 hidden
    ('ViT patch [1,196,768]', [1, 196, 768], [768]),
    // ViT spatial normalization: normalize over spatial dims
    ('ViT [1,196,14,14] over [14,14]', [1, 196, 14, 14], [14, 14]),
    // Conv2D instance norm style: normalize over spatial dims
    ('Conv2D [1,64,28,28] over [28,28]', [1, 64, 28, 28], [28, 28]),
  ];

  for (final (name, shape, normalizedShape) in multiDimConfigs) {
    final normalizedSize = normalizedShape.reduce((a, b) => a * b);

    _layerNormOp = LayerNormOp(
      normalizedShape: normalizedShape,
      weight: List.filled(normalizedSize, 1.0),
      bias: List.filled(normalizedSize, 0.0),
    );

    final numel = shape.reduce((a, b) => a * b);
    _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);

    final result = await benchmark(
      'LayerNorm $name',
      () => _layerNormOp.apply(_tensor),
      iterations: 50,
    );
    results.add(result);
    print(result);
  }

  // ===== Comparison Summary =====
  printHeader('Throughput Summary (elements/sec)');

  // Calculate throughput for representative configs
  final throughputTests = [
    ('BatchNorm ResNet-mid', [1, 256, 28, 28], 'batch', 256),
    ('LayerNorm BERT-base', [1, 512, 768], 'layer', 768),
  ];

  for (final (name, shape, type, param) in throughputTests) {
    final numel = shape.reduce((a, b) => a * b);

    if (type == 'batch') {
      _batchNormOp = BatchNormOp(
        runningMean: List.filled(param, 0.0),
        runningVar: List.filled(param, 1.0),
        weight: List.filled(param, 1.0),
        bias: List.filled(param, 0.0),
      );
      _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);

      final result = await benchmark(
        name,
        () => _batchNormOp.apply(_tensor),
        iterations: 100,
      );

      final elementsPerSec = numel * result.opsPerSecond;
      print('$name: ${_formatElements(elementsPerSec)} elements/sec');
    } else {
      _layerNormOp = LayerNormOp(
        normalizedShape: [param],
        weight: List.filled(param, 1.0),
        bias: List.filled(param, 0.0),
      );
      _tensor = TensorBuffer.fromFloat32List(Float32List(numel), shape);

      final result = await benchmark(
        name,
        () => _layerNormOp.apply(_tensor),
        iterations: 100,
      );

      final elementsPerSec = numel * result.opsPerSecond;
      print('$name: ${_formatElements(elementsPerSec)} elements/sec');
    }
  }

  return results;
}

String _formatElements(double elements) {
  if (elements >= 1e9) {
    return '${(elements / 1e9).toStringAsFixed(2)}B';
  } else if (elements >= 1e6) {
    return '${(elements / 1e6).toStringAsFixed(2)}M';
  } else if (elements >= 1e3) {
    return '${(elements / 1e3).toStringAsFixed(2)}K';
  }
  return elements.toStringAsFixed(0);
}

/// Runs benchmarks if executed directly.
void main() async {
  await runNormalizationBenchmarks();
}
