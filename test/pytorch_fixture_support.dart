import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:crypto/crypto.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

double fixtureNumber(dynamic value) =>
    value is num ? value.toDouble() : double.parse(value as String);

TensorBuffer fixtureTensor(Map<String, dynamic> fixture) {
  final dtype = DType.values.byName(fixture['dtype'] as String);
  final data = dtype.createBuffer((fixture['values'] as List).length);
  final storage = TensorStorage(data, dtype);
  final values = fixture['values'] as List;
  for (var i = 0; i < values.length; i++) {
    if (data is List<int>) {
      (data as List<int>)[i] = int.parse(values[i].toString());
    } else {
      storage.setFromDouble(i, fixtureNumber(values[i]));
    }
  }
  return TensorBuffer(
    storage: storage,
    shape: (fixture['shape'] as List).cast<int>(),
  );
}

void expectFixture(
  TensorBuffer actual,
  Map<String, dynamic> expected, {
  required double atol,
  required double rtol,
}) {
  expect(actual.shape, expected['shape'], reason: 'shape');
  expect(actual.dtype.name, expected['dtype'], reason: 'dtype');
  final values = expected['values'] as List;
  expect(actual.numel, values.length);
  var maxAbs = 0.0;
  var maxRel = 0.0;
  String? first;
  for (var i = 0; i < actual.numel; i++) {
    var remainder = i;
    var offset = actual.storageOffset;
    final coords = List<int>.filled(actual.rank, 0);
    for (var d = actual.rank - 1; d >= 0; d--) {
      coords[d] = remainder % actual.shape[d];
      remainder ~/= actual.shape[d];
      offset += coords[d] * actual.strides[d];
    }
    final data = actual.storage.data;
    if (data is List<int>) {
      expect(
        (data as List<int>)[offset],
        int.parse(values[i].toString()),
        reason: '$coords',
      );
      continue;
    }
    final a = actual.storage.getAsDouble(offset);
    final e = fixtureNumber(values[i]);
    final error = (a - e).abs();
    final matches = e.isNaN
        ? a.isNaN
        : e.isInfinite
        ? a == e
        : a.isFinite && error <= atol + rtol * e.abs();
    if (error.isFinite) {
      maxAbs = math.max(maxAbs, error);
      maxRel = math.max(maxRel, error / math.max(e.abs(), 1e-300));
    }
    if (!matches) first ??= '$coords expected $e, actual $a';
  }
  expect(first, isNull, reason: '$first; max abs=$maxAbs, max rel=$maxRel');
}

TransformOp fixtureOperation(Map<String, dynamic> c) {
  final p = c['params'] as Map<String, dynamic>;
  return switch (c['op']) {
    'tanh' => TanhOp(),
    'sigmoid' => SigmoidOp(),
    'relu' => ReLUOp(),
    'leaky_relu' => LeakyReLUOp(),
    'silu' => SiLUOp(),
    'mish' => MishOp(),
    'hardsigmoid' => HardsigmoidOp(),
    'hardswish' => HardswishOp(),
    'elu' => ELUOp(),
    'selu' => SELUOp(),
    'abs' => AbsOp(),
    'softmax' => SoftmaxOp(axis: p['axis'] as int),
    'center_crop' => CenterCropOp(
      height: p['height'] as int,
      width: p['width'] as int,
    ),
    'resize_shortest' => ResizeShortestOp(shortestEdge: p['size'] as int),
    'resize' => ResizeOp(
      height: p['height'] as int,
      width: p['width'] as int,
      mode: InterpolationMode.values.byName(p['mode'] as String),
      antialias: p['antialias'] as bool? ?? false,
      alignCorners: p['align_corners'] as bool? ?? false,
    ),
    _ => throw StateError('Unknown fixture operation: ${c['op']}'),
  };
}

TensorPipeline fixturePipeline(Map<String, dynamic> p) {
  final size = p['size'] as int;
  return switch (p['name']) {
    'imagenet' => PipelinePresets.imagenetClassification(
      shortestEdge: p['shortest_edge'] as int? ?? size + 2,
      cropSize: size,
    ),
    'resnet' => PipelinePresets.resnetClassification(height: size, width: size),
    'detection' => PipelinePresets.objectDetection(height: size, width: size),
    'segmentation' => PipelinePresets.segmentation(height: size, width: size),
    'face' => PipelinePresets.faceRecognition(height: size, width: size),
    'mobilenet' => PipelinePresets.mobileNet(height: size, width: size),
    'clip' => PipelinePresets.clip(size: size),
    'vit' => PipelinePresets.vit(size: size),
    'tflite' || 'tflite_raw' => PipelinePresets.tflite(
      height: size,
      width: size,
      normalize: p['name'] != 'tflite_raw',
    ),
    'minimal' => PipelinePresets.minimal(height: size, width: size),
    'custom' || 'custom_hwc' || 'custom_unbatched' => PipelinePresets.custom(
      height: size,
      width: size,
      mean: [0.5, 0.5, 0.5],
      std: [0.5, 0.5, 0.5],
      toChw: p['name'] != 'custom_hwc',
      addBatchDim: p['name'] != 'custom_unbatched',
    ),
    _ => throw StateError('Unknown fixture preset: ${p['name']}'),
  };
}

void registerPytorchFixtures(String directory) {
  final manifest =
      jsonDecode(File('$directory/manifest.json').readAsStringSync())
          as Map<String, dynamic>;
  for (final entry in manifest['files'] as List) {
    final cases =
        jsonDecode(File('$directory/${entry['path']}').readAsStringSync())
            as List;
    test('fixture integrity: ${entry['path']}', () {
      expect(cases.length, entry['cases']);
      expect(
        sha256
            .convert(File('$directory/${entry['path']}').readAsBytesSync())
            .toString(),
        entry['sha256'],
      );
    });
    for (final raw in cases) {
      final c = raw as Map<String, dynamic>;
      test(c['name'] as String, () async {
        final base = fixtureTensor(
          (c['base'] ?? c['input']) as Map<String, dynamic>,
        );
        final input = c.containsKey('base')
            ? TensorBuffer(
                storage: base.storage,
                shape: ((c['input'] as Map)['shape'] as List).cast<int>(),
                storageOffset: c['offset'] as int,
              )
            : base;
        final atol = fixtureNumber(c['atol']);
        final rtol = fixtureNumber(c['rtol']);
        if (c['op'] == 'preset') {
          final pipeline = fixturePipeline(c['params'] as Map<String, dynamic>);
          expect(
            pipeline.computeOutputShape(input.shape),
            (c['expected'] as Map)['shape'],
          );
          for (final actual in [
            pipeline.run(input),
            await pipeline.runAsync(input, isolateThreshold: 0),
            await pipeline.runAsync(input, isolateThreshold: 1000000),
          ]) {
            expectFixture(
              actual,
              c['expected'] as Map<String, dynamic>,
              atol: atol,
              rtol: rtol,
            );
          }
          expectFixture(
            input,
            c['input'] as Map<String, dynamic>,
            atol: 0,
            rtol: 0,
          );
          return;
        }
        final op = fixtureOperation(c);
        final TensorBuffer result;
        if (c['inplace'] == true) {
          (op as InPlaceTransform).applyInPlace(input);
          result = input;
        } else {
          result = op(input);
        }
        expectFixture(
          result,
          c['expected'] as Map<String, dynamic>,
          atol: atol,
          rtol: rtol,
        );
        if (c.containsKey('expected_base')) {
          expectFixture(
            base,
            c['expected_base'] as Map<String, dynamic>,
            atol: atol,
            rtol: rtol,
          );
        }
      });
    }
  }
}
