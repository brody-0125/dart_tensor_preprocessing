import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

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
  List<double>? numbers(String key) =>
      (p[key] as List?)?.map(fixtureNumber).toList();
  return switch (c['op']) {
    'masked_fill' => MaskedFillOp(
      mask: fixtureTensor(p['mask'] as Map<String, dynamic>),
      value: fixtureNumber(p['value']),
    ),
    'tile' => TileOp(reps: (p['reps'] as List).cast<int>()),
    'repeat' => RepeatOp(sizes: (p['reps'] as List).cast<int>()),
    'roll' => RollOp(
      shifts: (p['shifts'] as List).cast<int>(),
      dims: (p['dims'] as List?)?.cast<int>(),
    ),
    'slice' => SliceOp(
      (p['slices'] as List)
          .map(
            (s) =>
                s == null ? null : (s[0] as int?, s[1] as int?, s[2] as int?),
          )
          .toList(),
    ),
    'gather' => GatherOp(
      dim: p['dim'] as int,
      index: fixtureTensor(p['index'] as Map<String, dynamic>),
    ),
    'where' => WhereOp(
      condition: fixtureTensor(p['mask'] as Map<String, dynamic>),
      y: fixtureTensor(p['other'] as Map<String, dynamic>),
    ),
    'neg' => NegOp(),
    'sqrt' => SqrtOp(),
    'exp' => ExpOp(),
    'log' => LogOp(),
    'floor' => FloorOp(),
    'ceil' => CeilOp(),
    'round' => RoundOp(),
    'sin' => SinOp(),
    'cos' => CosOp(),
    'tan' => TanOp(),
    'asin' => AsinOp(),
    'acos' => AcosOp(),
    'atan' => AtanOp(),
    'gelu' => GELUOp(approximate: p['approximate'] as String),
    'glu' => GLUOp(dim: p['dim'] as int),
    'batch_norm' => BatchNormOp(
      runningMean: numbers('mean')!,
      runningVar: numbers('variance')!,
      weight: numbers('weight'),
      bias: numbers('bias'),
      eps: fixtureNumber(p['eps']),
    ),
    'instance_norm' => InstanceNormOp(
      numFeatures: p['channels'] as int,
      weight: numbers('weight'),
      bias: numbers('bias'),
      eps: fixtureNumber(p['eps']),
    ),
    'group_norm' => GroupNormOp(
      numChannels: p['channels'] as int,
      numGroups: p['groups'] as int,
      weight: numbers('weight'),
      bias: numbers('bias'),
      eps: fixtureNumber(p['eps']),
    ),
    'layer_norm' => LayerNormOp(
      normalizedShape: (p['shape'] as List).cast<int>(),
      weight: numbers('weight'),
      bias: numbers('bias'),
      eps: fixtureNumber(p['eps']),
    ),
    'rms_norm' => RMSNormOp(
      normalizedShape: (p['shape'] as List).cast<int>(),
      weight: numbers('weight'),
      eps: fixtureNumber(p['eps']),
    ),
    'normalize' => NormalizeOp(mean: numbers('mean')!, std: numbers('std')!),
    'lp_normalize' => LpNormalizeOp(
      p: fixtureNumber(p['p']),
      dim: p['dim'] as int,
      eps: fixtureNumber(p['eps']),
    ),
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

TensorBuffer fixtureCoreOperation(Map<String, dynamic> c, TensorBuffer input) {
  final p = c['params'] as Map<String, dynamic>;
  switch (c['op']) {
    case 'core_reduce':
      final keep = p['keep'] as bool? ?? false;
      final axes = (p['axes'] as List?)?.cast<int>();
      if (axes != null) {
        return switch (p['reduction']) {
          'sum' => input.sumAxes(axes, keepDims: keep),
          'mean' => input.meanAxes(axes, keepDims: keep),
          'min' => input.minAxes(axes, keepDims: keep),
          'max' => input.maxAxes(axes, keepDims: keep),
          _ => throw StateError('Unknown multi-axis reduction'),
        };
      }
      final axis = p['axis'] as int?;
      if (axis != null) {
        return switch (p['reduction']) {
          'sum' => input.sumAxis(axis, keepDims: keep),
          'mean' => input.meanAxis(axis, keepDims: keep),
          'min' => input.minAxis(axis, keepDims: keep),
          'max' => input.maxAxis(axis, keepDims: keep),
          'argmin' => input.argminAxis(axis, keepDims: keep),
          'argmax' => input.argmaxAxis(axis, keepDims: keep),
          _ => throw StateError('Unknown axis reduction'),
        };
      }
      final num value = switch (p['reduction']) {
        'sum' => input.sum(),
        'mean' => input.mean(),
        'min' => input.min(),
        'max' => input.max(),
        'argmin' => input.argmin(),
        'argmax' => input.argmax(),
        _ => throw StateError('Unknown scalar reduction'),
      };
      return TensorBuffer(
        storage: value is int
            ? TensorStorage(Int64List.fromList([value]), DType.int64)
            : TensorStorage(
                Float64List.fromList([value.toDouble()]),
                DType.float64,
              ),
        shape: [1],
      );
    case 'core_clone':
      return input.clone();
    case 'core_contiguous':
      return input.contiguous();
    case 'core_transpose':
      return input.transpose((p['axes'] as List).cast<int>());
    case 'core_reshape':
      return input.contiguous().reshape((p['shape'] as List).cast<int>());
    case 'core_stack':
      return stack([
        input,
        fixtureTensor(p['other'] as Map<String, dynamic>),
      ], dim: p['axis'] as int);
    case 'core_concat':
      return concat([
        input,
        fixtureTensor(p['other'] as Map<String, dynamic>),
      ], axis: p['axis'] as int);
    case 'core_split':
    case 'core_chunk':
      final parts = c['op'] == 'core_split'
          ? split(input, (p['sizes'] as List).cast<int>(), dim: p['dim'] as int)
          : chunk(input, p['chunks'] as int, dim: p['dim'] as int);
      expect(parts.length, p['count']);
      return parts[p['part'] as int];
    case 'core_topk':
      final op = TopKOp(
        k: p['k'] as int,
        axis: p['axis'] as int,
        largest: p['largest'] as bool,
      );
      final result = op.applyTopK(input);
      expect(op.computeOutputShape(input.shape), result.$1.shape);
      if (p['indices'] == false) {
        expectFixture(
          GatherOp(dim: p['axis'] as int, index: result.$2)(input),
          c['expected'] as Map<String, dynamic>,
          atol: fixtureNumber(c['atol']),
          rtol: fixtureNumber(c['rtol']),
        );
      }
      return p['indices'] == true ? result.$2 : result.$1;
    default:
      throw StateError('Unknown core fixture: ${c['op']}');
  }
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
      for (final source in {
        'scripts/generate_pytorch_fixtures.py': 'generator_sha256',
        'scripts/requirements-fixtures.txt': 'requirements_sha256',
        'scripts/requirements-fixtures-linux.txt': 'requirements_linux_sha256',
      }.entries) {
        final normalized = File(
          source.key,
        ).readAsStringSync().replaceAll('\r\n', '\n');
        expect(
          sha256.convert(utf8.encode(normalized)).toString(),
          manifest[source.value],
          reason: source.key,
        );
      }
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
                strides: (c['strides'] as List?)?.cast<int>(),
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
        if ((c['op'] as String).startsWith('core_')) {
          expectFixture(
            fixtureCoreOperation(c, input),
            c['expected'] as Map<String, dynamic>,
            atol: atol,
            rtol: rtol,
          );
          expectFixture(
            base,
            (c['base'] ?? c['input']) as Map<String, dynamic>,
            atol: 0,
            rtol: 0,
          );
          return;
        }
        final op = fixtureOperation(c);
        expect(
          op.computeOutputShape(input.shape),
          (c['expected'] as Map)['shape'],
        );
        final TensorBuffer result;
        if (c['inplace'] == true) {
          (op as InPlaceTransform).applyInPlace(input);
          result = input;
        } else {
          result = op(input);
          expectFixture(
            base,
            (c['base'] ?? c['input']) as Map<String, dynamic>,
            atol: 0,
            rtol: 0,
          );
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
