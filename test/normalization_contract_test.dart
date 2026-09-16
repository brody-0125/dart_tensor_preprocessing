import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  final factories = <String, TransformOp Function(double)>{
    'batch': (eps) =>
        BatchNormOp(runningMean: [0, 0], runningVar: [1, 1], eps: eps),
    'instance': (eps) => InstanceNormOp(numFeatures: 2, eps: eps),
    'group': (eps) => GroupNormOp(numGroups: 1, numChannels: 2, eps: eps),
    'layer': (eps) => LayerNormOp(normalizedShape: [2, 3], eps: eps),
    'rms': (eps) => RMSNormOp(normalizedShape: [2, 3], eps: eps),
    'lp': (eps) => LpNormalizeOp(eps: eps),
  };
  for (final entry in factories.entries) {
    test('${entry.key} rejects invalid epsilon', () {
      for (final eps in [0.0, -1.0, double.nan, double.infinity]) {
        expect(
          () => entry.value(eps),
          throwsA(isA<InvalidParameterException>()),
        );
      }
    });
    test(
      '${entry.key} in-place rejects non-contiguous input without mutation',
      () {
        final base = TensorBuffer.ones([2, 3, 2]);
        final input = base.transpose([0, 2, 1]);
        final op = entry.value(1e-5) as InPlaceTransform;
        expect(
          () => op.applyInPlace(input),
          throwsA(isA<NonContiguousException>()),
        );
        expect(base.toList(), everyElement(1.0));
      },
    );
  }
  test('Lp rejects unsupported orders and invalid axes', () {
    for (final p in [0.0, -1.0, double.nan, double.negativeInfinity]) {
      expect(
        () => LpNormalizeOp(p: p),
        throwsA(isA<InvalidParameterException>()),
      );
    }
    for (final axis in [-3, 2]) {
      expect(
        () => LpNormalizeOp(dim: axis)(TensorBuffer.ones([2, 3])),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    }
  });
  test('normalizers reject incompatible shapes and affine lengths', () {
    final input = TensorBuffer.ones([3, 2, 3]);
    for (final op in [
      BatchNormOp(runningMean: [0, 0], runningVar: [1, 1]),
      InstanceNormOp(numFeatures: 2),
      GroupNormOp(numGroups: 1, numChannels: 2),
      LayerNormOp(normalizedShape: [4]),
      RMSNormOp(normalizedShape: [4]),
      NormalizeOp(mean: [0, 0], std: [1, 1]),
    ]) {
      expect(() => op(input), throwsA(isA<ShapeMismatchException>()));
    }
    for (final create in [
      () => BatchNormOp(runningMean: [0, 0], runningVar: [1, 1], weight: [1]),
      () => InstanceNormOp(numFeatures: 2, bias: [1]),
      () => GroupNormOp(numGroups: 1, numChannels: 2, weight: [1]),
      () => LayerNormOp(normalizedShape: [2], bias: [1]),
      () => RMSNormOp(normalizedShape: [2], weight: [1]),
    ]) {
      expect(create, throwsA(isA<InvalidParameterException>()));
    }
  });
}
