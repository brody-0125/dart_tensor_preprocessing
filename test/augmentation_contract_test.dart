import 'dart:typed_data';
import 'dart:math' as math;
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('erasing validates finite parameters and inferred rank', () {
    for (final bad in [double.nan, double.infinity, double.negativeInfinity]) {
      for (final create in [
        () => RandomErasingOp(probability: bad),
        () => RandomErasingOp(scaleRange: (bad, 1)),
        () => RandomErasingOp(scaleRange: (0.1, bad)),
        () => RandomErasingOp(ratioRange: (bad, 1)),
        () => RandomErasingOp(ratioRange: (0.1, bad)),
        () => RandomErasingOp(value: bad),
      ]) {
        expect(create, throwsA(isA<InvalidParameterException>()));
      }
    }
    expect(
      () => RandomErasingOp().computeOutputShape([2, 3]),
      throwsA(isA<ShapeMismatchException>()),
    );
  });
  test('random erase fill follows native uniform stream per batch', () {
    final random = math.Random(41);
    final expected = <double>[];
    for (var batch = 0; batch < 2; batch++) {
      random.nextDouble(); // probability
      random.nextDouble(); // area
      random.nextDouble(); // ratio
      random.nextInt(1); // top
      random.nextInt(1); // left
      expected.addAll(List.generate(4, (_) => random.nextDouble()));
    }
    final result = RandomErasingOp(
      probability: 1,
      scaleRange: (1, 1),
      ratioRange: (1, 1),
      value: null,
      seed: 41,
    )(TensorBuffer.ones([2, 1, 2, 2], dtype: DType.float64));
    expect(result.storage.data, expected);
    final integers = RandomErasingOp(
      probability: 1,
      scaleRange: (1, 1),
      ratioRange: (1, 1),
      value: null,
      seed: 41,
    )(TensorBuffer.ones([2, 1, 2, 2], dtype: DType.int64));
    expect(integers.storage.data, List.filled(8, 0));
  });

  test('positional encoding validates base and inferred shape', () {
    for (final base in [0.0, -1.0, double.nan, double.infinity]) {
      expect(
        () => PositionalEncodingOp(dModel: 3, maxLen: 4, base: base),
        throwsA(isA<InvalidParameterException>()),
      );
    }
    final op = PositionalEncodingOp(dModel: 3, maxLen: 4);
    for (final shape in [
      [3],
      [2, 4],
    ]) {
      expect(
        () => op.computeOutputShape(shape),
        throwsA(isA<ShapeMismatchException>()),
      );
    }
    expect(
      () => op.computeOutputShape([5, 3]),
      throwsA(isA<InvalidParameterException>()),
    );
  });

  test('padding shape inference rejects unsupported ranks', () {
    for (final mode in PadMode.values) {
      final op = PadOp.all(2, mode: mode);
      expect(
        () => op.computeOutputShape([2, 3]),
        throwsA(isA<ShapeMismatchException>()),
      );
      expect(
        () => op(TensorBuffer.ones([2, 3])),
        throwsA(isA<ShapeMismatchException>()),
      );
    }
  });

  test(
    'blur validates sigma and shape and preserves exact identity kernels',
    () {
      for (final sigma in [double.nan, double.infinity, 0.0, -1.0]) {
        expect(
          () => GaussianBlurOp(sigma: sigma),
          throwsA(isA<InvalidParameterException>()),
        );
      }
      expect(
        () => GaussianBlurOp().computeOutputShape([2, 3]),
        throwsA(isA<ShapeMismatchException>()),
      );
      final x = TensorBuffer(
        storage: TensorStorage(
          Int64List.fromList([9007199254740993]),
          DType.int64,
        ),
        shape: [1, 1, 1],
      );
      final y = GaussianBlurOp(kernelSize: 1)(x);
      expect(y.storage.data, x.storage.data);
      expect(identical(y.storage, x.storage), isFalse);
    },
  );

  test('random crop rejects invalid rank and oversized crops in inference', () {
    final op = RandomCropOp(height: 4, width: 5, seed: 41);
    expect(
      () => op.computeOutputShape([2, 3]),
      throwsA(isA<ShapeMismatchException>()),
    );
    expect(
      () => op(TensorBuffer.ones([2, 3])),
      throwsA(isA<ShapeMismatchException>()),
    );
    expect(
      () => op.computeOutputShape([2, 3, 5]),
      throwsA(isA<InvalidParameterException>()),
    );
    expect(
      () => op(TensorBuffer.ones([2, 3, 5])),
      throwsA(isA<InvalidParameterException>()),
    );
  });

  test('flips reject invalid shape and non-finite probabilities', () {
    for (final p in [double.nan, double.infinity, -0.1, 1.1]) {
      expect(
        () => RandomHorizontalFlipOp(probability: p),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => RandomVerticalFlipOp(probability: p),
        throwsA(isA<InvalidParameterException>()),
      );
    }
    for (final op in <TransformOp>[
      HorizontalFlipOp(),
      VerticalFlipOp(),
      RandomHorizontalFlipOp(),
      RandomVerticalFlipOp(),
    ]) {
      expect(
        () => op(TensorBuffer.ones([2, 3])),
        throwsA(isA<ShapeMismatchException>()),
      );
      expect(
        () => op.computeOutputShape([2, 3]),
        throwsA(isA<ShapeMismatchException>()),
      );
    }
  });
}
