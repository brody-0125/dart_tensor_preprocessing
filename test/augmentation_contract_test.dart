import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
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
