import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
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
