import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test(
    'color execution and shape inference reject invalid rank and channels',
    () {
      for (final shape in [
        [2, 3],
        [1, 2, 3],
        [4, 2, 3],
        [2, 4, 2, 3],
        [1, 3, 2, 3, 1],
      ]) {
        for (final op in <TransformOp>[
          RgbToGrayscaleOp(),
          RgbToHsvOp(),
          HsvToRgbOp(),
        ]) {
          expect(
            () => op(TensorBuffer.zeros(shape)),
            throwsA(isA<ShapeMismatchException>()),
          );
          expect(
            () => op.computeOutputShape(shape),
            throwsA(isA<ShapeMismatchException>()),
          );
        }
      }
    },
  );
}
