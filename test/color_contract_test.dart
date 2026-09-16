import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('color adjustment factors must be finite', () {
    for (final value in [
      double.nan,
      double.infinity,
      double.negativeInfinity,
    ]) {
      for (final create in <TransformOp Function()>[
        () => AdjustBrightnessOp(factor: value),
        () => AdjustContrastOp(factor: value),
        () => AdjustSaturationOp(factor: value),
        () => AdjustHueOp(factor: value),
        () => ColorJitterOp(brightness: value),
        () => ColorJitterOp(contrast: value),
        () => ColorJitterOp(saturation: value),
        () => ColorJitterOp(hue: value),
      ]) {
        expect(create, throwsA(isA<InvalidParameterException>()));
      }
    }
  });

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
          AdjustBrightnessOp(factor: 0.1),
          AdjustContrastOp(factor: 1.2),
          AdjustSaturationOp(factor: 1.2),
          AdjustHueOp(factor: 0.1),
          ColorJitterOp(),
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
