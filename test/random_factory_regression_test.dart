import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  for (final dtype in [DType.float32, DType.float64]) {
    test('$dtype uniform excludes one even after storage rounding', () {
      // The first LCG state for this seed is 2^31 - 1.
      final x = TensorBuffer.random([7], dtype: dtype, seed: 230538014);
      expect(x.dtype, dtype);
      expect(x.toList(), everyElement(inExclusiveRange(-1e-300, 1.0)));
      expect(
        x.toList(),
        TensorBuffer.random([7], dtype: dtype, seed: 230538014).toList(),
      );
    });
    test('$dtype normal retains the tail and handles an odd length', () {
      // Python 3.12 math: u1=1/2**31, u2=1103527590/2**31;
      // r=sqrt(-2*log(u1)); (r*cos(2*pi*u2), r*sin(2*pi*u2)).
      // Seeds are package-specific, not PyTorch RNG seeds.
      final x = TensorBuffer.randn([3], dtype: dtype, seed: 1798410728);
      expect(x.dtype, dtype);
      final tolerance = dtype == DType.float32 ? 1e-6 : 1e-12;
      expect(x[[0]], closeTo(-6.530663232402052, tolerance));
      expect(x[[1]], closeTo(-0.5705812296847789, tolerance));
      expect(
        x.toList(),
        TensorBuffer.randn([3], dtype: dtype, seed: 1798410728).toList(),
      );
    });
    test('$dtype normal skips a zero uniform draw', () {
      final x = TensorBuffer.randn([3], dtype: dtype, seed: 2088216195);
      expect(x.toList().every((value) => value.isFinite), isTrue);
    });
  }
  for (final dtype in DType.values.where((d) => d.isInteger)) {
    test('random factories reject $dtype explicitly', () {
      expect(
        () => TensorBuffer.random([1], dtype: dtype),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => TensorBuffer.randn([1], dtype: dtype),
        throwsA(isA<InvalidParameterException>()),
      );
    });
  }
}
