import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

import 'pytorch_fixture_support.dart';

void main() {
  test('golden comparator detects a wrong value and unexpected NaN', () {
    final expected = <String, dynamic>{
      'shape': [2],
      'dtype': 'float32',
      'values': [0.0, 1.0],
    };
    for (final value in [2.0, double.nan]) {
      final actual = TensorBuffer.full([2], fillValue: value);
      expect(
        () => expectFixture(actual, expected, atol: 1e-6, rtol: 1e-5),
        throwsA(isA<TestFailure>()),
      );
    }
  });
}
