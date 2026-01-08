import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:dart_tensor_preprocessing/src/utils/validation_utils.dart';

void main() {
  group('TensorValidation extension', () {
    group('requireRank3Or4', () {
      test('passes for 3D tensor', () {
        final tensor = TensorBuffer.zeros([3, 4, 5]);
        expect(() => tensor.requireRank3Or4('TestOp'), returnsNormally);
      });

      test('passes for 4D tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
        expect(() => tensor.requireRank3Or4('TestOp'), returnsNormally);
      });

      test('throws for 1D tensor', () {
        final tensor = TensorBuffer.zeros([10]);
        expect(
          () => tensor.requireRank3Or4('TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws for 2D tensor', () {
        final tensor = TensorBuffer.zeros([3, 4]);
        expect(
          () => tensor.requireRank3Or4('TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws for 5D tensor', () {
        final tensor = TensorBuffer.zeros([1, 2, 3, 4, 5]);
        expect(
          () => tensor.requireRank3Or4('TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('exception message contains operation name', () {
        final tensor = TensorBuffer.zeros([3, 4]);
        expect(
          () => tensor.requireRank3Or4('MyCustomOp'),
          throwsA(
            isA<ShapeMismatchException>().having(
              (e) => e.message,
              'message',
              contains('MyCustomOp'),
            ),
          ),
        );
      });
    });

    group('requireExactRank', () {
      test('passes for matching rank', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        expect(() => tensor.requireExactRank(3, 'TestOp'), returnsNormally);
      });

      test('throws for non-matching rank', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        expect(
          () => tensor.requireExactRank(4, 'TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('works with 1D tensor', () {
        final tensor = TensorBuffer.zeros([10]);
        expect(() => tensor.requireExactRank(1, 'TestOp'), returnsNormally);
        expect(
          () => tensor.requireExactRank(2, 'TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('exception message contains expected rank', () {
        final tensor = TensorBuffer.zeros([2, 3]);
        expect(
          () => tensor.requireExactRank(4, 'TestOp'),
          throwsA(
            isA<ShapeMismatchException>().having(
              (e) => e.message,
              'message',
              contains('4D'),
            ),
          ),
        );
      });
    });

    group('requireMinRank', () {
      test('passes when rank equals minimum', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        expect(() => tensor.requireMinRank(3, 'TestOp'), returnsNormally);
      });

      test('passes when rank exceeds minimum', () {
        final tensor = TensorBuffer.zeros([1, 2, 3, 4]);
        expect(() => tensor.requireMinRank(2, 'TestOp'), returnsNormally);
      });

      test('throws when rank is below minimum', () {
        final tensor = TensorBuffer.zeros([2, 3]);
        expect(
          () => tensor.requireMinRank(3, 'TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('exception message contains minimum rank', () {
        final tensor = TensorBuffer.zeros([5]);
        expect(
          () => tensor.requireMinRank(2, 'TestOp'),
          throwsA(
            isA<ShapeMismatchException>().having(
              (e) => e.message,
              'message',
              contains('at least 2D'),
            ),
          ),
        );
      });
    });
  });

  group('requirePositive', () {
    test('passes for positive integer', () {
      expect(() => requirePositive(1, 'value'), returnsNormally);
      expect(() => requirePositive(100, 'value'), returnsNormally);
    });

    test('passes for positive double', () {
      expect(() => requirePositive(0.1, 'value'), returnsNormally);
      expect(() => requirePositive(1.5, 'value'), returnsNormally);
    });

    test('throws for zero', () {
      expect(
        () => requirePositive(0, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => requirePositive(0.0, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for negative value', () {
      expect(
        () => requirePositive(-1, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => requirePositive(-0.5, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('exception contains parameter name', () {
      expect(
        () => requirePositive(-1, 'kernelSize'),
        throwsA(
          isA<InvalidParameterException>().having(
            (e) => e.parameterName,
            'parameterName',
            equals('kernelSize'),
          ),
        ),
      );
    });
  });

  group('requireNonNegative', () {
    test('passes for positive value', () {
      expect(() => requireNonNegative(1, 'value'), returnsNormally);
      expect(() => requireNonNegative(0.5, 'value'), returnsNormally);
    });

    test('passes for zero', () {
      expect(() => requireNonNegative(0, 'value'), returnsNormally);
      expect(() => requireNonNegative(0.0, 'value'), returnsNormally);
    });

    test('throws for negative integer', () {
      expect(
        () => requireNonNegative(-1, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('throws for negative double', () {
      expect(
        () => requireNonNegative(-0.001, 'value'),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('exception contains parameter name', () {
      expect(
        () => requireNonNegative(-5, 'padding'),
        throwsA(
          isA<InvalidParameterException>().having(
            (e) => e.parameterName,
            'parameterName',
            equals('padding'),
          ),
        ),
      );
    });
  });
}
