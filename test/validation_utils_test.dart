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

  group('OpValidator', () {
    group('validateImageTensor', () {
      test('passes for 3D tensor', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        expect(
          () =>
              OpValidator.validateImageTensor(tensor, operationName: 'TestOp'),
          returnsNormally,
        );
      });

      test('passes for 4D tensor', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        expect(
          () =>
              OpValidator.validateImageTensor(tensor, operationName: 'TestOp'),
          returnsNormally,
        );
      });

      test('throws for 2D tensor', () {
        final tensor = TensorBuffer.zeros([224, 224]);
        expect(
          () =>
              OpValidator.validateImageTensor(tensor, operationName: 'TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('throws for 5D tensor', () {
        final tensor = TensorBuffer.zeros([1, 2, 3, 4, 5]);
        expect(
          () =>
              OpValidator.validateImageTensor(tensor, operationName: 'TestOp'),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('validates expected channels for 3D tensor', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        expect(
          () => OpValidator.validateImageTensor(
            tensor,
            operationName: 'TestOp',
            expectedChannels: 3,
          ),
          returnsNormally,
        );
      });

      test('throws on channel mismatch for 3D tensor', () {
        final tensor = TensorBuffer.zeros([1, 224, 224]);
        expect(
          () => OpValidator.validateImageTensor(
            tensor,
            operationName: 'TestOp',
            expectedChannels: 3,
          ),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('validates expected channels for 4D tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 224, 224]);
        expect(
          () => OpValidator.validateImageTensor(
            tensor,
            operationName: 'TestOp',
            expectedChannels: 3,
          ),
          returnsNormally,
        );
      });

      test('throws on channel mismatch for 4D tensor', () {
        final tensor = TensorBuffer.zeros([2, 1, 224, 224]);
        expect(
          () => OpValidator.validateImageTensor(
            tensor,
            operationName: 'TestOp',
            expectedChannels: 3,
          ),
          throwsA(isA<ShapeMismatchException>()),
        );
      });
    });

    group('validateBroadcast', () {
      test('returns same shape for identical shapes', () {
        final result = OpValidator.validateBroadcast(
          [2, 3],
          [2, 3],
          operationName: 'TestOp',
        );
        expect(result, equals([2, 3]));
      });

      test('broadcasts scalar-like dimension', () {
        final result = OpValidator.validateBroadcast(
          [2, 1],
          [2, 3],
          operationName: 'TestOp',
        );
        expect(result, equals([2, 3]));
      });

      test('broadcasts with different ranks', () {
        final result = OpValidator.validateBroadcast(
          [3],
          [2, 3],
          operationName: 'TestOp',
        );
        expect(result, equals([2, 3]));
      });

      test('throws for incompatible shapes', () {
        expect(
          () => OpValidator.validateBroadcast(
            [2, 3],
            [2, 4],
            operationName: 'TestOp',
          ),
          throwsA(isA<ShapeMismatchException>()),
        );
      });

      test('handles empty shapes', () {
        final result = OpValidator.validateBroadcast(
          [],
          [2, 3],
          operationName: 'TestOp',
        );
        expect(result, equals([2, 3]));
      });
    });

    group('validateFloatDType', () {
      test('passes for float32', () {
        expect(
          () => OpValidator.validateFloatDType(
            DType.float32,
            operationName: 'TestOp',
          ),
          returnsNormally,
        );
      });

      test('passes for float64', () {
        expect(
          () => OpValidator.validateFloatDType(
            DType.float64,
            operationName: 'TestOp',
          ),
          returnsNormally,
        );
      });

      test('throws for int32', () {
        expect(
          () => OpValidator.validateFloatDType(
            DType.int32,
            operationName: 'TestOp',
          ),
          throwsA(isA<DTypeMismatchException>()),
        );
      });

      test('throws for uint8', () {
        expect(
          () => OpValidator.validateFloatDType(
            DType.uint8,
            operationName: 'TestOp',
          ),
          throwsA(isA<DTypeMismatchException>()),
        );
      });
    });
  });
}
