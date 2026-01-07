// Exception hierarchy tests
//
// Tests for TensorException and its subclasses
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('TensorException hierarchy', () {
    group('IndexOutOfBoundsException', () {
      test('should contain index, min, max values', () {
        final exception = IndexOutOfBoundsException(
          index: 5,
          min: 0,
          max: 3,
        );

        expect(exception.index, equals(5));
        expect(exception.min, equals(0));
        expect(exception.max, equals(3));
        expect(exception, isA<TensorException>());
      });

      test('should include dimension name when provided', () {
        final exception = IndexOutOfBoundsException(
          index: 5,
          min: 0,
          max: 3,
          dimension: 'axis',
        );

        expect(exception.dimension, equals('axis'));
        expect(exception.message, contains('axis'));
      });

      test('should generate descriptive message', () {
        final exception = IndexOutOfBoundsException(
          index: 5,
          min: 0,
          max: 3,
        );

        expect(exception.message, contains('5'));
        expect(exception.message, contains('0'));
        expect(exception.message, contains('3'));
      });
    });

    group('DTypeMismatchException', () {
      test('should contain expected and actual dtypes', () {
        final exception = DTypeMismatchException(
          expected: DType.float32,
          actual: DType.int32,
        );

        expect(exception.expected, equals(DType.float32));
        expect(exception.actual, equals(DType.int32));
        expect(exception, isA<TensorException>());
      });

      test('should generate descriptive message', () {
        final exception = DTypeMismatchException(
          expected: DType.float32,
          actual: DType.int32,
        );

        expect(exception.message, contains('float32'));
        expect(exception.message, contains('int32'));
      });
    });
  });
}
