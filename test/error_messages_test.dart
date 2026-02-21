import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/exceptions/error_messages.dart';

void main() {
  group('ErrorMessages', () {
    test('rankMismatch returns message with operation and ranks', () {
      final msg = ErrorMessages.rankMismatch('PadOp', 3, 4);
      expect(msg, contains('PadOp'));
      expect(msg, contains('3'));
      expect(msg, contains('4'));
    });

    test('rankRange returns message with min/max range', () {
      final msg = ErrorMessages.rankRange('ResizeOp', 3, 4, 2);
      expect(msg, contains('ResizeOp'));
      expect(msg, contains('3-4'));
      expect(msg, contains('2'));
    });

    test('channelMismatch returns message with channel counts', () {
      final msg = ErrorMessages.channelMismatch('NormalizeOp', 3, 1);
      expect(msg, contains('NormalizeOp'));
      expect(msg, contains('3'));
      expect(msg, contains('1'));
    });

    test('shapeMismatch returns message with shapes', () {
      final msg = ErrorMessages.shapeMismatch('ConcatOp', [2, 3], [2, 4]);
      expect(msg, contains('ConcatOp'));
      expect(msg, contains('[2, 3]'));
      expect(msg, contains('[2, 4]'));
    });

    test('axisBounds returns message with axis and rank', () {
      final msg = ErrorMessages.axisBounds('SoftmaxOp', 5, 4);
      expect(msg, contains('SoftmaxOp'));
      expect(msg, contains('5'));
      expect(msg, contains('4'));
    });

    test('dtypeMismatch returns message with dtype names', () {
      final msg = ErrorMessages.dtypeMismatch('TypeCastOp', 'float32', 'int64');
      expect(msg, contains('TypeCastOp'));
      expect(msg, contains('float32'));
      expect(msg, contains('int64'));
    });

    test('parameterRange returns message with parameter info', () {
      final msg =
          ErrorMessages.parameterRange('ClipOp', 'min', -1.0, 0.0, 1.0);
      expect(msg, contains('ClipOp'));
      expect(msg, contains('min'));
      expect(msg, contains('-1.0'));
      expect(msg, contains('[0.0, 1.0]'));
    });

    test('broadcastIncompatible returns message with shapes', () {
      final msg =
          ErrorMessages.broadcastIncompatible('AddOp', [2, 3], [4, 5]);
      expect(msg, contains('AddOp'));
      expect(msg, contains('[2, 3]'));
      expect(msg, contains('[4, 5]'));
    });
  });
}
