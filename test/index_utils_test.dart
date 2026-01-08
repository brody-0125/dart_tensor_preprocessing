import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/src/utils/index_utils.dart';

void main() {
  group('reflectIndex', () {
    test('returns index unchanged when in range', () {
      expect(reflectIndex(0, 5), equals(0));
      expect(reflectIndex(2, 5), equals(2));
      expect(reflectIndex(4, 5), equals(4));
    });

    test('reflects negative indices', () {
      // idx=-1 → -(-1)-1 = 0
      expect(reflectIndex(-1, 5), equals(0));
      // idx=-2 → -(-2)-1 = 1
      expect(reflectIndex(-2, 5), equals(1));
    });

    test('reflects indices beyond size', () {
      // idx=5, size=5 → 2*5-5-1 = 4
      expect(reflectIndex(5, 5), equals(4));
      // idx=6, size=5 → 2*5-6-1 = 3
      expect(reflectIndex(6, 5), equals(3));
    });

    test('works with size 1', () {
      expect(reflectIndex(0, 1), equals(0));
      expect(reflectIndex(-1, 1), equals(0));
      expect(reflectIndex(1, 1), equals(0));
    });
  });

  group('replicateIndex', () {
    test('returns index unchanged when in range', () {
      expect(replicateIndex(0, 5), equals(0));
      expect(replicateIndex(2, 5), equals(2));
      expect(replicateIndex(4, 5), equals(4));
    });

    test('clamps negative indices to 0', () {
      expect(replicateIndex(-1, 5), equals(0));
      expect(replicateIndex(-10, 5), equals(0));
    });

    test('clamps indices beyond size to size-1', () {
      expect(replicateIndex(5, 5), equals(4));
      expect(replicateIndex(10, 5), equals(4));
    });

    test('works with size 1', () {
      expect(replicateIndex(0, 1), equals(0));
      expect(replicateIndex(-1, 1), equals(0));
      expect(replicateIndex(1, 1), equals(0));
    });
  });

  group('circularIndex', () {
    test('returns index unchanged when in range', () {
      expect(circularIndex(0, 5), equals(0));
      expect(circularIndex(2, 5), equals(2));
      expect(circularIndex(4, 5), equals(4));
    });

    test('wraps negative indices', () {
      expect(circularIndex(-1, 5), equals(4));
      expect(circularIndex(-2, 5), equals(3));
      expect(circularIndex(-5, 5), equals(0));
      expect(circularIndex(-6, 5), equals(4));
    });

    test('wraps indices beyond size', () {
      expect(circularIndex(5, 5), equals(0));
      expect(circularIndex(6, 5), equals(1));
      expect(circularIndex(10, 5), equals(0));
    });

    test('works with size 1', () {
      expect(circularIndex(0, 1), equals(0));
      expect(circularIndex(-1, 1), equals(0));
      expect(circularIndex(1, 1), equals(0));
    });
  });
}
