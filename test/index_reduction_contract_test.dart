import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test('gather rejects fractional indices and oversized non-gather axes', () {
    final input = TensorBuffer.ones([2, 3]);
    expect(
      () => GatherOp(dim: 1, index: TensorBuffer.full([2, 2], fillValue: 0.5))(
        input,
      ),
      throwsA(isA<InvalidParameterException>()),
    );
    expect(
      () => GatherOp(
        dim: 1,
        index: TensorBuffer.zeros([3, 1], dtype: DType.int64),
      )(input),
      throwsA(isA<ShapeMismatchException>()),
    );
  });
  test('roll requires shifts and a single flat shift', () {
    expect(() => RollOp(shifts: []), throwsA(isA<InvalidParameterException>()));
    expect(
      () => RollOp(shifts: [1, 2]),
      throwsA(isA<InvalidParameterException>()),
    );
  });
  test('tile and repeat reject zero, negative, or mismatched repetitions', () {
    for (final reps in [
      [0, 1],
      [-1, 2],
      [2],
    ]) {
      for (final op in [TileOp(reps: reps), RepeatOp(sizes: reps)]) {
        expect(
          () => op(TensorBuffer.ones([2, 3])),
          throwsA(isA<InvalidParameterException>()),
        );
        expect(
          () => op.computeOutputShape([2, 3]),
          throwsA(isA<InvalidParameterException>()),
        );
      }
    }
  });
  test(
    'axis mean rejects integer input instead of silently rounding to float32',
    () {
      final input = TensorBuffer.ones([2, 3], dtype: DType.int32);
      expect(
        () => input.meanAxis(-1),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        () => input.meanAxes([0, 1]),
        throwsA(isA<InvalidParameterException>()),
      );
      expect(
        input.mean(),
        1.0,
      ); // The scalar-valued API explicitly returns double.
    },
  );
  test('all reducers reject invalid axes without mutating input', () {
    final input = TensorBuffer.ones([2, 3]);
    for (final axis in [-3, 2]) {
      for (final call in [
        input.sumAxis,
        input.meanAxis,
        input.minAxis,
        input.maxAxis,
        input.argminAxis,
        input.argmaxAxis,
      ]) {
        expect(() => call(axis), throwsA(isA<IndexOutOfBoundsException>()));
      }
    }
    for (final call in [
      input.sumAxes,
      input.meanAxes,
      input.minAxes,
      input.maxAxes,
    ]) {
      expect(() => call([0, -2]), throwsA(isA<InvalidParameterException>()));
    }
    expect(input.toList(), everyElement(1.0));
  });
  test(
    'top-k tied indices are distinct and select their reported exact values',
    () {
      final input = TensorBuffer(
        storage: TensorStorage(
          Int64List.fromList([
            9007199254740993,
            9007199254740993,
            9007199254740992,
          ]),
          DType.int64,
        ),
        shape: [3],
      );
      for (final sorted in [false, true]) {
        final (values, indices) = TopKOp(k: 2, sorted: sorted).applyTopK(input);
        final selected = indices.storage.data as List<int>;
        expect(selected.toSet().length, 2);
        expect(values.storage.data as List<int>, [
          9007199254740993,
          9007199254740993,
        ]);
        for (var i = 0; i < 2; i++) {
          expect(
            (input.storage.data as List<int>)[selected[i]],
            (values.storage.data as List<int>)[i],
          );
        }
      }
    },
  );
}
