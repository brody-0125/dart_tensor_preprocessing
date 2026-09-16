import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  test(
    'logical conversion is independent of physical channels-last strides',
    () {
      final raw = Float64List.fromList(List.generate(24, (i) => i.toDouble()));
      final x = TensorBuffer(
        storage: TensorStorage(raw, DType.float64),
        shape: [1, 2, 3, 4],
        memoryFormat: MemoryFormat.channelsLast,
      );
      final y = LayoutConvertOp.toNhwc(forceContiguous: false)(x);
      expect(y.shape, [1, 3, 4, 2]);
      expect(y.isContiguous, isTrue);
      expect(y.toList(), raw);
      expect(y.sliceFirst(0, 1).strides, y.strides);
      expect(y.sliceFirst(0, 1).toList(), raw);
      final restored = LayoutConvertOp.toNchw()(y);
      expect(restored.shape, x.shape);
      expect(restored.toList(), [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
      ]);
    },
  );

  test('vector selection and unbind retain exact aliased single elements', () {
    final raw = Int64List.fromList([
      -7,
      9007199254740993,
      9007199254740995,
      -9,
    ]);
    final x = TensorBuffer(
      storage: TensorStorage(raw, DType.int64),
      shape: [2],
      storageOffset: 1,
    );
    final parts = x.unbind(0);
    expect(parts.length, 2);
    for (var i = 0; i < parts.length; i++) {
      expect(parts[i].shape, [1]);
      expect(identical(parts[i].storage, x.storage), isTrue);
      expect(parts[i].clone().storage.data, [raw[i + 1]]);
      expect(x.select(0, i).clone().storage.data, [raw[i + 1]]);
    }
    raw[2] = 11;
    expect(parts[1].clone().storage.data, [11]);
  });

  test('narrow rejects empty negative and overflowing ranges', () {
    final x = TensorBuffer.ones([3]);
    for (final range in [
      [0, 0],
      [0, -1],
      [2, 2],
      [9223372036854775807, 2],
    ]) {
      expect(() => x.narrow(0, range[0], range[1]), throwsRangeError);
    }
    expect(x.narrow(0, 2, 1).toList(), [1]);
  });

  test('shape and strides cannot invalidate cached contiguity', () {
    final shape = [2, 3];
    final strides = [3, 1];
    final x = TensorBuffer(
      storage: TensorStorage(Float32List(6), DType.float32),
      shape: shape,
      strides: strides,
    );
    expect(x.isContiguous, isTrue);
    shape[0] = 99;
    strides[0] = 99;
    expect(x.shape, [2, 3]);
    expect(x.strides, [3, 1]);
    for (final view in [
      x,
      x.transpose([1, 0]),
      x.reshape([6]),
      x.unsqueeze(0),
      x.squeeze(),
    ]) {
      expect(() => view.shape[0] = 99, throwsUnsupportedError);
      expect(() => view.strides[0] = 99, throwsUnsupportedError);
    }
  });

  test('constructor rejects invalid storage spans and strides', () {
    final storage = TensorStorage(Float32List(6), DType.float32);
    for (final create in <TensorBuffer Function()>[
      () => TensorBuffer(storage: storage, shape: []),
      () => TensorBuffer(storage: storage, shape: [0]),
      () => TensorBuffer(storage: storage, shape: [7]),
      () => TensorBuffer(storage: storage, shape: [2, 3], strides: [1]),
      () => TensorBuffer(storage: storage, shape: [2], strides: [-1]),
      () => TensorBuffer(storage: storage, shape: [2], strides: [6]),
      () => TensorBuffer(storage: storage, shape: [2], storageOffset: -1),
      () => TensorBuffer(storage: storage, shape: [2], storageOffset: 5),
      () => TensorBuffer(
        storage: storage,
        shape: [2],
        strides: [9223372036854775807],
      ),
    ]) {
      expect(create, throwsA(isA<InvalidParameterException>()));
    }
    final broadcast = TensorBuffer(
      storage: storage,
      shape: [3],
      strides: [0],
      storageOffset: 5,
    );
    expect(broadcast.toList(), [0, 0, 0]);
  });

  test('single-element squeeze remains usable by clone and reshape', () {
    final x = TensorBuffer.full([1, 1], fillValue: 7);
    final y = SqueezeOp()(x);
    expect(y.shape, [1]);
    expect(y.clone().reshape([1, 1]).toList(), [7]);
    expect(identical(y.storage, x.storage), isTrue);
  });

  test(
    'invalid axes and reshape contracts agree before and during execution',
    () {
      final x = TensorBuffer.ones([2, 3]);
      for (final op in <TransformOp>[
        SqueezeOp(-3),
        SqueezeOp(2),
        UnsqueezeOp(-4),
        UnsqueezeOp(3),
        ReshapeOp([5]),
      ]) {
        expect(() => op(x), throwsA(isA<TensorException>()));
        expect(
          () => op.computeOutputShape(x.shape),
          throwsA(isA<TensorException>()),
        );
      }
      for (final shape in <List<int>>[
        [],
        [-2, -3],
        [0, 6],
      ]) {
        expect(
          () => x.reshape(shape),
          throwsA(isA<InvalidParameterException>()),
        );
      }
      expect(() => ReshapeOp([]), throwsA(isA<InvalidParameterException>()));
      expect(
        () => x.transpose([1, 0]).reshape([6]),
        throwsA(isA<NonContiguousException>()),
      );
    },
  );

  test('offset views share mutations while clones own storage', () {
    final raw = Float64List.fromList([-1, 2, 3, 4, 5, -2]);
    final x = TensorBuffer(
      storage: TensorStorage(raw, DType.float64),
      shape: [2, 2],
      storageOffset: 1,
    );
    final y = x.transpose([1, 0]).unsqueeze(-1).squeeze(-1);
    final copied = y.contiguous();
    final cloned = x.clone();
    expect(identical(x.contiguous(), x), isTrue);
    raw[1] = 9;
    expect(y[[0, 0]], 9);
    expect(copied.toList(), [2, 4, 3, 5]);
    expect(cloned.toList(), [2, 3, 4, 5]);
    expect(raw.first, -1);
    expect(raw.last, -2);
  });
}
