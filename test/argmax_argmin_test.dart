// Argmax/argmin operations tests
//
// Tests cover:
// - argmax(), argmin() for global (flat) index
// - argmaxAxis(), argminAxis() for axis-wise index reduction
// - ArgMaxOp, ArgMinOp TransformOp subclasses
// - Different data types, non-contiguous tensors, keepDims, negative axis
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('argmax (global)', () {
    test('1D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5]),
        [5],
      );
      expect(tensor.argmax(), equals(4));
    });

    test('2D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3, 4, 5, 6]),
        [2, 3],
      );
      // Flat index of max (6) is 5
      expect(tensor.argmax(), equals(5));
    });

    test('single element', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([42]),
        [1],
      );
      expect(tensor.argmax(), equals(0));
    });

    test('duplicate max returns first occurrence', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 5, 3, 5, 2]),
        [5],
      );
      expect(tensor.argmax(), equals(1));
    });

    test('non-contiguous (transposed) tensor', () {
      // Original [2,3]: [[1,2,3],[4,5,6]]
      // Transposed [3,2]: [[1,4],[2,5],[3,6]]
      // In logical order: 1,4,2,5,3,6 → max=6 at flat index 5
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3, 4, 5, 6]),
        [2, 3],
      );
      final transposed = tensor.transpose([1, 0]);
      expect(transposed.argmax(), equals(5));
    });

    test('negative values', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([-5, -3, -1, -4, -2]),
        [5],
      );
      expect(tensor.argmax(), equals(2));
    });
  });

  group('argmin (global)', () {
    test('1D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5]),
        [5],
      );
      expect(tensor.argmin(), equals(1));
    });

    test('2D tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([6, 5, 4, 3, 2, 1]),
        [2, 3],
      );
      // Flat index of min (1) is 5
      expect(tensor.argmin(), equals(5));
    });

    test('single element', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([42]),
        [1],
      );
      expect(tensor.argmin(), equals(0));
    });

    test('duplicate min returns first occurrence', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([5, 1, 3, 1, 2]),
        [5],
      );
      expect(tensor.argmin(), equals(1));
    });

    test('non-contiguous (transposed) tensor', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 2, 1, 6, 5, 4]),
        [2, 3],
      );
      final transposed = tensor.transpose([1, 0]);
      // Transposed [3,2]: [[3,6],[2,5],[1,4]]
      // Logical order: 3,6,2,5,1,4 → min=1 at flat index 4
      expect(transposed.argmin(), equals(4));
    });
  });

  group('argmaxAxis', () {
    test('2D along axis 0', () {
      // [[3,1,4],[1,5,9]]
      // Along axis 0: argmax([3,1])=0, argmax([1,5])=1, argmax([4,9])=1
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argmaxAxis(0);
      expect(result.shape, equals([3]));
      expect(result.toList(), equals([0, 1, 1]));
    });

    test('2D along axis 1', () {
      // [[3,1,4],[1,5,9]]
      // Along axis 1: argmax([3,1,4])=2, argmax([1,5,9])=2
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argmaxAxis(1);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([2, 2]));
    });

    test('3D along axis 0', () {
      // Shape [2,2,3]:
      // [[[1,2,3],[4,5,6]],
      //  [[7,8,9],[10,11,12]]]
      // Along axis 0: compare [1,7]→1, [2,8]→1, ... all index 1
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        [2, 2, 3],
      );
      final result = tensor.argmaxAxis(0);
      expect(result.shape, equals([2, 3]));
      expect(result.toList(), equals([1, 1, 1, 1, 1, 1]));
    });

    test('3D along axis 2', () {
      // Shape [2,2,3]:
      // [[[1,2,3],[4,5,6]],
      //  [[7,8,9],[10,11,12]]]
      // Along axis 2: argmax([1,2,3])=2, argmax([4,5,6])=2, ...
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        [2, 2, 3],
      );
      final result = tensor.argmaxAxis(2);
      expect(result.shape, equals([2, 2]));
      expect(result.toList(), equals([2, 2, 2, 2]));
    });

    test('keepDims: true', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argmaxAxis(1, keepDims: true);
      expect(result.shape, equals([2, 1]));
      expect(result.toList(), equals([2, 2]));
    });

    test('negative axis', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      // axis -1 == axis 1
      final result = tensor.argmaxAxis(-1);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([2, 2]));
    });

    test('1D tensor (scalar result)', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5]),
        [5],
      );
      final result = tensor.argmaxAxis(0);
      expect(result.shape, equals([1]));
      expect(result.toList(), equals([4]));
    });

    test('output dtype is int64', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3]),
        [3],
      );
      final result = tensor.argmaxAxis(0);
      expect(result.dtype, equals(DType.int64));
    });

    test('non-contiguous (transposed) tensor', () {
      // Original [2,3]: [[1,2,3],[4,5,6]]
      // Transposed [3,2]: [[1,4],[2,5],[3,6]]
      // argmaxAxis(1): argmax([1,4])=1, argmax([2,5])=1, argmax([3,6])=1
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3, 4, 5, 6]),
        [2, 3],
      );
      final transposed = tensor.transpose([1, 0]);
      final result = transposed.argmaxAxis(1);
      expect(result.shape, equals([3]));
      expect(result.toList(), equals([1, 1, 1]));
    });

    test('uint8 input dtype', () {
      final tensor = TensorBuffer(
        storage: TensorStorage(Uint8List.fromList([10, 200, 50]), DType.uint8),
        shape: [3],
      );
      final result = tensor.argmaxAxis(0);
      expect(result.dtype, equals(DType.int64));
      expect(result.toList(), equals([1]));
    });

    test('invalid axis throws IndexOutOfBoundsException', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3]),
        [3],
      );
      expect(
        () => tensor.argmaxAxis(1),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });
  });

  group('argminAxis', () {
    test('2D along axis 0', () {
      // [[3,1,4],[1,5,9]]
      // Along axis 0: argmin([3,1])=1, argmin([1,5])=0, argmin([4,9])=0
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argminAxis(0);
      expect(result.shape, equals([3]));
      expect(result.toList(), equals([1, 0, 0]));
    });

    test('2D along axis 1', () {
      // [[3,1,4],[1,5,9]]
      // Along axis 1: argmin([3,1,4])=1, argmin([1,5,9])=0
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argminAxis(1);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([1, 0]));
    });

    test('keepDims: true', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argminAxis(0, keepDims: true);
      expect(result.shape, equals([1, 3]));
      expect(result.toList(), equals([1, 0, 0]));
    });

    test('negative axis', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final result = tensor.argminAxis(-2);
      expect(result.shape, equals([3]));
      expect(result.toList(), equals([1, 0, 0]));
    });

    test('output dtype is int64', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 2, 3]),
        [3],
      );
      final result = tensor.argminAxis(0);
      expect(result.dtype, equals(DType.int64));
    });

    test('non-contiguous (transposed) tensor', () {
      // Original [2,3]: [[6,5,4],[3,2,1]]
      // Transposed [3,2]: [[6,3],[5,2],[4,1]]
      // argminAxis(0): argmin([6,5,4])=2, argmin([3,2,1])=2
      // Wait, axis 0 of transposed [3,2]: argmin([6,3])=1, argmin([5,2])=1, argmin([4,1])=1
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([6, 5, 4, 3, 2, 1]),
        [2, 3],
      );
      final transposed = tensor.transpose([1, 0]);
      final result = transposed.argminAxis(0);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([2, 2]));
    });
  });

  group('ArgMaxOp', () {
    test('apply returns correct result', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final op = ArgMaxOp(axis: 1);
      final result = op.apply(tensor);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([2, 2]));
      expect(result.dtype, equals(DType.int64));
    });

    test('apply with keepDims', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final op = ArgMaxOp(axis: 1, keepDims: true);
      final result = op.apply(tensor);
      expect(result.shape, equals([2, 1]));
    });

    test('computeOutputShape', () {
      final op = ArgMaxOp(axis: 1);
      expect(op.computeOutputShape([2, 3, 4]), equals([2, 4]));
    });

    test('computeOutputShape with keepDims', () {
      final op = ArgMaxOp(axis: 1, keepDims: true);
      expect(op.computeOutputShape([2, 3, 4]), equals([2, 1, 4]));
    });

    test('computeOutputShape with negative axis', () {
      final op = ArgMaxOp(axis: -1);
      expect(op.computeOutputShape([2, 3, 4]), equals([2, 3]));
    });

    test('computeOutputShape 1D', () {
      final op = ArgMaxOp(axis: 0);
      expect(op.computeOutputShape([5]), equals([1]));
    });

    test('capabilities', () {
      final op = ArgMaxOp(axis: 0);
      expect(op.capabilities.preservesShape, isFalse);
      expect(op.capabilities.modifiesDType, isTrue);
      expect(op.capabilities.onnxOpType, equals('ArgMax'));
      expect(op.capabilities.pytorchEquivalent, equals('torch.argmax'));
    });

    test('name', () {
      final op = ArgMaxOp(axis: 0);
      expect(op.name, equals('ArgMax'));
    });

    test('call alias works', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([1, 3, 2]),
        [3],
      );
      final op = ArgMaxOp(axis: 0);
      final result = op.call(tensor);
      expect(result.toList(), equals([1]));
    });

    test('invalid axis throws', () {
      final op = ArgMaxOp(axis: 3);
      expect(
        () => op.computeOutputShape([2, 3]),
        throwsA(isA<IndexOutOfBoundsException>()),
      );
    });
  });

  group('ArgMinOp', () {
    test('apply returns correct result', () {
      final tensor = TensorBuffer.fromFloat32List(
        Float32List.fromList([3, 1, 4, 1, 5, 9]),
        [2, 3],
      );
      final op = ArgMinOp(axis: 1);
      final result = op.apply(tensor);
      expect(result.shape, equals([2]));
      expect(result.toList(), equals([1, 0]));
      expect(result.dtype, equals(DType.int64));
    });

    test('capabilities', () {
      final op = ArgMinOp(axis: 0);
      expect(op.capabilities.preservesShape, isFalse);
      expect(op.capabilities.modifiesDType, isTrue);
      expect(op.capabilities.onnxOpType, equals('ArgMin'));
      expect(op.capabilities.pytorchEquivalent, equals('torch.argmin'));
    });

    test('name', () {
      final op = ArgMinOp(axis: 0);
      expect(op.name, equals('ArgMin'));
    });

    test('computeOutputShape', () {
      final op = ArgMinOp(axis: 0);
      expect(op.computeOutputShape([2, 3, 4]), equals([3, 4]));
    });
  });
}
