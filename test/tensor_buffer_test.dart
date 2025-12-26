import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('TensorBuffer', () {
    group('creation', () {
      test('zeros creates zero-filled tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(tensor.shape, equals([2, 3, 4]));
        expect(tensor.dtype, equals(DType.float32));
        expect(tensor.numel, equals(24));
        expect(tensor.isContiguous, isTrue);

        // Check all values are zero
        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(0.0));
        }
      });

      test('ones creates one-filled tensor', () {
        final tensor = TensorBuffer.ones([2, 3]);

        expect(tensor.shape, equals([2, 3]));
        expect(tensor.numel, equals(6));

        // Check all values are one
        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(1.0));
        }
      });

      test('fromFloat32List creates tensor from data', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        expect(tensor.shape, equals([2, 3]));
        expect(tensor.dtype, equals(DType.float32));
        expect(tensor[[0, 0]], equals(1.0));
        expect(tensor[[1, 2]], equals(6.0));
      });

      test('fromUint8List creates tensor from data', () {
        final data = Uint8List.fromList([255, 128, 0, 64]);
        final tensor = TensorBuffer.fromUint8List(data, [2, 2]);

        expect(tensor.dtype, equals(DType.uint8));
        expect(tensor[[0, 0]], equals(255.0));
        expect(tensor[[1, 1]], equals(64.0));
      });
    });

    group('indexing', () {
      test('operator[] returns correct values', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        expect(tensor[[0, 0]], equals(1.0));
        expect(tensor[[0, 1]], equals(2.0));
        expect(tensor[[0, 2]], equals(3.0));
        expect(tensor[[1, 0]], equals(4.0));
        expect(tensor[[1, 1]], equals(5.0));
        expect(tensor[[1, 2]], equals(6.0));
      });

      test('throws on invalid indices', () {
        final tensor = TensorBuffer.zeros([2, 3]);

        expect(() => tensor[[0]], throwsArgumentError);
        expect(() => tensor[[0, 0, 0]], throwsArgumentError);
        expect(() => tensor[[2, 0]], throwsRangeError);
        expect(() => tensor[[0, 3]], throwsRangeError);
      });
    });

    group('transpose', () {
      test('transposes 2D tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
        final transposed = tensor.transpose([1, 0]);

        expect(transposed.shape, equals([3, 2]));
        expect(transposed[[0, 0]], equals(1.0));
        expect(transposed[[0, 1]], equals(4.0));
        expect(transposed[[1, 0]], equals(2.0));
        expect(transposed[[2, 1]], equals(6.0));
      });

      test('transpose is zero-copy', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        expect(transposed.shape, equals([4, 2, 3]));
        expect(identical(transposed.storage, tensor.storage), isTrue);
      });

      test('transpose NCHW to NHWC', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final nhwc = tensor.transpose([0, 2, 3, 1]);

        expect(nhwc.shape, equals([1, 224, 224, 3]));
      });
    });

    group('reshape', () {
      test('reshapes contiguous tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final reshaped = tensor.reshape([6, 4]);

        expect(reshaped.shape, equals([6, 4]));
        expect(reshaped.numel, equals(24));
      });

      test('throws on non-contiguous tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        expect(() => transposed.reshape([24]), throwsStateError);
      });

      test('throws on size mismatch', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(() => tensor.reshape([2, 3, 5]), throwsArgumentError);
      });
    });

    group('squeeze and unsqueeze', () {
      test('squeeze removes size-1 dimensions', () {
        final tensor = TensorBuffer.zeros([1, 3, 1, 224, 224]);
        final squeezed = tensor.squeeze();

        expect(squeezed.shape, equals([3, 224, 224]));
      });

      test('squeeze specific dimension', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final squeezed = tensor.squeeze(0);

        expect(squeezed.shape, equals([3, 224, 224]));
      });

      test('unsqueeze adds dimension', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        final unsqueezed = tensor.unsqueeze(0);

        expect(unsqueezed.shape, equals([1, 3, 224, 224]));
      });
    });

    group('contiguous', () {
      test('returns self for contiguous tensor', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(identical(tensor.contiguous(), tensor), isTrue);
      });

      test('creates copy for non-contiguous tensor', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
        final transposed = tensor.transpose([1, 0]);
        final contiguous = transposed.contiguous();

        expect(contiguous.isContiguous, isTrue);
        expect(contiguous.shape, equals([3, 2]));
        expect(contiguous[[0, 0]], equals(1.0));
        expect(contiguous[[0, 1]], equals(4.0));
        expect(contiguous[[1, 0]], equals(2.0));
      });
    });

    group('clone', () {
      test('creates independent copy', () {
        final tensor = TensorBuffer.ones([2, 3]);
        final cloned = tensor.clone();

        expect(cloned.shape, equals(tensor.shape));
        expect(identical(cloned.storage, tensor.storage), isFalse);

        // Modify original should not affect clone
        tensor.storage.setFromDouble(0, 99.0);
        expect(cloned.storage.getAsDouble(0), equals(1.0));
      });
    });

    group('strides', () {
      test('computes contiguous strides correctly', () {
        final strides = TensorBuffer.computeStrides(
          [2, 3, 4],
          MemoryFormat.contiguous,
        );

        expect(strides, equals([12, 4, 1]));
      });

      test('computes channels-last strides correctly', () {
        final strides = TensorBuffer.computeStrides(
          [1, 3, 224, 224],
          MemoryFormat.channelsLast,
        );

        // NHWC strides for [N=1, C=3, H=224, W=224]
        // N stride = H*W*C = 224*224*3 = 150528
        // C stride = 1 (fastest changing)
        // H stride = W*C = 224*3 = 672
        // W stride = C = 3
        expect(strides, equals([150528, 1, 672, 3]));
      });
    });
  });

  group('DType', () {
    test('creates correct buffers', () {
      expect(DType.float32.createBuffer(10), isA<Float32List>());
      expect(DType.float64.createBuffer(10), isA<Float64List>());
      expect(DType.int32.createBuffer(10), isA<Int32List>());
      expect(DType.uint8.createBuffer(10), isA<Uint8List>());
    });

    test('infers type from typed data', () {
      expect(DType.fromTypedData(Float32List(1)), equals(DType.float32));
      expect(DType.fromTypedData(Uint8List(1)), equals(DType.uint8));
      expect(DType.fromTypedData(Int64List(1)), equals(DType.int64));
    });

    test('reports correct byte size', () {
      expect(DType.float32.byteSize, equals(4));
      expect(DType.float64.byteSize, equals(8));
      expect(DType.int8.byteSize, equals(1));
      expect(DType.int16.byteSize, equals(2));
      expect(DType.int32.byteSize, equals(4));
      expect(DType.int64.byteSize, equals(8));
      expect(DType.uint8.byteSize, equals(1));
    });

    test('identifies floating point types', () {
      expect(DType.float32.isFloatingPoint, isTrue);
      expect(DType.float64.isFloatingPoint, isTrue);
      expect(DType.int32.isFloatingPoint, isFalse);
      expect(DType.uint8.isFloatingPoint, isFalse);
    });
  });
}
