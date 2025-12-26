/// Tensor creation and edge case tests based on PyTorch test_tensor_creation_ops.py patterns
///
/// Tests cover:
/// - Tensor creation (zeros, ones, from data)
/// - Edge cases (empty tensors, single element, large shapes)
/// - Shape validation
/// - Boundary conditions
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Creation', () {
    group('zeros', () {
      test('creates zero-filled tensor with correct shape', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(tensor.shape, equals([2, 3, 4]));
        expect(tensor.numel, equals(24));

        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(0.0));
        }
      });

      test('creates with specified dtype', () {
        final f64 = TensorBuffer.zeros([2, 3], dtype: DType.float64);
        expect(f64.dtype, equals(DType.float64));

        final i32 = TensorBuffer.zeros([2, 3], dtype: DType.int32);
        expect(i32.dtype, equals(DType.int32));

        final u8 = TensorBuffer.zeros([2, 3], dtype: DType.uint8);
        expect(u8.dtype, equals(DType.uint8));
      });

      test('creates with contiguous memory format', () {
        final tensor = TensorBuffer.zeros(
          [1, 3, 224, 224],
          memoryFormat: MemoryFormat.contiguous,
        );

        expect(tensor.isContiguous, isTrue);
        expect(tensor.strides, equals([150528, 50176, 224, 1]));
      });

      test('creates with channels-last memory format', () {
        final tensor = TensorBuffer.zeros(
          [1, 3, 224, 224],
          memoryFormat: MemoryFormat.channelsLast,
        );

        // NHWC strides
        expect(tensor.strides, equals([150528, 1, 672, 3]));
      });
    });

    group('ones', () {
      test('creates one-filled tensor', () {
        final tensor = TensorBuffer.ones([3, 4]);

        expect(tensor.numel, equals(12));
        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(1.0));
        }
      });

      test('creates with specified dtype', () {
        final tensor = TensorBuffer.ones([2, 3], dtype: DType.int32);

        expect(tensor.dtype, equals(DType.int32));
        expect(tensor.storage.getAsDouble(0), equals(1.0));
      });
    });

    group('fromFloat32List', () {
      test('creates tensor from list', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        expect(tensor.shape, equals([2, 3]));
        expect(tensor.dtype, equals(DType.float32));
        expect(tensor[[0, 0]], equals(1.0));
        expect(tensor[[1, 2]], equals(6.0));
      });

      test('throws on shape mismatch', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);

        expect(
          () => TensorBuffer.fromFloat32List(data, [2, 4]),
          throwsArgumentError,
        );
      });
    });

    group('fromUint8List', () {
      test('creates tensor from list', () {
        final data = Uint8List.fromList([0, 128, 255]);
        final tensor = TensorBuffer.fromUint8List(data, [3]);

        expect(tensor.dtype, equals(DType.uint8));
        expect(tensor[[0]], equals(0.0));
        expect(tensor[[1]], equals(128.0));
        expect(tensor[[2]], equals(255.0));
      });
    });
  });

  group('Edge Cases - Single Element', () {
    test('single element tensor', () {
      final data = Float32List.fromList([42.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1]);

      expect(tensor.shape, equals([1]));
      expect(tensor.numel, equals(1));
      expect(tensor[[0]], equals(42.0));
    });

    test('single element in multi-dimensional shape', () {
      final data = Float32List.fromList([42.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1, 1]);

      expect(tensor.shape, equals([1, 1, 1, 1]));
      expect(tensor.rank, equals(4));
      expect(tensor.numel, equals(1));
      expect(tensor[[0, 0, 0, 0]], equals(42.0));
    });

    test('single element reshape', () {
      final tensor = TensorBuffer.ones([1]);
      final reshaped = tensor.reshape([1, 1, 1]);

      expect(reshaped.shape, equals([1, 1, 1]));
      expect(reshaped[[0, 0, 0]], equals(1.0));
    });

    test('single element transpose', () {
      final data = Float32List.fromList([42.0]);
      final tensor = TensorBuffer.fromFloat32List(data, [1, 1]);
      final transposed = tensor.transpose([1, 0]);

      expect(transposed.shape, equals([1, 1]));
      expect(transposed[[0, 0]], equals(42.0));
    });

    test('single element squeeze removes all dims', () {
      final tensor = TensorBuffer.ones([1, 1, 1]);
      final squeezed = tensor.squeeze();

      expect(squeezed.shape, isEmpty);
    });
  });

  group('Edge Cases - High Dimensional', () {
    test('5D tensor operations', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5, 6]);

      expect(tensor.rank, equals(5));
      expect(tensor.numel, equals(720));

      final transposed = tensor.transpose([4, 3, 2, 1, 0]);
      expect(transposed.shape, equals([6, 5, 4, 3, 2]));
    });

    test('6D tensor operations', () {
      final tensor = TensorBuffer.zeros([2, 2, 2, 2, 2, 2]);

      expect(tensor.rank, equals(6));
      expect(tensor.numel, equals(64));
    });

    test('high dimensional reshape', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final reshaped = tensor.reshape([1, 2, 1, 3, 1, 4, 1]);

      expect(reshaped.rank, equals(7));
      expect(reshaped.numel, equals(24));
    });
  });

  group('Edge Cases - Large Shapes', () {
    test('large 1D tensor', () {
      final tensor = TensorBuffer.zeros([1000000]);

      expect(tensor.numel, equals(1000000));
      expect(tensor.sizeInBytes, equals(4000000)); // float32
    });

    test('large image-like tensor', () {
      final tensor = TensorBuffer.zeros([1, 3, 1920, 1080]);

      expect(tensor.numel, equals(6220800));
      expect(tensor.shape, equals([1, 3, 1920, 1080]));
    });

    test('stride calculation for large tensor', () {
      final tensor = TensorBuffer.zeros([8, 3, 512, 512]);
      final strides = tensor.strides;

      // Row-major strides
      expect(strides[0], equals(3 * 512 * 512)); // 786432
      expect(strides[1], equals(512 * 512)); // 262144
      expect(strides[2], equals(512));
      expect(strides[3], equals(1));
    });
  });

  group('Edge Cases - Size One Dimensions', () {
    test('tensor with multiple size-1 dimensions', () {
      final tensor = TensorBuffer.zeros([1, 1, 224, 1, 224, 1]);

      expect(tensor.shape, equals([1, 1, 224, 1, 224, 1]));
      expect(tensor.numel, equals(50176));
    });

    test('size-1 dimensions dont affect contiguity', () {
      final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1, 224]);
      expect(tensor.isContiguous, isTrue);
    });

    test('squeeze all size-1 dimensions', () {
      final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1]);
      final squeezed = tensor.squeeze();

      expect(squeezed.shape, equals([3, 224]));
    });

    test('squeeze specific size-1 dimensions', () {
      final tensor = TensorBuffer.zeros([1, 3, 1, 224]);

      expect(tensor.squeeze(0).shape, equals([3, 1, 224]));
      expect(tensor.squeeze(2).shape, equals([1, 3, 224]));
    });
  });

  group('Shape Validation', () {
    test('empty shape throws', () {
      expect(
        () => TensorBuffer.zeros([]),
        throwsArgumentError,
      );
    });

    test('zero dimension throws', () {
      expect(
        () => TensorBuffer.zeros([2, 0, 3]),
        throwsArgumentError,
      );
    });

    test('negative dimension throws', () {
      expect(
        () => TensorBuffer.zeros([2, -1, 3]),
        throwsArgumentError,
      );
    });

    test('data length must match shape', () {
      final data = Float32List(10);

      expect(
        () => TensorBuffer.fromFloat32List(data, [2, 6]),
        throwsArgumentError,
      );
    });
  });

  group('Indexing Edge Cases', () {
    test('indexing at boundaries', () {
      final tensor = TensorBuffer.zeros([3, 4, 5]);

      // First element
      expect(() => tensor[[0, 0, 0]], returnsNormally);

      // Last element
      expect(() => tensor[[2, 3, 4]], returnsNormally);

      // Out of bounds
      expect(() => tensor[[3, 0, 0]], throwsRangeError);
      expect(() => tensor[[0, 4, 0]], throwsRangeError);
      expect(() => tensor[[0, 0, 5]], throwsRangeError);
    });

    test('negative indices throw', () {
      final tensor = TensorBuffer.zeros([3, 4]);

      expect(() => tensor[[-1, 0]], throwsRangeError);
      expect(() => tensor[[0, -1]], throwsRangeError);
    });

    test('wrong number of indices throws', () {
      final tensor = TensorBuffer.zeros([3, 4, 5]);

      expect(() => tensor[[0, 0]], throwsArgumentError);
      expect(() => tensor[[0, 0, 0, 0]], throwsArgumentError);
    });
  });

  group('numel and sizeInBytes', () {
    test('numel is product of shape', () {
      expect(TensorBuffer.zeros([2, 3, 4]).numel, equals(24));
      expect(TensorBuffer.zeros([1]).numel, equals(1));
      expect(TensorBuffer.zeros([100, 100]).numel, equals(10000));
    });

    test('sizeInBytes depends on dtype', () {
      final f32 = TensorBuffer.zeros([100], dtype: DType.float32);
      expect(f32.sizeInBytes, equals(400));

      final f64 = TensorBuffer.zeros([100], dtype: DType.float64);
      expect(f64.sizeInBytes, equals(800));

      final u8 = TensorBuffer.zeros([100], dtype: DType.uint8);
      expect(u8.sizeInBytes, equals(100));

      final i64 = TensorBuffer.zeros([100], dtype: DType.int64);
      expect(i64.sizeInBytes, equals(800));
    });
  });

  group('rank property', () {
    test('rank equals number of dimensions', () {
      expect(TensorBuffer.zeros([5]).rank, equals(1));
      expect(TensorBuffer.zeros([2, 3]).rank, equals(2));
      expect(TensorBuffer.zeros([2, 3, 4]).rank, equals(3));
      expect(TensorBuffer.zeros([1, 2, 3, 4]).rank, equals(4));
      expect(TensorBuffer.zeros([1, 1, 1, 1, 1]).rank, equals(5));
    });
  });

  group('Data Access', () {
    test('data property requires contiguous tensor', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      expect(() => tensor.data, returnsNormally);

      final transposed = tensor.transpose([2, 0, 1]);
      expect(() => transposed.data, throwsStateError);
    });

    test('dataAsFloat32List requires float32 dtype', () {
      final f32 = TensorBuffer.zeros([2, 3], dtype: DType.float32);
      expect(() => f32.dataAsFloat32List, returnsNormally);

      final f64 = TensorBuffer.zeros([2, 3], dtype: DType.float64);
      expect(() => f64.dataAsFloat32List, throwsStateError);
    });
  });

  group('Memory Format', () {
    test('default is contiguous', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      expect(tensor.memoryFormat, equals(MemoryFormat.contiguous));
    });

    test('channels-last format for 4D', () {
      final tensor = TensorBuffer.zeros(
        [1, 3, 224, 224],
        memoryFormat: MemoryFormat.channelsLast,
      );

      expect(tensor.memoryFormat, equals(MemoryFormat.channelsLast));
    });

    test('channels-last throws for 2D', () {
      expect(
        () => TensorBuffer.zeros([224, 224], memoryFormat: MemoryFormat.channelsLast),
        throwsUnsupportedError,
      );
    });
  });

  group('Storage Offset', () {
    test('new tensors have zero offset', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      expect(tensor.storageOffset, equals(0));
    });

    test('view operations preserve or adjust offset', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final transposed = tensor.transpose([2, 0, 1]);

      // Transpose doesn't change offset
      expect(transposed.storageOffset, equals(0));
    });
  });
}
