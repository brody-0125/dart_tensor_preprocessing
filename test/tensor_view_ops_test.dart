/// View operations tests based on PyTorch test_view_ops.py patterns
///
/// Tests cover:
/// - View semantics (zero-copy operations)
/// - Reshape and layout preservation
/// - Dimension manipulation (transpose, permute, squeeze, unsqueeze)
/// - Contiguity checks
/// - Storage sharing verification
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('View Operations', () {
    group('transpose_view', () {
      test('transpose returns view sharing storage', () {
        final tensor = TensorBuffer.ones([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        // Verify storage is shared (zero-copy)
        expect(identical(transposed.storage, tensor.storage), isTrue);
        expect(transposed.shape, equals([4, 2, 3]));
      });

      test('transpose preserves data access', () {
        final data = Float32List.fromList(
            List.generate(24, (i) => i.toDouble()));
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4]);

        // [2, 3, 4] -> [4, 2, 3] via [2, 0, 1]
        final transposed = tensor.transpose([2, 0, 1]);

        // Verify original data is accessible through transposed view
        expect(transposed[[0, 0, 0]], equals(tensor[[0, 0, 0]]));
        expect(transposed[[3, 1, 2]], equals(tensor[[1, 2, 3]]));
      });

      test('double transpose restores original shape', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final t1 = tensor.transpose([2, 0, 1]); // [4, 2, 3]
        final t2 = t1.transpose([1, 2, 0]); // back to [2, 3, 4]

        expect(t2.shape, equals([2, 3, 4]));
      });

      test('transpose 2D is swap dims', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        final transposed = tensor.transpose([1, 0]);

        expect(transposed.shape, equals([3, 2]));
        // Row-major: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        expect(transposed[[0, 0]], equals(1));
        expect(transposed[[0, 1]], equals(4));
        expect(transposed[[1, 0]], equals(2));
        expect(transposed[[2, 1]], equals(6));
      });

      test('transpose with identity permutation', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final same = tensor.transpose([0, 1, 2]);

        expect(same.shape, equals(tensor.shape));
        expect(same.strides, equals(tensor.strides));
      });

      test('transpose throws on invalid axes length', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(() => tensor.transpose([0, 1]), throwsArgumentError);
        expect(() => tensor.transpose([0, 1, 2, 3]), throwsArgumentError);
      });

      test('transpose throws on out of range axis', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(() => tensor.transpose([0, 1, 5]), throwsRangeError);
        expect(() => tensor.transpose([-1, 0, 1]), throwsRangeError);
      });

      test('transpose throws on duplicate axis', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(() => tensor.transpose([0, 1, 1]), throwsArgumentError);
      });
    });

    group('reshape_view', () {
      test('reshape contiguous tensor returns view', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final reshaped = tensor.reshape([6, 4]);

        // For contiguous tensors, reshape is a view
        expect(identical(reshaped.storage, tensor.storage), isTrue);
        expect(reshaped.numel, equals(tensor.numel));
      });

      test('reshape preserves total elements', () {
        final tensor = TensorBuffer.ones([2, 3, 4]);
        final reshaped = tensor.reshape([24]);

        expect(reshaped.numel, equals(24));
        // All elements should still be 1.0
        for (int i = 0; i < 24; i++) {
          expect(reshaped.storage.getAsDouble(i), equals(1.0));
        }
      });

      test('reshape to higher dimensions', () {
        final tensor = TensorBuffer.zeros([24]);
        final reshaped = tensor.reshape([2, 3, 4]);

        expect(reshaped.shape, equals([2, 3, 4]));
        expect(reshaped.rank, equals(3));
      });

      test('reshape non-contiguous throws StateError', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        expect(transposed.isContiguous, isFalse);
        expect(() => transposed.reshape([24]), throwsStateError);
      });

      test('reshape with size mismatch throws', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(() => tensor.reshape([2, 3, 5]), throwsArgumentError);
        expect(() => tensor.reshape([25]), throwsArgumentError);
      });

      test('reshape single element tensor', () {
        final data = Float32List.fromList([42.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1]);

        final reshaped = tensor.reshape([1]);
        expect(reshaped.shape, equals([1]));
        expect(reshaped[[0]], equals(42.0));
      });
    });

    group('squeeze_view', () {
      test('squeeze removes all size-1 dimensions', () {
        final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1, 224, 1]);
        final squeezed = tensor.squeeze();

        expect(squeezed.shape, equals([3, 224, 224]));
        expect(identical(squeezed.storage, tensor.storage), isTrue);
      });

      test('squeeze specific dimension', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final squeezed = tensor.squeeze(0);

        expect(squeezed.shape, equals([3, 224, 224]));
      });

      test('squeeze non-1 dimension does nothing', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final squeezed = tensor.squeeze(1); // dim 1 has size 3

        expect(squeezed.shape, equals([1, 3, 224, 224]));
      });

      test('squeeze all dimensions of size 1', () {
        final tensor = TensorBuffer.zeros([1, 1, 1, 1]);
        final squeezed = tensor.squeeze();

        expect(squeezed.shape, isEmpty);
      });

      test('squeeze preserves strides correctly', () {
        final tensor = TensorBuffer.zeros([1, 3, 1, 4]);
        final squeezed = tensor.squeeze();

        expect(squeezed.shape, equals([3, 4]));
        // Strides should be recalculated for squeezed dimensions
        expect(squeezed.strides.length, equals(2));
      });
    });

    group('unsqueeze_view', () {
      test('unsqueeze adds dimension at position', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);

        expect(tensor.unsqueeze(0).shape, equals([1, 3, 224, 224]));
        expect(tensor.unsqueeze(1).shape, equals([3, 1, 224, 224]));
        expect(tensor.unsqueeze(2).shape, equals([3, 224, 1, 224]));
        expect(tensor.unsqueeze(3).shape, equals([3, 224, 224, 1]));
      });

      test('unsqueeze is view operation', () {
        final tensor = TensorBuffer.ones([3, 4]);
        final unsqueezed = tensor.unsqueeze(0);

        expect(identical(unsqueezed.storage, tensor.storage), isTrue);
      });

      test('unsqueeze then squeeze restores shape', () {
        final tensor = TensorBuffer.zeros([3, 4]);
        final result = tensor.unsqueeze(1).squeeze(1);

        expect(result.shape, equals([3, 4]));
      });

      test('unsqueeze at end', () {
        final tensor = TensorBuffer.zeros([2, 3]);
        final unsqueezed = tensor.unsqueeze(2);

        expect(unsqueezed.shape, equals([2, 3, 1]));
      });

      test('unsqueeze throws on out of range', () {
        final tensor = TensorBuffer.zeros([2, 3]);

        expect(() => tensor.unsqueeze(-1), throwsRangeError);
        expect(() => tensor.unsqueeze(4), throwsRangeError);
      });
    });

    group('contiguous', () {
      test('contiguous on contiguous tensor returns self', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(tensor.isContiguous, isTrue);
        expect(identical(tensor.contiguous(), tensor), isTrue);
      });

      test('contiguous on transposed tensor creates copy', () {
        final data = Float32List.fromList(
            List.generate(24, (i) => i.toDouble()));
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        expect(transposed.isContiguous, isFalse);

        final contiguous = transposed.contiguous();

        expect(contiguous.isContiguous, isTrue);
        expect(identical(contiguous.storage, tensor.storage), isFalse);
        expect(contiguous.shape, equals([4, 2, 3]));
      });

      test('contiguous preserves data values', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
        final transposed = tensor.transpose([1, 0]);
        final contiguous = transposed.contiguous();

        // After transpose [1,0]: shape [3,2]
        // Original: [[1,2,3],[4,5,6]]
        // Transposed: [[1,4],[2,5],[3,6]]
        expect(contiguous[[0, 0]], equals(1));
        expect(contiguous[[0, 1]], equals(4));
        expect(contiguous[[1, 0]], equals(2));
        expect(contiguous[[2, 1]], equals(6));
      });

      test('contiguous strides are row-major', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);
        final contiguous = transposed.contiguous();

        // Row-major strides for [4, 2, 3]: [6, 3, 1]
        expect(contiguous.strides, equals([6, 3, 1]));
      });
    });

    group('clone', () {
      test('clone creates independent copy', () {
        final tensor = TensorBuffer.ones([2, 3]);
        final cloned = tensor.clone();

        expect(identical(cloned.storage, tensor.storage), isFalse);

        // Modify original
        tensor.storage.setFromDouble(0, 99.0);

        // Clone should be unaffected
        expect(cloned.storage.getAsDouble(0), equals(1.0));
      });

      test('clone of transposed tensor is contiguous', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);
        final cloned = transposed.clone();

        expect(cloned.isContiguous, isTrue);
        expect(cloned.shape, equals([4, 2, 3]));
      });

      test('clone preserves data', () {
        final data = Float32List.fromList(
            List.generate(12, (i) => i.toDouble()));
        final tensor = TensorBuffer.fromFloat32List(data, [3, 4]);
        final cloned = tensor.clone();

        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 4; j++) {
            expect(cloned[[i, j]], equals(tensor[[i, j]]));
          }
        }
      });
    });

    group('isContiguous checks', () {
      test('freshly created tensor is contiguous', () {
        expect(TensorBuffer.zeros([2, 3, 4]).isContiguous, isTrue);
        expect(TensorBuffer.ones([1, 3, 224, 224]).isContiguous, isTrue);
      });

      test('transposed tensor is not contiguous', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final transposed = tensor.transpose([1, 0, 2]);

        expect(transposed.isContiguous, isFalse);
      });

      test('identity transpose is contiguous', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final same = tensor.transpose([0, 1, 2]);

        expect(same.isContiguous, isTrue);
      });

      test('squeeze/unsqueeze preserves contiguity', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);

        expect(tensor.squeeze().isContiguous, isTrue);
        expect(tensor.unsqueeze(2).isContiguous, isTrue);
      });

      test('reshape preserves contiguity', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final reshaped = tensor.reshape([6, 4]);

        expect(reshaped.isContiguous, isTrue);
      });

      test('size-1 dimensions do not affect contiguity', () {
        final tensor = TensorBuffer.zeros([1, 3, 1, 4]);
        expect(tensor.isContiguous, isTrue);
      });
    });

    group('chained view operations', () {
      test('multiple transposes', () {
        final tensor = TensorBuffer.zeros([2, 3, 4, 5]);

        final t1 = tensor.transpose([3, 2, 1, 0]); // [5,4,3,2]
        final t2 = t1.transpose([3, 2, 1, 0]); // back to [2,3,4,5]

        expect(t2.shape, equals([2, 3, 4, 5]));
        expect(identical(t2.storage, tensor.storage), isTrue);
      });

      test('transpose then squeeze', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final result = tensor.transpose([0, 2, 3, 1]).squeeze(0);

        expect(result.shape, equals([224, 224, 3]));
      });

      test('unsqueeze then transpose', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        final result = tensor.unsqueeze(0).transpose([0, 2, 3, 1]);

        expect(result.shape, equals([1, 224, 224, 3]));
      });

      test('complex chain of operations', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);

        final result = tensor
            .unsqueeze(0) // [1, 3, 224, 224]
            .transpose([0, 2, 3, 1]) // [1, 224, 224, 3]
            .squeeze(0); // [224, 224, 3]

        expect(result.shape, equals([224, 224, 3]));
        expect(identical(result.storage, tensor.storage), isTrue);
      });
    });
  });

  group('Stride Calculations', () {
    test('contiguous strides are row-major', () {
      final strides = TensorBuffer.computeStrides(
        [2, 3, 4],
        MemoryFormat.contiguous,
      );

      // Row-major: stride[i] = product of shape[i+1:]
      expect(strides, equals([12, 4, 1]));
    });

    test('channels-last strides for 4D', () {
      final strides = TensorBuffer.computeStrides(
        [2, 3, 4, 5],
        MemoryFormat.channelsLast,
      );

      // NHWC layout: N, H, W, C -> C is fastest
      // N stride = H*W*C = 4*5*3 = 60
      // C stride = 1
      // H stride = W*C = 5*3 = 15
      // W stride = C = 3
      expect(strides, equals([60, 1, 15, 3]));
    });

    test('channels-last strides for 3D', () {
      final strides = TensorBuffer.computeStrides(
        [3, 4, 5],
        MemoryFormat.channelsLast,
      );

      // HWC layout: C is fastest
      // C stride = 1
      // H stride = W*C = 5*3 = 15
      // W stride = C = 3
      expect(strides, equals([1, 15, 3]));
    });

    test('stride calculation throws for invalid rank with channelsLast', () {
      expect(
        () => TensorBuffer.computeStrides([2, 3], MemoryFormat.channelsLast),
        throwsUnsupportedError,
      );
    });

    test('transposed tensor has permuted strides', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final transposed = tensor.transpose([2, 0, 1]);

      // Original strides: [12, 4, 1]
      // After transpose [2,0,1]: strides become [1, 12, 4]
      expect(transposed.strides, equals([1, 12, 4]));
    });
  });
}
