/// Memory Layout Tests
///
/// Tests based on PyTorch test_view_ops.py patterns for memory layout operations.
///
/// Test coverage:
/// - MemoryFormat enum and extension methods
/// - Stride calculations for various ranks and formats
/// - NCHW ↔ NHWC format conversions
/// - Storage sharing and view semantics
/// - Contiguity checks and edge cases
/// - Stride-based data access patterns
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  // ============================================================
  // MemoryFormat Enum Tests
  // ============================================================
  group('MemoryFormat Enum', () {
    test('contiguous has correct layout name', () {
      // Verify NCHW layout name
      expect(MemoryFormat.contiguous.layoutName, equals('NCHW'));
    });

    test('channelsLast has correct layout name', () {
      // Verify NHWC layout name
      expect(MemoryFormat.channelsLast.layoutName, equals('NHWC'));
    });

    test('logicalOrder for contiguous', () {
      // NCHW logical order: N(0), C(1), H(2), W(3)
      expect(MemoryFormat.contiguous.logicalOrder, equals([0, 1, 2, 3]));
    });

    test('logicalOrder for channelsLast', () {
      // NHWC logical order: N(0), H(2), W(3), C(1)
      expect(MemoryFormat.channelsLast.logicalOrder, equals([0, 2, 3, 1]));
    });

    test('permuteToOther for contiguous gives NHWC permutation', () {
      // Axes order for NCHW → NHWC conversion
      expect(MemoryFormat.contiguous.permuteToOther, equals([0, 2, 3, 1]));
    });

    test('permuteToOther for channelsLast gives NCHW permutation', () {
      // Axes order for NHWC → NCHW conversion
      expect(MemoryFormat.channelsLast.permuteToOther, equals([0, 3, 1, 2]));
    });
  });

  // ============================================================
  // Stride Calculations - Contiguous (Row-Major)
  // ============================================================
  group('Stride Calculations - Contiguous', () {
    test('1D contiguous strides', () {
      // 1D tensor always has stride of 1
      final strides = TensorBuffer.computeStrides([10], MemoryFormat.contiguous);
      expect(strides, equals([1]));
    });

    test('2D contiguous strides', () {
      // Shape [3, 4]: stride[0] = 4, stride[1] = 1
      final strides = TensorBuffer.computeStrides([3, 4], MemoryFormat.contiguous);
      expect(strides, equals([4, 1]));
    });

    test('3D contiguous strides', () {
      // [2, 3, 4]: stride = [3*4, 4, 1] = [12, 4, 1]
      final strides =
          TensorBuffer.computeStrides([2, 3, 4], MemoryFormat.contiguous);
      expect(strides, equals([12, 4, 1]));
    });

    test('4D contiguous strides (NCHW)', () {
      // [2, 3, 4, 5]: stride = [3*4*5, 4*5, 5, 1] = [60, 20, 5, 1]
      final strides =
          TensorBuffer.computeStrides([2, 3, 4, 5], MemoryFormat.contiguous);
      expect(strides, equals([60, 20, 5, 1]));
    });

    test('5D contiguous strides', () {
      // Same pattern for 5D tensors
      final strides =
          TensorBuffer.computeStrides([2, 3, 4, 5, 6], MemoryFormat.contiguous);
      expect(strides, equals([360, 120, 30, 6, 1]));
    });

    test('6D contiguous strides', () {
      // 6D [2,2,2,2,2,2]: 32, 16, 8, 4, 2, 1
      final strides = TensorBuffer.computeStrides(
          [2, 2, 2, 2, 2, 2], MemoryFormat.contiguous);
      expect(strides, equals([32, 16, 8, 4, 2, 1]));
    });

    test('size-1 dimensions have correct strides', () {
      // Size-1 dimensions also have correct stride computation
      // [1, 3, 1, 224]: stride = [672, 224, 224, 1]
      final strides =
          TensorBuffer.computeStrides([1, 3, 1, 224], MemoryFormat.contiguous);
      expect(strides, equals([672, 224, 224, 1]));
    });

    test('large tensor strides do not overflow', () {
      // 8K resolution batch images (no overflow)
      final strides = TensorBuffer.computeStrides(
          [4, 3, 4320, 7680], MemoryFormat.contiguous);
      expect(strides[0], equals(3 * 4320 * 7680)); // 99,532,800
      expect(strides[3], equals(1));
    });
  });

  // ============================================================
  // Stride Calculations - Channels Last (NHWC)
  // ============================================================
  group('Stride Calculations - Channels Last', () {
    test('4D channelsLast strides (NHWC)', () {
      // NHWC layout: Channel (C) varies fastest
      // Shape [N=2, C=3, H=4, W=5]
      // N stride = H*W*C = 4*5*3 = 60
      // C stride = 1 (fastest)
      // H stride = W*C = 5*3 = 15
      // W stride = C = 3
      final strides =
          TensorBuffer.computeStrides([2, 3, 4, 5], MemoryFormat.channelsLast);
      expect(strides, equals([60, 1, 15, 3]));
    });

    test('3D channelsLast strides (HWC)', () {
      // HWC layout for 3D tensors
      // Shape [C=3, H=4, W=5]
      // C stride = 1, H stride = W*C, W stride = C
      final strides =
          TensorBuffer.computeStrides([3, 4, 5], MemoryFormat.channelsLast);
      expect(strides, equals([1, 15, 3]));
    });

    test('typical ImageNet shape channelsLast', () {
      // NHWC strides for ImageNet standard shape [1, 3, 224, 224]
      final strides =
          TensorBuffer.computeStrides([1, 3, 224, 224], MemoryFormat.channelsLast);
      expect(strides[0], equals(224 * 224 * 3)); // 150528
      expect(strides[1], equals(1)); // C is fastest
      expect(strides[2], equals(224 * 3)); // 672
      expect(strides[3], equals(3)); // W
    });

    test('channelsLast throws for 2D', () {
      // 2D tensors don't support channelsLast
      expect(
        () => TensorBuffer.computeStrides([3, 4], MemoryFormat.channelsLast),
        throwsUnsupportedError,
      );
    });

    test('channelsLast throws for 5D', () {
      // 5D tensors also don't support channelsLast
      expect(
        () => TensorBuffer.computeStrides(
            [1, 2, 3, 4, 5], MemoryFormat.channelsLast),
        throwsUnsupportedError,
      );
    });
  });

  // ============================================================
  // Memory Format Conversion - Data Integrity Tests
  // ============================================================
  group('Memory Format Conversion - Data Integrity', () {
    test('NCHW to NHWC preserves all values', () {
      // Create 4D tensor with unique values
      final data = Float32List.fromList(
          List.generate(24, (i) => i.toDouble()));
      final nchw = TensorBuffer.fromFloat32List(data, [1, 2, 3, 4]);

      // Permute to NHWC: [0, 2, 3, 1]
      final nhwc = nchw.transpose([0, 2, 3, 1]);
      expect(nhwc.shape, equals([1, 3, 4, 2]));

      // Verify all original values are correctly accessible
      for (int c = 0; c < 2; c++) {
        for (int h = 0; h < 3; h++) {
          for (int w = 0; w < 4; w++) {
            expect(nhwc[[0, h, w, c]], equals(nchw[[0, c, h, w]]));
          }
        }
      }
    });

    test('NHWC to NCHW round-trip', () {
      // NCHW → NHWC → NCHW round-trip conversion
      final original = TensorBuffer.ones([1, 3, 224, 224]);

      final nhwc = original.transpose([0, 2, 3, 1]);
      final back = nhwc.transpose([0, 3, 1, 2]);

      expect(back.shape, equals([1, 3, 224, 224]));
      // Verify same storage is shared
      expect(identical(back.storage, original.storage), isTrue);
    });

    test('CHW to HWC preserves data (3D)', () {
      // CHW → HWC conversion for 3D tensors
      final data = Float32List.fromList(
          List.generate(12, (i) => i.toDouble()));
      final chw = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

      final hwc = chw.transpose([1, 2, 0]);
      expect(hwc.shape, equals([2, 2, 3]));

      // Verify specific values
      expect(hwc[[0, 0, 0]], equals(chw[[0, 0, 0]]));
      expect(hwc[[1, 1, 2]], equals(chw[[2, 1, 1]]));
    });

    test('contiguous after permute creates new layout', () {
      // Calling contiguous after permute creates new contiguous memory
      final nchw = TensorBuffer.ones([1, 3, 4, 4]);
      final nhwc = nchw.transpose([0, 2, 3, 1]);
      final contiguousNhwc = nhwc.contiguous();

      expect(contiguousNhwc.isContiguous, isTrue);
      // Verify new storage is created
      expect(identical(contiguousNhwc.storage, nchw.storage), isFalse);
      // NHWC [1, 4, 4, 3] contiguous strides: [48, 12, 3, 1]
      expect(contiguousNhwc.strides, equals([48, 12, 3, 1]));
    });
  });

  // ============================================================
  // Storage Sharing and View Semantics Tests
  // ============================================================
  group('Storage Sharing and View Semantics', () {
    test('transpose shares storage', () {
      // transpose returns a view that shares storage
      final tensor = TensorBuffer.ones([2, 3, 4]);
      final transposed = tensor.transpose([2, 0, 1]);

      expect(identical(transposed.storage, tensor.storage), isTrue);
    });

    test('reshape shares storage for contiguous tensor', () {
      // reshape of contiguous tensor shares storage
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final reshaped = tensor.reshape([6, 4]);

      expect(identical(reshaped.storage, tensor.storage), isTrue);
    });

    test('squeeze shares storage', () {
      // squeeze also shares storage
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final squeezed = tensor.squeeze(0);

      expect(identical(squeezed.storage, tensor.storage), isTrue);
    });

    test('unsqueeze shares storage', () {
      // unsqueeze also shares storage
      final tensor = TensorBuffer.zeros([3, 224, 224]);
      final unsqueezed = tensor.unsqueeze(0);

      expect(identical(unsqueezed.storage, tensor.storage), isTrue);
    });

    test('contiguous creates new storage only when needed', () {
      // Already contiguous tensor returns itself
      final contiguousTensor = TensorBuffer.zeros([2, 3, 4]);
      expect(identical(contiguousTensor.contiguous(), contiguousTensor), isTrue);

      // Non-contiguous tensor creates new storage
      final nonContiguous = contiguousTensor.transpose([2, 0, 1]);
      final madeContinuous = nonContiguous.contiguous();
      expect(identical(madeContinuous.storage, contiguousTensor.storage), isFalse);
    });

    test('clone always creates new storage', () {
      // clone always creates new storage
      final tensor = TensorBuffer.ones([2, 3]);
      final cloned = tensor.clone();

      expect(identical(cloned.storage, tensor.storage), isFalse);
    });

    test('chained view operations share storage', () {
      // Chained view operations also share storage
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final result = tensor
          .transpose([0, 2, 3, 1]) // NCHW → NHWC
          .squeeze(0) // Remove batch
          .unsqueeze(0); // Add batch back

      expect(identical(result.storage, tensor.storage), isTrue);
    });
  });

  // ============================================================
  // Contiguity Check Tests
  // ============================================================
  group('Contiguity Checks', () {
    test('new tensor is contiguous', () {
      // Newly created tensors are contiguous
      expect(TensorBuffer.zeros([2, 3, 4]).isContiguous, isTrue);
      expect(TensorBuffer.ones([1, 3, 224, 224]).isContiguous, isTrue);
    });

    test('identity transpose preserves contiguity', () {
      // Identity permutation preserves contiguity
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final same = tensor.transpose([0, 1, 2]);
      expect(same.isContiguous, isTrue);
    });

    test('non-identity transpose breaks contiguity', () {
      // Non-identity transpose breaks contiguity
      final tensor = TensorBuffer.zeros([2, 3, 4]);

      expect(tensor.transpose([1, 0, 2]).isContiguous, isFalse);
      expect(tensor.transpose([0, 2, 1]).isContiguous, isFalse);
      expect(tensor.transpose([2, 1, 0]).isContiguous, isFalse);
    });

    test('size-1 dimensions do not affect contiguity', () {
      // Size-1 dimensions don't affect contiguity
      final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1]);
      expect(tensor.isContiguous, isTrue);
    });

    test('squeeze preserves contiguity', () {
      // squeeze preserves contiguity
      final tensor = TensorBuffer.zeros([1, 3, 1, 224]);
      expect(tensor.squeeze().isContiguous, isTrue);
    });

    test('unsqueeze preserves contiguity', () {
      // unsqueeze also preserves contiguity
      final tensor = TensorBuffer.zeros([3, 224, 224]);
      expect(tensor.unsqueeze(0).isContiguous, isTrue);
      expect(tensor.unsqueeze(2).isContiguous, isTrue);
    });

    test('reshape preserves contiguity', () {
      // reshape preserves contiguity
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      expect(tensor.reshape([24]).isContiguous, isTrue);
      expect(tensor.reshape([6, 4]).isContiguous, isTrue);
    });

    test('transpose then contiguous results in contiguous', () {
      // Calling contiguous after transpose returns a contiguous tensor
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final transposed = tensor.transpose([2, 0, 1]);
      expect(transposed.isContiguous, isFalse);

      final made = transposed.contiguous();
      expect(made.isContiguous, isTrue);
    });
  });

  // ============================================================
  // Stride-Based Data Access Tests
  // ============================================================
  group('Data Access with Strides', () {
    test('correct value access after transpose', () {
      // Verify correct value access after transpose
      // [0,1,2,3,4,5] → 2x3 matrix → transpose → 3x2 matrix
      final data = Float32List.fromList(
          List.generate(6, (i) => i.toDouble()));
      final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);
      // [[0, 1, 2], [3, 4, 5]]

      final transposed = tensor.transpose([1, 0]);
      // [[0, 3], [1, 4], [2, 5]]

      expect(transposed[[0, 0]], equals(0));
      expect(transposed[[0, 1]], equals(3));
      expect(transposed[[1, 0]], equals(1));
      expect(transposed[[1, 1]], equals(4));
      expect(transposed[[2, 0]], equals(2));
      expect(transposed[[2, 1]], equals(5));
    });

    test('correct value access with 4D NCHW to NHWC', () {
      // Value access after 4D NCHW → NHWC conversion
      // [N=1, C=2, H=2, W=2] tensor
      final data = Float32List.fromList(
          List.generate(8, (i) => i.toDouble()));
      final nchw = TensorBuffer.fromFloat32List(data, [1, 2, 2, 2]);

      final nhwc = nchw.transpose([0, 2, 3, 1]);

      // NCHW[0,0,0,0] = NHWC[0,0,0,0]
      expect(nhwc[[0, 0, 0, 0]], equals(nchw[[0, 0, 0, 0]]));
      // NCHW[0,1,0,0] = NHWC[0,0,0,1]
      expect(nhwc[[0, 0, 0, 1]], equals(nchw[[0, 1, 0, 0]]));
      // NCHW[0,0,1,1] = NHWC[0,1,1,0]
      expect(nhwc[[0, 1, 1, 0]], equals(nchw[[0, 0, 1, 1]]));
    });

    test('non-contiguous tensor iteration via indices', () {
      // Index-based traversal of non-contiguous tensor
      final data = Float32List.fromList(
          List.generate(12, (i) => i.toDouble()));
      final tensor = TensorBuffer.fromFloat32List(data, [3, 4]);
      final transposed = tensor.transpose([1, 0]); // [4, 3]

      // Collect values in transposed order
      final values = <double>[];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
          values.add(transposed[[i, j]]);
        }
      }

      // Transposed traversal order: column-major
      expect(values, equals([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]));
    });

    test('channels last memory format data access', () {
      // Data access with channelsLast memory format
      final tensor = TensorBuffer.zeros(
        [1, 3, 2, 2],
        memoryFormat: MemoryFormat.channelsLast,
      );

      // Verify C is fastest changing stride pattern
      expect(tensor.strides[1], equals(1)); // Channel stride = 1
    });
  });

  // ============================================================
  // Edge Case Tests
  // ============================================================
  group('Edge Cases', () {
    test('single element tensor layout', () {
      // Single element tensor
      final tensor = TensorBuffer.ones([1, 1, 1, 1]);
      expect(tensor.isContiguous, isTrue);
      expect(tensor.strides, equals([1, 1, 1, 1]));
    });

    test('batch size 1 NCHW', () {
      // Batch size 1 NCHW tensor
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      expect(tensor.strides[0], equals(3 * 224 * 224)); // 150528
    });

    test('empty batch handling', () {
      // Batch 1 with all dimensions 1
      final tensor = TensorBuffer.zeros([1, 1, 1, 1]);
      final squeezed = tensor.squeeze();
      expect(squeezed.shape, isEmpty);
    });

    test('very high dimensional tensor (6D)', () {
      // 6D tensor operations
      final tensor = TensorBuffer.zeros([2, 2, 2, 2, 2, 2]);
      final permuted = tensor.transpose([5, 4, 3, 2, 1, 0]);

      expect(permuted.shape, equals([2, 2, 2, 2, 2, 2]));
      expect(identical(permuted.storage, tensor.storage), isTrue);
    });

    test('multiple size-1 dimensions with channelsLast', () {
      // Multiple size-1 dimensions with channelsLast
      final tensor = TensorBuffer.zeros(
        [1, 1, 224, 224],
        memoryFormat: MemoryFormat.channelsLast,
      );

      // C=1 still follows NHWC stride pattern
      expect(tensor.strides[1], equals(1)); // C stride
    });
  });

  // ============================================================
  // Tensor Creation with Memory Format Tests
  // ============================================================
  group('Tensor Creation with Memory Format', () {
    test('zeros with contiguous format', () {
      // Create zeros with contiguous format
      final tensor = TensorBuffer.zeros(
        [1, 3, 224, 224],
        memoryFormat: MemoryFormat.contiguous,
      );

      expect(tensor.memoryFormat, equals(MemoryFormat.contiguous));
      // NCHW strides: [150528, 50176, 224, 1]
      expect(tensor.strides, equals([150528, 50176, 224, 1]));
    });

    test('zeros with channelsLast format', () {
      // Create zeros with channelsLast format
      final tensor = TensorBuffer.zeros(
        [1, 3, 224, 224],
        memoryFormat: MemoryFormat.channelsLast,
      );

      expect(tensor.memoryFormat, equals(MemoryFormat.channelsLast));
      // NHWC strides: [150528, 1, 672, 3]
      expect(tensor.strides, equals([150528, 1, 672, 3]));
    });

    test('different dtypes with memory format', () {
      // Different dtype and memory format combinations
      final f32 = TensorBuffer.zeros(
        [1, 3, 4, 4],
        dtype: DType.float32,
        memoryFormat: MemoryFormat.channelsLast,
      );
      expect(f32.strides[1], equals(1)); // C is fastest

      final u8 = TensorBuffer.zeros(
        [1, 3, 4, 4],
        dtype: DType.uint8,
        memoryFormat: MemoryFormat.channelsLast,
      );
      expect(u8.strides[1], equals(1)); // Same stride pattern
    });
  });
}
