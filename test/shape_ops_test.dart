/// Shape operations tests based on PyTorch test_shape_ops.py patterns
///
/// Tests cover:
/// - PermuteOp, UnsqueezeOp, SqueezeOp
/// - ReshapeOp with -1 dimension
/// - FlattenOp
/// - LayoutConvertOp
/// - Memory format conversions (NCHW <-> NHWC)
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('PermuteOp', () {
    test('permutes NCHW to NHWC', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final op = PermuteOp.nchwToNhwc();
      final result = op(tensor);

      expect(result.shape, equals([2, 4, 5, 3]));
    });

    test('permutes NHWC to NCHW', () {
      final tensor = TensorBuffer.zeros([2, 4, 5, 3]);
      final op = PermuteOp.nhwcToNchw();
      final result = op(tensor);

      expect(result.shape, equals([2, 3, 4, 5]));
    });

    test('permutes CHW to HWC (3D)', () {
      final tensor = TensorBuffer.zeros([3, 224, 224]);
      final op = PermuteOp.chwToHwc();
      final result = op(tensor);

      expect(result.shape, equals([224, 224, 3]));
    });

    test('permutes HWC to CHW (3D)', () {
      final tensor = TensorBuffer.zeros([224, 224, 3]);
      final op = PermuteOp.hwcToChw();
      final result = op(tensor);

      expect(result.shape, equals([3, 224, 224]));
    });

    test('custom permutation', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final op = PermuteOp([3, 1, 0, 2]);
      final result = op(tensor);

      expect(result.shape, equals([5, 3, 2, 4]));
    });

    test('permute is zero-copy', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = PermuteOp([2, 0, 1]);
      final result = op(tensor);

      expect(identical(result.storage, tensor.storage), isTrue);
    });

    test('double permute restores shape', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final result = PermuteOp.nchwToNhwc()(PermuteOp.nhwcToNchw()(tensor));

      expect(result.shape, equals([2, 3, 4, 5]));
    });

    test('throws on rank mismatch', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = PermuteOp([0, 1]); // Wrong length

      expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
    });

    test('computeOutputShape', () {
      final op = PermuteOp([2, 0, 1]);

      expect(op.computeOutputShape([2, 3, 4]), equals([4, 2, 3]));
    });
  });

  group('UnsqueezeOp', () {
    test('adds batch dimension', () {
      final tensor = TensorBuffer.zeros([3, 224, 224]);
      final op = UnsqueezeOp.batch();
      final result = op(tensor);

      expect(result.shape, equals([1, 3, 224, 224]));
    });

    test('adds dimension at various positions', () {
      final tensor = TensorBuffer.zeros([3, 4, 5]);

      expect(UnsqueezeOp(0)(tensor).shape, equals([1, 3, 4, 5]));
      expect(UnsqueezeOp(1)(tensor).shape, equals([3, 1, 4, 5]));
      expect(UnsqueezeOp(2)(tensor).shape, equals([3, 4, 1, 5]));
      expect(UnsqueezeOp(3)(tensor).shape, equals([3, 4, 5, 1]));
    });

    test('is zero-copy operation', () {
      final tensor = TensorBuffer.zeros([3, 4]);
      final result = UnsqueezeOp(0)(tensor);

      expect(identical(result.storage, tensor.storage), isTrue);
    });

    test('computeOutputShape', () {
      final op = UnsqueezeOp(1);

      expect(op.computeOutputShape([3, 4, 5]), equals([3, 1, 4, 5]));
    });

    test('computeOutputShape at end', () {
      final op = UnsqueezeOp(3);

      expect(op.computeOutputShape([3, 4, 5]), equals([3, 4, 5, 1]));
    });
  });

  group('SqueezeOp', () {
    test('removes batch dimension', () {
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final op = SqueezeOp.batch();
      final result = op(tensor);

      expect(result.shape, equals([3, 224, 224]));
    });

    test('removes all size-1 dimensions', () {
      final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1]);
      final op = SqueezeOp.all();
      final result = op(tensor);

      expect(result.shape, equals([3, 224]));
    });

    test('squeeze specific dimension', () {
      final tensor = TensorBuffer.zeros([1, 3, 1, 224]);

      expect(SqueezeOp(0)(tensor).shape, equals([3, 1, 224]));
      expect(SqueezeOp(2)(tensor).shape, equals([1, 3, 224]));
    });

    test('squeeze non-1 dimension does nothing', () {
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final result = SqueezeOp(1)(tensor);

      expect(result.shape, equals([1, 3, 224, 224]));
    });

    test('is zero-copy operation', () {
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final result = SqueezeOp.batch()(tensor);

      expect(identical(result.storage, tensor.storage), isTrue);
    });

    test('computeOutputShape', () {
      expect(SqueezeOp.all().computeOutputShape([1, 3, 1, 224]), equals([3, 224]));
      expect(SqueezeOp(0).computeOutputShape([1, 3, 224]), equals([3, 224]));
      expect(SqueezeOp(1).computeOutputShape([1, 3, 224]), equals([1, 3, 224])); // Not size-1
    });
  });

  group('ReshapeOp', () {
    test('reshapes to flat', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = ReshapeOp([24]);
      final result = op(tensor);

      expect(result.shape, equals([24]));
    });

    test('reshapes with -1 infers dimension', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);

      expect(ReshapeOp([-1]).computeOutputShape([2, 3, 4]), equals([24]));
      expect(ReshapeOp([2, -1]).computeOutputShape([2, 3, 4]), equals([2, 12]));
      expect(ReshapeOp([-1, 4]).computeOutputShape([2, 3, 4]), equals([6, 4]));
      expect(ReshapeOp([2, -1, 4]).computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });

    test('multiple -1 throws', () {
      expect(
        () => ReshapeOp([-1, -1]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('zero dimension throws', () {
      expect(
        () => ReshapeOp([2, 0, 3]),
        throwsA(isA<InvalidParameterException>()),
      );
    });

    test('size mismatch with -1 throws', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]); // 24 elements
      final op = ReshapeOp([-1, 5]); // 24 not divisible by 5

      expect(() => op(tensor), throwsA(isA<InvalidParameterException>()));
    });

    test('is zero-copy for contiguous tensor', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final result = ReshapeOp([6, 4])(tensor);

      expect(identical(result.storage, tensor.storage), isTrue);
    });
  });

  group('FlattenOp', () {
    test('flattens from startDim to end', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final op = FlattenOp(startDim: 1);
      final result = op(tensor);

      expect(result.shape, equals([2, 60]));
    });

    test('flattens entire tensor', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = FlattenOp(startDim: 0);
      final result = op(tensor);

      expect(result.shape, equals([24]));
    });

    test('flattens specific range', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final op = FlattenOp(startDim: 1, endDim: 2);
      final result = op(tensor);

      expect(result.shape, equals([2, 12, 5]));
    });

    test('flattens with negative endDim', () {
      final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
      final op = FlattenOp(startDim: 1, endDim: -1);
      final result = op(tensor);

      expect(result.shape, equals([2, 60]));
    });

    test('computeOutputShape', () {
      final op = FlattenOp(startDim: 1);

      expect(op.computeOutputShape([2, 3, 4, 5]), equals([2, 60]));
    });
  });

  group('LayoutConvertOp', () {
    test('converts to NCHW', () {
      // Note: This doesn't change shape, just ensures contiguity
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final transposed = tensor.transpose([0, 2, 3, 1]); // Make NHWC-like
      final op = LayoutConvertOp.toNchw();
      final result = op(transposed);

      expect(result.isContiguous, isTrue);
    });

    test('converts to NHWC', () {
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final op = LayoutConvertOp.toNhwc();
      final result = op(tensor);

      expect(result.shape, equals([1, 224, 224, 3]));
    });

    test('forceContiguous creates copy if needed', () {
      final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
      final op = LayoutConvertOp.toNhwc(forceContiguous: true);
      final result = op(tensor);

      expect(result.isContiguous, isTrue);
    });

    test('throws on non-4D tensor', () {
      final tensor = TensorBuffer.zeros([3, 224, 224]);
      final op = LayoutConvertOp.toNhwc();

      expect(() => op(tensor), throwsA(isA<ShapeMismatchException>()));
    });
  });

  group('ContiguousOp', () {
    test('returns same tensor if already contiguous', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = ContiguousOp();
      final result = op(tensor);

      expect(identical(result, tensor), isTrue);
    });

    test('creates contiguous copy of non-contiguous tensor', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final transposed = tensor.transpose([2, 0, 1]);
      final op = ContiguousOp();
      final result = op(transposed);

      expect(result.isContiguous, isTrue);
      expect(identical(result.storage, tensor.storage), isFalse);
    });
  });

  group('IdentityOp', () {
    test('returns same tensor', () {
      final tensor = TensorBuffer.zeros([2, 3, 4]);
      final op = IdentityOp();
      final result = op(tensor);

      expect(identical(result, tensor), isTrue);
    });

    test('computeOutputShape returns same shape', () {
      final op = IdentityOp();

      expect(op.computeOutputShape([2, 3, 4]), equals([2, 3, 4]));
    });
  });

  group('MemoryFormat Conversions', () {
    test('NCHW to NHWC preserves data', () {
      // Create NCHW tensor with known pattern
      final data = Float32List.fromList(List.generate(24, (i) => i.toDouble()));
      final nchw = TensorBuffer.fromFloat32List(data, [1, 2, 3, 4]);

      // Convert to NHWC
      final nhwc = PermuteOp.nchwToNhwc()(nchw).contiguous();

      expect(nhwc.shape, equals([1, 3, 4, 2]));

      // Verify data is accessible correctly
      // NCHW[0,0,0,0] should equal NHWC[0,0,0,0]
      expect(nhwc[[0, 0, 0, 0]], equals(nchw[[0, 0, 0, 0]]));
    });

    test('round-trip NCHW -> NHWC -> NCHW', () {
      final original = TensorBuffer.ones([1, 3, 224, 224]);

      final nhwc = PermuteOp.nchwToNhwc()(original);
      final back = PermuteOp.nhwcToNchw()(nhwc);

      expect(back.shape, equals([1, 3, 224, 224]));
    });

    test('channels-last strides for image tensor', () {
      final tensor = TensorBuffer.zeros(
        [1, 3, 224, 224],
        memoryFormat: MemoryFormat.channelsLast,
      );

      // In NHWC layout, C has stride 1
      expect(tensor.strides[1], equals(1)); // C dimension
    });
  });

  group('Combined Shape Operations', () {
    test('typical image preprocessing chain', () {
      // HWC uint8 -> CHW float32 -> batch
      final hwc = TensorBuffer.zeros([224, 224, 3], dtype: DType.uint8);

      // Manual permute (HWC to CHW)
      final chw = PermuteOp([2, 0, 1])(hwc);
      expect(chw.shape, equals([3, 224, 224]));

      // Add batch dim
      final batched = UnsqueezeOp.batch()(chw);
      expect(batched.shape, equals([1, 3, 224, 224]));
    });

    test('remove batch then flatten', () {
      final tensor = TensorBuffer.zeros([1, 512, 7, 7]);

      final squeezed = SqueezeOp.batch()(tensor);
      expect(squeezed.shape, equals([512, 7, 7]));

      final flattened = FlattenOp(startDim: 0)(squeezed);
      expect(flattened.shape, equals([25088]));
    });

    test('reshape and permute', () {
      final tensor = TensorBuffer.zeros([768]);

      // Reshape to 3D
      final reshaped = ReshapeOp([3, 16, 16])(tensor);
      expect(reshaped.shape, equals([3, 16, 16]));

      // Permute to HWC
      final hwc = PermuteOp.chwToHwc()(reshaped);
      expect(hwc.shape, equals([16, 16, 3]));
    });
  });
}
