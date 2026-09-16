import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  for (final dtype in [DType.float32, DType.float64, DType.int32]) {
    test('storage kernels respect offset and length ($dtype)', () async {
      final base = TensorBuffer.full([8], fillValue: -99, dtype: dtype);
      for (var i = 0; i < 4; i++) {
        base.storage.setFromDouble(i + 2, (i - 2).toDouble());
      }
      final view = TensorBuffer(
        storage: base.storage,
        shape: [2, 2],
        storageOffset: 2,
      );
      final indices = TensorBuffer(
        storage: TensorStorage(
          Int64List.fromList([9, 1, 0, 0, 1, 9]),
          DType.int64,
        ),
        shape: [2, 2],
        storageOffset: 1,
      );
      final mask = TensorBuffer(
        storage: TensorStorage(
          Uint8List.fromList([9, 1, 0, 0, 1, 9]),
          DType.uint8,
        ),
        shape: [2, 2],
        storageOffset: 1,
      );
      expect(GatherOp(dim: 1, index: indices)(view).toList(), [-1, -2, 0, 1]);
      expect(TopKOp(k: 1)(view).toList(), [-1, 1]);
      expect(stack([view, view]).toList(), [-2, -1, 0, 1, -2, -1, 0, 1]);
      expect(concat([view, view], axis: 0).toList(), [
        -2,
        -1,
        0,
        1,
        -2,
        -1,
        0,
        1,
      ]);
      expect(split(view, [1, 1], dim: 0).map((t) => t.toList()), [
        [-2, -1],
        [0, 1],
      ]);
      expect(
        tensorWhere(
          mask,
          view,
          TensorBuffer.ones([2, 2], dtype: dtype),
        ).toList(),
        [-2, 1, 1, 1],
      );
      MaskedFillOp(mask: mask, value: 8).applyInPlace(view);
      expect(base.toList(), [-99, -99, 8, -1, 0, 8, -99, -99]);
      ReLUOp().applyInPlace(view);
      expect(base.toList(), [-99, -99, 8, 0, 0, 8, -99, -99]);
      final result = await TensorPipeline([
        IdentityOp(),
      ]).runAsync(view, isolateThreshold: 0);
      expect(result.toList(), [8, 0, 0, 8]);
      expect(result.shape, [2, 2]);
    });
  }
  test(
    'RGB preset rejects unsupported channels and partial normalization config',
    () {
      for (final channels in [1, 4]) {
        expect(
          () => PipelinePresets.minimal().run(
            TensorBuffer.zeros([4, 5, channels]),
          ),
          throwsA(isA<ShapeMismatchException>()),
        );
      }
      expect(
        () => PipelinePresets.custom(height: 2, width: 2, mean: [0.5]),
        throwsA(isA<InvalidParameterException>()),
      );
    },
  );
}
