import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

void main() {
  group('TopKOp', () {
    group('basic 1D', () {
      test('selects top-3 largest from 1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9, 2, 6]),
          [8],
        );
        final (values, indices) = TopKOp(k: 3).applyTopK(tensor);

        expect(values.shape, equals([3]));
        expect(indices.shape, equals([3]));
        expect(indices.dtype, equals(DType.int64));

        // sorted descending: 9, 6, 5
        expect(values[[0]], closeTo(9.0, 1e-6));
        expect(values[[1]], closeTo(6.0, 1e-6));
        expect(values[[2]], closeTo(5.0, 1e-6));

        // original indices: 9 at 5, 6 at 7, 5 at 4
        expect(indices[[0]], equals(5));
        expect(indices[[1]], equals(7));
        expect(indices[[2]], equals(4));
      });

      test('selects top-3 smallest from 1D tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5, 9, 2, 6]),
          [8],
        );
        final (values, indices) = TopKOp(
          k: 3,
          largest: false,
        ).applyTopK(tensor);

        expect(values.shape, equals([3]));

        // sorted ascending: 1, 1, 2
        expect(values[[0]], closeTo(1.0, 1e-6));
        expect(values[[1]], closeTo(1.0, 1e-6));
        expect(values[[2]], closeTo(2.0, 1e-6));
      });
    });

    group('2D tensor', () {
      test('topk along axis 1 (per row)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            3, 1, 4, 1, 5, // row 0
            9, 2, 6, 5, 3, // row 1
          ]),
          [2, 5],
        );
        final (values, indices) = TopKOp(k: 2, axis: 1).applyTopK(tensor);

        expect(values.shape, equals([2, 2]));
        expect(indices.shape, equals([2, 2]));

        // Row 0: top-2 are 5 (idx=4), 4 (idx=2)
        expect(values[[0, 0]], closeTo(5.0, 1e-6));
        expect(values[[0, 1]], closeTo(4.0, 1e-6));
        expect(indices[[0, 0]], equals(4));
        expect(indices[[0, 1]], equals(2));

        // Row 1: top-2 are 9 (idx=0), 6 (idx=2)
        expect(values[[1, 0]], closeTo(9.0, 1e-6));
        expect(values[[1, 1]], closeTo(6.0, 1e-6));
        expect(indices[[1, 0]], equals(0));
        expect(indices[[1, 1]], equals(2));
      });

      test('topk along axis 0 (per column)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            3, 1, 4, // row 0
            9, 2, 6, // row 1
            5, 8, 0, // row 2
          ]),
          [3, 3],
        );
        final (values, indices) = TopKOp(k: 2, axis: 0).applyTopK(tensor);

        expect(values.shape, equals([2, 3]));

        // Col 0: top-2 are 9 (idx=1), 5 (idx=2)
        expect(values[[0, 0]], closeTo(9.0, 1e-6));
        expect(values[[1, 0]], closeTo(5.0, 1e-6));
        expect(indices[[0, 0]], equals(1));
        expect(indices[[1, 0]], equals(2));

        // Col 1: top-2 are 8 (idx=2), 2 (idx=1)
        expect(values[[0, 1]], closeTo(8.0, 1e-6));
        expect(values[[1, 1]], closeTo(2.0, 1e-6));
        expect(indices[[0, 1]], equals(2));
        expect(indices[[1, 1]], equals(1));

        // Col 2: top-2 are 6 (idx=1), 4 (idx=0)
        expect(values[[0, 2]], closeTo(6.0, 1e-6));
        expect(values[[1, 2]], closeTo(4.0, 1e-6));
        expect(indices[[0, 2]], equals(1));
        expect(indices[[1, 2]], equals(0));
      });
    });

    group('3D tensor', () {
      test('topk along last axis with negative index', () {
        // Shape [2, 2, 4]
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            // batch 0
            1, 4, 2, 3, // [0, 0, :]
            8, 5, 7, 6, // [0, 1, :]
            // batch 1
            10, 20, 30, 40, // [1, 0, :]
            44, 33, 22, 11, // [1, 1, :]
          ]),
          [2, 2, 4],
        );
        final (values, indices) = TopKOp(k: 2, axis: -1).applyTopK(tensor);

        expect(values.shape, equals([2, 2, 2]));

        // [0, 0, :] = [1, 4, 2, 3] -> top-2: 4 (idx=1), 3 (idx=3)
        expect(values[[0, 0, 0]], closeTo(4.0, 1e-6));
        expect(values[[0, 0, 1]], closeTo(3.0, 1e-6));
        expect(indices[[0, 0, 0]], equals(1));
        expect(indices[[0, 0, 1]], equals(3));

        // [1, 0, :] = [10, 20, 30, 40] -> top-2: 40 (idx=3), 30 (idx=2)
        expect(values[[1, 0, 0]], closeTo(40.0, 1e-6));
        expect(values[[1, 0, 1]], closeTo(30.0, 1e-6));
        expect(indices[[1, 0, 0]], equals(3));
        expect(indices[[1, 0, 1]], equals(2));
      });
    });

    group('sorted vs unsorted', () {
      test('sorted=true returns descending order for largest', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [6],
        );
        final (values, _) = TopKOp(k: 4, sorted: true).applyTopK(tensor);

        // Should be: 9, 8, 5, 3
        expect(values[[0]], closeTo(9.0, 1e-6));
        expect(values[[1]], closeTo(8.0, 1e-6));
        expect(values[[2]], closeTo(5.0, 1e-6));
        expect(values[[3]], closeTo(3.0, 1e-6));
      });

      test('sorted=true returns ascending order for smallest', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [6],
        );
        final (values, _) = TopKOp(
          k: 4,
          largest: false,
          sorted: true,
        ).applyTopK(tensor);

        // Should be: 1, 2, 3, 5
        expect(values[[0]], closeTo(1.0, 1e-6));
        expect(values[[1]], closeTo(2.0, 1e-6));
        expect(values[[2]], closeTo(3.0, 1e-6));
        expect(values[[3]], closeTo(5.0, 1e-6));
      });

      test('sorted=false returns correct values (order not guaranteed)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 2, 8, 1, 9, 3]),
          [6],
        );
        final (values, indices) = TopKOp(k: 3, sorted: false).applyTopK(tensor);

        // Collect returned values
        final returnedValues = <double>{};
        for (int i = 0; i < 3; i++) {
          returnedValues.add(values[[i]]);
        }

        // Top-3 largest are 9, 8, 5
        expect(returnedValues, containsAll([9.0, 8.0, 5.0]));

        // Indices should be valid
        for (int i = 0; i < 3; i++) {
          final idx = indices[[i]].toInt();
          expect(idx, greaterThanOrEqualTo(0));
          expect(idx, lessThan(6));
        }
      });
    });

    group('edge cases', () {
      test('k == axis size (select all)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4]),
          [3],
        );
        final (values, indices) = TopKOp(k: 3).applyTopK(tensor);

        expect(values.shape, equals([3]));

        // All values present, sorted descending
        expect(values[[0]], closeTo(4.0, 1e-6));
        expect(values[[1]], closeTo(3.0, 1e-6));
        expect(values[[2]], closeTo(1.0, 1e-6));
      });

      test('k == 1 (equivalent to argmax)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 7, 2]),
          [4],
        );
        final (values, indices) = TopKOp(k: 1).applyTopK(tensor);

        expect(values.shape, equals([1]));
        expect(values[[0]], closeTo(7.0, 1e-6));
        expect(indices[[0]], equals(2));
      });

      test('k == 1 smallest (equivalent to argmin)', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 7, 2]),
          [4],
        );
        final (values, indices) = TopKOp(
          k: 1,
          largest: false,
        ).applyTopK(tensor);

        expect(values[[0]], closeTo(1.0, 1e-6));
        expect(indices[[0]], equals(1));
      });

      test('handles duplicate values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([5, 5, 5, 1, 2]),
          [5],
        );
        final (values, indices) = TopKOp(k: 3).applyTopK(tensor);

        // All three 5s should be selected
        for (int i = 0; i < 3; i++) {
          expect(values[[i]], closeTo(5.0, 1e-6));
        }

        // Indices should be valid and unique
        final idxSet = <int>{};
        for (int i = 0; i < 3; i++) {
          idxSet.add(indices[[i]].toInt());
        }
        expect(idxSet.length, equals(3));
        expect(idxSet, everyElement(isIn([0, 1, 2])));
      });
    });

    group('validation', () {
      test('throws on k <= 0', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        expect(
          () => TopKOp(k: 0).applyTopK(tensor),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on k > axis size', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        expect(
          () => TopKOp(k: 5).applyTopK(tensor),
          throwsA(isA<InvalidParameterException>()),
        );
      });

      test('throws on axis out of range', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        expect(
          () => TopKOp(k: 1, axis: 2).applyTopK(tensor),
          throwsA(isA<IndexOutOfBoundsException>()),
        );
      });

      test('throws on negative axis out of range', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1, 2, 3]),
          [3],
        );
        expect(
          () => TopKOp(k: 1, axis: -2).applyTopK(tensor),
          throwsA(isA<IndexOutOfBoundsException>()),
        );
      });
    });

    group('dtype support', () {
      test('float64 input produces float64 values and int64 indices', () {
        final tensor = TensorBuffer(
          storage: TensorStorage(
            Float64List.fromList([3, 1, 4, 1, 5]),
            DType.float64,
          ),
          shape: [5],
        );
        final (values, indices) = TopKOp(k: 2).applyTopK(tensor);

        expect(values.dtype, equals(DType.float64));
        expect(indices.dtype, equals(DType.int64));
        expect(values[[0]], closeTo(5.0, 1e-6));
        expect(values[[1]], closeTo(4.0, 1e-6));
      });

      test('float32 input produces float32 values and int64 indices', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5]),
          [5],
        );
        final (values, indices) = TopKOp(k: 2).applyTopK(tensor);

        expect(values.dtype, equals(DType.float32));
        expect(indices.dtype, equals(DType.int64));
      });
    });

    group('computeOutputShape', () {
      test('replaces axis dimension with k', () {
        final op = TopKOp(k: 3, axis: 1);
        expect(op.computeOutputShape([2, 5, 4]), equals([2, 3, 4]));
      });

      test('negative axis works', () {
        final op = TopKOp(k: 2, axis: -1);
        expect(op.computeOutputShape([2, 5, 4]), equals([2, 5, 2]));
      });

      test('axis 0', () {
        final op = TopKOp(k: 1, axis: 0);
        expect(op.computeOutputShape([3, 4]), equals([1, 4]));
      });
    });

    group('extension method', () {
      test('tensor.topk works', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5]),
          [5],
        );
        final (values, indices) = tensor.topk(2);

        expect(values.shape, equals([2]));
        expect(values[[0]], closeTo(5.0, 1e-6));
        expect(values[[1]], closeTo(4.0, 1e-6));
        expect(indices[[0]], equals(4));
        expect(indices[[1]], equals(2));
      });

      test('extension forwards named parameters', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            3, 1, 4, // row 0
            1, 5, 9, // row 1
          ]),
          [2, 3],
        );
        final (values, indices) = tensor.topk(
          2,
          axis: 1,
          largest: false,
          sorted: true,
        );

        expect(values.shape, equals([2, 2]));
        // Row 0 smallest 2: 1 (idx=1), 3 (idx=0)
        expect(values[[0, 0]], closeTo(1.0, 1e-6));
        expect(values[[0, 1]], closeTo(3.0, 1e-6));
        // Row 1 smallest 2: 1 (idx=0), 5 (idx=1)
        expect(values[[1, 0]], closeTo(1.0, 1e-6));
        expect(values[[1, 1]], closeTo(5.0, 1e-6));
      });
    });

    group('pipeline compatibility', () {
      test('apply returns only values tensor', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([3, 1, 4, 1, 5]),
          [5],
        );
        final result = TopKOp(k: 2).apply(tensor);

        expect(result, isA<TensorBuffer>());
        expect(result.shape, equals([2]));
        expect(result[[0]], closeTo(5.0, 1e-6));
        expect(result[[1]], closeTo(4.0, 1e-6));
      });

      test('works in TensorPipeline', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            3, 1, 4, 1, 5, // row 0
            9, 2, 6, 5, 3, // row 1
          ]),
          [2, 5],
        );
        final pipeline = TensorPipeline([TopKOp(k: 3, axis: 1)]);
        final result = pipeline.run(tensor);

        expect(result.shape, equals([2, 3]));
      });
    });

    group('index consistency', () {
      test('returned indices can reconstruct values', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([
            3, 1, 4, 1, 5, // row 0
            9, 2, 6, 5, 3, // row 1
          ]),
          [2, 5],
        );
        final (values, indices) = TopKOp(k: 3, axis: 1).applyTopK(tensor);

        // For each row, verify that input[row][indices[row][i]] == values[row][i]
        for (int row = 0; row < 2; row++) {
          for (int i = 0; i < 3; i++) {
            final idx = indices[[row, i]].toInt();
            expect(
              tensor[[row, idx]],
              closeTo(values[[row, i]], 1e-6),
              reason:
                  'Row $row, topk position $i: '
                  'input[row][$idx] should equal values[row][$i]',
            );
          }
        }
      });
    });

    group('capabilities', () {
      test('has correct capabilities', () {
        final op = TopKOp(k: 3);
        expect(op.capabilities.preservesShape, isFalse);
        expect(op.capabilities.pytorchEquivalent, equals('torch.topk'));
        expect(op.capabilities.onnxOpType, equals('TopK'));
        expect(op.capabilities.onnxOpsetVersion, equals(1));
      });

      test('name is TopK', () {
        expect(TopKOp(k: 1).name, equals('TopK'));
      });
    });
  });
}
