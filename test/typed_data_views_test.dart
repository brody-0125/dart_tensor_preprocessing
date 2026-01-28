import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('TypedDataViews', () {
    group('float32SublistView', () {
      test('creates zero-copy view', () {
        final data = Float32List.fromList([1, 2, 3, 4, 5]);
        final view = TypedDataViews.float32SublistView(data, 1, 4);

        expect(view.length, equals(3));
        expect(view[0], equals(2));
        expect(view[1], equals(3));
        expect(view[2], equals(4));

        // Verify it's truly a view (modifications reflect in original)
        view[0] = 99;
        expect(data[1], equals(99));
      });
    });

    group('float64SublistView', () {
      test('creates zero-copy view', () {
        final data = Float64List.fromList([1, 2, 3, 4, 5]);
        final view = TypedDataViews.float64SublistView(data, 2);

        expect(view.length, equals(3));
        expect(view[0], equals(3));
      });
    });

    group('viewAs', () {
      test('creates float32 view from buffer', () {
        final buffer = Float32List(10).buffer;
        final view =
            TypedDataViews.viewAs(buffer, DType.float32) as Float32List;

        expect(view.length, equals(10));
      });

      test('creates view at offset', () {
        final original = Float32List.fromList([1, 2, 3, 4, 5]);
        final view = TypedDataViews.viewAs(
          original.buffer,
          DType.float32,
          offsetInBytes: 4, // Skip first float32 (4 bytes)
          length: 3,
        ) as Float32List;

        expect(view.length, equals(3));
        expect(view[0], equals(2));
      });
    });
  });

  group('TensorViewExtension', () {
    group('sliceFirst', () {
      test('creates view of first dimension slice', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList(List.generate(24, (i) => i.toDouble())),
          [4, 2, 3],
        );

        final slice = tensor.sliceFirst(1, 3);

        expect(slice.shape, equals([2, 2, 3]));
        // Check first element of slice (should be element 6 of original)
        expect(slice[[0, 0, 0]], equals(6));
      });

      test('shares storage with original', () {
        final data =
            Float32List.fromList(List.generate(12, (i) => i.toDouble()));
        final tensor = TensorBuffer.fromFloat32List(data, [3, 4]);

        final slice = tensor.sliceFirst(1, 2);

        expect(identical(slice.storage, tensor.storage), isTrue);
      });

      test('throws for non-contiguous tensor', () {
        final tensor = TensorBuffer.zeros([3, 4]).transpose([1, 0]);

        expect(() => tensor.sliceFirst(0, 2), throwsStateError);
      });

      test('throws for invalid range', () {
        final tensor = TensorBuffer.zeros([5, 3]);

        expect(() => tensor.sliceFirst(-1, 3), throwsRangeError);
        expect(() => tensor.sliceFirst(0, 6), throwsRangeError);
        expect(() => tensor.sliceFirst(3, 2), throwsRangeError);
      });
    });

    group('isViewable', () {
      test('returns true for fresh tensor', () {
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(tensor.isViewable, isTrue);
      });

      test('returns false for transposed tensor', () {
        final tensor = TensorBuffer.zeros([3, 4]).transpose([1, 0]);

        expect(tensor.isViewable, isFalse);
      });

      test('returns false for tensor with offset', () {
        final tensor = TensorBuffer.zeros([5, 3]);
        final slice = tensor.sliceFirst(1, 3);

        expect(slice.isViewable, isFalse);
      });
    });

    group('toChannelsLast', () {
      test('converts NCHW to NHWC', () {
        final nchw = TensorBuffer.zeros([2, 3, 4, 5]);

        final nhwc = nchw.toChannelsLast();

        expect(nhwc.shape, equals([2, 4, 5, 3]));
        expect(identical(nhwc.storage, nchw.storage), isTrue);
      });

      test('throws for non-4D tensor', () {
        final tensor = TensorBuffer.zeros([3, 4, 5]);

        expect(() => tensor.toChannelsLast(), throwsStateError);
      });
    });

    group('toChannelsFirst', () {
      test('converts NHWC to NCHW', () {
        final nhwc = TensorBuffer.zeros([2, 4, 5, 3]);

        final nchw = nhwc.toChannelsFirst();

        expect(nchw.shape, equals([2, 3, 4, 5]));
      });
    });

    group('flatten', () {
      test('creates 1D view', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        final flat = tensor.flatten();

        expect(flat.shape, equals([24]));
        expect(flat.numel, equals(24));
      });

      test('shares storage', () {
        final tensor = TensorBuffer.zeros([2, 3]);

        final flat = tensor.flatten();

        expect(identical(flat.storage, tensor.storage), isTrue);
      });

      test('throws for non-contiguous tensor', () {
        final tensor = TensorBuffer.zeros([3, 4]).transpose([1, 0]);

        expect(() => tensor.flatten(), throwsStateError);
      });
    });

    group('unbind', () {
      test('splits tensor into views along dimension', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList(List.generate(12, (i) => i.toDouble())),
          [3, 4],
        );

        final views = tensor.unbind(0);

        expect(views.length, equals(3));
        expect(views[0].shape, equals([4]));
        expect(views[1].shape, equals([4]));
        expect(views[2].shape, equals([4]));

        // Check values
        expect(views[0].toList(), equals([0, 1, 2, 3]));
        expect(views[1].toList(), equals([4, 5, 6, 7]));
        expect(views[2].toList(), equals([8, 9, 10, 11]));
      });

      test('all views share storage', () {
        final tensor = TensorBuffer.zeros([5, 3, 4]);

        final views = tensor.unbind(0);

        for (final view in views) {
          expect(identical(view.storage, tensor.storage), isTrue);
        }
      });

      test('throws for invalid dim', () {
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(() => tensor.unbind(-1), throwsRangeError);
        expect(() => tensor.unbind(2), throwsRangeError);
      });
    });

    group('select', () {
      test('selects single index reducing rank', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList(List.generate(24, (i) => i.toDouble())),
          [2, 3, 4],
        );

        final selected = tensor.select(0, 1);

        expect(selected.shape, equals([3, 4]));
        expect(selected[[0, 0]], equals(12)); // First element of second batch
      });

      test('shares storage', () {
        final tensor = TensorBuffer.zeros([5, 3, 4]);

        final selected = tensor.select(1, 2);

        expect(identical(selected.storage, tensor.storage), isTrue);
      });

      test('throws for invalid dim', () {
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(() => tensor.select(2, 0), throwsRangeError);
      });

      test('throws for invalid index', () {
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(() => tensor.select(0, 5), throwsRangeError);
      });
    });

    group('narrow', () {
      test('narrows dimension', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList(List.generate(20, (i) => i.toDouble())),
          [5, 4],
        );

        final narrowed = tensor.narrow(0, 1, 3);

        expect(narrowed.shape, equals([3, 4]));
        expect(narrowed[[0, 0]], equals(4)); // Row 1 of original
        expect(narrowed[[2, 3]], equals(15)); // Row 3, col 3 of original
      });

      test('shares storage', () {
        final tensor = TensorBuffer.zeros([5, 3, 4]);

        final narrowed = tensor.narrow(0, 1, 2);

        expect(identical(narrowed.storage, tensor.storage), isTrue);
      });

      test('throws for invalid range', () {
        final tensor = TensorBuffer.zeros([5, 4]);

        expect(() => tensor.narrow(0, -1, 2), throwsRangeError);
        expect(() => tensor.narrow(0, 4, 2), throwsRangeError);
      });

      test('throws for invalid dim', () {
        final tensor = TensorBuffer.zeros([3, 4]);

        expect(() => tensor.narrow(2, 0, 1), throwsRangeError);
      });
    });
  });
}
