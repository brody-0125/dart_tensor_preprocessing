import 'package:dart_tensor_preprocessing/src/utils/tensor_indexing.dart';
import 'package:test/test.dart';

void main() {
  group('TensorIndexer', () {
    group('index4D', () {
      test('computes correct index for NCHW tensor', () {
        // Shape [2, 3, 4, 5] → strides [60, 20, 5, 1]
        // Element at [1, 2, 3, 4]
        const C = 3, H = 4, W = 5;
        final idx = TensorIndexer.index4D(1, 2, 3, 4, C, H, W);

        // Manual: 1*60 + 2*20 + 3*5 + 4 = 60 + 40 + 15 + 4 = 119
        expect(idx, 119);
      });

      test('index at origin is 0', () {
        final idx = TensorIndexer.index4D(0, 0, 0, 0, 3, 4, 5);
        expect(idx, 0);
      });

      test('index at last position is numel-1', () {
        const N = 2, C = 3, H = 4, W = 5;
        final idx = TensorIndexer.index4D(N - 1, C - 1, H - 1, W - 1, C, H, W);
        expect(idx, N * C * H * W - 1);
      });
    });

    group('index3D', () {
      test('computes correct index for CHW tensor', () {
        // Shape [3, 4, 5] → strides [20, 5, 1]
        const H = 4, W = 5;
        final idx = TensorIndexer.index3D(2, 3, 4, H, W);

        // Manual: 2*20 + 3*5 + 4 = 40 + 15 + 4 = 59
        expect(idx, 59);
      });
    });

    group('index2D', () {
      test('computes correct index for HW tensor', () {
        const W = 5;
        final idx = TensorIndexer.index2D(3, 4, W);

        // Manual: 3*5 + 4 = 19
        expect(idx, 19);
      });

      test('handles single row', () {
        expect(TensorIndexer.index2D(0, 3, 10), 3);
      });

      test('handles single column', () {
        expect(TensorIndexer.index2D(5, 0, 1), 5);
      });
    });

    group('linearToCoords', () {
      test('converts linear index to 3D coords', () {
        // Shape [2, 3, 4], strides [12, 4, 1]
        // Index 15 = 1*12 + 0*4 + 3 → [1, 0, 3]
        final coords = TensorIndexer.linearToCoords(15, [2, 3, 4]);
        expect(coords, [1, 0, 3]);
      });

      test('handles index 0', () {
        final coords = TensorIndexer.linearToCoords(0, [2, 3, 4]);
        expect(coords, [0, 0, 0]);
      });

      test('handles last index', () {
        // Shape [2, 3, 4], numel = 24, last = 23
        // 23 = 1*12 + 2*4 + 3 → [1, 2, 3]
        final coords = TensorIndexer.linearToCoords(23, [2, 3, 4]);
        expect(coords, [1, 2, 3]);
      });

      test('handles 1D shape', () {
        final coords = TensorIndexer.linearToCoords(5, [10]);
        expect(coords, [5]);
      });

      test('handles 4D shape', () {
        // Shape [2, 3, 4, 5], strides [60, 20, 5, 1]
        // Index 119 = 1*60 + 2*20 + 3*5 + 4 → [1, 2, 3, 4]
        final coords = TensorIndexer.linearToCoords(119, [2, 3, 4, 5]);
        expect(coords, [1, 2, 3, 4]);
      });
    });

    group('coordsToLinear', () {
      test('converts 3D coords to linear index', () {
        // Strides [12, 4, 1] for shape [2, 3, 4]
        final idx = TensorIndexer.coordsToLinear([1, 2, 3], [12, 4, 1]);

        // Manual: 1*12 + 2*4 + 3*1 = 23
        expect(idx, 23);
      });

      test('handles origin', () {
        final idx = TensorIndexer.coordsToLinear([0, 0, 0], [12, 4, 1]);
        expect(idx, 0);
      });

      test('handles non-contiguous strides', () {
        // Transposed tensor: logical shape [4, 3], strides [1, 4]
        // Coords [2, 1] → physical index 2*1 + 1*4 = 6
        final idx = TensorIndexer.coordsToLinear([2, 1], [1, 4]);
        expect(idx, 6);
      });
    });

    group('computeStrides', () {
      test('computes contiguous strides for 3D shape', () {
        final strides = TensorIndexer.computeStrides([2, 3, 4]);
        expect(strides, [12, 4, 1]);
      });

      test('computes contiguous strides for 4D shape', () {
        final strides = TensorIndexer.computeStrides([2, 3, 4, 5]);
        expect(strides, [60, 20, 5, 1]);
      });

      test('computes strides for 1D shape', () {
        final strides = TensorIndexer.computeStrides([10]);
        expect(strides, [1]);
      });

      test('computes strides for 2D shape', () {
        final strides = TensorIndexer.computeStrides([3, 4]);
        expect(strides, [4, 1]);
      });

      test('handles empty shape', () {
        final strides = TensorIndexer.computeStrides([]);
        expect(strides, []);
      });
    });

    group('index4DNHWC', () {
      test('computes correct index for NHWC tensor', () {
        // Shape [2, 4, 5, 3] (N,H,W,C) → strides [60, 15, 3, 1]
        // Element at [1, 2, 3, 2]
        const H = 4, W = 5, C = 3;
        final idx = TensorIndexer.index4DNHWC(1, 2, 3, 2, H, W, C);

        // Manual: 1*60 + 2*15 + 3*3 + 2 = 60 + 30 + 9 + 2 = 101
        expect(idx, 101);
      });

      test('produces different index than NCHW for same logical position', () {
        // Same element position, different layout
        // Choose values that definitely produce different indices
        const b = 0, c = 1, h = 2, w = 3;
        const C = 3, H = 4, W = 5;

        final nchwIdx = TensorIndexer.index4D(b, c, h, w, C, H, W);
        // NCHW: 0*60 + 1*20 + 2*5 + 3 = 33
        final nhwcIdx = TensorIndexer.index4DNHWC(b, h, w, c, H, W, C);
        // NHWC: 0*60 + 2*15 + 3*3 + 1 = 40

        expect(nchwIdx, 33);
        expect(nhwcIdx, 40);
        expect(nchwIdx, isNot(equals(nhwcIdx)));
      });
    });

    group('index3DHWC', () {
      test('computes correct index for HWC tensor', () {
        // Shape [4, 5, 3] (H,W,C) → strides [15, 3, 1]
        const W = 5, C = 3;
        final idx = TensorIndexer.index3DHWC(2, 3, 2, W, C);

        // Manual: 2*15 + 3*3 + 2 = 30 + 9 + 2 = 41
        expect(idx, 41);
      });
    });

    group('round-trip consistency', () {
      test('linearToCoords and coordsToLinear are inverse operations', () {
        final shape = [2, 3, 4, 5];
        final strides = TensorIndexer.computeStrides(shape);
        final numel = shape.reduce((a, b) => a * b);

        for (int i = 0; i < numel; i++) {
          final coords = TensorIndexer.linearToCoords(i, shape);
          final backToLinear = TensorIndexer.coordsToLinear(coords, strides);
          expect(backToLinear, i);
        }
      });

      test('index4D matches manual stride computation', () {
        final shape = [2, 3, 4, 5];
        final strides = TensorIndexer.computeStrides(shape);

        for (int b = 0; b < shape[0]; b++) {
          for (int c = 0; c < shape[1]; c++) {
            for (int h = 0; h < shape[2]; h++) {
              for (int w = 0; w < shape[3]; w++) {
                final fromIndex4D = TensorIndexer.index4D(
                    b, c, h, w, shape[1], shape[2], shape[3]);
                final fromCoords =
                    TensorIndexer.coordsToLinear([b, c, h, w], strides);
                expect(fromIndex4D, fromCoords);
              }
            }
          }
        }
      });
    });
  });
}
