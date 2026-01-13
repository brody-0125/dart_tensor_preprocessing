import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('DTypeDispatcher', () {
    group('dispatch', () {
      test('calls onFloat32 for float32 tensor', () {
        final tensor = TensorBuffer.zeros([2, 2], dtype: DType.float32);
        String called = '';

        final result = DTypeDispatcher.dispatch<String>(
          tensor,
          onFloat32: (data, numel) {
            called = 'float32';
            expect(data, isA<Float32List>());
            expect(numel, 4);
            return 'float32-result';
          },
          onFloat64: (data, numel) {
            called = 'float64';
            return 'float64-result';
          },
          fallback: (t) {
            called = 'fallback';
            return 'fallback-result';
          },
        );

        expect(called, 'float32');
        expect(result, 'float32-result');
      });

      test('calls onFloat64 for float64 tensor', () {
        final tensor = TensorBuffer.zeros([2, 3], dtype: DType.float64);
        String called = '';

        final result = DTypeDispatcher.dispatch<String>(
          tensor,
          onFloat32: (data, numel) {
            called = 'float32';
            return 'float32-result';
          },
          onFloat64: (data, numel) {
            called = 'float64';
            expect(data, isA<Float64List>());
            expect(numel, 6);
            return 'float64-result';
          },
          fallback: (t) {
            called = 'fallback';
            return 'fallback-result';
          },
        );

        expect(called, 'float64');
        expect(result, 'float64-result');
      });

      test('calls fallback for int64 tensor', () {
        final tensor = TensorBuffer.zeros([2, 2], dtype: DType.int64);
        String called = '';

        final result = DTypeDispatcher.dispatch<String>(
          tensor,
          onFloat32: (data, numel) {
            called = 'float32';
            return 'float32-result';
          },
          onFloat64: (data, numel) {
            called = 'float64';
            return 'float64-result';
          },
          fallback: (t) {
            called = 'fallback';
            expect(t.numel, 4);
            return 'fallback-result';
          },
        );

        expect(called, 'fallback');
        expect(result, 'fallback-result');
      });

      test('returns computed value from callback', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
        );

        final sum = DTypeDispatcher.dispatch<double>(
          tensor,
          onFloat32: (data, numel) {
            double s = 0;
            for (int i = 0; i < numel; i++) {
              s += data[i];
            }
            return s;
          },
          onFloat64: (data, numel) {
            double s = 0;
            for (int i = 0; i < numel; i++) {
              s += data[i];
            }
            return s;
          },
          fallback: (t) => 0.0,
        );

        expect(sum, 10.0);
      });
    });

    group('dispatchVoid', () {
      test('modifies float32 tensor in-place', () {
        final tensor = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, -2.0, 3.0, -4.0]),
          [4],
        );

        DTypeDispatcher.dispatchVoid(
          tensor,
          onFloat32: (data, numel) {
            for (int i = 0; i < numel; i++) {
              if (data[i] < 0) data[i] = 0;
            }
          },
          onFloat64: (data, numel) {
            for (int i = 0; i < numel; i++) {
              if (data[i] < 0) data[i] = 0;
            }
          },
          fallback: (t) {
            for (int i = 0; i < t.numel; i++) {
              final v = t.storage.getAsDouble(i);
              if (v < 0) t.storage.setFromDouble(i, 0);
            }
          },
        );

        expect(tensor.toList(), [1.0, 0.0, 3.0, 0.0]);
      });

      test('modifies float64 tensor in-place', () {
        final tensor = TensorBuffer.fromFloat64List(
          Float64List.fromList([1.0, -2.0, 3.0, -4.0]),
          [4],
        );

        DTypeDispatcher.dispatchVoid(
          tensor,
          onFloat32: (data, numel) {
            for (int i = 0; i < numel; i++) {
              if (data[i] < 0) data[i] = 0;
            }
          },
          onFloat64: (data, numel) {
            for (int i = 0; i < numel; i++) {
              if (data[i] < 0) data[i] = 0;
            }
          },
          fallback: (t) {
            for (int i = 0; i < t.numel; i++) {
              final v = t.storage.getAsDouble(i);
              if (v < 0) t.storage.setFromDouble(i, 0);
            }
          },
        );

        expect(tensor.toList(), [1.0, 0.0, 3.0, 0.0]);
      });
    });

    group('dispatchPair', () {
      test('handles matching float32 dtypes', () {
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
        );
        final output = TensorBuffer.zeros([4], dtype: DType.float32);
        String called = '';

        DTypeDispatcher.dispatchPair<void>(
          input,
          output,
          onFloat32: (inData, outData, numel) {
            called = 'float32';
            for (int i = 0; i < numel; i++) {
              outData[i] = inData[i] * 2;
            }
          },
          onFloat64: (inData, outData, numel) {
            called = 'float64';
          },
          fallback: (inp, out) {
            called = 'fallback';
          },
        );

        expect(called, 'float32');
        expect(output.toList(), [2.0, 4.0, 6.0, 8.0]);
      });

      test('handles matching float64 dtypes', () {
        final input = TensorBuffer.fromFloat64List(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
        );
        final output = TensorBuffer.zeros([4], dtype: DType.float64);
        String called = '';

        DTypeDispatcher.dispatchPair<void>(
          input,
          output,
          onFloat32: (inData, outData, numel) {
            called = 'float32';
          },
          onFloat64: (inData, outData, numel) {
            called = 'float64';
            for (int i = 0; i < numel; i++) {
              outData[i] = inData[i] * 3;
            }
          },
          fallback: (inp, out) {
            called = 'fallback';
          },
        );

        expect(called, 'float64');
        expect(output.toList(), [3.0, 6.0, 9.0, 12.0]);
      });

      test('falls back on dtype mismatch', () {
        final input = TensorBuffer.zeros([4], dtype: DType.float32);
        final output = TensorBuffer.zeros([4], dtype: DType.float64);
        String called = '';

        DTypeDispatcher.dispatchPair<void>(
          input,
          output,
          onFloat32: (inData, outData, numel) {
            called = 'float32';
          },
          onFloat64: (inData, outData, numel) {
            called = 'float64';
          },
          fallback: (inp, out) {
            called = 'fallback';
            expect(inp.dtype, DType.float32);
            expect(out.dtype, DType.float64);
          },
        );

        expect(called, 'fallback');
      });

      test('falls back for non-float dtypes', () {
        final input = TensorBuffer.zeros([4], dtype: DType.int32);
        final output = TensorBuffer.zeros([4], dtype: DType.int32);
        String called = '';

        DTypeDispatcher.dispatchPair<void>(
          input,
          output,
          onFloat32: (inData, outData, numel) {
            called = 'float32';
          },
          onFloat64: (inData, outData, numel) {
            called = 'float64';
          },
          fallback: (inp, out) {
            called = 'fallback';
          },
        );

        expect(called, 'fallback');
      });

      test('returns value from pair dispatch', () {
        final input = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.0, 3.0]),
          [3],
        );
        final output = TensorBuffer.zeros([3], dtype: DType.float32);

        final result = DTypeDispatcher.dispatchPair<double>(
          input,
          output,
          onFloat32: (inData, outData, numel) {
            double sum = 0;
            for (int i = 0; i < numel; i++) {
              outData[i] = inData[i] * 2;
              sum += outData[i];
            }
            return sum;
          },
          onFloat64: (inData, outData, numel) => 0.0,
          fallback: (inp, out) => -1.0,
        );

        expect(result, 12.0); // 2 + 4 + 6
        expect(output.toList(), [2.0, 4.0, 6.0]);
      });
    });
  });
}
