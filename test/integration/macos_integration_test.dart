@TestOn('mac-os')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

/// macOS-specific integration tests.
///
/// These tests validate tensor operations in the macOS environment,
/// including SIMD behavior, isolate-based async execution, and
/// memory management on macOS.
void main() {
  group('macOS Platform Integration', () {
    test('platform is macOS', () {
      expect(Platform.isMacOS, isTrue);
    });

    group('SIMD operations on macOS', () {
      test('Float32 SIMD multiply scalar', () {
        final data = Float32List.fromList(
          List.generate(1024, (i) => i.toDouble()),
        );
        SimdOps.multiplyScalar(data, 2.0);
        for (int i = 0; i < 1024; i++) {
          expect(data[i], closeTo(i * 2.0, 1e-5));
        }
      });

      test('Float32 SIMD normalize', () {
        final data = Float32List.fromList(
          List.generate(256, (i) => i.toDouble()),
        );
        SimdOps.normalize(data, 128.0, 64.0);
        expect(data[0], closeTo((0 - 128.0) / 64.0, 1e-5));
        expect(data[128], closeTo(0.0, 1e-5));
      });

      test('Float32 SIMD ReLU', () {
        final data = Float32List.fromList(
          List.generate(128, (i) => (i - 64).toDouble()),
        );
        SimdOps.relu(data);
        for (int i = 0; i < 64; i++) {
          expect(data[i], equals(0.0));
        }
        for (int i = 64; i < 128; i++) {
          expect(data[i], equals((i - 64).toDouble()));
        }
      });
    });

    group('Isolate-based async pipeline on macOS', () {
      test('runAsync executes in isolate for large tensors', () async {
        final data = Float32List(3 * 224 * 224);
        for (int i = 0; i < data.length; i++) {
          data[i] = (i % 256).toDouble();
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 224, 224]);

        final pipeline = TensorPipeline([
          NormalizeOp.imagenet(),
        ]);

        final result = await pipeline.runAsync(tensor, isolateThreshold: 0);

        expect(result.shape, equals([3, 224, 224]));
        expect(result.dtype, equals(DType.float32));
      });

      test('runAsync falls back to sync for small tensors', () async {
        final data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 1, 2]);

        final pipeline = TensorPipeline([
          NormalizeOp.symmetric(),
        ]);

        final result = await pipeline.runAsync(tensor);

        expect(result.shape, equals([3, 1, 2]));
      });
    });

    group('Full preprocessing pipeline on macOS', () {
      test('ImageNet pipeline end-to-end', () {
        final imageData = Uint8List(224 * 224 * 3);
        for (int i = 0; i < imageData.length; i++) {
          imageData[i] = i % 256;
        }
        final tensor = TensorBuffer.fromUint8List(imageData, [224, 224, 3]);

        final pipeline = TensorPipeline([
          ToTensorOp(normalize: true),
          PermuteOp.hwcToChw(),
          NormalizeOp.imagenet(),
          UnsqueezeOp.batch(),
        ]);

        final result = pipeline.run(tensor);

        expect(result.shape, equals([1, 3, 224, 224]));
        expect(result.dtype, equals(DType.float32));
      });

      test('normalization layers pipeline', () {
        final data = Float32List(2 * 64 * 8 * 8);
        for (int i = 0; i < data.length; i++) {
          data[i] = (i % 100) / 50.0 - 1.0;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [2, 64, 8, 8]);

        final batchNorm = BatchNormOp(
          runningMean: List.filled(64, 0.0),
          runningVar: List.filled(64, 1.0),
        );

        final result = batchNorm.apply(tensor);

        expect(result.shape, equals([2, 64, 8, 8]));
        expect(result.dtype, equals(DType.float32));
      });
    });

    group('Memory management on macOS', () {
      test('BufferPool acquire and release cycle', () {
        final pool = BufferPool.create();
        pool.clear();

        final buffer1 = pool.acquireFloat32(1024);
        expect(buffer1.length, greaterThanOrEqualTo(1024));

        pool.release(buffer1);

        final buffer2 = pool.acquireFloat32(1024);
        expect(identical(buffer1, buffer2), isTrue);

        pool.clear();
      });

      test('large tensor allocation and deallocation', () {
        for (int i = 0; i < 10; i++) {
          final tensor = TensorBuffer.zeros([256, 256, 3]);
          expect(tensor.numel, equals(256 * 256 * 3));
        }
      });
    });

    group('DType operations on macOS', () {
      test('all dtype conversions work on macOS', () {
        final source = TensorBuffer.fromFloat32List(
          Float32List.fromList([1.0, 2.5, 3.7, 4.0]),
          [2, 2],
        );

        for (final targetDType in [
          DType.float64,
          DType.int32,
          DType.int64,
          DType.uint8,
          DType.uint16,
        ]) {
          final castOp = TypeCastOp(targetDType);
          final result = castOp.apply(source);
          expect(result.dtype, equals(targetDType));
          expect(result.shape, equals([2, 2]));
        }
      });
    });
  });
}
