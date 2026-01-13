import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('BufferPool', () {
    late BufferPool pool;

    setUp(() {
      pool = BufferPool.create();
    });

    group('acquire', () {
      test('allocates new buffer when pool is empty', () {
        final buffer = pool.acquireFloat32(100);

        expect(buffer, isA<Float32List>());
        expect(buffer.length, greaterThanOrEqualTo(100));
      });

      test('rounds up to power of 2 bucket size', () {
        final buffer = pool.acquireFloat32(100);

        // 100 rounds up to 128
        expect(buffer.length, equals(128));
      });

      test('returns pooled buffer when available', () {
        final original = pool.acquireFloat32(100);
        pool.release(original);

        final reused = pool.acquireFloat32(100);

        expect(identical(reused, original), isTrue);
      });

      test('handles different dtypes separately', () {
        final float32 = pool.acquireFloat32(100);
        pool.release(float32);

        final float64 = pool.acquireFloat64(100);

        expect(float64, isA<Float64List>());
        expect(identical(float64, float32), isFalse);
      });

      test('handles different sizes separately', () {
        final small = pool.acquireFloat32(64);
        pool.release(small);

        final large = pool.acquireFloat32(256);

        expect(identical(large, small), isFalse);
        expect(large.length, equals(256));
      });
    });

    group('release', () {
      test('adds buffer to pool', () {
        expect(pool.pooledCount, equals(0));

        final buffer = pool.acquire(100, DType.float32);
        pool.release(buffer);

        expect(pool.pooledCount, equals(1));
      });

      test('respects maxBuffersPerBucket limit', () {
        final buffers = <TypedData>[];
        for (int i = 0; i < BufferPool.maxBuffersPerBucket + 5; i++) {
          buffers.add(pool.acquire(100, DType.float32));
        }

        for (final buffer in buffers) {
          pool.release(buffer);
        }

        expect(pool.pooledCount, equals(BufferPool.maxBuffersPerBucket));
      });
    });

    group('clear', () {
      test('removes all pooled buffers', () {
        final buffer1 = pool.acquire(100, DType.float32);
        final buffer2 = pool.acquire(100, DType.float64);
        pool.release(buffer1);
        pool.release(buffer2);

        expect(pool.pooledCount, equals(2));

        pool.clear();

        expect(pool.pooledCount, equals(0));
      });
    });

    group('pooledCount', () {
      test('returns correct count across dtypes and sizes', () {
        final b1 = pool.acquire(64, DType.float32);
        final b2 = pool.acquire(128, DType.float32);
        final b3 = pool.acquire(64, DType.float64);

        pool.release(b1);
        pool.release(b2);
        pool.release(b3);

        expect(pool.pooledCount, equals(3));
      });
    });

    group('pooledBytes', () {
      test('returns correct total bytes', () {
        final b1 = pool.acquire(64, DType.float32); // 64 * 4 = 256 bytes
        final b2 = pool.acquire(32, DType.float64); // 32 * 8 = 256 bytes

        pool.release(b1);
        pool.release(b2);

        expect(pool.pooledBytes, equals(512));
      });
    });

    group('singleton', () {
      test('instance returns same pool', () {
        final pool1 = BufferPool.instance;
        final pool2 = BufferPool.instance;

        expect(identical(pool1, pool2), isTrue);
      });
    });

    group('BufferPoolExtension', () {
      test('acquireFloat32 returns Float32List', () {
        final buffer = pool.acquireFloat32(100);

        expect(buffer, isA<Float32List>());
      });

      test('acquireFloat64 returns Float64List', () {
        final buffer = pool.acquireFloat64(100);

        expect(buffer, isA<Float64List>());
      });

      test('acquireUint8 returns Uint8List', () {
        final buffer = pool.acquireUint8(100);

        expect(buffer, isA<Uint8List>());
      });

      test('acquireInt32 returns Int32List', () {
        final buffer = pool.acquireInt32(100);

        expect(buffer, isA<Int32List>());
      });
    });

    group('bucket sizing', () {
      test('size 0 returns bucket size 1', () {
        final buffer = pool.acquireFloat32(0);

        expect(buffer.length, equals(1));
      });

      test('size 1 returns bucket size 1', () {
        final buffer = pool.acquireFloat32(1);

        expect(buffer.length, equals(1));
      });

      test('exact power of 2 returns same size', () {
        final buffer = pool.acquireFloat32(256);

        expect(buffer.length, equals(256));
      });

      test('size 257 returns bucket size 512', () {
        final buffer = pool.acquireFloat32(257);

        expect(buffer.length, equals(512));
      });
    });
  });
}
