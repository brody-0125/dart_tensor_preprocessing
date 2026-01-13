import 'dart:typed_data';

import 'dtype.dart';

/// A pool of reusable TypedData buffers for memory optimization.
///
/// Reduces GC pressure by reusing buffers instead of allocating new ones.
/// Buffers are organized by dtype and size buckets for efficient reuse.
///
/// ```dart
/// final pool = BufferPool.instance;
///
/// // Acquire a buffer
/// final buffer = pool.acquire(1000, DType.float32);
///
/// // ... use buffer ...
///
/// // Release back to pool
/// pool.release(buffer);
/// ```
class BufferPool {
  /// The singleton pool instance.
  static final BufferPool instance = BufferPool._();

  BufferPool._();

  /// Creates a new buffer pool instance (for testing or isolated use).
  factory BufferPool.create() => BufferPool._();

  // Pools organized by dtype name -> size bucket -> list of buffers
  final Map<String, Map<int, List<TypedData>>> _pools = {};

  /// Maximum number of buffers to keep per bucket.
  static const int maxBuffersPerBucket = 8;

  /// Acquires a buffer of at least [minSize] elements with dtype [dtype].
  ///
  /// Returns a buffer from the pool if available, otherwise allocates new.
  /// The returned buffer may be larger than requested (rounded up to bucket size).
  TypedData acquire(int minSize, DType dtype) {
    final bucketSize = _computeBucketSize(minSize);
    final dtypePools = _pools[dtype.typeName] ??= {};
    final bucket = dtypePools[bucketSize] ??= [];

    if (bucket.isNotEmpty) {
      return bucket.removeLast();
    }

    return dtype.createBuffer(bucketSize);
  }

  /// Releases a buffer back to the pool for reuse.
  ///
  /// The buffer will only be pooled if:
  /// - The bucket hasn't reached [maxBuffersPerBucket]
  /// - The buffer dtype can be determined
  void release(TypedData buffer) {
    final dtype = DType.fromTypedData(buffer);
    final bucketSize = buffer.lengthInBytes ~/ dtype.byteSize;
    final dtypePools = _pools[dtype.typeName] ??= {};
    final bucket = dtypePools[bucketSize] ??= [];

    // Limit pool size to prevent unbounded memory growth
    if (bucket.length < maxBuffersPerBucket) {
      bucket.add(buffer);
    }
  }

  /// Clears all pooled buffers, releasing memory.
  void clear() => _pools.clear();

  /// The total number of buffers currently in the pool.
  int get pooledCount {
    int count = 0;
    for (final dtypePools in _pools.values) {
      for (final bucket in dtypePools.values) {
        count += bucket.length;
      }
    }
    return count;
  }

  /// The total memory held by pooled buffers in bytes.
  int get pooledBytes {
    int bytes = 0;
    for (final dtypePools in _pools.values) {
      for (final bucket in dtypePools.values) {
        for (final buffer in bucket) {
          bytes += buffer.lengthInBytes;
        }
      }
    }
    return bytes;
  }

  /// Compute size bucket (round up to power of 2 for efficient reuse).
  int _computeBucketSize(int size) {
    if (size <= 0) return 1;
    int bucket = 1;
    while (bucket < size) {
      bucket *= 2;
    }
    return bucket;
  }
}

/// Extension to add pool-aware operations to TensorBuffer.
///
/// This extension provides methods to work with [BufferPool] for
/// memory-efficient tensor operations.
extension BufferPoolExtension on BufferPool {
  /// Acquires a Float32List of at least [minSize] elements.
  Float32List acquireFloat32(int minSize) {
    return acquire(minSize, DType.float32) as Float32List;
  }

  /// Acquires a Float64List of at least [minSize] elements.
  Float64List acquireFloat64(int minSize) {
    return acquire(minSize, DType.float64) as Float64List;
  }

  /// Acquires a Uint8List of at least [minSize] elements.
  Uint8List acquireUint8(int minSize) {
    return acquire(minSize, DType.uint8) as Uint8List;
  }

  /// Acquires an Int32List of at least [minSize] elements.
  Int32List acquireInt32(int minSize) {
    return acquire(minSize, DType.int32) as Int32List;
  }
}
