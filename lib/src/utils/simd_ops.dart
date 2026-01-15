import 'dart:typed_data';

/// SIMD-accelerated tensor operations.
///
/// These functions operate on contiguous Float32List data using
/// 128-bit SIMD instructions (Float32x4) for vectorized processing.
/// Provides 2-4x speedup for element-wise operations on Float32 tensors.
///
/// **Platform Support:**
/// - Dart VM (native): Full SIMD support on x86_64 and ARM64
/// - Flutter Web: SIMD support varies by browser, may fall back to scalar
///
/// **Usage:**
/// ```dart
/// final data = Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// SimdOps.multiplyScalar(data, 2.0); // [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
/// ```
class SimdOps {
  SimdOps._();

  /// Multiplies all elements by a scalar.
  ///
  /// Processes 4 elements at a time using SIMD, falling back to
  /// scalar operations for remaining elements.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void multiplyScalar(Float32List data, double scalar) {
    final length = data.length;
    if (length == 0) return;

    final scalar4 = Float32x4.splat(scalar);

    // SIMD loop (4 elements per iteration)
    final simdLength = length ~/ 4 * 4;

    // Check alignment - Float32x4List.view requires 16-byte aligned data
    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        simdView[i] = simdView[i] * scalar4;
      }
    } else {
      // Unaligned fallback - still use SIMD but manually
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final result = v * scalar4;
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    // Scalar loop for remaining elements
    for (var i = simdLength; i < length; i++) {
      data[i] *= scalar;
    }
  }

  /// Adds a scalar to all elements.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void addScalar(Float32List data, double scalar) {
    final length = data.length;
    if (length == 0) return;

    final scalar4 = Float32x4.splat(scalar);
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        simdView[i] = simdView[i] + scalar4;
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final result = v + scalar4;
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      data[i] += scalar;
    }
  }

  /// Subtracts a scalar from all elements.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void subtractScalar(Float32List data, double scalar) {
    addScalar(data, -scalar);
  }

  /// Element-wise addition: out = a + b
  ///
  /// All arrays must have the same length.
  ///
  /// **Complexity:** O(n) where n = a.length
  static void add(Float32List a, Float32List b, Float32List out) {
    assert(a.length == b.length && b.length == out.length);
    final length = a.length;
    if (length == 0) return;

    final simdLength = length ~/ 4 * 4;

    // Check if all arrays are aligned
    final allAligned = a.offsetInBytes % 16 == 0 &&
        b.offsetInBytes % 16 == 0 &&
        out.offsetInBytes % 16 == 0;

    if (allAligned) {
      final aView =
          Float32x4List.view(a.buffer, a.offsetInBytes, simdLength ~/ 4);
      final bView =
          Float32x4List.view(b.buffer, b.offsetInBytes, simdLength ~/ 4);
      final outView =
          Float32x4List.view(out.buffer, out.offsetInBytes, simdLength ~/ 4);

      for (var i = 0; i < aView.length; i++) {
        outView[i] = aView[i] + bView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final va = Float32x4(a[i], a[i + 1], a[i + 2], a[i + 3]);
        final vb = Float32x4(b[i], b[i + 1], b[i + 2], b[i + 3]);
        final result = va + vb;
        out[i] = result.x;
        out[i + 1] = result.y;
        out[i + 2] = result.z;
        out[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      out[i] = a[i] + b[i];
    }
  }

  /// Element-wise subtraction: out = a - b
  ///
  /// All arrays must have the same length.
  ///
  /// **Complexity:** O(n) where n = a.length
  static void subtract(Float32List a, Float32List b, Float32List out) {
    assert(a.length == b.length && b.length == out.length);
    final length = a.length;
    if (length == 0) return;

    final simdLength = length ~/ 4 * 4;

    // Check if all arrays are aligned
    final allAligned = a.offsetInBytes % 16 == 0 &&
        b.offsetInBytes % 16 == 0 &&
        out.offsetInBytes % 16 == 0;

    if (allAligned) {
      final aView =
          Float32x4List.view(a.buffer, a.offsetInBytes, simdLength ~/ 4);
      final bView =
          Float32x4List.view(b.buffer, b.offsetInBytes, simdLength ~/ 4);
      final outView =
          Float32x4List.view(out.buffer, out.offsetInBytes, simdLength ~/ 4);

      for (var i = 0; i < aView.length; i++) {
        outView[i] = aView[i] - bView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final va = Float32x4(a[i], a[i + 1], a[i + 2], a[i + 3]);
        final vb = Float32x4(b[i], b[i + 1], b[i + 2], b[i + 3]);
        final result = va - vb;
        out[i] = result.x;
        out[i + 1] = result.y;
        out[i + 2] = result.z;
        out[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      out[i] = a[i] - b[i];
    }
  }

  /// Element-wise division: out = a / b
  ///
  /// All arrays must have the same length.
  ///
  /// **Complexity:** O(n) where n = a.length
  static void divide(Float32List a, Float32List b, Float32List out) {
    assert(a.length == b.length && b.length == out.length);
    final length = a.length;
    if (length == 0) return;

    final simdLength = length ~/ 4 * 4;

    // Check if all arrays are aligned
    final allAligned = a.offsetInBytes % 16 == 0 &&
        b.offsetInBytes % 16 == 0 &&
        out.offsetInBytes % 16 == 0;

    if (allAligned) {
      final aView =
          Float32x4List.view(a.buffer, a.offsetInBytes, simdLength ~/ 4);
      final bView =
          Float32x4List.view(b.buffer, b.offsetInBytes, simdLength ~/ 4);
      final outView =
          Float32x4List.view(out.buffer, out.offsetInBytes, simdLength ~/ 4);

      for (var i = 0; i < aView.length; i++) {
        outView[i] = aView[i] / bView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final va = Float32x4(a[i], a[i + 1], a[i + 2], a[i + 3]);
        final vb = Float32x4(b[i], b[i + 1], b[i + 2], b[i + 3]);
        final result = va / vb;
        out[i] = result.x;
        out[i + 1] = result.y;
        out[i + 2] = result.z;
        out[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      out[i] = a[i] / b[i];
    }
  }

  /// Element-wise multiplication: out = a * b
  ///
  /// All arrays must have the same length.
  ///
  /// **Complexity:** O(n) where n = a.length
  static void multiply(Float32List a, Float32List b, Float32List out) {
    assert(a.length == b.length && b.length == out.length);
    final length = a.length;
    if (length == 0) return;

    final simdLength = length ~/ 4 * 4;

    final allAligned = a.offsetInBytes % 16 == 0 &&
        b.offsetInBytes % 16 == 0 &&
        out.offsetInBytes % 16 == 0;

    if (allAligned) {
      final aView =
          Float32x4List.view(a.buffer, a.offsetInBytes, simdLength ~/ 4);
      final bView =
          Float32x4List.view(b.buffer, b.offsetInBytes, simdLength ~/ 4);
      final outView =
          Float32x4List.view(out.buffer, out.offsetInBytes, simdLength ~/ 4);

      for (var i = 0; i < aView.length; i++) {
        outView[i] = aView[i] * bView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final va = Float32x4(a[i], a[i + 1], a[i + 2], a[i + 3]);
        final vb = Float32x4(b[i], b[i + 1], b[i + 2], b[i + 3]);
        final result = va * vb;
        out[i] = result.x;
        out[i + 1] = result.y;
        out[i + 2] = result.z;
        out[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      out[i] = a[i] * b[i];
    }
  }

  /// Element-wise ReLU: data[i] = max(0, data[i])
  ///
  /// Uses SIMD comparison and selection for vectorized processing.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void relu(Float32List data) {
    final length = data.length;
    if (length == 0) return;

    final zero4 = Float32x4.zero();
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        final v = simdView[i];
        // max(0, x): use comparison + select
        final mask = v.greaterThan(zero4);
        simdView[i] = mask.select(v, zero4);
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final mask = v.greaterThan(zero4);
        final result = mask.select(v, zero4);
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      if (data[i] < 0) data[i] = 0;
    }
  }

  /// Element-wise Leaky ReLU: data[i] = x > 0 ? x : alpha * x
  ///
  /// Uses SIMD comparison and selection for vectorized processing.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void leakyRelu(Float32List data, double alpha) {
    final length = data.length;
    if (length == 0) return;

    final zero4 = Float32x4.zero();
    final alpha4 = Float32x4.splat(alpha);
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        final v = simdView[i];
        final mask = v.greaterThan(zero4);
        simdView[i] = mask.select(v, v * alpha4);
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final mask = v.greaterThan(zero4);
        final result = mask.select(v, v * alpha4);
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      if (data[i] < 0) data[i] *= alpha;
    }
  }

  /// Normalizes values: data[i] = (data[i] - mean) / std
  ///
  /// Combines subtraction and division in a single pass using
  /// pre-computed inverse standard deviation.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void normalize(Float32List data, double mean, double std) {
    final length = data.length;
    if (length == 0) return;

    final mean4 = Float32x4.splat(mean);
    final invStd4 = Float32x4.splat(1.0 / std);
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        simdView[i] = (simdView[i] - mean4) * invStd4;
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final result = (v - mean4) * invStd4;
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    final invStd = 1.0 / std;
    for (var i = simdLength; i < length; i++) {
      data[i] = (data[i] - mean) * invStd;
    }
  }

  /// Fast memory copy using SIMD.
  ///
  /// More efficient than standard list copy for large Float32 arrays.
  ///
  /// **Complexity:** O(n) where n = src.length
  static void copy(Float32List src, Float32List dst) {
    assert(src.length == dst.length);
    final length = src.length;
    if (length == 0) return;

    final simdLength = length ~/ 4 * 4;

    final allAligned =
        src.offsetInBytes % 16 == 0 && dst.offsetInBytes % 16 == 0;

    if (allAligned) {
      final srcView =
          Float32x4List.view(src.buffer, src.offsetInBytes, simdLength ~/ 4);
      final dstView =
          Float32x4List.view(dst.buffer, dst.offsetInBytes, simdLength ~/ 4);
      for (var i = 0; i < srcView.length; i++) {
        dstView[i] = srcView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        dst[i] = src[i];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i + 2];
        dst[i + 3] = src[i + 3];
      }
    }

    for (var i = simdLength; i < length; i++) {
      dst[i] = src[i];
    }
  }

  /// Fills array with a constant value using SIMD.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void fill(Float32List data, double value) {
    final length = data.length;
    if (length == 0) return;

    final value4 = Float32x4.splat(value);
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        simdView[i] = value4;
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
        data[i + 3] = value;
      }
    }

    for (var i = simdLength; i < length; i++) {
      data[i] = value;
    }
  }

  /// Computes sum of all elements using SIMD.
  ///
  /// **Complexity:** O(n) where n = data.length
  static double sum(Float32List data) {
    final length = data.length;
    if (length == 0) return 0.0;

    var sum4 = Float32x4.zero();
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        sum4 = sum4 + simdView[i];
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        sum4 = sum4 + Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
      }
    }

    var total = sum4.x + sum4.y + sum4.z + sum4.w;

    for (var i = simdLength; i < length; i++) {
      total += data[i];
    }

    return total;
  }

  /// Clips values to [min, max] range using SIMD.
  ///
  /// **Complexity:** O(n) where n = data.length
  static void clip(Float32List data, double min, double max) {
    final length = data.length;
    if (length == 0) return;

    final min4 = Float32x4.splat(min);
    final max4 = Float32x4.splat(max);
    final simdLength = length ~/ 4 * 4;

    if (data.offsetInBytes % 16 == 0) {
      final simdView = Float32x4List.view(
        data.buffer,
        data.offsetInBytes,
        simdLength ~/ 4,
      );
      for (var i = 0; i < simdView.length; i++) {
        simdView[i] = simdView[i].clamp(min4, max4);
      }
    } else {
      for (var i = 0; i < simdLength; i += 4) {
        final v = Float32x4(data[i], data[i + 1], data[i + 2], data[i + 3]);
        final result = v.clamp(min4, max4);
        data[i] = result.x;
        data[i + 1] = result.y;
        data[i + 2] = result.z;
        data[i + 3] = result.w;
      }
    }

    for (var i = simdLength; i < length; i++) {
      if (data[i] < min) {
        data[i] = min;
      } else if (data[i] > max) {
        data[i] = max;
      }
    }
  }
}
