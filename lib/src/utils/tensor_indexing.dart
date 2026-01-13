/// Utilities for tensor index calculations.
///
/// Provides optimized index computation methods commonly used in tensor
/// operations like resize, pad, and convolution.
library;

/// Utilities for tensor index calculations.
///
/// These methods encapsulate common index patterns found throughout the
/// codebase, reducing duplication and potential for errors.
///
/// For hot paths (inner loops), consider using inline expressions or
/// `@pragma('vm:prefer-inline')` annotated methods for best performance.
class TensorIndexer {
  TensorIndexer._();

  /// Computes linear index for 4D tensor [N,C,H,W] (NCHW format).
  ///
  /// This is the most common layout for image tensors in PyTorch and ONNX.
  ///
  /// Example:
  /// ```dart
  /// // For tensor of shape [2, 3, 4, 5] (N=2, C=3, H=4, W=5)
  /// final idx = TensorIndexer.index4D(1, 2, 3, 4, 3, 4, 5);
  /// // idx = 1*60 + 2*20 + 3*5 + 4 = 60 + 40 + 15 + 4 = 119
  /// ```
  @pragma('vm:prefer-inline')
  static int index4D(int b, int c, int h, int w, int C, int H, int W) {
    return ((b * C + c) * H + h) * W + w;
  }

  /// Computes linear index for 3D tensor [C,H,W].
  ///
  /// Used for single-image processing without batch dimension.
  @pragma('vm:prefer-inline')
  static int index3D(int c, int h, int w, int H, int W) {
    return (c * H + h) * W + w;
  }

  /// Computes linear index for 2D tensor [H,W].
  ///
  /// Used for grayscale images or single-channel operations.
  @pragma('vm:prefer-inline')
  static int index2D(int h, int w, int W) {
    return h * W + w;
  }

  /// Converts linear index to multi-dimensional coordinates.
  ///
  /// Given a flat index into contiguous storage, returns the corresponding
  /// multi-dimensional coordinates for the given shape.
  ///
  /// Example:
  /// ```dart
  /// final coords = TensorIndexer.linearToCoords(11, [2, 3, 2]);
  /// // 11 = 1*6 + 1*2 + 1 → coords = [1, 2, 1]
  /// ```
  static List<int> linearToCoords(int linearIdx, List<int> shape) {
    final rank = shape.length;
    final coords = List<int>.filled(rank, 0);
    int remaining = linearIdx;

    for (int d = rank - 1; d >= 0; d--) {
      coords[d] = remaining % shape[d];
      remaining ~/= shape[d];
    }

    return coords;
  }

  /// Converts multi-dimensional coordinates to linear index using strides.
  ///
  /// For non-contiguous tensors (e.g., transposed), use the tensor's
  /// actual strides rather than computing from shape.
  ///
  /// Example:
  /// ```dart
  /// // Contiguous tensor of shape [2, 3, 4] has strides [12, 4, 1]
  /// final idx = TensorIndexer.coordsToLinear([1, 2, 3], [12, 4, 1]);
  /// // idx = 1*12 + 2*4 + 3*1 = 23
  /// ```
  @pragma('vm:prefer-inline')
  static int coordsToLinear(List<int> coords, List<int> strides) {
    int offset = 0;
    for (int d = 0; d < coords.length; d++) {
      offset += coords[d] * strides[d];
    }
    return offset;
  }

  /// Pre-computes contiguous strides for a shape.
  ///
  /// Returns row-major (C-order) strides where the last dimension
  /// has stride 1.
  ///
  /// Example:
  /// ```dart
  /// final strides = TensorIndexer.computeStrides([2, 3, 4]);
  /// // strides = [12, 4, 1]
  /// ```
  static List<int> computeStrides(List<int> shape) {
    final rank = shape.length;
    if (rank == 0) return [];
    final strides = List<int>.filled(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /// Computes linear index for 4D tensor in NHWC format.
  ///
  /// This layout is common in TensorFlow and Flutter's image library.
  @pragma('vm:prefer-inline')
  static int index4DNHWC(int b, int h, int w, int c, int H, int W, int C) {
    return ((b * H + h) * W + w) * C + c;
  }

  /// Computes linear index for 3D tensor in HWC format.
  @pragma('vm:prefer-inline')
  static int index3DHWC(int h, int w, int c, int W, int C) {
    return (h * W + w) * C + c;
  }
}
