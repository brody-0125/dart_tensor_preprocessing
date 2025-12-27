/// Defines the memory layout format for tensor data.
///
/// Different frameworks use different memory layouts for image tensors:
/// - PyTorch/ONNX typically uses NCHW ([contiguous])
/// - TensorFlow typically uses NHWC ([channelsLast])
enum MemoryFormat {
  /// Row-major NCHW layout (batch, channels, height, width).
  ///
  /// This is the default format used by PyTorch and ONNX models.
  contiguous,

  /// Channels-last NHWC layout (batch, height, width, channels).
  ///
  /// This is the default format used by TensorFlow and Dart's image library.
  channelsLast,
}

/// Extension providing utility methods for [MemoryFormat].
extension MemoryFormatExtension on MemoryFormat {
  /// The logical dimension ordering for this format.
  List<int> get logicalOrder {
    return switch (this) {
      MemoryFormat.contiguous => [0, 1, 2, 3],
      MemoryFormat.channelsLast => [0, 2, 3, 1],
    };
  }

  /// The permutation axes to convert to the other format.
  List<int> get permuteToOther {
    return switch (this) {
      MemoryFormat.contiguous => [0, 2, 3, 1],
      MemoryFormat.channelsLast => [0, 3, 1, 2],
    };
  }

  /// The human-readable layout name (e.g., "NCHW" or "NHWC").
  String get layoutName {
    return switch (this) {
      MemoryFormat.contiguous => 'NCHW',
      MemoryFormat.channelsLast => 'NHWC',
    };
  }
}
