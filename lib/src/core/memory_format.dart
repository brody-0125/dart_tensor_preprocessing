/// Defines the memory layout format for tensor data.
///
/// Controls default physical strides, not the names of logical axes.
/// Channels-last storage retains logical CHW/NCHW shape, like PyTorch.
/// Use an explicit permutation to change shape to HWC/NHWC.
enum MemoryFormat {
  /// Row-major storage for the supplied logical shape.
  ///
  /// This is the default format used by PyTorch and ONNX models.
  contiguous,

  /// Channels-last physical storage for a logical CHW/NCHW shape.
  ///
  /// The channel dimension has stride one; logical axes are unchanged.
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
