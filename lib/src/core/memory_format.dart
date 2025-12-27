/// Memory layout formats for multi-dimensional tensors.
///
/// Different frameworks use different memory layouts for tensors:
/// - PyTorch/ONNX typically use NCHW ([contiguous])
/// - TensorFlow/Dart image library use NHWC ([channelsLast])
///
/// ## Example
///
/// ```dart
/// // Create a tensor with channels-last format
/// final tensor = TensorBuffer(
///   storage: storage,
///   shape: [1, 3, 224, 224],
///   memoryFormat: MemoryFormat.channelsLast,
/// );
///
/// print(tensor.memoryFormat.layoutName);  // 'NHWC'
/// ```
enum MemoryFormat {
  /// Standard contiguous memory layout (NCHW for 4D tensors).
  ///
  /// This is the default format used by PyTorch and ONNX, where data is
  /// stored in row-major order with channels as the second dimension.
  contiguous,

  /// Channels-last memory layout (NHWC for 4D tensors).
  ///
  /// This format is used by TensorFlow and the Dart image library, where
  /// channels are stored as the last dimension.
  channelsLast,
}

/// Extension methods for [MemoryFormat].
extension MemoryFormatExtension on MemoryFormat {
  /// The logical dimension order for this memory format.
  ///
  /// For a 4D tensor (N, C, H, W):
  /// - [contiguous]: `[0, 1, 2, 3]` (NCHW order)
  /// - [channelsLast]: `[0, 2, 3, 1]` (NHWC order)
  List<int> get logicalOrder {
    return switch (this) {
      MemoryFormat.contiguous => [0, 1, 2, 3],
      MemoryFormat.channelsLast => [0, 2, 3, 1],
    };
  }

  /// The permutation axes needed to convert to the other memory format.
  ///
  /// Returns the axes to pass to [TensorBuffer.transpose] for conversion.
  List<int> get permuteToOther {
    return switch (this) {
      MemoryFormat.contiguous => [0, 2, 3, 1],
      MemoryFormat.channelsLast => [0, 3, 1, 2],
    };
  }

  /// The conventional name for this layout (e.g., 'NCHW' or 'NHWC').
  String get layoutName {
    return switch (this) {
      MemoryFormat.contiguous => 'NCHW',
      MemoryFormat.channelsLast => 'NHWC',
    };
  }
}
