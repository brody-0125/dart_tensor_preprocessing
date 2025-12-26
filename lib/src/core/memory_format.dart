enum MemoryFormat {
  contiguous,
  channelsLast,
}

extension MemoryFormatExtension on MemoryFormat {
  List<int> get logicalOrder {
    return switch (this) {
      MemoryFormat.contiguous => [0, 1, 2, 3],
      MemoryFormat.channelsLast => [0, 2, 3, 1],
    };
  }

  List<int> get permuteToOther {
    return switch (this) {
      MemoryFormat.contiguous => [0, 2, 3, 1],
      MemoryFormat.channelsLast => [0, 3, 1, 2],
    };
  }

  String get layoutName {
    return switch (this) {
      MemoryFormat.contiguous => 'NCHW',
      MemoryFormat.channelsLast => 'NHWC',
    };
  }
}
