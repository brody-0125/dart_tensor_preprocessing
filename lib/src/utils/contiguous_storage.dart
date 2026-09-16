import '../core/tensor_buffer.dart';
import '../core/tensor_storage.dart';
import 'typed_data_views.dart';

/// Gives storage-based kernels exactly the tensor's contiguous region.
/// Contiguous views keep sharing memory; strided inputs require a copy.
TensorBuffer contiguousStorageView(TensorBuffer input) {
  if (!input.isContiguous) return input.contiguous();
  if (input.storageOffset == 0 && input.storage.length == input.numel) {
    return input;
  }
  final data = input.storage.data;
  final view = TypedDataViews.viewAs(
    data.buffer,
    input.dtype,
    offsetInBytes:
        data.offsetInBytes + input.storageOffset * input.dtype.byteSize,
    length: input.numel,
  );
  return TensorBuffer(
    storage: TensorStorage(view, input.dtype),
    shape: input.shape,
    strides: input.strides,
    memoryFormat: input.memoryFormat,
  );
}
