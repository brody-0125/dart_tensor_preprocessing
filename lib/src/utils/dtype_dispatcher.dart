/// Utility for dispatching operations based on tensor dtype.
///
/// Eliminates repetitive dtype switch statements by providing typed callbacks.
/// This reduces code duplication and ensures consistent dtype handling across
/// the codebase.
library;

import 'dart:typed_data';

import '../core/dtype.dart';
import '../core/tensor_buffer.dart';

/// Utility for dispatching operations based on tensor dtype.
///
/// The dtype switch pattern appears 15+ times across the codebase.
/// This utility centralizes that logic for consistency and maintainability.
///
/// Example usage:
/// ```dart
/// void relu(TensorBuffer tensor) {
///   DTypeDispatcher.dispatchVoid(
///     tensor,
///     onFloat32: (list, numel) {
///       for (int i = 0; i < numel; i++) {
///         if (list[i] < 0) list[i] = 0;
///       }
///     },
///     onFloat64: (list, numel) {
///       for (int i = 0; i < numel; i++) {
///         if (list[i] < 0) list[i] = 0;
///       }
///     },
///     fallback: (tensor) {
///       for (int i = 0; i < tensor.numel; i++) {
///         final value = tensor.storage.getAsDouble(i);
///         if (value < 0) tensor.storage.setFromDouble(i, 0.0);
///       }
///     },
///   );
/// }
/// ```
class DTypeDispatcher {
  DTypeDispatcher._();

  /// Dispatches to typed callback based on tensor dtype.
  ///
  /// Returns result from the appropriate callback:
  /// - [onFloat32] for Float32 tensors
  /// - [onFloat64] for Float64 tensors
  /// - [fallback] for all other types
  static R dispatch<R>(
    TensorBuffer tensor, {
    required R Function(Float32List data, int numel) onFloat32,
    required R Function(Float64List data, int numel) onFloat64,
    required R Function(TensorBuffer tensor) fallback,
  }) {
    final numel = tensor.numel;
    switch (tensor.dtype) {
      case DType.float32:
        return onFloat32(tensor.storage.data as Float32List, numel);
      case DType.float64:
        return onFloat64(tensor.storage.data as Float64List, numel);
      default:
        return fallback(tensor);
    }
  }

  /// Dispatches void operations (in-place modifications).
  ///
  /// Convenience method for operations that don't return a value.
  static void dispatchVoid(
    TensorBuffer tensor, {
    required void Function(Float32List data, int numel) onFloat32,
    required void Function(Float64List data, int numel) onFloat64,
    required void Function(TensorBuffer tensor) fallback,
  }) {
    dispatch<void>(
      tensor,
      onFloat32: onFloat32,
      onFloat64: onFloat64,
      fallback: fallback,
    );
  }

  /// Pair dispatch for operations involving two tensors.
  ///
  /// Used when input and output tensors are separate (e.g., normalization).
  /// Falls back to generic handling if dtypes don't match.
  static R dispatchPair<R>(
    TensorBuffer input,
    TensorBuffer output, {
    required R Function(Float32List inData, Float32List outData, int numel)
        onFloat32,
    required R Function(Float64List inData, Float64List outData, int numel)
        onFloat64,
    required R Function(TensorBuffer input, TensorBuffer output) fallback,
  }) {
    if (input.dtype != output.dtype) {
      return fallback(input, output);
    }
    final numel = input.numel;
    switch (input.dtype) {
      case DType.float32:
        return onFloat32(
          input.storage.data as Float32List,
          output.storage.data as Float32List,
          numel,
        );
      case DType.float64:
        return onFloat64(
          input.storage.data as Float64List,
          output.storage.data as Float64List,
          numel,
        );
      default:
        return fallback(input, output);
    }
  }
}
