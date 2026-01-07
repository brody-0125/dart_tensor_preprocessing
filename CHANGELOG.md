# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-08

### Added
- `ClipOp` - Element-wise value clamping with factory presets (unit, symmetric, uint8)
- `PadOp` - Padding with multiple modes (constant, reflect, replicate, circular)
- `SliceOp` - Python-like tensor slicing with support for negative indices and steps
- `RandomCropOp` - Random cropping for data augmentation with deterministic seed support
- `GaussianBlurOp` - Gaussian blur using separable convolution with factory presets
- `concat()` - Utility function for tensor concatenation along specified axis

### Fixed
- `concat()` axis-based copy logic now correctly handles multi-axis concatenation

### Changed
- **BREAKING**: Unified exception handling across the library
  - All exceptions now extend `TensorException` sealed class
  - `ArgumentError` → `ShapeMismatchException`, `InvalidParameterException`
  - `RangeError` → `IndexOutOfBoundsException`

## [0.2.0] - 2026-01-04

### Added
- `IndexOutOfBoundsException` - Thrown when an index or axis is out of valid range
- `DTypeMismatchException` - Thrown when tensor data types do not match

### Changed
- **BREAKING**: Unified exception handling across the library
  - All exceptions now extend `TensorException` sealed class
  - `ArgumentError` → `ShapeMismatchException`, `InvalidParameterException`
  - `RangeError` → `IndexOutOfBoundsException`
  - `StateError` → `NonContiguousException`, `DTypeMismatchException`
- Shape validation now happens before buffer creation in `zeros()` and `ones()`

### Migration Guide

If you were catching standard Dart exceptions, update your code:

| Before | After |
|--------|-------|
| `on RangeError` | `on IndexOutOfBoundsException` |
| `on ArgumentError` | `on ShapeMismatchException` or `on InvalidParameterException` |
| `on StateError` | `on NonContiguousException` or `on DTypeMismatchException` |

## [0.1.4] - 2026-01-04

### Added

- Reduction operations for `TensorBuffer`:
  - `sum()` - Returns the sum of all elements
  - `mean()` - Returns the arithmetic mean of all elements
  - `min()` - Returns the minimum value
  - `max()` - Returns the maximum value
- Axis-wise reduction operations:
  - `sumAxis(int axis, {bool keepDims})` - Sum along a specific axis
  - `meanAxis(int axis, {bool keepDims})` - Mean along a specific axis
  - `minAxis(int axis, {bool keepDims})` - Min along a specific axis
  - `maxAxis(int axis, {bool keepDims})` - Max along a specific axis
- Support for negative axis indexing in axis-wise operations
- Comprehensive test coverage for all reduction operations (49 tests)

## [0.1.3] - 2026-01-03

### Added

- `TensorBuffer.toList()` method for extracting tensor data as `List<double>`

### Fixed

- Unused import in test file (`dart:math`)
- Unused variable in test file
- `prefer_final_locals` lint warnings in test files
- `dangling_library_doc_comments` lint warnings in test files

## [0.1.2] - 2025-12-27

### Added

- `.gitignore` file for Git
- `.pubignore` file for pub.dev publishing

## [0.1.1] - 2025-12-27

### Added

- Comprehensive dartdoc comments for all public API elements
- Library-level documentation with usage examples

## [0.1.0] - 2025-12-27

### Added

- Core tensor operations
  - `TensorBuffer` with shape, strides, and view/storage separation
  - `TensorStorage` for immutable typed data wrapper
  - `DType` enum with ONNX-compatible data types

- Transform operations
  - `ResizeOp` with nearest, bilinear, bicubic interpolation
  - `ResizeShortestOp` for aspect-ratio preserving resize
  - `CenterCropOp` for center cropping
  - `NormalizeOp` with ImageNet, CIFAR-10, symmetric presets
  - `ScaleOp` for value scaling
  - `PermuteOp` for axis reordering
  - `ToTensorOp` for HWC uint8 to CHW float32 conversion
  - `ToImageOp` for CHW float32 to HWC uint8 conversion
  - `UnsqueezeOp`, `SqueezeOp`, `ReshapeOp`, `FlattenOp` for shape manipulation
  - `TypeCastOp` for dtype conversion

- Pipeline system
  - `TensorPipeline` for chaining operations
  - `PipelinePresets` with ImageNet, ResNet, YOLO, CLIP, ViT, MobileNet presets
  - Async execution via `Isolate.run`

- Zero-copy operations
  - `transpose()` via stride manipulation
  - `squeeze()`, `unsqueeze()` as shape-only changes
