# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-01-16

### Added

- **Multi-axis Reductions** - Reduce along multiple axes at once (`tensor_buffer_reduce.dart`):
  - `sumAxes(List<int> axes, {bool keepDims})` - Sum along multiple axes
  - `meanAxes(List<int> axes, {bool keepDims})` - Mean along multiple axes
  - `minAxes(List<int> axes, {bool keepDims})` - Min along multiple axes
  - `maxAxes(List<int> axes, {bool keepDims})` - Max along multiple axes
  - Supports negative axis indexing
  - Validates duplicate axes

- **GroupNormOp** - Group normalization for modern CNNs (`group_norm_op.dart`):
  - Full PyTorch-compatible `torch.nn.GroupNorm` implementation
  - Normalizes across groups of channels (used in U-Net, modern CNNs with small batch sizes)
  - Supports 3D `[C,H,W]` and 4D `[N,C,H,W]` tensors
  - `GroupNormOp.withAffine()` factory for PyTorch-style initialization
  - `GroupNormOp.fromStateDict()` factory for loading PyTorch weights
  - Welford's algorithm for numerically stable mean/variance computation
  - Dtype-specialized loops for Float32/Float64
  - In-place support via `applyInPlace()`

- **SIMD Operations** - Vectorized tensor operations (`simd_ops.dart`):
  - Uses Float32x4 SIMD instructions for 2-4x speedup on Float32 tensors
  - `SimdOps.multiplyScalar()`, `SimdOps.addScalar()`, `SimdOps.subtractScalar()` - Scalar operations
  - `SimdOps.add()`, `SimdOps.subtract()`, `SimdOps.multiply()`, `SimdOps.divide()` - Element-wise binary operations
  - `SimdOps.relu()`, `SimdOps.leakyRelu()` - Activation functions
  - `SimdOps.normalize()` - Mean/std normalization
  - `SimdOps.copy()`, `SimdOps.fill()`, `SimdOps.sum()`, `SimdOps.clip()`
  - Handles both aligned and unaligned memory

- **Interpolation Modes** - Additional resize algorithms (`resize_op.dart`):
  - `InterpolationMode.area` - Weighted area averaging for high-quality downsampling with anti-aliasing (OpenCV INTER_AREA equivalent)
  - `InterpolationMode.lanczos` - Lanczos3 (6x6 kernel) for high-quality resize with sinc-based interpolation

### Changed

- **BREAKING**: Reduction operations moved to extension (`TensorBufferReduce`)
  - `sum()`, `mean()`, `min()`, `max()` - Full tensor reductions
  - `sumAxis()`, `meanAxis()`, `minAxis()`, `maxAxis()` - Single-axis reductions
  - `toList()` - Data extraction
  - Existing code using these methods will work unchanged, but users importing only `tensor_buffer.dart` must now also import `tensor_buffer_reduce.dart` or the main library

### Performance

- **SIMD-accelerated operations**: `ScaleOp`, `ReLUOp`, `LeakyReLUOp` now use SIMD for Float32 tensors
- **SIMD-accelerated ArithmeticOp**: `AddOp`, `SubOp`, `MulOp`, `DivOp` now use SIMD for Float32 tensors (both scalar and tensor modes)
- **Cache-friendly bicubic resize**: 64x64 block processing for better L1 cache utilization on large tensors

### Internal

- Extracted reduction operations from `tensor_buffer.dart` (1170 → 740 lines) to `tensor_buffer_reduce.dart`
- Added 14 new tests for multi-axis reductions
- Added 61 new tests for SIMD operations, GroupNormOp, and resize modes

### PyTorch Compatibility

| Operation | PyTorch Equivalent |
|-----------|-------------------|
| `GroupNormOp` | `torch.nn.GroupNorm` |

## [0.5.1] - 2026-01-13

### Added

- **BufferPool** - Memory pooling API for buffer reuse (`buffer_pool.dart`):
  - Singleton `BufferPool.instance` for global buffer reuse
  - Power-of-2 size bucketing for efficient allocation
  - Per-dtype buffer pools (Float32, Float64, Int32, Uint8, etc.)
  - `acquire(minSize, dtype)` and `release(buffer)` methods
  - `acquireFloat32()`, `acquireFloat64()`, etc. convenience extensions
  - Max buffers per bucket limit (8) to prevent unbounded memory growth
  - `pooledCount` and `pooledBytes` for monitoring

- **TypedData Views** - Zero-copy tensor view utilities (`typed_data_views.dart`):
  - `TypedDataViews.float32SublistView()` - Zero-copy Float32List slicing
  - `TypedDataViews.float64SublistView()` - Zero-copy Float64List slicing
  - `TypedDataViews.viewAs()` - Create typed view from ByteBuffer at offset
  - `TensorViewExtension` on TensorBuffer:
    - `sliceFirst(start, end)` - Zero-copy slice along first dimension
    - `isViewable` - Check if tensor can be used as a view
    - `toChannelsLast()` - NCHW to NHWC without copying
    - `toChannelsFirst()` - NHWC to NCHW without copying
    - `flatten()` - 1D view of contiguous tensor
    - `unbind(dim)` - Split tensor into views along dimension
    - `select(dim, index)` - Select single index with reduced rank
    - `narrow(dim, start, length)` - Narrow dimension without copying

- **Utility Libraries** (`lib/src/utils/`):
  - `dtype_dispatcher.dart` - DTypeDispatcher for dtype-specialized dispatch
  - `tensor_indexing.dart` - TensorIndexer for index calculations (index2D, index3D, index4D, linearToCoords, coordsToLinear, computeStrides)

- **TensorBuffer/TensorStorage Factory Methods**:
  - `TensorBuffer.fromFloat64List()` - Create tensor from Float64List
  - `TensorStorage.fromFloat64List()` - Create storage from Float64List

### Changed

- **SoftmaxOp Optimization**: Now preserves input dtype (Float32/Float64) instead of always using Float64. Added dtype-specialized implementations for better performance.

- **Double-copy elimination**: Operations now use `cloneForModification()` pattern (`input.isContiguous ? input.clone() : input.contiguous()`) to avoid unnecessary copies:
  - `ReLUOp`, `LeakyReLUOp`, `SigmoidOp`, `TanhOp`, `SoftmaxOp`
  - `AbsOp`, `NegOp`, `SqrtOp`, `ExpOp`, `LogOp` (UnaryMathOp)
  - `NormalizeOp`, `ScaleOp`

### Internal

- Added `cloneForModification()` helper to `RequiresContiguous` mixin in `transform_op.dart`
- Integrated `DTypeDispatcher` into activation ops (`ReLUOp`, `LeakyReLUOp`, `SigmoidOp`, `TanhOp`) for dtype-specialized loops
- Integrated `DTypeDispatcher` into `ScaleOp` for consistent dtype handling
- Replaced stride computation with `TensorIndexer.computeStrides()` in `SoftmaxOp` (removed 3x code duplication)

## [0.5.0] - 2026-01-10

### Added

- **BatchNormOp** - Batch normalization for CNN inference (`batch_norm_op.dart`):
  - Full PyTorch-compatible `torch.nn.BatchNorm2d` implementation
  - Pre-computed scale/shift coefficients for efficient inference: `y = x * scale + shift`
  - Supports 3D `[C,H,W]` and 4D `[N,C,H,W]` tensors
  - `BatchNormOp.fromStateDict()` factory for loading PyTorch weights
  - Dtype-specialized loops for Float32/Float64
  - In-place support via `applyInPlace()`

- **LayerNormOp** - Layer normalization for Transformer inference (`layer_norm_op.dart`):
  - Full PyTorch-compatible `torch.nn.LayerNorm` implementation
  - Normalizes over last N dimensions (e.g., `[768]` for BERT)
  - **Welford's algorithm** for numerically stable mean/variance computation
  - `LayerNormOp.bert()` and `LayerNormOp.bertLarge()` factory presets
  - `LayerNormOp.fromStateDict()` factory for loading PyTorch weights
  - Dtype-specialized loops for Float32/Float64
  - In-place support via `applyInPlace()`

### PyTorch Compatibility

| Operation | PyTorch Equivalent |
|-----------|-------------------|
| `BatchNormOp` | `torch.nn.BatchNorm2d` (inference) |
| `LayerNormOp` | `torch.nn.LayerNorm` |

## [0.4.1] - 2026-01-09

### Performance Optimizations

- **Dtype-specialized loops**: Hot paths in transform operations now use dtype-specific code paths with direct `Float32List`/`Float64List` access, avoiding per-element switch overhead:
  - `NormalizeOp._normalize3D()`, `NormalizeOp._normalize4D()`
  - `ScaleOp._scale()`
  - `ClipOp._clip()`
  - `GaussianBlurOp._applySeparableBlur()`
  - `ResizeOp._resizeNearest()`, `_resizeBilinear()`, `_resizeBicubic()`
  - `CenterCropOp._crop3D()`, `_crop4D()`
  - `concat()` with optimized axis=0 bulk copy

- **Clone-Before-Modify optimization**: `ClipOp.apply()` now avoids double copy by checking `isContiguous` before deciding whether to `clone()` or `contiguous()`

- **Isolate threshold**: `TensorPipeline.runAsync()` now accepts optional `isolateThreshold` parameter (default: 100,000 elements). Small tensors skip isolate overhead and run synchronously

- **Buffer reuse**: `GaussianBlurOp` now pre-allocates and reuses temp buffer across channels, reducing allocations

- **Concat linear copy**: `concat()` now uses pre-computed strides for linear index calculation instead of recursive index computation. Axis=0 concatenation of contiguous tensors uses bulk `setRange()` copy

- **Loop unrolling**: `ResizeOp._resizeBicubic()` unrolls 4x4 kernel with pre-computed weights and indices

## [0.4.0] - 2026-01-09

### Added
- **Arithmetic Operations** (`arithmetic_op.dart`):
  - `AddOp` - Element-wise addition (scalar or tensor)
  - `SubOp` - Element-wise subtraction (scalar or tensor)
  - `MulOp` - Element-wise multiplication (scalar or tensor)
  - `DivOp` - Element-wise division (scalar or tensor)
  - `PowOp` - Element-wise power operation
- **Math Operations** (`math_op.dart`):
  - `AbsOp` - Element-wise absolute value
  - `NegOp` - Element-wise negation
  - `SqrtOp` - Element-wise square root
  - `ExpOp` - Element-wise exponential (e^x)
  - `LogOp` - Element-wise natural logarithm
- **Activation Functions** (`activation_op.dart`):
  - `ReLUOp` - Rectified Linear Unit
  - `LeakyReLUOp` - Leaky ReLU with configurable negative slope
  - `SigmoidOp` - Sigmoid activation
  - `TanhOp` - Hyperbolic tangent activation
  - `SoftmaxOp` - Softmax along specified axis
- **TensorBuffer Factory Methods**:
  - `TensorBuffer.full()` - Create tensor filled with specified value
  - `TensorBuffer.random()` - Create tensor with uniform random values [0, 1)
  - `TensorBuffer.randn()` - Create tensor with standard normal distribution
  - `TensorBuffer.eye()` - Create identity matrix (supports rectangular)
  - `TensorBuffer.linspace()` - Create tensor with evenly spaced values
  - `TensorBuffer.arange()` - Create tensor with sequence values
- **Utility Libraries** (`lib/src/utils/`):
  - `index_utils.dart` - Index manipulation utilities (reflectIndex, replicateIndex, circularIndex)
  - `validation_utils.dart` - Common tensor validation patterns

### Changed
- **Exception Consistency**: `TensorStorage._checkBounds()` now throws `IndexOutOfBoundsException` instead of `RangeError` for consistent exception handling across the library

### Internal
- Extracted duplicate `_reflectIndex` code from `pad_op.dart` and `augmentation_op.dart` into shared utility
- Added `TensorValidation` extension with `requireRank3Or4()`, `requireExactRank()`, `requireMinRank()` methods

## [0.3.1] - 2026-01-08

### Added
- Performance benchmark suite (`benchmark/` directory):
  - `tensor_creation_benchmark.dart` - Tensor creation performance
  - `tensor_ops_benchmark.dart` - Zero-copy and copy operations
  - `pipeline_benchmark.dart` - Pipeline sync/async comparison
  - `memory_benchmark.dart` - Memory usage measurement
  - `run_all.dart` - Unified benchmark runner
  - `utils/benchmark_utils.dart` - Benchmark utilities

### Fixed
- Removed unused variables in benchmark files
- Fixed lint issues in benchmark files

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
