# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-02-02

### Added

- **New Activation Functions** (PyTorch compatible):
  - `GELUOp` - Gaussian Error Linear Unit, standard in Transformers (BERT, GPT, ViT)
    - Supports exact computation and `tanh` approximation modes
  - `SiLUOp` (Swish) - Sigmoid Linear Unit, used in EfficientNet and YOLOv5
  - `SwishOp` - Alias for `SiLUOp`
  - `HardsigmoidOp` - Hardware-efficient sigmoid approximation for MobileNetV3
  - `HardswishOp` - Hardware-efficient swish approximation for MobileNetV3
  - `MishOp` - Self-regularizing activation used in YOLOv4+
  - `ELUOp` - Exponential Linear Unit with configurable alpha

- **`stack()` Function** - Stack tensors along a new dimension (torch.stack equivalent)
  - Supports arbitrary dimension insertion with negative indexing
  - All input tensors must have identical shapes
  - Dtype-specialized for Float32/Float64 performance

- **New Normalization Operations** (PyTorch compatible):
  - `InstanceNormOp` - Instance normalization for style transfer and GANs
    - Normalizes per sample per channel (each spatial region independently)
    - Supports 3D `[C,H,W]` and 4D `[N,C,H,W]` tensors
    - `InstanceNormOp.fromStateDict()` factory for loading PyTorch weights
    - Equivalent to `torch.nn.InstanceNorm2d`
  - `RMSNormOp` - Root Mean Square normalization for modern LLMs
    - More efficient than LayerNorm (no mean subtraction)
    - Used in LLaMA, Gemma, and other modern transformers
    - Factory presets: `llama7B`, `llama13B`, `llama70B`, `gemma2B`
    - `RMSNormOp.fromStateDict()` factory for loading weights
    - Equivalent to `torch.nn.RMSNorm` (PyTorch 2.4+)

### Documentation

- Updated PyTorch compatibility table in README.md
- Added new activation functions to Available Operations list

## [0.6.5] - 2026-02-02

### Added

- **OperationCapabilities metadata** - All operations with `InPlaceTransform` and `RequiresContiguous` mixins now override `capabilities` getter:
  - `ReLUOp`, `LeakyReLUOp`, `SigmoidOp`, `TanhOp`, `SoftmaxOp`
  - `UnaryMathOp` (AbsOp, NegOp, SqrtOp, ExpOp, LogOp)
  - `ArithmeticOp` (AddOp, SubOp, MulOp, DivOp), `PowOp`
  - `BatchNormOp`, `LayerNormOp`, `GroupNormOp`
  - `NormalizeOp`, `ScaleOp`, `ClipOp`
  - `ResizeOp`, `CenterCropOp`, `RandomCropOp`, `GaussianBlurOp`, `PadOp`
  - `TypeCastOp`, `ToTensorOp`, `ToImageOp`

- **NaN/Infinity edge case tests** - 23 new tests in `simd_ops_test.dart`:
  - Float32/Float64 NaN handling for clip, abs, sqrt, normalize, relu operations
  - Float32/Float64 Infinity handling for clip, abs, sqrt, normalize, relu operations
  - Op-level NaN/Inf handling tests for ClipOp, AbsOp, SqrtOp, ReLUOp

### Changed

- **Code consistency** - Standardized `cloneForModification()` usage across all in-place operations:
  - `BatchNormOp`, `LayerNormOp`, `ClipOp`, `ArithmeticOp`, `PowOp` now use `cloneForModification()`
  - Eliminates potential double-copy issues from manual contiguity checks

### Performance

- **`PowOp` dtype specialization** - Added Float32/Float64 specialized loops for direct TypedList access

### Documentation

- **Time/space complexity** - Added Big-O complexity documentation to key operations:
  - `ResizeOp` - Complexity table for all interpolation modes (nearest, bilinear, bicubic, area, lanczos)
  - `NormalizeOp` - O(n) time with SIMD acceleration
  - `BatchNormOp` - O(n) time with pre-computed coefficients
  - `LayerNormOp` - O(n) time with Welford's algorithm
  - `GroupNormOp` - O(n) time with per-group normalization
  - `SoftmaxOp` - O(n) time with 3-pass algorithm
  - `GaussianBlurOp` - O(C×H×W×k) time using separable convolution
  - `ResizeNormalizeFusedOp` - O(C×H_out×W_out) with no intermediate tensor

### Tests

- Total test count: 897 (23 new NaN/Inf edge case tests)

## [0.6.4] - 2026-01-28

### Added

- **`ResizeNormalizeFusedOp`** - Fused resize + normalize operation that eliminates intermediate tensor allocation
  - Combines bilinear resize and per-channel normalization in a single pass
  - `factory ResizeNormalizeFusedOp.imagenet(...)` convenience constructor
  - Supports 3D `[C, H, W]` and 4D `[N, C, H, W]` inputs
  - Cache-friendly 64x64 blocking for optimal L1 cache usage

### Changed

- **Cache-friendly blocking for bilinear resize** - Applied 64x64 blocking pattern to `_resizeBilinear()` for both Float32-specialized and generic fallback paths
- **Cache-friendly blocking for area resize** - Applied 64x64 blocking pattern to `_resizeArea()` for both Float32-specialized and generic fallback paths
- **`ResizeNormalizeFusedOp.name`** - Now includes `alignCorners` parameter for better debugging visibility
- **Generic path style consistency** - Pre-computes `oneMinusFy`/`oneMinusFx` in `_bilinearNormalizeGeneric` matching Float32 path style

### Tests

- Added edge case tests for `ResizeNormalizeFusedOp`: 1x1 input, same-size resize, alignCorners with dim=1, 25x upscale
- Added validation tests: negative width, 5D input rejection, `alignCorners` in name
- Added path coverage tests: 4D+alignCorners, 4D+Float64 generic fallback, factory default alignCorners
- Added shape coverage tests: 4D non-contiguous, 4+ channel, batch=1 4D, computeOutputShape 2D behavior

## [0.6.3] - 2026-01-28

### Added

- **`TensorBuffer.uninitialized()` factory** - Creates tensor buffer without zero-fill for cases where all elements will be immediately overwritten
  - Supports all DTypes and MemoryFormat options
  - Semantically signals intent to overwrite, avoiding redundant initialization

### Changed

- **Uninitialized buffer usage** - Operations that fully overwrite output now use `TensorBuffer.uninitialized()` instead of `zeros()`:
  - `ResizeOp` (3D/4D), `CenterCropOp` (3D/4D), `concat()`, `SliceOp`, `RandomCropOp` (3D/4D), `GaussianBlurOp` (3D/4D), `PadOp` (all modes)

- **BufferPool integration in GaussianBlurOp** - Temporary `Float64List` buffers now acquired from `BufferPool` and properly released via `try/finally` to prevent leaks on exceptions

## [0.6.2] - 2026-01-20

### Internal

- **TensorBuffer Factory Separation** - Moved factory methods to separate file:
  - `tensor_buffer_factory.dart` contains: zeros, ones, full, random, randn, eye, linspace, arange, fromFloat32List, fromFloat64List, fromUint8List
  - Reduces `tensor_buffer.dart` from ~840 lines to ~530 lines
  - No API changes

- **OpValidator** - Added centralized operation validation (`validation_utils.dart`):
  - `OpValidator.validateRank()` - Validates tensor rank range
  - `OpValidator.validateAxis()` - Validates and normalizes axis (supports negative indexing)
  - `OpValidator.validateChannels()` - Validates channel count
  - `OpValidator.validatePositiveDimension()` - Validates positive dimension
  - `OpValidator.validateListLength()` - Validates list length

- **OperationCapabilities** - Added operation metadata (`transform_op.dart`):
  - `supportsInPlace` - Whether op can modify tensor in place
  - `requiresContiguous` - Whether op requires contiguous memory
  - `preservesShape` - Whether op preserves input shape
  - `modifiesDType` - Whether op may change data type
  - Default `capabilities` getter on `TransformOp`

## [0.6.1] - 2026-01-17

### Added

- **Float64 SIMD Operations** - Vectorized operations for Float64 tensors (`simd_ops.dart`):
  - `SimdOps.clipF64()` - Clips values using Float64x2.clamp()
  - `SimdOps.absF64()` - Absolute value using Float64x2.abs()
  - `SimdOps.sqrtF64()` - Square root using Float64x2.sqrt()
  - `SimdOps.normalizeF64()` - Mean/std normalization with SIMD
  - Uses Float64x2List.view() for aligned data (16-byte alignment)
  - Scalar fallback for unaligned data to avoid object creation overhead
  - ~2.5x speedup for aligned Float64 data vs scalar

- **SIMD Microbenchmark** - Performance verification for SIMD operations (`benchmark/simd_microbenchmark.dart`):
  - Direct SimdOps performance measurement (clip, abs, sqrt, normalize)
  - Aligned vs unaligned data comparison (~4.4x performance difference)
  - Float32 SIMD vs Float64 SIMD comparison
  - Edge case testing for non-multiple-of-4 lengths

- **SIMD Tests** - 64 tests in `simd_ops_test.dart`:
  - Float32 SIMD tests with alignment edge cases
  - Float64 SIMD tests (clipF64, absF64, sqrtF64, normalizeF64)
  - Op integration tests for both Float32 and Float64

### Changed

- **SimdOps.abs()** and **SimdOps.sqrt()** - Now applied to `AbsOp` and `SqrtOp` for Float32 tensors
- **SimdOps.clip()** - Now used in `ClipOp` for Float32 tensors
- **SimdOps.normalize()** - Now used in `NormalizeOp` for Float32 tensors (per-channel)
- **NegOp** - Now uses `SimdOps.multiplyScalar(-1)` for Float32 tensors
- **ClipOp** - Now uses `SimdOps.clipF64()` for Float64 tensors
- **AbsOp** - Now uses `SimdOps.absF64()` for Float64 tensors
- **SqrtOp** - Now uses `SimdOps.sqrtF64()` for Float64 tensors
- **NormalizeOp** - Now uses `SimdOps.normalizeF64()` for Float64 tensors (3D and 4D)

### Performance

- Float32 SIMD (aligned): ~6.2 GE/s
- Float64 SIMD (aligned): ~3.3 GE/s (53% of Float32, expected due to Float64x2 vs Float32x4)
- Unaligned fallback: ~1.3-1.5 GE/s

### Internal

- Integrated SIMD microbenchmark into `benchmark/run_all.dart`

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
