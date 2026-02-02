# dart_tensor_preprocessing

![Dart](https://img.shields.io/badge/Dart-3.0+-0175C2.svg?logo=dart)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-EE4C2C.svg?logo=pytorch)

Tensor preprocessing library for Flutter/Dart. NumPy-like transforms pipeline for ONNX Runtime, TFLite, and other AI inference engines.

## Features

- **PyTorch Compatible**: Matches PyTorch/torchvision tensor operations
- **Non-blocking**: Isolate-based async execution prevents UI jank
- **Type-safe**: ONNX-compatible tensor types (Float32, Int64, Uint8, etc.)
- **Zero-copy**: View/stride manipulation for reshape/transpose operations
- **Declarative**: Chain operations into reusable pipelines
- **SIMD Accelerated**: Float32/Float64 vectorized operations for 2-4x speedup
- **Memory Efficient**: Buffer pooling, uninitialized allocation, fused operations

## Installation

```yaml
dependencies:
  dart_tensor_preprocessing: ^0.6.5
```

## Quick Start

```dart
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';

// Create a tensor from image data (HWC format, Uint8)
final imageData = Uint8List.fromList([/* RGBA pixel data */]);
final tensor = TensorBuffer.fromUint8List(imageData, [height, width, channels]);

// Use a preset pipeline for ImageNet models
final pipeline = PipelinePresets.imagenetClassification();
final result = await pipeline.runAsync(tensor);

// result.shape: [1, 3, 224, 224] (NCHW, Float32, normalized)
```

## Pipeline Presets

| Preset | Output Shape | Use Case |
|--------|--------------|----------|
| `imagenetClassification()` | [1, 3, 224, 224] | ResNet, VGG, etc. |
| `objectDetection()` | [1, 3, 640, 640] | YOLO, SSD |
| `faceRecognition()` | [1, 3, 112, 112] | ArcFace, FaceNet |
| `clip()` | [1, 3, 224, 224] | CLIP models |
| `mobileNet()` | [1, 3, 224, 224] | MobileNet family |

## Custom Pipeline

```dart
final pipeline = TensorPipeline([
  ResizeOp(height: 224, width: 224),
  ToTensorOp(normalize: true),  // HWC -> CHW, scale to [0,1]
  NormalizeOp.imagenet(),       // ImageNet mean/std
  UnsqueezeOp.batch(),          // Add batch dimension
]);

// Sync execution
final result = pipeline.run(input);

// Async execution (runs in isolate)
final result = await pipeline.runAsync(input);

// Async with custom isolate threshold (default: 100,000 elements)
// Small tensors skip isolate overhead and run synchronously
final result = await pipeline.runAsync(input, isolateThreshold: 50000);
```

## Available Operations

### Resize & Crop
- `ResizeOp` - Resize to fixed dimensions (nearest, bilinear, bicubic, area, lanczos)
- `ResizeShortestOp` - Resize preserving aspect ratio
- `CenterCropOp` - Center crop to fixed dimensions
- `ClipOp` - Element-wise value clamping (presets: unit, symmetric, uint8)
- `PadOp` - Padding with multiple modes (constant, reflect, replicate, circular)
- `SliceOp` - Python-like tensor slicing with negative index support

### Normalization
- `NormalizeOp` - Channel-wise normalization (presets: ImageNet, CIFAR-10, symmetric)
- `ScaleOp` - Scale values (e.g., [0-255] to [0-1])
- `BatchNormOp` - Batch normalization for CNN inference (PyTorch compatible)
- `LayerNormOp` - Layer normalization for Transformer inference (presets: BERT, BERT-Large)
- `GroupNormOp` - Group normalization for modern CNNs (PyTorch compatible)

### Layout
- `PermuteOp` - Axis reordering (e.g., HWC to CHW)
- `ToTensorOp` - HWC uint8 to CHW float32 with optional scaling
- `ToImageOp` - CHW float32 to HWC uint8

### Data Augmentation
- `RandomCropOp` - Random cropping with deterministic seed support
- `GaussianBlurOp` - Gaussian blur using separable convolution

### Fused Operations
- `ResizeNormalizeFusedOp` - Combines resize + normalize in single pass (eliminates intermediate tensor)

### Activation Functions
- `ReLUOp` - Rectified Linear Unit (SIMD accelerated)
- `LeakyReLUOp` - Leaky ReLU with configurable slope (SIMD accelerated)
- `SigmoidOp` - Sigmoid activation
- `TanhOp` - Hyperbolic tangent activation
- `SoftmaxOp` - Softmax along specified axis

### Math Operations
- `AbsOp` - Absolute value (SIMD accelerated)
- `NegOp` - Negation (SIMD accelerated)
- `SqrtOp` - Square root (SIMD accelerated)
- `ExpOp` - Exponential (e^x)
- `LogOp` - Natural logarithm
- `PowOp` - Power operation

### Arithmetic Operations
- `AddOp` / `SubOp` - Element-wise addition/subtraction (SIMD accelerated)
- `MulOp` / `DivOp` - Element-wise multiplication/division (SIMD accelerated)

### Utility
- `concat()` - Concatenates tensors along specified axis

### Shape
- `UnsqueezeOp` - Add dimension
- `SqueezeOp` - Remove size-1 dimensions
- `ReshapeOp` - Reshape tensor (supports -1 for inference)
- `FlattenOp` - Flatten dimensions

### Type
- `TypeCastOp` - Convert between data types

## Core Classes

### TensorBuffer

Tensor with shape and stride metadata over physical storage.

```dart
// Create tensors
final zeros = TensorBuffer.zeros([3, 224, 224]);
final ones = TensorBuffer.ones([3, 224, 224], dtype: DType.float32);
final fromData = TensorBuffer.fromFloat32List(data, [3, 224, 224]);

// Access elements
final value = tensor[[0, 100, 100]];

// Zero-copy operations
final transposed = tensor.transpose([2, 0, 1]);  // Changes strides only
final squeezed = tensor.squeeze();

// Copy operations
final contiguous = tensor.contiguous();  // Force contiguous memory
final cloned = tensor.clone();
```

### DType

ONNX-compatible data types with `onnxId` for runtime integration.

```dart
DType.float32  // ONNX ID: 1
DType.int64    // ONNX ID: 7
DType.uint8    // ONNX ID: 2
```

### BufferPool

Memory pooling for buffer reuse, reducing GC pressure in hot paths.

```dart
final pool = BufferPool.instance;

// Acquire buffer (reuses from pool if available)
final buffer = pool.acquireFloat32(1000);

// ... use buffer ...

// Release back to pool for reuse
pool.release(buffer);

// Monitor pool usage
print('Pooled: ${pool.pooledCount} buffers, ${pool.pooledBytes} bytes');
```

### Zero-Copy View Operations

TensorBuffer extension methods for zero-copy tensor manipulation:

```dart
// Slice along first dimension (batch slicing)
final batch = tensor.sliceFirst(2, 5);  // Views elements 2..4

// Split tensor into views
final items = tensor.unbind(0);  // List of views along dim 0

// Select single index (reduces rank)
final first = tensor.select(0, 0);  // First item, shape reduced

// Narrow dimension
final narrowed = tensor.narrow(0, 1, 3);  // 3 elements starting at 1

// Format conversion without copying
final nhwc = nchwTensor.toChannelsLast();   // NCHW -> NHWC view
final nchw = nhwcTensor.toChannelsFirst();  // NHWC -> NCHW view

// Flatten to 1D view
final flat = tensor.flatten();
```

## In-Place Operations

Many operations support in-place modification to avoid allocation overhead:

```dart
// In-place operations (modify tensor directly)
ReLUOp().applyInPlace(tensor);
NormalizeOp.imagenet().applyInPlace(tensor);
ClipOp(min: 0, max: 1).applyInPlace(tensor);
BatchNormOp(...).applyInPlace(tensor);

// Query operation capabilities
final op = ReLUOp();
print(op.capabilities.supportsInPlace);    // true
print(op.capabilities.requiresContiguous); // true
print(op.capabilities.preservesShape);     // true
```

Operations supporting in-place: `ReLUOp`, `LeakyReLUOp`, `SigmoidOp`, `TanhOp`, `AbsOp`, `NegOp`, `SqrtOp`, `ExpOp`, `LogOp`, `PowOp`, `AddOp`, `SubOp`, `MulOp`, `DivOp`, `ClipOp`, `NormalizeOp`, `ScaleOp`, `BatchNormOp`, `LayerNormOp`, `GroupNormOp`.

## Memory Formats

| Format | Layout | Strides (for [1,3,224,224]) |
|--------|--------|----------------------------|
| `contiguous` | NCHW | [150528, 50176, 224, 1] |
| `channelsLast` | NHWC | [150528, 1, 672, 3] |

## PyTorch Compatibility

This library is designed to produce identical results to PyTorch/torchvision operations:

| Operation | PyTorch Equivalent |
|-----------|-------------------|
| `TensorBuffer.zeros()` | `torch.zeros()` |
| `TensorBuffer.ones()` | `torch.ones()` |
| `tensor.transpose()` | `tensor.permute()` |
| `tensor.reshape()` | `tensor.reshape()` |
| `tensor.squeeze()` | `tensor.squeeze()` |
| `tensor.unsqueeze()` | `tensor.unsqueeze()` |
| `tensor.sum()` / `sumAxis()` | `tensor.sum()` |
| `tensor.sumAxes([...])` | `tensor.sum(dim=[...])` |
| `tensor.mean()` / `meanAxis()` | `tensor.mean()` |
| `tensor.meanAxes([...])` | `tensor.mean(dim=[...])` |
| `tensor.min()` / `max()` | `tensor.min()` / `max()` |
| `tensor.minAxes([...])` | `tensor.amin(dim=[...])` |
| `tensor.maxAxes([...])` | `tensor.amax(dim=[...])` |
| `NormalizeOp.imagenet()` | `transforms.Normalize(mean, std)` |
| `ResizeOp(mode: bilinear)` | `F.interpolate(mode='bilinear')` |
| `ResizeOp(mode: area)` | `F.interpolate(mode='area')` |
| `ResizeOp(mode: lanczos)` | Lanczos3 interpolation |
| `ToTensorOp()` | `transforms.ToTensor()` |
| `ClipOp(min, max)` | `torch.clamp(min, max)` |
| `PadOp(mode: reflect)` | `F.pad(mode='reflect')` |
| `SliceOp([(start, end, step)])` | `tensor[start:end:step]` |
| `concat(tensors, axis)` | `torch.cat(tensors, dim)` |
| `RandomCropOp` | `transforms.RandomCrop()` |
| `GaussianBlurOp` | `transforms.GaussianBlur()` |
| `AddOp` / `SubOp` | `torch.add()` / `torch.sub()` |
| `MulOp` / `DivOp` | `torch.mul()` / `torch.div()` |
| `PowOp` | `torch.pow()` |
| `AbsOp` / `NegOp` | `torch.abs()` / `torch.neg()` |
| `SqrtOp` / `ExpOp` / `LogOp` | `torch.sqrt()` / `exp()` / `log()` |
| `ReLUOp` / `LeakyReLUOp` | `F.relu()` / `F.leaky_relu()` |
| `SigmoidOp` / `TanhOp` | `torch.sigmoid()` / `torch.tanh()` |
| `SoftmaxOp` | `F.softmax()` |
| `BatchNormOp` | `torch.nn.BatchNorm2d` (inference) |
| `LayerNormOp` | `torch.nn.LayerNorm` |
| `GroupNormOp` | `torch.nn.GroupNorm` |
| `TensorBuffer.full()` | `torch.full()` |
| `TensorBuffer.random()` | `torch.rand()` |
| `TensorBuffer.randn()` | `torch.randn()` |
| `TensorBuffer.eye()` | `torch.eye()` |
| `TensorBuffer.linspace()` | `torch.linspace()` |
| `TensorBuffer.arange()` | `torch.arange()` |
| `tensor.select(dim, index)` | `tensor.select(dim, index)` |
| `tensor.narrow(dim, start, len)` | `tensor.narrow(dim, start, len)` |
| `tensor.unbind(dim)` | `tensor.unbind(dim)` |
| `tensor.flatten()` | `tensor.flatten()` |
| `ResizeNormalizeFusedOp` | `F.interpolate()` + `transforms.Normalize()` (fused) |

## Performance Benchmarks

Run benchmarks with `dart run benchmark/run_all.dart`.

### SIMD Acceleration

Operations with Float32x4/Float64x2 SIMD vectorization:

| Operation | SIMD Throughput | Speedup |
|-----------|-----------------|---------|
| `ClipOp` | ~6.2 GE/s (Float32) | ~4x |
| `AbsOp` | ~6.2 GE/s (Float32) | ~4x |
| `SqrtOp` | ~6.2 GE/s (Float32) | ~4x |
| `NormalizeOp` | ~6.2 GE/s (Float32) | ~4x |
| `ReLUOp` / `LeakyReLUOp` | ~6.2 GE/s (Float32) | ~4x |
| `ScaleOp` | ~6.2 GE/s (Float32) | ~4x |
| `AddOp` / `SubOp` / `MulOp` / `DivOp` | ~6.2 GE/s (Float32) | ~4x |

> GE/s = Giga Elements per second. Float64 SIMD achieves ~53% of Float32 performance due to Float64x2 vs Float32x4.

### Operation Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| `ResizeOp` (bilinear) | O(C × H × W) | O(C × H × W) |
| `ResizeOp` (bicubic) | O(C × H × W × 16) | O(C × H × W) |
| `ResizeOp` (lanczos) | O(C × H × W × 36) | O(C × H × W) |
| `NormalizeOp` | O(n) | O(n) or O(1) in-place |
| `BatchNormOp` | O(n) | O(n) or O(1) in-place |
| `LayerNormOp` | O(n) | O(n) or O(1) in-place |
| `GaussianBlurOp` | O(C × H × W × k) | O(C × H × W) |
| `ResizeNormalizeFusedOp` | O(C × H × W) | O(C × H × W) |

### Zero-Copy Operations (O(1))

| Operation | Time | Ops/sec |
|-----------|------|---------|
| `transpose()` | ~1µs | 700K+ |
| `reshape()` | ~1µs | 1.6M+ |
| `squeeze()` | <1µs | 3.2M+ |
| `unsqueeze()` | ~1µs | 780K+ |

### Pipeline Performance

| Pipeline | Input Shape | Time |
|----------|-------------|------|
| Simple (Normalize + Unsqueeze) | [3, 224, 224] | ~3.4ms |
| ImageNet Classification | [3, 224, 224] | ~3.0ms |
| Object Detection | [3, 640, 640] | ~25ms |

### Sync vs Async

| Execution | 224x224 | 640x640 |
|-----------|---------|---------|
| `run()` (sync) | ~3.5ms | ~29ms |
| `runAsync()` (isolate) | ~11ms | ~93ms |
| Isolate overhead | ~7ms | ~64ms |

> **Note**: Use `runAsync()` for large tensors or when UI responsiveness is critical.

## Requirements

- Dart SDK ^3.0.0

## License

MIT
