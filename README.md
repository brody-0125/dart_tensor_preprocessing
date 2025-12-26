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

## Installation

```yaml
dependencies:
  dart_tensor_preprocessing: ^0.1.0
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
```

## Available Operations

### Resize & Crop
- `ResizeOp` - Resize to fixed dimensions (nearest, bilinear, bicubic)
- `ResizeShortestOp` - Resize preserving aspect ratio
- `CenterCropOp` - Center crop to fixed dimensions

### Normalization
- `NormalizeOp` - Channel-wise normalization (presets: ImageNet, CIFAR-10, symmetric)
- `ScaleOp` - Scale values (e.g., [0-255] to [0-1])

### Layout
- `PermuteOp` - Axis reordering (e.g., HWC to CHW)
- `ToTensorOp` - HWC uint8 to CHW float32 with optional scaling
- `ToImageOp` - CHW float32 to HWC uint8

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
| `NormalizeOp.imagenet()` | `transforms.Normalize(mean, std)` |
| `ResizeOp(mode: bilinear)` | `F.interpolate(mode='bilinear')` |
| `ToTensorOp()` | `transforms.ToTensor()` |

## Requirements

- Dart SDK ^3.0.0

## License

MIT
