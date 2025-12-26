# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
