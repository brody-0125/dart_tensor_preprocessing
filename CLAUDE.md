# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A high-performance tensor preprocessing library for Flutter/Dart that enables NumPy-like transforms for ONNX Runtime inference. Key features:
- Non-blocking isolate-based execution
- Type-safe ONNX tensor types (Float32, Int64, Uint8, etc.)
- Zero-copy view/stride manipulation for reshape/transpose
- Declarative preprocessing pipelines

## Commands

```bash
# Install dependencies
dart pub get

# Run all tests
dart test

# Run a single test file
dart test test/tensor_buffer_test.dart

# Static analysis
dart analyze
```

## Architecture

### Layer Structure

```
User API (TensorPipeline, Presets)
         ↓
Transform Operations (ResizeOp, NormalizeOp, PermuteOp, TypeCastOp)
         ↓
Core Tensor (TensorBuffer, TensorStorage, DType, MemoryFormat)
         ↓
Execution Engine (sync/async via Isolate.run)
         ↓
Platform (dart:typed_data, dart:isolate)
```

### Core Design: View/Storage Separation

- **TensorStorage**: Immutable wrapper around TypedData (physical data)
- **TensorBuffer**: View with shape, strides, and offset metadata
- This enables O(1) `transpose()` via stride manipulation without copying data
- `reshape()` requires contiguity; use `contiguous()` to force copy when needed

### Memory Formats

- **NCHW** (`MemoryFormat.contiguous`): PyTorch/ONNX standard
- **NHWC** (`MemoryFormat.channelsLast`): TensorFlow/Dart image lib standard

### DType Enum

Maps directly to ONNX TensorProto.DataType with `onnxId` and `byteSize` properties.

## Key Patterns

### Pipeline Usage

```dart
final pipeline = TensorPipeline()
  ..append(ResizeOp(224, 224))
  ..append(NormalizeOp.imageNet())
  ..append(PermuteOp.hwcToChw());

// Sync execution
final result = pipeline.run(input);

// Async execution (isolate-based)
final result = await pipeline.runAsync(input);
```

### Zero-Copy Operations

- `transpose()`: Changes strides only, no data copy
- `squeeze()`, `unsqueeze()`: Shape-only changes
- `reshape()`: Only works on contiguous tensors (throws otherwise)

## Code Style

Uses `package:lints/recommended.yaml` with additional rules:
- `prefer_const_constructors`
- `prefer_const_declarations`
- `prefer_final_locals`
- `avoid_print`
- `prefer_single_quotes`
