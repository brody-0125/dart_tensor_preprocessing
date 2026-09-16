# Compatibility audit for 1.0.0 (in progress)

This is an evidence map, not a claim that all APIs match PyTorch. A passing
legacy test alone does not close an independent-oracle requirement. The
manifest and generator under `test/fixtures/pytorch` / `scripts` are the
source of truth for the named case prefixes below.

## Established contracts

- CPU tensor oracle: torch 2.10.0+cpu / torchvision 0.25.0+cpu, float32/64.
  Cases assert shape, dtype, every value, and NaN/infinity classification.
- Image presets: RGB uint8 or float32/64 HWC/NHWC input, float32 output;
  float inputs are already in [0,1]. The complete preset recipes are explicit
  and do not represent every pretrained weight configuration.
- Batch normalization is inference using supplied running statistics.
  Batch/Group/Instance normalization accepts CHW/NCHW; rank-three input is an
  unbatched image, unlike PyTorch's general NCL interpretation.
- Layer/RMS normalize the supplied trailing dimensions. Normalization uses
  finite positive epsilon. Lp supports positive p, including infinity, and
  divides by max(norm, eps). Integer normalization behavior is still under audit.
- Resize nearest means PyTorch nearest, not nearest-exact. Bilinear/bicubic
  antialias is supported for float32/64. Lanczos has no torchvision Tensor
  counterpart and requires a separate documented compatibility decision.
- `RoundOp` deliberately rounds half away from zero, unlike torch.round's
  half-to-even rule. Its expected values use torch copysign/floor/abs to express
  that documented contract, not a claim of torch.round equivalence.
- Seeded random/randn are package-specific LCG / Box-Muller streams, not torch
  RNG streams. Both accept float32/64 only; `random` remains below one after
  storage rounding. Mathematical tail and reproducibility regressions live in
  `random_factory_regression_test.dart`.
- `TensorBuffer.clone` and strided `contiguous` preserve int64/uint64 exactly;
  `integer_copy_regression_test.dart` checks values above 2^53 without floating
  comparisons. `storage.getAsDouble`, tensor element access and `toList` expose
  doubles, so they cannot represent every int64 value exactly. Use typed storage
  for exact integer inspection. Remaining integer movement paths are pending.

## Transform operations

Every concrete exported transform class is listed, including classes reached
through re-exports. Prefixes identify existing independent golden cases;
coverage beyond those cases remains subject to the release checklist.

| API | Implementation | Golden case prefix / status |
|---|---|---|
| `AbsOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `abs` |
| `AcosOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `acos` |
| `AddOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | **Pending independent oracle / contract audit** |
| `AdjustBrightnessOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | **Pending independent oracle / contract audit** |
| `AdjustContrastOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | **Pending independent oracle / contract audit** |
| `AdjustHueOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | **Pending independent oracle / contract audit** |
| `AdjustSaturationOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | **Pending independent oracle / contract audit** |
| `ArgMaxOp` | [lib/src/ops/argmax_op.dart](lib/src/ops/argmax_op.dart) | **Pending independent oracle / contract audit** |
| `ArgMinOp` | [lib/src/ops/argmax_op.dart](lib/src/ops/argmax_op.dart) | **Pending independent oracle / contract audit** |
| `AsinOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `asin` |
| `Atan2Op` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | **Pending independent oracle / contract audit** |
| `AtanOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `atan` |
| `BatchNormOp` | [lib/src/ops/batch_norm_op.dart](lib/src/ops/batch_norm_op.dart) | `batch_norm` |
| `CeilOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `ceil` |
| `CenterCropOp` | [lib/src/ops/crop_op.dart](lib/src/ops/crop_op.dart) | `center-crop` |
| `ClipOp` | [lib/src/ops/clip_op.dart](lib/src/ops/clip_op.dart) | **Pending independent oracle / contract audit** |
| `ColorJitterOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | **Pending independent oracle / contract audit** |
| `ContiguousOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `CosOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `cos` |
| `DivOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | **Pending independent oracle / contract audit** |
| `ELUOp` | [lib/src/ops/activation/elu_op.dart](lib/src/ops/activation/elu_op.dart) | `elu` |
| `ExpOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `exp` |
| `FlattenOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `FloorOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `floor` |
| `GELUOp` | [lib/src/ops/activation/gelu_op.dart](lib/src/ops/activation/gelu_op.dart) | `gelu / gelu-dense / gelu-special` |
| `GLUOp` | [lib/src/ops/activation/glu_op.dart](lib/src/ops/activation/glu_op.dart) | `glu` |
| `GatherOp` | [lib/src/ops/gather_op.dart](lib/src/ops/gather_op.dart) | **Pending independent oracle / contract audit** |
| `GaussianBlurOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `GroupNormOp` | [lib/src/ops/group_norm_op.dart](lib/src/ops/group_norm_op.dart) | `group_norm` |
| `HardsigmoidOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `hardsigmoid` |
| `HardswishOp` | [lib/src/ops/activation/swish_ops.dart](lib/src/ops/activation/swish_ops.dart) | `hardswish` |
| `HorizontalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `HsvToRgbOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | **Pending independent oracle / contract audit** |
| `IdentityOp` | [lib/src/ops/transform_op.dart](lib/src/ops/transform_op.dart) | **Pending independent oracle / contract audit** |
| `InstanceNormOp` | [lib/src/ops/instance_norm_op.dart](lib/src/ops/instance_norm_op.dart) | `instance_norm` |
| `LayerNormOp` | [lib/src/ops/layer_norm_op.dart](lib/src/ops/layer_norm_op.dart) | `layer_norm` |
| `LayoutConvertOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `LeakyReLUOp` | [lib/src/ops/activation/relu_ops.dart](lib/src/ops/activation/relu_ops.dart) | `leaky_relu` |
| `LogOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `log` |
| `LpNormalizeOp` | [lib/src/ops/lp_normalize_op.dart](lib/src/ops/lp_normalize_op.dart) | `lp / lp-special` |
| `MaskedFillOp` | [lib/src/ops/masked_fill_op.dart](lib/src/ops/masked_fill_op.dart) | **Pending independent oracle / contract audit** |
| `MishOp` | [lib/src/ops/activation/mish_op.dart](lib/src/ops/activation/mish_op.dart) | `mish` |
| `MulOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | **Pending independent oracle / contract audit** |
| `NegOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `neg` |
| `NormalizeOp` | [lib/src/ops/normalize_op.dart](lib/src/ops/normalize_op.dart) | `normalize / preset` |
| `PadOp` | [lib/src/ops/pad_op.dart](lib/src/ops/pad_op.dart) | **Pending independent oracle / contract audit** |
| `PermuteOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `PositionalEncodingOp` | [lib/src/ops/positional_encoding_op.dart](lib/src/ops/positional_encoding_op.dart) | **Pending independent oracle / contract audit** |
| `PowOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | **Pending independent oracle / contract audit** |
| `RMSNormOp` | [lib/src/ops/rms_norm_op.dart](lib/src/ops/rms_norm_op.dart) | `rms_norm` |
| `RandomCropOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `RandomErasingOp` | [lib/src/ops/random_erasing_op.dart](lib/src/ops/random_erasing_op.dart) | **Pending independent oracle / contract audit** |
| `RandomHorizontalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `RandomVerticalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `ReLUOp` | [lib/src/ops/activation/relu_ops.dart](lib/src/ops/activation/relu_ops.dart) | `relu` |
| `RepeatOp` | [lib/src/ops/repeat_op.dart](lib/src/ops/repeat_op.dart) | **Pending independent oracle / contract audit** |
| `ReshapeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `ResizeNormalizeFusedOp` | [lib/src/ops/fused_ops.dart](lib/src/ops/fused_ops.dart) | **Pending independent oracle / contract audit** |
| `ResizeOp` | [lib/src/ops/resize_op.dart](lib/src/ops/resize_op.dart) | `resize / resize-aa` |
| `ResizeShortestOp` | [lib/src/ops/resize_op.dart](lib/src/ops/resize_op.dart) | `shortest` |
| `RgbToGrayscaleOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | **Pending independent oracle / contract audit** |
| `RgbToHsvOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | **Pending independent oracle / contract audit** |
| `RollOp` | [lib/src/ops/roll_op.dart](lib/src/ops/roll_op.dart) | **Pending independent oracle / contract audit** |
| `RoundOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `round-half-away (deliberate difference from torch.round)` |
| `SELUOp` | [lib/src/ops/activation/selu_op.dart](lib/src/ops/activation/selu_op.dart) | `selu` |
| `ScaleOp` | [lib/src/ops/normalize_op.dart](lib/src/ops/normalize_op.dart) | **Pending independent oracle / contract audit** |
| `SiLUOp` | [lib/src/ops/activation/swish_ops.dart](lib/src/ops/activation/swish_ops.dart) | `silu` |
| `SigmoidOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `sigmoid` |
| `SinOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `sin` |
| `SliceOp` | [lib/src/ops/slice_op.dart](lib/src/ops/slice_op.dart) | **Pending independent oracle / contract audit** |
| `SoftmaxOp` | [lib/src/ops/activation/softmax_op.dart](lib/src/ops/activation/softmax_op.dart) | `softmax` |
| `SqrtOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `sqrt` |
| `SqueezeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `SubOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | **Pending independent oracle / contract audit** |
| `TanOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `tan` |
| `TanhOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `tanh` |
| `TileOp` | [lib/src/ops/tile_op.dart](lib/src/ops/tile_op.dart) | **Pending independent oracle / contract audit** |
| `ToImageOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | **Pending independent oracle / contract audit** |
| `ToTensorOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | **Pending independent oracle / contract audit** |
| `TopKOp` | [lib/src/ops/topk_op.dart](lib/src/ops/topk_op.dart) | **Pending independent oracle / contract audit** |
| `TypeCastOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | **Pending independent oracle / contract audit** |
| `UnsqueezeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | **Pending independent oracle / contract audit** |
| `VerticalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `WhereOp` | [lib/src/ops/where_op.dart](lib/src/ops/where_op.dart) | **Pending independent oracle / contract audit** |

## Core, convenience APIs and execution infrastructure

| Public surface | Evidence / remaining gate |
|---|---|
| `TensorBuffer` constructor; shape/strides/storageOffset/memoryFormat; dtype/rank/numel/sizeInBytes/isContiguous/data/dataAsFloat32List | Offset and storage regression tests; constructor bounds, shape metadata and scalar/empty consistency audit pending |
| `transpose`, `reshape`, `squeeze`, `unsqueeze`, `contiguous`, `clone`, element access, `toList`, `computeStrides` | Exact integer clone/contiguous regressions; independent core/view goldens pending |
| `zeros`, `ones`, `full`, `uninitialized`, `eye`, `linspace`, `arange`, `fromFloat32List`, `fromFloat64List`, `fromUint8List` | Existing factory tests; dtype-wide independent goldens pending |
| `random`, `randn` | Deliberately different RNG; documented contract and mathematical regression suite above |
| `sum`, `mean`, `min`, `max`, `sumAxis`, `meanAxis`, `minAxis`, `maxAxis`, `argmax`, `argmin`, `argmaxAxis`, `argminAxis` | Independent reductions including ties, NaN, dtype and view tests pending |
| `stack`, `concat`, `split`, `chunk`, `tensorWhere`, top-k extension | View/sentinel regressions; independent integer/value/tie cases pending |
| `TensorViewExtension` (`sliceFirst`, `isViewable`, `toChannelsLast`, `toChannelsFirst`, `flatten`, `select`, `narrow`) | Existing utility tests; exact layout/index audit pending |
| `DType`, `MemoryFormat`, `TensorStorage`, typed views, buffer pool, dtype dispatcher, tensor indexing, `SimdOps` | Native storage/utility contracts, not separate PyTorch numerical operations; dtype/memory/SIMD-tail audit pending |
| `TransformOp`, `InPlaceTransform`, `RequiresContiguous`, `OperationCapabilities`, error types/messages | Native composition/validation contracts; tests must cover invalid arguments and mutation boundaries |
| `TensorPipeline` run/runAsync/call/shape validation/composition and `PipelinePresets` factories | All preset values tested in sync/forced isolate/fallback modes, HWC/NHWC and uint8/float32; remaining custom/fused paths pending |

No row marked pending may be treated as passed solely because another row or
CI is green. Finish the pending audit before changing the package to 1.0.0.
