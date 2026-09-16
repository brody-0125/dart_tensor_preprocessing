# Compatibility audit for 1.0.0 (in progress)

This is an evidence map, not a claim that all APIs match PyTorch. A passing
legacy test alone does not close an independent-oracle requirement. The
manifest and generator under `test/fixtures/pytorch` / `scripts` are the
source of truth for the named case prefixes below.

## Established contracts

- Color adjustment retains the package's existing recipes: additive brightness,
  per-channel-mean contrast, HSV saturation and wrapped hue, all clamped to [0,1].
  These are not torchvision's default brightness/contrast/saturation recipes.
  Float64 hue keeps double precision instead of torchvision adjust_hue's float32
  conversion. Integer normalized colors are 0/1 and outputs truncate; their
  identity and half-turn references use exact torch algebra to avoid HSV
  roundoff crossing integer boundaries. `adjust-*` and `jitter-*` verify these
  recipes, including a recorded native Dart seed-41 schedule with independently
  computed torch image values. Dart seeds do not imply PyTorch RNG parity.

- RGB/HSV conversion expects finite normalized channels in [0,1], as stated
  in its API documentation. Integer inputs are converted numerically to float32
  without dividing by 255; integer HSV tests therefore use binary channel values.
  Grayscale uses BT.601 coefficients and promotes integer input to float32.
  `color-*` covers primaries, black/white/ties, gradients, batches, offsets and
  non-contiguous input. HSV references call the pinned torchvision 0.25.0
  `_functional_tensor._rgb2hsv/_hsv2rgb` implementations directly.

- ToTensor explicitly maps HWC/NHWC to CHW/NCHW and outputs float32. Its
  normalize flag divides all input dtypes by 255, including floating inputs;
  this is distinct from preset handling of already-normalized floats.
  ToImage reverses axes, optionally multiplies by 255, then rounds halfway
  away from zero and clamps to uint8. `to-tensor-*`/`to-image-*` cover 1/3/4
  channels, batches, flags, offsets and strided storage. Unsupported ranks
  are rejected by execution and shape inference.

- Scale and Atan2 retain integer output dtype using double arithmetic followed
  by truncation and storage conversion (uint8/16 clamp, other integer types wrap).
  This differs from torch dtype promotion and cannot preserve all int64 inputs.
  Clip compares integers exactly and converts only values outside its double
  bounds. Numeric integer recipes cover all eight destination dtypes; exact
  int64 clip also has offset/in-place/strided goldens.

- Binary arithmetic requires identical tensor shapes (no broadcasting).
  It retains the input dtype; mixed float32/64 goldens explicitly compute in
  double then cast back, rather than claiming PyTorch promotion equivalence.
  Integral operands use native signed-64-bit arithmetic (including wrapping),
  with truncating integer division. Nonnegative integer powers use the same
  integer domain. Uint8/uint16 outputs clamp; other integer buffers wrap.
  Fractional scalars or floating tensor operands use double arithmetic then
  truncate integer outputs, so that path cannot retain every int64 value.
  Uint64 high bits follow native Dart's signed representation, not arbitrary
  unsigned arithmetic. All eight integer destination types have explicit recipe
  goldens; signed int64 precision is also checked above 2^53. Integer zero
  divisors are rejected before mutation. Other invalid floating-to-integer
  conversions follow Dart conversion errors. In-place calls are not generally
  transactional on such errors. Overlapping tensor operands are snapshotted.

- Rank-zero and empty tensors are unsupported. Squeeze retains `[1]` for a
  single element; both squeeze and unsqueeze normalize negative axes.
  Shape/stride metadata is immutable, strides nonnegative, and every reachable
  storage index must be in bounds. Zero-stride read views are supported.
  `view_contract_test.dart` verifies aliases, copies and invalid metadata.

- `eye` supports every dtype. `linspace`/`arange` compute a double sequence,
  then truncate for integer destinations; this is deliberately different from
  PyTorch's integer-endpoint kernels. Endpoints/step must be finite and the
  sequence nonempty. Public arguments are doubles, so they cannot express every
  int64 value. `factory-*` goldens cover all ten dtypes.
- `TypeCastOp` preserves same-dtype view identity. Integer sources are read
  exactly; floating sources round halfway away from zero for integer outputs.
  Int8/int16/uint8/uint16/uint32 destinations clamp, while int32/int64/uint64
  use typed-buffer wrapping. `cast-*` goldens express this existing contract
  with explicit torch rounding/clamping recipes. Native Dart exposes uint64
  high-bit values as signed integers; no JavaScript 64-bit parity is promised.

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
  for exact integer inspection. Gather/slice/stack/concat/split/where/tile/roll/top-k now have exact int64
  cases; integer casting and other image movement paths are still pending.

## Transform operations

Every concrete exported transform class is listed, including classes reached
through re-exports. Prefixes identify existing independent golden cases;
coverage beyond those cases remains subject to the release checklist.

| API | Implementation | Golden case prefix / status |
|---|---|---|
| `AbsOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `abs` |
| `AcosOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `acos` |
| `AddOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | binary-*/integer-* goldens: floating, integer, mixed/fractional recipes, non-finite values, offsets/strides and in-place boundaries; explicit dtype contract above |
| `AdjustBrightnessOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | adjust-*/jitter-* independent recipes, normalized dtype/batch/view variants, exact integer identities, recorded native seed schedule and finite-factor/shape validation |
| `AdjustContrastOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | adjust-*/jitter-* independent recipes, normalized dtype/batch/view variants, exact integer identities, recorded native seed schedule and finite-factor/shape validation |
| `AdjustHueOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | adjust-*/jitter-* independent recipes, normalized dtype/batch/view variants, exact integer identities, recorded native seed schedule and finite-factor/shape validation |
| `AdjustSaturationOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | adjust-*/jitter-* independent recipes, normalized dtype/batch/view variants, exact integer identities, recorded native seed schedule and finite-factor/shape validation |
| `ArgMaxOp` | [lib/src/ops/argmax_op.dart](lib/src/ops/argmax_op.dart) | `core_reduce (delegated argmaxAxis) / reduce-adjacent / ties / nan` |
| `ArgMinOp` | [lib/src/ops/argmax_op.dart](lib/src/ops/argmax_op.dart) | `core_reduce (delegated argminAxis) / reduce-adjacent / ties / nan` |
| `AsinOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `asin` |
| `Atan2Op` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | numeric-*/clip-*/scale-*/atan2-* independent floating and integer recipes; offsets/strides, NaN/Inf, cancellation, shape/overlap and bound checks; dtype limits documented above |
| `AtanOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `atan` |
| `BatchNormOp` | [lib/src/ops/batch_norm_op.dart](lib/src/ops/batch_norm_op.dart) | `batch_norm` |
| `CeilOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `ceil` |
| `CenterCropOp` | [lib/src/ops/crop_op.dart](lib/src/ops/crop_op.dart) | `center-crop` |
| `ClipOp` | [lib/src/ops/clip_op.dart](lib/src/ops/clip_op.dart) | numeric-*/clip-*/scale-*/atan2-* independent floating and integer recipes; offsets/strides, NaN/Inf, cancellation, shape/overlap and bound checks; dtype limits documented above |
| `ColorJitterOp` | [lib/src/ops/color_jitter_op.dart](lib/src/ops/color_jitter_op.dart) | adjust-*/jitter-* independent recipes, normalized dtype/batch/view variants, exact integer identities, recorded native seed schedule and finite-factor/shape validation |
| `ContiguousOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | Independent shape-* float32/64/int32/int64 values, offset/strided cases and shape inference; contiguous preparation is explicit for reshape/flatten and rejection is tested |
| `CosOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `cos` |
| `DivOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | binary-*/integer-* goldens: floating, integer, mixed/fractional recipes, non-finite values, offsets/strides and in-place boundaries; explicit dtype contract above |
| `ELUOp` | [lib/src/ops/activation/elu_op.dart](lib/src/ops/activation/elu_op.dart) | `elu` |
| `ExpOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `exp` |
| `FlattenOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | Independent shape-* float32/64/int32/int64 values, offset/strided cases and shape inference; contiguous preparation is explicit for reshape/flatten and rejection is tested |
| `FloorOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `floor` |
| `GELUOp` | [lib/src/ops/activation/gelu_op.dart](lib/src/ops/activation/gelu_op.dart) | `gelu / gelu-dense / gelu-special` |
| `GLUOp` | [lib/src/ops/activation/glu_op.dart](lib/src/ops/activation/glu_op.dart) | `glu` |
| `GatherOp` | [lib/src/ops/gather_op.dart](lib/src/ops/gather_op.dart) | `index-*gather` |
| `GaussianBlurOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `GroupNormOp` | [lib/src/ops/group_norm_op.dart](lib/src/ops/group_norm_op.dart) | `group_norm` |
| `HardsigmoidOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `hardsigmoid` |
| `HardswishOp` | [lib/src/ops/activation/swish_ops.dart](lib/src/ops/activation/swish_ops.dart) | `hardswish` |
| `HorizontalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `HsvToRgbOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | color-* goldens: float32/64, integer promotion, normalized domain, batch/offset/strided inputs and invalid rank/channel contracts |
| `IdentityOp` | [lib/src/ops/transform_op.dart](lib/src/ops/transform_op.dart) | Independent shape-* float32/64/int32/int64 values, offset/strided cases and shape inference; contiguous preparation is explicit for reshape/flatten and rejection is tested |
| `InstanceNormOp` | [lib/src/ops/instance_norm_op.dart](lib/src/ops/instance_norm_op.dart) | `instance_norm` |
| `LayerNormOp` | [lib/src/ops/layer_norm_op.dart](lib/src/ops/layer_norm_op.dart) | `layer_norm` |
| `LayoutConvertOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | `layout-*`: float32/64/int32/int64, offset/strided input, both directions and forceContiguous settings, independent values and round trips; physical channels-last regression |
| `LeakyReLUOp` | [lib/src/ops/activation/relu_ops.dart](lib/src/ops/activation/relu_ops.dart) | `leaky_relu` |
| `LogOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `log` |
| `LpNormalizeOp` | [lib/src/ops/lp_normalize_op.dart](lib/src/ops/lp_normalize_op.dart) | `lp / lp-special` |
| `MaskedFillOp` | [lib/src/ops/masked_fill_op.dart](lib/src/ops/masked_fill_op.dart) | `masked-fill` |
| `MishOp` | [lib/src/ops/activation/mish_op.dart](lib/src/ops/activation/mish_op.dart) | `mish` |
| `MulOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | binary-*/integer-* goldens: floating, integer, mixed/fractional recipes, non-finite values, offsets/strides and in-place boundaries; explicit dtype contract above |
| `NegOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `neg` |
| `NormalizeOp` | [lib/src/ops/normalize_op.dart](lib/src/ops/normalize_op.dart) | `normalize / preset` |
| `PadOp` | [lib/src/ops/pad_op.dart](lib/src/ops/pad_op.dart) | **Pending independent oracle / contract audit** |
| `PermuteOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | Independent shape-* float32/64/int32/int64 values, offset/strided cases and shape inference; contiguous preparation is explicit for reshape/flatten and rejection is tested |
| `PositionalEncodingOp` | [lib/src/ops/positional_encoding_op.dart](lib/src/ops/positional_encoding_op.dart) | **Pending independent oracle / contract audit** |
| `PowOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | binary-*/integer-* goldens: floating, integer, mixed/fractional recipes, non-finite values, offsets/strides and in-place boundaries; explicit dtype contract above |
| `RMSNormOp` | [lib/src/ops/rms_norm_op.dart](lib/src/ops/rms_norm_op.dart) | `rms_norm` |
| `RandomCropOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `RandomErasingOp` | [lib/src/ops/random_erasing_op.dart](lib/src/ops/random_erasing_op.dart) | **Pending independent oracle / contract audit** |
| `RandomHorizontalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `RandomVerticalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `ReLUOp` | [lib/src/ops/activation/relu_ops.dart](lib/src/ops/activation/relu_ops.dart) | `relu` |
| `RepeatOp` | [lib/src/ops/repeat_op.dart](lib/src/ops/repeat_op.dart) | `index-*repeat` |
| `ReshapeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | Independent shape-* float32/64/int32/int64 values, offset/strided cases and shape inference; contiguous preparation is explicit for reshape/flatten and rejection is tested |
| `ResizeNormalizeFusedOp` | [lib/src/ops/fused_ops.dart](lib/src/ops/fused_ops.dart) | **Pending independent oracle / contract audit** |
| `ResizeOp` | [lib/src/ops/resize_op.dart](lib/src/ops/resize_op.dart) | `resize / resize-aa` |
| `ResizeShortestOp` | [lib/src/ops/resize_op.dart](lib/src/ops/resize_op.dart) | `shortest` |
| `RgbToGrayscaleOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | color-* goldens: float32/64, integer promotion, normalized domain, batch/offset/strided inputs and invalid rank/channel contracts |
| `RgbToHsvOp` | [lib/src/ops/color_space_op.dart](lib/src/ops/color_space_op.dart) | color-* goldens: float32/64, integer promotion, normalized domain, batch/offset/strided inputs and invalid rank/channel contracts |
| `RollOp` | [lib/src/ops/roll_op.dart](lib/src/ops/roll_op.dart) | `index-*roll (including repeated axes)` |
| `RoundOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `round-half-away (deliberate difference from torch.round)` |
| `SELUOp` | [lib/src/ops/activation/selu_op.dart](lib/src/ops/activation/selu_op.dart) | `selu` |
| `ScaleOp` | [lib/src/ops/normalize_op.dart](lib/src/ops/normalize_op.dart) | numeric-*/clip-*/scale-*/atan2-* independent floating and integer recipes; offsets/strides, NaN/Inf, cancellation, shape/overlap and bound checks; dtype limits documented above |
| `SiLUOp` | [lib/src/ops/activation/swish_ops.dart](lib/src/ops/activation/swish_ops.dart) | `silu` |
| `SigmoidOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `sigmoid` |
| `SinOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `sin` |
| `SliceOp` | [lib/src/ops/slice_op.dart](lib/src/ops/slice_op.dart) | `index-*slice` |
| `SoftmaxOp` | [lib/src/ops/activation/softmax_op.dart](lib/src/ops/activation/softmax_op.dart) | `softmax` |
| `SqrtOp` | [lib/src/ops/math_op.dart](lib/src/ops/math_op.dart) | `sqrt` |
| `SqueezeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | `squeeze-*`: float32/64/int32/int64, offsets/strides, negative axes, single-element [1] contract |
| `SubOp` | [lib/src/ops/arithmetic_op.dart](lib/src/ops/arithmetic_op.dart) | binary-*/integer-* goldens: floating, integer, mixed/fractional recipes, non-finite values, offsets/strides and in-place boundaries; explicit dtype contract above |
| `TanOp` | [lib/src/ops/trig_op.dart](lib/src/ops/trig_op.dart) | `tan` |
| `TanhOp` | [lib/src/ops/activation/sigmoid_ops.dart](lib/src/ops/activation/sigmoid_ops.dart) | `tanh` |
| `TileOp` | [lib/src/ops/tile_op.dart](lib/src/ops/tile_op.dart) | `index-*tile` |
| `ToImageOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | `to-image-*`: explicit rounding/clamp recipe, 1/3/4 channels, rank3/4, flags, offsets and strides |
| `ToTensorOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | `to-tensor-*`: uint8/int64/float32/64, 1/3/4 channels, rank3/4, flags, offsets and strides |
| `TopKOp` | [lib/src/ops/topk_op.dart](lib/src/ops/topk_op.dart) | `index-*core_topk / topk-special / topk-ties` |
| `TypeCastOp` | [lib/src/ops/type_cast_op.dart](lib/src/ops/type_cast_op.dart) | `cast-*`: all destination dtypes, float32/64 and exact int64 sources, offset/strided inputs; native identity/wrapping regressions |
| `UnsqueezeOp` | [lib/src/ops/permute_op.dart](lib/src/ops/permute_op.dart) | `unsqueeze-*`: float32/64/int32/int64, offsets/strides, negative axes, alias checks |
| `VerticalFlipOp` | [lib/src/ops/augmentation_op.dart](lib/src/ops/augmentation_op.dart) | **Pending independent oracle / contract audit** |
| `WhereOp` | [lib/src/ops/where_op.dart](lib/src/ops/where_op.dart) | `index-*where` |

## Core, convenience APIs and execution infrastructure

| Public surface | Evidence / remaining gate |
|---|---|
| `TensorBuffer` constructor; shape/strides/storageOffset/memoryFormat; dtype/rank/numel/sizeInBytes/isContiguous/data/dataAsFloat32List | Offset/storage regressions and `view_contract_test.dart`: bounds, immutable metadata, scalar/empty rejection, alias/copy boundaries; layout utility audit remains below |
| `transpose`, `reshape`, `squeeze`, `unsqueeze`, `contiguous`, `clone`, element access, `toList`, `computeStrides` | Float32/64/int32/int64 offset/strided goldens for movement and squeeze/unsqueeze; invalid axes, shape/count checks and alias/copy regressions; numeric element access explicitly returns double |
| `zeros`, `ones`, `full`, `uninitialized`, `eye`, `linspace`, `arange`, `fromFloat32List`, `fromFloat64List`, `fromUint8List` | `factory-*` covers all seven generated factories and all ten dtypes; typed-list constructors retain supplied storage, covered by native storage and view regressions |
| `random`, `randn` | Deliberately different RNG; documented contract and mathematical regression suite above |
| `sum`, `mean`, `min`, `max`, `sumAxis`, `meanAxis`, `minAxis`, `maxAxis`, `argmax`, `argmin`, `argmaxAxis`, `argminAxis` | Independent single/multi/global reductions, keepDims, ties, NaN, offset/strided, integer overflow/adjacent int64 cases; integer axis mean rejected; global value API is double, scalar tensor shape is [1] |
| `stack`, `concat`, `split`, `chunk`, `tensorWhere`, top-k extension | Independent values and exact int64, offsets/strides, split/chunk part counts, top-k values/indices/NaN/ties; no broadcasting |
| `TensorViewExtension` (`sliceFirst`, `isViewable`, `toChannelsLast`, `toChannelsFirst`, `flatten`, `select`, `unbind`, `narrow`) | `view-select/unbind/narrow-*` cover values, exact integers, offset/strided storage and alias identity; vector select/unbind retain [1]. Layout/remaining utility audit pending |
| `DType`, `MemoryFormat`, `TensorStorage`, typed views, buffer pool, dtype dispatcher, tensor indexing, `SimdOps` | Native storage/utility contracts, not separate PyTorch numerical operations; dtype/memory/SIMD-tail audit pending |
| `TransformOp`, `InPlaceTransform`, `RequiresContiguous`, `OperationCapabilities`, error types/messages | Native composition/validation contracts; tests must cover invalid arguments and mutation boundaries |
| `TensorPipeline` run/runAsync/call/shape validation/composition and `PipelinePresets` factories | All preset values tested in sync/forced isolate/fallback modes, HWC/NHWC and uint8/float32; remaining custom/fused paths pending |

No row marked pending may be treated as passed solely because another row or
CI is green. Finish the pending audit before changing the package to 1.0.0.
