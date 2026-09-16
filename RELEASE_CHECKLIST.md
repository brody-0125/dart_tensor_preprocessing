# 1.0.0 release work in progress

This is an execution ledger, not a declaration that the package is ready.
Original scope: independent PyTorch expectations, pinned network fixtures,
bug fixes, complete supported-API audit, CI, documentation, and verified pub.dev
1.0.0 publication. PR #65 remains draft until all gates below are satisfied.

## Completed foundation

- [x] Pin torch 2.10.0+cpu / torchvision 0.25.0+cpu and Python dependencies.
- [x] Record exact Linux CPython 3.12 wheel URLs/SHA-256 for all 13 dependencies;
  CI installs with --require-hashes and the golden integrity test binds both locks.
- [x] Generate 161 offline operation/preset cases without invoking Dart.
- [x] Verify shape, dtype, every numeric element, non-finite values, and hashes.
- [x] Prove the comparator rejects wrong finite values and unexpected NaN.
- [x] Download fixed-commit RGB/grayscale/RGBA PNGs, verify encoded/decoded hashes.
- [x] Compare full ImageNet 256→224, CLIP 224, detection 640 tensors; reject non-RGB.
- [x] Test all presets with HWC/NHWC, uint8/float32, sync/forced isolate/fallback.
- [x] Fix view bounds, Tanh overflow, unseeded seed collisions, preset layout/batch.
- [x] Fix nearest, bicubic, area, shortest-edge sizing, center-crop padding/rounding.
- [x] Add antialias float32/64 and verify align_corners combinations.
- [x] Extend offset tests to gather/topk/stack/concat/split/masks/isolate transport.
- [x] Reproduce and fix int64/uint64 clone and strided contiguous-copy rounding
  above 2^53; exact-integer tests verify offset bounds and independent storage.
- [x] Establish Linux canonical goldens; Windows/Linux differences stay within existing tolerances.
- [x] CI success at cc85cd5: Linux Dart 3.9.0/stable, Windows/macOS stable, format,
  analyze, dry-run, network, and exact Linux oracle regeneration (run 35088804903).
- [x] Local full suite with network enabled: 1,662 passed against Linux goldens.
- [x] Remove tracked .DS_Store; publish archive excludes fixture/tooling payloads.
- [x] Clean-commit publish dry-run: 0 warnings (package still version 0.9.0).

## Required before 1.0.0

- [ ] Finish the public API map in COMPATIBILITY.md: all 84 concrete transforms
  are inventoried, with pending rows remaining explicit release gates.
- [x] Expand normalization goldens: Batch/Layer/Group/Instance/RMS/Lp, affine,
  epsilon, constant inputs, invalid axes/shapes, in-place view sentinels.
  Float32/64 coverage adds 270 cases (431 total), including non-contiguous
  views and Lp non-finite inputs. Fix Lp denominator and validate epsilon/order.
  Canonical Linux regeneration passed in runs 35090715234 and 35090718980.
- [ ] Finish core/indexing/reduction goldens: transpose/reshape/clone/contiguous,
  integer precision, scalar/empty restrictions, dtype conversion, sum/mean/min/max,
  argmin/max/topk (including ties), gather/slice/split/concat/repeat/tile/roll/where.
  Added 752 cases (1,307 total): copy/index operations with exact int64,
  scalar/single/multi-axis reductions, dtypes, NaN/ties, offsets/strides,
  top-k indices and integer overflow/adjacent values. Fixed discovered copy,
  reduction, repeated-roll and index validation defects. CI runs 35094405848
  and 35094410310 passed, including exact fixed-CPU regeneration.
  Added another 150 factory/cast cases (1,457 total): all destination dtypes,
  sequence truncation, legacy cast rounding/clamps and exact integer movement.
  Fixed dtype allocation in eye/linspace/arange and integer cast precision.
  Factory/cast CI runs 35095300717 and 35095305969 passed all gates.
  Added 136 view/remaining-factory cases (1,593 total), plus native constructor
  bounds, immutable metadata, negative axes, reshape and alias/copy tests.
  Single-element squeeze now retains [1]; all scalar/empty restrictions are
  documented as a migration change. Local network-inclusive suite: 3,138 passed.
  View increment CI runs 35095978514 and 35095984086 passed all gates.
  Select/unbind/narrow add 48 cases (1,641 total); vector selection retains [1]
  and shared exact integer storage. Local network-inclusive suite: 3,188 passed.
  CI runs 35096599370 and 35096602375 passed all gates.
  Layout conversion adds 48 cases (1,689 total): both logical directions,
  view/copy modes and physical channels-last regression. Fixed skipped NCHW
  conversion and sliceFirst stride reinterpretation. Local full suite: 3,237 passed.
  Layout CI runs 35097013269 and 35097017677 passed all gates.
  Identity/Contiguous/Permute/Reshape/Flatten add 60 cases (1,749 total), with
  explicit contiguous preparation and rejection checks for strided reshape.
  Fixed invalid permutation shape inference and mutable operation parameters.
  Local network-inclusive suite: 3,298 passed; static analysis clean.
  Shape CI runs 35097349704 and 35097354321 passed all gates.
  Add/Sub/Mul/Div/Pow float32/64 scalar/tensor cases add 54 goldens (1,803 total).
  Fixed mismatched-shape acceptance and overlapping in-place tensor operands.
  Local network-inclusive suite: 3,354 passed. Integer/mixed arithmetic dtype
  contracts remain open; float goldens do not prove integer parity.
  Arithmetic CI runs 35097708542 and 35097716567 passed all gates.
  Added 96 exact signed-integer, mixed-float and non-finite cases (1,899 total).
  Fixed integer add/sub/mul double rounding and Pow sqrt/rsqrt non-finite semantics.
  Local network-inclusive suite: 3,450 passed. Unsigned, fractional integer,
  integer division/power and remaining numeric contracts still require audit.
  CI runs 35098131246 and 35098136497 passed all gates.
  Added 131 division/power/all-integer-dtype/fractional/mixed-operand cases
  (2,030 total). Fixed exact integer division/power and prevalidated zero
  divisors. Documented native integer wrapping and double-fallback limitations.
  Local network-inclusive suite: 3,582 passed; static analysis clean.
  Integer arithmetic CI runs 35098495443 and 35098501058 passed all gates.
  Scale/Clip/Atan2 add 48 float32/64 cases (2,078 total), including NaN/Inf,
  cancellation, scalar/tensor, offsets and strided input. Fixed severe Scale
  cancellation, Atan2 shape/overlap handling and Clip NaN bounds. Integer
  contracts for these three operations remain pending. Local full suite: 3,632 passed.
  Added 27 integer recipes/exact int64 clip cases (2,105 total); fixed unchanged
  Clip values losing integer precision. Local network-inclusive suite: 3,659 passed.
  Numeric CI 35098903435/35098907602 passed every job except exact oracle
  regeneration: both produced identical artifacts with three float32 Atan2
  cases differing by at most 1.1920928955078125e-7. Reviewed every changed value
  under unchanged tolerances, verified network payloads unchanged and adopted
  canonical values. Latest commit must verify exact regeneration again.
  Canonical adoption CI 35099322211/35099325397 passed all gates.
  ToTensor/ToImage add 216 cases (2,321 total), including channel/batch/flag/
  offset/stride combinations and explicit non-torch rounding/normalization
  recipes. No implementation change was required for these valid inputs.
  Local network-inclusive suite passed 3,875 tests before one invalid-rank
  regression was added; that regression is verified separately.
  Image conversion CI 35099624249/35099627341 passed all gates.
  RGB/grayscale/HSV adds 144 cases (2,465 total): normalized float32/64 and
  promoted uint8/int64 inputs, primaries/ties/gradients, batch/view variants.
  Fixed shape-inference rank/channel validation; full local network suite:
  4,021 passed, static analysis clean. Color adjustment/augmentation remains pending.
  Color-space CI 35099984523/35099989484 passed all gates.
  Adjustment/jitter adds 324 cases (2,789 total): fixed factors, dtype/batch/view
  combinations, neutral jitter and a recorded seed schedule with independent
  torch image values. Preserve documented package recipes rather than claiming
  torchvision default equivalence. Fixed integer hue half-turn truncation,
  non-finite factors and shape inference. Full local network suite: 4,346 passed.
  Color adjustment CI 35100922731/35100928393 passed all gates.
  Flip variants add 144 cases (2,933 total): both axes, CHW/NCHW, float32/64,
  uint8/exact int64, offset/strided and random probability 0/1. Fixed integer
  copy precision, finite probability and inferred-rank validation. Full local
  network suite: 4,491 passed; static analysis clean.
  RandomCrop adds 120 cases (3,053 total): all ten dtypes, full/partial crops,
  CHW/NCHW and offset/strided views, using recorded native seed coordinates
  and independent torch slices. Fixed integer copy precision and inferred
  rank/size validation. Full local network suite: 4,612 passed; analysis clean.
  GaussianBlur adds 240 cases (3,293 total): explicit symmetric-padding torch
  recipe, kernels 1/3/7, tiny/huge sigma, small images, dtype/batch/view variants.
  Fixed repeated reflection bounds, sigma underflow, exact identity copying and
  parameter/shape validation. Full local network suite: 4,853 passed; analysis clean.
- [ ] Expand color/augmentation: RGB/HSV/grayscale, brightness/contrast/saturation/
  hue, blur, fixed crop/flip/erase/jitter parameters. Do not equate RNG seeds across languages.
- [ ] Cover remaining activations/math/trig/positional operations and supported
  dtypes, non-finite edge cases, non-contiguous input, and invalid parameters.
  Unary math/trig, GELU/GLU and documented half-away RoundOp now add 124 cases
  (555 total); positional/binary arithmetic/remaining edge contracts are pending.
  Exact GELU float64 accuracy is fixed and checked on an 801-point dense grid.
  Two independent Linux runs regenerated identical files; reviewed 52 last-bit
  differences (max 4.64e-16) and adopted the Linux corpus without changing tolerance.
- [ ] Compare fused operations and SIMD/scalar tails against independent PyTorch.
- [x] Fix random factories' float64 allocation and Box-Muller math; reject
  integer dtype, exclude the uniform endpoint after float32 rounding, skip
  zero before log, preserve normal tails, and document package-specific seeds.
  Targeted regression tests reproduce all three defects before their fixes.
- [ ] Audit storage copy paths for int64 values beyond double's exact range;
  `_copyToContiguous` is fixed and tested; audit remaining movement/conversion
  operations. Do not hide precision loss with floating-point comparisons.
- [ ] Finish any bugs found by expanded parity tests and preserve invalid-input checks.
- [x] Review antialias precomputation/performance at ordinary image sizes;
  reuse weights/scratch across batch and channel planes and use direct typed
  access. Full 2,086-test suite passes. Local before/after timings and raw
  samples are recorded in benchmark/ANTIALIAS_RESULTS.md; no universal speed claim.
- [ ] Finalize compatibility table, migration guide, README snippets/examples,
  random/dtype/view contracts, and fixture dependency hashes/provenance.
- [ ] Validate fixture licenses/attribution and package contents at final version.
- [ ] Change pubspec/CHANGELOG/README to 1.0.0 only when functional audit is complete.
- [ ] Run all gates on the actual final commit; verify required case counts, no hidden skips.
- [ ] Merge the reviewed release, publish through existing pub.dev authorization,
  tag/release the identical commit, and install 1.0.0 in a clean consumer project.
- [ ] Verify pub.dev metadata/archive and consumer smoke against golden subset.

## Operational notes

The implementation checkout is the release/1.0.0 branch. Python 3.12 was used
instead of the plan's provisional 3.11 because the available local runtime is
3.12 and the pinned oracle pair supports it. Exact fixture regeneration belongs
to Linux CI; use `--output` for Windows investigations. When changing the
generator, review Linux CI regeneration artifacts before updating canonical
goldens. All source changes must be tested against those same goldens.

No 1.0.0 version, tag, GitHub release, or pub.dev publication has been created yet.

Canonical unary regeneration exposed hardware-dependent last bits even with
ATen/MKL flags. The oracle job now uses Ubuntu 24.04 and QEMU Haswell-v4;
independent runs 35092487771 and 35092491682 regenerated identical files.
Reviewed and adopted six sqrt cases with one-ULP changes within unchanged
tolerances; all other cases and network payloads stayed identical. Native Dart CI remains unchanged.

Padding increment: 720 independent cases (4,013 total), all ten dtypes, modes,
batch/view variants and padding beyond dimensions. Reproduced 144 integer
precision failures before fixing eight copy paths. Network-inclusive suite:
5,573 passed, plus a separate invalid-rank regression; static analysis clean.
GaussianBlur CI 35102092956/35102099683 passed all gates.

Positional encoding: 540 independent torch sin/cos cases (4,553 total), all ten
dtypes, odd/even model dimensions, rank 2/3/4 and offset/in-place/strided inputs.
Fixed base and inferred-shape validation; corrected RoPE/Embedding claims.
Network-inclusive full suite: 6,114 passed before the additional parameter
regression; that regression passed separately.

RandomErasing: 240 independent torch assignment cases (4,793 total) across all
dtypes, views/batches and skip/full/partial/impossible regions. Added native
uniform-stream and integer-fill tests; reject non-finite parameters and invalid
inferred rank. Full network suite results recorded in the progress artifact.

Fused resize/normalize float32/64: 120 independent torch double-interpolate and
normalize cases (4,913 total), alignCorners, singleton/identity/up/down sizes,
64-pixel blocking boundary, batches and offset/strided views. Shape inference
now validates rank/channels. Integer and remaining parameter contracts stay open.

Normalization parameter audit: reproduced mutation of validated std to zero
producing -Infinity instead of 2. Both NormalizeOp and ResizeNormalizeFusedOp
now copy mean/std into unmodifiable lists; regression verifies caller and
public-field mutations. Remaining fused integer audit remains open.

Fused integer increment: 288 exact-comparison cases (5,201 total), all eight
integer dtypes, positive in-range values, both alignCorners settings, batches
and offset/strided views. Independent double torch interpolation followed by
normalization and integer cast passes. Out-of-range/non-finite cases remain open.

Fused numerical edge: std=1e-320 with resized==mean produced NaN instead of
zero because the reciprocal overflowed. Reproduced before fixing direct division;
float32/float64/int64 regression covers the corrected result.

SIMD normalization reproduces the same reciprocal overflow for subnormal std.
Guarded direct-division fallback now covers float32/64, scalar/vector paths and
tails at lengths 1/4/9/128; standard-range SIMD behavior remains unchanged.

Storage contract audit: exact native clamp/wrap/truncate expectations, rejected
NaN/Inf writes leave storage unchanged, all ten dtype bounded-view clones,
independent copies, byte sizes and bounds validated in storage_contract_test.
Typed-view clones preserve int64/uint64 values above 2^53. Buffer-pool/dispatcher
and remaining SIMD utility audit are separate pending gates.

Fixture review-size correction: operations JSON changed from 834,349 to 5,203
lines (12,712,023 to 5,631,250 bytes), one case per line. Decoded data equality
verified for all 5,201 cases; generator and manifest updated together. Full
network-inclusive suite: 6,770 passed. No case or expectation was removed.

BufferPool audit found duplicate release enqueued the same object twice.
A bucket-local duplicate check (maximum eight entries) fixes double lending;
regression reproduces the failure and checks independent acquired buffers.
Ownership transfer and the prohibition on separately releasing aliases are
now explicit in the API documentation.

Dispatcher audit found non-contiguous mutation destinations were copied, so
callback writes disappeared. dispatchVoid and dispatchPair now reject strided
destinations before callbacks; read/input dispatch retains contiguous copying.
Regression reproduces the old silent success and checks callbacks stay uncalled.
