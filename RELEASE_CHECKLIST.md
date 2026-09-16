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
