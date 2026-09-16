# 1.0.0 release work in progress

This is an execution ledger, not a declaration that the package is ready.
Original scope: independent PyTorch expectations, pinned network fixtures,
bug fixes, complete supported-API audit, CI, documentation, and verified pub.dev
1.0.0 publication. PR #65 remains draft until all gates below are satisfied.

## Completed foundation

- [x] Pin torch 2.10.0+cpu / torchvision 0.25.0+cpu and Python dependencies.
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
- [ ] Expand core/indexing/reduction goldens: transpose/reshape/clone/contiguous,
  integer precision, scalar/empty restrictions, dtype conversion, sum/mean/min/max,
  argmin/max/topk (including ties), gather/slice/split/concat/repeat/tile/roll/where.
- [ ] Expand color/augmentation: RGB/HSV/grayscale, brightness/contrast/saturation/
  hue, blur, fixed crop/flip/erase/jitter parameters. Do not equate RNG seeds across languages.
- [ ] Cover remaining activations/math/trig/positional operations and supported
  dtypes, non-finite edge cases, non-contiguous input, and invalid parameters.
  Unary math/trig, GELU/GLU and documented half-away RoundOp now add 124 cases
  (555 total); positional/binary arithmetic/remaining edge contracts are pending.
  Exact GELU float64 accuracy is fixed and checked on an 801-point dense grid.
- [ ] Compare fused operations and SIMD/scalar tails against independent PyTorch.
- [x] Fix random factories' float64 allocation and Box-Muller math; reject
  integer dtype, exclude the uniform endpoint after float32 rounding, skip
  zero before log, preserve normal tails, and document package-specific seeds.
  Targeted regression tests reproduce all three defects before their fixes.
- [ ] Audit storage copy paths for int64 values beyond double's exact range;
  `_copyToContiguous` is fixed and tested; audit remaining movement/conversion
  operations. Do not hide precision loss with floating-point comparisons.
- [ ] Finish any bugs found by expanded parity tests and preserve invalid-input checks.
- [ ] Review antialias precomputation/performance at ordinary image sizes; no speed claims without measurement.
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
