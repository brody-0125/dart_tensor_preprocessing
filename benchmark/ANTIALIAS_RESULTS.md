# Antialias resize timing observation

2026-09-16, Windows x64, Dart 3.13.3 JIT. Before: commit 702100e.
After: the scratch-plane/weight reuse and typed-access change in this PR.
`dart run benchmark/antialias_benchmark.dart`; five rounds, each with five warmups
and ten timed calls. Inputs are allocated before timing. Numbers are the median
of the five per-call averages, in milliseconds. This is a local observation,
not a cross-device guarantee or a CI performance threshold. Full tensor golden
tests, including batched presets and network images, validate numerical output.

| dtype | Input → output | mode | Before ms | After ms |
|---|---|---|---:|---:|
| float32 | [3, 480, 640] → 224² | bilinear | 18.657 | 4.314 |
| float32 | [3, 224, 224] → 640² | bicubic | 45.209 | 12.377 |
| float32 | [4, 3, 480, 640] → 224² | bilinear | 66.084 | 15.775 |
| float64 | [3, 480, 640] → 224² | bilinear | 24.620 | 8.075 |
| float64 | [3, 224, 224] → 640² | bicubic | 45.044 | 22.608 |
| float64 | [4, 3, 480, 640] → 224² | bilinear | 99.881 | 30.017 |

Weights and the intermediate plane are reused across channels and batch
items. Direct float32/64 list access avoids repeated dynamic dtype dispatch
and redundant explicit bounds checks inside each filter tap. Dart list access
still enforces bounds; filter arithmetic order and float32 rounding are unchanged.

## Raw samples

```json
{
  "before": [
    {
      "dtype": "float32",
      "shape": [
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        24.886200000000002,
        22.764599999999998,
        18.657,
        17.0367,
        16.6943
      ],
      "median_ms": 18.657,
      "checksum": 37.793366611003876
    },
    {
      "dtype": "float32",
      "shape": [
        3,
        224,
        224
      ],
      "size": 640,
      "mode": "bicubic",
      "samples_ms": [
        47.013,
        38.9347,
        41.743900000000004,
        53.6905,
        45.2091
      ],
      "median_ms": 45.2091,
      "checksum": 47.64570593833923
    },
    {
      "dtype": "float32",
      "shape": [
        4,
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        94.0428,
        66.0842,
        74.8742,
        59.3675,
        61.8063
      ],
      "median_ms": 66.0842,
      "checksum": 37.793366611003876
    },
    {
      "dtype": "float64",
      "shape": [
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        24.6196,
        24.5269,
        24.8607,
        24.8692,
        24.5251
      ],
      "median_ms": 24.6196,
      "checksum": 37.79337016431251
    },
    {
      "dtype": "float64",
      "shape": [
        3,
        224,
        224
      ],
      "size": 640,
      "mode": "bicubic",
      "samples_ms": [
        46.2914,
        45.1004,
        44.6172,
        44.448699999999995,
        45.044
      ],
      "median_ms": 45.044,
      "checksum": 47.64571828416862
    },
    {
      "dtype": "float64",
      "shape": [
        4,
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        93.4527,
        93.27680000000001,
        99.88119999999999,
        115.1524,
        137.50379999999998
      ],
      "median_ms": 99.88119999999999,
      "checksum": 37.79337016431251
    }
  ],
  "after": [
    {
      "dtype": "float32",
      "shape": [
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        4.1884,
        4.267600000000001,
        4.3138000000000005,
        4.334,
        4.3704
      ],
      "median_ms": 4.3138000000000005,
      "checksum": 37.793366611003876
    },
    {
      "dtype": "float32",
      "shape": [
        3,
        224,
        224
      ],
      "size": 640,
      "mode": "bicubic",
      "samples_ms": [
        12.5693,
        12.3551,
        12.6981,
        12.3774,
        12.3526
      ],
      "median_ms": 12.3774,
      "checksum": 47.64570593833923
    },
    {
      "dtype": "float32",
      "shape": [
        4,
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        15.799100000000001,
        15.744,
        15.7119,
        17.2147,
        15.7754
      ],
      "median_ms": 15.7754,
      "checksum": 37.793366611003876
    },
    {
      "dtype": "float64",
      "shape": [
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        7.869800000000001,
        8.0749,
        7.7732,
        8.225700000000002,
        8.3805
      ],
      "median_ms": 8.0749,
      "checksum": 37.79337016431251
    },
    {
      "dtype": "float64",
      "shape": [
        3,
        224,
        224
      ],
      "size": 640,
      "mode": "bicubic",
      "samples_ms": [
        22.1837,
        22.6078,
        23.9069,
        23.751099999999997,
        22.336599999999997
      ],
      "median_ms": 22.6078,
      "checksum": 47.64571828416862
    },
    {
      "dtype": "float64",
      "shape": [
        4,
        3,
        480,
        640
      ],
      "size": 224,
      "mode": "bilinear",
      "samples_ms": [
        30.3711,
        30.016599999999997,
        29.82,
        29.9509,
        30.024099999999997
      ],
      "median_ms": 30.016599999999997,
      "checksum": 37.79337016431251
    }
  ]
}
```
