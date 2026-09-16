"""Generate independent CPU goldens. Never import or run the Dart implementation."""
import argparse
import hashlib
import json
import math
import gzip
import io
import importlib.metadata
import sys
import os
from urllib.request import urlopen
from pathlib import Path

# Select the same kernel family on AVX2/AVX512 GitHub-hosted runners.
os.environ["ATEN_CPU_CAPABILITY"] = "default"

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TV
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def encode_number(x):
    if isinstance(x, int):
        return str(x) if abs(x) > 2**53 else x
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def tensor(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
            "values": [encode_number(x) for x in t.flatten().tolist()]}


def preset(x, name, size=4, shortest_edge=None):
    """Explicit tensor recipes; not a promise about arbitrary pretrained weights."""
    y = x.float()
    if x.dtype == torch.uint8 and name != "tflite_raw":
        y = y / 255
    y = y.permute(2, 0, 1) if y.ndim == 3 else y.permute(0, 3, 1, 2)
    if name in ("imagenet", "clip"):
        mode = torchvision.transforms.InterpolationMode.BICUBIC if name == "clip" else torchvision.transforms.InterpolationMode.BILINEAR
        y = TV.resize(y, size if name == "clip" else (shortest_edge or size + 2), interpolation=mode, antialias=True)
        y = TV.center_crop(y, [size, size])
    else:
        y = TV.resize(y, [size, size], antialias=True)
    if name in ("imagenet", "resnet", "segmentation"):
        y = TV.normalize(y, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif name in ("face", "mobilenet", "vit", "custom", "custom_hwc", "custom_unbatched"):
        y = TV.normalize(y, [0.5] * 3, [0.5] * 3)
    elif name == "clip":
        y = TV.normalize(y, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    if name in ("tflite", "tflite_raw", "custom_hwc"):
        y = y.permute(1, 2, 0) if y.ndim == 3 else y.permute(0, 2, 3, 1)
    if y.ndim == 3 and name != "custom_unbatched":
        y = y.unsqueeze(0)
    return y


def generate():
    cases = []

    def add(name, op, x, y, params=None, **extra):
        cases.append({"name": name, "op": op, "params": params or {},
                      "input": tensor(x), "expected": tensor(y),
                      "atol": 1e-12 if x.dtype == torch.float64 else 1e-6,
                      "rtol": 1e-10 if x.dtype == torch.float64 else 1e-5, **extra})

    operations = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": F.relu,
                  "silu": F.silu, "mish": F.mish, "hardsigmoid": F.hardsigmoid,
                  "hardswish": F.hardswish, "elu": F.elu, "selu": F.selu,
                  "leaky_relu": F.leaky_relu, "abs": torch.abs}
    for dtype in (torch.float32, torch.float64):
        dtype_name = str(dtype).removeprefix("torch.")
        x = torch.tensor([-1000, -30, -3, -1, -1e-8, 0, 1e-8, 1, 3, 30, 1000], dtype=dtype)
        for op, fn in operations.items():
            add(f"{op}-{dtype_name}", op, x, fn(x))
            # Offset view and SIMD tail: both prefix and suffix must survive in-place.
            base = torch.tensor([-777, -555, -3, -1, 0, 1, 3, -333, -111], dtype=dtype)
            view = base[2:7]
            original = view.clone()
            before = tensor(base)
            expected = fn(view)
            base[2:7] = expected
            add(f"{op}-{dtype_name}-offset", op, original, expected,
                base=before, offset=2, inplace=True, expected_base=tensor(base))
        special = torch.tensor([float("-inf"), -0.0, 0.0, float("inf"), float("nan")], dtype=dtype)
        for op in ("tanh", "sigmoid"):
            add(f"{op}-{dtype_name}-special", op, special, operations[op](special))
        for axis in (0, -1):
            x = torch.tensor([[1000, 999, 998], [-1000, -999, -998]], dtype=dtype)
            add(f"softmax-{dtype_name}-{axis}", "softmax", x, F.softmax(x, dim=axis), {"axis": axis})

    for dtype in (torch.float32, torch.float64):
        x = torch.arange(45, dtype=dtype).reshape(3, 3, 5) / 7
        add(f"shortest-{dtype}", "resize_shortest", x,
            TV.resize(x, 4, antialias=False), {"size": 4})
        for mode in ("nearest", "bilinear", "bicubic", "area"):
            for h, w in ((2, 3), (5, 8), (1, 1)):
                params = {"height": h, "width": w, "mode": mode}
                kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
                y = F.interpolate(x.unsqueeze(0), size=(h, w), mode=mode, **kwargs).squeeze(0)
                add(f"resize-{dtype}-{mode}-{h}x{w}", "resize", x, y, params)
                if mode in ("bilinear", "bicubic"):
                    for align in (False, True):
                        y = F.interpolate(x.unsqueeze(0), size=(h, w), mode=mode,
                                          align_corners=align, antialias=True).squeeze(0)
                        add(f"resize-aa-{dtype}-{mode}-{h}x{w}-align{align}", "resize", x, y,
                            {**params, "antialias": True, "align_corners": align})
    for batch in (False, True):
        for dtype in (torch.uint8, torch.float32):
            x = ((torch.arange((2 if batch else 1) * 7 * 11 * 3) * 37 + 13) % 256).to(torch.uint8)
            x = x.reshape(2, 7, 11, 3) if batch else x.reshape(7, 11, 3)
            if dtype == torch.float32:
                x = x.float() / 255
            for name in ("imagenet", "resnet", "detection", "segmentation", "face", "mobilenet",
                         "clip", "vit", "tflite", "tflite_raw", "minimal", "custom", "custom_hwc", "custom_unbatched"):
                add(f"preset-{name}-{dtype}-batch{batch}", "preset", x, preset(x, name), {"name": name, "size": 4})
    for h, w in ((4, 4), (2, 2), (9, 13)):
        x = torch.arange(7 * 11, dtype=torch.float32).reshape(1, 7, 11)
        add(f"center-crop-{h}x{w}", "center_crop", x, TV.center_crop(x, [h, w]), {"height": h, "width": w})
    return cases


def generate_network(output):
    sources = json.loads((ROOT / "test/fixtures/pytorch/network-manifest.json").read_text(encoding="utf-8"))
    cache = ROOT / ".dart_tool/pytorch-fixtures"
    cache.mkdir(parents=True, exist_ok=True)
    goldens = []
    for source in sources:
        path = cache / (source["name"] + ".png")
        if not path.exists():
            with urlopen(source["url"], timeout=60) as response:
                path.write_bytes(response.read())
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"], source["name"]
        pixels = np.array(Image.open(io.BytesIO(raw)))
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == source["decoded_sha256"]
        for recipe in source["recipes"]:
            expected = preset(torch.from_numpy(pixels.copy()), recipe["name"], recipe["size"], recipe.get("shortest_edge"))
            data = expected.contiguous().numpy().astype("<f4").tobytes()
            filename = f'{source["name"]}-{recipe["name"]}-{recipe["size"]}.f32.gz'
            compressed = io.BytesIO()
            with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as stream:
                stream.write(data)
            (output / filename).write_bytes(compressed.getvalue())
            goldens.append({"source": source["name"], "recipe": recipe, "shape": list(expected.shape),
                            "dtype": "float32", "file": filename, "sha256": hashlib.sha256(data).hexdigest(),
                            "atol": 1e-6, "rtol": 1e-5})
    (output / "network-goldens.json").write_text(json.dumps(goldens, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "test/fixtures/pytorch")
    parser.add_argument("--network", action="store_true", help="Also regenerate pinned remote-image goldens")
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 12), sys.version
    for line in (ROOT / "scripts/requirements-fixtures.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            assert importlib.metadata.version(name) == version, (name, importlib.metadata.version(name), version)
    assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
    assert torchvision.__version__.split("+")[0] == "0.25.0", torchvision.__version__
    torch.set_num_threads(1)
    assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
    torch.use_deterministic_algorithms(True)
    cases = generate()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(cases, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    (args.output / "operations.golden.json").write_bytes(raw)
    manifest = {"schema_version": 1,
                "oracle": {"torch": "2.10.0+cpu", "torchvision": "0.25.0+cpu", "python": "3.12", "device": "cpu", "capability": "DEFAULT", "threads": 1},
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "requirements_sha256": hashlib.sha256((ROOT / "scripts/requirements-fixtures.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "files": [{"path": "operations.golden.json", "sha256": hashlib.sha256(raw).hexdigest(),
                           "cases": len(cases)}]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.network:
        generate_network(args.output)
    print(f"Generated {len(cases)} cases in {args.output}")


if __name__ == "__main__":
    main()
