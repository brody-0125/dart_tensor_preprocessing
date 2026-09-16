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
# ATen's unary VML calls have their own MKL dispatch independent of ATen.
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"

import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TV
from torchvision.transforms import _functional_tensor as TVT
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
                      "atol": 1e-12 if y.dtype == torch.float64 else 1e-6,
                      "rtol": 1e-10 if y.dtype == torch.float64 else 1e-5, **extra})

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

    # Normalization: independent torch calls, affine parameters, CHW/NCHW,
    # constant inputs, offset mutation sentinels and non-contiguous views.
    def view_cases(name, x, fn, params, inplace=True):
        y = fn(x)
        add(name, params["op"], x, y, params)
        unsigned = x.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)
        base = torch.cat((x.new_tensor([77, 55] if unsigned else [-777, -555]), x.flatten(), x.new_tensor([33] if unsigned else [-333])))
        before = tensor(base)
        if inplace:
            base[2:-1] = y.flatten()
        add(name + "-offset", params["op"], x, y, params, base=before,
            offset=2, inplace=inplace, expected_base=tensor(base))
        if x.ndim == 1:
            backing = torch.stack((x, torch.zeros_like(x)), dim=-1)
            view = backing[:, 0]
        else:
            backing = x.transpose(-1, -2).contiguous()
            view = backing.transpose(-1, -2)
        add(name + "-strided", params["op"], view, fn(view), params,
            base=tensor(backing), offset=0, strides=list(view.stride()))

    for dtype in (torch.float32, torch.float64):
        for batched in (False, True):
            shape = (2, 4, 2, 3) if batched else (4, 2, 3)
            for variant in ("plain", "affine", "constant"):
                x = (torch.arange(math.prod(shape), dtype=dtype).reshape(shape) % 17 - 8) / 4
                if variant == "constant":
                    x = torch.full_like(x, 0.25)
                eps = 0.125 if variant == "affine" else 1e-5
                affine = variant != "plain"
                weight = torch.tensor([0.5, -1, 2, 0], dtype=dtype) if affine else None
                bias = torch.tensor([-0.25, 0.5, 1, -2], dtype=dtype) if affine else None
                mean = torch.tensor([0.5, -1, 2, 0], dtype=dtype)
                var = torch.tensor([0.5, 0, 2, 4], dtype=dtype)
                kw = {"eps": eps, "channels": 4,
                      "weight": weight.tolist() if affine else None,
                      "bias": bias.tolist() if affine else None}
                def image_norm(z, fn):
                    return fn(z) if batched else fn(z.unsqueeze(0)).squeeze(0)
                recipes = [
                    ("batch_norm", lambda z: image_norm(z, lambda v: F.batch_norm(v, mean, var, weight, bias, training=False, eps=eps)),
                     {**kw, "mean": mean.tolist(), "variance": var.tolist()}),
                    ("instance_norm", lambda z: image_norm(z, lambda v: F.instance_norm(v, weight=weight, bias=bias, eps=eps)), kw),
                    ("group_norm", lambda z: image_norm(z, lambda v: F.group_norm(v, 2, weight, bias, eps)), {**kw, "groups": 2}),
                    ("normalize", lambda z: TV.normalize(z, mean.tolist(), [0.5, 1, 2, 4]),
                     {"mean": mean.tolist(), "std": [0.5, 1, 2, 4]})]
                layer_weight = torch.tensor([[0.5, 1, -1], [2, 0, 0.25]], dtype=dtype) if affine else None
                layer_bias = torch.tensor([[1, -1, 0], [0.5, 2, -2]], dtype=dtype) if affine else None
                layer_kw = {"shape": [2, 3], "eps": eps,
                            "weight": layer_weight.flatten().tolist() if affine else None,
                            "bias": layer_bias.flatten().tolist() if affine else None}
                recipes.extend([
                    ("layer_norm", lambda z: F.layer_norm(z, [2, 3], layer_weight, layer_bias, eps), layer_kw),
                    ("rms_norm", lambda z: F.rms_norm(z, [2, 3], layer_weight, eps), layer_kw)])
                for op, fn, params in recipes:
                    view_cases(f"{op}-{dtype}-batch{batched}-{variant}", x, fn, {**params, "op": op})
        for order in (1.0, 2.0, 3.0, float("inf")):
            for axis in (0, -1):
                x = torch.tensor([[0, 0, 0], [0.01, -0.02, 0.03], [3, -4, 5]], dtype=dtype)
                params = {"op": "lp_normalize", "p": encode_number(order), "dim": axis, "eps": 0.125}
                view_cases(f"lp-{dtype}-p{order}-dim{axis}", x,
                           lambda z: F.normalize(z, p=order, dim=axis, eps=0.125), params)

    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[float("nan"), 1], [float("inf"), 1], [0, 0]], dtype=dtype)
        for order in (1.0, 2.0, float("inf")):
            add(f"lp-special-{dtype}-{order}", "lp_normalize", x,
                F.normalize(x, p=order, dim=-1, eps=0.125),
                {"p": encode_number(order), "dim": -1, "eps": 0.125})

    unary = {"neg": torch.neg, "sqrt": torch.sqrt, "exp": torch.exp,
             "log": torch.log, "floor": torch.floor, "ceil": torch.ceil,
             "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
             "asin": torch.asin, "acos": torch.acos, "atan": torch.atan}
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([-1000, -3, -2.5, -1, -0.5, -1e-8, -0.0, 0.0, 1e-8, 0.5, 1, 2.5, 3, 1000], dtype=dtype)
        for op, fn in unary.items():
            view_cases(f"{op}-{dtype}-views", x, fn, {"op": op})
            special = torch.tensor([float("-inf"), float("inf"), float("nan")], dtype=dtype)
            add(f"{op}-{dtype}-special", op, special, fn(special))
        for approximate in ("none", "tanh"):
            view_cases(f"gelu-{dtype}-{approximate}", x,
                       lambda z: F.gelu(z, approximate=approximate),
                       {"op": "gelu", "approximate": approximate})
        dense = torch.arange(-400, 401, dtype=dtype) / 40
        add(f"gelu-dense-{dtype}", "gelu", dense, F.gelu(dense), {"approximate": "none"})
        for approximate in ("none", "tanh"):
            special = torch.tensor([float("-inf"), float("inf"), float("nan")], dtype=dtype)
            add(f"gelu-special-{dtype}-{approximate}", "gelu", special,
                F.gelu(special, approximate=approximate), {"approximate": approximate})
        for axis in (0, -1):
            z = torch.tensor([[-1000, -1, 1, 1000], [3, 1, -2, -3]], dtype=dtype)
            add(f"glu-{dtype}-dim{axis}", "glu", z, F.glu(z, dim=axis), {"dim": axis})
        # RoundOp deliberately retains its documented half-away-from-zero rule.
        view_cases(f"round-half-away-{dtype}", x,
                   lambda z: torch.copysign(torch.floor(torch.abs(z) + 0.5), z),
                   {"op": "round"})

    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
        values = [4, 1, -7, 2, 9, -3] if dtype != torch.int64 else [
            9007199254740993, 9007199254740995, -9007199254740993,
            9007199254740997, 9007199254740999, -9007199254740995]
        x = torch.tensor(values, dtype=dtype).reshape(2, 3)
        idx = torch.tensor([[2, 0], [0, 1]], dtype=torch.int64)
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.uint8)
        y = x.flip(-1)
        recipes = [
            ("core_clone", lambda z: z.clone(), {}),
            ("core_contiguous", lambda z: z.contiguous(), {}),
            ("core_transpose", lambda z: z.permute(1, 0), {"axes": [1, 0]}),
            ("core_reshape", lambda z: z.contiguous().reshape(3, 2), {"shape": [3, 2]}),
            ("tile", lambda z: z.repeat(2, 3), {"reps": [2, 3]}),
            ("repeat", lambda z: z.repeat(2, 3), {"reps": [2, 3]}),
            ("roll", lambda z: torch.roll(z, (1, -2), (0, -1)), {"shifts": [1, -2], "dims": [0, -1]}),
            ("roll", lambda z: torch.roll(z, 2), {"shifts": [2], "dims": None}),
            ("roll", lambda z: torch.roll(z, (1, 1), (1, 1)), {"shifts": [1, 1], "dims": [1, 1]}),
            ("slice", lambda z: z[:, 0:3:2], {"slices": [None, [0, 3, 2]]}),
            ("gather", lambda z: torch.gather(z, -1, idx), {"dim": -1, "index": tensor(idx)}),
            ("where", lambda z: torch.where(mask.bool(), z, y), {"mask": tensor(mask), "other": tensor(y)}),
        ]
        for axis in (0, -1):
            recipes.append(("core_stack", lambda z, d=axis: torch.stack((z, y), d), {"axis": axis, "other": tensor(y)}))
            recipes.append(("core_concat", lambda z, d=axis: torch.cat((z, y), d), {"axis": axis, "other": tensor(y)}))
        for part in (0, 1):
            recipes.append(("core_split", lambda z, i=part: torch.split(z, [1, 2], dim=-1)[i], {"sizes": [1, 2], "dim": -1, "part": part, "count": 2}))
            recipes.append(("core_chunk", lambda z, i=part: torch.chunk(z, 2, dim=-1)[i], {"chunks": 2, "dim": -1, "part": part, "count": 2}))
        for largest in (False, True):
            for indices in (False, True):
                recipes.append(("core_topk", lambda z, l=largest, i=indices: torch.topk(z, 2, dim=-1, largest=l)[int(i)],
                                {"k": 2, "axis": -1, "largest": largest, "indices": indices}))
        for reduction in ("sum", "min", "max", "argmin", "argmax", "mean"):
            if reduction == "mean" and not x.is_floating_point():
                continue  # PyTorch rejects integer mean; Dart must reject axis mean too.
            for axis in (0, -1):
                for keep in (False, True):
                    def reduce(z, r=reduction, d=axis, k=keep):
                        if r == "min": return torch.amin(z, d, k)
                        if r == "max": return torch.amax(z, d, k)
                        return getattr(torch, r)(z, dim=d, keepdim=k)
                    recipes.append(("core_reduce", reduce,
                                    {"reduction": reduction, "axis": axis, "keep": keep}))
            # The scalar-valued Dart API explicitly returns double for value reductions.
            fn = getattr(torch, reduction)
            expected = fn(x if reduction.startswith("arg") else x.double()).reshape(1)
            add(f"reduce-global-{dtype}-{reduction}", "core_reduce", x, expected,
                {"reduction": reduction, "axis": None})
        for reduction in ("sum", "min", "max", "mean"):
            if reduction == "mean" and not x.is_floating_point(): continue
            fn = torch.amin if reduction == "min" else torch.amax if reduction == "max" else getattr(torch, reduction)
            for keep in (False, True):
                y_multi = fn(x, dim=(0, 1), keepdim=keep)
                if y_multi.ndim == 0: y_multi = y_multi.reshape(1)  # Scalar tensor outputs use [1] in this package.
                add(f"reduce-multi-{dtype}-{reduction}-{keep}", "core_reduce", x, y_multi,
                    {"reduction": reduction, "axes": [0, -1], "keep": keep})
        view_cases(f"masked-fill-{dtype}", x, lambda z: z.masked_fill(mask.bool(), -2),
                   {"op": "masked_fill", "mask": tensor(mask), "value": -2})
        for number, (op, fn, params) in enumerate(recipes):
            view_cases(f"index-{dtype}-{number}-{op}", x, fn, {**params, "op": op}, inplace=False)

    for dtype in (torch.float32, torch.float64):
        for values in ([[2, 2, 1], [3, 1, 1]], [[1, float("nan"), 2], [float("inf"), -1, float("nan")]]):
            x = torch.tensor(values, dtype=dtype)
            label = "nan" if torch.isnan(x).any() else "ties"
            for reduction in ("min", "max", "argmin", "argmax"):
                fn = torch.amin if reduction == "min" else torch.amax if reduction == "max" else getattr(torch, reduction)
                view_cases(f"reduce-{label}-{dtype}-{reduction}", x, lambda z: fn(z, dim=-1),
                           {"op": "core_reduce", "reduction": reduction, "axis": -1, "keep": False}, inplace=False)
                result = getattr(torch, reduction)(x.double()).reshape(1)
                add(f"reduce-global-{label}-{dtype}-{reduction}", "core_reduce", x, result,
                    {"reduction": reduction, "axis": None})

    for dtype in (torch.float32, torch.float64):
        for count in (3, 20):
            x = torch.arange(count * 2, dtype=dtype).reshape(2, count)
            x[:, count // 2] = float("nan")
            for largest in (False, True):
                for indices in (False, True):
                    k = min(count, 5)
                    view_cases(f"topk-special-{dtype}-{count}-{largest}-{indices}", x,
                               lambda z: torch.topk(z, k, dim=-1, largest=largest)[int(indices)],
                               {"op": "core_topk", "k": k, "axis": -1, "largest": largest, "indices": indices}, inplace=False)
        tied = torch.tensor([[2, 2, 1], [3, 1, 1]], dtype=dtype)
        for largest in (False, True):
            view_cases(f"topk-ties-{dtype}-{largest}", tied,
                       lambda z: torch.topk(z, 2, dim=-1, largest=largest).values,
                       {"op": "core_topk", "k": 2, "axis": -1, "largest": largest, "indices": False}, inplace=False)

    adjacent = torch.tensor([[2**53, 2**53 + 1, 2**53], [-2**53, -2**53 - 1, -2**53]], dtype=torch.int64)
    for reduction in ("argmin", "argmax"):
        for axis in (None, -1):
            fn = getattr(torch, reduction)
            view_cases(f"reduce-adjacent-int64-{reduction}-{axis}", adjacent,
                       lambda z: fn(z).reshape(1) if axis is None else fn(z, dim=axis),
                       {"op": "core_reduce", "reduction": reduction, "axis": axis, "keep": False}, inplace=False)
    for dtype, largest in ((torch.int32, 2**31 - 1), (torch.int64, 2**63 - 1)):
        x = torch.tensor([[largest, 1]], dtype=dtype)
        add(f"reduce-integer-overflow-{dtype}", "core_reduce", x, torch.sum(x, dim=-1),
            {"reduction": "sum", "axis": -1, "keep": False})

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
    # Sequence factories compute in double, then truncate integer destinations.
    # This intentionally differs from torch's integer-endpoint linspace kernel.
    for dtype in (torch.float32, torch.float64, torch.int8, torch.int16,
                  torch.int32, torch.int64, torch.uint8, torch.uint16,
                  torch.uint32, torch.uint64):
        dummy = torch.zeros(1, dtype=dtype)
        for factory, value in (("zeros", 0), ("uninitialized", 0), ("ones", 1), ("full", 2.75)):
            add(f"factory-{factory}-{dtype}", "core_factory", dummy,
                torch.full((2, 3), value, dtype=torch.float64).to(dtype),
                {"factory": factory, "shape": [2, 3], "value": value})
        add(f"factory-eye-{dtype}", "core_factory", dummy, torch.eye(3, 5, dtype=torch.int64).to(dtype),
            {"factory": "eye", "n": 3, "m": 5})
        for steps in (1, 2, 7):
            add(f"factory-linspace-{dtype}-{steps}", "core_factory", dummy,
                torch.linspace(0.5, 9.5, steps, dtype=torch.float64).to(dtype),
                {"factory": "linspace", "start": 0.5, "end": 9.5, "steps": steps})
        for start, end, step in ((0.5, 8.5, 0.75), (9.5, 0.5, -1.25)):
            add(f"factory-arange-{dtype}-{step}", "core_factory", dummy,
                torch.arange(start, end, step, dtype=torch.float64).to(dtype),
                {"factory": "arange", "start": start, "end": end, "step": step})

    # Existing public cast contract is half-away rounding, with selected clamps.
    clamps = {torch.int8: (-128, 127), torch.int16: (-32768, 32767),
              torch.uint8: (0, 255), torch.uint16: (0, 65535), torch.uint32: (0, 4294967295)}
    for source in (torch.float32, torch.float64, torch.int64):
        x = torch.tensor([-9007199254740993, -257, -1, 0, 257, 9007199254740993], dtype=source) if source == torch.int64 else torch.tensor([-32769.5, -128.5, -1.5, 0.5, 255.5, 65536.5], dtype=source)
        for dest in (torch.float32, torch.float64, torch.int8, torch.int16, torch.int32,
                     torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
            def cast(z, dest=dest):
                if dest in (torch.float32, torch.float64) or dest == z.dtype:
                    return z.to(dest)
                y = torch.copysign(torch.floor(z.abs() + 0.5), z) if z.is_floating_point() else z
                if dest in clamps:
                    y = y.clamp(*clamps[dest])
                return y.to(dest)
            view_cases(f"cast-{source}-{dest}", x.abs() if dest == torch.uint64 else x, cast,
                       {"op": "core_cast", "dtype": str(dest).removeprefix("torch.")}, inplace=False)
    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
        x = torch.arange(6, dtype=dtype).reshape(1, 2, 1, 3)
        if dtype == torch.int64:
            x += 9007199254740993
        for op, fn, params in (
            ("select", lambda z: z.select(1, 1), {"axis": 1, "index": 1}),
            ("unbind", lambda z: z.unbind(1)[1], {"axis": 1, "index": 1, "parts": 2}),
            ("narrow", lambda z: z.narrow(3, 1, 2), {"axis": 3, "start": 1, "length": 2}),
        ):
            view_cases(f"view-{op}-{dtype}", x, fn, {"op": "core_" + op, **params}, inplace=False)
        view_cases(f"view-select-vector-{dtype}", x.flatten(), lambda z: z.select(0, 1).reshape(1),
                   {"op": "core_select", "axis": 0, "index": 1}, inplace=False)
        for axis in (None, 0, -2, 1):
            view_cases(f"squeeze-{dtype}-{axis}", x,
                       lambda z, axis=axis: z.squeeze() if axis is None else z.squeeze(axis),
                       {"op": "core_squeeze", "axis": axis}, inplace=False)
        for axis in (0, -1, -5):
            view_cases(f"unsqueeze-{dtype}-{axis}", x, lambda z, axis=axis: z.unsqueeze(axis),
                       {"op": "core_unsqueeze", "axis": axis}, inplace=False)
        view_cases(f"squeeze-single-{dtype}", x.flatten()[:1].reshape(1, 1),
                   lambda z: z.squeeze().reshape(1), {"op": "core_squeeze"}, inplace=False)
    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
        x = torch.arange(24, dtype=dtype).reshape(1, 2, 3, 4)
        if dtype == torch.int64:
            x += 9007199254740993
        for target, axes in (("nhwc", (0, 2, 3, 1)), ("nchw", (0, 3, 1, 2))):
            for contiguous in (False, True):
                view_cases(f"layout-{dtype}-{target}-{contiguous}", x,
                           lambda z, axes=axes: z.permute(axes),
                           {"op": "core_layout", "target": target, "contiguous": contiguous}, inplace=False)
    for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
        x = torch.arange(24, dtype=dtype).reshape(2, 3, 4)
        if dtype == torch.int64:
            x += 9007199254740993
        for op, fn, params in (
            ("identity", lambda z: z, {}),
            ("contiguous_op", lambda z: z.contiguous(), {}),
            ("permute_op", lambda z: z.permute(2, 0, 1), {"axes": [2, 0, 1]}),
            ("reshape_op", lambda z: z.reshape(4, 6), {"shape": [4, -1]}),
            ("flatten_op", lambda z: z.flatten(1, 2), {"start": 1, "end": -1}),
        ):
            view_cases(f"shape-{op}-{dtype}", x, fn, {"op": "core_" + op, **params}, inplace=False)
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([0.125, 0.5, 1, 2, 3, 7, 11], dtype=dtype)
        other = torch.tensor([2, 0.25, -1, 3, 0.5, -2, 4], dtype=dtype)
        for name, fn in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div)):
            for scalar in (True, False):
                operand = 1.25 if scalar else other
                view_cases(f"binary-{name}-{dtype}-scalar{scalar}", x,
                           lambda z, fn=fn, operand=operand: fn(z, operand),
                           {"op": "binary_" + name, **({"scalar": operand} if scalar else {"other": tensor(other)})})
        view_cases(f"binary-pow-{dtype}", x, lambda z: torch.pow(z, 1.5), {"op": "binary_pow", "scalar": 1.5})
    for dtype in (torch.int32, torch.int64):
        x = torch.tensor([-11, -3, -1, 0, 1, 3, 11], dtype=dtype)
        if dtype == torch.int64:
            x += 9007199254740993
        other = torch.tensor([1, 2, -1, 3, -2, 1, 2], dtype=dtype)
        for name, fn in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul)):
            for scalar in (True, False):
                operand = 2 if scalar else other
                view_cases(f"binary-integer-{name}-{dtype}-{scalar}", x,
                           lambda z, fn=fn, operand=operand: fn(z, operand),
                           {"op": "binary_" + name, **({"scalar": operand} if scalar else {"other": tensor(other)})})
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([-float("inf"), -0.0, 0.0, float("inf"), float("nan"), 1, -1], dtype=dtype)
        other = torch.tensor([0, 0, 1, -1, 2, 0, 3], dtype=dtype)
        for name, fn in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div)):
            view_cases(f"binary-special-{name}-{dtype}", x, lambda z, fn=fn: fn(z, other),
                       {"op": "binary_" + name, "other": tensor(other)})
            mixed = torch.tensor([0.25, 1.5, 2, -1.25, 3, 7, -2], dtype=torch.float64 if dtype == torch.float32 else torch.float32)
            finite = torch.tensor([0.125, 0.5, 1, 2, 3, 7, 11], dtype=dtype)
            view_cases(f"binary-mixed-{name}-{dtype}", finite,
                       lambda z, fn=fn: fn(z.double(), mixed.double()).to(dtype),
                       {"op": "binary_" + name, "other": tensor(mixed)})
        for exponent in (0.5, -0.5):
            view_cases(f"binary-special-pow-{dtype}-{exponent}", x,
                       lambda z, exponent=exponent: torch.pow(z, exponent), {"op": "binary_pow", "scalar": exponent})
    for dtype in (torch.int32, torch.int64):
        x = torch.tensor([-11, -3, -1, 0, 1, 3, 11], dtype=dtype)
        if dtype == torch.int64:
            x += 9007199254740993
        view_cases(f"integer-div-{dtype}", x, lambda z: torch.div(z, 2, rounding_mode="trunc"), {"op": "binary_div", "scalar": 2})
        for exponent in (0, 1, 2, 3):
            view_cases(f"integer-pow-{dtype}-{exponent}", x, lambda z, exponent=exponent: torch.pow(z, exponent), {"op": "binary_pow", "scalar": exponent})
    for dtype in (torch.int8, torch.int16, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        x = torch.tensor([0, 1, 3, 11, 127, 255], dtype=torch.int64).to(dtype)
        for name, fn in (("add", lambda z: z + 2), ("sub", lambda z: z - 2), ("mul", lambda z: z * 2), ("div", lambda z: torch.div(z, 2, rounding_mode="trunc")), ("pow", lambda z: z ** 2)):
            y = fn(x.to(torch.int64))
            if dtype in (torch.uint8, torch.uint16):
                y = y.clamp(0, 255 if dtype == torch.uint8 else 65535)
            if dtype == torch.uint64 and name == "sub":
                continue  # Native Dart exposes high-bit uint64 values as signed.
            add(f"integer-dtype-{name}-{dtype}", "binary_" + name, x, y.to(dtype), {"scalar": 2})
    for dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        x = torch.tensor([2, 3, 5, 7, 11, 20], dtype=dtype)
        for name, fn in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div), ("pow", torch.pow)):
            y = fn(x.double(), 1.5).trunc()
            if dtype in (torch.uint8, torch.uint16):
                y = y.clamp(0, 255 if dtype == torch.uint8 else 65535)
            add(f"integer-fractional-{name}-{dtype}", "binary_" + name, x, y.to(dtype), {"scalar": 1.5})
            if name != "pow":
                other = torch.full((6,), 1.5, dtype=torch.float64)
                add(f"integer-mixed-{name}-{dtype}", "binary_" + name, x, y.to(dtype), {"other": tensor(other)})
    for dtype in (torch.float32, torch.float64):
        for special in (False, True):
            x = torch.tensor([-7, -2, -0.0, 0.0, 0.25, 3, 11], dtype=dtype)
            if special:
                x = torch.tensor([float('nan'), -float('inf'), -0.0, 0, float('inf'), float('nan'), 1], dtype=dtype)
            for name, fn, params in (
                ('clip', lambda z: z.clamp(-1, 2), {'min': -1, 'max': 2}),
                ('scale', lambda z: (z - 0.5) / 2.5, {'scale': 2.5, 'offset': 0.5}),
                ('atan2_scalar', lambda z: torch.atan2(z, z.new_tensor(-0.5)), {'scalar': -0.5}),
            ):
                view_cases(f'numeric-{name}-{dtype}-{special}', x, fn, {'op': name, **params})
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([100000000, 100000008, 99999992, 100000016, 99999984], dtype=dtype)
        view_cases(f'scale-cancellation-{dtype}', x, lambda z: (z - 100000000) / 3,
                   {'op': 'scale', 'scale': 3, 'offset': 100000000})
        other = torch.tensor([-1, 0, 1, -float('inf'), float('inf'), 0, float('nan')], dtype=dtype)
        x = torch.tensor([-2, -0.0, 0, -float('inf'), float('inf'), 3, 1], dtype=dtype)
        view_cases(f'atan2-tensor-{dtype}', x, lambda z: torch.atan2(z, other), {'op': 'atan2_tensor', 'other': tensor(other)})
    x = torch.tensor([-1000000000000000000, -9007199254740993, 0, 9007199254740993, 1000000000000000000], dtype=torch.int64)
    view_cases('clip-exact-int64', x, lambda z: z.clamp(-100000000000000000, 100000000000000000),
               {'op': 'clip', 'min': -100000000000000000, 'max': 100000000000000000})
    for dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        x = torch.tensor([0, 1, 3, 7, 20], dtype=dtype)
        add(f'clip-integer-{dtype}', 'clip', x, x.to(torch.float64).clamp(1.5, 9.5).trunc().to(dtype), {'min': 1.5, 'max': 9.5})
        add(f'scale-integer-{dtype}', 'scale', x, ((x.double() - 0.5) / 2.5).trunc().to(dtype), {'scale': 2.5, 'offset': 0.5})
        add(f'atan2-integer-{dtype}', 'atan2_scalar', x, torch.atan2(x.double(), torch.tensor(0.5, dtype=torch.float64)).trunc().to(dtype), {'scalar': 0.5})
    for dtype in (torch.uint8, torch.int64, torch.float32, torch.float64):
        for channels in (1, 3, 4):
            for batched in (False, True):
                shape = (2, 3, 5, channels) if batched else (3, 5, channels)
                x = ((torch.arange(math.prod(shape)) * 37) % 256).reshape(shape).to(dtype)
                if dtype in (torch.float32, torch.float64):
                    x = x / 255
                axes = (0, 3, 1, 2) if batched else (2, 0, 1)
                for normalize in (False, True):
                    view_cases(f'to-tensor-{dtype}-{channels}-{batched}-{normalize}', x,
                               lambda z, axes=axes, normalize=normalize: (z.double().permute(axes) / (255 if normalize else 1)).float(),
                               {'op': 'to_tensor', 'normalize': normalize}, inplace=False)
    for dtype in (torch.float32, torch.float64):
        for channels in (1, 3, 4):
            for batched in (False, True):
                shape = (2, channels, 3, 5) if batched else (channels, 3, 5)
                x = (torch.arange(math.prod(shape), dtype=dtype).reshape(shape) % 17 - 2) / 11
                axes = (0, 2, 3, 1) if batched else (1, 2, 0)
                for denormalize in (False, True):
                    def to_image(z, axes=axes, denormalize=denormalize):
                        y = z.double().permute(axes) * (255 if denormalize else 1)
                        return torch.copysign(torch.floor(y.abs() + 0.5), y).clamp(0, 255).to(torch.uint8)
                    view_cases(f'to-image-{dtype}-{channels}-{batched}-{denormalize}', x, to_image,
                               {'op': 'to_image', 'denormalize': denormalize}, inplace=False)
    for dtype in (torch.float32, torch.float64, torch.uint8, torch.int64):
        for batched in (False, True):
            for variant in ('primary', 'gradient'):
                x = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1]], dtype=dtype).reshape(3, 2, 4)
                if variant == 'gradient':
                    x = (torch.arange(24).reshape(3, 2, 4) % 11).to(dtype)
                    if dtype.is_floating_point:
                        x = x / 10
                    else:
                        x = x % 2
                if batched:
                    x = torch.stack((x, x.flip(-1)))
                work = lambda z: z if z.is_floating_point() else z.float()
                for op, fn in (('grayscale', TV.rgb_to_grayscale), ('rgb_hsv', TVT._rgb2hsv), ('hsv_rgb', TVT._hsv2rgb)):
                    view_cases(f'color-{op}-{dtype}-{batched}-{variant}', x,
                               lambda z, fn=fn: fn(work(z)), {'op': op}, inplace=False)
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
                "oracle": {"torch": "2.10.0+cpu", "torchvision": "0.25.0+cpu", "python": "3.12", "device": "cpu", "cpu_model": os.environ.get("PYTORCH_ORACLE_CPU_MODEL", "native-investigation"), "capability": "DEFAULT", "mkl_cbwr": "COMPATIBLE", "mkl_instructions": "SSE4_2", "threads": 1},
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "requirements_sha256": hashlib.sha256((ROOT / "scripts/requirements-fixtures.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "requirements_linux_sha256": hashlib.sha256((ROOT / "scripts/requirements-fixtures-linux.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "files": [{"path": "operations.golden.json", "sha256": hashlib.sha256(raw).hexdigest(),
                           "cases": len(cases)}]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.network:
        generate_network(args.output)
    print(f"Generated {len(cases)} cases in {args.output}")


if __name__ == "__main__":
    main()
