"""Standalone CPU inference for FGRNet (PBFG, Zhu & Lee, ICCV 2025).

Loads the official pretrained checkpoint (net_g_last.pth, 6-channel Uformer
variant with SFEM modules) without installing the repo's basicsr package:
the arch file is imported directly with a stubbed ARCH_REGISTRY.

Usage:
  python3 fgrnet_infer.py <input_dir_or_image> <output_dir> \
      [--repo PATH] [--ckpt PATH] [--gt DIR]

Outputs per image: <name>_deflare.png (flare-free), <name>_flare.png
(predicted flare/glare layer = the "glow map" analogue). If --gt is given,
also writes <name>_blend.png with the paper's histogram-matching step.
"""
import argparse
import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image


def load_arch(repo):
    """Import FGRNet_arch.py standalone, stubbing basicsr.utils.registry.

    The released checkpoint carries two extra module groups per block, `cab`
    and `fusion`, plus the SFEM stored under the name `sfb`. Statistical
    inspection shows cab and fusion are DEAD parameters: fusion's weights sit
    exactly at their kaiming-uniform init bounds and cab's BatchNorm has
    running_mean=0, running_var=1, num_batches_tracked=0 — they never
    received a forward pass during training. The trained computation is
    therefore exactly the released forward; loading only requires renaming
    sfb->sfem and dropping the dead cab keys (see remap_state_dict).
    """
    pkg = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")

    class _Reg:
        def register(self, *a, **k):
            def deco(cls):
                return cls
            return deco if not a else a[0]

    registry.ARCH_REGISTRY = _Reg()
    utils.registry = registry
    pkg.utils = utils
    sys.modules.setdefault("basicsr", pkg)
    sys.modules.setdefault("basicsr.utils", utils)
    sys.modules.setdefault("basicsr.utils.registry", registry)

    path = os.path.join(repo, "code", "PBFG", "basicsr", "archs", "FGRNet_arch.py")
    spec = importlib.util.spec_from_file_location("fgrnet_arch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Uformer


def remap_state_dict(sd):
    """sfb -> sfem; drop dead cab.* keys (fusion is declared in the released
    arch, so its — untrained — keys load as-is and stay unused)."""
    out = {}
    for k, v in sd.items():
        if ".cab." in k:
            continue
        out[k.replace(".sfb.", ".sfem.")] = v
    return out


def adjust_gamma(t, gamma):
    return t.clamp(1e-7, 1.0) ** gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--repo", default=os.environ.get("PBFG_REPO", "pbfg"))
    ap.add_argument("--ckpt", default=os.environ.get(
        "PBFG_CKPT", "pbfg_ckpt/checkpoint/net_g_last.pth"))
    ap.add_argument("--gt", default=None)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    Uformer = load_arch(args.repo)
    model = Uformer(img_size=args.size, img_ch=3, output_ch=6)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]
    model.load_state_dict(remap_state_dict(sd), strict=True)
    model.eval()
    print("checkpoint loaded")

    os.makedirs(args.out, exist_ok=True)
    if os.path.isdir(args.inp):
        files = sorted(os.path.join(args.inp, f) for f in os.listdir(args.inp))
    else:
        files = [args.inp]
    gts = sorted(os.path.join(args.gt, f) for f in os.listdir(args.gt)) \
        if args.gt else [None] * len(files)

    gamma = torch.tensor([2.2])
    for path, gtp in zip(files, gts):
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        orig_size = img.size
        x = torch.from_numpy(
            np.asarray(img.resize((args.size, args.size), Image.BICUBIC),
                       dtype=np.float32) / 255.).permute(2, 0, 1)[None]
        t0 = time.time()
        with torch.no_grad():
            y = model(x)
        deflare, flare = y[:, :3], y[:, 3:]
        print(f"{name}: {time.time()-t0:.1f}s")

        def save(t, suffix):
            arr = (t[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).resize(orig_size, Image.BICUBIC).save(
                os.path.join(args.out, f"{name}_{suffix}.png"))

        save(deflare, "deflare")
        save(flare, "flare")
        if gtp:
            from skimage.exposure import match_histograms
            gt = Image.open(gtp).convert("RGB").resize((args.size, args.size),
                                                       Image.BICUBIC)
            gt = torch.from_numpy(np.asarray(gt, dtype=np.float32) / 255.
                                  ).permute(2, 0, 1)[None]
            blend = match_histograms(deflare[0].numpy(), gt[0].numpy(),
                                     channel_axis=0)
            save(torch.from_numpy(blend)[None], "blend")


if __name__ == "__main__":
    main()
