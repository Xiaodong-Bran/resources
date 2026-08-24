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


def load_arch(repo, act="relu", order="sf"):
    """Import FGRNet_arch.py standalone (stubbing basicsr.utils.registry) and
    patch New_TransformerBlock to match the released checkpoint.

    The released code and checkpoint diverge: the checkpoint names the SFEM
    'sfb' and adds a channel-attention branch 'cab' fused with the declared
    (but unused) `self.fusion` conv. CAB's structure is reconstructed from
    the checkpoint tensor shapes:
        cab.cab = Sequential(Conv(d,d//3,3), BN(d//3), <act>, Conv(d//3,d,3),
                             CA(d))     with CA: x * Sigmoid(Conv(1,d,1)(
                             ReLU(Conv(d,1,1)(AvgPool(x)))))
    The paramless activation and the concat order in fusion are ambiguous;
    `act` in {relu,gelu} and `order` in {sf,fs} select a variant, to be
    validated empirically against the official GT test data.
    """
    import math
    import torch.nn as nn
    from einops import rearrange

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

    class ChanAtt(nn.Module):
        def __init__(self, dim):
            super().__init__()
            mid = max(1, dim // 30)
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, mid, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, dim, 1),
                nn.Sigmoid())

        def forward(self, x):
            return x * self.attention(x)

    class CAB(nn.Module):
        def __init__(self, dim):
            super().__init__()
            a = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()
            self.cab = nn.Sequential(
                nn.Conv2d(dim, dim // 3, 3, 1, 1),
                nn.BatchNorm2d(dim // 3),
                a,
                nn.Conv2d(dim // 3, dim, 3, 1, 1),
                ChanAtt(dim))

        def forward(self, x):
            return self.cab(x)

    Blk = mod.New_TransformerBlock
    orig_init = Blk.__init__

    def new_init(self, dim, *a, **k):
        orig_init(self, dim, *a, **k)
        self.sfb = self._modules.pop("sfem")
        self.cab = CAB(dim)

    def new_forward(self, x, mask=None):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        if mask is not None:
            import torch.nn.functional as F
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            imw = mod.window_partition(input_mask, self.win_size)
            attn_mask = imw.view(-1, self.win_size * self.win_size)
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)
                                              ).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        if self.shift_size > 0:
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            slices = (slice(0, -self.win_size),
                      slice(-self.win_size, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            smw = mod.window_partition(shift_mask, self.win_size)
            smw = smw.view(-1, self.win_size * self.win_size)
            sam = smw.unsqueeze(1) - smw.unsqueeze(2)
            sam = sam.masked_fill(sam != 0, float(-100.0)).masked_fill(sam == 0, 0.0)
            attn_mask = attn_mask + sam if attn_mask is not None else sam

        shortcut = x
        x = self.norm1(x).view(B, H, W, C).permute(0, 3, 1, 2)
        shifted_x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2)) \
            if self.shift_size > 0 else x
        xw = mod.window_partition(shifted_x, self.win_size)
        xw = xw.view(-1, self.win_size * self.win_size, C)
        aw = self.attn(xw, mask=attn_mask)
        aw = aw.view(-1, self.win_size, self.win_size, C)
        shifted_x = mod.window_reverse(aw, self.win_size, H, W)
        x1 = torch.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2)) \
            if self.shift_size > 0 else shifted_x
        x1 = x1.view(B, H * W, C)
        x = shortcut + self.drop_path(x1)

        a1 = self.mlp(self.norm2(x))
        a1 = rearrange(a1, ' b (h w) c -> b c h w ', h=H, w=W)
        s = self.sfb(a1)
        c = self.cab(a1)
        pair = [s, c] if order == "sf" else [c, s]
        out = self.fusion(torch.cat(pair, dim=1))
        out = rearrange(out, ' b c h w -> b (h w) c')
        return x + self.drop_path(out)

    Blk.__init__ = new_init
    Blk.forward = new_forward
    return mod.Uformer


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
    ap.add_argument("--act", default="relu", choices=["relu", "gelu"])
    ap.add_argument("--order", default="sf", choices=["sf", "fs"])
    args = ap.parse_args()

    Uformer = load_arch(args.repo, act=args.act, order=args.order)
    model = Uformer(img_size=args.size, img_ch=3, output_ch=6)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]
    model.load_state_dict(sd)
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
