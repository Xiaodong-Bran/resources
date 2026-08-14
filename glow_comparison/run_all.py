"""Orchestrates the glow-map comparison between ONVE (LBDN, zero-shot deep
prior) and Li et al. ICCV 2015 (relative-smoothness layer separation).

For every test image:
  1. run ONVE step-1 glow extraction (onve_cpu/run_glow.py, subprocess)
  2. run Li 2015 glow decomposition on the identical working-resolution input
  3. compute metrics (full-reference vs GT for synthetic; no-reference for all)
  4. save a side-by-side composite (input | ONVE glow | Li glow [| GT glow])

Results land in results/: per-image folders, metrics.json, composites.
"""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from li2015 import glow_decomposition
from metrics import full_reference, no_reference

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
SCRATCH = os.environ.get("GLOW_SCRATCH", "/tmp/glow_runs")
ONVE_ITERS = int(os.environ.get("ONVE_ITERS", "2500"))
MAX_SIDE = int(os.environ.get("ONVE_MAX_SIDE", "256"))
LI_LAMBDA = float(os.environ.get("LI_LAMBDA", "1562.5"))


def crop_to_multiple(img, d=32):
    """Replicates utils.image_io.crop_image (center crop to multiple of d)."""
    h, w = img.shape[:2]
    nh, nw = h - h % d, w - w % d
    top, left = (h - nh) // 2, (w - nw) // 2
    return img[top:top + nh, left:left + nw]


def imread01(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return bgr[..., ::-1].astype(np.float64) / 255.0


def imwrite01(path, rgb):
    cv2.imwrite(path, (np.clip(rgb, 0, 1)[..., ::-1] * 255).round().astype(np.uint8))


def run_onve(image_path, out_dir):
    env = dict(os.environ)
    env.setdefault("VGG16BN_FEATURES_WEIGHTS",
                   os.path.join(SCRATCH, "vgg16bn_features_tv.pth"))
    t0 = time.time()
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "onve_cpu", "run_glow.py"),
         image_path, out_dir, "--iters", str(ONVE_ITERS),
         "--max-side", str(MAX_SIDE)],
        check=True, env=env, cwd=os.path.join(ROOT, "onve_cpu"),
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return time.time() - t0


def process(name, image_path, gt_glow_path=None):
    out = os.path.join(RESULTS, name)
    onve_dir = os.path.join(out, "onve")
    os.makedirs(onve_dir, exist_ok=True)

    print(f"=== {name}: ONVE ({ONVE_ITERS} iters, max side {MAX_SIDE}) ...",
          flush=True)
    t_onve = run_onve(image_path, onve_dir)

    # identical working-resolution input used by ONVE
    I = crop_to_multiple(imread01(os.path.join(onve_dir, "input_resized.png")))
    imwrite01(os.path.join(out, "input.png"), I)

    G_onve = imread01(os.path.join(onve_dir, "glow_map.jpg"))
    if G_onve.shape != I.shape:
        G_onve = cv2.resize(G_onve, (I.shape[1], I.shape[0]))

    print(f"=== {name}: Li2015 ...", flush=True)
    t0 = time.time()
    _, G_li = glow_decomposition(I, ds_scale=1.0, base_lambda=LI_LAMBDA,
                                 presmooth_sigma=2.0)
    t_li = time.time() - t0

    imwrite01(os.path.join(out, "glow_onve.png"), G_onve)
    imwrite01(os.path.join(out, "glow_li2015.png"), G_li)

    entry = {"onve_seconds": round(t_onve, 1), "li_seconds": round(t_li, 2),
             "onve": {}, "li2015": {}}
    panels = [I, G_onve, G_li]

    if gt_glow_path:
        G_gt = crop_to_multiple(imread01(gt_glow_path))
        if G_gt.shape != I.shape:
            G_gt = cv2.resize(G_gt, (I.shape[1], I.shape[0]))
        imwrite01(os.path.join(out, "glow_gt.png"), G_gt)
        entry["onve"].update(full_reference(G_onve, G_gt))
        entry["li2015"].update(full_reference(G_li, G_gt))
        panels.append(G_gt)

    entry["onve"].update(no_reference(G_onve, I))
    entry["li2015"].update(no_reference(G_li, I))

    comp = np.concatenate(panels, axis=1)
    imwrite01(os.path.join(RESULTS, f"compare_{name}.png"), comp)
    print(json.dumps({name: entry}, indent=2), flush=True)
    return entry


def main():
    os.makedirs(RESULTS, exist_ok=True)
    jobs = []
    syn_dir = os.path.join(ROOT, "data", "synthetic")
    for n in ["syn1", "syn2", "syn3"]:
        jobs.append((n, os.path.join(syn_dir, f"{n}_input.png"),
                     os.path.join(syn_dir, f"{n}_glow_gt.png")))
    img_dir = os.path.join(ROOT, "onve_cpu", "images")
    for f in ["DSC01015.jpg", "DSC00983.jpg", "DSC01173.jpg", "input_000009.png"]:
        jobs.append((os.path.splitext(f)[0], os.path.join(img_dir, f), None))

    metrics = {}
    mpath = os.path.join(RESULTS, "metrics.json")
    if os.path.exists(mpath):
        metrics = json.load(open(mpath))
    for name, img, gt in jobs:
        if name in metrics:
            print(f"=== {name}: already done, skipping", flush=True)
            continue
        metrics[name] = process(name, img, gt)
        json.dump(metrics, open(mpath, "w"), indent=2)
    print("all done ->", mpath)


if __name__ == "__main__":
    main()
