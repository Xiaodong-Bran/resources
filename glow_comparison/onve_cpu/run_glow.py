"""CPU runner for ONVE glow-map extraction (step 1 of Deglow.py only).

Usage: python run_glow.py <image_path> <output_dir> [--iters N] [--max-side S]

Runs the LBDN glow decomposition from Deglow.py (Segmentation.optimize_step1)
on CPU and saves glow_map.jpg / gray_map.jpg / noGlow.jpg into <output_dir>.
The algorithm is unchanged from the original repo; only device (CUDA -> CPU),
input resolution and iteration count are adjustable for CPU feasibility.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Deglow
from Deglow import Segmentation
from Segment import create_soft_mask
from utils.image_io import prepare_image


def make_hints(image_path, tmp_dir):
    """Reproduce Segment.py hint generation for a single image."""
    os.makedirs(tmp_dir, exist_ok=True)
    light_mask = create_soft_mask(image_path, thresh=250, blur_size=31, morph_size=7)
    fg_mask = create_soft_mask(image_path, thresh=100, blur_size=51, morph_size=9)
    img = cv2.imread(image_path, 0)
    _, bg_mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    paths = {}
    for name, m in [("light", light_mask), ("fg", fg_mask), ("bg", bg_mask)]:
        p = os.path.join(tmp_dir, f"{name}.png")
        cv2.imwrite(p, m)
        paths[name] = p
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output_dir")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--max-side", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # resize input for CPU feasibility (keep aspect, multiple of 32 handled by prepare_image)
    im = Image.open(args.image).convert("RGB")
    scale = args.max_side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)), Image.LANCZOS)
    resized_path = os.path.join(args.output_dir, "input_resized.png")
    im.save(resized_path)

    hints = make_hints(resized_path, os.path.join(args.output_dir, "hints"))

    i, _, _ = prepare_image(resized_path)
    bg, _, _ = prepare_image(hints["bg"])
    fg, _, _ = prepare_image(hints["fg"])
    light, _, _ = prepare_image(hints["light"])

    # Deglow.Segmentation reads the module-level global `output_dir`
    Deglow.output_dir = args.output_dir

    t0 = time.time()
    seg = Segmentation("img", i, bg_hint=bg, fg_hint=fg, light_hint=light,
                       output_dir=args.output_dir,
                       first_step_iter_num=args.iters,
                       plot_during_training=True)
    seg.show_every = min(500, args.iters)
    seg.optimize_step1()
    # save final glow map explicitly
    seg._plot_with_name()
    print(f"\nDone in {time.time() - t0:.1f}s -> {args.output_dir}/glow_map.jpg")


if __name__ == "__main__":
    main()
