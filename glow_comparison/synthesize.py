"""Build a synthetic nighttime-glow benchmark with ground-truth glow layers.

Model: I = clip(J + G), G = L (x) APSF, following the glow imaging model of
Li et al. ICCV'15 / Narasimhan-Nayar APSF. The APSF is approximated by a
heavy-tailed generalized Gaussian kernel, which produces the characteristic
long glow tails around light sources.

Outputs per scene under data/synthetic/:
  <name>_input.png     I  (observed nighttime image with glow)
  <name>_glow_gt.png   G  (ground-truth glow layer)
  <name>_scene_gt.png  J  (glow-free scene)
"""
import os

import cv2
import numpy as np
from skimage import data
from skimage.transform import resize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic")
SIZE = 256  # multiple of 32 so both methods run at native resolution


def apsf_kernel(size=201, sigma=6.0, p=0.55):
    """Heavy-tailed generalized-Gaussian approximation of the APSF."""
    r = np.hypot(*np.mgrid[-(size // 2): size // 2 + 1,
                           -(size // 2): size // 2 + 1].astype(np.float64))
    k = np.exp(-(r / sigma) ** p)
    return k / k.sum()


def make_scene(base_rgb, darken=0.32, gamma=1.6):
    """Turn a clean photo into a plausible glow-free night scene."""
    J = base_rgb.astype(np.float64) / 255.0
    J = resize(J, (SIZE, SIZE, 3), anti_aliasing=True)
    J = darken * (J ** gamma)
    return J


def add_lights(shape, lights):
    """lights: list of (y, x, radius, (r,g,b), intensity)."""
    L = np.zeros(shape, dtype=np.float64)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (y, x, rad, color, inten) in lights:
        disk = ((yy - y) ** 2 + (xx - x) ** 2) <= rad ** 2
        for c in range(3):
            L[..., c][disk] = np.maximum(L[..., c][disk], inten * color[c])
    return L


def build(name, base, lights, sigma, p, hdr_gain=60.0):
    J = make_scene(base)
    L = add_lights(J.shape, lights)
    # real light sources are HDR: far brighter than the display range, so the
    # APSF-convolved glow stays visible even though the kernel integrates to 1
    k = apsf_kernel(sigma=sigma, p=p)
    G = np.stack([cv2.filter2D(hdr_gain * L[..., c], -1, k,
                               borderType=cv2.BORDER_REPLICATE)
                  for c in range(3)], axis=-1)
    G = np.clip(G, 0, 1)
    # burn saturated light cores into the scene so hint generation finds them
    cores = np.clip(L, 0, 1)
    I = np.clip(J + G + cores, 0, 1)
    Jgt = np.clip(J + cores, 0, 1)

    os.makedirs(OUT, exist_ok=True)
    for suffix, img in [("input", I), ("glow_gt", G), ("scene_gt", Jgt)]:
        cv2.imwrite(os.path.join(OUT, f"{name}_{suffix}.png"),
                    (img[..., ::-1] * 255).round().astype(np.uint8))
    print(f"{name}: glow mean={G.mean():.4f} max={G.max():.3f}")


def main():
    rng = np.random.default_rng(0)

    # scene 1: street-like, two strong warm lights + one cool light
    build("syn1", data.astronaut(),
          lights=[(60, 70, 4, (1.0, 0.85, 0.55), 1.0),
                  (90, 200, 3, (1.0, 0.9, 0.7), 1.0),
                  (200, 130, 5, (0.7, 0.85, 1.0), 1.0)],
          sigma=5.0, p=0.6)

    # scene 2: single dominant colored light, broader glow
    build("syn2", data.chelsea(),
          lights=[(110, 128, 6, (1.0, 0.6, 0.3), 1.0)],
          sigma=9.0, p=0.5)

    # scene 3: many small lights, tighter glow
    lights = [(int(y), int(x), 2, (1.0, 0.95, 0.8), 1.0)
              for y, x in rng.integers(20, SIZE - 20, size=(6, 2))]
    build("syn3", data.coffee(), lights, sigma=3.5, p=0.7)


if __name__ == "__main__":
    main()
