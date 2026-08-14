"""GAR — Glow via APSF Refit. An improved glow-map constructor built on
Li et al. ICCV 2015, inspired by the guided-APSF idea of Jin et al.
(ACMMM 2023, "Enhancing Visibility in Nighttime Haze Images Using Guided
APSF and Gradient Adaptive Convolution").

Pipeline:
  1. Relative-smoothness decomposition (li2015.glow_decomposition) -> G_li.
  2. Detect saturated light-source cores in the input.
  3. Fit a physically parameterized APSF (heavy-tailed generalized Gaussian,
     shared sigma/p, per-channel amplitude) so that APSF (x) cores matches
     G_li in least squares.
  4. Re-render the glow from the fitted APSF, and add back only the
     heavily low-passed residual of G_li (large-scale ambient cast).

The re-rendered part is structure-free by construction (it is a convolution
of a point-source map), which removes the scene-texture leakage that the
plain relative-smoothness solution suffers from, while keeping — and often
sharpening — the glow amplitude around the true light sources.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from li2015 import glow_decomposition


def _apsf_kernel(size, sigma, p):
    r = np.hypot(*np.mgrid[-(size // 2): size // 2 + 1,
                           -(size // 2): size // 2 + 1].astype(np.float64))
    k = np.exp(-(r / max(sigma, 1e-3)) ** max(p, 0.05))
    # taper to zero at the support boundary so heavy tails do not truncate
    # into a visible square edge
    k = np.maximum(k - np.exp(-((size // 2) / max(sigma, 1e-3)) ** max(p, 0.05)), 0)
    s = k.sum()
    return k / s if s > 0 else k


def detect_cores(I, thresh=0.9, min_pix=1):
    """Bright light-source cores (max-channel >= thresh); returns HxWx3
    colored core map and binary mask. The threshold is deliberately below
    full saturation so strongly colored sources (e.g. a bright blue sign,
    whose max channel peaks near but under 1.0) still count as sources."""
    gray = I.max(axis=2)
    mask = (gray >= thresh).astype(np.float64)
    if mask.sum() < min_pix:
        mask = (gray >= np.quantile(gray, 0.999)).astype(np.float64)
    return I * mask[..., None], mask


def fit_apsf(G_li, cores, size=129, fit_scale=0.5):
    """Fit (sigma, p) shared + per-channel amplitude a_c minimizing the
    weighted LS error || w * (a_c * (k(sigma,p) (x) cores_c) - G_li_c) ||^2.
    The weight w ~ glow intensity focuses the fit on the halo shape near the
    sources instead of the vast dark background. Fitting runs at reduced
    resolution for speed; the final render is full resolution."""
    H, W, D = G_li.shape
    size = min(size, (min(H, W) // 2) * 2 + 1)

    # reduced-resolution copies for the parameter search
    Gs = cv2.resize(G_li, dsize=None, fx=fit_scale, fy=fit_scale,
                    interpolation=cv2.INTER_AREA)
    Cs = cv2.resize(cores, dsize=None, fx=fit_scale, fy=fit_scale,
                    interpolation=cv2.INTER_AREA)
    w = Gs.mean(axis=2) ** 0.7 + 1e-3
    w3 = w[..., None]

    def render(C, sigma, p, scale):
        sz = min(int(size * scale) | 1, (min(C.shape[:2]) // 2) * 2 + 1)
        k = _apsf_kernel(sz, sigma * scale, p)
        return np.stack([cv2.filter2D(C[..., c], -1, k,
                                      borderType=cv2.BORDER_REPLICATE)
                         for c in range(D)], axis=-1)

    def amps(B, G, w3):
        a = np.zeros(D)
        for c in range(D):
            denom = float((w3[..., 0] * B[..., c] ** 2).sum())
            a[c] = (float((w3[..., 0] * B[..., c] * G[..., c]).sum()) / denom
                    if denom > 0 else 0.0)
        return np.clip(a, 0, None)

    def cost(x):
        sigma, p = np.exp(x[0]), x[1]
        if not (0.2 <= p <= 1.3) or not (0.5 <= sigma <= 100):
            return 1e9
        B = render(Cs, sigma, p, fit_scale)
        a = amps(B, Gs, w3)
        r = a[None, None, :] * B - Gs
        return float((w3 * r * r).sum())

    best = None
    for s0 in [2.0, 5.0, 12.0, 30.0]:
        for p0 in [0.5, 0.9]:
            res = minimize(cost, [np.log(s0), p0], method="Nelder-Mead",
                           options={"maxiter": 50, "xatol": 1e-2, "fatol": 1e-7})
            if best is None or res.fun < best.fun:
                best = res
    sigma, p = np.exp(best.x[0]), best.x[1]
    B_full = render(cores, sigma, p, 1.0)
    Bs = render(Cs, sigma, p, fit_scale)
    a = amps(Bs, Gs, w3)
    return {"sigma": sigma, "p": p, "amps": a.tolist()}, a[None, None, :] * B_full


def fit_apsf_envelope(I, cores, core_mask, size=257, fit_scale=0.5,
                      halo_radius=70, over_penalty=8.0):
    """Fit the APSF directly to the input image as an asymmetric lower
    envelope: within the halo neighborhood of the light sources the input is
    glow + (mostly dark) scene, so the rendered glow must stay below I —
    overshoot is penalized `over_penalty` times harder than undershoot.
    This bypasses the shape bias of the relative-smoothness decomposition."""
    H, W, D = I.shape
    size = min(size, (min(H, W) // 2) * 2 + 1)

    Is = cv2.resize(I, dsize=None, fx=fit_scale, fy=fit_scale,
                    interpolation=cv2.INTER_AREA)
    Cs = cv2.resize(cores, dsize=None, fx=fit_scale, fy=fit_scale,
                    interpolation=cv2.INTER_AREA)
    Ms = cv2.resize(core_mask.astype(np.float64), dsize=None,
                    fx=fit_scale, fy=fit_scale,
                    interpolation=cv2.INTER_AREA) > 0.25

    # halo region: near sources but outside the saturated cores
    dist = cv2.distanceTransform((~Ms).astype(np.uint8), cv2.DIST_L2, 3)
    region = (dist > 1) & (dist <= halo_radius * fit_scale)
    if region.sum() < 50:
        region = dist > 1
    w = 1.0 / (1.0 + dist / (10.0 * fit_scale))  # nearer halo counts more
    w = w * region

    def render(C, sigma, p, scale):
        sz = min(int(size * scale) | 1, (min(C.shape[:2]) // 2) * 2 + 1)
        k = _apsf_kernel(sz, sigma * scale, p)
        return np.stack([cv2.filter2D(C[..., c], -1, k,
                                      borderType=cv2.BORDER_REPLICATE)
                         for c in range(C.shape[2])], axis=-1)

    def asym_cost(r, w):  # r = model - target
        return float((w * np.where(r > 0, over_penalty * r * r, r * r)).sum())

    def amp_for(Bc, Ic, w):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda a: asym_cost(a * Bc - Ic, w),
                              bounds=(0.0, 500.0), method="bounded",
                              options={"xatol": 1e-3})
        return float(res.x)

    def fit_one(target, w, sig_lo, sig_hi, seeds, base=None):
        """Fit one (sigma, p) + per-channel amps; `base` is an already
        rendered component added to the model inside the envelope cost."""
        base_s = 0.0 if base is None else base

        def cost(x):
            sigma, p = np.exp(x[0]), x[1]
            if not (0.2 <= p <= 1.3) or not (sig_lo <= sigma <= sig_hi):
                return 1e9
            B = render(Cs, sigma, p, fit_scale)
            total = 0.0
            for c in range(D):
                bs = base_s if base is None else base_s[..., c]
                a = amp_for(B[..., c], target[..., c] - bs, w)
                total += asym_cost(bs + a * B[..., c] - target[..., c], w)
            return total

        best = None
        for s0 in seeds:
            for p0 in [0.5, 0.9]:
                res = minimize(cost, [np.log(s0), p0], method="Nelder-Mead",
                               options={"maxiter": 40, "xatol": 1e-2,
                                        "fatol": 1e-8})
                if best is None or res.fun < best.fun:
                    best = res
        sigma, p = np.exp(best.x[0]), best.x[1]
        Bs = render(Cs, sigma, p, fit_scale)
        a = []
        for c in range(D):
            bs = base_s if base is None else base_s[..., c]
            a.append(amp_for(Bs[..., c], target[..., c] - bs, w))
        a = np.array(a)
        return sigma, p, a, a[None, None, :] * Bs

    # two-scale APSF mixture: a real APSF has a sharp peak AND a heavy tail.
    # A single kernel collapses to one or the other (a broad ambient fit
    # erases the compact halo around point lights), so fit a narrow kernel
    # on the near-halo first, then a broad kernel on what remains.
    w_near = w * (dist <= 25 * fit_scale)
    if w_near.sum() < 20:
        w_near = w
    s_n, p_n, a_n, Bn_s = fit_one(Is, w_near, 0.5, 12.0, [1.5, 4.0, 8.0])
    s_b, p_b, a_b, _ = fit_one(Is, w, 8.0, 100.0, [15.0, 40.0], base=Bn_s)

    B_full_n = render(cores, s_n, p_n, 1.0)
    B_full_b = render(cores, s_b, p_b, 1.0)
    G_fit = a_n[None, None, :] * B_full_n + a_b[None, None, :] * B_full_b
    return {"sigma": s_n, "p": p_n, "amps": a_n.tolist(),
            "sigma_broad": s_b, "p_broad": p_b,
            "amps_broad": a_b.tolist()}, G_fit


def gar_glow(I, ambient_sigma=25.0, li_kwargs=None):
    """Full GAR pipeline. I: HxWx3 float [0,1]. Returns (J, G, info)."""
    I = np.asarray(I, dtype=np.float64)
    _, G_li = glow_decomposition(I, **(li_kwargs or dict(ds_scale=1.0,
                                                         base_lambda=1562.5,
                                                         presmooth_sigma=2.0)))
    cores, mask = detect_cores(I)
    params, G_fit = fit_apsf_envelope(I, cores, mask > 0)
    G_fit = np.minimum(G_fit, 1.0)

    # large-scale ambient cast that the point-source model cannot express
    resid = G_li - G_fit
    G_amb = np.clip(np.stack([gaussian_filter(resid[..., c], ambient_sigma)
                              for c in range(3)], axis=-1), 0, None)
    # max-blend: fitted point-source glow provides the peaks, the low-passed
    # residual provides the broad ambient bed — max avoids double counting
    G = np.clip(np.maximum(G_fit, G_amb), 0, 1)
    # physical bound: the observed intensity is an upper envelope of the glow.
    # Use a morphological closing of I as the bound so fine dark scene texture
    # is not imprinted (inverted) into the glow map.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    upper = np.stack([cv2.morphologyEx(I[..., c], cv2.MORPH_CLOSE, kern)
                      for c in range(3)], axis=-1)
    upper = np.stack([gaussian_filter(upper[..., c], 4.0) for c in range(3)],
                     axis=-1)
    # smooth-min toward the envelope: G' = G / (1+(G/U)^k)^(1/k) leaves
    # values below the bound nearly untouched (unlike exponential
    # compression, which attenuates them too) while still approaching U
    # smoothly, so no hard clamped plateaus imprint scene texture edges
    U = np.maximum(upper, 1e-3)
    k = 4.0
    G = G / (1.0 + (G / U) ** k) ** (1.0 / k)
    J = np.clip(I - G, 0, 1)
    params["core_pixels"] = int(mask.sum())
    return J, G, params


if __name__ == "__main__":
    import json
    import os
    import sys

    inp, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    I = cv2.imread(inp)[..., ::-1].astype(np.float64) / 255.0
    J, G, info = gar_glow(I)
    cv2.imwrite(os.path.join(outdir, "gar_glow.png"),
                (np.clip(G[..., ::-1], 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(outdir, "gar_scene.png"),
                (np.clip(J[..., ::-1], 0, 1) * 255).astype(np.uint8))
    print(json.dumps(info, indent=2))
