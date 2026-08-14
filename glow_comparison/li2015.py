"""Glow decomposition from Li, Tan & Brown, "Nighttime Haze Removal with Glow
and Multiple Light Colors", ICCV 2015 (Sec. 4: glow separation via the
relative-smoothness layer separation of Li & Brown, CVPR 2014).

Model:  I = J + G, where
  - J (scene / nighttime-haze layer) has sparse, long-tail gradients,
  - G (glow layer, APSF-convolved light) is smooth: short-tail gradients,
    penalized through the squared Laplacian.

Objective (paper Eq. 7):
    min_J  sum_x |grad J|_0  +  lambda * || Laplacian(I - J) ||^2
    s.t.   0 <= J <= I
solved by half-quadratic splitting: auxiliary gradient variables g with
group hard-thresholding (sum_c |g| < 1/beta -> 0), quadratic subproblem
solved in closed form via FFT, beta = 2^(i-1)/thr doubled each outer
iteration. Port aligned with the authors' released MATLAB solver and the
reference implementation glow_estimation.py provided by the user.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import correlate1d, gaussian_filter
import cv2


def psf2otf(psf, shape):
    """MATLAB-style psf2otf: zero-pad PSF to `shape`, circularly shift the
    kernel center to (0,0), return its 2-D FFT."""
    psf = np.asarray(psf, dtype=np.float64)
    pad = np.zeros(shape, dtype=np.float64)
    pad[: psf.shape[0], : psf.shape[1]] = psf
    for axis, s in enumerate(psf.shape):
        pad = np.roll(pad, -(s // 2), axis=axis)
    return fft2(pad)


def sept(I, lambda_, lb, hb, L1_0=None, n_iter=3, thr=0.05):
    """Relative-smoothness layer separation. I: HxWxD in [0,1].
    Returns (L1, L2): textured scene layer and smooth glow layer L2 = I - L1."""
    I = np.asarray(I, dtype=np.float64)
    L1 = I.copy() if L1_0 is None else L1_0.copy()
    H, W, D = I.shape

    f1 = np.array([1, -1], dtype=np.float64)
    f3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    otfFx = psf2otf(f1[None, :], (H, W))
    otfFy = psf2otf(f1[:, None], (H, W))
    otfL = psf2otf(f3, (H, W))

    I_fft = fft2(I, axes=(0, 1))
    absL2 = np.abs(otfL) ** 2
    Normin1 = absL2[:, :, None] * I_fft
    Denormin1 = absL2[:, :, None]
    Denormin2 = (np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2)[:, :, None]

    eps = 1e-16
    for i in range(1, n_iter + 1):
        beta = (2.0 ** (i - 1)) / thr
        Den = lambda_ * Denormin1 + beta * Denormin2

        # g-subproblem: group hard-thresholding of gradients across channels
        gFx = -correlate1d(L1, f1, axis=1, mode='wrap')
        gFy = -correlate1d(L1, f1, axis=0, mode='wrap')
        t = np.sum(np.abs(gFx), axis=2) < 1.0 / beta
        gFx[np.repeat(t[:, :, None], D, axis=2)] = 0
        t = np.sum(np.abs(gFy), axis=2) < 1.0 / beta
        gFy[np.repeat(t[:, :, None], D, axis=2)] = 0

        # divergence (transpose of forward difference, circular)
        div_x = np.concatenate(
            [gFx[:, -1:, :] - gFx[:, :1, :], -np.diff(gFx, axis=1)], axis=1)
        div_y = np.concatenate(
            [gFy[-1:, :, :] - gFy[:1, :, :], -np.diff(gFy, axis=0)], axis=0)
        Normin2 = div_x + div_y

        F_L1 = (lambda_ * Normin1 + beta * fft2(Normin2, axes=(0, 1))) / (Den + eps)
        L1 = np.real(ifft2(F_L1, axes=(0, 1)))

        # per-channel scalar re-centering into [lb, hb], then clip
        for c in range(D):
            L1c = L1[:, :, c]
            for _ in range(500):
                dt = (L1c[L1c < lb[:, :, c]].sum() +
                      (L1c - hb[:, :, c])[L1c > hb[:, :, c]].sum()) * 2 / L1c.size
                L1c = L1c - dt
                if abs(dt) < 1.0 / L1c.size:
                    break
            L1[:, :, c] = np.clip(L1c, lb[:, :, c], hb[:, :, c])

    L2 = I - L1
    return L1, L2


def glow_decomposition(I, ds_scale=0.25, base_lambda=25000.0, presmooth_sigma=2.0,
                       gain=1.0):
    """Full glow-map pipeline following the reference implementation:
    downsample, per-channel Gaussian pre-smoothing, relative-smoothness
    separation with resolution-scaled lambda, upsample glow back.

    I: HxWx3 float in [0,1]. Returns (J, G) at the input resolution.
    """
    I = np.asarray(I, dtype=np.float64)
    H0, W0 = I.shape[:2]
    small = cv2.resize(I, dsize=None, fx=ds_scale, fy=ds_scale,
                       interpolation=cv2.INTER_AREA)
    for c in range(small.shape[2]):
        small[:, :, c] = gaussian_filter(small[:, :, c], sigma=presmooth_sigma)

    H, W, D = small.shape
    lam = base_lambda * ds_scale * ds_scale
    _, G_small = sept(small, lam, np.zeros((H, W, D)), small)

    G = cv2.resize(G_small, (W0, H0), interpolation=cv2.INTER_LINEAR)
    G = np.clip(G * gain, 0, 1)
    J = np.clip(I - G, 0, 1)
    return J, G


if __name__ == "__main__":
    import os
    import sys

    inp, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    bgr = cv2.imread(inp).astype(np.float64) / 255.0
    I = bgr[..., ::-1]
    J, G = glow_decomposition(I)
    cv2.imwrite(os.path.join(outdir, "li_scene.png"),
                (np.clip(J[..., ::-1], 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(outdir, "li_glow.png"),
                (np.clip(G[..., ::-1], 0, 1) * 255).astype(np.uint8))
    print("saved to", outdir)
