"""Metrics for comparing estimated glow maps.

Full-reference (synthetic, vs ground-truth glow):
  - PSNR, SSIM, MAE

No-reference (real images, no GT):
  - smoothness:  mean |Laplacian(G)| — a physically plausible glow (APSF
    convolution) is smooth, lower is better.
  - leakage:     Pearson correlation between |grad G| and |grad I| over
    non-highlight pixels — measures how much scene texture leaked into the
    glow map, lower is better.
  - neg_residual: fraction of pixels where I - G < -1/255 — glow
    over-estimation that would darken the scene below zero, lower is better.
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_gray(img):
    if img.ndim == 3:
        return img.mean(axis=2)
    return img


def full_reference(G_est, G_gt):
    G_est = np.clip(G_est.astype(np.float64), 0, 1)
    G_gt = np.clip(G_gt.astype(np.float64), 0, 1)
    return {
        "psnr": float(peak_signal_noise_ratio(G_gt, G_est, data_range=1.0)),
        "ssim": float(structural_similarity(to_gray(G_gt), to_gray(G_est),
                                            data_range=1.0)),
        "mae": float(np.abs(G_est - G_gt).mean()),
    }


def grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def no_reference(G_est, I, highlight_thresh=0.85):
    G = np.clip(G_est.astype(np.float64), 0, 1)
    I = np.clip(I.astype(np.float64), 0, 1)
    gG, gI = to_gray(G), to_gray(I)

    lap = cv2.Laplacian(gG, cv2.CV_64F, ksize=3)
    smoothness = float(np.abs(lap).mean())

    mask = to_gray(I) < highlight_thresh  # exclude saturated light cores
    mG, mI = grad_mag(gG)[mask], grad_mag(gI)[mask]
    if mG.std() < 1e-9 or mI.std() < 1e-9:
        leakage = 0.0
    else:
        leakage = float(np.corrcoef(mG, mI)[0, 1])

    neg = float(((I - G) < -1.0 / 255).mean())
    return {"smoothness": smoothness, "leakage": leakage, "neg_residual": neg}
