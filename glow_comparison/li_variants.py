"""Robustness-first improvements over Li 2015 — no detection stage, few
parameters, weak priors only (they inherit Li's graceful-degradation
behavior, unlike the strong-assumption GAR pipeline).

A. li_wls: Li decomposition + WLS edge-aware refinement of the glow layer,
   guided by the scene layer — the published post-process of the sea-fog
   defogging paper (TIP 2020, layer_decom.m "fast" path), adapted to keep
   color (per-channel filtering instead of min-channel collapse).

B. li_multiscale: fuse Li run at full resolution (compact halos survive)
   with Li at half resolution (smoother large-scale glow) by per-pixel max.
"""
import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from li2015 import glow_decomposition


def wls_filter(img, guide, lam=0.5, alpha=1.2, eps=1e-4):
    """Farbman et al. 2008 weighted least squares smoothing of `img` with
    gradient weights computed from `guide` (both HxW, float)."""
    H, W = img.shape
    n = H * W
    g = np.log(guide + eps)

    # row-major flattening: index = y*W + x; right neighbor +1, down neighbor +W
    dxr = np.diff(g, axis=1)
    dxr = -lam / (np.abs(dxr) ** alpha + eps)
    dxr = np.hstack([dxr, np.zeros((H, 1))]).ravel()
    dyd = np.diff(g, axis=0)
    dyd = -lam / (np.abs(dyd) ** alpha + eps)
    dyd = np.vstack([dyd, np.zeros((1, W))]).ravel()

    A = sp.diags([dxr[:-1], dyd[:-W]], [1, W], shape=(n, n))
    A = A + A.T
    D = 1.0 - np.asarray(A.sum(axis=1)).ravel()
    A = A + sp.diags(D)
    out = spsolve(A.tocsr(), img.ravel())
    return out.reshape(H, W)


def li_wls(I, base_lambda=1562.5, wls_lam=0.5, wls_alpha=1.2):
    """Li 2015 + WLS refinement: smooth each glow channel with the scene
    layer as guide, killing leaked scene texture while following glow."""
    I = np.asarray(I, dtype=np.float64)
    J, G = glow_decomposition(I, ds_scale=1.0, base_lambda=base_lambda,
                              presmooth_sigma=2.0)
    guide = J.min(axis=2)  # scene structure to smooth across
    G_ref = np.stack([wls_filter(G[..., c], guide, wls_lam, wls_alpha)
                      for c in range(3)], axis=-1)
    G_ref = np.clip(G_ref, 0, 1)
    return np.clip(I - G_ref, 0, 1), G_ref


def li_multiscale(I, base_lambda=1562.5):
    """Fuse Li at full and half resolution by per-pixel max."""
    I = np.asarray(I, dtype=np.float64)
    _, G1 = glow_decomposition(I, ds_scale=1.0, base_lambda=base_lambda,
                               presmooth_sigma=2.0)
    _, G2 = glow_decomposition(I, ds_scale=0.5, base_lambda=base_lambda,
                               presmooth_sigma=2.0)
    G = np.clip(np.maximum(G1, G2), 0, 1)
    return np.clip(I - G, 0, 1), G
