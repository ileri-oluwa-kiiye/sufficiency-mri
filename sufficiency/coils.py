from __future__ import annotations

import numpy as np

from .fields import smooth_field_2d
from .kspace import fft2c, ifft2c


def coil_maps(shape, n_coils: int, rng: np.random.Generator, width: float = 0.7,
              phase_scale: float = 0.5) -> np.ndarray:
    """Birdcage-like sensitivities: Gaussian profiles centred on a ring around the head.

    Returns complex64 array (C, H, W), normalised so sum_c |S_c|^2 = 1.
    """
    h, w = shape
    if n_coils <= 1:
        return np.ones((1, h, w), dtype=np.complex64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    offset = rng.uniform(0, 2 * np.pi / n_coils)
    maps = []
    for c in range(n_coils):
        angle = offset + 2 * np.pi * c / n_coils
        cy, cx = 1.1 * np.sin(angle), 1.1 * np.cos(angle)
        magnitude = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width ** 2))
        phase = phase_scale * smooth_field_2d(shape, rng, order=2) + angle
        maps.append(magnitude * np.exp(1j * phase))
    maps = np.stack(maps)
    norm = np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return (maps / np.maximum(norm, 1e-8)).astype(np.complex64)


def coil_combine(coil_images: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """Matched-filter (adjoint) combination over the coil axis 0."""
    return np.sum(np.conj(sens) * coil_images, axis=0)


def sense_cg(kspace: np.ndarray, mask: np.ndarray, sens: np.ndarray, iters: int = 10,
             lam: float = 0.0, tol: float = 1e-6) -> np.ndarray:
    """CG-SENSE reconstruction (Pruessmann et al., 2001).

    kspace: (C, H, W, Z) undersampled multi-coil k-space (zeros where not acquired)
    mask:   broadcastable to kspace, True where acquired
    sens:   (C, H, W, 1) coil sensitivities
    Solves (A^H A + lam I) x = A^H y with A = mask * F * S, batched over slices.
    A small number of iterations acts as regularisation, as in clinical practice.
    """
    def normal(x):
        k = mask * fft2c(sens * x[None], axes=(1, 2))
        return coil_combine(ifft2c(k, axes=(1, 2)), sens) + lam * x

    b = coil_combine(ifft2c(mask * kspace, axes=(1, 2)), sens)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    b_norm = rs
    for _ in range(iters):
        if rs <= tol * b_norm:
            break
        ap = normal(p)
        alpha = rs / float(np.vdot(p, ap).real)
        x += alpha * p
        r -= alpha * ap
        rs_new = float(np.vdot(r, r).real)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x
