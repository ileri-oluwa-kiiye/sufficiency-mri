from __future__ import annotations

import numpy as np


def _grid(shape):
    h, w = shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    return yy, xx


def smooth_field_2d(shape, rng: np.random.Generator, order: int = 3) -> np.ndarray:
    """Random low-order polynomial field over the image plane, scaled so max |f| = 1."""
    yy, xx = _grid(shape)
    field = np.zeros(shape, dtype=np.float32)
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0:
                continue
            coef = np.float32(rng.normal() / (1.0 + i + j))  # weaker high-order terms
            field += coef * (yy ** i) * (xx ** j)
    peak = float(np.abs(field).max())
    return field / peak if peak > 0 else field


def phase_map(shape, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """Smooth background phase (radians) with max |phase| = amplitude.

    Real MR images are complex with slowly varying phase (B0 inhomogeneity, coil phase).
    Without it, k-space of a magnitude image is conjugate-symmetric, which makes partial
    Fourier and motion artifacts unrealistically mild.
    """
    if amplitude <= 0:
        return np.zeros(shape, dtype=np.float32)
    return (amplitude * smooth_field_2d(shape, rng, order=3)).astype(np.float32)


def bias_field(shape3d, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Multiplicative intensity inhomogeneity exp(strength * f), with max |f| = 1.

    strength = 0.2 gives intensities varying by roughly +/-20% across the brain.
    """
    h, w, nz = shape3d
    if strength <= 0:
        return np.ones(shape3d, dtype=np.float32)
    f = smooth_field_2d((h, w), rng, order=3)[:, :, None]
    f = f + np.float32(rng.uniform(-0.5, 0.5)) * np.linspace(-1.0, 1.0, nz, dtype=np.float32)[None, None, :]
    f = f / max(float(np.abs(f).max()), 1e-6)
    return np.exp(strength * f).astype(np.float32)
