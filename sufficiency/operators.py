from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

from .acquisition import Acquisition

# ------------------------------------------------------------------ tissue contrast


def kmeans_1d(values: np.ndarray, k: int = 3, iters: int = 30, max_fit: int = 200_000):
    """1D k-means on intensities. Returns sorted centres and labels for all values."""
    values = np.asarray(values, dtype=np.float64)
    step = max(1, values.size // max_fit)
    fit = values[::step]
    centres = np.percentile(fit, np.linspace(100 / (2 * k), 100 - 100 / (2 * k), k))
    for _ in range(iters):
        labels = np.argmin(np.abs(fit[:, None] - centres[None, :]), axis=1)
        new = np.array([fit[labels == j].mean() if np.any(labels == j) else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    centres = np.sort(centres)
    labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
    return centres, labels


def compress_contrast(vol: np.ndarray, mask: np.ndarray, factor: float, n_classes: int = 3,
                      smooth_sigma: float = 1.0) -> np.ndarray:
    """Shrink between-tissue contrast by `factor`, keeping within-tissue texture.

    Voxels are grouped into intensity classes (roughly CSF, grey matter, white matter).
    Each class mean m is moved towards the brain mean mu:  x' = mu + factor*(m - mu) + (x - m).
    This is a proxy for the reduced T1 contrast at low field. It is a heuristic, so
    calibrate `factor` against real low-field scans (scripts/calibrate.py).
    """
    if factor >= 1.0 or not mask.any():
        return vol
    vals = vol[mask]
    centres, labels = kmeans_1d(vals, n_classes)
    class_map = np.zeros_like(vol, dtype=np.float32)
    class_map[mask] = centres[labels]
    if smooth_sigma > 0:  # avoid hard steps at class boundaries
        weight = ndimage.gaussian_filter(mask.astype(np.float32), smooth_sigma)
        smoothed = ndimage.gaussian_filter(class_map, smooth_sigma)
        class_map = np.where(mask, smoothed / np.maximum(weight, 1e-6), 0.0)
    mu = float(vals.mean())
    out = vol + (factor - 1.0) * (class_map - mu)
    return np.where(mask, np.maximum(out, 0.0), 0.0).astype(np.float32)


# ------------------------------------------------------------------ through-plane


def slice_sampling(vol: np.ndarray, thickness_mm: float, gap_mm: float = 0.0, profile: str = "rect"):
    """Integrate over the slice profile and sample one slice per (thickness + gap).

    Input is 1 mm isotropic. Returns the acquired slices and their positions (indices in 1 mm space).
    """
    nz = vol.shape[2]
    if thickness_mm <= 1 and gap_mm <= 0:
        return vol, np.arange(nz)
    if profile == "rect":
        t = max(1, int(round(thickness_mm)))
        smoothed = ndimage.uniform_filter1d(vol, size=t, axis=2, mode="nearest") if t > 1 else vol
    else:
        smoothed = ndimage.gaussian_filter1d(vol, sigma=thickness_mm / 2.3548, axis=2, mode="nearest")
    step = max(1, int(round(thickness_mm + gap_mm)))
    idx = np.arange(step // 2, nz, step)
    return smoothed[..., idx], idx


def resample_to_iso(slices: np.ndarray, idx: np.ndarray, nz: int) -> np.ndarray:
    """Linear interpolation of acquired slices back onto the 1 mm grid (no extrapolation)."""
    if len(idx) == nz and np.array_equal(idx, np.arange(nz)):
        return slices
    if len(idx) == 1:
        return np.repeat(slices, nz, axis=2)
    z = np.clip(np.arange(nz), idx[0], idx[-1])
    return interp1d(idx, slices, axis=2, kind="linear", assume_sorted=True)(z).astype(np.float32)


# ------------------------------------------------------------------ sampling masks


def lowpass_mask(shape, keep_pe: float, keep_ro: float) -> np.ndarray:
    """Keep the central fraction of k-space along each axis (lower acquired resolution)."""
    h, w = shape

    def axis_mask(n, keep):
        m = np.ones(n, dtype=bool)
        if keep < 1.0:
            half = max(1, int(round(n * keep / 2)))
            m[:] = False
            m[n // 2 - half: n // 2 + half] = True
        return m

    return axis_mask(h, keep_pe)[:, None] & axis_mask(w, keep_ro)[None, :]


def undersampling_lines(n: int, accel: int, center_fraction: float, pattern: str,
                        rng: np.random.Generator) -> np.ndarray:
    """Phase-encode lines acquired with acceleration R plus a fully sampled centre."""
    m = np.zeros(n, dtype=bool)
    if accel <= 1:
        m[:] = True
        return m
    c = max(1, int(round(n * center_fraction / 2)))
    m[n // 2 - c: n // 2 + c] = True
    if pattern == "equispaced":
        m[::accel] = True
    else:  # variable-density random sampling, denser near the centre
        target = int(round(n / accel))
        remaining = target - int(m.sum())
        if remaining > 0:
            candidates = np.flatnonzero(~m)
            dist = np.abs(candidates - n // 2) / (n / 2)
            p = (1.0 - dist) ** 2 + 0.05
            chosen = rng.choice(candidates, size=min(remaining, candidates.size), replace=False, p=p / p.sum())
            m[chosen] = True
    return m


def partial_fourier_lines(n: int, fraction: float) -> np.ndarray:
    """Acquire only the last `fraction` of phase-encode lines (asymmetric k-space)."""
    m = np.zeros(n, dtype=bool)
    if fraction >= 1.0:
        m[:] = True
    else:
        m[int(round((1.0 - fraction) * n)):] = True
    return m


def sampling_mask(shape, acq: Acquisition, rng: np.random.Generator) -> np.ndarray:
    """All samples the scanner acquires, as an (H, W) boolean mask."""
    h, _ = shape
    lines = undersampling_lines(h, acq.accel, acq.center_fraction, acq.undersampling, rng)
    lines &= partial_fourier_lines(h, acq.partial_fourier)
    return lowpass_mask(shape, acq.keep_pe, acq.keep_ro) & lines[:, None]


# ------------------------------------------------------------------ motion


def acquisition_order(n: int, ordering: str) -> np.ndarray:
    """Order in which phase-encode lines are acquired."""
    lines = np.arange(n)
    if ordering == "linear":
        return lines
    if ordering == "centric":
        return np.argsort(np.abs(lines - n // 2), kind="stable")
    if ordering == "interleaved":
        return np.concatenate([lines[0::2], lines[1::2]])
    raise ValueError(f"Unknown ordering '{ordering}'")


def motion_lines(n: int, frac: float, ordering: str) -> np.ndarray:
    """Phase-encode lines acquired after a single motion event at time (1 - frac)."""
    k = int(round(frac * n))
    if k <= 0:
        return np.array([], dtype=int)
    return np.sort(acquisition_order(n, ordering)[n - k:])


def draw_motion(rng: np.random.Generator, shift_mm: float, rotation_deg: float):
    """Random direction of an in-plane translation of fixed size, and a rotation of random sign."""
    angle = rng.uniform(0, 2 * np.pi)
    shift = (shift_mm * np.cos(angle), shift_mm * np.sin(angle))
    rotation = rotation_deg * rng.choice([-1.0, 1.0])
    return shift, rotation


def move_object(obj: np.ndarray, shift, rotation_deg: float) -> np.ndarray:
    """Rigid in-plane motion of a complex object (H, W, Z). Pixels are 1 mm."""
    def transform(a):
        a = ndimage.rotate(a, rotation_deg, axes=(0, 1), reshape=False, order=1, mode="constant")
        return ndimage.shift(a, (shift[0], shift[1], 0), order=1, mode="constant")

    return (transform(obj.real) + 1j * transform(obj.imag)).astype(np.complex64)
