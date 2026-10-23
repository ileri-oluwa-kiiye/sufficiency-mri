"""Scan statistics that can be measured on any scan, real or simulated.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .operators import kmeans_1d

FEATURES = ("log_snr", "inplane_hf", "through_plane_ratio", "contrast", "bias_index")


def auto_mask(vol: np.ndarray) -> np.ndarray:
    """Brain mask: non-zero voxels if the scan is skull-stripped, otherwise an Otsu threshold."""
    if (vol == 0).mean() > 0.2:
        return vol > 0
    return vol > threshold_otsu(vol)


def noise_sigma(vol: np.ndarray, mask: np.ndarray) -> float:
    """Noise sd from neighbouring-voxel differences inside an eroded mask (robust to edges)."""
    inner = ndimage.binary_erosion(mask, iterations=2)
    both = inner[1:] & inner[:-1]
    d = np.diff(vol, axis=0)[both]
    if d.size == 0:
        return float("nan")
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def log_snr(vol: np.ndarray, mask: np.ndarray) -> float:
    sigma = noise_sigma(vol, mask)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(np.log10(np.median(vol[mask]) / sigma))


def _hf_fraction_along(x: np.ndarray, axis: int, cutoff: float) -> float:
    power = np.abs(np.fft.fft(x, axis=axis)) ** 2
    power = power.sum(axis=tuple(a for a in range(x.ndim) if a != axis))
    freq = np.abs(np.fft.fftfreq(x.shape[axis])) / 0.5
    total = power.sum()
    return float(power[freq > cutoff].sum() / total) if total > 0 else float("nan")


def inplane_hf(vol: np.ndarray, mask: np.ndarray, cutoff: float = 0.5) -> float:
    """Share of in-plane spectral energy above `cutoff` x Nyquist (lower = blurrier)."""
    x = np.where(mask, vol, 0.0)
    return 0.5 * (_hf_fraction_along(x, 0, cutoff) + _hf_fraction_along(x, 1, cutoff))


def through_plane_ratio(vol: np.ndarray, mask: np.ndarray, cutoff: float = 0.5) -> float:
    """High-frequency energy along z relative to in-plane. Thick slices make it fall below 1."""
    x = np.where(mask, vol, 0.0)
    hz = _hf_fraction_along(x, 2, cutoff)
    hxy = 0.5 * (_hf_fraction_along(x, 0, cutoff) + _hf_fraction_along(x, 1, cutoff))
    return float(hz / hxy) if hxy > 0 else float("nan")


def contrast_index(vol: np.ndarray, mask: np.ndarray) -> float:
    """(brightest - darkest tissue class mean) / brain mean, using 3-class k-means."""
    vals = vol[mask]
    if vals.size < 100:
        return float("nan")
    centres, _ = kmeans_1d(vals, 3)
    return float((centres[-1] - centres[0]) / max(vals.mean(), 1e-6))


def bias_index(vol: np.ndarray, mask: np.ndarray, sigma_mm: float = 12.0) -> float:
    """Coefficient of variation of a heavily smoothed brain image (slow intensity drift)."""
    m = mask.astype(np.float32)
    low = ndimage.gaussian_filter(vol * m, sigma_mm) / np.maximum(ndimage.gaussian_filter(m, sigma_mm), 1e-6)
    inner = ndimage.binary_erosion(mask, iterations=4)
    vals = low[inner]
    return float(vals.std() / max(vals.mean(), 1e-6)) if vals.size else float("nan")


def scan_features(vol: np.ndarray, mask: np.ndarray | None = None) -> dict:
    vol = np.asarray(vol, dtype=np.float32)
    mask = auto_mask(vol) if mask is None else mask
    return {
        "log_snr": log_snr(vol, mask),
        "inplane_hf": inplane_hf(vol, mask),
        "through_plane_ratio": through_plane_ratio(vol, mask),
        "contrast": contrast_index(vol, mask),
        "bias_index": bias_index(vol, mask),
    }
