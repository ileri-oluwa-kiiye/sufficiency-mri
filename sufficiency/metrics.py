from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """PSNR inside the brain mask, with the reference maximum as data range."""
    diff = (ref - test)[mask]
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(ref[mask].max())
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """3D SSIM over the whole volume."""
    data_range = float(ref.max() - ref.min())
    return float(structural_similarity(ref, test, data_range=data_range))
