from __future__ import annotations

import itertools

import numpy as np

from .acquisition import Acquisition, apply_overrides
from .simulator import simulate_volume
from .stats import auto_mask, scan_features

DEFAULT_CAL_FEATURES = ("log_snr", "inplane_hf", "through_plane_ratio", "contrast")


def central_slab(vol: np.ndarray, mask: np.ndarray, slab: int):
    """The `slab` axial slices centred on the brain's centre of mass along z."""
    if vol.shape[2] <= slab:
        return vol, mask
    zs = np.where(mask.any(axis=(0, 1)))[0]
    centre = int(zs.mean()) if zs.size else vol.shape[2] // 2
    lo = int(np.clip(centre - slab // 2, 0, vol.shape[2] - slab))
    return vol[:, :, lo:lo + slab], mask[:, :, lo:lo + slab]


def calibrate(source: np.ndarray, target: np.ndarray, grid: dict, base: Acquisition,
              features=DEFAULT_CAL_FEATURES, seed: int = 0, source_mask=None, target_mask=None,
              progress=None, slab: int | None = 48) -> tuple[dict, list[dict]]:
    """Return (target features, results sorted by distance). Each result holds params, features, distance.

    `slab` simulates only that many central brain slices of the source (faster; the statistics
    are global, so a thick central slab is enough). None uses the whole volume.
    """
    smask = auto_mask(source) if source_mask is None else source_mask
    tmask = auto_mask(target) if target_mask is None else target_mask
    if slab:  # source and target are both cut to their central brain slab, so statistics stay comparable
        source, smask = central_slab(source, smask, slab)
        target, tmask = central_slab(target, tmask, slab)
    target_f = scan_features(target, tmask)

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    iterator = progress(combos) if progress else combos
    results = []
    for combo in iterator:
        params = dict(zip(keys, combo))
        acq, _ = apply_overrides(base, params)
        sim = simulate_volume(source, acq, np.random.default_rng(seed), brain_mask=smask)
        results.append({"params": params, "features": scan_features(sim, smask)})

    # standardise each feature by its spread across the grid
    scales = {}
    for f in features:
        vals = np.array([r["features"][f] for r in results], dtype=float)
        vals = vals[np.isfinite(vals)]
        scales[f] = float(vals.std()) if vals.size > 1 and vals.std() > 0 else 1.0

    for r in results:
        d = [((r["features"][f] - target_f[f]) / scales[f]) ** 2 for f in features
             if np.isfinite(r["features"][f]) and np.isfinite(target_f[f])]
        r["distance"] = float(np.sqrt(np.sum(d))) if d else float("inf")
    results.sort(key=lambda r: r["distance"])
    return target_f, results
