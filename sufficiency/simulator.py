"""The acquisition simulator.

Per sequence, in physical order:

  object                tissue contrast change (low-field proxy) -> receive bias field
  through-plane         slice-profile integration, one slice per (thickness + gap)
  complex object        smooth background phase
  receive               multi-coil sensitivities
  k-space               2D Fourier transform per slice
  motion                lines acquired after a motion event come from the moved head
  noise                 complex Gaussian thermal noise on every acquired sample
  sampling              lower resolution, undersampling, partial Fourier
  reconstruction        CG-SENSE (accelerated) or matched-filter coil combination
  output                magnitude -> back to 1 mm -> original brain mask re-applied

Inputs are skull-stripped BraTS volumes at 1 mm isotropic. Phase-encode is image axis 0.
"""

from __future__ import annotations

import numpy as np

from .acquisition import Acquisition
from .coils import coil_combine, coil_maps, sense_cg
from .fields import bias_field, phase_map
from .kspace import fft2c, ifft2c
from .operators import (
    compress_contrast,
    draw_motion,
    motion_lines,
    move_object,
    resample_to_iso,
    sampling_mask,
    slice_sampling,
)

SEQUENCES = ("t1n", "t1c", "t2w", "t2f")  # BraTS 2023: T1, T1 post-contrast, T2, FLAIR
DEFAULT_MISSING_ORDER = ("t1c", "t2f", "t2w")  # contrast agent is the first thing to go


def brain_mask_from(vols: dict) -> np.ndarray:
    """Union of non-zero voxels across sequences (BraTS volumes are skull-stripped)."""
    first = next(iter(vols.values()))
    mask = np.zeros(first.shape, dtype=bool)
    for v in vols.values():
        mask |= v > 0
    return mask


def simulate_volume(
    vol: np.ndarray,
    acq: Acquisition,
    rng: np.random.Generator,
    brain_mask: np.ndarray | None = None,
    signal_ref: float | None = None,
    coil_sens: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    chunk_slices: int = 16,
) -> np.ndarray:
    """Simulate acquiring one 3D volume (H, W, Z) under `acq`. Returns float32, same shape.

    coil_sens and bias can be passed in so that all sequences of a case share them.
    """
    acq.validate()
    vol = np.asarray(vol, dtype=np.float32)
    h, w, nz = vol.shape
    mask = (vol > 0) if brain_mask is None else brain_mask
    if signal_ref is None:
        signal_ref = float(vol[mask].mean()) if mask.any() else 1.0

    # object
    x = vol
    if acq.contrast < 1.0:
        x = compress_contrast(x, mask, acq.contrast)
    if acq.bias > 0:
        x = x * (bias if bias is not None else bias_field(x.shape, rng, acq.bias))

    # through-plane acquisition
    thick, idx = slice_sampling(x, acq.slice_mm, acq.slice_gap_mm, acq.slice_profile)

    # complex object with smooth background phase
    obj = thick.astype(np.complex64)
    if acq.phase_amplitude > 0:
        obj = obj * np.exp(1j * phase_map((h, w), rng, acq.phase_amplitude))[:, :, None].astype(np.complex64)

    sens = coil_maps((h, w), acq.coils, rng) if coil_sens is None else coil_sens
    sens4 = sens[:, :, :, None]
    mask4 = sampling_mask((h, w), acq, rng)[None, :, :, None]

    lines = motion_lines(h, acq.motion_frac, acq.motion_ordering)
    if lines.size:
        shift, rotation = draw_motion(rng, acq.motion_shift_mm, acq.motion_rotation_deg)
    sigma = None if acq.snr is None else np.float32(signal_ref / acq.snr)
    use_sense = acq.recon == "sense" and acq.accel > 1 and sens.shape[0] > 1

    out = np.empty(thick.shape, dtype=np.float32)
    for z0 in range(0, thick.shape[2], chunk_slices):
        o = obj[:, :, z0:z0 + chunk_slices]
        k = fft2c(sens4 * o[None], axes=(1, 2))
        if lines.size:
            k_moved = fft2c(sens4 * move_object(o, shift, rotation)[None], axes=(1, 2))
            k[:, lines] = k_moved[:, lines]
        if sigma is not None:
            noise = rng.standard_normal(k.shape, dtype=np.float32) + 1j * rng.standard_normal(k.shape, dtype=np.float32)
            k += (sigma * noise).astype(np.complex64)
        k *= mask4
        if use_sense:
            rec = sense_cg(k, mask4, sens4, iters=acq.sense_iters)
        else:
            rec = coil_combine(ifft2c(k, axes=(1, 2)), sens4)
        out[:, :, z0:z0 + chunk_slices] = np.abs(rec)

    img = resample_to_iso(out, idx, nz)
    return (img * mask).astype(np.float32)


def simulate_case(
    vols: dict,
    acq: Acquisition,
    rng: np.random.Generator,
    missing: int = 0,
    missing_order=DEFAULT_MISSING_ORDER,
    chunk_slices: int = 16,
) -> dict:
    """Simulate all sequences of a case.

    Coil sensitivities and the bias field are shared across sequences (same session);
    background phase, noise and motion are drawn independently per sequence.
    The first `missing` sequences in `missing_order` are returned as zeros.
    """
    mask = brain_mask_from(vols)
    shape = next(iter(vols.values())).shape
    sens = coil_maps(shape[:2], acq.coils, rng)
    bias = bias_field(shape, rng, acq.bias) if acq.bias > 0 else None
    dropped = set(list(missing_order)[:missing])
    out = {}
    names = [s for s in SEQUENCES if s in vols] + [s for s in vols if s not in SEQUENCES]
    for name in names:
        child = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))  # drawn even if dropped
        if name in dropped:
            out[name] = np.zeros(shape, dtype=np.float32)
        else:
            out[name] = simulate_volume(vols[name], acq, child, brain_mask=mask, coil_sens=sens,
                                        bias=bias, chunk_slices=chunk_slices)
    return out
