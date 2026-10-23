"""Tests for the physics operators and the simulator, on small synthetic phantoms.

Run with: python -m pytest -q
"""

import numpy as np
import pytest

from sufficiency.acquisition import Acquisition, apply_overrides
from sufficiency.coils import coil_combine, coil_maps, sense_cg
from sufficiency.kspace import fft2c, ifft2c
from sufficiency.operators import (
    compress_contrast,
    motion_lines,
    partial_fourier_lines,
    resample_to_iso,
    slice_sampling,
    undersampling_lines,
)
from sufficiency.simulator import simulate_case, simulate_volume


def phantom(shape=(64, 64, 12)):
    """Brain-like phantom: background 0, 'white matter' 100, 'grey matter' ring 70, a bright 'lesion' 160."""
    h, w, _ = shape
    yy, xx = np.meshgrid(np.arange(h) - h / 2, np.arange(w) - w / 2, indexing="ij")
    r = np.sqrt(yy ** 2 + xx ** 2)
    sl = np.zeros((h, w), np.float32)
    sl[r < 26] = 70.0
    sl[r < 20] = 100.0
    sl[(yy - 5) ** 2 + (xx + 6) ** 2 < 25] = 160.0
    return np.repeat(sl[:, :, None], shape[2], axis=2)


CLEAN = dict(snr=None, coils=1, phase_amplitude=0.0)


# ------------------------------------------------------------------ k-space and coils

def test_fft_roundtrip():
    x = np.random.default_rng(0).standard_normal((32, 32, 3)).astype(np.complex64)
    assert np.allclose(ifft2c(fft2c(x)), x, atol=1e-5)


def test_coil_maps_are_normalised():
    s = coil_maps((48, 48), 8, np.random.default_rng(0))
    assert s.shape == (8, 48, 48)
    assert np.allclose(np.sum(np.abs(s) ** 2, axis=0), 1.0, atol=1e-5)


def test_coil_combination_is_exact_when_fully_sampled():
    x = phantom()[:, :, :2].astype(np.complex64)
    s = coil_maps(x.shape[:2], 8, np.random.default_rng(1))[:, :, :, None]
    combined = coil_combine(ifft2c(fft2c(s * x[None], axes=(1, 2)), axes=(1, 2)), s)
    assert np.allclose(combined, x, atol=1e-3)


def test_sense_unfolds_aliasing():
    x = phantom()[:, :, :2].astype(np.complex64)
    s = coil_maps(x.shape[:2], 8, np.random.default_rng(2))[:, :, :, None]
    lines = undersampling_lines(64, 2, 0.08, "equispaced", np.random.default_rng(0))
    m = lines[None, :, None, None]
    k = m * fft2c(s * x[None], axes=(1, 2))
    zero_filled = coil_combine(ifft2c(k, axes=(1, 2)), s)
    sense = sense_cg(k, m, s, iters=30)
    err = lambda y: np.linalg.norm(np.abs(y) - np.abs(x)) / np.linalg.norm(np.abs(x))
    assert err(sense) < 0.05
    assert err(sense) < 0.5 * err(zero_filled)


# ------------------------------------------------------------------ masks and operators

def test_undersampling_mask():
    m = undersampling_lines(240, 4, 0.08, "equispaced", np.random.default_rng(0))
    assert m[110:130].all()
    assert 0.28 < m.mean() < 0.35
    r = undersampling_lines(240, 4, 0.08, "random", np.random.default_rng(0))
    assert r[110:130].all()
    assert abs(r.sum() - 60) <= 1


def test_partial_fourier_mask():
    m = partial_fourier_lines(240, 0.625)
    assert m.sum() == 150 and m[-1] and not m[0]


def test_motion_lines():
    assert motion_lines(100, 0.0, "linear").size == 0
    assert np.array_equal(motion_lines(100, 0.2, "linear"), np.arange(80, 100))
    centric = motion_lines(100, 0.2, "centric")
    assert centric.size == 20 and 50 not in centric  # centre acquired first


def test_slice_sampling_with_gap():
    v = np.random.default_rng(0).random((16, 16, 40)).astype(np.float32)
    thick, idx = slice_sampling(v, 5, gap_mm=1)
    assert np.array_equal(idx, np.arange(3, 40, 6))
    assert resample_to_iso(thick, idx, 40).shape == v.shape


def test_contrast_compression():
    v = phantom()
    mask = v > 0
    assert np.array_equal(compress_contrast(v, mask, 1.0), v)
    c = compress_contrast(v, mask, 0.5, smooth_sigma=0)
    gap = lambda a: a[32, 32, 0] - a[32, 8, 0]  # white matter minus grey matter
    assert abs(gap(c) - 0.5 * gap(v)) < 1.0


# ------------------------------------------------------------------ simulator

def test_clean_acquisition_is_identity():
    v = phantom()
    out = simulate_volume(v, Acquisition(**CLEAN), np.random.default_rng(0))
    assert np.allclose(out, v, atol=1e-2)


def test_multicoil_with_phase_is_identity_without_noise():
    v = phantom()
    out = simulate_volume(v, Acquisition(snr=None, coils=8, phase_amplitude=1.5), np.random.default_rng(0))
    assert np.allclose(out, v, atol=1e-2)


@pytest.mark.parametrize("coils", [1, 8])
def test_noise_matches_target_snr(coils):
    v = np.zeros((64, 64, 12), np.float32)
    v[8:56, 8:56, :] = 100.0
    out = simulate_volume(v, Acquisition(snr=20, coils=coils, phase_amplitude=0.0), np.random.default_rng(0))
    inner = out[16:48, 16:48, :]
    assert abs(inner.std() - 5.0) / 5.0 < 0.1
    assert abs(inner.mean() - 100.0) < 1.0


def test_sense_beats_zero_filled_in_simulator():
    v = phantom()
    psnr_like = lambda a: -np.mean((a - v) ** 2)
    base = dict(snr=None, coils=8, accel=3, phase_amplitude=1.0)
    sense = simulate_volume(v, Acquisition(**base, recon="sense", sense_iters=30), np.random.default_rng(0))
    zf = simulate_volume(v, Acquisition(**base, recon="zero_filled"), np.random.default_rng(0))
    assert psnr_like(sense) > psnr_like(zf)


@pytest.mark.parametrize("spec", [
    {"slice_mm": 5}, {"inplane_keep": 0.5}, {"accel": 4}, {"partial_fourier": 0.625},
    {"motion_frac": 0.3}, {"contrast": 0.5}, {"bias": 0.3}, {"snr": 5},
])
def test_every_axis_degrades_and_keeps_shape(spec):
    v = phantom((64, 64, 20))
    v[:, :, ::4] *= 0.6  # structure along z, so thick slices have something to blur
    acq, _ = apply_overrides(Acquisition(snr=None, coils=8, recon="zero_filled"), spec)
    out = simulate_volume(v, acq, np.random.default_rng(0))
    assert out.shape == v.shape and out.dtype == np.float32
    assert not np.allclose(out, v, atol=0.5)
    assert np.all(out[v == 0] == 0)  # brain mask re-applied


def test_simulate_case_drops_and_is_reproducible():
    vols = {s: phantom() * f for s, f in zip(("t1n", "t1c", "t2w", "t2f"), (1.0, 1.2, 0.8, 0.9))}
    acq = Acquisition(snr=10, coils=4)
    a = simulate_case(vols, acq, np.random.default_rng(5), missing=1)
    b = simulate_case(vols, acq, np.random.default_rng(5), missing=1)
    assert not a["t1c"].any() and a["t1n"].any()
    assert all(np.array_equal(a[s], b[s]) for s in vols)


def test_invalid_acquisition_is_rejected():
    with pytest.raises(ValueError):
        Acquisition(contrast=1.5).validate()
    with pytest.raises(ValueError):
        apply_overrides(Acquisition(), {"not_a_field": 1})
