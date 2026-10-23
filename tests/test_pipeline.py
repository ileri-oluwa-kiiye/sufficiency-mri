"""Tests for the lattice, data I/O, splits, nnU-Net export and scan statistics."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from sufficiency.acquisition import Acquisition
from sufficiency.data import find_cases, is_complete, load_case, make_splits, patient_id, save_case
from sufficiency.lattice import build_lattice, condition_rng, random_condition
from sufficiency.nnunet import brats2023_to_nnunet_labels, export_condition
from sufficiency.simulator import simulate_volume
from sufficiency.stats import scan_features

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lattice.yaml"


def load_cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def write_fake_case(root: Path, case_id: str, shape=(32, 32, 16)):
    d = root / case_id
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for seq in ("t1n", "t1c", "t2w", "t2f"):
        v = np.zeros(shape, np.float32)
        v[4:28, 4:28, 2:14] = 100 + 5 * rng.standard_normal((24, 24, 12))
        nib.save(nib.Nifti1Image(v, np.eye(4)), str(d / f"{case_id}-{seq}.nii.gz"))
    seg = np.zeros(shape, np.uint8)
    seg[10:14, 10:14, 6:9] = 1
    seg[14:18, 10:14, 6:9] = 2
    seg[10:14, 14:18, 6:9] = 3
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(d / f"{case_id}-seg.nii.gz"))
    return d


def test_lattice_has_no_duplicates_and_valid_conditions():
    cfg = load_cfg()
    conds = build_lattice(cfg)
    names = [c.name for c in conds]
    assert len(names) == len(set(names))
    assert len({(c.acq, c.missing) for c in conds}) == len(conds)
    assert names[:2] == ["original", "reference"]
    assert "snr=40" not in names  # identical to the reference, removed
    assert any(c.kind == "profile" for c in conds)


def test_condition_rng_is_order_independent():
    a = condition_rng(0, "BraTS-GLI-00001-000", "snr=10").random()
    b = condition_rng(0, "BraTS-GLI-00001-000", "snr=10").random()
    c = condition_rng(0, "BraTS-GLI-00002-000", "snr=10").random()
    assert a == b and a != c


def test_random_condition_is_valid():
    cfg = load_cfg()
    for i in range(20):
        cond = random_condition(np.random.default_rng(i), cfg)
        cond.acq.validate()


def test_patient_level_splits():
    ids = [f"BraTS-GLI-{i:05d}-{t:03d}" for i in range(50) for t in (0, 1)]
    splits = make_splits(ids, {"train": 0.7, "calibration": 0.1, "test": 0.2}, seed=0)
    assert sum(len(v) for v in splits.values()) == len(ids)
    owner = {patient_id(c): s for s, cs in splits.items() for c in cs}
    for s, cs in splits.items():
        assert all(owner[patient_id(c)] == s for c in cs)
    assert patient_id("BraTS-GLI-00007-001") == "BraTS-GLI-00007"


def test_label_conversion():
    seg = np.array([0, 1, 2, 3], np.uint8)
    assert brats2023_to_nnunet_labels(seg).tolist() == [0, 2, 1, 3]


def test_save_load_and_export_roundtrip(tmp_path):
    src = tmp_path / "data"
    write_fake_case(src, "BraTS-GLI-00001-000")
    write_fake_case(src, "BraTS-GLI-00002-000")
    cases = find_cases(src)
    assert len(cases) == 2

    out = tmp_path / "lattice" / "snr=10"
    for c in cases:
        case = load_case(c)
        vols = {s: simulate_volume(v, Acquisition(snr=10, coils=2), np.random.default_rng(0))
                for s, v in case["vols"].items()}
        save_case(out, case, vols, case["seg"], dtype="int16", sidecar={"case": case["id"]})
        assert is_complete(out, case["id"])
        reloaded = load_case(out / case["id"])
        assert np.allclose(reloaded["vols"]["t1n"], vols["t1n"], atol=0.1)  # int16 scaling is near-lossless

    root = export_condition(out, tmp_path / "nnUNet_raw", 501, "Suff_snr=10",
                            train_ids=["BraTS-GLI-00001-000"], test_ids=["BraTS-GLI-00002-000"])
    assert root.name == "Dataset501_Suff_snr_10"
    assert (root / "imagesTr" / "BraTS-GLI-00001-000_0003.nii.gz").exists()
    assert (root / "labelsTs" / "BraTS-GLI-00002-000.nii.gz").exists()
    meta = json.loads((root / "dataset.json").read_text())
    assert meta["numTraining"] == 1 and meta["regions_class_order"] == [1, 2, 3]


def test_scan_features_track_degradation():
    v = np.zeros((64, 64, 32), np.float32)
    yy, xx = np.meshgrid(np.arange(64) - 32, np.arange(64) - 32, indexing="ij")
    v[(yy ** 2 + xx ** 2) < 26 ** 2] = 70
    v[(yy ** 2 + xx ** 2) < 18 ** 2] = 100
    mask = v > 0
    clean = scan_features(simulate_volume(v, Acquisition(snr=40, coils=1, phase_amplitude=0), np.random.default_rng(0)), mask)
    noisy = scan_features(simulate_volume(v, Acquisition(snr=5, coils=1, phase_amplitude=0), np.random.default_rng(0)), mask)
    thick = scan_features(simulate_volume(v + np.random.default_rng(1).normal(0, 5, v.shape).astype(np.float32) * mask,
                                          Acquisition(snr=None, coils=1, phase_amplitude=0, slice_mm=5),
                                          np.random.default_rng(0)), mask)
    low_contrast = scan_features(simulate_volume(v, Acquisition(snr=40, coils=1, phase_amplitude=0, contrast=0.4),
                                                 np.random.default_rng(0)), mask)
    assert noisy["log_snr"] < clean["log_snr"]
    assert thick["through_plane_ratio"] < 0.8
    assert low_contrast["contrast"] < clean["contrast"]
