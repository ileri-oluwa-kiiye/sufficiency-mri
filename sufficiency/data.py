from __future__ import annotations

import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from .simulator import SEQUENCES

SIDECAR_SUFFIX = "-acquisition.json"


def find_cases(root) -> list[Path]:
    """All case folders under `root` (searched recursively)."""
    return sorted({p.parent for p in Path(root).rglob("*-t1n.nii.gz")})


def _find(case_dir: Path, suffix: str):
    hits = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    return hits[0] if hits else None


def load_volume(path):
    """Load any NIfTI volume as float32. Returns (array, affine, header)."""
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32)), img.affine, img.header


def load_case(case_dir) -> dict:
    """Load the four sequences (float32) and the segmentation (uint8, or None)."""
    case_dir = Path(case_dir)
    vols, ref = {}, None
    for seq in SEQUENCES:
        path = _find(case_dir, seq)
        if path is None:
            raise FileNotFoundError(f"No *-{seq}.nii.gz in {case_dir}")
        img = nib.load(str(path))
        if ref is None:
            ref = img
            zooms = img.header.get_zooms()[:3]
            if not np.allclose(zooms, 1.0, atol=0.05):
                warnings.warn(f"{case_dir.name}: voxel size {zooms}; the simulator assumes 1 mm isotropic.")
        vols[seq] = np.asarray(img.get_fdata(dtype=np.float32))
    seg_path = _find(case_dir, "seg")
    seg = np.asarray(nib.load(str(seg_path)).dataobj).astype(np.uint8) if seg_path else None
    return {"id": case_dir.name, "vols": vols, "seg": seg, "affine": ref.affine, "header": ref.header}


def save_case(out_root, case: dict, vols: dict, seg=None, dtype: str = "float32", sidecar: dict | None = None) -> Path:
    """Write sequences (and the unchanged label map) with BraTS names.

    dtype="int16" stores scaled integers (about half the disk space); nnU-Net normalises
    intensities anyway. The sidecar JSON is written last and marks the case as complete.
    """
    out_dir = Path(out_root) / case["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_dtype = np.int16 if dtype == "int16" else np.float32
    for seq, arr in vols.items():
        hdr = case["header"].copy()
        hdr.set_data_dtype(disk_dtype)
        nib.save(nib.Nifti1Image(arr.astype(np.float32), case["affine"], hdr),
                 str(out_dir / f"{case['id']}-{seq}.nii.gz"))
    if seg is not None:
        hdr = case["header"].copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), case["affine"], hdr), str(out_dir / f"{case['id']}-seg.nii.gz"))
    if sidecar is not None:
        with open(out_dir / f"{case['id']}{SIDECAR_SUFFIX}", "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
    return out_dir


def is_complete(out_root, case_id: str) -> bool:
    return (Path(out_root) / case_id / f"{case_id}{SIDECAR_SUFFIX}").exists()


# ------------------------------------------------------------------ splits


def patient_id(case_id: str) -> str:
    """BraTS-GLI-00000-000 -> BraTS-GLI-00000 (the last block is the timepoint)."""
    parts = case_id.split("-")
    if len(parts) >= 4 and parts[-1].isdigit():
        return "-".join(parts[:-1])
    return case_id


def make_splits(case_ids, fractions: dict, seed: int = 0) -> dict:
    """Split cases by patient, so timepoints of one patient never cross splits."""
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    patients = sorted({patient_id(c) for c in case_ids})
    order = np.random.default_rng(seed).permutation(len(patients))
    patients = [patients[i] for i in order]
    names = list(fractions)
    bounds = np.round(np.cumsum([fractions[n] for n in names]) * len(patients)).astype(int)
    assignment, start = {}, 0
    for name, end in zip(names, bounds):
        for p in patients[start:end]:
            assignment[p] = name
        start = end
    splits = {n: [] for n in names}
    for c in sorted(case_ids):
        splits[assignment[patient_id(c)]].append(c)
    return splits
