from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .simulator import SEQUENCES

CHANNEL_NAMES = {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}  # order of SEQUENCES


def brats2023_to_nnunet_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS 2023 (1 necrosis, 2 oedema, 3 enhancing) -> nnU-Net convention (1 oedema, 2 necrosis, 3 enhancing).

    With regions whole tumour = {1,2,3}, tumour core = {2,3}, enhancing tumour = {3}.
    """
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 2] = 1
    out[seg == 1] = 2
    out[seg == 3] = 3
    return out


def dataset_json(n_training: int) -> dict:
    return {
        "channel_names": CHANNEL_NAMES,
        "labels": {"background": 0, "whole tumor": [1, 2, 3], "tumor core": [2, 3], "enhancing tumor": [3]},
        "regions_class_order": [1, 2, 3],
        "numTraining": n_training,
        "file_ending": ".nii.gz",
    }


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _write_case(case_dir: Path, case_id: str, images_dir: Path, labels_dir: Path | None):
    for i, seq in enumerate(SEQUENCES):
        shutil.copy2(case_dir / f"{case_id}-{seq}.nii.gz", images_dir / f"{case_id}_{i:04d}.nii.gz")
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    if labels_dir is not None and seg_path.exists():
        img = nib.load(str(seg_path))
        seg = brats2023_to_nnunet_labels(np.asarray(img.dataobj))
        nib.save(nib.Nifti1Image(seg, img.affine, img.header), str(labels_dir / f"{case_id}.nii.gz"))


def export_condition(condition_dir, nnunet_raw, dataset_id: int, dataset_name: str,
                     train_ids=(), test_ids=()) -> Path:
    """Write DatasetXXX_Name/{imagesTr,labelsTr,imagesTs,labelsTs,dataset.json}."""
    condition_dir = Path(condition_dir)
    root = Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{safe_name(dataset_name)}"
    dirs = {n: root / n for n in ("imagesTr", "labelsTr", "imagesTs", "labelsTs")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_train = 0
    for case_id in train_ids:
        if (condition_dir / case_id).exists():
            _write_case(condition_dir / case_id, case_id, dirs["imagesTr"], dirs["labelsTr"])
            n_train += 1
    for case_id in test_ids:
        if (condition_dir / case_id).exists():
            _write_case(condition_dir / case_id, case_id, dirs["imagesTs"], dirs["labelsTs"])

    with open(root / "dataset.json", "w") as f:
        json.dump(dataset_json(n_train), f, indent=2)
    return root
