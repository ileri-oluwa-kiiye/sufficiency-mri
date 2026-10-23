# sufficiency-mri

A physics-based MRI acquisition simulator. It takes clean 1 mm brain MRI (BraTS format) and re-acquires it in simulated k-space under controlled scanner settings: noise, slice thickness, acceleration, resolution, motion, contrast and missing sequences. Use it to measure how much acquisition a segmentation model needs before its output can be trusted.

## How it works

For each sequence, the simulator follows the scanner's physics:

```
1 mm image → contrast / bias → slice profile → background phase → multi-coil sensitivities
→ 2-D FFT → motion → thermal noise in k-space → sampling mask → CG-SENSE reconstruction
→ magnitude → back to 1 mm
```

Noise is added per coil in k-space, so images have Rician noise and accelerated scans show realistic g-factor noise. Every scan is deterministic: one seed per case and condition.

## Acquisition axes


| Axis              | Levels                           |
| ----------------- | -------------------------------- |
| `snr`             | 40, 20, 10, 5, 3                 |
| `slice_mm`        | 1, 2, 3, 4, 5, 6                 |
| `accel`           | 1, 2, 3, 4, 6, 8                 |
| `motion_frac`     | 0 – 0.4                          |
| `missing`         | 0 – 3 (T1c, then FLAIR, then T2) |
| `inplane_keep`    | 1 – 0.25                         |
| `partial_fourier` | 1 – 0.625                        |
| `contrast`        | 1 – 0.4                          |
| `bias`            | 0 – 0.4                          |


Combined profiles (e.g. 64 mT portable, older 1.5 T, typical low-resource protocol) are defined in `configs/lattice.yaml`.

## Install

```bash
git clone https://github.com/ileri-oluwa-kiiye/sufficiency-mri.git
cd sufficiency-mri
pip install -e ".[dev]"
python -m pytest -q
```



## Data

Download [BraTS 2023](https://www.synapse.org/brats2023) (GLI and, optionally, BraTS-Africa) and place cases as:

```
data/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-{t1n,t1c,t2w,t2f,seg}.nii.gz
```

No patient data is included in this repository.

## Usage

```bash
# visualise one case
python scripts/figure_gallery.py --case_dir data/BraTS2023-GLI/BraTS-GLI-00000-000 --seq t1c
python scripts/plot_ladder.py    --case_dir data/BraTS2023-GLI/BraTS-GLI-00000-000 --axis accel

# patient-level splits
python scripts/make_splits.py --data_root data/BraTS2023-GLI --out splits/brats2023_gli.json

# generate the lattice (parallel, resumable)
python scripts/make_lattice.py --data_root data/BraTS2023-GLI --splits splits/brats2023_gli.json \
    --split test --workers 8 --out_dir outputs/lattice

# export to nnU-Net v2
python scripts/export_nnunet.py --lattice_dir outputs/lattice --conditions all \
    --splits splits/brats2023_gli.json --nnunet_raw $nnUNet_raw

# place real low-resource scans on the lattice
python scripts/scan_stats.py --data_root outputs/lattice   --out outputs/stats_lattice.csv
python scripts/scan_stats.py --data_root data/BraTS-Africa --out outputs/stats_africa.csv
python scripts/place_on_lattice.py --real outputs/stats_africa.csv --lattice outputs/stats_lattice.csv

# calibrate profiles against a real low-field scan
python scripts/calibrate.py --source 3T.nii.gz --target 64mT_1mm.nii.gz
```

Or use it directly in Python:

```python
from sufficiency import Acquisition, simulate_volume
import numpy as np

degraded = simulate_volume(volume, Acquisition(snr=10, slice_mm=5, accel=4), np.random.default_rng(0))
```



## License

MIT