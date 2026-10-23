# sufficiency-mri

A physics-parameterised MRI acquisition simulator for studying **diagnostic sufficiency**: how much acquisition (SNR, resolution, slice thickness, acceleration, motion, contrast, available sequences) a segmentation model needs before its output is trustworthy.

It takes 1 mm high-field brain MRI (BraTS 2023 format), re-acquires it in simulated k-space under controlled, documented acquisition parameters, and writes an **acquisition lattice**: the same patients scanned under dozens of conditions, each with a JSON record of every parameter. The lattice exports straight to nnU-Net v2, and real low-resource scans can be placed on it by matching image statistics.

Built for the MICCAI 2027 study *Diagnostic Sufficiency: task-usable information as a function of the acquisition budget* (Ilerioluwakiiye Abolade).

---

## What the simulator does

Each sequence of each case goes through the same forward model, slice by slice in k-space:

```
1 mm image
 → contrast compression (low-field T1/T2 proxy)        contrast
 → B1 / receive bias field                             bias
 → slice profile + gap, sampled every (thickness+gap)  slice_mm, slice_gap_mm, slice_profile
 → smooth background phase                             phase_amplitude
 → multi-coil sensitivities (birdcage ring)            coils
 → 2-D FFT per slice (orthonormal)
 → rigid in-plane motion for late phase-encode lines   motion_frac, motion_shift_mm, motion_rotation_deg, motion_ordering
 → complex Gaussian thermal noise, σ = signal / SNR    snr
 → k-space sampling mask
     low-pass truncation (resolution)                  keep_pe, keep_ro  (axis: inplane_keep)
     undersampling with fully sampled centre           accel, center_fraction, undersampling
     partial Fourier (zero-filled)                     partial_fourier
 → reconstruction: CG-SENSE or matched-filter combine  recon, sense_iters
 → magnitude (Rician statistics arise naturally)
 → interpolation back to the 1 mm grid, brain mask re-applied
```

Design points that matter for the paper:

* **Noise is added in k-space, per coil, before reconstruction**, so magnitude images have Rician statistics and SENSE shows real g-factor noise amplification. The noise level is defined relative to the mean brain signal, so `snr=10` means the same thing for every case.
* **The reference acquisition is not noise-free** (`snr: 40`, 8 coils, SENSE). Otherwise accelerated conditions would look artificially perfect, because SENSE recovers noiseless data exactly. A separate `original` condition is the untouched input.
* **Thick slices are acquired, not blurred**: the volume is averaged over the slice profile, sampled every (thickness + gap) mm, then interpolated back. Information between slices is genuinely lost.
* **Deterministic**: every (case, condition) pair has its own seed derived from the global seed, the case ID and the condition name, so any single scan can be regenerated alone, in any order, on any machine.
* **Coil maps and bias field are shared** across the four sequences of a case (same patient, same session).

## Acquisition axes (configs/lattice.yaml)

| Axis | Levels | What it means physically |
|---|---|---|
| `snr` | 40, 20, 10, 5, 3 | thermal noise; field strength, voxel volume, scan time |
| `slice_mm` | 1, 2, 3, 4, 5, 6 | 2-D multi-slice thickness (clinical 2-D protocols are 4 to 6 mm) |
| `accel` | 1, 2, 3, 4, 6, 8 | parallel-imaging acceleration, CG-SENSE reconstruction |
| `motion_frac` | 0 to 0.4 | fraction of phase-encode lines acquired after a rigid head movement |
| `missing` | 0 to 3 | sequences not acquired: T1c first (no contrast agent), then FLAIR, then T2 |
| `inplane_keep` | 1 to 0.25 | fraction of k-space kept per in-plane direction (0.5 ≈ 2 mm pixels) |
| `partial_fourier` | 1 to 0.625 | asymmetric k-space, zero-filled |
| `contrast` | 1 to 0.4 | tissue-contrast compression, a low-field proxy (calibrate before use) |
| `bias` | 0 to 0.4 | strength of the smooth intensity inhomogeneity |

A **ladder** changes one axis from the reference. A **profile** changes several at once to resemble a real setting (64 mT portable, older 1.5 T, typical low-resource protocol, worst case). The profile values in the config are starting points; replace them with calibrated values (below) before the final experiments.

The full default lattice is **47 conditions** (`original`, `reference`, 37 ladder levels, 8 profiles). Run `python scripts/make_lattice.py --data_root <data> --dry_run` to list them.

## Install

```bash
git clone https://github.com/<you>/sufficiency-mri.git
cd sufficiency-mri
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q          # 31 tests, about 2 seconds, synthetic phantoms only
```

Python 3.10 or newer. No GPU is needed for simulation.

## Data

| Dataset | Use | Layout expected |
|---|---|---|
| BraTS 2023 GLI (training set, 1251 cases) | train / calibration / test lattice | `data/BraTS2023-GLI/BraTS-GLI-00000-000/BraTS-GLI-00000-000-{t1n,t1c,t2w,t2f,seg}.nii.gz` |
| BraTS-Africa (BraTS 2023 SSA) | external test set of real low-resource scans | same naming with `BraTS-SSA-…` |
| ULF-EnC or another paired 64 mT / 3 T set | calibrating the profiles | any NIfTI pair of the same contrast |

BraTS data are skull-stripped, co-registered and 1 mm isotropic (240 × 240 × 155), which is what the simulator assumes. Download them from Synapse after accepting the data use agreement. Nothing in this repository contains patient data.

## MICCAI workflow

### 1. Patient-level splits

```bash
python scripts/make_splits.py --data_root data/BraTS2023-GLI --out splits/brats2023_gli.json
python scripts/make_splits.py --data_root data/BraTS-Africa  --out splits/brats_africa.json --all_as external
```

Follow-up scans of the same patient (`-000`, `-001`) stay in the same split. Default 70 / 10 / 20 (train / calibration / test). The calibration split is for the conformal flag, not for the simulator.

### 2. Calibrate the profiles against real low-field scans

```bash
python scripts/calibrate.py --source data/ulf/sub01_3T_T1.nii.gz \
                            --target data/ulf/sub01_64mT_T1_1mm.nii.gz \
                            --out_dir outputs/calibration/sub01_T1
```

Grid search over `calibration.grid` in the config: degrade the 3 T scan, measure noise, in-plane sharpness, through-plane sharpness and tissue contrast, and rank settings by distance to the 64 mT scan. Resample the target to 1 mm first. Repeat over several subjects and contrasts, take the median best parameters, and write them into `profiles:`. Report this procedure and the residual distances in the paper.

### 3. Look at what you are simulating

```bash
python scripts/figure_gallery.py --case_dir data/BraTS2023-GLI/BraTS-GLI-00000-000 --seq t1c
python scripts/plot_ladder.py    --case_dir data/BraTS2023-GLI/BraTS-GLI-00000-000 --axis accel --seq t1c
python scripts/plot_ladder.py    --case_dir data/BraTS2023-GLI/BraTS-GLI-00000-000 --axis slice_mm   # coronal view
```

`figure_gallery.py` also draws the k-space sampling masks and coil sensitivities (method figure material).

### 4. Generate the lattice

```bash
# test split, main-paper axes, 16 processes
python scripts/make_lattice.py --data_root data/BraTS2023-GLI --splits splits/brats2023_gli.json --split test \
    --include original reference ladders profiles --axes snr slice_mm accel motion_frac missing \
    --workers 16 --metrics --out_dir outputs/lattice

# training data for robustness baselines: only the conditions you train on
python scripts/make_lattice.py --data_root data/BraTS2023-GLI --splits splits/brats2023_gli.json --split train \
    --conditions reference snr=10 slice_mm=5 accel=4 --workers 16 --out_dir outputs/lattice
```

The script is resumable: kill it and rerun the same command, and finished case-conditions are skipped (a case-condition counts as finished once its `-acquisition.json` sidecar is written, which happens last). `manifest.csv` is rebuilt from the sidecars at the end. Volumes are stored as int16 with a scale factor by default (lossless to well below the noise floor); use `--dtype float32` if you prefer.

### 5. Export to nnU-Net v2 and train / predict

```bash
export nnUNet_raw=/path/nnUNet_raw nnUNet_preprocessed=/path/nnUNet_preprocessed nnUNet_results=/path/nnUNet_results

python scripts/export_nnunet.py --lattice_dir outputs/lattice --conditions reference \
    --splits splits/brats2023_gli.json --nnunet_raw $nnUNet_raw --dataset_id_start 500

nnUNetv2_plan_and_preprocess -d 500 --verify_dataset_integrity
nnUNetv2_train 500 3d_fullres 0            # folds 0-4 for the 5-model ensemble baseline
nnUNetv2_predict -i $nnUNet_raw/Dataset5XX_Suff_snr_10/imagesTs -o outputs/pred/snr_10 \
    -d 500 -c 3d_fullres -f 0 --save_probabilities
```

Export writes BraTS labels in nnU-Net's region format: whole tumour, tumour core and enhancing tumour, with channels `0000` T1, `0001` T1c, `0002` T2, `0003` FLAIR. A dropped sequence is exported as an all-zero channel, which is how a model trained with modality dropout sees it. `--test_only` exports every case as test data (for BraTS-Africa). Each condition becomes its own dataset; `nnunet_datasets.json` records the mapping.

To save disk, export the test conditions, run prediction, then delete the exported images: the lattice folder can regenerate them.

### 6. Place real scans on the lattice

```bash
python scripts/scan_stats.py --data_root outputs/lattice --out outputs/stats_lattice.csv --max_cases_per_group 30 --workers 8
python scripts/scan_stats.py --data_root data/BraTS-Africa --out outputs/stats_africa.csv --workers 8
python scripts/place_on_lattice.py --real outputs/stats_africa.csv --lattice outputs/stats_lattice.csv \
    --out outputs/placement_africa.csv
```

Each real scan is assigned the nearest simulated condition per sequence. This is the bridge between the controlled lattice and the real low-resource cohort: the sufficiency threshold learned on simulation predicts where real scans fall.

## Compute and storage

Measured on 2 CPU cores, one 240 × 240 × 155 sequence:

| Condition | Time per sequence |
|---|---|
| reference (SNR 40, 8 coils, full sampling) | ≈ 6 s |
| `slice_mm=5` | ≈ 1.5 s |
| `motion_frac=0.3` | ≈ 9 s |
| `accel=4` (CG-SENSE, 10 iterations) | ≈ 36 s |

A case has 4 sequences, so a typical condition costs 10 to 40 s per case and the accelerated conditions dominate. The main-paper test lattice (≈ 250 test cases × ≈ 30 conditions) is roughly 40 to 60 CPU-hours: a few hours on a 16-core node. Use `--workers` for processes and `--fft_threads` for FFT threads within each process.

Storage: one case-condition is about 30 to 45 MB as int16 NIfTI (4 sequences and the label). 250 cases × 47 conditions ≈ 400 to 500 GB, so generate the test split and only the training conditions you need, and use the export-predict-delete loop above.

## Reproducibility

* Global seed in the config; per-(case, condition) seeds from `SeedSequence([seed, crc32(case), crc32(condition)])`.
* Every output folder has a sidecar with the full `Acquisition`, dropped sequences, seed, package version and, with `--metrics`, PSNR per sequence.
* `conditions.json` in the lattice folder records the lattice definition used.
* Unit tests check: exact identity without degradation (1 and 8 coils, with phase); measured noise SD matching the target SNR within 10 %; SENSE unfolding aliasing (error < 5 %, under half of zero-filled); mask geometry; slice sampling with gaps; contrast compression; determinism; sequence dropping; label conversion; lattice de-duplication; patient-level splits; scan statistics moving in the right direction.

## Repository layout

```
sufficiency/            the package
  acquisition.py        Acquisition dataclass (every parameter), validation, axis overrides
  kspace.py             centred orthonormal 2-D FFT
  coils.py              coil sensitivities, coil combination, CG-SENSE
  fields.py             smooth phase and bias fields
  operators.py          contrast compression, slice sampling, k-space masks, motion
  simulator.py          the forward model: simulate_volume, simulate_case
  lattice.py            conditions, lattice construction, per-condition seeds, random conditions
  data.py               BraTS I/O, int16 saving, sidecars, patient-level splits
  nnunet.py             nnU-Net v2 export and BraTS label conversion
  stats.py              scan statistics (noise, sharpness, through-plane, contrast, bias)
  calibration.py        grid-search calibration against a real scan
  metrics.py            PSNR, SSIM
configs/lattice.yaml    the reference acquisition, axes, profiles, splits, calibration grid
scripts/                command-line entry points (each has --help)
tests/                  pytest suite on synthetic phantoms
```

`random_condition()` in `lattice.py` draws random multi-axis acquisitions from the config ranges, for degradation-aware training augmentation.

## Limitations (say these in the paper)

* **2-D Cartesian model on already-reconstructed magnitude images.** The input has its own noise and resolution limits; we can only degrade, never improve. Phase is synthetic.
* **Contrast at low field is approximated**, not simulated from T1/T2/PD maps. Use calibration and treat the contrast axis as supplementary unless validated.
* **Motion is rigid and in-plane**, one event per scan; no through-plane motion, spin history or ghosting from periodic motion.
* **No B0 effects** (distortion, susceptibility, chemical shift), no Gibbs-specific filtering, no 3-D acquisitions.
* **Coil maps are analytic**, not measured; SENSE uses the true maps (no calibration error).
* **Missing sequences are zeroed**; a model never trained with dropout will be unfairly penalised, so state the training protocol.

## Roadmap

* Bloch-based contrast from synthetic T1/T2/PD maps (e.g. SynthSeg labels plus literature relaxation times at 64 mT).
* Measured coil sensitivities and ESPIRiT calibration error.
* 3-D and through-plane motion; radial and spiral trajectories.
* GPU (CuPy / PyTorch) back end for SENSE.

## Citation

```bibtex
@misc{abolade2026sufficiency,
  author = {Abolade, Ilerioluwakiiye},
  title  = {sufficiency-mri: a physics-parameterised acquisition simulator for diagnostic sufficiency},
  year   = {2026},
  url    = {https://github.com/<you>/sufficiency-mri}
}
```

MIT licence.
