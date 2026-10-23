import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sufficiency.acquisition import apply_overrides  # noqa: E402
from sufficiency.coils import coil_maps  # noqa: E402
from sufficiency.data import load_case  # noqa: E402
from sufficiency.lattice import base_acquisition, condition_rng, format_level  # noqa: E402
from sufficiency.metrics import psnr  # noqa: E402
from sufficiency.operators import sampling_mask  # noqa: E402
from sufficiency.simulator import brain_mask_from, simulate_volume  # noqa: E402

# One illustrative level per axis (shown with its view).
EXAMPLES = [
    ("snr", 5, "axial"),
    ("slice_mm", 5, "coronal"),
    ("accel", 4, "axial"),
    ("motion_frac", 0.3, "axial"),
    ("inplane_keep", 0.375, "axial"),
    ("partial_fourier", 0.625, "axial"),
    ("contrast", 0.55, "axial"),
    ("bias", 0.4, "axial"),
]


def take(vol, view, centre):
    if view == "axial":
        return np.rot90(vol[:, :, centre[2]])
    return np.rot90(vol[:, centre[1], :])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--seq", default="t1c", choices=["t1n", "t1c", "t2w", "t2f"])
    ap.add_argument("--config", default="configs/lattice.yaml")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 0)
    case = load_case(args.case_dir)
    mask = brain_mask_from(case["vols"])
    ref = case["vols"][args.seq]
    seg = case["seg"]
    centre = ([int(round(c)) for c in np.argwhere(seg > 0).mean(axis=0)]
              if seg is not None and seg.any() else [s // 2 for s in ref.shape])
    base = base_acquisition(cfg)
    sens = coil_maps(ref.shape[:2], base.coils, np.random.default_rng(seed))
    vmax = float(np.percentile(ref[mask], 99.5))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- image gallery
    panels = [("reference", base, "axial")]
    for axis, level, view in EXAMPLES:
        acq, _ = apply_overrides(base, {axis: level})
        panels.append((f"{axis}={format_level(level)}", acq, view))
    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows))
    for ax, (name, acq, view) in zip(axs.ravel(), panels):
        img = simulate_volume(ref, acq, condition_rng(seed, case["id"], name), brain_mask=mask, coil_sens=sens)
        ax.imshow(take(img, view, centre), cmap="gray", vmin=0, vmax=vmax)
        if seg is not None and (take(seg, view, centre) == 3).any():
            ax.contour(take(seg, view, centre) == 3, levels=[0.5], colors="#ff5a5a", linewidths=0.7)
        ax.set_title(f"{name} ({view})\nPSNR {psnr(ref, img, mask):.1f} dB", fontsize=9)
        ax.axis("off")
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{case['id']} | {args.seq} | red: enhancing tumour", fontsize=10)
    fig.tight_layout()
    path = out / f"gallery_{args.seq}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")

    # --- k-space gallery
    shape = ref.shape[:2]
    rng = np.random.default_rng(seed)
    masks = [
        ("accel=4 equispaced", apply_overrides(base, {"accel": 4})[0]),
        ("accel=4 random", apply_overrides(base, {"accel": 4, "undersampling": "random"})[0]),
        ("partial_fourier=0.625", apply_overrides(base, {"partial_fourier": 0.625})[0]),
        ("inplane_keep=0.5", apply_overrides(base, {"inplane_keep": 0.5})[0]),
    ]
    fig, axs = plt.subplots(1, len(masks) + 2, figsize=(2.6 * (len(masks) + 2), 3.0))
    for ax, (name, acq) in zip(axs, masks):
        ax.imshow(sampling_mask(shape, acq, rng), cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    axs[-2].imshow(np.abs(sens[0]), cmap="magma")
    axs[-2].set_title("coil 1 sensitivity", fontsize=9)
    axs[-2].axis("off")
    axs[-1].imshow(np.sqrt(np.sum(np.abs(sens) ** 2, axis=0)), cmap="magma", vmin=0, vmax=1.2)
    axs[-1].set_title(f"sum over {sens.shape[0]} coils", fontsize=9)
    axs[-1].axis("off")
    fig.suptitle("k-space sampling (white = acquired; vertical = phase encode)", fontsize=10)
    fig.tight_layout()
    path = out / "gallery_kspace.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
