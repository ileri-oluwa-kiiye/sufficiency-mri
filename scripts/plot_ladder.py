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
from sufficiency.simulator import brain_mask_from, simulate_volume  # noqa: E402


def tumour_centre(seg, shape):
    if seg is not None and seg.any():
        return [int(round(c)) for c in np.argwhere(seg > 0).mean(axis=0)]
    return [s // 2 for s in shape]


def take(vol, view, centre):
    if view == "axial":
        return np.rot90(vol[:, :, centre[2]])
    if view == "coronal":
        return np.rot90(vol[:, centre[1], :])
    return np.rot90(vol[centre[0], :, :])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--axis", default="snr")
    ap.add_argument("--seq", default="t1c", choices=["t1n", "t1c", "t2w", "t2f"])
    ap.add_argument("--view", default=None, choices=["axial", "coronal", "sagittal"],
                    help="default: coronal for slice_mm, axial otherwise")
    ap.add_argument("--config", default="configs/lattice.yaml")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.axis == "missing":
        raise SystemExit("'missing' drops whole sequences; plot one of the acquisition axes instead.")
    if args.axis not in cfg["axes"]:
        raise SystemExit(f"Axis '{args.axis}' is not in {args.config}.")
    view = args.view or ("coronal" if args.axis == "slice_mm" else "axial")

    case = load_case(args.case_dir)
    mask = brain_mask_from(case["vols"])
    ref = case["vols"][args.seq]
    seg = case["seg"]
    centre = tumour_centre(seg, ref.shape)
    base = base_acquisition(cfg)
    sens = coil_maps(ref.shape[:2], base.coils, np.random.default_rng(cfg.get("seed", 0)))

    levels = cfg["axes"][args.axis]
    vmax = float(np.percentile(ref[mask], 99.5))
    fig, axs = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 3.2))
    for ax, level in zip(np.atleast_1d(axs), levels):
        acq, _ = apply_overrides(base, {args.axis: level})
        name = f"{args.axis}={format_level(level)}"
        img = simulate_volume(ref, acq, condition_rng(cfg.get("seed", 0), case["id"], name),
                              brain_mask=mask, coil_sens=sens)
        ax.imshow(take(img, view, centre), cmap="gray", vmin=0, vmax=vmax)
        if seg is not None and (take(seg, view, centre) == 3).any():
            ax.contour(take(seg, view, centre) == 3, levels=[0.5], colors="#ff5a5a", linewidths=0.8)
        ax.set_title(f"{name}\nPSNR {psnr(ref, img, mask):.1f} dB", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{case['id']} | {args.seq} | {view} | red: enhancing tumour", fontsize=10)
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ladder_{args.axis}_{args.seq}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
