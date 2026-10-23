import argparse
import csv
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from sufficiency.calibration import DEFAULT_CAL_FEATURES, calibrate
from sufficiency.data import load_volume
from sufficiency.lattice import base_acquisition


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="high-field NIfTI (1 mm)")
    ap.add_argument("--target", required=True, help="real low-field or low-resource NIfTI")
    ap.add_argument("--config", default="configs/lattice.yaml")
    ap.add_argument("--out_dir", default="outputs/calibration")
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_CAL_FEATURES))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--slab", type=int, default=48, help="central slices simulated per grid point (0 = all)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    source, _, src_hdr = load_volume(args.source)
    target, _, tgt_hdr = load_volume(args.target)
    print(f"source voxel size {src_hdr.get_zooms()[:3]}  (simulator assumes 1 mm isotropic)")
    if any(abs(z - 1.0) > 0.05 for z in tgt_hdr.get_zooms()[:3]):
        print(f"WARNING: target voxel size {tgt_hdr.get_zooms()[:3]}; resample it to 1 mm for comparable statistics")

    base = base_acquisition(cfg)
    target_f, results = calibrate(source, target, cfg["calibration"]["grid"], base,
                                  features=tuple(args.features), seed=cfg.get("seed", 0),
                                  progress=lambda it: tqdm(it, desc="grid"), slab=args.slab or None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(cfg["calibration"]["grid"])
    with open(out_dir / "calibration.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "distance", *keys, *[f"sim_{k}" for k in target_f]])
        for i, r in enumerate(results, 1):
            writer.writerow([i, round(r["distance"], 4), *[r["params"][k] for k in keys],
                             *[round(r["features"][k], 5) for k in target_f]])
    best = {"target_features": target_f, "best_params": results[0]["params"],
            "best_features": results[0]["features"], "distance": results[0]["distance"]}
    with open(out_dir / "best.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\ntarget: " + "  ".join(f"{k}={v:.4f}" for k, v in target_f.items()))
    print(f"\n{'rank':<6}{'distance':>10}  params")
    for i, r in enumerate(results[: args.top], 1):
        print(f"{i:<6}{r['distance']:>10.3f}  {r['params']}")
    print(f"\nSaved {out_dir / 'calibration.csv'} and {out_dir / 'best.json'}")


if __name__ == "__main__":
    main()
