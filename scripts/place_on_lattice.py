import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sufficiency.stats import FEATURES


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="scan_stats.py output for real scans")
    ap.add_argument("--lattice", required=True, help="scan_stats.py output for lattice scans")
    ap.add_argument("--out", default="outputs/placement.csv")
    ap.add_argument("--features", nargs="+", default=["log_snr", "inplane_hf", "through_plane_ratio", "contrast"],
                    choices=list(FEATURES))
    ap.add_argument("--exclude_prefix", nargs="*", default=["missing=", "original"],
                    help="conditions to ignore; missing=k is acquired like the reference, so it is excluded by default")
    args = ap.parse_args()

    feats = args.features
    lattice = defaultdict(list)
    for r in read_rows(args.lattice):
        if any(r["group"].startswith(p) for p in args.exclude_prefix):
            continue
        lattice[(r["sequence"], r["group"])].append([float(r[k]) for k in feats])

    centres = defaultdict(dict)  # sequence -> condition -> median feature vector
    for (seq, cond), vals in lattice.items():
        centres[seq][cond] = np.nanmedian(np.array(vals), axis=0)

    out_rows = []
    counts = defaultdict(Counter)
    for r in read_rows(args.real):
        seq = r["sequence"]
        if seq not in centres:
            continue
        names = sorted(centres[seq])
        mat = np.stack([centres[seq][n] for n in names])
        scale = np.nanstd(mat, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        x = np.array([float(r[k]) for k in feats])
        d = np.sqrt(np.nansum(((mat - x) / scale) ** 2, axis=1))
        best = int(np.argmin(d))
        out_rows.append({"case": r["case"], "sequence": seq, "nearest_condition": names[best],
                         "distance": round(float(d[best]), 4), **{k: r[k] for k in feats}})
        counts[seq][names[best]] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "sequence", "nearest_condition", "distance", *feats])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out}\n")
    for seq in sorted(counts):
        print(f"{seq}:")
        for cond, n in counts[seq].most_common():
            print(f"  {cond:<34}{n:>5}")


if __name__ == "__main__":
    main()
