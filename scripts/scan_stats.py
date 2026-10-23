import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sufficiency.data import find_cases, load_case
from sufficiency.simulator import brain_mask_from
from sufficiency.stats import FEATURES, scan_features

_ROOT = {}


def _init(root):
    _ROOT["root"] = Path(root)


def _stats(case_dir):
    root = _ROOT["root"]
    case = load_case(case_dir)
    mask = brain_mask_from(case["vols"])
    rel = case_dir.parent.relative_to(root)
    group = str(rel) if str(rel) != "." else "."
    rows = []
    for seq, vol in case["vols"].items():
        if not vol.any():
            continue  # dropped sequence
        f = scan_features(vol, mask)
        rows.append({"group": group, "case": case["id"], "sequence": seq, **{k: round(v, 6) for k, v in f.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="outputs/scan_stats.csv")
    ap.add_argument("--max_cases_per_group", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.data_root)
    cases = find_cases(root)
    if args.max_cases_per_group:
        by_group = defaultdict(list)
        for c in cases:
            by_group[c.parent].append(c)
        cases = [c for g in sorted(by_group) for c in by_group[g][: args.max_cases_per_group]]
    if not cases:
        raise SystemExit(f"No cases found under {root}")

    _init(root)
    rows = []
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(str(root),)) as pool:
            for r in tqdm(pool.imap_unordered(_stats, cases), total=len(cases), desc="cases"):
                rows.extend(r)
    else:
        for c in tqdm(cases, desc="cases"):
            rows.extend(_stats(c))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "case", "sequence", *FEATURES])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["group"], r["case"], r["sequence"])))
    print(f"Wrote {len(rows)} rows to {out}\n")

    summary = defaultdict(list)
    for r in rows:
        summary[(r["group"], r["sequence"])].append([r[k] for k in FEATURES])
    print(f"{'group':<34}{'seq':<6}" + "".join(f"{k:>21}" for k in FEATURES))
    for (group, seq), vals in sorted(summary.items()):
        med = np.nanmedian(np.array(vals, dtype=float), axis=0)
        print(f"{group:<34}{seq:<6}" + "".join(f"{v:>21.4f}" for v in med))


if __name__ == "__main__":
    main()
