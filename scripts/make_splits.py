import argparse
import json
from pathlib import Path

import yaml

from sufficiency.data import find_cases, make_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="splits/splits.json")
    ap.add_argument("--config", default="configs/lattice.yaml")
    ap.add_argument("--all_as", default=None,
                    help="put every case in one split with this name (e.g. 'external' for BraTS-Africa)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    case_ids = [p.name for p in find_cases(args.data_root)]
    if not case_ids:
        raise SystemExit(f"No cases found under {args.data_root}")

    if args.all_as:
        splits = {args.all_as: sorted(case_ids)}
    else:
        splits = make_splits(case_ids, cfg["splits"]["fractions"], seed=cfg.get("seed", 0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(splits, f, indent=2)
    for name, ids in splits.items():
        print(f"{name:<12}{len(ids):>6} cases")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
