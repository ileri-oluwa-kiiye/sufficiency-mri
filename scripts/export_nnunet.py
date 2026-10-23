import argparse
import json
from pathlib import Path

from sufficiency.nnunet import export_condition


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lattice_dir", required=True)
    ap.add_argument("--conditions", nargs="+", required=True, help="condition names, or 'all'")
    ap.add_argument("--nnunet_raw", required=True, help="your nnUNet_raw folder")
    ap.add_argument("--dataset_id_start", type=int, default=500)
    ap.add_argument("--prefix", default="Suff", help="dataset name prefix")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--test_only", action="store_true", help="export every case as test data")
    args = ap.parse_args()

    lattice = Path(args.lattice_dir)
    if args.conditions == ["all"]:
        names = sorted(p.name for p in lattice.iterdir() if p.is_dir())
    else:
        names = args.conditions

    mapping = {}
    for i, name in enumerate(names):
        cond_dir = lattice / name
        if not cond_dir.exists():
            raise SystemExit(f"Condition folder not found: {cond_dir}")
        all_ids = sorted(p.name for p in cond_dir.iterdir() if p.is_dir())
        if args.test_only:
            train_ids, test_ids = [], all_ids
        else:
            if not args.splits:
                raise SystemExit("--splits is required unless --test_only is set")
            with open(args.splits) as f:
                splits = json.load(f)
            train_ids, test_ids = splits[args.train_split], splits[args.test_split]
        dataset_id = args.dataset_id_start + i
        root = export_condition(cond_dir, args.nnunet_raw, dataset_id, f"{args.prefix}_{name}",
                                train_ids=train_ids, test_ids=test_ids)
        mapping[name] = {"dataset_id": dataset_id, "path": str(root)}
        print(f"{name:<34}-> {root}")

    out = lattice / "nnunet_datasets.json"
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {out}")


if __name__ == "__main__":
    main()
