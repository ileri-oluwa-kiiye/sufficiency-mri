import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import yaml
from tqdm import tqdm

import sufficiency
from sufficiency.data import SIDECAR_SUFFIX, find_cases, is_complete, load_case, save_case
from sufficiency.kspace import set_fft_workers
from sufficiency.lattice import build_lattice, condition_rng
from sufficiency.metrics import psnr
from sufficiency.simulator import brain_mask_from, simulate_case

_STATE = {}


def _init_worker(state):
    _STATE.update(state)
    set_fft_workers(state["fft_threads"])


def _process_case(case_dir):
    s = _STATE
    case = load_case(case_dir)
    mask = brain_mask_from(case["vols"])
    done = 0
    for cond in s["conditions"]:
        out_root = Path(s["out_dir"]) / cond.name
        if is_complete(out_root, case["id"]) and not s["overwrite"]:
            continue
        t0 = time.time()
        rng = condition_rng(s["seed"], case["id"], cond.name)
        vols = simulate_case(case["vols"], cond.acq, rng, missing=cond.missing,
                             missing_order=s["missing_order"], chunk_slices=s["chunk_slices"])
        dropped = list(s["missing_order"])[:cond.missing]
        sidecar = {
            "case": case["id"],
            "condition": cond.to_dict(),
            "dropped_sequences": dropped,
            "seed": s["seed"],
            "package_version": sufficiency.__version__,
            "seconds": round(time.time() - t0, 2),
        }
        if s["metrics"]:
            sidecar["psnr"] = {seq: (None if seq in dropped else round(psnr(case["vols"][seq], vols[seq], mask), 3))
                               for seq in vols}
        save_case(out_root, case, vols, case["seg"], dtype=s["dtype"], sidecar=sidecar)
        done += 1
    return case["id"], done


def write_manifest(out_dir: Path):
    rows = []
    for path in sorted(out_dir.rglob(f"*{SIDECAR_SUFFIX}")):
        with open(path) as f:
            sc = json.load(f)
        cond = sc["condition"]
        row = {"condition": cond["name"], "kind": cond["kind"], "axis": cond["axis"], "level": cond["level"],
               "case": sc["case"], "missing": cond["missing"], "seconds": sc.get("seconds")}
        for seq, value in (sc.get("psnr") or {}).items():
            row[f"psnr_{seq}"] = value
        rows.append(row)
    if not rows:
        return None
    fieldnames = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    path = out_dir / "manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="outputs/lattice")
    ap.add_argument("--config", default="configs/lattice.yaml")
    ap.add_argument("--include", nargs="+", default=["original", "reference", "ladders", "profiles"],
                    choices=["original", "reference", "ladders", "profiles"])
    ap.add_argument("--axes", nargs="+", default=None, help="restrict ladders to these axes")
    ap.add_argument("--conditions", nargs="+", default=None, help="restrict to these condition names")
    ap.add_argument("--splits", default=None, help="splits JSON from make_splits.py")
    ap.add_argument("--split", nargs="+", default=None, help="which splits to generate, e.g. test")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fft_threads", type=int, default=1, help="FFT threads per worker")
    ap.add_argument("--chunk_slices", type=int, default=16, help="slices per FFT chunk (memory vs speed)")
    ap.add_argument("--dtype", choices=["float32", "int16"], default="int16")
    ap.add_argument("--metrics", action="store_true", help="store PSNR per sequence in the sidecar")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="list conditions and exit")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    conditions = build_lattice(cfg, include=tuple(args.include), axes=args.axes)
    if args.conditions:
        conditions = [c for c in conditions if c.name in set(args.conditions)]

    cases = find_cases(args.data_root)
    if args.splits and args.split:
        with open(args.splits) as f:
            splits = json.load(f)
        keep = {c for s in args.split for c in splits[s]}
        cases = [c for c in cases if c.name in keep]
    cases = cases[: args.max_cases]

    print(f"{len(conditions)} conditions x {len(cases)} cases = {len(conditions) * len(cases)} case-conditions")
    for c in conditions:
        print(f"  {c.name:<34}{c.kind:<10}missing={c.missing}")
    if args.dry_run:
        return
    if not cases:
        raise SystemExit(f"No cases found under {args.data_root} (after split filtering)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "conditions.json", "w") as f:
        json.dump([c.to_dict() for c in conditions], f, indent=2, default=str)

    state = {"conditions": conditions, "out_dir": str(out_dir), "seed": cfg.get("seed", 0),
             "missing_order": cfg.get("missing_order", ["t1c", "t2f", "t2w"]), "dtype": args.dtype,
             "metrics": args.metrics, "overwrite": args.overwrite, "chunk_slices": args.chunk_slices,
             "fft_threads": args.fft_threads}

    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=(state,)) as pool:
            for _ in tqdm(pool.imap_unordered(_process_case, cases), total=len(cases), desc="cases"):
                pass
    else:
        _init_worker(state)
        for case_dir in tqdm(cases, desc="cases"):
            _process_case(case_dir)

    manifest = write_manifest(out_dir)
    print(f"Done. Manifest: {manifest}")


if __name__ == "__main__":
    main()
