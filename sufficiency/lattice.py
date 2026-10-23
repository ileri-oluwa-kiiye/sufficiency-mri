from __future__ import annotations

from dataclasses import dataclass, replace
from zlib import crc32

import numpy as np

from .acquisition import AXES, Acquisition, apply_overrides


@dataclass(frozen=True)
class Condition:
    name: str                  # folder-safe name, e.g. "snr=10" or "profile=portable_64mT_like"
    kind: str                  # original | reference | ladder | profile
    acq: Acquisition
    missing: int = 0
    axis: str | None = None
    level: object = None

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "axis": self.axis, "level": self.level,
                "missing": self.missing, "acquisition": self.acq.to_dict()}


def format_level(level) -> str:
    if level is None:
        return "none"
    if isinstance(level, float) and level.is_integer():
        return str(int(level))
    return str(level)


def base_acquisition(cfg: dict) -> Acquisition:
    return Acquisition.from_dict(cfg.get("defaults", {}))


def build_lattice(cfg: dict, include=("original", "reference", "ladders", "profiles"), axes=None) -> list[Condition]:
    """All conditions described by the config, without duplicates.

    original   the untouched input (no added noise; coils and phase cancel exactly)
    reference  the default acquisition (by default mild noise, so reconstruction effects show)
    ladders    one axis changed at a time, starting from the reference
    profiles   several axes changed together (realistic low-resource acquisitions)
    """
    base = base_acquisition(cfg)
    conditions, seen = [], set()

    def add(cond: Condition):
        key = (cond.acq, cond.missing)
        if key in seen:
            return
        seen.add(key)
        conditions.append(cond)

    if "original" in include:
        add(Condition("original", "original", replace(base, snr=None)))
    if "reference" in include:
        add(Condition("reference", "reference", base))
    else:
        seen.add((base, 0))  # ladders never regenerate the reference itself

    if "ladders" in include:
        for axis, levels in cfg.get("axes", {}).items():
            if axis not in AXES:
                raise ValueError(f"Unknown axis '{axis}' in config. Options: {AXES}")
            if axes and axis not in axes:
                continue
            for level in levels:
                acq, missing = apply_overrides(base, {axis: level})
                add(Condition(f"{axis}={format_level(level)}", "ladder", acq, missing, axis, level))

    if "profiles" in include:
        for name, spec in (cfg.get("profiles") or {}).items():
            acq, missing = apply_overrides(base, spec)
            add(Condition(f"profile={name}", "profile", acq, missing))
    return conditions


def condition_rng(seed: int, case_id: str, condition_name: str) -> np.random.Generator:
    """Deterministic RNG per (seed, case, condition), independent of processing order."""
    ss = np.random.SeedSequence([int(seed), crc32(case_id.encode()), crc32(condition_name.encode())])
    return np.random.default_rng(ss)


def random_condition(rng: np.random.Generator, cfg: dict, max_axes: int = 3) -> Condition:
    """Sample a random point of the lattice: 1..max_axes axes set to non-reference levels.

    Useful for training a per-scan sufficiency predictor over many budgets.
    """
    base = base_acquisition(cfg)
    axes = [a for a, levels in cfg["axes"].items() if len(levels) > 1]
    k = int(rng.integers(1, min(max_axes, len(axes)) + 1))
    chosen = rng.choice(axes, size=k, replace=False)
    spec = {}
    for axis in chosen:
        levels = cfg["axes"][axis][1:]  # skip the reference level
        spec[str(axis)] = levels[int(rng.integers(len(levels)))]
    acq, missing = apply_overrides(base, spec)
    name = "random:" + ",".join(f"{a}={format_level(v)}" for a, v in sorted(spec.items()))
    return Condition(name, "random", acq, missing)
