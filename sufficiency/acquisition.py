from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

RECONS = ("sense", "zero_filled")
PATTERNS = ("equispaced", "random")
ORDERINGS = ("linear", "centric", "interleaved")
SLICE_PROFILES = ("rect", "gaussian")


@dataclass(frozen=True)
class Acquisition:
    # field strength and tissue contrast
    snr: float | None = None            # mean brain signal / thermal noise sd; None = no added noise
    contrast: float = 1.0               # between-tissue contrast kept (1 = unchanged); low-field proxy
    bias: float = 0.0                   # receive inhomogeneity strength (0.2 ~ +/-20%)
    # through-plane
    slice_mm: float = 1.0               # slice thickness
    slice_gap_mm: float = 0.0           # gap between slices
    slice_profile: str = "rect"         # rect | gaussian
    # in-plane sampling (phase-encode axis = image axis 0)
    keep_pe: float = 1.0                # fraction of k-space kept along phase encode
    keep_ro: float = 1.0                # fraction of k-space kept along readout
    accel: int = 1                      # undersampling factor R
    center_fraction: float = 0.08       # fully sampled central fraction (calibration lines)
    undersampling: str = "equispaced"   # equispaced (parallel imaging) | random (compressed sensing)
    partial_fourier: float = 1.0        # fraction of phase-encode lines acquired (0.5-1)
    # receiver and reconstruction
    coils: int = 8
    recon: str = "sense"                # sense | zero_filled
    sense_iters: int = 10
    phase_amplitude: float = 1.5        # max synthetic background phase (radians)
    # motion
    motion_frac: float = 0.0            # fraction of phase-encode lines acquired after the head moved
    motion_shift_mm: float = 2.0
    motion_rotation_deg: float = 2.0
    motion_ordering: str = "linear"     # linear | centric | interleaved

    def validate(self) -> "Acquisition":
        def bad(msg):
            raise ValueError(f"Invalid acquisition: {msg}")

        if self.snr is not None and self.snr <= 0:
            bad("snr must be > 0 or None")
        if not 0 < self.contrast <= 1:
            bad("contrast must be in (0, 1]")
        if self.bias < 0:
            bad("bias must be >= 0")
        if self.slice_mm < 1 or self.slice_gap_mm < 0:
            bad("slice_mm must be >= 1 and slice_gap_mm >= 0")
        if self.slice_profile not in SLICE_PROFILES:
            bad(f"slice_profile must be one of {SLICE_PROFILES}")
        if not (0 < self.keep_pe <= 1 and 0 < self.keep_ro <= 1):
            bad("keep_pe and keep_ro must be in (0, 1]")
        if int(self.accel) != self.accel or self.accel < 1:
            bad("accel must be an integer >= 1")
        if not 0 < self.center_fraction < 1:
            bad("center_fraction must be in (0, 1)")
        if self.undersampling not in PATTERNS:
            bad(f"undersampling must be one of {PATTERNS}")
        if not 0.5 <= self.partial_fourier <= 1:
            bad("partial_fourier must be in [0.5, 1]")
        if self.coils < 1:
            bad("coils must be >= 1")
        if self.recon not in RECONS:
            bad(f"recon must be one of {RECONS}")
        if not 0 <= self.motion_frac < 1:
            bad("motion_frac must be in [0, 1)")
        if self.motion_ordering not in ORDERINGS:
            bad(f"motion_ordering must be one of {ORDERINGS}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Acquisition":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown acquisition fields: {sorted(unknown)}")
        return cls(**d).validate()


# Lattice axes and the Acquisition fields each one sets. 'missing' is handled per case.
AXIS_FIELDS = {
    "snr": ("snr",),
    "contrast": ("contrast",),
    "bias": ("bias",),
    "slice_mm": ("slice_mm",),
    "inplane_keep": ("keep_pe", "keep_ro"),
    "accel": ("accel",),
    "partial_fourier": ("partial_fourier",),
    "motion_frac": ("motion_frac",),
}
AXES = tuple(AXIS_FIELDS) + ("missing",)


def apply_overrides(base: Acquisition, spec: dict) -> tuple[Acquisition, int]:
    """Apply axis names or field names from `spec` to `base`. Returns (acquisition, n_missing)."""
    spec = dict(spec)
    missing = int(spec.pop("missing", 0))
    updates = {}
    for key, value in spec.items():
        if key in AXIS_FIELDS:
            for f in AXIS_FIELDS[key]:
                updates[f] = value
        elif key in Acquisition.__dataclass_fields__:
            updates[key] = value
        else:
            raise ValueError(f"Unknown acquisition parameter or axis: {key!r}")
    if "accel" in updates:
        updates["accel"] = int(updates["accel"])
    acq = replace(base, **updates).validate()
    return acq, missing
