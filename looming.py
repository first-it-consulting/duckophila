"""ToF frames -> Poisson rates on the fly's looming detectors (LPLC2, LC4).

This is the hand-written part: everything here is our translation, not the fly's. Keep it small
and say what it assumes.

**The signal is inverse time-to-contact**, tau^-1 = -d'/d per zone. For a small object it equals
the relative expansion rate of its image (theta'/theta), which is what looming detectors respond
to, and it needs no object size — which a ToF cannot know.

**Retinotopy, crudely.** Column 0 is the sensor's left (`kinematics/src/tof.rs`), so columns 0-3
feed the left optic lobe and 4-7 the right. Every LPLC2/LC4 neuron of a side is assigned to one
zone of that side's 32, round-robin in a fixed order: ~3 LPLC2 + ~1.5 LC4 per zone. A large
looming object drives many neurons, a small one few — the size dependence the giant fiber reads.
Real receptive fields would be better; this is v1. The ToF sees 45 degrees straight ahead, so this
also stretches the fly's frontal view over each whole eye.

**Arrival is not approach.** A zone whose range halves in one frame saw an occluder slide in,
not something coming at it (at 1.5 m/s, 30 cm away, a frame closes 10 cm, a third); it restarts.

**Only fast approach drives — a duck choice, not a fly fact.** Anything that ends near the duck
has a short time-to-contact, a hand reaching to pet it included. In the fly, fast looming is what
recruits the giant fiber (slow looming escapes go through other descending neurons); here a zone
drives only when it closes faster than `v_min` (0.4 m/s), ramping to full at twice that. Without
this a slow hand drew a startle every time.

**Noise.** Range and its rate come from a line fit over the last `window` frames (~0.27 s at
15 Hz), not a two-frame difference, which turned 1 cm of sensor noise into fake looming while
walking at a wall. A zone needs three valid frames in a row before it can drive. Returns under
10 cm are dropped (sensor crosstalk, `kinematics::tof::MIN_RANGE_M`).

**The floor and the duck's own walking** are the bridge's to supply (`geometry.py`): the first live
run in duck-sim startled at the floor, which comes and goes as a standing duck sways its head.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import neurons as N

ROWS = COLS = 8
STATUS_VALID = (5, 9)
MIN_RANGE_M = 0.10


@dataclass
class Params:
    window: int = 4             # frames per line fit
    tau_k: float = 2.0          # 1/s: tau^-1 at which a zone drives half of max_rate
    tau_min: float = 0.3        # 1/s: below this, no drive (sensor noise, slow drift)
    v_min: float = 0.4          # m/s: slower approaches do not drive (see the module doc)
    jump: float = 0.5           # a zone whose range falls below this fraction of the last in one
                                # frame saw something *arrive* (an edge, an occluder), not approach
    max_rate: float = 150.0     # Hz, the reference model's default drive


@dataclass
class LoomingEncoder:
    params: Params = field(default_factory=Params)
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # zone index (row-major) -> list of FlyWire ids
        self.zone_ids: list[list[int]] = [[] for _ in range(ROWS * COLS)]
        for side, cols in (('left', range(0, 4)), ('right', range(4, 8))):
            zones = [r * COLS + c for r in range(ROWS) for c in cols]
            for cell_type in N.LOOMING:
                ids = N.ids(cell_type, side)
                rng.shuffle(ids)
                for k, fid in enumerate(ids):
                    self.zone_ids[zones[k % len(zones)]].append(fid)
        self.all_ids = [f for z in self.zone_ids for f in z]
        self._hist: list[tuple[float, np.ndarray]] = []   # (t, range per zone, NaN where invalid)
        self._last = np.full(ROWS * COLS, np.nan)
        self.inv_tau = np.zeros(ROWS * COLS)
        self.speed = np.zeros(ROWS * COLS)

    def update(self, distance_mm, status, t: float, floor=None, ego=None) -> dict[int, float]:
        """One frame -> rates for every looming neuron (0 where not driven). `t` in seconds.

        `floor`: zones whose return is the floor (`geometry.Geometry.floor`), dropped like an
        invalid zone. `ego`: m/s each zone closes because the duck itself is walking
        (`Geometry.ego_closing`), subtracted from the approach speed."""
        p = self.params
        d = np.asarray(distance_mm, dtype=float) / 1000.0
        ok = np.isin(np.asarray(status), STATUS_VALID) & (d >= MIN_RANGE_M)
        if floor is not None:
            ok &= ~np.asarray(floor, bool)
        d = np.where(ok, d, np.nan)
        # an occluder sliding in starts that zone's history over: arrival is not approach
        arrived = d < p.jump * self._last
        for _, h in self._hist:
            h[arrived] = np.nan
        self._last = d
        self._hist = (self._hist + [(t, d.copy())])[-p.window:]

        ts = np.array([h[0] for h in self._hist])
        ds = np.stack([h[1] for h in self._hist])          # frames x zones
        ok = ~np.isnan(ds)
        n = ok.sum(0)
        tc = np.where(ok, ts[:, None] - ts.mean(), 0.0)
        tm = (tc * ok).sum(0) / np.maximum(n, 1)           # mean time of the valid frames
        tc = np.where(ok, tc - tm, 0.0)
        dm = np.where(ok, ds, 0.0).sum(0) / np.maximum(n, 1)
        slope = (tc * np.where(ok, ds - dm, 0.0)).sum(0) / np.maximum((tc * tc).sum(0), 1e-9)
        d_now = dm + slope * (ts[-1] - ts.mean() - tm)
        usable = (n >= 3) & ok[-1]
        self.speed = np.where(usable, -slope - (0.0 if ego is None else np.asarray(ego)), 0.0)
        self.inv_tau = np.where(usable & (self.speed > 0), self.speed / np.maximum(d_now, MIN_RANGE_M), 0.0)

        drive = np.maximum(self.inv_tau - p.tau_min, 0.0)
        gate = np.clip((self.speed - p.v_min) / p.v_min, 0.0, 1.0)
        return self._rates(gate * p.max_rate * drive / (drive + p.tau_k))

    def _rates(self, zone_rate) -> dict[int, float]:
        return {f: float(zone_rate[z]) for z, ids in enumerate(self.zone_ids) for f in ids}
