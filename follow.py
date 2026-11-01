"""Camera frames -> a small target's bearing -> Poisson rates on LC10a. Stage 3's translation, "follow me".

This is the hand-written part, like `looming.py` and `motion.py`: everything here is ours, not the fly's.

**What the fly has, and what it does not.** FlyWire is a female brain: there is no P1, so no
courtship pursuit — "the duck courts you" is not in this graph. What is in it is the visual object
pathway that pursuit is built on: LC10a, a lobula columnar type that answers small moving objects
(Ribeiro et al. 2018; Hindmarsh Sten et al. 2021), into the anterior optic tubercle, and from there
AOTU019 (GABA, mostly onto the *other* side's DNa02) and AOTU025 (ACh) to DNa02, the steering
descending neuron. Open loop, a target on the left makes left DNa02 fire more than right — a turn
toward it — graded with bearing, and only with the real wiring (`experiments/follow_dn.py`).

**Where each LC10a looks**, from the wiring: the synapse-weighted mean viewing direction of its
inputs that sit on FlyWire's column map (Tm20, Tm4, ... `retinotopy.py`), on its own eye.

The two eyes' maps are not mirror images: the left eye's LC10a look further front, so a target
dead ahead drives the left eye more (931 Hz against 782 in all), and DNa02 leans a little left there
(+1.6 Hz at 0 degrees). From 10 degrees out the target's own side dominates. Whether that is the
fly or our column-to-angle approximation (`retinotopy.py`) we cannot tell from here.

**The detector** stands in for a person or object detector: in the grey sim camera the target
(`sim/make_scene.py`) lights itself and is the only thing at full white, so it is the largest
saturated blob after an opening that removes the floor's thin grid lines. Its centroid gives a
bearing through the pinhole model (45 degrees across the 360-pixel width). On a real duck this is
where a detector's box would come in.

**The encoder**: the target at azimuth `az` (+ = left) and elevation `el` drives every LC10a by how
close its receptive field is, max_rate * exp(-d^2 / 2 sigma^2). The left eye sees it at az, the
right eye at -az (each eye's own side is +). 120 Hz and sigma = 10 degrees from a dose sweep:
DNa02 is graded there; at 250 Hz it saturates and folds. No target, no drive. LC10a's preference
for *moving* objects is not modelled: a target standing still drives it as much.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

import neurons as N
import retinotopy as R


@lru_cache(maxsize=None)
def receptive_fields(cell_type: str = 'LC10a') -> pd.DataFrame:
    """Per neuron: side, and the mean viewing direction (az, el, degrees) of its column-mapped inputs."""
    d = R.directions()
    con = pd.read_parquet(R.CONNECTIVITY, columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    rows = []
    for side in ('left', 'right'):
        e = con[con.Postsynaptic_ID.isin(set(N.ids(cell_type, side))) & con.Presynaptic_ID.isin(d.index)].copy()
        e = e[d.hemisphere.reindex(e.Presynaptic_ID).to_numpy() == side]
        e['az'] = d.az_deg.reindex(e.Presynaptic_ID).to_numpy() * e.Connectivity
        e['el'] = d.el_deg.reindex(e.Presynaptic_ID).to_numpy() * e.Connectivity
        g = e.groupby('Postsynaptic_ID')[['az', 'el', 'Connectivity']].sum()
        rows.append(pd.DataFrame({'side': side, 'az': g.az / g.Connectivity, 'el': g.el / g.Connectivity}))
    return pd.concat(rows)


@dataclass
class Params:
    max_rate: float = 120.0      # Hz
    sigma_deg: float = 10.0
    hfov_deg: float = 45.0       # across the upright frame's width
    white: int = 250             # grey level of the target
    min_area: int = 30           # pixels, after the opening
    bar_aspect: float = 1.6      # height / width above which a target is the bar


def bearing(grey: np.ndarray, p: Params = Params()) -> tuple[float, float, int, str] | None:
    """The target in an upright grey frame: azimuth (+ = left) and elevation (+ = up) in degrees,
    its area in pixels, and what it is — 'bar' when its box is more than `bar_aspect` times taller
    than wide, else 'ball'; None when there is none."""
    h, w = grey.shape
    m = cv2.morphologyEx((grey >= p.white).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = int(st[k, cv2.CC_STAT_AREA])
    if area < p.min_area:
        return None
    f = (w / 2) / math.tan(math.radians(p.hfov_deg / 2))
    x, y = cen[k]
    kind = 'bar' if st[k, cv2.CC_STAT_HEIGHT] > p.bar_aspect * st[k, cv2.CC_STAT_WIDTH] else 'ball'
    return math.degrees(math.atan((w / 2 - x) / f)), math.degrees(math.atan((h / 2 - y) / f)), area, kind


class FollowEncoder:
    def __init__(self, params: Params | None = None, cell_type: str = 'LC10a'):
        self.p = params or Params()
        self.rf = receptive_fields(cell_type)
        self.ids = [int(f) for f in self.rf.index]
        self.target: tuple[float, float, int, str] | None = None   # az, el (deg), area (px), kind, last frame

    def rates_at(self, az: float, el: float) -> dict[int, float]:
        own = np.where(self.rf.side == 'left', az, -az)
        d2 = (self.rf.az.to_numpy() - own) ** 2 + (self.rf.el.to_numpy() - el) ** 2
        r = self.p.max_rate * np.exp(-d2 / (2 * self.p.sigma_deg ** 2))
        return {f: float(x) if x > 1.0 else 0.0 for f, x in zip(self.ids, r)}

    def update(self, grey: np.ndarray, *_, **__) -> dict[int, float]:
        """Rates for every LC10a — zeros included, so a target that leaves stops driving."""
        self.target = bearing(grey, self.p)
        if self.target is None:
            return dict.fromkeys(self.ids, 0.0)
        return self.rates_at(self.target[0], self.target[1])
