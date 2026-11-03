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


@dataclass
class MovingParams:
    lag: int = 3                 # frames between the two compared (0.1 s at 30 fps)
    threshold: int = 20          # grey levels of change that count as motion
    min_area: float = 0.002      # of the frame: smaller is flicker (an LED, noise)
    max_area: float = 0.35       # of the frame: larger is not a small object (a leg right at the lens)
    ego_area: float = 0.5        # of the frame moving at once: the camera moved, or the exposure changed
    compensate: bool = True      # subtract the camera's own motion first (a duck turning)
    margin_px: int = 6           # slack at the edges a shift brings in
    turning_px: float = 3.0      # camera shift over `lag` above which the parallax test applies
    parallax: float = 0.7        # a candidate shifting the room's way by this share of the room's shift, or more, is scenery
    persist: int = 3             # frames a target must be seen running before it counts
    hold_s: float = 1.0          # a target that pauses is kept this long: a stepping foot rests between steps
    jump_px: float = 80.0        # farther than this from the last target is a new one ...
    jump_widths: float = 1.5     # ... or than this many of its widths: a near pair of feet steps further apart
    pick: str = 'lowest'         # which blob: 'lowest' (the nearest floor point) or 'largest'
    min_fill: float = 0.0        # of its box a blob must fill: a shimmering line fills little of it
    bottom_px: int = -1          # a blob reaching this close to the bottom edge is the duck's own near floor (-1: off)
    travel_px: float = 8.0       # it must have moved this far over the last `travel_s`: a blinking LED does not
    travel_s: float = 0.5


class MovingTarget:
    """A real-world detector: the nearest small thing that moves. Ours, like `bearing`.

    The sim's target is the only white thing in its world; a room is full of white things — chairs,
    a rug, a window — and a phone or webcam never saturates. What LC10a answers in a fly is a small
    *moving* object, so this looks for exactly that, and for nothing about brightness: what changed
    over the last `lag` frames, opened to remove noise, closed to join a leg's parts; blobs between
    `min_area` and `max_area` of the frame; the lowest one (the floor is nearer lower down, and a
    walking person's feet are the bottom of their blob), at the bottom of its box. It must also
    travel (`travel_px` in `travel_s`): an LED blinking or an edge shimmering changes in place. When most of the
    frame changes at once, the camera moved: no target (that is what the efference copy is for, on a
    walking duck). The target must run for `persist` frames, and survives a pause of `hold_s`.
    Unlike `bearing`, a target standing still disappears — closer to LC10a, and not what the sim's
    glowing ball does."""

    def __init__(self, p: MovingParams | None = None, hfov_deg: float = 45.0, fps: float = 30.0):
        self.p, self.hfov, self.fps = p or MovingParams(), hfov_deg, fps
        self.past: list[np.ndarray] = []
        self.last: tuple[float, float] | None = None
        self.run = 0
        self.missing = 0
        self.mask: np.ndarray | None = None          # the last motion mask, for display
        self.track: list[tuple[float, float]] = []
        self.shift = (0.0, 0.0)                      # the last camera shift taken off, px
        self.width = 0.0                             # the current target's width, px
        self._width = 0.0

    def __call__(self, grey: np.ndarray, shift: tuple[float | None, float | None] | None = None):
        """`shift`: how far (dx, dy, pixels) the camera's own motion moved the image over the last `lag`
        frames, if known — on a duck, from the gyro and the efference copy. Without it, and with
        `compensate`, it is estimated from the image (phase correlation): the room's shift."""
        p = self.p
        mo = self.motion(grey, shift)
        if mo is None:
            return None
        m, st, ok, h, w = mo
        hit = None
        if ok:
            k = max(ok, key=(lambda k: st[k, cv2.CC_STAT_AREA]) if p.pick == 'largest'
                    else (lambda k: st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]))
            x0, y0, bw, bh, area = st[k]
            foot = m[y0 + bh - max(1, bh // 4): y0 + bh, x0: x0 + bw]
            xs = np.flatnonzero(foot.any(0))
            hit = (x0 + float(xs.mean()) if len(xs) else x0 + bw / 2, float(y0 + bh), int(area))
            self._width = float(bw)
        return self._track_hit(hit, h, w)

    def motion(self, grey: np.ndarray, shift=None):
        """The motion image of this frame against `lag` frames ago, the camera's own motion taken off,
        and which of its blobs pass the filters: (mask, blob stats, kept blob labels, h, w), or None
        while the first frames come in."""
        p = self.p
        g = cv2.GaussianBlur(grey, (5, 5), 0)
        self.past.append(g)
        if len(self.past) <= p.lag:
            return None
        old = self.past.pop(0)
        h, w = g.shape
        valid = np.ones_like(g)
        if p.compensate:
            dx, dy = shift if shift is not None else (None, None)
            if dx is None or dy is None:            # what is not known is estimated from the image
                win = cv2.createHanningWindow((w, h), cv2.CV_32F)
                (ex, ey), _ = cv2.phaseCorrelate(old.astype(np.float32), g.astype(np.float32), win)
                dx = ex if dx is None else dx
                dy = ey if dy is None else dy
            self.shift = (dx, dy)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            raw = old
            old = cv2.warpAffine(old, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            valid = cv2.erode(cv2.warpAffine(valid, M, (w, h)), np.ones((2 * p.margin_px + 1,) * 2, np.uint8))
        m = ((cv2.absdiff(g, old) > p.threshold) & (valid > 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # thin edge shimmer goes
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        self.mask = m
        n, labels, st, _ = cv2.connectedComponentsWithStats(m)
        self.labels = labels
        ok = []
        if m.mean() < p.ego_area:
            ok = [k for k in range(1, n) if p.min_area * h * w <= st[k, cv2.CC_STAT_AREA] <= p.max_area * h * w
                  and st[k, cv2.CC_STAT_AREA] >= p.min_fill * st[k, cv2.CC_STAT_WIDTH] * st[k, cv2.CC_STAT_HEIGHT]
                  and (p.bottom_px < 0 or st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] < h - p.bottom_px)]
            if p.compensate and abs(self.shift[0]) > p.turning_px:
                # one border strip, as wide as the shift, is dropped while turning: on the four clips the false
                # targets sat at the border the view leaves from (found by trying both, not derived)
                edge = abs(self.shift[0]) + p.margin_px * 2
                if self.shift[0] < 0:
                    ok = [k for k in ok if st[k, cv2.CC_STAT_LEFT] > edge]
                else:
                    ok = [k for k in ok if st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] < w - edge]
                ok = [k for k in ok if not self._scenery(raw, g, st[k])]
        return m, st, ok, h, w

    def _track_hit(self, hit, h, w):
        p = self.p
        if hit and self.last and math.dist(hit[:2], self.last) > max(p.jump_px, p.jump_widths * self.width):
            self.run = 0                              # somewhere else: a new target starts over
        if hit:
            if self.run == 0:
                self.track = []
            self.last, self.run, self.missing = hit[:2], self.run + 1, 0
            self.width = self._width
            self.track = (self.track + [hit[:2]])[-max(2, int(p.travel_s * self.fps)):]
        else:
            self.missing += 1
            if self.missing > p.hold_s * self.fps:
                self.last, self.run, self.track = None, 0, []
        if self.last is None or self.run < p.persist:
            return None
        xy = np.array(self.track)
        if np.ptp(xy, axis=0).max() < p.travel_px:
            return None
        f = (w / 2) / math.tan(math.radians(self.hfov / 2))
        x, y = self.last
        return math.degrees(math.atan((w / 2 - x) / f)), math.degrees(math.atan((h / 2 - y) / f)), \
            hit[2] if hit else 0, 'moving'


    def _scenery(self, old: np.ndarray, new: np.ndarray, box) -> bool:
        """While the camera turns, is this candidate just nearer scenery? Near things slide the room's way,
        only further (parallax: a duck's camera sits ahead of its turning axis, a hand-held phone moves);
        a target held in view, or walking its own way, does not. The candidate's own shift, in its box,
        against the room's."""
        x0, y0, bw, bh = (int(v) for v in box[:4])
        pad = 12
        xa, ya = max(0, x0 - pad), max(0, y0 - pad)
        xb, yb = min(new.shape[1], x0 + bw + pad), min(new.shape[0], y0 + bh + pad)
        if xb - xa < 16 or yb - ya < 16:
            return False
        a, b = old[ya:yb, xa:xb].astype(np.float32), new[ya:yb, xa:xb].astype(np.float32)
        win = cv2.createHanningWindow((xb - xa, yb - ya), cv2.CV_32F)
        (bx, _), _ = cv2.phaseCorrelate(a, b, win)
        room = self.shift[0]
        return bx * room > 0 and abs(bx) >= self.p.parallax * abs(room)


@dataclass
class PopulationParams:
    cell_deg: float = 2.0        # the motion image is pooled into cells about this size
    disc_deg: float = 2.5        # a moving disc this big, centred on a receptive field, drives it at max_rate (motion is mostly edges)
    tau_s: float = 0.3           # LC10a drive decays with this time constant when the motion stops
    foot_deg: float = 8.0        # of each moving blob only its lowest this many degrees drive (0: all of it)
    suppress_deg_s: float = 20.0 # while the camera turns faster than this, no new drive (saccadic suppression)


class PopulationMotion:
    """The real-world encoder without a target: every LC10a driven by the motion in its own receptive field.

    `MovingTarget` turns the motion image into one bearing and so has to pick: which blob, whether it
    is the same one as before, whether it has run long enough. A person has two feet, and they move in
    turn; picking makes the bearing hop between them. A fly does not pick here — LC10a are ~100
    columnar neurons a side, each looking at its own patch — so this does not either: the same motion
    image (`MovingTarget.motion`: the camera's own motion taken off, flicker, edge shimmer, parallax
    and blobs too big to be a small object removed), of each blob only its lowest `foot_deg` — where a
    walking person meets the floor (open loop, `experiments/population_dn.py`: two feet low in the view
    turn DNa02 as the glowing ball does, +12 Hz at 15 degrees; the whole legs with them, up to +30
    degrees, turn it by 0.3: too much of LC10a driven at once and the pathway no longer steers — it is
    a small-object pathway), pooled into cells, and each LC10a driven by the
    moving area under its receptive field, weighted as `FollowEncoder.rates_at` weights a point target
    (a Gaussian of `sigma_deg`). A moving disc of `disc_deg` at a receptive field's centre gives
    `max_rate`, as the point encoder does for a target there; more moving area does not give more. The
    drive decays with `tau_s` when the motion stops, so a foot resting between steps fades, not
    vanishes. Ours, all of it — what DNa02 makes of the two sides is the connectome's."""

    def __init__(self, rf, p: Params, pp: PopulationParams | None = None, fps: float = 30.0):
        self.rf, self.p, self.pp, self.fps = rf, p, pp or PopulationParams(), fps
        self.det = MovingTarget(hfov_deg=p.hfov_deg, fps=fps)
        self.grid = None
        self.drive = np.zeros(len(rf))
        self.mask = None
        self.shift = (0.0, 0.0)

    def _weights(self, h, w):
        """Per LC10a, the weight of each cell (deg^2), from its receptive field's centre on its own eye."""
        f = (w / 2) / math.tan(math.radians(self.p.hfov_deg / 2))
        c = max(1, int(round(f * math.radians(self.pp.cell_deg))))
        gh, gw = h // c, w // c
        xs = (np.arange(gw) + 0.5) * c; ys = (np.arange(gh) + 0.5) * c
        az = np.degrees(np.arctan((w / 2 - xs) / f)); el = np.degrees(np.arctan((h / 2 - ys) / f))
        cell_area = (np.degrees(c / f)) ** 2
        AZ, EL = np.meshgrid(az, el)                                     # image az (+ left), el (+ up)
        own = np.where(self.rf.side.to_numpy()[:, None] == 'left', AZ.ravel()[None], -AZ.ravel()[None])
        d2 = (self.rf.az.to_numpy()[:, None] - own) ** 2 + (self.rf.el.to_numpy()[:, None] - EL.ravel()[None]) ** 2
        W = np.exp(-d2 / (2 * self.p.sigma_deg ** 2)) * cell_area
        s, r = self.p.sigma_deg, self.pp.disc_deg
        ref = 2 * math.pi * s ** 2 * (1 - math.exp(-r ** 2 / (2 * s ** 2)))   # a disc of r at the centre
        self.grid = (c, gh, gw, W / ref, AZ, EL)

    def __call__(self, grey, shift=None):
        """Updates the drive; returns a summary for logs — the motion-weighted bearing of what drives
        LC10a (az, el, moving cells, 'moving') — or None when nothing does."""
        mo = self.det.motion(grey, shift)
        self.shift = self.det.shift
        k = math.exp(-1.0 / (self.fps * self.pp.tau_s))
        if mo is None:
            self.drive *= k
            return None
        m, st, ok, h, w = mo
        f = (w / 2) / math.tan(math.radians(self.p.hfov_deg / 2))
        turn_deg_s = math.degrees(math.atan(abs(self.det.shift[0]) / f)) * self.fps / self.det.p.lag
        if turn_deg_s > self.pp.suppress_deg_s:
            # the duck's own fast turn: what is left of it after the shift is taken off reads as motion
            # on one side, and a turn toward it feeds itself — the duck spins. Flies damp their vision
            # during saccades; here the drive only decays until the turn slows.
            self.drive *= k
            self.mask = np.zeros_like(m)
            return None
        keep = np.zeros_like(m, np.float32)
        foot_px = f * math.tan(math.radians(self.pp.foot_deg)) if self.pp.foot_deg > 0 else h
        for j in ok:                                       # each blob's lowest foot_deg: where it meets the floor
            x0, y0, bw, bh = (int(v) for v in st[j, :4])
            top = max(y0, int(y0 + bh - foot_px))
            keep[top:y0 + bh, x0:x0 + bw] = (self.det.labels[top:y0 + bh, x0:x0 + bw] == j)
        self.mask = keep.astype(np.uint8)
        if self.grid is None or self.grid[1] != h // self.grid[0]:
            self._weights(h, w)
        c, gh, gw, W, AZ, EL = self.grid
        frac = cv2.resize(keep[:gh * c, :gw * c], (gw, gh), interpolation=cv2.INTER_AREA).ravel()
        now = np.minimum(1.0, W @ frac)
        self.drive = np.maximum(now, self.drive * k)          # rises with the motion, decays after it
        if frac.sum() == 0:
            return None
        wt = frac / frac.sum()
        return float(wt @ AZ.ravel()), float(wt @ EL.ravel()), int((frac > 0).sum()), 'moving'

    def rates(self) -> np.ndarray:
        return self.p.max_rate * self.drive


class FollowEncoder:
    def __init__(self, params: Params | None = None, cell_type: str = 'LC10a', detector: str = 'white'):
        self.p = params or Params()
        self.rf = receptive_fields(cell_type)
        if detector == 'white':
            self.detect = lambda g: bearing(g, self.p)
        elif detector == 'moving':
            self.detect = MovingTarget(hfov_deg=self.p.hfov_deg)
        else:
            self.detect = PopulationMotion(self.rf, self.p)
        self.ids = [int(f) for f in self.rf.index]
        self.target: tuple[float, float, int, str] | None = None   # az, el (deg), area (px), kind, last frame

    def rates_at(self, az: float, el: float) -> dict[int, float]:
        own = np.where(self.rf.side == 'left', az, -az)
        d2 = (self.rf.az.to_numpy() - own) ** 2 + (self.rf.el.to_numpy() - el) ** 2
        r = self.p.max_rate * np.exp(-d2 / (2 * self.p.sigma_deg ** 2))
        return {f: float(x) if x > 1.0 else 0.0 for f, x in zip(self.ids, r)}

    def update(self, grey: np.ndarray, *_, camera_yaw_rate_deg: float | None = None,
               camera_turn_deg: float | None = None, **__) -> dict[int, float]:
        """Rates for every LC10a — zeros included, so a target that leaves stops driving.
        `camera_yaw_rate_deg` (+ = left; the gyro plus the head's own turn), or better `camera_turn_deg`,
        the camera's turn between the two frames the detector compares, gives the moving detector the
        camera's turn instead of an estimate from the image: turning left moves the room right."""
        if isinstance(self.detect, (MovingTarget, PopulationMotion)):
            shift = None
            d = self.detect.det if isinstance(self.detect, PopulationMotion) else self.detect
            f = (grey.shape[1] / 2) / math.tan(math.radians(d.hfov / 2))
            if camera_turn_deg is not None:
                shift = (f * math.tan(math.radians(camera_turn_deg)), None)
            elif camera_yaw_rate_deg is not None:
                # the turn from the gyro; the bob (dy) still from the image. A large target dominates an
                # image estimate of the sideways shift and is 'compensated' away; it moves little vertically.
                shift = (f * math.tan(math.radians(camera_yaw_rate_deg * d.p.lag / d.fps)), None)
            self.target = self.detect(grey, shift)
            if isinstance(self.detect, PopulationMotion):
                r = self.detect.rates()
                return {f: float(x) if x > 1.0 else 0.0 for f, x in zip(self.ids, r)}
        else:
            self.target = self.detect(grey)
        if self.target is None:
            return dict.fromkeys(self.ids, 0.0)
        return self.rates_at(self.target[0], self.target[1])
