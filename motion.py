"""Camera frames -> Poisson rates on the fly's motion detectors (T4/T5). Step 2's translation.

This is the hand-written part, like `looming.py`: everything here is ours, not the fly's.

**What T4/T5 are.** Each eye has ~750 of each of eight types: T4 sees moving ON edges, T5 OFF
edges, and each comes in four preferred directions — a: front-to-back, b: back-to-front,
c: upward, d: downward (Maisak et al. 2013). The connectome takes it from there: HS cells, H2,
DNp15/DNHS1, and DNa02 — whose left/right difference is what steers (`experiments/optomotor_dn.py`) —
and LPLC2, whose four arms each want *outward* motion around its own patch of the field.

**Retinotopy, real.** Every T4/T5 is on a visual column (FlyWire's column map), and
`retinotopy.py` turns its column into a viewing direction. A neuron whose direction falls inside
the camera's view is driven by the optic flow *at that spot*, averaged over about one column
(4.8 degrees), projected onto its own preferred direction. A neuron looking anywhere else gets
nothing — the camera sees 45 x 73 degrees straight ahead, about one in eight of the fly's motion
detectors. (The first version drew T4/T5 onto image cells at random. Wide-field motion reached
LPLC2's arms scrambled, and the drum fired the giant fiber now and then: spurious startles.)

**Directions in the image.** Image left is the duck's left (checked: a ball at the duck's left lands
on the left of the frame). Front-to-back for the left eye is leftward image motion wherever that
eye looks, including the strip it sees across the midline; for the right eye it is rightward.
Up is up in both.

**Signal.** Dense optic flow (Farneback) on a quarter-size grey image, in degrees per second (the
sim camera: 45 degrees across the 360-pixel axis, about 0.125 degrees per pixel at the centre).

**Simplifications, said out loud.** T4 and T5 are driven alike (no ON/OFF split). The camera is
the eye: no separate head-to-eye geometry, and the lattice is laid flat (`retinotopy.py`).

**Efference copies.** Image motion the duck causes on purpose is subtracted before it reaches the
fly, so it cannot be mistaken for the world moving:
  * *Head turns* by `robotd`'s idle look-around: a head turning at w (deg/s, positive left) shifts
    the whole image rightward by w — measured in duck-sim as u = 0.94 w.
  * *Walking.* A walking duck's view is mostly its own doing: the floor streams outward as it
    moves, and — measured, the larger part — the whole image bobs and sways with the gait's pitch
    and roll. With the duck walking the giant fiber fired from it now and then. Both are predicted
    per pixel and subtracted (`self_flow`): translation from odometry (which tracks the
    simulator's truth to ~10%) on pixels that see the floor (camera pose `frames.camera`, height
    from `robot.state`, "down" from the IMU) — a wall's distance is not known to the camera, and
    the drum moves with the duck — and the camera's rotation, from its orientation in the world
    state to state, with the turn about the vertical taken out: yaw slip is what the reflex
    steers by, and its own turning is the loop. Each prediction is subtracted only where the
    image has edges to show it (`texture`): on a blank wall the estimator, like a T4, sees nothing
    move, and subtracting there would invent motion the other way.
Turning caused by the optomotor reflex itself is *not* subtracted, head or body: that slip is
exactly what the loop is there to null, and a fly does not cancel its reflexes either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import retinotopy as R

VIEW_ROWS, VIEW_COLS = 8, 4      # per half-image, only for the view's flow arrows
# preferred direction in image axes (u right, v down), per eye
PREFERRED = {'left': {'a': (-1, 0), 'b': (1, 0), 'c': (0, -1), 'd': (0, 1)},
             'right': {'a': (1, 0), 'b': (-1, 0), 'c': (0, -1), 'd': (0, 1)}}


@dataclass
class Params:
    width: int = 360             # upright frame, full resolution
    height: int = 640
    hfov_deg: float = 45.0       # across `width`
    scale: float = 0.25          # image downsampling before flow
    s_min: float = 2.0           # deg/s: flow below this drives nothing (rendering noise, jitter)
    s_half: float = 30.0         # deg/s at half of max_rate
    max_rate: float = 60.0       # Hz; ~50 Hz on a+b gave a clean DNa02 flip
    # Where the image has no edges the estimator sees no motion, whatever moves — and T4/T5 see
    # none either. A predicted self-motion flow is subtracted only as far as the image there could
    # show it: weight g / (g + texture_half), g the local squared gradient (per pixel, 0-255 grey).
    # (Averaged over a whole frame this looks like the estimator reading speed low — 0.63 of a
    # drum's, 0.70 of a head nod's — but on edges it reads them right; the blank parts read zero.)
    texture_half: float = 100.0
    flow_gain: float = 1.0


@dataclass
class MotionEncoder:
    params: Params = field(default_factory=Params)

    def __post_init__(self):
        p = self.params
        d = R.directions()
        d = d[d.type.str.match(r'T[45][a-d]$')]
        f = (p.width / 2) / math.tan(math.radians(p.hfov_deg / 2))
        az = np.radians(np.where(d.hemisphere == 'left', d.az_deg, -d.az_deg))   # + is the duck's left
        el = np.radians(d.el_deg.to_numpy())
        ok = np.abs(az) < math.radians(80)
        u = p.width / 2 - f * np.tan(az)
        v = p.height / 2 - f * np.tan(el) / np.cos(az)
        ok &= (u >= 0) & (u < p.width) & (v >= 0) & (v < p.height)
        d = d[ok]
        self.ids = [int(i) for i in d.index]
        self.u = (u[ok] * p.scale).astype(int)
        self.v = (v[ok] * p.scale).astype(int)
        self.pref = np.array([PREFERRED[h][t[2]] for h, t in zip(d.hemisphere, d.type)], float)
        self.column_px = max(3, int(round(R.DEG_PER_COLUMN / (p.hfov_deg / p.width) * p.scale)) | 1)
        # every quarter-size pixel's ray in the camera frame (OpenCV: x right, y down, z ahead)
        hs, ws = int(p.height * p.scale), int(p.width * p.scale)
        self.f_small = f * p.scale
        gv, gu = np.mgrid[0:hs, 0:ws].astype(float)
        self.rays = np.stack([(gu - ws / 2) / self.f_small, (gv - hs / 2) / self.f_small, np.ones_like(gu)], -1)
        self.ego_deg = None       # last predicted self-motion flow, deg/s, for the view
        self._prev = None
        self._t = None
        self._last: dict[int, float] = {}
        self.flow_deg = None      # (2, VIEW_ROWS, VIEW_COLS, 2) coarse flow per half-image, deg/s, for the view
        self.mean_u = 0.0

    @property
    def in_view(self) -> int:
        return len(self.ids)

    def self_flow(self, cam_rot, cam_pos, down, height, v, w_cam=None) -> np.ndarray:
        """Flow (deg/s as the estimator reads it, image axes) the duck's own motion makes at every pixel.

        Translation at `v` (m/s, trunk frame) moves only what has a known distance — the floor:
        `cam_rot` takes camera to trunk, `cam_pos` is the camera in the trunk, `down` world-down in
        the trunk, `height` the trunk above the floor. Rotation at `w_cam` (rad/s, camera frame)
        moves everything, whatever its distance. Both are predicted by moving each pixel's ray and
        re-projecting it."""
        p = self.params
        dt = 0.01
        rays = self.rays
        Pc = rays.copy()                                           # a direction per pixel, camera frame
        r = rays @ cam_rot.T                                       # rays, trunk frame
        cos = r @ down
        h = height - cam_pos @ down                                # camera above the floor
        hit = cos > 1e-3
        dist = np.where(hit, h / np.where(hit, cos, 1.0), 0.0)
        # translation: floor points, relative to the camera, a moment later
        P = r * dist[..., None]
        Q = np.where(hit[..., None], (P - np.asarray(v) * dt) @ cam_rot, Pc)
        # rotation: the camera turning by w moves every static direction by -w x d
        if w_cam is not None:
            Q = Q - np.cross(np.asarray(w_cam), Q) * dt
        with np.errstate(divide='ignore', invalid='ignore'):
            du = self.f_small * (Q[..., 0] / Q[..., 2] - Pc[..., 0] / Pc[..., 2]) / dt
            dv = self.f_small * (Q[..., 1] / Q[..., 2] - Pc[..., 1] / Pc[..., 2]) / dt
        flow = np.stack([du, dv], -1) * ((p.hfov_deg / p.width) / p.scale) * p.flow_gain
        return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    def texture(self, small: np.ndarray) -> np.ndarray:
        """0..1 per pixel: how well the flow estimator can see motion there (local edge energy)."""
        f = small.astype(np.float32)
        g = cv2.Sobel(f, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(f, cv2.CV_32F, 0, 1) ** 2
        g = cv2.blur(g, (self.column_px, self.column_px))
        return g / (g + self.params.texture_half)

    def update(self, grey: np.ndarray, t: float, head_yaw_rate_deg: float = 0.0, ego=None) -> dict[int, float]:
        """One frame (H x W uint8, upright) -> rates for every in-view T4/T5 neuron. `t` in seconds.
        `ego`: (cam_rot, cam_pos, down, height, v, w_cam) for the walking efference copy, or None."""
        p = self.params
        small = cv2.resize(grey, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)
        if self._prev is not None and small.shape == self._prev.shape and np.array_equal(small, self._prev):
            # the same render handed out twice: no new information, and differencing it would read
            # as zero flow followed by double flow
            return self._last
        if self._prev is None or small.shape != self._prev.shape:
            self._prev, self._t = small, t
            self._last = self._rates(np.zeros(len(self.ids)))
            return self._last
        dt = max(t - self._t, 1e-3)
        flow = cv2.calcOpticalFlowFarneback(self._prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self._prev, self._t = small, t
        deg = flow * ((p.hfov_deg / p.width) / p.scale) / dt        # deg/s, image axes (u right, v down)
        deg[..., 0] -= head_yaw_rate_deg                             # efference copy: head turns are ours
        if ego is not None:                                          # efference copy: so is walking
            self.ego_deg = self.self_flow(*ego) * self.texture(small)[..., None]
            deg -= self.ego_deg
        local = cv2.blur(deg, (self.column_px, self.column_px))      # about one column's worth
        along = (local[self.v, self.u] * self.pref).sum(1)           # motion along each neuron's preference
        self._view(deg)
        self._last = self._rates(along)
        return self._last

    def _view(self, deg):
        h, w = deg.shape[:2]
        cells = np.zeros((2, VIEW_ROWS, VIEW_COLS, 2))
        for e, half in enumerate((deg[:, : w // 2], deg[:, w // 2:])):
            hh, ww = half.shape[:2]
            for r in range(VIEW_ROWS):
                for c in range(VIEW_COLS):
                    blk = half[r * hh // VIEW_ROWS:(r + 1) * hh // VIEW_ROWS, c * ww // VIEW_COLS:(c + 1) * ww // VIEW_COLS]
                    cells[e, r, c] = blk.reshape(-1, 2).mean(0)
        self.flow_deg = cells
        self.mean_u = float(deg[..., 0].mean())

    def _rates(self, along) -> dict[int, float]:
        p = self.params
        s = np.maximum(along - p.s_min, 0.0)
        r = p.max_rate * s / (s + p.s_half)
        return {f: float(x) for f, x in zip(self.ids, r)}
