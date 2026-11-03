"""Where the duck's own motion moves each pixel: a displacement field, for the follow encoder.

`motion.MotionEncoder.self_flow` predicts the flow of the duck's own motion for the fly's motion
detectors (the walking efference copy): translation moves only what has a known distance, the floor,
and rotation moves everything. The follow encoder needs the same, as a displacement in full-size
pixels over the interval between the two frames it compares, and with the turn about the vertical
included (the drum's efference copy leaves it out on purpose; here it is the largest part). With it,
the earlier frame is warped onto the later one and what is left is what moved by itself — feet, not
the floor sliding under the camera as the duck walks. Ours: geometry, not the fly's.

Camera axes as OpenCV and `motion.py`: x right, y down, z ahead. `ego` is `bridge.Body.ego()`:
(cam_rot camera->trunk, cam_pos in the trunk, world-down in the trunk, trunk height, velocity m/s in
the trunk frame, camera pitch/roll rate rad/s in the camera frame with the vertical turn taken out).
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class EgoWarp:
    def __init__(self, width: int = 360, height: int = 640, hfov_deg: float = 45.0, step: int = 8):
        """`step`: the field is computed every `step` pixels and interpolated — it is smooth."""
        self.w, self.h, self.step = width, height, step
        self.f = (width / 2) / math.tan(math.radians(hfov_deg / 2))
        gv, gu = np.mgrid[0:height:step, 0:width:step].astype(float) + step / 2
        self.gu, self.gv = gu, gv
        self.rays = np.stack([(gu - width / 2) / self.f, (gv - height / 2) / self.f, np.ones_like(gu)], -1)
        Y, X = np.mgrid[0:height, 0:width].astype(np.float32)
        self.X, self.Y = X, Y

    def field(self, ego, dt: float, turn_rad: float = 0.0) -> np.ndarray:
        """(h, w, 2) float32: how far (px) a static point at each pixel of the earlier frame moves by
        the later one, `dt` s on, the duck having turned `turn_rad` about the vertical (+ = left)."""
        cam_rot, cam_pos, down, height, v, w_pr = ego
        cam_rot = np.asarray(cam_rot, float); down = np.asarray(down, float)
        P = self.rays                                                # camera frame
        r = P @ cam_rot.T                                            # trunk frame
        cos = r @ down
        h = height - np.asarray(cam_pos, float) @ down               # camera above the floor
        hit = cos > 1e-3
        dist = np.where(hit, h / np.where(hit, cos, 1.0), 0.0)
        Q = P.copy()
        # translation: a floor point, relative to the camera, after the duck moved v * dt
        moved = (r * dist[..., None] - np.asarray(v, float) * dt) @ cam_rot
        Q = np.where(hit[..., None], moved, Q)
        # rotation: the camera turning by omega moves a static direction d by -omega x d
        up_cam = -(cam_rot.T @ down)
        omega = up_cam * turn_rad + (np.asarray(w_pr, float) * dt if w_pr is not None else 0.0)
        Q = Q - np.cross(omega, Q)
        with np.errstate(divide='ignore', invalid='ignore'):
            du = self.f * (Q[..., 0] / Q[..., 2] - P[..., 0] / P[..., 2])
            dv = self.f * (Q[..., 1] / Q[..., 2] - P[..., 1] / P[..., 2])
        d = np.nan_to_num(np.stack([du, dv], -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def field_between(self, ego0, r0: np.ndarray, r1: np.ndarray, dpos_world) -> np.ndarray:
        """As `field`, from the camera's actual poses at the two capture times: `r0`, `r1` camera->world
        rotations, `dpos_world` how far (m) the camera moved in the world between them. The gait rocks the
        camera at ~3 Hz, so a rotation rate at one instant times the interval is not the rotation over it."""
        cam_rot, cam_pos, down, height, _, _ = ego0
        cam_rot = np.asarray(cam_rot, float); down = np.asarray(down, float)
        P = self.rays
        r = P @ cam_rot.T
        cos = r @ down
        h = height - np.asarray(cam_pos, float) @ down
        hit = cos > 1e-3
        dist = np.where(hit, h / np.where(hit, cos, 1.0), 0.0)
        R = r1.T @ r0                                                # old camera frame -> new
        Q = P @ R.T                                                  # far points: rotation only
        floor = (P * dist[..., None]) @ r0.T - np.asarray(dpos_world, float)   # world, from the new camera
        Q = np.where(hit[..., None], floor @ r1, Q)
        with np.errstate(divide='ignore', invalid='ignore'):
            du = self.f * (Q[..., 0] / Q[..., 2] - P[..., 0] / P[..., 2])
            dv = self.f * (Q[..., 1] / Q[..., 2] - P[..., 1] / P[..., 2])
        d = np.nan_to_num(np.stack([du, dv], -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return cv2.resize(d, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def warp(self, old: np.ndarray, d: np.ndarray):
        """The earlier frame moved onto the later one by the field `d`, and where that is valid."""
        mx, my = self.X - d[..., 0], self.Y - d[..., 1]
        out = cv2.remap(old, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = ((mx >= 0) & (mx < self.w - 1) & (my >= 0) & (my < self.h - 1)).astype(np.uint8)
        return out, valid
