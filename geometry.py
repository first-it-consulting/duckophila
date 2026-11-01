"""Which ToF zones see the floor, and how fast the duck's own walking closes each zone.

The floor test is `kinematics::tof::Reprojector::project`'s, ported, plus an angular margin
(`PITCH_MARGIN_DEG`) for pose error near the horizon, because a duck
that sways while standing pitches its head, and the floor then "approaches" — which the looming
encoder took for a threat in the first live run. Everything it needs comes from `robotd`, not from
a transcription of the robot: `robot.model` (beam directions, rest trunk height, asked once) and
each `robot.state` tick (`frames.tof`: the sensor's pose in the trunk from the head FK; `imu.quat`
for which way is down; `odom.position[2]`: measured trunk height).

The ego-motion term is the efference copy: `move.applied` is the twist the daemon is actually
executing, so a static point along a beam closes at `v . beam` — that much approach is the duck's
own doing and is subtracted before it can look like a threat. Yaw rate is ignored for now.
"""
from __future__ import annotations

import math

import numpy as np

FLOOR_SAFETY = 0.85     # kinematics::tof::Reprojector::FLOOR_SAFETY
# The pose is a few degrees off: the head FK put the sim camera's horizon 2-3 degrees from where it
# rendered. At a steep beam that is nothing; at a beam grazing the floor near the horizon it is the
# whole answer — 3 degrees moves where a 5-degree beam lands from 2.5 m to 1.6 m, and the sway of a
# standing head then reads as a floor rushing in (5 startles in 25 s with nothing in the room).
# So a beam is floor if its range fits the floor at up to this much more depression.
PITCH_MARGIN_DEG = 3.0


def rotate(q, v):
    """Rotate v (…, 3) by the scalar-first unit quaternion q = [w, x, y, z]."""
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return np.asarray(v) @ r.T


def quat_matrix(q) -> np.ndarray:
    """The rotation matrix of a scalar-first quaternion (normalised here)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return np.array([rotate(q, e) for e in np.eye(3)]).T


def rotation_rate(r0: np.ndarray, r1: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity (rad/s, the frame r0 and r1 are expressed in) taking orientation r0 to r1."""
    d = r1 @ r0.T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(d) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (2 * math.sin(angle))
    return axis * angle / dt


class Geometry:
    def __init__(self, model: dict):
        self.beams = np.array(model['tof_beams'], float)          # sensor frame, row-major
        self.trunk_height = float(model['trunk_height_m'])

    def _down_in_trunk(self, state):
        imu = state.get('imu') or {}
        q = imu.get('quat')
        if not q or np.linalg.norm(q) < 0.5:
            return np.array([0.0, 0.0, -1.0])
        q = np.asarray(q, float) / np.linalg.norm(q)
        # trunk -> world is q, so world down in the trunk is the inverse rotation of (0, 0, -1)
        return rotate(q * [1, -1, -1, -1], [0.0, 0.0, -1.0])

    def beams_in_trunk(self, state):
        tof = (state.get('frames') or {}).get('tof')
        if not tof:
            return None, None
        return rotate(np.asarray(tof['quat'], float), self.beams), np.asarray(tof['pos'], float)

    def floor(self, ranges_m, state) -> np.ndarray | None:
        """True where the zone's return is the floor. None without a pose to judge by."""
        dirs, pos = self.beams_in_trunk(state)
        if dirs is None:
            return None
        down = self._down_in_trunk(state)
        h = (state.get('odom') or {}).get('position', [0, 0, 0])[2] or self.trunk_height
        above_floor = h + float(pos @ -down)
        downward = dirs @ down
        steeper = np.sin(np.arcsin(np.clip(downward, -1, 1)) + np.radians(PITCH_MARGIN_DEG))
        r = np.nan_to_num(np.asarray(ranges_m, float), nan=0.0)
        return (above_floor > 0) & (steeper > 0) & (r * steeper >= above_floor * FLOOR_SAFETY)

    def ego_closing(self, state) -> np.ndarray:
        """m/s each zone closes on a static world purely because the duck is walking."""
        dirs, _ = self.beams_in_trunk(state)
        applied = (state.get('move') or {}).get('applied') or [0.0, 0.0, 0.0]
        if dirs is None:
            return np.zeros(len(self.beams))
        return dirs[:, :2] @ np.array(applied[:2], float)
