"""Synthetic ToF frames: the sensor's 64 beams ray-cast against a wall and a sphere, with noise.

Beams are `kinematics/src/tof.rs`'s: 45 deg square FOV, row 0 top, column 0 the sensor's left,
sensor frame +x forward, +y left, +z up. 15 Hz, as tofd ranges. Shared by tof_scenes.py and
controls.py, so both see exactly the same sensor.
"""
import math

import numpy as np

ROWS = COLS = 8
HZ = 15
FOV = 45.0
half = math.radians(FOV / 2 - FOV / COLS / 2)
step = 2 * half / (COLS - 1)
BEAMS = np.array([[math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
                  for i in range(ROWS * COLS)
                  for e, a in [(half - (i // COLS) * step, half - (i % COLS) * step)]])

def render(wall_x, sphere=None, rng=None):
    """Slant range per beam to a wall at x=wall_x and an optional sphere (center, radius)."""
    d = wall_x / BEAMS[:, 0]
    if sphere is not None:
        c, r = np.asarray(sphere[0], float), sphere[1]
        b = BEAMS @ c
        disc = b * b - (c @ c - r * r)
        hit = (disc >= 0) & (b > 0)
        t = b - np.sqrt(np.where(hit, disc, 0))
        d = np.where(hit & (t > 0), np.minimum(d, t), d)
    if rng is not None:
        d = d + rng.normal(0, 0.005 + 0.01 * d)
    status = np.where(d <= 4.0, 5, 255)
    return (d * 1000).round().astype(int), status

def approach(start, end, speed, r):
    """A sphere moving in a straight line at `speed` m/s, holding at `end`."""
    start, end = np.array(start, float), np.array(end, float)
    L = np.linalg.norm(end - start)
    return lambda t: (start + (end - start) * min(1.0, speed * t / L), r)
