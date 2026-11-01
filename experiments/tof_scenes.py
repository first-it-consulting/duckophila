"""Synthetic ToF scenes -> looming encoder -> whole brain. Does the giant fiber fire for threats,
on the right side, and stay quiet for noise, the duck's own walking, and things merely passing?

Beams are `kinematics/src/tof.rs`'s: 45 deg square FOV, row 0 top, column 0 the sensor's left,
sensor frame +x forward, +y left, +z up. 15 Hz, as tofd ranges.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from flysim import FlyBrain
from looming import LoomingEncoder, ROWS, COLS
import neurons as N

from tof_geom import HZ, BEAMS, render, approach

SCENES = {
    # name: (seconds, wall distance(t), sphere(t) or None, expect escape?)
    'static wall, noise':      (3, lambda t: 1.5,              None, False),
    'walk to wall 0.15 m/s':   (3, lambda t: 1.0 - 0.15 * t,   None, False),
    'walk to wall 0.3 m/s':    (3, lambda t: 1.2 - 0.3 * t,    None, False),
    'hand 0.4 m/s':            (3, lambda t: 1.5, approach([0.7, 0, 0], [0.15, 0, 0], 0.4, 0.06), None),
    'hand 0.6 m/s':            (3, lambda t: 1.5, approach([0.7, 0, 0], [0.15, 0, 0], 0.6, 0.06), None),
    'ball crossing at 40 cm':  (3, lambda t: 1.5, (lambda t: (np.array([0.4, 0.5 - 0.5 * t, 0.0]), 0.05)), False),
    'hand slow 0.2 m/s':       (3, lambda t: 1.5, approach([0.7, 0, 0], [0.15, 0, 0], 0.2, 0.06), False),
    'hand fast 1 m/s':         (3, lambda t: 1.5, approach([0.8, 0, 0], [0.15, 0, 0], 1.0, 0.06), True),
    'ball 1.5 m/s from left':  (3, lambda t: 1.5, approach([1.2, 0.3, 0], [0.15, 0.06, 0], 1.5, 0.05), True),
    'ball 1.5 m/s from right': (3, lambda t: 1.5, approach([1.2, -0.3, 0], [0.15, -0.06, 0], 1.5, 0.05), True),
}

base = FlyBrain(783, seed=0)
gf = {s: [base.index[f] for f in N.ids(N.GIANT_FIBER, s)] for s in ('left', 'right')}
dna = {s: [base.index[f] for t in N.STEER for f in N.ids(t, s)] for s in ('left', 'right')}
mdn = [base.index[f] for f in N.ids(N.BACKWARD)]

print(f"{'scene':26s} {'expect':>6s} {'GF L':>5s} {'GF R':>5s} {'1st GF':>7s} {'DNa L':>6s} {'DNa R':>6s} {'MDN':>4s}  max tau^-1")
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3):
    for name, (secs, wall, sphere, expect) in SCENES.items():
        rng = np.random.default_rng(seed)
        brain = base.copy()
        enc = LoomingEncoder(seed=0)
        tot = {k: 0 for k in ('GF L', 'GF R', 'DNa L', 'DNa R', 'MDN')}
        first = None; peak = 0.0
        for k in range(secs * HZ):
            t = k / HZ
            dist, status = render(wall(t), sphere(t) if sphere else None, rng)
            brain.stimulate(enc.update(dist, status, t))
            peak = max(peak, enc.inv_tau.max())
            c = brain.run(1000 / HZ)
            g = c[gf['left']].sum() + c[gf['right']].sum()
            if g and first is None:
                first = t
            tot['GF L'] += c[gf['left']].sum(); tot['GF R'] += c[gf['right']].sum()
            tot['DNa L'] += c[dna['left']].sum(); tot['DNa R'] += c[dna['right']].sum(); tot['MDN'] += c[mdn].sum()
        ok = expect is None or (tot['GF L'] + tot['GF R'] > 0) == expect
        print(f"{name:26s} {'?' if expect is None else 'yes' if expect else 'no':>6s} {tot['GF L']:5d} {tot['GF R']:5d} "
              f"{(f'{first:.2f}s' if first is not None else '-'):>7s} {tot['DNa L']:6d} {tot['DNa R']:6d} {tot['MDN']:4d}  {peak:5.1f}{'' if ok else '   <-- WRONG'}")
    print()
