"""Open-loop optomotor test on recorded duck-sim footage: camera -> motion.py -> brain -> DNa02.

The drum turns counter-clockwise for the first `period` seconds, then clockwise. A fly turns
with it, so expect DNa02 L > R in the first half and R > L in the second. Nothing is steering.

    .venv/bin/python -I experiments/optomotor_replay.py <recording.npz> [period_s] [--no-efference]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flysim import FlyBrain
from motion import MotionEncoder
import neurons as N

rec = np.load(sys.argv[1]); period = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 4.0
efference = '--no-efference' not in sys.argv
frames, ts, hyv = rec['frames'], rec['t'] - rec['t'][0], np.degrees(rec['head_yaw_vel'])
brain = FlyBrain(783, seed=0); brain.run(1)
enc = MotionEncoder()
ix = {f'{t}_{s[0].upper()}': [brain.index[f] for f in N.ids(t, s)] for t in ('DNa02', 'DNp15', 'HSE') for s in ('left', 'right')}
acc = {ph: {k: 0 for k in ix} for ph in ('ccw', 'cw')}; dur = {'ccw': 0.0, 'cw': 0.0}; us = {'ccw': [], 'cw': []}
prev = None
for g, t, w in zip(frames, ts, hyv):
    rates = enc.update(g, t, head_yaw_rate_deg=w if efference else 0.0)
    if prev is None:
        prev = t; continue
    ms = (t - prev) * 1000; prev = t
    brain.stimulate(rates); c = brain.run(ms)
    if t < 0.5 or abs(t - period) < 0.5:  # skip the onset and the reversal
        continue
    ph = 'ccw' if int(t / period) % 2 == 0 else 'cw'
    dur[ph] += ms; us[ph].append(enc.mean_u)
    for k, v in ix.items():
        acc[ph][k] += c[v].sum()
for ph in ('ccw', 'cw'):
    r = {k: round(v * 1000 / dur[ph] / len(ix[k]), 1) for k, v in acc[ph].items()}
    print(f"{ph:4s} mean image flow u {np.mean(us[ph]):+6.1f} deg/s   " + '  '.join(f'{k} {v:5.1f}' for k, v in r.items()))
