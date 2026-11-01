"""Does driving the looming detectors make the giant fiber fire — with a threshold, on the right side?

LPLC2 + LC4 on one side at a range of rates, and on a fraction of the population (a small object
loomes in part of the visual field). Reads GF (both sides) and the walking DNs. 783 connectome.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flysim import FlyBrain
import neurons as N

brain = FlyBrain(783, seed=0)
watch = {f'GF {s[0].upper()}': N.ids(N.GIANT_FIBER, s) for s in ('left', 'right')}
for t in (*N.STEER, N.FORWARD, N.BACKWARD):
    for s in ('left', 'right'):
        watch[f'{t} {s[0].upper()}'] = N.ids(t, s)

def trial(rate, frac=1.0, side='right', ms=500, seed=0):
    rng = np.random.default_rng(seed)
    b = FlyBrain.__new__(FlyBrain); b.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in brain.__dict__.items()}
    b._poi = {}
    src = [i for t in N.LOOMING for i in N.ids(t, side)]
    src = list(rng.choice(src, max(1, int(len(src) * frac)), replace=False))
    b.stimulate({f: rate for f in src})
    c = b.run(ms)
    return {k: b.rate(c, v, ms).mean() for k, v in watch.items()}, len(src)

side = sys.argv[1] if len(sys.argv) > 1 else 'right'
print('baseline (no drive):', {k: round(v, 1) for k, v in trial(0)[0].items() if v})
print(f"\n{'drive':>14s} " + ' '.join(f'{k:>9s}' for k in watch))
for frac in (1.0, 0.25, 0.1):
    for rate in (10, 20, 40, 80, 150):
        r, n = trial(rate, frac, side)
        print(f'{n:3d} {side[0].upper()} @{rate:3d} Hz ' + ' '.join(f'{v:9.1f}' for v in r.values()))
