"""The composite readout's lean, per brain: how far it turns for a target dead ahead, per Hz of LC10a drive.

    .venv/bin/python -I experiments/lean.py          # -> results/lean.json, read by bridge.py --readout composite

`bridge.py --readout composite` turns on the left-right difference of the four DN pairs that follow a
target's bearing best (DNa02, DNae002, DNa16, and DNg97 with its sign turned). In the intact brain
that difference leans left whenever LC10a is driven at all — for a target dead ahead, in proportion
to how much — and a lean is a turn, so the bridge takes it off. The same procedure runs on every
brain the bridge can be given, so that a control is corrected by its own lean, not the intact one's:
synthetic pairs of moving feet dead ahead (2-8 degrees apart or wide, at -8, -14, -20 degrees of the fly's
elevation), encoded as `follow.PopulationMotion` encodes them (point targets, or by area), 1 s each, 4 seeds; the lean per Hz of drive is the least-squares
slope through zero. Open loop, nothing from the twin.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

import neurons as N
from flysim import FlyBrain
from follow import FollowEncoder

READOUT = [('DNa02', 1.0), ('DNae002', 1.0), ('DNa16', 1.0), ('DNg97', -1.0)]   # as bridge.READOUT['composite']


def lean(brain, seeds=4):
    e = FollowEncoder(detector='population'); pm = e.detect; pm._weights(640, 360)
    c, gh, gw, W, AZ, EL = pm.grid
    ix = [(sg, brain.index[N.ids(n, 'left')[0]], brain.index[N.ids(n, 'right')[0]]) for n, sg in READOUT]
    D, V = [], []
    for w in (2, 4, 6, 8):
        for el in (-8, -14, -20):
            if pm.pp.objects:          # as the encoder drives: each foot a point target (half-separation w)
                r = e.p.max_rate * np.maximum(pm._point(-w, el), pm._point(w, el))
            else:                      # by moving area, as the encoder did before
                m = np.zeros((gh, gw))
                for a0, a1 in ((-w - 2, -2), (2, w + 2)):
                    m[(AZ >= a0) & (AZ <= a1) & (EL >= el - 3) & (EL <= el)] = 1
                r = e.p.max_rate * np.minimum(1, W @ m.ravel())
            rates = {f: float(x) for f, x in zip(e.ids, r) if x > 1}
            v = []
            for s in range(seeds):
                b = brain.copy(); b.stimulate(rates); cc = b.run(1000)
                v.append(sum(sg * (cc[lft] - cc[rgt]) for sg, lft, rgt in ix))
            D.append(r.sum()); V.append(np.mean(v))
    D, V = np.array(D), np.array(V)
    k = float(D @ V / (D @ D))
    return {'k': k, 'residual_sd_hz': float(np.std(V - k * D)), 'dead_ahead_hz': V.tolist(), 'drive_hz': D.tolist()}


def main():
    intact = FlyBrain(783, seed=0); intact.run(1)
    brains = {'intact': intact, 'shuffled': intact.null('shuffled', seed=1)}   # as bridge.py --brain
    out = {}
    for name, b in brains.items():
        out[name] = lean(b)
        print(f'{name:9s} lean {out[name]["k"]:+.5f} Hz per Hz of LC10a drive (residual sd {out[name]["residual_sd_hz"]:.1f} Hz)', flush=True)
    (ROOT / 'results' / 'lean.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
