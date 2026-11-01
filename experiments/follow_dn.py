"""Follow me, open loop: a small target at a bearing -> LC10a -> which descending neurons turn toward it?

    .venv/bin/python -I experiments/follow_dn.py [n_seeds]      # -> results/follow_dn.json

FlyWire is a female brain: no P1, so no courtship pursuit. What it has is the visual object pathway
that pursuit is built on — LC10a, a lobula columnar type that answers small moving objects
(Ribeiro et al. 2018; Hindmarsh Sten et al. 2021), into the anterior optic tubercle, from where
AOTU019 (GABA) and AOTU025 (ACh) reach DNa02, the steering DN.

Where each LC10a looks, and the encoder from a bearing to LC10a rates, are `follow.py`'s — the
same the duck uses closed loop. The target sits at -10 degrees elevation, inside the camera.

Brains and nulls, as in controls.py: intact, shuffled; and an input null — the same rates on as many
random *other* visual projection neurons of the same side. Readout: DNa02's left-minus-right rate,
and every DN pair with one neuron a side scored by how its left-minus-right follows the bearing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from flysim import FlyBrain
from controls import ci_mean
from follow import FollowEncoder, receptive_fields
import neurons as N

BEARINGS = (-20.0, -10.0, 0.0, 10.0, 20.0)   # inside the camera's +-22.5 deg
EL = -10.0
ELEVATIONS = (15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0)
MS = 1000.0
ENC = FollowEncoder()      # 120 Hz, sigma 10 deg: the dose is chosen in follow.py


def rates(az: float, el: float = EL) -> dict[int, float]:
    return {f: x for f, x in ENC.rates_at(az, el).items() if x > 0}


def remap(seed=0) -> dict[int, int]:
    t = N.table()
    rng = np.random.default_rng(seed)
    rf = receptive_fields()
    out = {}
    for side in ('left', 'right'):
        pool = t.index[(t.super_class == 'visual_projection') & (t.side == side) & ~t.cell_type.str.startswith('LC10', na=False)]
        mine = rf.index[rf.side == side]
        out.update({int(f): int(g) for f, g in zip(mine, rng.choice(pool, len(mine), replace=False))})
    return out


def dn_pairs(brain):
    tt = N.table().reindex(brain.ids)
    desc = tt[tt.super_class == 'descending']
    per = desc.groupby(['cell_type', 'side']).size().unstack(fill_value=0)
    paired = [c for c in per.index if per.loc[c].get('left', 0) == 1 and per.loc[c].get('right', 0) == 1]
    return {c: (brain.index[int(desc.index[(desc.cell_type == c) & (desc.side == 'left')][0])],
                brain.index[int(desc.index[(desc.cell_type == c) & (desc.side == 'right')][0])]) for c in paired}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    intact = FlyBrain(783, seed=0); intact.run(1)
    pairs = dn_pairs(intact)
    names = list(pairs)
    L = np.array([pairs[c][0] for c in names]); Rr = np.array([pairs[c][1] for c in names])
    runs = {'intact': (intact, None), 'shuffled': (intact.null('shuffled', seed=1), None), 'input null': (intact, remap())}
    print(f'LC10a with a receptive field: {len(receptive_fields())}; DN pairs: {len(names)}', flush=True)
    out = {'n_seeds': n, 'bearings': BEARINGS, 'pairs': names, 'runs': {}}
    t0 = time.time()
    for run, (brain, rm) in runs.items():
        diff = np.zeros((len(BEARINGS), n, len(names)))
        for i, az in enumerate(BEARINGS):
            r = rates(az)
            if rm:
                r = {rm[f]: x for f, x in r.items()}
            for s in range(n):
                b = brain.copy()
                b.stimulate(r)
                c = b.run(MS)
                diff[i, s] = (c[L] - c[Rr]) * 1000.0 / MS
        x = np.repeat(np.array(BEARINGS), n)
        y = diff.reshape(-1, len(names))
        # how each pair's left-minus-right follows the bearing (+ = left): a slope, Hz per 10 deg, and r
        slope = np.array([np.polyfit(x, y[:, k], 1)[0] * 10 for k in range(len(names))])
        corr = np.array([np.corrcoef(x, y[:, k])[0, 1] if y[:, k].std() > 0 else 0.0 for k in range(len(names))])
        k = names.index('DNa02')
        rank = int((corr > corr[k]).sum()) + 1
        per_bearing = {str(az): ci_mean(diff[i, :, k]) for i, az in enumerate(BEARINGS)}
        order = np.argsort(-np.abs(corr))[:15]
        out['runs'][run] = {'dna02_LminusR': per_bearing, 'dna02_corr': float(corr[k]), 'dna02_rank': rank,
                            'n_follow': int((corr > 0.5).sum()), 'n_against': int((corr < -0.5).sum()),
                            'top': [(names[j], float(corr[j]), float(slope[j])) for j in order]}
        print(f'\n== {run} ({time.time() - t0:.0f} s)')
        for az, (m, lo, hi) in per_bearing.items():
            print(f'  bearing {float(az):+5.0f}  DNa02 L-R {m:+6.1f} Hz [{lo:+.1f}, {hi:+.1f}]')
        print(f'  DNa02 r = {corr[k]:+.2f}, rank {rank}/{len(names)}; pairs with r > 0.5: {(corr > 0.5).sum()}, r < -0.5: {(corr < -0.5).sum()}')
        for j in order[:10]:
            print(f'    {names[j]:20s} r {corr[j]:+.2f}  {slope[j]:+.1f} Hz/10deg')
    # where in the field the pathway steers: the turn toward a target 10 deg out, by elevation
    k = names.index('DNa02')
    out['elevation'] = {}
    print('\n== intact, DNa02 toward-turn (half the L-R difference between +10 and -10 deg) by elevation')
    for el in ELEVATIONS:
        toward = np.zeros(n)
        for sgn in (1, -1):
            r = rates(sgn * 10.0, el)
            for s in range(n):
                b = intact.copy(); b.stimulate(r); c = b.run(MS)
                toward[s] += sgn * (c[L[k]] - c[Rr[k]]) * 1000.0 / MS / 2
        out['elevation'][str(el)] = ci_mean(toward)
        m, lo, hi = out['elevation'][str(el)]
        print(f'  elevation {el:+5.0f}  {m:+5.1f} Hz [{lo:+.1f}, {hi:+.1f}]', flush=True)
    (ROOT / 'results' / 'follow_dn.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
