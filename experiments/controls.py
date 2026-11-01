"""Is it the fly? The open-loop claims, with statistics, against two null brains.

    .venv/bin/python -I experiments/controls.py [n_seeds]      # -> results/controls.json, printed summary

Three brains, the same sensory translations, the same readouts:
  * intact    — FlyWire's wiring, as everywhere else
  * shuffled  — every connection keeps its source neuron and its weight but gets a random target
                (each neuron's in- and out-degree preserved: a configuration-model null)
  * flipped   — every sign inverted
A behaviour the nulls also show is ours, not the connectome's. Rewiring silences the brain (input
no longer converges), so two narrower nulls keep it intact and change one thing:
  * input null   — the startle encoder drives as many *other* visual projection neurons (same side,
                   same zones, same rates) instead of LPLC2/LC4: is the giant fibre's answer specific?
  * readout null — every left/right pair of descending neurons is scored for "following": are
                   DNp15 and DNa02 special, or does any DN pair do it?

Startle: a ball (r = 5 cm) approaches the ToF from 1.2 m at 0.2-2.0 m/s, from the left or the
right, on the same synthetic sensor as tof_scenes.py; plus four scenes that must not startle.
Per trial: did a giant fibre spike, when, and on which side.
Optomotor: yaw rotation (40 deg/s) on the T4/T5 that look into the camera, and on the whole eye;
per trial, DNp15's and DNa02's right-minus-left rate. A following turn needs it positive for a
world turning right and negative for one turning left.
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
from looming import LoomingEncoder
from motion import MotionEncoder, PREFERRED
from tof_geom import HZ, render, approach
import neurons as N
import retinotopy as R

SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
NEGATIVES = {
    'static wall, noise':     (lambda t: 1.5, None),
    'walk to wall 0.15 m/s':  (lambda t: 1.0 - 0.15 * t, None),
    'walk to wall 0.3 m/s':   (lambda t: 1.2 - 0.3 * t, None),
    'ball crossing at 40 cm': (lambda t: 1.5, lambda t: (np.array([0.4, 0.5 - 0.5 * t, 0.0]), 0.05)),
}
SECONDS = 3.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def ci_mean(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float('nan'),) * 3
    b = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


class RemappedLooming(LoomingEncoder):
    """The input null: the same encoder, its every LPLC2/LC4 replaced by a random other visual
    projection neuron of the same side — so the same zones drive as many neurons, the wrong ones."""

    def __init__(self, seed=0):
        super().__init__()
        t = N.table()
        rng = np.random.default_rng(seed)
        self.remap = {}
        for side in ('left', 'right'):
            pool = t.index[(t.super_class == 'visual_projection') & (t.side == side) & ~t.cell_type.isin(N.LOOMING)]
            mine = [f for z in self.zone_ids for f in z if t.side.get(f) == side]
            for f, g in zip(mine, rng.choice(pool, len(mine), replace=False)):
                self.remap[f] = int(g)

    def update(self, *a, **k):
        return {self.remap.get(f, f): r for f, r in super().update(*a, **k).items()}


def startle_trial(brain, gf, wall, sphere, seed, encoder=LoomingEncoder):
    rng = np.random.default_rng(1000 + seed)
    b = brain.copy()
    enc = encoder()
    first, n_l, n_r, spikes = None, 0, 0, 0
    for k in range(int(SECONDS * HZ)):
        t = k / HZ
        dist, status = render(wall(t), sphere(t) if sphere else None, rng)
        b.stimulate(enc.update(dist, status, t))
        c = b.run(1000 / HZ)
        spikes += int(c.sum())
        l, r = int(c[gf['left']].sum()), int(c[gf['right']].sum())
        n_l += l; n_r += r
        if (l or r) and first is None:
            first = t
    return {'startle': first is not None, 'latency': first, 'gf_left': n_l, 'gf_right': n_r, 'spikes': spikes}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    intact = FlyBrain(783, seed=0); intact.run(1)
    brains = {'intact': intact, 'shuffled': intact.null('shuffled', seed=1), 'flipped': intact.null('flipped')}
    gf = {s: [intact.index[f] for f in N.ids(N.GIANT_FIBER, s)] for s in ('left', 'right')}
    out = {'n_seeds': n, 'startle': {}, 'negatives': {}, 'optomotor': {}}
    t0 = time.time()

    startle_runs = dict(brains)
    startle_runs['input null'] = intact
    for name, brain in startle_runs.items():
        encoder = RemappedLooming if name == 'input null' else LoomingEncoder
        rows = []
        for speed in SPEEDS:
            for side, y in (('left', 0.3), ('right', -0.3)):
                sph = approach([1.2, y, 0], [0.15, y / 5, 0], speed, 0.05)
                for s in range(n):
                    r = startle_trial(brain, gf, lambda t: 1.5, sph, s, encoder)
                    rows.append({'speed': speed, 'side': side, 'seed': s, **r})
        out['startle'][name] = rows
        neg = {}
        for scene, (wall, sph) in NEGATIVES.items():
            neg[scene] = [startle_trial(brain, gf, wall, sph, s, encoder)['startle'] for s in range(n)]
        out['negatives'][name] = neg
        print(f'startle: {name} done ({time.time() - t0:.0f} s)', flush=True)

    enc = MotionEncoder()
    d = R.directions(); d = d[d.type.str.match(r'T[45][a-d]$')]
    whole = (list(d.index), np.array([PREFERRED[h][t[2]] for h, t in zip(d.hemisphere, d.type)], float))
    views = {'camera only': (enc.ids, enc.pref), 'whole eye': whole}
    dn = {k: {s: [intact.index[f] for f in N.ids(k, s)] for s in ('left', 'right')} for k in ('DNp15', 'DNa02', 'DNp01')}
    # the readout null: every descending type with exactly one neuron a side
    tt = N.table().reindex(intact.ids)
    desc = tt[tt.super_class == 'descending']
    per = desc.groupby(['cell_type', 'side']).size().unstack(fill_value=0)
    paired = [c for c in per.index if per.loc[c].get('left', 0) == 1 and per.loc[c].get('right', 0) == 1]
    pair_ix = {c: (intact.index[int(desc.index[(desc.cell_type == c) & (desc.side == 'left')][0])],
                   intact.index[int(desc.index[(desc.cell_type == c) & (desc.side == 'right')][0])]) for c in paired}
    for name, brain in brains.items():
        res = {}
        for view, (ids, pref) in views.items():
            for rot, u in (('world turns left', -40.0), ('world turns right', 40.0)):
                along = (np.array([u, 0.0]) * pref).sum(1)
                sv = np.maximum(along - 2.0, 0.0)
                rates = {int(f): float(x) for f, x in zip(ids, 60 * sv / (sv + 30))}
                trials = []
                for s in range(n):
                    b = brain.copy(); b.stimulate(rates); c = b.run(1000)
                    tr = {k: float(c[v['right']].sum() / len(v['right']) - c[v['left']].sum() / len(v['left']))
                          for k, v in dn.items()}
                    tr['gf_any'] = bool(c[dn['DNp01']['left']].sum() + c[dn['DNp01']['right']].sum())
                    tr['spikes'] = int(c.sum())
                    if name == 'intact':
                        tr['pairs'] = {cname: float(c[r] - c[l]) for cname, (l, r) in pair_ix.items()}
                    trials.append(tr)
                res[f'{view}|{rot}'] = trials
        out['optomotor'][name] = res
        print(f'optomotor: {name} done ({time.time() - t0:.0f} s)', flush=True)

    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'controls.json').write_text(json.dumps(out, default=float))
    summarize(out)


def summarize(out):
    n = out['n_seeds']
    print(f'\n=== startle: P(giant fibre spike) by approach speed, n = {2 * n} per cell (both sides), 95% Wilson CI')
    print(f"{'speed m/s':>10s}" + ''.join(f'{b:>24s}' for b in out['startle']))
    for sp in SPEEDS:
        cells = []
        for name, rows in out['startle'].items():
            k = sum(r['startle'] for r in rows if r['speed'] == sp)
            m = sum(1 for r in rows if r['speed'] == sp)
            p, lo, hi = wilson(k, m)
            cells.append(f'{k:3d}/{m:<3d} {p:4.2f} [{lo:.2f},{hi:.2f}]')
        print(f'{sp:10.1f}' + ''.join(f'{c:>24s}' for c in cells))
    print('\n=== startle laterality (fast approaches, >= 1.0 m/s, trials with a spike): ipsilateral GF share, mean [95% CI]')
    for name, rows in out['startle'].items():
        li = []
        for r in rows:
            if r['speed'] >= 1.0 and r['startle']:
                ip, co = (r['gf_left'], r['gf_right']) if r['side'] == 'left' else (r['gf_right'], r['gf_left'])
                li.append(ip / (ip + co))
        m, lo, hi = ci_mean(li)
        lat = [r['latency'] for r in rows if r['speed'] >= 1.0 and r['startle']]
        print(f'   {name:9s} ipsilateral share {m:.2f} [{lo:.2f}, {hi:.2f}] over {len(li)} trials; '
              f'latency {np.mean(lat) if lat else float("nan"):.2f} s')
    print('\n=== activity: mean spikes per startle trial (all neurons, 3 s), fast approaches')
    for name, rows in out['startle'].items():
        sp = [r['spikes'] for r in rows if r['speed'] >= 1.0]
        print(f'   {name:10s} {np.mean(sp):10.0f}')
    print(f'\n=== false startles, n = {n} per scene')
    for name, neg in out['negatives'].items():
        print(f'   {name:9s} ' + '   '.join(f'{scene}: {sum(v)}/{len(v)}' for scene, v in neg.items()))
    print('\n=== optomotor: right-minus-left rate (Hz), mean [95% CI]; following needs + for world-right, - for world-left')
    for name, res in out['optomotor'].items():
        for view in ('camera only', 'whole eye'):
            for k in ('DNp15', 'DNa02'):
                L = ci_mean([t[k] for t in res[f'{view}|world turns left']])
                Rr = ci_mean([t[k] for t in res[f'{view}|world turns right']])
                ok = 'FOLLOWS' if (L[2] < 0 and Rr[1] > 0) else '-'
                print(f'   {name:9s} {view:11s} {k}: world left {L[0]:+6.1f} [{L[1]:+.1f},{L[2]:+.1f}]  '
                      f'world right {Rr[0]:+6.1f} [{Rr[1]:+.1f},{Rr[2]:+.1f}]  {ok}')
        gfa = sum(t['gf_any'] for v in res.values() for t in v)
        sp = np.mean([t['spikes'] for v in res.values() for t in v])
        print(f'   {name:9s} giant fibre spikes during rotation: {gfa}/{sum(len(v) for v in res.values())} trials; '
              f'mean spikes per 1 s trial {sp:.0f}')
    res = out['optomotor']['intact']
    for view in ('camera only', 'whole eye'):
        L, Rr = res[f'{view}|world turns left'], res[f'{view}|world turns right']
        names = list(L[0]['pairs'])
        score, follow = {}, []
        for c in names:
            l = np.array([t['pairs'][c] for t in L]); r = np.array([t['pairs'][c] for t in Rr])
            score[c] = r.mean() - l.mean()
            lo_l, hi_l = ci_mean(l)[1:]; lo_r, hi_r = ci_mean(r)[1:]
            if hi_l < 0 and lo_r > 0:
                follow.append(c)
            elif lo_l > 0 and hi_r < 0:
                follow.append('-' + c)
        ranked = sorted(score, key=lambda c: -abs(score[c]))
        print(f'\n=== readout null, {view}: {len(names)} paired DN types; {len([f for f in follow if not f.startswith("-")])} follow, '
              f'{len([f for f in follow if f.startswith("-")])} oppose (95% CIs exclude 0 in both directions)')
        for c in ('DNp15', 'DNa02'):
            if c in score:
                print(f'   {c}: follow index {score[c]:+.1f} Hz, rank {ranked.index(c) + 1} of {len(ranked)} by |index|'
                      f'{"  (follows)" if c in follow else ""}')
        print('   top 6:', ', '.join(f'{c} {score[c]:+.0f}' for c in ranked[:6]))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        summarize(json.loads((ROOT / 'results' / 'controls.json').read_text()))
    else:
        main()
