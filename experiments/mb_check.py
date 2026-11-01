"""Does the mushroom body, in this model, code anything a duck could learn about? Mostly no — measured here.

    .venv/bin/python -I experiments/mb_check.py      # -> results/mb_check.json

  1. What reaches the Kenyon cells from the eyes: per KC type, the share of input from visual
     projection neurons (KCg-d, 30 %; KCab-p, 3 %; the others, ~0).
  2. What the duck's camera can drive gets there: drum motion on the camera's T4/T5, the follow
     target on LC10a, a 10-degree "card" on every columnar neuron, all 45k columnar neurons — the
     Kenyon cells and the visual projection neurons that feed them, how many fire.
  3. Is the code input-specific: four odours (one ORN type each) and taste, at 20 and 100 Hz —
     how many Kenyon cells fire, and the Jaccard overlap of the sets.
  4. Whether cutting KC -> KC synapses (38 % of their output) makes it specific.
  5. Where the runaway comes from: a random 15 % of Kenyon cells at 20 Hz, 10 seeds, intact vs the
     antennal lobe silenced (`learning.anosmic`).
Each stimulus 1 s, 3 seeds unless said.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from flysim import FlyBrain
from follow import FollowEncoder
from learning import MushroomBody, Params, anosmic
from motion import MotionEncoder
import neurons as N
import retinotopy as R

VIS_PN = ['aMe12', 'MTe32', 'MTe30', 'LTe25', 'MTe40', 'aMe20', 'aMe26', 'LTe16', 'LTe02', 'MTe38']


def mean_counts(brain, rates, seeds=3):
    acc = np.zeros(brain.n)
    for _ in range(seeds):
        b = brain.copy(); b.stimulate(rates); acc += b.run(1000) / seeds
    return acc


def main():
    t = N.table(); ct = t.cell_type.fillna(''); sc = t.super_class.fillna(''); cc = t.cell_class.fillna('')
    b0 = FlyBrain(783, seed=0); b0.run(1)
    ix = lambda m: np.array([b0.index[int(f)] for f in t.index[m] if int(f) in b0.index])
    ids = lambda m: [int(f) for f in t.index[m] if int(f) in b0.index]
    kc, vis = ix(ct.str.startswith('KC')), ix(ct.isin(VIS_PN))
    out = {}

    con = pd.read_parquet(R.CONNECTIVITY, columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    e = con[con.Postsynaptic_ID.isin(t.index[ct.str.startswith('KC')])]
    g = pd.DataFrame({'kc': ct.reindex(e.Postsynaptic_ID).to_numpy(), 'visual': sc.reindex(e.Presynaptic_ID).isin(
        ['visual_projection', 'visual_centrifugal', 'optic']).to_numpy(), 'w': e.Connectivity.to_numpy()})
    share = g.groupby('kc').apply(lambda x: x.w[x.visual].sum() / x.w.sum())
    out['visual_share'] = share.round(4).to_dict()
    print('1. visual share of Kenyon-cell input:', {k: f'{v:.1%}' for k, v in share.items() if v > 0.005})

    d = R.directions(); enc = MotionEncoder()
    sv = np.maximum((np.array([40.0, 0.0]) * enc.pref).sum(1) - 2, 0)
    own = np.where(d.hemisphere == 'left', 0.0, -0.0)
    card = ((d.az_deg - own) ** 2 + (d.el_deg + 10) ** 2) < 100
    stim = {'drum on camera T4/T5': {int(f): float(x) for f, x in zip(enc.ids, 60 * sv / (sv + 30))},
            'follow target on LC10a': {f: x for f, x in FollowEncoder().rates_at(10, -10).items() if x > 0},
            '10-deg card, all columnar': {int(f): 50.0 for f in d.index[card]},
            'all columnar, both eyes': {int(f): 30.0 for f in d.index}}
    out['camera'] = {}
    print('2. camera-shaped input -> Kenyon cells / their visual inputs:')
    for name, r in stim.items():
        c = mean_counts(b0, r)
        out['camera'][name] = {'kc': int((c[kc] > 0.5).sum()), 'vis_pn': int((c[vis] > 0.5).sum())}
        print(f'   {name:28s} KCs {out["camera"][name]["kc"]}/{len(kc)}, visual PNs {out["camera"][name]["vis_pn"]}/{len(vis)}')
    c = mean_counts(b0, {int(f): 100.0 for f in t.index[ct.isin(VIS_PN)]})
    out['vis_pn_direct'] = {'kcg_d': int((c[ix(ct == 'KCg-d')] > 0.5).sum())}
    print(f'   the visual PNs driven directly at 100 Hz: KCg-d {out["vis_pn_direct"]["kcg_d"]}/{(ct == "KCg-d").sum()}')

    smells = {'ORN_DM1': ids(ct == 'ORN_DM1'), 'ORN_DA1': ids(ct == 'ORN_DA1'), 'ORN_VA1d': ids(ct == 'ORN_VA1d'),
              'ORN_DL3': ids(ct == 'ORN_DL3'), 'taste': ids((sc == 'sensory') & (cc == 'gustatory'))}
    pre = np.repeat(np.arange(b0.n), np.diff(b0.indptr))
    is_kc = np.zeros(b0.n, bool); is_kc[kc] = True
    cut = b0.copy(); cut.weights = b0.weights.copy(); cut.weights[is_kc[pre] & is_kc[b0.indices]] = 0.0
    out['code'] = {}
    for brain_name, brain in (('intact', b0), ('KC->KC cut', cut)):
        for rate in (20, 100):
            sets, n_kc, hz = {}, {}, {}
            for name, f in smells.items():
                c = mean_counts(brain, {x: float(rate) for x in f})
                sets[name] = c[kc] > 0.5; n_kc[name] = int(sets[name].sum()); hz[name] = float(c.sum())
            big = [k for k in sets if n_kc[k] > 100]
            jac = [float((sets[a] & sets[b]).sum() / (sets[a] | sets[b]).sum()) for i, a in enumerate(big) for b in big[i + 1:]]
            out['code'][f'{brain_name}|{rate}'] = {'kcs': n_kc, 'brain_hz': hz, 'jaccard_min': min(jac) if jac else None}
            print(f'3/4. {brain_name:10s} {rate:3d} Hz: KCs firing {n_kc}; overlap of the non-empty sets >= '
                  f'{min(jac) if jac else float("nan"):.2f}; whole brain up to {max(hz.values()):.0f} spikes/s', flush=True)

    out['runaway'] = {}
    for name, brain in (('intact', b0), ('antennal lobe silenced', anosmic(b0))):
        n_run = 0
        for s in range(10):
            mb = MushroomBody(brain, Params(cue_fraction=0.15, cue_rate=20.0), seed=s)
            c = brain.copy(); c.stimulate(mb.cue('A')); k = c.run(1000)
            n_run += int((k[mb.kc] > 0).sum() - len(mb.cues['A']) > 500)
        out['runaway'][name] = n_run
        print(f'5. a random 15 % of Kenyon cells at 20 Hz, {name}: runaway in {n_run}/10')
    (ROOT / 'results' / 'mb_check.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
