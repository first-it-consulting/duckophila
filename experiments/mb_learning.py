"""Can the duck learn to like what it is petted for? Open loop, the mushroom body with `learning.py`'s rule.

    .venv/bin/python -I experiments/mb_learning.py [n_seeds]      # -> results/mb_learning.json

Differential conditioning, as with flies: two cues, A and B, each a random sparse set of Kenyon
cells (new sets per seed). Before and after training, each is shown alone for 1 s and its value
read out (approach-MBON minus avoidance-MBON rate). Training, four rounds of:
  * paired    — A with petting (every PAM driven), then B alone
  * unpaired  — A alone, petting alone, B alone: the same cue and the same reward, never together
  * no reward — A alone, B alone
The learning index is A's change in value minus B's. Brains: intact and rewired (`FlyBrain.null`),
both with the antennal lobe silenced (`learning.anosmic`: the runaway's source; the duck has no nose).
Between windows the brain gets 0.5 s with no input.
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
from learning import MushroomBody, anosmic

MS = 1000.0
ROUNDS = 4


def trial(b, mb, cue=None, reward=False, learn=False):
    r = mb.off()
    if cue:
        r.update(mb.cue(cue))
    r.update(mb.reward(b, reward))
    b.stimulate(r)
    c = b.run(MS)
    if learn:
        mb.learn(b, c, MS)
    b.stimulate({**mb.off(), **mb.reward(b, False)})
    b.run(500)
    return c


def run(brain, protocol, seed):
    b = brain.copy()
    mb = MushroomBody(b, seed=seed)
    w0 = b.weights[mb.syn].copy()
    pre = {k: trial(b, mb, k) for k in 'AB'}
    for _ in range(ROUNDS):
        if protocol == 'paired':
            trial(b, mb, 'A', reward=True, learn=True); trial(b, mb, 'B', learn=True)
        elif protocol == 'unpaired':
            trial(b, mb, 'A', learn=True); trial(b, mb, None, reward=True, learn=True); trial(b, mb, 'B', learn=True)
        else:
            trial(b, mb, 'A', learn=True); trial(b, mb, 'B', learn=True)
    post = {k: trial(b, mb, k) for k in 'AB'}
    v = {f'{w}_{k}': mb.value(c, MS) for w, d in (('pre', pre), ('post', post)) for k, c in d.items()}
    driven = {k: set(int(b.index[int(f)]) for f in mb.cues[k]) for k in 'AB'}
    spill = max(int((c[mb.kc] > 0).sum()) - len(driven[k]) for k, c in post.items())
    mbon_change = ((post['A'][mb.mbon] - pre['A'][mb.mbon]) - (post['B'][mb.mbon] - pre['B'][mb.mbon])).tolist()
    return {**v, 'li': (v['post_A'] - v['pre_A']) - (v['post_B'] - v['pre_B']),
            'depressed': float(np.mean(b.weights[mb.syn] < 0.99 * w0)) if len(w0) else 0.0,
            'kc_spill': spill, 'brain_hz': float(post['A'].sum()), 'mbon_change': mbon_change}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    intact = FlyBrain(783, seed=0); intact.run(1)
    brains = {'intact': anosmic(intact), 'rewired': anosmic(intact.null('shuffled', seed=1))}
    mb0 = MushroomBody(brains['intact'])
    out = {'n_seeds': n, 'mbon_sign': mb0.sign.tolist(), 'runs': {}}
    t0 = time.time()
    print(f'{len(mb0.kc)} Kenyon cells, {len(mb0.mbon)} MBONs ({int((mb0.sign > 0).sum())} approach, '
          f'{int((mb0.sign < 0).sum())} avoid), {len(mb0.syn)} KC->MBON synapses', flush=True)
    for name, brain in brains.items():
        for protocol in ('paired', 'unpaired', 'no reward'):
            rows = [run(brain, protocol, s) for s in range(n)]
            out['runs'][f'{name}|{protocol}'] = rows
            m, lo, hi = ci_mean([r['li'] for r in rows])
            pa = ci_mean([r['post_A'] - r['pre_A'] for r in rows])
            pb = ci_mean([r['post_B'] - r['pre_B'] for r in rows])
            print(f'{name:8s} {protocol:9s} learning index {m:+6.2f} Hz [{lo:+.2f}, {hi:+.2f}]   '
                  f'A {pa[0]:+.2f}, B {pb[0]:+.2f}; synapses depressed {np.mean([r["depressed"] for r in rows]):.1%}; '
                  f'KC spill max {max(r["kc_spill"] for r in rows)}  ({time.time() - t0:.0f} s)', flush=True)
    (ROOT / 'results' / 'mb_learning.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
