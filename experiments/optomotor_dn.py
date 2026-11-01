"""Optomotor go/no-go: does wide-field yaw motion on T4/T5 reach the steering DNs, with the sign
a fly would use?

World rotating clockwise seen from above (= the duck yawing left): the right eye sees
front-to-back motion (T4a/T5a), the left eye back-to-front (T4b/T5b). A fly turns *with* it — to
the right. DNa02 (Rayshubskiy et al. 2020) predicts ipsilateral turning, so expect DNa01/02 R > L.
Counter-clockwise is the mirror. Every descending neuron is read; none is wired by us.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from flysim import FlyBrain
import neurons as N

base = FlyBrain(783, seed=0)
t = N.table().reindex(base.ids)
dn = np.flatnonzero((t.super_class == 'descending').to_numpy())
lptc = {k: [base.index[f] for c in cs for s in [side] for f in N.ids(c, s)]
        for k, cs, side in [('HS L', ('HSN', 'HSE', 'HSS'), 'left'), ('HS R', ('HSN', 'HSE', 'HSS'), 'right'),
                            ('H2 L', ('H2',), 'left'), ('H2 R', ('H2',), 'right')]}

def motion(prog_side, rate):
    """front-to-back on `prog_side`, back-to-front on the other: a yaw rotation of the world"""
    other = 'left' if prog_side == 'right' else 'right'
    ids = [f for c in ('T4a', 'T5a') for f in N.ids(c, prog_side)] + [f for c in ('T4b', 'T5b') for f in N.ids(c, other)]
    return {f: rate for f in ids}

MS = 1000
rows = {}
for name, drive in [('cw  (world turns right)', lambda r: motion('right', r)),
                    ('ccw (world turns left)', lambda r: motion('left', r))]:
    for rate in (20, 50):
        b = base.copy(); b.stimulate(drive(rate)); c = b.run(MS)
        rows[(name, rate)] = c
        hs = {k: round(c[v].sum() * 1000 / MS / len(v), 1) for k, v in lptc.items()}
        print(f'{name} @{rate} Hz  active DNs: {(c[dn] > 0).sum():3d}  {hs}')

def dn_table(c):
    d = pd.DataFrame({'type': t.cell_type.to_numpy()[dn], 'side': t.side.to_numpy()[dn], 'hz': c[dn] * 1000 / MS})
    return d.pivot_table(index='type', columns='side', values='hz', aggfunc='mean', fill_value=0)

for rate in (20, 50):
    cw, ccw = dn_table(rows[('cw  (world turns right)', rate)]), dn_table(rows[('ccw (world turns left)', rate)])
    both = pd.concat({'cw': cw, 'ccw': ccw}, axis=1).fillna(0)
    both['asym'] = (both[('cw', 'right')] - both[('cw', 'left')]) - (both[('ccw', 'right')] - both[('ccw', 'left')])
    act = both[(both.filter(like='left').sum(1) + both.filter(like='right').sum(1)) > 0]
    print(f'\n@{rate} Hz: DN types that respond ({len(act)}), sorted by R-L flip between cw and ccw:')
    print(act.reindex(act.asym.abs().sort_values(ascending=False).index).head(15).round(1).to_string())
    for ty in ('DNa01', 'DNa02'):
        if ty in both.index:
            print(ty, both.loc[ty].round(1).to_dict())
