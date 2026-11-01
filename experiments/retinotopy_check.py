"""Does the column map give LPLC2 what it is built for? Synthetic flow fields straight into
`motion.MotionEncoder` (no camera), then the whole brain.

Expansion centred in the view is looming: LPLC2 and the giant fiber should answer. Contraction is
the same edges moving the other way: they should not, or far less. Wide-field yaw rotation should
leave the giant fiber alone and flip DNa02/DNp15. `shuffled` repeats the rotation with the in-view
neurons' positions permuted — the old random assignment — as the control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from flysim import FlyBrain
from motion import MotionEncoder
import neurons as N

enc = MotionEncoder()
p = enc.params
cx, cy = p.width * p.scale / 2, p.height * p.scale / 2
deg_px = (p.hfov_deg / p.width) / p.scale
pos = np.stack([(enc.u - cx) * deg_px, (enc.v - cy) * deg_px], 1)   # deg from the image centre, v down
print(f'{enc.in_view} T4/T5 in view (of 11,820 with columns)')

def along(field, perm=None):
    f = field(pos if perm is None else pos[perm])
    return (f * enc.pref).sum(1)

SPEED = 40.0  # deg/s at 20 deg from the centre
fields = {
    'expansion': lambda x: x / 20.0 * SPEED,
    'contraction': lambda x: -x / 20.0 * SPEED,
    'yaw, world turning left (ccw)': lambda x: np.tile([-SPEED, 0.0], (len(x), 1)),   # image moves left
    'yaw, world turning right (cw)': lambda x: np.tile([SPEED, 0.0], (len(x), 1)),
}
rng = np.random.default_rng(0)
perm = rng.permutation(len(pos))

base = FlyBrain(783, seed=0); base.run(1)
ix = {k: [base.index[f] for f in N.ids(t, s)] for k, t, s in
      [('LPLC2 L', 'LPLC2', 'left'), ('LPLC2 R', 'LPLC2', 'right'), ('GF L', 'DNp01', 'left'), ('GF R', 'DNp01', 'right'),
       ('DNa02 L', 'DNa02', 'left'), ('DNa02 R', 'DNa02', 'right'), ('DNp15 L', 'DNp15', 'left'), ('DNp15 R', 'DNp15', 'right')]}
print(f"{'':34s}" + ''.join(f'{k:>9s}' for k in ix))
runs = [(n, f, None) for n, f in fields.items()] + [('shuffled: expansion', fields['expansion'], perm),
                                                    ('shuffled: yaw ccw', fields['yaw, world turning left (ccw)'], perm)]
for name, field, pm in runs:
    b = base.copy(); b.stimulate(enc._rates(along(field, pm))); c = b.run(1000)
    print(f'{name:34s}' + ''.join(f'{c[v].sum() / len(v):9.1f}' for v in ix.values()))
