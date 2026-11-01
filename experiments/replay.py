"""Replay Brian2's own sugar spike times (one trial) into flysim; downstream is deterministic,
so every other neuron's spikes should match Brian2's exactly if the dynamics are the same."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from flysim import FlyBrain, DT
from compare_ids import SUGAR

ref = pd.read_parquet(Path(__file__).resolve().parents[1] / 'results' / 'sugarR.parquet')
tr = ref[ref.trial == 0]
b = FlyBrain(630)
sug = {b.index[f] for f in SUGAR}
steps = {}
for f, t in zip(tr.flywire_id, tr.t):
    i = b.index[int(f)]
    if i in sug:
        steps.setdefault(round(t / 1e-4), []).append(i)
for i in sug:
    b.rfc_steps[i] = 0
ours = []
for k in range(10000):
    for i in steps.get(k, ()):
        b.v[i] = -40.0
        if not b.is_act[i]:
            b.is_act[i] = True; b.act[b.n_act[0]] = i; b.n_act[0] += 1
    c = b.run(DT)
    ours += [(int(b.ids[i]), k) for i in np.flatnonzero(c)]
ours = pd.DataFrame(ours, columns=['fid', 'step'])
theirs = pd.DataFrame({'fid': tr.flywire_id.astype('int64'), 'step': (tr.t / 1e-4).round().astype(int)})
o = set(map(tuple, ours.values)); t = set(map(tuple, theirs.values))
print('spikes brian2', len(t), ' flysim', len(o), ' identical', len(o & t))
first = sorted(o ^ t, key=lambda x: x[1])[:5]
print('first differences (fid, step, in):', [(f, s, 'flysim' if (f, s) in o else 'brian2') for f, s in first])
