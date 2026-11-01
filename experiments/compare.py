"""flysim vs the Brian2 reference: sugar GRNs at 150 Hz, rates over N x 1 s trials, and speed."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from flysim import FlyBrain

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
         720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
         720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
         720575940611875570]
MN9 = 720575940660219265
n_run = int(sys.argv[1]) if len(sys.argv) > 1 else 6

t0 = time.time(); brain = FlyBrain(630, seed=1); print(f'load: {time.time()-t0:.1f} s')
brain.run(1)  # jit
counts = np.zeros(brain.n)
for k in range(n_run):
    b = FlyBrain(630, seed=k)  # fresh state per trial, as the reference does
    b.stimulate({f: 150 for f in SUGAR})
    t0 = time.time()
    for _ in range(50):           # 50 x 20 ms: the closed-loop shape
        counts += b.run(20)
    if k == 0: print(f"1 s of brain in 20 ms steps: {time.time()-t0:.2f} s wall, {b.active} active at end")
ours = pd.Series(counts / n_run, index=brain.ids)

ref = pd.read_parquet(Path(__file__).resolve().parents[1] / 'results' / 'sugarR.parquet')
n_ref = ref['trial'].nunique()
theirs = ref.groupby('flywire_id').size() / n_ref
both = pd.concat({'brian2': theirs, 'flysim': ours[ours > 0]}, axis=1).fillna(0)
print('active neurons  brian2:', (theirs > 0).sum(), ' flysim:', (ours > 0).sum())
print('MN9 Hz          brian2:', theirs.get(MN9, 0), ' flysim:', ours[MN9])
print('rate correlation (active in either):', round(np.corrcoef(both['brian2'], both['flysim'])[0, 1], 3))
top = both.sort_values('brian2', ascending=False).head(12)
print(top.round(1))

mid = both[(both.brian2 > 10) | (both.flysim > 10)]
ratio = (mid.flysim + 1) / (mid.brian2 + 1)
print('neurons >10 Hz:', len(mid), ' median flysim/brian2 rate ratio:', round(ratio.median(), 3))
print('non-stimulated median ratio:', round(ratio[~ratio.index.isin(SUGAR)].median(), 3))
print(mid.assign(ratio=ratio).sort_values('ratio').iloc[[0, 1, 2, -3, -2, -1]].round(1))

# where does the bias start? rate ratio by synaptic hops from the stimulated neurons
hop = {brain.index[f]: 0 for f in SUGAR}
frontier = list(hop)
for h in (1, 2, 3):
    nxt = []
    for i in frontier:
        for j in brain.indices[brain.indptr[i]:brain.indptr[i + 1]]:
            if j not in hop:
                hop[int(j)] = h; nxt.append(int(j))
    frontier = nxt
mid_hop = pd.Series([hop.get(brain.index[int(f)], 9) for f in mid.index], index=mid.index)
print(pd.DataFrame({'ratio': ratio, 'hop': mid_hop}).groupby('hop').ratio.agg(['count', 'median']).round(3))
