"""Every figure in the paper, from data: `uv sync --group paper`, then

    .venv/bin/python -I paper/figures.py          # -> paper/figures/*.pdf, *.jpg

Needs the setup in the README (the reference model, the FlyWire annotations and columns, and
results/sugarR.parquet from experiments/sugar_mn9.py 30), and media/duckophila.mp4 for figure 1.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'figure.dpi': 150})


def fig_frames():
    """Figure 1: two frames of the published video (a startle; the arena)."""
    for name, t in (('frame_startle', 1.4), ('frame_arena', 39.0)):   # the video opens on act 1, title at the end
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(ROOT / 'media' / 'duckophila.mp4'),
                        '-frames:v', '1', '-q:v', '3', str(OUT / f'{name}.jpg')], check=True)


def fig_validation():
    """Figure 2: flysim against Brian2, sugar GRNs at 150 Hz, 30 trials each (FlyWire 630)."""
    from flysim import FlyBrain
    sys.path.insert(0, str(ROOT / 'experiments'))
    from compare_ids import SUGAR, MN9
    ref = pd.read_parquet(ROOT / 'results' / 'sugarR.parquet')
    theirs = ref.groupby('flywire_id').size() / ref.trial.nunique()
    acc = None
    for k in range(30):
        b = FlyBrain(630, seed=k)
        b.stimulate({f: 150 for f in SUGAR})
        c = b.run(1000)
        acc = c if acc is None else acc + c
    ours = pd.Series(acc / 30, index=b.ids)
    both = pd.concat({'brian2': theirs, 'flysim': ours[ours > 0]}, axis=1).fillna(0)
    r = float(np.corrcoef(both.brian2, both.flysim)[0, 1])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(both.brian2, both.flysim, s=6, c='0.3', lw=0)
    ax.scatter([both.brian2.get(MN9, 0)], [both.flysim.get(MN9, 0)], s=24, c='C3', lw=0, label='MN9 (proboscis motor neuron)')
    lim = max(both.max()) * 1.05
    ax.plot([0, lim], [0, lim], c='0.7', lw=0.8, zorder=0)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel='Brian2 reference (Hz)', ylabel='flysim (Hz)',
           title=f'{len(both)} active neurons, r = {r:.3f}')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(OUT / 'validation.pdf'); plt.close(fig)
    return {'validation_r': r, 'validation_n': int(len(both)),
            'mn9_brian2': float(both.brian2.get(MN9, 0)), 'mn9_flysim': float(both.flysim.get(MN9, 0))}


def fig_eyes():
    """Figure 3: where each T4/T5 looks (FlyWire columns, orientation from the wiring), the camera's
    field of view, and the LPLC2 input offsets that fix front and up."""
    import retinotopy as R
    d = R.directions()
    t = d[d.type.str.match(r'T4[a-d]$')].drop_duplicates('column_id')
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1, 1, 1.1]})
    for ax, side in zip(axs[:2], ('left', 'right')):
        s = t[t.hemisphere == side]
        az = s.az_deg if side == 'left' else -s.az_deg
        ax.scatter(az, s.el_deg, s=3, c='0.55', lw=0)
        ax.add_patch(plt.Rectangle((-22.5, -36.4), 45, 72.8, fill=False, ec='C3', lw=1.2))
        ax.set(xlabel='azimuth (deg, + = left)', ylabel='elevation (deg)', aspect='equal',
               title=f'{side} eye, {len(s)} columns')
        ax.axvline(0, c='0.85', lw=0.6, zorder=0); ax.axhline(0, c='0.85', lw=0.6, zorder=0)
        if side == 'left':
            ax.text(0, 40, 'camera', color='C3', ha='center', va='bottom', fontsize=7)
    ax = axs[2]
    names = {'a': 'T4a/T5a front-to-back', 'b': 'T4b/T5b back-to-front', 'c': 'T4c/T5c up', 'd': 'T4d/T5d down'}
    for k, side in enumerate(('left', 'right')):
        o = R.lplc2_offsets(side)
        for dname, col in zip('abcd', ('C0', 'C1', 'C2', 'C4')):
            ax.annotate('', xy=(o.loc[dname, 'dX'], o.loc[dname, 'dY']), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.4, alpha=1.0 if k == 0 else 0.5))
            if k == 0:
                pos = {'a': (-0.2, -0.45, 'right'), 'b': (0.2, -0.45, 'left'), 'c': (0.15, 0.3, 'left'), 'd': (0.15, -0.5, 'left')}[dname]
                ax.text(o.loc[dname, 'dX'] + pos[0], o.loc[dname, 'dY'] + pos[1], names[dname], color=col, fontsize=6, ha=pos[2])
    ax.set(xlim=(-9.0, 8.0), ylim=(-3.6, 3.2), aspect='equal', xlabel='hex X (columns; + = front)', ylabel='hex Y (+ = up)',
           title='LPLC2: where its inputs sit')
    fig.tight_layout(); fig.savefig(OUT / 'eyes.pdf'); plt.close(fig)
    view = d[d.type.str.match(r'T[45][a-d]$')]
    return {'eye_az': [float(t.az_deg.min()), float(t.az_deg.max())], 'eye_el': [float(t.el_deg.min()), float(t.el_deg.max())],
            'columns_left': int((t.hemisphere == 'left').sum()), 'columns_right': int((t.hemisphere == 'right').sum()),
            't45_with_columns': int(len(view))}


def fig_flips():
    """Figure 4: steering DNs under yaw rotation — the whole eye vs only what the camera sees."""
    from flysim import FlyBrain
    from motion import MotionEncoder, PREFERRED
    import neurons as N
    import retinotopy as R
    enc = MotionEncoder()
    d = R.directions(); d = d[d.type.str.match(r'T[45][a-d]$')]
    pref_all = np.array([PREFERRED[h][t[2]] for h, t in zip(d.hemisphere, d.type)], float)
    base = FlyBrain(783, seed=0); base.run(1)
    dns = {n: {s: [base.index[f] for f in N.ids(n, s)] for s in ('left', 'right')} for n in ('DNa02', 'DNp15')}

    def run(ids, pref, u):
        along = (np.array([u, 0.0]) * pref).sum(1)
        s = np.maximum(along - 2.0, 0.0)
        rates = {int(f): float(x) for f, x in zip(ids, 60 * s / (s + 30))}
        out = {}
        for seed in range(3):
            b = base.copy(); b.stimulate(rates); c = b.run(1000)
            for n, sides in dns.items():
                for side, ix in sides.items():
                    out[(n, side)] = out.get((n, side), 0) + c[ix].sum() / len(ix) / 3
        return out

    res = {}
    for view, ids, pref in (('whole eye', list(d.index), pref_all), ('camera only', enc.ids, enc.pref)):
        for rot, u in (('world turns left', -40.0), ('world turns right', 40.0)):
            res[(view, rot)] = run(ids, pref, u)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, n in zip(axs, ('DNa02', 'DNp15')):
        labels, L, Rr = [], [], []
        for view in ('whole eye', 'camera only'):
            for rot in ('world turns left', 'world turns right'):
                labels.append(f'{view}\n{rot}'); r = res[(view, rot)]
                L.append(r[(n, 'left')]); Rr.append(r[(n, 'right')])
        x = np.arange(len(labels))
        ax.bar(x - 0.18, L, 0.36, label=f'{n} left', color='C0'); ax.bar(x + 0.18, Rr, 0.36, label=f'{n} right', color='C1')
        ax.set_xticks(x, labels, fontsize=6); ax.set_title(n + (' (body steering)' if n == 'DNa02' else ' (DNHS1, head)'))
        ax.legend(frameon=False, fontsize=6)
    axs[0].set_ylabel('rate (Hz), 1 s, mean of 3')
    fig.tight_layout(); fig.savefig(OUT / 'flips.pdf'); plt.close(fig)
    return {f'{v}|{r}|{n}|{s}': float(x) for (v, r), o in res.items() for (n, s), x in o.items()}


def fig_follow():
    """Figure 5: following. DNa02 against the target's bearing (three brains) and elevation, open loop;
    and the duck's heading against the target's in closed loop (paper/data, from experiments/follow_*.py)."""
    data = ROOT / 'paper' / 'data'
    fd = json.loads((data / 'follow_dn.json').read_text())
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    ax = axs[0]
    for run, col, label in (('intact', 'C0', 'intact'), ('shuffled', '0.5', 'rewired'), ('input null', 'C1', 'input null')):
        pb = fd['runs'][run]['dna02_LminusR']
        az = [float(a) for a in pb]; m = [pb[a][0] for a in pb]
        err = np.array([[pb[a][0] - pb[a][1] for a in pb], [pb[a][2] - pb[a][0] for a in pb]])
        ax.errorbar(az, m, yerr=err, color=col, marker='o', ms=3, lw=1, capsize=2, label=label)
    ax.axhline(0, c='0.85', lw=0.6, zorder=0)
    ax.set(xlabel='target bearing (deg, + = left)', ylabel='DNa02 left - right (Hz)', title='by bearing, el -10 deg')
    ax.legend(frameon=False, fontsize=6)
    ax = axs[1]
    el = fd['elevation']
    e = [float(k) for k in el]; m = [el[k][0] for k in el]
    err = np.array([[el[k][0] - el[k][1] for k in el], [el[k][2] - el[k][0] for k in el]])
    ax.errorbar(m, e, xerr=err, color='C0', marker='o', ms=3, lw=1, capsize=2)
    ax.axhline(0, c='0.85', lw=0.6, zorder=0)
    ax.set(xlabel='turn toward, +-10 deg (Hz)', ylabel='target elevation (deg)', title='by elevation')
    ax = axs[2]
    for name, col, label in (('intact_1', 'C0', 'intact, kept up'), ('intact_0', 'C3', 'intact, fell behind'),
                             ('shuffled_1', '0.5', 'rewired')):
        rows = json.loads((data / 'follow_runs' / f'{name}.json').read_text())['rows']
        t = [r['t'] for r in rows]
        ax.plot(t, [r['duck_deg'] - rows[0]['duck_deg'] for r in rows], c=col, lw=1, label=label)
    ax.plot(t, [r['target_deg'] - rows[0]['target_deg'] for r in rows], c='k', lw=0.8, ls='--', label='target')
    ax.set(xlabel='time (s)', ylabel='heading (deg, + = left)', title='closed loop')
    ax.legend(frameon=False, fontsize=6, loc='lower left', bbox_to_anchor=(0.0, 0.0))
    ax.set_ylim(-140, 120)
    fig.tight_layout(); fig.savefig(OUT / 'follow.pdf'); plt.close(fig)
    return {'follow_dna02_corr': fd['runs']['intact']['dna02_corr'], 'follow_dna02_rank': fd['runs']['intact']['dna02_rank']}


if __name__ == '__main__':
    numbers = {}
    fig_frames()
    numbers.update(fig_validation())
    numbers.update(fig_eyes())
    numbers.update(fig_flips())
    numbers.update(fig_follow())
    (OUT / 'numbers.json').write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
