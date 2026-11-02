"""Teach the duck, closed loop: it follows what it was petted for, and not the other thing.

    DUCK_SIM_CAMERAS=a sim/duck-sim
    .venv/bin/python -I experiments/teach.py [n_runs]      # -> results/teach.json

Per run, a fresh `bridge.py --steer follow --learn` (intact, then rewired), and through the real
stack:
  1. naive: the ball leads the duck (`lead.py`, 30 deg each way), then the bar — the duck has not
     learned anything, so it should stand and look at both; this also sets each cue's naive value;
  2. training: four rounds of one target standing in front of the duck while it is petted
     (`pet.py`, 3 s), then the other standing there, not petted — the ball petted in even runs, the
     bar in odd ones, so what is learned is the pairing, not the ball;
  3. test: the ball leads, then the bar.
Scored from the simulator's truth (lead.py's log): how far the duck got (net, start to end: the
gait sways the head several centimetres, so a path length counts standing as walking), and its
heading gain, while each target led. Learning shows as walking after the petted target and not after the
other, after training only.
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from closed_loop_stats import start, stop, wait_for, PY
from controls import ci_mean, wilson

OUT = ROOT / 'results' / 'teach'
OUT.mkdir(parents=True, exist_ok=True)
LEAD = ['--angle', '30', '--hold', '3']             # 15 s of holds, two 30-degree swings each way: ~30 s


def lead(which, log):
    subprocess.run([PY, '-I', 'lead.py', '--which', which, '--log', str(log), *LEAD], cwd=ROOT, check=True, capture_output=True)
    rows = json.loads(Path(log).read_text())['rows']
    xy = np.array([[r['x'], r['y']] for r in rows])
    tg = np.array([r['target_deg'] for r in rows]); dk = np.array([r['duck_deg'] for r in rows])
    return {'moved_m': float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 1 else 0.0,
            'gain': float(np.polyfit(tg - tg[0], dk - dk[0], 1)[0]),
            'error_median': float(np.median(np.abs([r['error_deg'] for r in rows])))}


def present(which, pet, tag):
    """The target standing 0.45 m in front for 4 s; petted for 3 s from 0.5 s in, if `pet`."""
    p = subprocess.Popen([PY, '-I', 'lead.py', '--which', which, '--angle', '0', '--hold', '0.8'], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if pet:
        time.sleep(0.5)
        subprocess.run([PY, '-I', 'pet.py', '3'], cwd=ROOT, check=True, capture_output=True)
    p.wait()
    time.sleep(1.0)


def run(brain, k):
    tag = f'{brain}_{k}'
    log = OUT / f'{tag}.log'
    subprocess.run([PY, '-I', 'drum.py', '--off'], cwd=ROOT, check=True)   # a drum left standing hides the targets
    br = start(['bridge.py', '--no-viz', '--steer', 'follow', '--learn', '--brain', brain], log)
    wait_for(log, 'wired'); time.sleep(3)
    petted, other = ('ball', 'bar') if k % 2 == 0 else ('bar', 'ball')
    res = {'brain': brain, 'run': k, 'petted': petted}
    for which in ('ball', 'bar'):
        res[f'naive_{"petted" if which == petted else "other"}'] = lead(which, OUT / f'{tag}_naive_{which}.json'); time.sleep(2)
    for _ in range(4):
        present(petted, True, tag); present(other, False, tag)
    for which in ('ball', 'bar'):
        res[f'test_{"petted" if which == petted else "other"}'] = lead(which, OUT / f'{tag}_test_{which}.json'); time.sleep(2)
    stop(br)
    subprocess.run([PY, '-I', 'lead.py', '--off'], cwd=ROOT, check=True)
    res['startles'] = log.read_text().count('STARTLE')
    time.sleep(3)
    return res


def main():
    if '--summary' in sys.argv:
        summarize(json.loads((ROOT / 'results' / 'teach.json').read_text())); return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = []
    for k in range(n):
        for brain in ('intact', 'shuffled'):
            r = run(brain, k); out.append(r)
            print(json.dumps(r), flush=True)
            (ROOT / 'results' / 'teach.json').write_text(json.dumps(out, indent=1))
    summarize(out)


def summarize(out):
    print(f'\n=== teach: metres moved while each target led (~30 s, net), mean [95% CI]; followed = moved > 0.3 m')
    for brain in ('intact', 'shuffled'):
        rs = [r for r in out if r['brain'] == brain]
        cells = []
        for key in ('naive_petted', 'naive_other', 'test_petted', 'test_other'):
            w = [r[key]['moved_m'] for r in rs]
            m, lo, hi = ci_mean(w)
            k = sum(x > 0.3 for x in w)
            cells.append(f'{key} {m:.2f} m [{lo:.2f}, {hi:.2f}] {k}/{len(rs)}')
        print(f'   {brain:9s} ' + '; '.join(cells) + f'; startles {sum(r["startles"] for r in rs)}')
        g = ci_mean([r['test_petted']['gain'] for r in rs])
        print(f'   {"":9s} heading gain after the petted target, test: {g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}]')


if __name__ == '__main__':
    main()
