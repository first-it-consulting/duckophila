"""Follow me, closed loop, repeated: the real stack (duck-sim, robotd, tofd, mediad), intact vs shuffled.

    DUCK_SIM_CAMERAS=a sim/duck-sim
    .venv/bin/python -I experiments/follow_stats.py [n_runs]      # -> results/follow.json
    .venv/bin/python -I experiments/follow_stats.py --summary     # the table again, from the JSON
    .venv/bin/python -I experiments/follow_stats.py 8 --feet      # a person's feet and the real-world encoder
                                                                  # (follow.PopulationMotion) -> results/follow_feet.json

Per run: `bridge.py --steer follow` with the brain given, and `lead.py` swinging the target round
the duck (8 deg/s; 90 left, back, 90 right, back; 65 s). Scored from the simulator's truth:
  * error     — median |target heading - duck heading|
  * in view   — the share of the run with |error| < 20 deg (the camera sees +-22.5)
  * gain      — the slope of the duck's heading on the target's, over the run (1 = follows, 0 = does not)
  * startles  — giant-fibre startles during the run (should be none: the target keeps its distance)
  * completed — the target stayed in view for at least 90% of the run (a lost target is not found
                again: with nothing in view the duck stands)
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from closed_loop_stats import start, stop, wait_for, PY
from controls import ci_mean, wilson
import subprocess

FEET = '--feet' in sys.argv
OUT = ROOT / 'results' / ('follow_feet' if FEET else 'follow')
RESULT = ROOT / 'results' / ('follow_feet.json' if FEET else 'follow.json')
BRIDGE_EXTRA = (['--detector', 'population'] if FEET else []) + (['--no-startle'] if '--no-startle' in sys.argv else [])
BRIDGE_EXTRA += os.environ.get('DUCKOPHILA_STATS_BRIDGE', '').split()   # e.g. '--readout composite'
if os.environ.get('DUCKOPHILA_STATS_TAG'):
    RESULT = RESULT.with_name(RESULT.stem + '_' + os.environ['DUCKOPHILA_STATS_TAG'] + '.json')
    OUT = OUT.with_name(OUT.name + '_' + os.environ['DUCKOPHILA_STATS_TAG'])
if '--no-startle' in sys.argv:    # the legs loom on the ToF; to see the following on its own
    RESULT = RESULT.with_name(RESULT.stem + '_nostartle.json')
LEAD_EXTRA = (['--which', 'feet'] if FEET else []) + (['--path'] if '--away' in sys.argv else [])
if '--away' in sys.argv:          # the person walks away along a route; the duck has to come after them
    RESULT = RESULT.with_name(RESULT.stem + '_away.json'); OUT = OUT.with_name(OUT.name + '_away')
OUT.mkdir(parents=True, exist_ok=True)


def run(brain, k):
    tag = f'{brain}_{k}'
    subprocess.run([PY, '-I', 'drum.py', '--off'], cwd=ROOT, check=True)   # a drum left standing hides the target
    br = start(['bridge.py', '--no-viz', '--steer', 'follow', '--brain', brain, *BRIDGE_EXTRA], OUT / f'{tag}.log')
    wait_for(OUT / f'{tag}.log', 'wired'); time.sleep(3)
    subprocess.run([PY, '-I', 'lead.py', *LEAD_EXTRA, '--log', str(OUT / f'{tag}.json')], cwd=ROOT, check=True, capture_output=True)
    stop(br)
    subprocess.run([PY, '-I', 'lead.py', '--off'], cwd=ROOT, check=True)
    rows = json.loads((OUT / f'{tag}.json').read_text())['rows']
    e = np.abs([r['error_deg'] for r in rows])
    tg = np.array([r['target_deg'] for r in rows]); dk = np.array([r['duck_deg'] for r in rows])
    gain = float(np.polyfit(tg - tg[0], dk - dk[0], 1)[0])
    startles = (OUT / f'{tag}.log').read_text().count('STARTLE')
    time.sleep(3)   # let the duck settle before the next run
    travel = float(np.abs(np.diff(dk)).sum() / max(1e-9, np.abs(np.diff(tg)).sum()))   # >2: it spun, not followed
    dist = [r['dist_m'] for r in rows if 'dist_m' in r]
    away = {'dist_median': float(np.median(dist)), 'dist_end': float(dist[-1]),
            'walked_m': float(np.hypot(rows[-1]['x'] - rows[0]['x'], rows[-1]['y'] - rows[0]['y']))} if dist else {}
    return {'brain': brain, 'run': k, 'error_median': float(np.median(e)), 'in_view': float(np.mean(e < 20)),
            'gain': gain, 'startles': startles, 'travel': travel, **away}


def main():
    if '--summary' in sys.argv:
        summarize(json.loads(RESULT.read_text())); return
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = int(pos[0]) if pos else 5
    out = []
    for k in range(n):
        for brain in ('intact', 'shuffled'):
            r = run(brain, k)
            out.append(r)
            print(r, flush=True)
    RESULT.write_text(json.dumps(out, indent=1))
    summarize(out)


def summarize(out):
    n = sum(r['brain'] == 'intact' for r in out)
    print(f'\n=== follow me, n = {n} runs per brain, mean [95% bootstrap CI]')
    for brain in ('intact', 'shuffled'):
        rs = [r for r in out if r['brain'] == brain]
        cells = []
        for key, fmt in (('error_median', '{:.1f} deg'), ('in_view', '{:.0%}'), ('gain', '{:.2f}')):
            m, lo, hi = ci_mean([r[key] for r in rs])
            cells.append(f'{key} ' + fmt.format(m) + ' [' + fmt.format(lo) + ', ' + fmt.format(hi) + ']')
        spun = sum(r.get('travel', 0) > 2 and r['error_median'] > 45 for r in rs)   # zig-zagging close behind is not spinning
        if spun:
            print(f'   {brain:9s} {spun} run(s) spun (turned more than twice the target\'s path): their gain is not following')
        zig = [round(r.get('travel', 0), 1) for r in rs]
        print(f'   {brain:9s} turned per degree the target moved: {zig}')
        if rs and 'dist_median' in rs[0]:
            for key in ('dist_median', 'dist_end', 'walked_m'):
                m, lo, hi = ci_mean([r[key] for r in rs]); print(f'   {brain:9s} {key} {m:.2f} m [{lo:.2f}, {hi:.2f}]')
        k = sum(r['in_view'] >= 0.9 for r in rs)
        p, lo, hi = wilson(k, len(rs))
        print(f'   {brain:9s} ' + '; '.join(cells) + f'; startles {sum(r["startles"] for r in rs)}'
              f'; completed {k}/{len(rs)} {p:.2f} [{lo:.2f}, {hi:.2f}]')
        done = [r for r in rs if r['in_view'] >= 0.9]
        if done:
            m, lo, hi = ci_mean([r['error_median'] for r in done])
            print(f'   {"":9s} completed runs: error {m:.1f} deg [{lo:.1f}, {hi:.1f}], gain {np.mean([r["gain"] for r in done]):.2f}')


if __name__ == '__main__':
    main()
