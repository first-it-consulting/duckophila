"""The closed-loop claims, repeated, through the real stack (duck-sim, robotd, tofd, mediad).

    DUCK_SIM_CAMERAS=a sim/duck-sim
    .venv/bin/python -I experiments/closed_loop_stats.py [n_throws]    # -> results/closed_loop.json

For the intact brain and the shuffled null (`bridge.py --brain shuffled`):
  * startle: n fast balls (1.5 m/s; centre, left, right in turn) and n slow approaches (0.2 m/s), each
    timestamped; a startle is a giant-fibre spike from the throw's start to 0.5 s after it arrives,
    and its side is which giant fibre fired more;
  * head optomotor reflex: 60 s of a drum reversing every 5 s, the head yaw from the recorded joint
    and the drum's yaw from its recorded orientation, gain = head rate / drum rate per phase
    (the first second of each phase skipped).
"""
import json, math, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from controls import wilson, ci_mean

PY = str(ROOT / '.venv' / 'bin' / 'python')
OUT = ROOT / 'results' / 'closed_loop'
OUT.mkdir(parents=True, exist_ok=True)


def start(args, log):
    f = open(log, 'w')
    p = subprocess.Popen([PY, '-I', *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p


def wait_for(log, text, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        if Path(log).exists() and text in Path(log).read_text():
            return
        time.sleep(0.5)
    raise SystemExit(f'{log}: no "{text}"')


def stop(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()


def startle(brain, n):
    tag = f'startle_{brain}'
    br = start(['bridge.py', '--no-viz', '--brain', brain, '--trace', str(OUT / f'{tag}_trace.npz')], OUT / f'{tag}.log')
    wait_for(OUT / f'{tag}.log', 'wired'); time.sleep(3)
    throws = []
    sides = ('center', 'left', 'right')
    for kind, speed, k in [('fast', 1.5, i) for i in range(n)] + [('slow', 0.2, i) for i in range(n)]:
        side = sides[k % 3]
        t0 = time.time()
        subprocess.run([PY, '-I', 'throw.py', '--speed', str(speed), '--side', side], cwd=ROOT, check=True, capture_output=True)
        throws.append({'kind': kind, 'speed': speed, 'side': side, 't0': t0, 't1': time.time() - 1.0})  # throw.py holds 1 s at the end
        time.sleep(3.0)
    stop(br)
    tr = np.load(OUT / f'{tag}_trace.npz')
    names = list(tr['key_names'])
    gl, gr = tr['key'][:, names.index('DNp01_L')], tr['key'][:, names.index('DNp01_R')]
    for th in throws:
        m = (tr['t'] >= th['t0']) & (tr['t'] <= th['t1'] + 0.5)
        th['gf_left'], th['gf_right'] = float(gl[m].sum()), float(gr[m].sum())
        th['startle'] = bool(th['gf_left'] + th['gf_right'] > 0)
        hit = np.flatnonzero(m & ((gl > 0) | (gr > 0)))
        th['latency'] = float(tr['t'][hit[0]] - th['t0']) if len(hit) else None
    return throws


def optomotor(brain, seconds=60, period=5.0):
    tag = f'head_{brain}'
    br = start(['bridge.py', '--no-viz', '--steer', 'head', '--no-startle', '--brain', brain], OUT / f'{tag}.log')
    wait_for(OUT / f'{tag}.log', 'wired'); time.sleep(2)
    rec = start(['video/record_sim.py', str(OUT / f'{tag}_sim.npz')], OUT / f'{tag}_rec.log')
    drum = start(['drum.py', '--speed', '30', '--period', str(period), '--no-follow'], OUT / f'{tag}_drum.log')
    time.sleep(seconds)
    stop(drum); stop(rec); stop(br)
    subprocess.run([PY, '-I', 'drum.py', '--off'], cwd=ROOT, check=True)   # stopped is not parked: it would hide what comes next
    s = np.load(OUT / f'{tag}_sim.npz')
    t = s['t'] - s['t'][0]
    head = np.unwrap(s['joints'][:, 7])                               # head_yaw, wire order
    q = s['drum_quat']
    drum_yaw = np.unwrap(2 * np.arctan2(q[:, 3], q[:, 0]))
    gains = []
    for k in range(int(seconds // period)):
        m = (t >= k * period + 1.0) & (t < (k + 1) * period)
        if m.sum() < 10:
            continue
        dr = np.polyfit(t[m], drum_yaw[m], 1)[0]
        hr = np.polyfit(t[m], head[m], 1)[0]
        if abs(dr) > 0.1:
            gains.append(hr / dr)
    return gains


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = {}
    for brain in ('intact', 'shuffled'):
        out[brain] = {'throws': startle(brain, n), 'head_gain': optomotor(brain)}
        (OUT.parent / 'closed_loop.json').write_text(json.dumps(out))
        print(f'{brain} done', flush=True)
    summarize(out)


def summarize(out):
    for brain, r in out.items():
        print(f'\n=== {brain}')
        for kind in ('fast', 'slow'):
            th = [t for t in r['throws'] if t['kind'] == kind]
            k = sum(t['startle'] for t in th)
            p, lo, hi = wilson(k, len(th))
            lat = [t['latency'] for t in th if t['latency'] is not None]
            print(f'   {kind} ({th[0]["speed"]} m/s): startle {k}/{len(th)} = {p:.2f} [{lo:.2f}, {hi:.2f}]'
                  + (f', latency {np.mean(lat):.2f} s' if lat else ''))
        lat = [t for t in r['throws'] if t['kind'] == 'fast' and t['side'] != 'center' and t['startle']]
        ip = [(t['gf_left'] if t['side'] == 'left' else t['gf_right']) / (t['gf_left'] + t['gf_right']) for t in lat]
        if ip:
            m, lo, hi = ci_mean(ip)
            print(f'   lateral fast throws: ipsilateral giant-fibre share {m:.2f} [{lo:.2f}, {hi:.2f}] over {len(ip)}')
        g = r['head_gain']
        if g:
            m, lo, hi = ci_mean(g)
            print(f'   head optomotor gain {m:.2f} [{lo:.2f}, {hi:.2f}] over {len(g)} drum phases')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--summary':
        summarize(json.loads((OUT.parent / 'closed_loop.json').read_text()))
    else:
        main()
