"""Record acts 3 and 4 of the demo — following, and learning — for make_video.py.

    .venv/bin/python -I video/acts.py follow demo/    # -> demo/follow_{sim,trace}.npz
    .venv/bin/python -I video/acts.py learn demo/     # -> demo/learn_{sim,trace,phases}.npz/.json

Each act starts its own `bridge.py --trace` and `record_sim.py`, as `demo.sh` does for acts 1-2.
  * follow: the ball leads the duck 45 degrees left, back, right and back (lead.py).
  * learn: `bridge.py --learn`. Naive, the ball leads, then the bar: the duck only looks. Then four
    rounds of the ball in front while it is petted (pet.py), and the bar, not petted. Then the ball
    leads again — the duck follows — and the bar — it does not. The phases are written beside the
    recording, with their times, for the captions.
"""
import json, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / '.venv' / 'bin' / 'python')


def start(args, log):
    return subprocess.Popen([PY, '-I', *args], cwd=ROOT, stdout=open(log, 'w'), stderr=subprocess.STDOUT)


def stop(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()


def run(*args):
    subprocess.run([PY, '-I', *args], cwd=ROOT, check=True, capture_output=True)


def main():
    act, d = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else 'demo')
    d = d if d.is_absolute() else ROOT / d
    d.mkdir(exist_ok=True)
    run('lead.py', '--off'); run('drum.py', '--off')
    flags = ['--steer', 'follow', '--no-viz', '--trace', str(d / f'{act}_trace.npz')] + (['--learn'] if act == 'learn' else [])
    log = d / f'{act}.log'
    br = start(['bridge.py', *flags], log)
    while 'wired' not in log.read_text():
        time.sleep(0.5)
    time.sleep(3)
    rec = start(['video/record_sim.py', str(d / f'{act}_sim.npz')], d / f'{act}_rec.log')
    time.sleep(1)
    phases = []
    mark = lambda s: phases.append((time.time(), s))
    lead = ['--speed', '10', '--angle', '45', '--hold', '1.5']
    if act == 'follow':
        run('lead.py', *lead)
    else:
        mark('1. naive: the ball leads - the duck only looks'); run('lead.py', '--which', 'ball', *lead)
        mark('1. naive: the bar leads - the duck only looks'); run('lead.py', '--which', 'bar', *lead)
        for k in range(4):
            mark(f'2. training {k + 1}/4: the ball, and petting')
            p = subprocess.Popen([PY, '-I', 'lead.py', '--which', 'ball', '--angle', '0', '--hold', '0.8'], cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5); run('pet.py', '3'); p.wait(); time.sleep(0.5)
            mark(f'2. training {k + 1}/4: the bar, no petting')
            run('lead.py', '--which', 'bar', '--angle', '0', '--hold', '0.8'); time.sleep(0.5)
        mark('3. test: the ball leads - the duck follows it'); run('lead.py', '--which', 'ball', *lead)
        mark('3. test: the bar leads - the duck stays'); run('lead.py', '--which', 'bar', *lead)
    time.sleep(1)
    stop(rec); stop(br)
    run('lead.py', '--off')
    if phases:
        (d / f'{act}_phases.json').write_text(json.dumps(phases))
    print(f'{act}: recorded -> {d}')


if __name__ == '__main__':
    main()
