"""Throw the orange prop at the duck in `duckophila/sim/duck-sim`.

    .venv/bin/python -I throw.py                       # 1.5 m/s, straight on
    .venv/bin/python -I throw.py --side left --speed 1
    .venv/bin/python -I throw.py --speed 0.2           # a hand at petting speed: no startle

Aims from where the duck's ToF sensor is right now (the hook's `where`), so it works wherever the
duck has wandered: starts 1 m out, level-ish and inside the field of view, ends 12 cm in front of
the sensor, holds a second, then parks under the floor again.
"""
import argparse
import json
import socket
import time

import numpy as np

PARKED = [0.0, 0.0, -5.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=1.5, help='m/s')
    ap.add_argument('--side', choices=('left', 'center', 'right'), default='center')
    ap.add_argument('--port', type=int, default=7899)
    args = ap.parse_args()

    s = socket.create_connection(('127.0.0.1', args.port))
    f = s.makefile('rw')

    def ask(req):
        f.write(json.dumps(req) + '\n'); f.flush()
        ans = json.loads(f.readline())
        if 'error' in ans:
            raise SystemExit(f"prop hook: {ans['error']} — is the sim up via duckophila/sim/duck-sim?")
        return ans

    tof = ask({'op': 'where'})['tof']
    p = np.array(tof['pos'])
    fwd = np.array(tof['xmat']).reshape(3, 3)[:, 0]          # sensor +x, in the world
    h = np.array([fwd[0], fwd[1], 0.0]); h /= np.linalg.norm(h)
    left = np.array([-h[1], h[0], 0.0])
    lat = {'left': 1, 'center': 0, 'right': -1}[args.side]
    start = p + 1.0 * h + 0.3 * lat * left
    start[2] = max(p[2] - 0.08, 0.08)                         # above the floor, in the beam
    end = p + 0.12 * fwd + 0.03 * lat * left

    dur = np.linalg.norm(end - start) / args.speed
    print(f'throwing {args.side} at {args.speed} m/s: {dur:.2f} s from 1 m out')
    t0 = time.monotonic()
    while (k := (time.monotonic() - t0) / dur) < 1.0:
        ask({'op': 'prop', 'pos': (start + (end - start) * k).tolist()})
        time.sleep(0.01)
    ask({'op': 'prop', 'pos': end.tolist()})
    time.sleep(1.0)
    ask({'op': 'prop', 'pos': PARKED})


if __name__ == '__main__':
    main()
