"""Lead the duck: walk the glowing target ahead of it, turning, for `bridge.py --steer follow`.

    .venv/bin/python -I lead.py                         # 8 deg/s; left 90, back, right 90, back
    .venv/bin/python -I lead.py --speed 12 --log run.json
    .venv/bin/python -I lead.py --off                   # park it under the floor

The target rolls on the floor `--distance` (0.45 m) ahead of the duck's head, the way someone's
feet walking ahead of it would — the duck cannot catch up, only keep facing it. On the floor, not
at eye height, because that is where the fly's pathway looks: through the wiring, LC10a turns
DNa02 for a target below the eye's equator and hardly at all above it (`experiments/follow_dn.py`),
and the camera looks down ~14 degrees, so a target level with it sits above the image centre. Its compass heading starts where the duck
faces, holds, then swings left by `--angle`, holds, swings back, and the same to the right.
Logged at 20 Hz from the simulator's truth (the hook's `where`): the duck's heading, the target's,
and the error between them — what a run is scored on (`experiments/follow_stats.py`).
"""
import argparse
import json
import math
import socket
import time

import numpy as np

PARKED = [0.0, 0.0, -5.0]
HEIGHT_M = 0.04           # the target's centre: its radius, so it sits on the floor


def heading(xmat) -> float:
    f = np.array(xmat).reshape(3, 3)[:, 0]
    return math.atan2(f[1], f[0])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def schedule(speed_deg, angle_deg, hold_s):
    """(duration s, rate deg/s) segments: hold, left, hold, back, hold, right, hold, back, hold."""
    turn = angle_deg / speed_deg
    return [(hold_s, 0), (turn, speed_deg), (hold_s, 0), (turn, -speed_deg), (hold_s, 0),
            (turn, -speed_deg), (hold_s, 0), (turn, speed_deg), (hold_s, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=8.0, help='deg/s the target swings round the duck')
    ap.add_argument('--angle', type=float, default=90.0, help='deg each swing')
    ap.add_argument('--hold', type=float, default=4.0, help='s between swings')
    ap.add_argument('--distance', type=float, default=0.45, help='m from the duck')
    ap.add_argument('--log', help='write the 20 Hz log here (JSON)')
    ap.add_argument('--off', action='store_true')
    ap.add_argument('--port', type=int, default=7899)
    args = ap.parse_args()

    s = socket.create_connection(('127.0.0.1', args.port)); f = s.makefile('rw')

    def ask(r):
        f.write(json.dumps(r) + '\n'); f.flush(); return json.loads(f.readline())

    if args.off:
        ask({'op': 'target', 'pos': PARKED}); return
    w = ask({'op': 'where'})['tof']
    psi = heading(w['xmat'])
    segs = schedule(args.speed, args.angle, args.hold)
    rows, t0, last = [], time.monotonic(), time.monotonic()
    k, seg_end = 0, segs[0][0]
    print(f'leading: {args.speed} deg/s, +-{args.angle} deg, {args.distance} m ahead, {sum(d for d, _ in segs):.0f} s')
    try:
        while k < len(segs):
            now = time.monotonic()
            t = now - t0
            while k < len(segs) and t >= seg_end:
                k += 1
                seg_end += segs[k][0] if k < len(segs) else 0
            if k >= len(segs):
                break
            psi += math.radians(segs[k][1]) * (now - last); last = now
            w = ask({'op': 'where'})['tof']
            p = np.array(w['pos'])
            ask({'op': 'target', 'pos': [p[0] + args.distance * math.cos(psi), p[1] + args.distance * math.sin(psi),
                                          HEIGHT_M]})
            duck = heading(w['xmat'])
            rows.append({'t': round(t, 3), 'unix': time.time(), 'segment': k, 'target_deg': math.degrees(psi),
                         'duck_deg': math.degrees(duck), 'error_deg': math.degrees(wrap(psi - duck)),
                         'x': p[0], 'y': p[1]})
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        ask({'op': 'target', 'pos': PARKED})
    # unwrap both headings, so a run that went round reads as such
    for key in ('target_deg', 'duck_deg'):
        u = np.degrees(np.unwrap(np.radians([r[key] for r in rows])))
        for r, v in zip(rows, u):
            r[key] = float(v)
    e = np.abs([r['error_deg'] for r in rows])
    print(f'done: |error| median {np.median(e):.1f} deg, in view (|error| < 20) {np.mean(e < 20):.0%}, '
          f'duck turned {rows[-1]["duck_deg"] - rows[0]["duck_deg"]:+.0f} deg overall')
    if args.log:
        with open(args.log, 'w') as fh:
            json.dump({'args': vars(args), 'segments': segs, 'rows': rows}, fh)


if __name__ == '__main__':
    main()
