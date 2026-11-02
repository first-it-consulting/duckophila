"""Lead the duck: walk the glowing target ahead of it, turning, for `bridge.py --steer follow`.

    .venv/bin/python -I lead.py                         # 8 deg/s; left 90, back, right 90, back
    .venv/bin/python -I lead.py --speed 12 --log run.json
    .venv/bin/python -I lead.py --off                   # park it under the floor
    .venv/bin/python -I lead.py --which feet            # a person walking round the duck, for --detector moving

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
HEIGHT_M = 0.04           # the ball's centre: its radius, so it sits on the floor
BAR_HALF_M = 0.05         # the bar's centre: half its height, standing on the floor
FEET_DISTANCE_M = 0.6     # a person keeps a little more room than a ball
STEP_HZ = 1.0             # steps per second, each foot
LIFT_M = 0.04             # how high a stepping foot comes up


def feet(c, psi, rate, t, distance):
    """A person centred at c: walking round the duck at `rate` rad/s, they face their way and stride
    (stride from the speed); holding, they face the duck and step on the spot — people rarely stand
    still, and the real-world detector only sees what moves. -> (left, right, yaw)."""
    yaw = psi + (math.copysign(math.pi / 2, rate) if rate else math.pi)
    v = abs(rate) * distance
    stride = min(0.15, v / (2 * STEP_HZ))
    ph = 2 * math.pi * STEP_HZ * t
    f, side = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    out = []
    for sgn in (1, -1):                       # left, right: in antiphase
        s = sgn * math.sin(ph)
        xy = c[:2] + 0.1 * sgn * side + stride * s * f
        out.append([float(xy[0]), float(xy[1]), LIFT_M * max(0.0, s)])
    return out[0], out[1], yaw


def heading(xmat) -> float:
    f = np.array(xmat).reshape(3, 3)[:, 0]
    return math.atan2(f[1], f[0])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def schedule(speed_deg, angle_deg, hold_s):
    """(duration s, rate deg/s) segments: hold, left, hold, back, hold, right, hold, back, hold."""
    turn = angle_deg / speed_deg if speed_deg else 0.0
    return [(hold_s, 0), (turn, speed_deg), (hold_s, 0), (turn, -speed_deg), (hold_s, 0),
            (turn, -speed_deg), (hold_s, 0), (turn, speed_deg), (hold_s, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=8.0, help='deg/s the target swings round the duck')
    ap.add_argument('--angle', type=float, default=90.0, help='deg each swing')
    ap.add_argument('--hold', type=float, default=4.0, help='s between swings')
    ap.add_argument('--distance', type=float, default=0.45, help='m from the duck')
    ap.add_argument('--which', choices=('ball', 'bar', 'feet'), default='ball', help='which target leads')
    ap.add_argument('--log', help='write the 20 Hz log here (JSON)')
    ap.add_argument('--off', action='store_true')
    ap.add_argument('--port', type=int, default=7899)
    args = ap.parse_args()

    s = socket.create_connection(('127.0.0.1', args.port)); f = s.makefile('rw')

    def ask(r):
        f.write(json.dumps(r) + '\n'); f.flush(); return json.loads(f.readline())

    if args.off:
        for which in ('ball', 'bar'):
            ask({'op': 'target', 'pos': PARKED, 'which': which})
        ask({'op': 'feet', 'left': PARKED, 'right': PARKED})
        return
    if args.which == 'feet' and args.distance == ap.get_default('distance'):
        args.distance = FEET_DISTANCE_M
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
            c = np.array([p[0] + args.distance * math.cos(psi), p[1] + args.distance * math.sin(psi)])
            if args.which == 'feet':
                l, r, yaw = feet(c, psi, math.radians(segs[k][1]), t, args.distance)
                ask({'op': 'feet', 'left': l, 'right': r, 'yaw': yaw})
            else:
                ask({'op': 'target', 'pos': [c[0], c[1], HEIGHT_M if args.which == 'ball' else BAR_HALF_M], 'which': args.which})
            duck = heading(w['xmat'])
            rows.append({'t': round(t, 3), 'unix': time.time(), 'segment': k, 'target_deg': math.degrees(psi),
                         'duck_deg': math.degrees(duck), 'error_deg': math.degrees(wrap(psi - duck)),
                         'x': p[0], 'y': p[1]})
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if args.which == 'feet':
            ask({'op': 'feet', 'left': PARKED, 'right': PARKED})
        else:
            ask({'op': 'target', 'pos': PARKED, 'which': args.which})
    # unwrap both headings, so a run that went round reads as such
    for key in ('target_deg', 'duck_deg'):
        u = np.degrees(np.unwrap(np.radians([r[key] for r in rows])))
        for r, v in zip(rows, u):
            r[key] = float(v)
    e = np.abs([r['error_deg'] for r in rows])
    if not rows:
        return
    print(f'done: |error| median {np.median(e):.1f} deg, in view (|error| < 20) {np.mean(e < 20):.0%}, '
          f'duck turned {rows[-1]["duck_deg"] - rows[0]["duck_deg"]:+.0f} deg overall')
    if args.log:
        with open(args.log, 'w') as fh:
            json.dump({'args': vars(args), 'segments': segs, 'rows': rows}, fh)


if __name__ == '__main__':
    main()
