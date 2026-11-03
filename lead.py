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


ROUTE = [('hold', 3.0), ('straight', 1.0), ('hold', 2.0), ('turn', 60.0, 0.8), ('straight', 0.8), ('hold', 2.0),
         ('turn', -90.0, 1.0), ('straight', 0.8), ('hold', 3.0)]   # metres, degrees (+ left) over metres


def route(start, yaw0, v):
    """(t, x, y, yaw, moving) samples of the person's centre along ROUTE at `v` m/s, 20 Hz."""
    x, y, yaw, t, out = start[0], start[1], yaw0, 0.0, []
    dt = 0.05
    for seg in ROUTE:
        if seg[0] == 'hold':
            for _ in range(int(seg[1] / dt)):
                out.append((t, x, y, yaw, False)); t += dt
        else:
            length = seg[1] if seg[0] == 'straight' else seg[2]
            turn = 0.0 if seg[0] == 'straight' else math.radians(seg[1])
            n = max(1, int(length / v / dt))
            for _ in range(n):
                yaw += turn / n; x += v * dt * math.cos(yaw); y += v * dt * math.sin(yaw)
                out.append((t, x, y, yaw, True)); t += dt
    return out


def feet_walking(c, yaw, moving, t):
    """Legs at centre `c` facing `yaw`: striding while walking, stepping on the spot while waiting."""
    stride = 0.12 if moving else 0.0
    ph = 2 * math.pi * STEP_HZ * t
    f, side = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    out = []
    for sgn in (1, -1):
        s = sgn * math.sin(ph)
        xy = np.asarray(c) + 0.1 * sgn * side + stride * s * f
        out.append([float(xy[0]), float(xy[1]), LIFT_M * max(0.0, s)])
    return out[0], out[1]


def walk_away(args, ask, w, psi):
    """The person starts `--distance` ahead of the duck and walks ROUTE; logged as the circle: the
    person's bearing from the duck against the duck's heading, and their distance."""
    p = np.array(w['pos'])
    start = (p[0] + 0.7 * math.cos(psi), p[1] + 0.7 * math.sin(psi))
    path = route(start, psi, args.walk_speed)
    rows, t0 = [], time.monotonic()
    print(f'walking away: {args.walk_speed} m/s, {path[-1][0]:.0f} s')
    try:
        for (ts, x, y, yaw, moving) in path:
            while time.monotonic() - t0 < ts:
                time.sleep(0.005)
            l, r = feet_walking((x, y), yaw, moving, ts)
            ask({'op': 'feet', 'left': l, 'right': r, 'yaw': yaw})
            w = ask({'op': 'where'})['tof']; q = np.array(w['pos'])
            bearing = math.atan2(y - q[1], x - q[0]); duck = heading(w['xmat'])
            rows.append({'t': round(ts, 3), 'unix': time.time(), 'target_deg': math.degrees(bearing), 'duck_deg': math.degrees(duck),
                         'error_deg': math.degrees(wrap(bearing - duck)), 'dist_m': float(math.hypot(x - q[0], y - q[1])),
                         'x': q[0], 'y': q[1], 'px': x, 'py': y})
    finally:
        ask({'op': 'feet', 'left': PARKED, 'right': PARKED})
    for key in ('target_deg', 'duck_deg'):
        u = np.degrees(np.unwrap(np.radians([r[key] for r in rows])))
        for r, v in zip(rows, u):
            r[key] = float(v)
    d = np.array([r['dist_m'] for r in rows]); e = np.abs([r['error_deg'] for r in rows])
    walked = float(np.hypot(rows[-1]['x'] - rows[0]['x'], rows[-1]['y'] - rows[0]['y']))
    print(f'done: distance median {np.median(d):.2f} m (end {d[-1]:.2f}), |error| median {np.median(e):.1f} deg, '
          f'in view {np.mean(e < 20):.0%}, duck walked {walked:.2f} m (net)')
    if args.log:
        with open(args.log, 'w') as fh:
            json.dump({'args': vars(args), 'route': ROUTE, 'rows': rows}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=8.0, help='deg/s the target swings round the duck')
    ap.add_argument('--angle', type=float, default=90.0, help='deg each swing')
    ap.add_argument('--hold', type=float, default=4.0, help='s between swings')
    ap.add_argument('--distance', type=float, default=0.45, help='m from the duck')
    ap.add_argument('--which', choices=('ball', 'bar', 'feet'), default='ball', help='which target leads')
    ap.add_argument('--path', action='store_true', help='with --which feet: the person walks away along a route '
                    '(straight, left curve, straight, right curve), and the duck has to come after them')
    ap.add_argument('--walk-speed', type=float, default=0.07, help='m/s the person walks with --path')
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
    if args.path:
        return walk_away(args, ask, w, psi)
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
