"""Spin the striped drum around the duck — the optomotor arena.

    .venv/bin/python -I drum.py                     # 30 deg/s, alternating direction every 6 s
    .venv/bin/python -I drum.py --speed 60 --period 0   # one direction, forever
    .venv/bin/python -I drum.py --off               # park it under the floor

The drum is centred on the duck's head (its ToF sensor) at floor level and, by default, keeps
following the head as the duck walks — a translation-locked arena, as in virtual reality for a
walking fly: the duck can turn relative to the stripes but cannot walk out of them, and walking
adds no motion of its own to the stripes. `--no-follow` fixes the drum where it started.
Positive speed is counter-clockwise seen from above.
"""
import argparse, json, math, socket, time

ap = argparse.ArgumentParser()
ap.add_argument('--speed', type=float, default=30.0, help='deg/s, + is counter-clockwise from above')
ap.add_argument('--period', type=float, default=6.0, help='s per direction; 0 = never reverse')
ap.add_argument('--off', action='store_true')
ap.add_argument('--no-follow', action='store_true', help='leave the drum where it started')
ap.add_argument('--port', type=int, default=7899)
args = ap.parse_args()

s = socket.create_connection(('127.0.0.1', args.port)); f = s.makefile('rw')
def ask(r):
    f.write(json.dumps(r) + '\n'); f.flush(); return json.loads(f.readline())

if args.off:
    ask({'op': 'drum', 'pos': [0, 0, -5], 'yaw': 0}); raise SystemExit
p = ask({'op': 'where'})['tof']['pos']
centre = [p[0], p[1], 0.0]
yaw, last, t0 = 0.0, time.monotonic(), time.monotonic()
print(f'drum at {centre[0]:.2f}, {centre[1]:.2f}; {args.speed} deg/s'
      + (f', reversing every {args.period} s' if args.period else '') + ' — ctrl-c to stop')
try:
    while True:
        now = time.monotonic()
        sign = 1 if not args.period or int((now - t0) / args.period) % 2 == 0 else -1
        yaw += sign * math.radians(args.speed) * (now - last); last = now
        if not args.no_follow:
            p = ask({'op': 'where'})['tof']['pos']
            centre = [p[0], p[1], 0.0]
        ask({'op': 'drum', 'pos': centre, 'yaw': yaw})
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    ask({'op': 'drum', 'pos': [0, 0, -5], 'yaw': 0})
