"""Record the simulated duck and its props at 30 Hz, until SIGTERM or ctrl-c, for make_video.py.

    .venv/bin/python -I video/record_sim.py OUT.npz

Every joint and the trunk's pose from the body server's own `read` (as a second client, the way
`scripts/duck-sim drive` reads it), the ball, the drum and the two follow targets from the prop hook. Wall-clock stamped,
so it lines up with `bridge.py --trace`.
"""
import json, signal, socket, sys, time
import numpy as np

out = sys.argv[1]
body = socket.create_connection(('127.0.0.1', 7801)).makefile('rw')
body.write(json.dumps({'op': 'hello', 'protocol': 1, 'joints': 15}) + '\n'); body.flush(); body.readline()
props = socket.create_connection(('127.0.0.1', 7899)).makefile('rw')
rows = {'t': [], 'joints': [], 'trunk': [], 'quat': [], 'prop': [], 'drum_pos': [], 'drum_quat': [], 'target': [], 'bar': [], 'feet': [], 'feet_quat': []}
stop = False
def done(*_):
    global stop; stop = True
signal.signal(signal.SIGTERM, done); signal.signal(signal.SIGINT, done)
while not stop:
    t0 = time.time()
    body.write('{"op":"read"}\n'); body.flush(); r = json.loads(body.readline())
    props.write('{"op":"props"}\n'); props.flush(); p = json.loads(props.readline())
    rows['t'].append(t0); rows['joints'].append(r['positions']); rows['trunk'].append(r['trunk'])
    rows['quat'].append(r['imu']['quat']); rows['prop'].append(p['prop'])
    rows['target'].append(p.get('target', [0, 0, -5])); rows['bar'].append(p.get('bar', [0, 0, -5]))
    rows['feet'].append(p.get('feet', [[0, 0, -5], [0, 0, -5]])); rows['feet_quat'].append(p.get('feet_quat', [1, 0, 0, 0]))
    rows['drum_pos'].append(p.get('drum', {}).get('pos', [0, 0, -5])); rows['drum_quat'].append(p.get('drum', {}).get('quat', [1, 0, 0, 0]))
    time.sleep(max(0.0, 1 / 30 - (time.time() - t0)))
np.savez_compressed(out, **{k: np.array(v) for k, v in rows.items()})
print(f'{len(rows["t"])} samples -> {out}')
