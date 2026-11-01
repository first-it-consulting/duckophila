"""The rotation half of the walking efference copy, on its own: the duck stands, its head nods
(pitch, +-0.3 rad at 0.5 Hz, slow enough that frame timing barely matters), drum parked. Per frame,
measured vs predicted vertical flow. With the geometry and sign right they agree."""
import json, math, socket, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from bridge import Body, Camera, STATE, FRAME_LAG_S
from motion import MotionEncoder

body = Body(STATE / 'duck-a.sock')
cam = Camera.__new__(Camera); cam.path = STATE / 'duck-a-frame.sock'
enc = MotionEncoder(); p = enc.params
if len(sys.argv) > 1: p.texture_half = float(sys.argv[1])
# what the motion neurons see: flow averaged over about one column, as MotionEncoder does
seen = lambda f: cv2.blur(np.ascontiguousarray(f, np.float32), (enc.column_px, enc.column_px))
d = socket.socket(socket.AF_UNIX); d.connect(str(STATE / 'duck-a.sock')); df = d.makefile('w')
time.sleep(1)
h0 = list(body.state['head'])
stop = False
def nod():
    t0 = time.time()
    while not stop:
        a = 0.3 * math.sin(2 * math.pi * 0.5 * (time.time() - t0))
        df.write(json.dumps({'jsonrpc': '2.0', 'method': 'robot.head', 'params': {'neck_pitch': h0[0], 'head_pitch': h0[1] + a, 'head_yaw': h0[2], 'head_roll': h0[3]}}) + '\n'); df.flush()
        time.sleep(0.02)
threading.Thread(target=nod, daemon=True).start()
time.sleep(2)
rows, prev = [], None
end = time.time() + 8
while time.time() < end:
    g, t = cam._frame()
    s = cv2.resize(g, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)
    if prev is not None and not np.array_equal(s, prev[0]):
        m = cv2.calcOpticalFlowFarneback(prev[0], s, None, 0.5, 3, 15, 3, 5, 1.2, 0) * ((p.hfov_deg / p.width) / p.scale) / (t - prev[1])
        e = body.ego_at((t + prev[1]) / 2 - FRAME_LAG_S)
        q = enc.self_flow(*e) * enc.texture(s)[..., None]
        mb, qb = seen(m), seen(q)
        rows.append((m[..., 1].mean(), q[..., 1].mean(), m[..., 0].mean(), q[..., 0].mean(), np.degrees(np.linalg.norm(e[5])),
                     np.abs(mb[..., 1]).mean(), np.abs(mb[..., 1] - qb[..., 1]).mean()))
    prev = (s, t)
stop = True
r = np.array(rows)
print(f'{len(r)} frames; camera pitch/roll rate up to {r[:, 4].max():.0f} deg/s')
print(f'vertical:   measured std {r[:, 0].std():5.1f}, predicted std {r[:, 1].std():5.1f}, correlation {np.corrcoef(r[:, 0], r[:, 1])[0, 1]:+.2f}, '
      f'slope measured/predicted {np.polyfit(r[:, 1], r[:, 0], 1)[0]:+.2f}')
print(f'per column, vertical: |measured| {r[:, 5].mean():.1f} -> after subtracting {r[:, 6].mean():.1f} ({100 * r[:, 6].mean() / r[:, 5].mean():.0f}%)')
print('   samples (measured, predicted):', ' '.join(f'({a:+.0f},{b:+.0f})' for a, b in r[::12, :2]))
