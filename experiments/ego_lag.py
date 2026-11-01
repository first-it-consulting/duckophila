"""Cross-correlate measured image flow with the predicted self-motion flow over time lags, to find
the camera/state timing offset and catch a sign error. Frames are keyed by `captured_at`, states
by arrival — what `Body.ego_at` pairs — so with the pairing right the peak sits at lag 0. Walk straight, drum parked."""
import json, socket, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from bridge import Body, Camera, STATE
from motion import MotionEncoder

body = Body(STATE / 'duck-a.sock')
states = []
orig = body._track
def track(st):
    orig(st); states.append((time.time(), body.ego()))   # arrival time, as Body.ego_at keys them
body._track = track
cam = Camera.__new__(Camera); cam.path = STATE / 'duck-a-frame.sock'
enc = MotionEncoder(); p = enc.params
d = socket.socket(socket.AF_UNIX); d.connect(str(STATE / 'duck-a.sock')); df = d.makefile('w')
stop = False
def walk():
    while not stop:
        v = 0.25 if time.time() - t_start < 2 else 0.15    # velstand often will not take a slow first step
        df.write(json.dumps({'jsonrpc': '2.0', 'method': 'robot.move', 'params': {'vx': v, 'vy': 0, 'vyaw': 0}}) + '\n'); df.flush(); time.sleep(0.05)
t_start = time.time()
threading.Thread(target=walk, daemon=True).start()
time.sleep(4)
frames, prev = [], None
end = time.time() + 10
while time.time() < end:
    g, t = cam._frame(); wall = time.time()
    s = cv2.resize(g, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)
    if prev is not None and not np.array_equal(s, prev[0]):
        fl = cv2.calcOpticalFlowFarneback(prev[0], s, None, 0.5, 3, 15, 3, 5, 1.2, 0) * ((p.hfov_deg / p.width) / p.scale) / (t - prev[1])
        frames.append(((prev[1] + t) / 2, fl))   # capture time, as the bridge pairs them
    prev = (s, t, wall)
stop = True; time.sleep(0.1)
df.write(json.dumps({'jsonrpc': '2.0', 'method': 'robot.move', 'params': {'vx': 0, 'vy': 0, 'vyaw': 0}}) + '\n'); df.flush()
st = [(w, e) for w, e in states if e is not None]
st_t = np.array([w for w, _ in st])
print(f'{len(frames)} frame pairs, {len(st)} states')
print('lag (state before frame stamp) -> vertical residual after subtracting, % of measured, and correlation')
for lag in np.arange(-0.30, 0.101, 0.02):
    num = den = 0.0; m_all, q_all = [], []
    for w, fl in frames:
        k = np.searchsorted(st_t, w + lag)
        if 0 < k < len(st):
            q = enc.self_flow(*st[k][1])
            num += np.abs(fl[..., 1] - q[..., 1]).mean(); den += np.abs(fl[..., 1]).mean()
            m_all.append(fl[..., 1].mean()); q_all.append(q[..., 1].mean())
    print(f'   {lag:+.2f} s: {100 * num / den:5.0f}%   r {np.corrcoef(m_all, q_all)[0, 1]:+.2f}')
