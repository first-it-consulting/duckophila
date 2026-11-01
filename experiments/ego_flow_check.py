"""Does the walking efference copy predict what walking does to the image? Walk the duck straight
over the bare floor (drum parked), and per camera frame compare the measured flow with
`MotionEncoder.self_flow` from the duck's own state. Run with duck-sim up and the camera on."""
import json, math, socket, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from bridge import Body, Camera, STATE, FRAME_LAG_S
from motion import MotionEncoder

body = Body(STATE / 'duck-a.sock')
cam = Camera.__new__(Camera); cam.path = STATE / 'duck-a-frame.sock'          # just its frame grabber
enc = MotionEncoder(); p = enc.params
# what the motion neurons see: flow averaged over about one column, as MotionEncoder does
seen = lambda f: cv2.blur(np.ascontiguousarray(f, np.float32), (enc.column_px, enc.column_px))
drive = socket.socket(socket.AF_UNIX); drive.connect(str(STATE / 'duck-a.sock')); df = drive.makefile('w')
vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
stop = False
t_start = time.time()
def walk():
    while not stop:
        # a firmer first two seconds: velstand often will not take the first step at a slow ask
        v = max(vx, 0.25) if time.time() - t_start < 2 else vx
        df.write(json.dumps({'jsonrpc': '2.0', 'method': 'robot.move', 'params': {'vx': v, 'vy': 0.0, 'vyaw': 0.0}}) + '\n'); df.flush(); time.sleep(0.05)
threading.Thread(target=walk, daemon=True).start()
time.sleep(4)                                               # get going
if np.linalg.norm(body.velocity()) < 0.02:
    raise SystemExit('the duck did not start walking; run it again')
prev = None; rows = []; raw = []
t_end = time.time() + 8
while time.time() < t_end:
    grey, t = cam._frame()
    small = cv2.resize(grey, None, fx=p.scale, fy=p.scale, interpolation=cv2.INTER_AREA)
    if prev is not None and not np.array_equal(small, prev[0]):
        dt = t - prev[1]
        meas = cv2.calcOpticalFlowFarneback(prev[0], small, None, 0.5, 3, 15, 3, 5, 1.2, 0) * ((p.hfov_deg / p.width) / p.scale) / dt
        ego = body.ego_at((t + prev[1]) / 2 - FRAME_LAG_S)
        pred = enc.self_flow(*ego) * enc.texture(small)[..., None]
        floor = np.ones(pred.shape[:2], bool)   # every pixel: rotation moves all of them
        m, q = seen(meas)[floor], seen(pred)[floor]
        rows.append((np.linalg.norm(m, axis=1).mean(), np.linalg.norm(q, axis=1).mean(), np.linalg.norm(m - q, axis=1).mean(),
                     float(np.corrcoef(m.ravel(), q.ravel())[0, 1]), ego[4][0],
                     np.abs(m[:, 1]).mean(), np.abs(m[:, 1] - q[:, 1]).mean(), np.abs(m[:, 0]).mean(), np.abs(m[:, 0] - q[:, 0]).mean()))
    prev = (small, t)
stop = True; time.sleep(0.1)
df.write(json.dumps({'jsonrpc': '2.0', 'method': 'robot.move', 'params': {'vx': 0, 'vy': 0, 'vyaw': 0}}) + '\n'); df.flush()
r = np.array(rows)
print(f'{len(r)} frame pairs walking at {vx} m/s asked, odometry {r[:, 4].mean():.3f} m/s')
print(f'flow, deg/s (mean per frame):  measured {r[:, 0].mean():5.1f}   predicted {r[:, 1].mean():5.1f}   '
      f'residual {r[:, 2].mean():5.1f}  ({100 * r[:, 2].mean() / r[:, 0].mean():.0f}% of measured)   correlation {np.nanmean(r[:, 3]):.2f}')
print(f'  vertical   |measured| {r[:, 5].mean():5.1f} -> after subtracting {r[:, 6].mean():5.1f}  ({100 * r[:, 6].mean() / r[:, 5].mean():.0f}%)')
print(f'  horizontal |measured| {r[:, 7].mean():5.1f} -> after subtracting {r[:, 8].mean():5.1f}  ({100 * r[:, 8].mean() / r[:, 7].mean():.0f}%)'
      '   (gait yaw wobble is left in on purpose)')
