"""The fly brain, wired to a duck. Steps 1 and 2 of `docs/ideas/fly-brain.md`.

    .venv/bin/python -I bridge.py              # startle only (step 1), with the live brain view
    .venv/bin/python -I bridge.py --steer head  # and the head optomotor reflex (step 2) — needs the camera
    .venv/bin/python -I bridge.py --steer body  # the head reflex, and the walking body following the head
    .venv/bin/python -I bridge.py --steer dna02 # the body turned by DNa02 directly, for comparison
    .venv/bin/python -I bridge.py --no-viz     # without the viewer
    .venv/bin/python -I bridge.py --dry        # brain and view only; send the duck nothing
    .venv/bin/python -I bridge.py --record run.rrd   # and save the view to replay or share

**Step 1, startle.** Every `tof.frame` from `tofd` goes through `LoomingEncoder` onto LPLC2/LC4;
a giant-fiber spike is a startle. The startle is composed here from intents `robotd` already has
(alarm honk, look at the zone, crouch, back off) — only *whether* and *which side* come from the
fly; the shape of the flinch is ours. A 2 s refractory makes one threat one startle.

**Step 2, the optomotor reflex.** Camera frames (`mediad`'s local `media.frame`, one per
connection) go through `MotionEncoder` onto T4/T5. Two outputs, both left/right differences of a
descending neuron pair, smoothed, R > L meaning "turn right" (the sign `experiments/optomotor_dn.py`
found and the literature has):
  * `--steer head`: DNp15, alias DNHS1 — the HS cells' own descending neuron, which turns a fly's
    head with the motion. Integrated into `robot.head` yaw. While we hold the head, `robotd`'s
    idle look-around stands down, so no efference copy is needed.
  * `--steer body`: the head reflex above, and the body follows the head — **our coordination
    rule, not the fly's wiring**: walking at 0.1 m/s, `vyaw = k * head_yaw`, turning the body
    toward wherever the head has turned, while the head counter-rotates by the trunk's measured
    turn (gyro) so the gaze holds — the way an animal keeps its eyes still while its body comes
    round. Flies lead with the head too; the coupling here is a stand-in, said as such.
  * `--steer dna02`: DNa02 (Rayshubskiy et al. 2020) -> `robot.move` vyaw directly, with the
    head's idle look-around taken out of the image (efference copy). The literature's turning
    neuron — but it needs more visual field than the camera's ±22° x ±36°: over that view it does
    not flip with rotation (`experiments/retinotopy_check.py`). Kept to show exactly that.
Which way and how hard come from the fly; the gains, the smoothing and the clamps are ours. A
flinch outranks steering while it lasts.

The brain runs on the ToF clock (~15 Hz) for exactly the time since the previous frame, with the
latest camera rates; the camera thread keeps them fresh at ~30 Hz. The sockets and the JSON-RPC
are the robot's own, so the same script reaches a real duck's local sockets. (A brain on a laptop
next to a real duck should take the camera over WebRTC instead; that is for when we get there.)

The view (Rerun) shows every neuron at its real position in the brain, lit when it spikes, beside
the depth frame, the per-zone looming signal, the camera and its flow, and the descending neurons.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import socket
import threading
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from flysim import FlyBrain
from geometry import Geometry
from looming import COLS, ROWS, LoomingEncoder
import neurons as N

STATE = Path(os.environ.get('DUCK_SIM_STATE', Path.home() / '.cache' / 'duck-sim'))
REFRACTORY_S = 2.0
FLINCH_S = 0.6
FOV_DEG = 45.0
# rad/s per Hz of R - L, per output. DNp15 swings ~3x wider than DNa02 for the same drum, and
# the head answers faster than a walking body; 0.012 on the head oscillates (~0.3 s loop latency).
STEER_GAIN = {'head': float(os.environ.get('DUCKOPHILA_HEAD_GAIN', 0.004)),
              'dna02': float(os.environ.get('DUCKOPHILA_BODY_GAIN', 0.02))}
STEER_GAIN['body'] = STEER_GAIN['head']
# --steer body: rad/s of body turn per rad of head yaw — the head is recentred in ~1/k seconds
HEAD_TO_BODY = float(os.environ.get('DUCKOPHILA_HEAD_TO_BODY', 1.0))
# How much older a camera frame is than its `captured_at` says, against the robot.state stream.
# Measured in duck-sim (`experiments/ego_lag.py`: the walking efference copy correlates best,
# r = 0.90, with the state 0-0.02 s before the stamp); a real duck's pipeline needs its own.
FRAME_LAG_S = float(os.environ.get('DUCKOPHILA_FRAME_LAG_S', 0.01))
STEER_TAU_S = 0.15       # smoothing of the R - L difference
STEER_MAX = 1.0          # rad/s
HEAD_YAW_MAX = 0.9       # rad either side of straight ahead
# Steering the body means walking while turning, as a fly does: velstand ignores yaw commands under
# ~0.55 rad/s on the spot, and below ~0.1 m/s it often does not step at all; walking at 0.1 it
# turns at 0.2-0.3 rad/s both ways (duck-sim drive). It also drifts right by itself, ~-5 deg/s.
BODY_VX = float(os.environ.get('DUCKOPHILA_BODY_VX', 0.1))


class Rpc:
    """Newline-delimited JSON-RPC 2.0 on a unix socket — `duck_ipc_proto`'s framing."""

    def __init__(self, path: Path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(str(path))
        self.file = self.sock.makefile('rw')
        self.ids = itertools.count(1)
        self.lock = threading.Lock()

    def send(self, method: str, params=None, request: bool = False):
        msg = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            msg['params'] = params
        if request:
            msg['id'] = next(self.ids)
        with self.lock:
            self.file.write(json.dumps(msg) + '\n')
            self.file.flush()

    def lines(self):
        for raw in self.file:
            yield json.loads(raw)


class Duck:
    """What a startle and a turn are, as intents. Replies on the robot socket are drained and ignored."""

    def __init__(self, path: Path):
        self.rpc = Rpc(path)
        self.flinching_until = 0.0
        threading.Thread(target=lambda: [None for _ in self.rpc.lines()], daemon=True).start()

    def startle(self, zone: int):
        self.flinching_until = time.monotonic() + FLINCH_S
        threading.Thread(target=self._flinch, args=(zone,), daemon=True).start()

    def turn(self, vyaw: float, vx: float = 0.0):
        """A continuous intent, sent every brain step while steering; a flinch outranks it."""
        if time.monotonic() >= self.flinching_until:
            self.rpc.send('robot.move', {'vx': vx, 'vy': 0.0, 'vyaw': vyaw})

    def head(self, joints: list[float]):
        """Head joint targets [neck_pitch, head_pitch, head_yaw, head_roll], continuous."""
        n, p, y, r = joints
        self.rpc.send('robot.head', {'neck_pitch': n, 'head_pitch': p, 'head_yaw': y, 'head_roll': r})

    def _flinch(self, zone: int):
        r = self.rpc
        r.send('robot.sound', {'tag': 'alarm'}, request=True)
        # Look where it came from: the zone's beam, half a metre out, roughly in the trunk frame
        # (the sensor's pitch is ignored — a glance, not a measurement).
        az = math.radians(FOV_DEG / 2 - FOV_DEG / COLS / 2 - (zone % COLS) * FOV_DEG / COLS)
        r.send('robot.look', {'x': 0.5, 'y': 0.5 * math.tan(az), 'z': 0.05}, request=True)
        end = time.monotonic() + FLINCH_S
        while time.monotonic() < end:  # continuous intents: keep them arriving, or the deadman zeroes them
            r.send('robot.pose', {'z': -0.025, 'roll': 0.0, 'pitch': 0.0, 'active': True})
            r.send('robot.move', {'vx': -0.12, 'vy': 0.0, 'vyaw': 0.0})
            time.sleep(0.05)
        r.send('robot.move', {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0})
        r.send('robot.pose', {'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'active': False})


class Body:
    """What the duck knows about itself: `robot.model` once, then the latest `robot.state`.

    Its own connection, separate from the one intents go out on, and opened in `--dry` too: the
    floor filter and the efference copies need it whether or not the duck is being told anything.
    """

    def __init__(self, path: Path):
        self.state: dict = {}
        self.geometry: Geometry | None = None
        self.head_yaw = None
        self._hist: list = []
        self._egos: list = []
        rpc = Rpc(path)
        rpc.send('robot.model', request=True)
        lines = rpc.lines()
        for msg in lines:
            if 'result' in msg or 'error' in msg:
                model = msg.get('result') or {}
                if model.get('tof_beams'):
                    self.geometry = Geometry(model)
                else:
                    print('robot.model gave no ToF geometry: no floor filter', flush=True)
                if 'head_yaw' in model.get('joint_names', []):
                    self.head_yaw = model['joint_names'].index('head_yaw')
                break
        rpc.send('robot.subscribe', {'hz': 50}, request=True)

        def read():
            for msg in lines:
                if msg.get('method') == 'robot.state':
                    self.state = msg['params']
                    self._track(self.state)
        threading.Thread(target=read, daemon=True).start()

    def _track(self, st: dict):
        """Per state: odometry for the velocity, and the camera's orientation in the world for its
        rotation rate. Called on the reader thread, so `ego()` is cheap."""
        o, cam, imu = st.get('odom') or {}, (st.get('frames') or {}).get('camera'), st.get('imu') or {}
        t = st.get('t')
        if t is None or 'position' not in o or not cam or not imu.get('quat'):
            return
        from geometry import quat_matrix
        r_cam = quat_matrix(imu['quat']) @ quat_matrix(cam['quat'])     # camera -> world
        self._hist = [h for h in self._hist if t - h[0] < 0.5] + [(t, np.asarray(o['position'][:2]), o.get('yaw', 0.0), r_cam)]
        # what self-motion looked like at each moment, by arrival time, so a camera frame can be
        # paired with the state of the moment it was captured: paired with the newest state
        # instead it is ~90 ms out, a third of a ~3 Hz stride, and the prediction does not correlate
        e = self.ego()
        if e is not None:
            now = time.time()
            self._egos = [x for x in self._egos if now - x[0] < 1.0] + [(now, e)]

    def velocity(self) -> np.ndarray:
        """The trunk's own velocity, m/s in the trunk frame, from odometry over the last 0.5 s —
        the robot's estimate, not the simulator's truth (they agree to ~10% walking). Half a
        second, because the gait sways the trunk sideways by ~0.1 m/s within every stride."""
        if len(self._hist) < 2 or self._hist[-1][0] - self._hist[0][0] < 0.1:
            return np.zeros(3)
        (t0, p0, _, _), (t1, p1, yaw, _) = self._hist[0], self._hist[-1]
        vw = (p1 - p0) / (t1 - t0)                                    # odometry frame: gravity-aligned
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.array([c * vw[0] + s_ * vw[1], -s_ * vw[0] + c * vw[1], 0.0])

    def camera_pitch_roll_rate(self) -> np.ndarray:
        """The camera's rotation rate, rad/s in the camera frame, with the turn about the vertical
        taken out: the bob and sway of walking and of the head, not the yaw the reflex steers."""
        if len(self._hist) < 2:
            return np.zeros(3)
        from geometry import rotation_rate
        (t0, _, _, r0), (t1, _, _, r1) = self._hist[max(0, len(self._hist) - 4)], self._hist[-1]   # ~60 ms: 2 states are noise
        if t1 - t0 <= 0:
            return np.zeros(3)
        w = rotation_rate(r0, r1, t1 - t0)                            # world frame (IMU: z up)
        w[2] = 0.0
        return r1.T @ w

    def ego(self):
        """What `MotionEncoder.self_flow` needs, or None without a camera pose."""
        st = self.state
        cam = (st.get('frames') or {}).get('camera')
        imu = st.get('imu') or {}
        if not cam or not imu.get('quat') or self.geometry is None:
            return None
        from geometry import quat_matrix
        down = quat_matrix(imu['quat']).T @ np.array([0.0, 0.0, -1.0])
        rot = quat_matrix(cam['quat'])
        height = (st.get('odom') or {}).get('position', [0, 0, 0])[2] or self.geometry.trunk_height
        return rot, np.asarray(cam['pos'], float), down, height, self.velocity(), self.camera_pitch_roll_rate()

    def ego_at(self, unix_time: float):
        """`ego()` as it was at `unix_time` (nearest state by arrival), or None."""
        egos = self._egos
        if not egos:
            return None
        return min(egos, key=lambda x: abs(x[0] - unix_time))[1]

    def head_yaw_rate_deg(self) -> float:
        """How fast the head is turning on its own, deg/s, positive left — the camera's efference copy."""
        v = self.state.get('velocities')
        return math.degrees(v[self.head_yaw]) if v and self.head_yaw is not None else 0.0

    def yaw(self) -> float | None:
        return (self.state.get('odom') or {}).get('yaw')


class Camera:
    """`mediad`'s local frames through `MotionEncoder`, on a thread; the latest rates on demand."""

    def __init__(self, path: Path, body: Body, efference: bool = True):
        from motion import MotionEncoder
        self.path, self.body, self.efference = path, body, efference
        self.enc = MotionEncoder()
        self.rates: dict[int, float] = {}
        self.grey = None
        self.fps = 0.0
        self._t_prev = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _frame(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(str(self.path))
        f = s.makefile('rwb')
        f.write(b'{"jsonrpc":"2.0","id":1,"method":"media.frame"}\n')
        f.flush()
        h = json.loads(f.readline())
        if 'error' in h:
            raise RuntimeError(h['error'])
        h = h['result']
        raw = f.read(h['bytes'])
        s.close()
        y = np.frombuffer(raw, np.uint8)[1::2].reshape(h['height'], h['width'])
        return np.ascontiguousarray(np.rot90(y, k=-h['rotate'] // 90)), h['captured_at_unix_us'] / 1e6

    def _run(self):
        n, t0 = 0, time.monotonic()
        while True:
            try:
                grey, t = self._frame()
            except (OSError, RuntimeError, ValueError) as e:
                print(f'camera: {e}; retrying', flush=True)
                time.sleep(1.0)
                continue
            w = self.body.head_yaw_rate_deg() if self.efference else 0.0
            # walking efference copy: opt-in until experiments/ego_flow_check.py shows it predicts the floor
            # the flow spans this frame and the last, so the state from between their captures
            ego = None if os.environ.get('DUCKOPHILA_NO_EGO') else self.body.ego_at((t + self._t_prev) / 2 - FRAME_LAG_S)
            self._t_prev = t
            self.rates = self.enc.update(grey, t, head_yaw_rate_deg=w, ego=ego)
            self.grey = grey
            n += 1
            if n % 30 == 0:
                self.fps, n, t0 = 30 / (time.monotonic() - t0), 0, time.monotonic()


class Trace:
    """Every brain step, for `video/make_video.py`: wall time, which neurons spiked, the key
    descending neurons' rates, the depth frame, a camera thumbnail, startles and steering."""

    KEY = (N.GIANT_FIBER, 'DNp15', 'DNa02', 'LPLC2')

    def __init__(self, path: Path, brain: FlyBrain):
        self.path, self.rows = path, []
        self.key = {f'{c}_{s[0].upper()}': [brain.index[f] for f in N.ids(c, s)] for c in self.KEY for s in ('left', 'right')}

    def add(self, counts, ms, frame, camera, note, head, steer):
        thumb = camera.grey[::4, ::4].copy() if camera is not None and camera.grey is not None else None
        d = np.where(np.isin(frame['status'], (5, 9)), np.asarray(frame['distance_mm'], np.int16), 0)
        self.rows.append((time.time(), np.flatnonzero(counts).astype(np.int32), counts[np.flatnonzero(counts)].astype(np.int16),
                          {k: counts[v].sum() * 1000.0 / ms / len(v) for k, v in self.key.items()},
                          d, thumb, note or '', None if head is None else float(head[2]), steer))

    def save(self):
        r = self.rows
        if not r:
            return
        spk = [x[1] for x in r]
        thumbs = [x[5] for x in r]
        shape = next((t.shape for t in thumbs if t is not None), (1, 1))
        np.savez_compressed(
            self.path, t=np.array([x[0] for x in r]),
            spk=np.concatenate(spk), spk_n=np.array([len(x) for x in spk]), spk_c=np.concatenate([x[2] for x in r]),
            key_names=np.array(list(self.key)), key=np.array([[x[3][k] for k in self.key] for x in r]),
            depth=np.array([x[4] for x in r]), camera=np.array([t if t is not None else np.zeros(shape, np.uint8) for t in thumbs]),
            note=np.array([x[6] for x in r]), head_yaw=np.array([np.nan if x[7] is None else x[7] for x in r]),
            steer=np.array([np.nan if x[8] is None else x[8] for x in r]))
        print(f'trace: {len(r)} steps -> {self.path}', flush=True)


class View:
    """The live brain in Rerun."""

    CLASS_COLORS = {
        'optic': (70, 110, 200), 'central': (220, 220, 220), 'sensory': (90, 200, 120),
        'visual_projection': (80, 180, 250), 'descending': (255, 90, 60), 'ascending': (240, 200, 60),
        'motor': (255, 60, 200), 'visual_centrifugal': (120, 140, 255),
    }

    def __init__(self, brain: FlyBrain, live: bool = True, record: Path | None = None):
        import rerun as rr
        self.rr = rr
        # the viewer binary ships in the venv's bin/, which is not on PATH when the venv's python
        # is run directly
        os.environ['PATH'] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
        rr.init('duckophila', spawn=live)
        if record is not None:
            # live and to a file at once: the recording replays with `rerun FILE.rrd`
            rr.set_sinks(*([rr.GrpcSink()] if live else []), rr.FileSink(str(record)))
        t = N.table()
        ids = brain.ids
        anchor = t.reindex(ids)[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
        # FlyWire voxels are 4 x 4 x 40 nm; micrometres, with y flipped so dorsal is up
        self.pos = (anchor * [0.004, -0.004, 0.04]).astype(np.float32)
        self.pos[np.isnan(self.pos)] = 0
        cls = t.reindex(ids)['super_class'].fillna('').to_numpy()
        self.col = np.array([self.CLASS_COLORS.get(c, (180, 180, 180)) for c in cls], np.uint8)
        rr.log('brain/all', rr.Points3D(self.pos, colors=np.full((len(ids), 4), (120, 120, 120, 25), np.uint8),
                                        radii=0.6), static=True)
        named = {'GF': N.ids(N.GIANT_FIBER), 'LPLC2': N.ids('LPLC2'), 'LC4': N.ids('LC4'),
                 'DNa01/02': [f for c in N.STEER for f in N.ids(c)], 'MDN': N.ids(N.BACKWARD),
                 'DNp09': N.ids(N.FORWARD), 'HS': [f for c in ('HSN', 'HSE', 'HSS') for f in N.ids(c)],
                 'DNp15 (DNHS1)': N.ids('DNp15')}
        for name, fids in named.items():
            ix = [brain.index[f] for f in fids if f in brain.index]
            rr.log(f'brain/named/{name}', rr.Points3D(self.pos[ix], radii=2.0 if len(ix) < 10 else 1.0,
                                                     labels=[name] * len(ix) if len(ix) < 10 else None), static=True)
        self.watch = {f'{c}_{s[0].upper()}': [brain.index[f] for f in N.ids(c, s)]
                      for c in (N.GIANT_FIBER, *N.STEER, 'DNp15', N.FORWARD, N.BACKWARD) for s in ('left', 'right')}

    def frame(self, t, dist_mm, status, inv_tau, counts, ms, note, floor=None, camera=None, steer=None):
        rr = self.rr
        rr.set_time('t', duration=t)
        spk = np.flatnonzero(counts)
        rr.log('brain/spiking', rr.Points3D(self.pos[spk], colors=self.col[spk],
                                            radii=np.clip(1.0 + counts[spk] * 0.6, 1, 4)))
        rr.log('brain/spiking_count', rr.Scalars(len(spk)))
        d = np.where(np.isin(status, (5, 9)), np.asarray(dist_mm, float), 0).reshape(ROWS, COLS)
        rr.log('tof/depth_mm', rr.DepthImage(d.astype(np.float32), meter=1000.0))
        if floor is not None:
            rr.log('tof/floor', rr.Tensor(np.asarray(floor, np.uint8).reshape(ROWS, COLS)))
        rr.log('tof/looming', rr.Tensor(inv_tau.reshape(ROWS, COLS).astype(np.float32)))
        if camera is not None and camera.grey is not None:
            g = camera.grey[::4, ::4]
            rr.log('camera/image', rr.Image(g))
            fl = camera.enc.flow_deg
            if fl is not None:
                gh, gw = g.shape
                rows, cols = fl.shape[1], fl.shape[2]
                origins, vectors = [], []
                for e in range(2):
                    for r in range(rows):
                        for c in range(cols):
                            origins.append([(e * cols + c + 0.5) * gw / (2 * cols), (r + 0.5) * gh / rows])
                            vectors.append(fl[e, r, c] * 0.5)
                rr.log('camera/image/flow', rr.Arrows2D(origins=origins, vectors=vectors, colors=(255, 200, 0)))
            rr.log('camera/mean_flow_deg_s', rr.Scalars(camera.enc.mean_u))
        for name, ix in self.watch.items():
            rr.log(f'dn/{name}', rr.Scalars(counts[ix].sum() * 1000.0 / ms / len(ix)))
        if steer is not None:
            rr.log('steer/vyaw_rad_s', rr.Scalars(steer))
        if note:
            rr.log('events', rr.TextLog(note, level='WARN'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--duck', default='duck-a')
    ap.add_argument('--steer', choices=('head', 'body', 'dna02'), help='the optomotor reflex (step 2); needs the camera')
    ap.add_argument('--no-startle', action='store_true', help='leave step 1 out: no flinch (the giant fiber is still read)')
    ap.add_argument('--no-viz', action='store_true')
    ap.add_argument('--record', type=Path, help='also save the view to this .rrd file (replay: rerun FILE.rrd)')
    ap.add_argument('--trace', type=Path, help='save every brain step (spikes, key neurons, depth, camera) to this .npz, for video/make_video.py')
    ap.add_argument('--dry', action='store_true', help='do not send the duck anything')
    args = ap.parse_args()

    print('loading the brain (FlyWire 783)…', flush=True)
    brain = FlyBrain(783, seed=0)
    brain.run(1)  # jit
    enc = LoomingEncoder()
    gf = {s: [brain.index[f] for f in N.ids(N.GIANT_FIBER, s)] for s in ('left', 'right')}
    view = None if args.no_viz and not args.record else View(brain, live=not args.no_viz, record=args.record)
    duck = None if args.dry else Duck(STATE / f'{args.duck}.sock')
    body = Body(STATE / f'{args.duck}.sock')
    camera = Camera(STATE / f'{args.duck}-frame.sock', body, efference=args.steer == 'dna02') if args.steer else None
    steer_dn = 'DNa02' if args.steer == 'dna02' else 'DNp15'
    pair = {s: [brain.index[f] for f in N.ids(steer_dn, s)] for s in ('left', 'right')}

    trace = Trace(args.trace, brain) if args.trace else None
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))       # so `kill` still writes the trace
    try:
        loop(args, brain, enc, gf, view, duck, body, camera, steer_dn, pair, trace)
    finally:
        if trace:
            trace.save()


def loop(args, brain, enc, gf, view, duck, body, camera, steer_dn, pair, trace):
    head = None
    tof = Rpc(STATE / f'{args.duck}-tof.sock')
    tof.send('tof.stream', request=True)
    last_t, last_startle, frames, busy = None, -1e9, 0, 0.0
    diff = 0.0                       # smoothed DNa02 R - L, Hz
    yaw0, t_yaw0 = None, None
    print('wired: waiting for depth frames' + (' and camera' if camera else ''), flush=True)
    for msg in tof.lines():
        if msg.get('method') != 'tof.frame':
            if 'result' in msg:
                print('tof.stream:', msg['result'], flush=True)
            continue
        f = msg['params']
        t = f['at_us'] / 1e6
        ms = 1000.0 / 15 if last_t is None else min(max((t - last_t) * 1000.0, 10.0), 500.0)
        last_t = t
        t0 = time.perf_counter()
        floor = ego = None
        if body.geometry is not None and body.state:
            ranges = np.where(np.isin(f['status'], (5, 9)), np.asarray(f['distance_mm'], float) / 1000.0, np.nan)
            floor = body.geometry.floor(ranges, body.state)
            ego = body.geometry.ego_closing(body.state)
        rates = enc.update(f['distance_mm'], f['status'], t, floor=floor, ego=ego)
        if camera is not None:
            rates = {**camera.rates, **rates}
        brain.stimulate(rates)
        counts = brain.run(ms)
        busy += time.perf_counter() - t0
        frames += 1

        # step 1: startle
        gl, gr = int(counts[gf['left']].sum()), int(counts[gf['right']].sum())
        note = None
        if gl + gr and t - last_startle > REFRACTORY_S:
            last_startle = t
            zone = int(np.argmax(enc.inv_tau))
            side = 'left' if gl > gr else 'right' if gr > gl else 'both'
            note = f'STARTLE  giant fiber L{gl} R{gr} ({side}), looming peak zone {zone} (tau^-1 {enc.inv_tau[zone]:.1f}/s)'
            print(f'{t:8.2f}s  {time.strftime("%H:%M:%S")}  {note}', flush=True)
            if duck and not args.no_startle:
                duck.startle(zone)

        # step 2: the optomotor reflex
        vyaw = None
        if camera is not None:
            hz = {s: counts[ix].sum() * 1000.0 / ms / len(ix) for s, ix in pair.items()}
            a = 1.0 - math.exp(-ms / 1000.0 / STEER_TAU_S)
            diff += a * ((hz['right'] - hz['left']) - diff)
            vyaw = float(np.clip(-STEER_GAIN[args.steer] * diff, -STEER_MAX, STEER_MAX))
            if args.steer in ('head', 'body'):
                if head is None and body.state.get('head'):
                    head = list(body.state['head'])      # hold neck, pitch and roll where they are
                if head is not None:
                    dt = ms / 1000.0
                    turn = 0.0
                    if args.steer == 'body':
                        # the body comes round toward the head; the head gives back what the trunk
                        # turned, so the gaze — what the camera, and the fly, sees — holds
                        turn = float(np.clip(HEAD_TO_BODY * head[2], -STEER_MAX, STEER_MAX))
                        gyro_z = ((body.state.get('imu') or {}).get('gyro') or [0, 0, 0])[2]
                        head[2] -= gyro_z * dt
                    head[2] = float(np.clip(head[2] + vyaw * dt, -HEAD_YAW_MAX, HEAD_YAW_MAX))
                    if duck and time.monotonic() >= duck.flinching_until:
                        duck.head(head)
                        if args.steer == 'body':
                            duck.turn(turn, BODY_VX)
            elif duck:
                duck.turn(vyaw, BODY_VX)

        if view:
            view.frame(t, f['distance_mm'], f['status'], enc.inv_tau, counts, ms, note, floor, camera, vyaw)
        if trace:
            trace.add(counts, ms, f, camera, note, head, vyaw)
        if frames % (15 if os.environ.get("DUCKOPHILA_VERBOSE") else 150) == 0:
            line = (f'{t:8.2f}s  {frames} frames, brain {busy / frames * 1000:.1f} ms per frame, '
                    f'{brain.active} neurons active, '
                    f'{"no floor filter" if floor is None else f"{int(np.sum(floor))} floor zones"}')
            if camera is not None:
                yaw = body.yaw()
                rate = ''
                if yaw is not None:
                    if yaw0 is not None:
                        dy = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))
                        rate = f', duck yaw rate {math.degrees(dy) / (time.monotonic() - t_yaw0):+5.1f} deg/s'
                    yaw0, t_yaw0 = yaw, time.monotonic()
                line += (f', camera {camera.fps:.0f} fps, flow {camera.enc.mean_u:+5.1f} deg/s, '
                         f'{steer_dn} R-L {diff:+5.1f} Hz -> {vyaw:+.2f} rad/s'
                         + (f', head yaw {head[2]:+.2f} rad (measured {body.state["joints"][body.head_yaw]:+.2f})'
                            if head is not None and body.head_yaw is not None else '') + rate)
            print(line, flush=True)


if __name__ == '__main__':
    main()
