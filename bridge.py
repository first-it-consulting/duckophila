"""The fly brain, wired to a duck. Stages 1-4 of microduck's `docs/ideas/fly-brain.md`.

    .venv/bin/python -I bridge.py              # startle only (step 1), with the live brain view
    .venv/bin/python -I bridge.py --steer head  # and the head optomotor reflex (step 2) — needs the camera
    .venv/bin/python -I bridge.py --steer body  # the head reflex, and the walking body following the head
    .venv/bin/python -I bridge.py --steer dna02 # the body turned by DNa02 directly, for comparison
    .venv/bin/python -I bridge.py --steer follow  # follow me: a small target -> LC10a -> DNa02 -> walk toward it
    .venv/bin/python -I bridge.py --steer follow --learn   # and only what it was petted for (pet.py)
    .venv/bin/python -I bridge.py --no-viz     # without the viewer
    .venv/bin/python -I bridge.py --dry        # brain and view only; send the duck nothing
    .venv/bin/python -I bridge.py --record run.rrd   # and save the view to replay or share
    .venv/bin/python -I bridge.py --link ws://duck.local:8443 --steer follow   # a duck on the LAN, over WebRTC

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
  * `--steer follow` (step 3, "follow me"): no optic flow — the camera's target bearing
    (`follow.py`'s detector) onto LC10a, and DNa02's left-minus-right -> `robot.move` vyaw while
    walking at 0.1 m/s; standing while no target is in view. The head is held straight, so the
    camera looks where the body does and `robotd`'s idle look-around stands down.
  * `--learn` (stage 4, with `--steer follow`): the mushroom body learns (`learning.py`, all
    ours but the wiring): the target in view is a cue — the ball or the bar, each its own sparse
    set of Kenyon cells — petting (`pet.py`, on port 7897) drives the reward dopamine neurons, and
    the rule depresses KC -> MBON synapses where the two coincide. The duck follows a target only
    once its value has risen; otherwise it stands and looks. The antennal lobe is silenced for it.
Which way and how hard come from the fly; the gains, the smoothing and the clamps are ours. A
flinch outranks steering while it lasts.

The brain runs on the ToF clock (~15 Hz) for exactly the time since the previous frame, with the
latest camera rates; the camera thread keeps them fresh at ~30 Hz. The sockets and the JSON-RPC
are the robot's own, so the same script runs on a duck against its local sockets — or, with
`--link ws://<duck>:8443`, anywhere on its network over the WebRTC session the robot publishes
(`link.py`): the camera as an H.264 track, everything else over the `control` datachannel, and
each frame dated on the robot's clock so the efference copies still pair it with its own state.

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
# follow: DNa02 answers a small target sparsely (a few Hz of L - R at 10 deg), so more gain, more smoothing
STEER_GAIN['follow'] = float(os.environ.get('DUCKOPHILA_FOLLOW_GAIN', 0.05))
# --steer body: rad/s of body turn per rad of head yaw — the head is recentred in ~1/k seconds
HEAD_TO_BODY = float(os.environ.get('DUCKOPHILA_HEAD_TO_BODY', 1.0))
# How much older a camera frame is than its `captured_at` says, against the robot.state stream.
# Measured in duck-sim (`experiments/ego_lag.py`: the walking efference copy correlates best,
# r = 0.90, with the state 0-0.02 s before the stamp); a real duck's pipeline needs its own.
FRAME_LAG_S = float(os.environ.get('DUCKOPHILA_FRAME_LAG_S', 0.01))
# Over WebRTC (`--link`) a frame is dated by RTCP on the robot's own clock, and states by their t_ns,
# so the network drops out. Measured in duck-sim with `ego_lag.py --link`: the correlation peaks with
# the state 0.04 s before the frame's date (r = 0.65; locally, same run, also 0.04 s, r = 0.86 — the
# H.264 track costs the flow some precision). A real duck's camera pipeline needs its own.
LINK_FRAME_LAG_S = float(os.environ.get('DUCKOPHILA_LINK_FRAME_LAG_S', 0.04))
STEER_TAU_S = {'follow': float(os.environ.get('DUCKOPHILA_FOLLOW_TAU_S', 0.4))}   # smoothing of the R - L difference
STEER_MAX = 1.0          # rad/s
GOAL_MARGIN_DEG = float(os.environ.get('DUCKOPHILA_GOAL_MARGIN_DEG', 10.0))   # --readout composite: goal heading (0: off)
COMPOSITE_SCALE_WALK = float(os.environ.get('DUCKOPHILA_COMPOSITE_SCALE_WALK', 0.15))   # while walking after a person
ADAPT_S = float(os.environ.get('DUCKOPHILA_ADAPT_S', 0.0))   # --readout composite: adaptation (0: off; it rebounds)
def lean_k(brain_kind: str) -> float:
    """--readout composite: the readout's lean per Hz of LC10a drive for this brain, calibrated on that
    brain (experiments/lean.py -> results/lean.json), so a control is corrected by its own lean."""
    if os.environ.get('DUCKOPHILA_LEAN_K'):
        return float(os.environ['DUCKOPHILA_LEAN_K'])
    here = Path(__file__).resolve().parent
    for f in (here / 'results' / 'lean.json', here / 'paper' / 'data' / 'lean.json'):   # yours, or the shipped one
        if f.exists() and brain_kind in (d := json.loads(f.read_text())):
            return float(d[brain_kind]['k'])
    print(f'no lean calibrated for the {brain_kind} brain (experiments/lean.py): none taken off', flush=True)
    return 0.0
COMPOSITE_SCALE = float(os.environ.get('DUCKOPHILA_COMPOSITE_SCALE', 0.3))    # the composite's slope is ~5x DNa02's (24 against 4.6 Hz per 10 deg): the same turn per degree
READOUT = {'dna02': [('DNa02', 1.0)],
           'composite': [('DNa02', 1.0), ('DNae002', 1.0), ('DNa16', 1.0), ('DNg97', -1.0)]}
FOLLOW_MEMORY_S = float(os.environ.get('DUCKOPHILA_FOLLOW_MEMORY_S', 0.8))   # --detector moving: walk on this long
PURSUIT_GATE_DEG = float(os.environ.get('DUCKOPHILA_PURSUIT_GATE_DEG', 20.0))   # no flinch at what is followed...
PURSUIT_MEMORY_S = float(os.environ.get('DUCKOPHILA_PURSUIT_MEMORY_S', 1.0))   # ...seen this recently
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


def connect(where):
    """A JSON-RPC connection: a unix socket's path, or a channel of a WebRTC session (`link.RtcLink`)."""
    return Rpc(where) if isinstance(where, Path) else where.channel()


class Duck:
    """What a startle and a turn are, as intents. Replies on the robot socket are drained and ignored."""

    def __init__(self, where):
        self.rpc = connect(where)
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

    def __init__(self, where):
        # over WebRTC, states are keyed by the robot's clock (t_ns), as the frames are; locally by arrival
        self.robot_clock = not isinstance(where, Path)
        self.state: dict = {}
        self.geometry: Geometry | None = None
        self.head_yaw = None
        self._hist: list = []
        self._egos: list = []
        rpc = connect(where)
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
            now = st['t_ns'] / 1e9 if self.robot_clock and st.get('t_ns') else time.time()
            self._egos = [x for x in self._egos if now - x[0] < 1.0] + [(now, e)]
            # the camera's heading in the world, on the same clock, for the moving detector's turn
            a = r_cam[:, int(np.argmin(np.abs(r_cam[2])))]          # its most horizontal axis
            self._headings = [x for x in getattr(self, '_headings', []) if now - x[0] < 1.0] + [(now, math.atan2(a[1], a[0]))]
            # and its whole pose (camera->world rotation, trunk position), for egoflow.field_between
            self._poses = [x for x in getattr(self, '_poses', []) if now - x[0] < 1.0] + \
                [(now, r_cam, np.asarray(o['position'][:3] if len(o['position']) > 2 else list(o['position'][:2]) + [0.0], float))]

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
        """`ego()` as it was at `unix_time` (nearest state by arrival — or, over WebRTC, by the
        robot's monotonic clock, which is what a frame's time is then on), or None."""
        egos = self._egos
        if not egos:
            return None
        return min(egos, key=lambda x: abs(x[0] - unix_time))[1]

    def camera_pose_at(self, when: float):
        """(camera->world rotation, trunk position) at `when`, on `ego_at`'s clock, or None."""
        p = getattr(self, '_poses', None)
        return min(p, key=lambda x: abs(x[0] - when))[1:] if p else None

    def camera_heading_at(self, when: float) -> float | None:
        """The camera's heading in the world (rad, + = left) at `when`, on `ego_at`'s clock, or None."""
        h = getattr(self, '_headings', None)
        return min(h, key=lambda x: abs(x[0] - when))[1] if h else None

    def head_yaw_rate_deg(self) -> float:
        """How fast the head is turning on its own, deg/s, positive left — the camera's efference copy."""
        v = self.state.get('velocities')
        return math.degrees(v[self.head_yaw]) if v and self.head_yaw is not None else 0.0

    def yaw(self) -> float | None:
        return (self.state.get('odom') or {}).get('yaw')


class Camera:
    """`mediad`'s local frames through `MotionEncoder`, on a thread; the latest rates on demand."""

    def __init__(self, where, body: Body, efference: bool = True, follow: bool = False, detector: str = 'white'):
        self.path, self.body, self.efference = where, body, efference
        self.lag = FRAME_LAG_S if isinstance(where, Path) else LINK_FRAME_LAG_S
        self.follow = follow
        if follow:
            from follow import FollowEncoder
            self.enc = FollowEncoder(detector=detector)
        else:
            from motion import MotionEncoder
            self.enc = MotionEncoder()
        self.rates: dict[int, float] = {}
        self.grey = None
        self.fps = 0.0
        self._t_prev = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _frame(self):
        if not isinstance(self.path, Path):
            while True:                   # until the first RTCP sender report, a frame has no date: skip it
                grey, t, _ = self.path.next_frame()
                if t is not None:
                    return grey, t
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
            except (OSError, RuntimeError, ValueError, TimeoutError) as e:
                print(f'camera: {e}; retrying', flush=True)
                time.sleep(1.0)
                continue
            w = self.body.head_yaw_rate_deg() if self.efference else 0.0
            # walking efference copy: opt-in until experiments/ego_flow_check.py shows it predicts the floor
            # the flow spans this frame and the last, so the state from between their captures
            ego = None if os.environ.get('DUCKOPHILA_NO_EGO') else self.body.ego_at((t + self._t_prev) / 2 - self.lag)
            self._t_prev = t
            gyro_z = ((self.body.state.get('imu') or {}).get('gyro') or [0, 0, 0])[2]
            cam_yaw = math.degrees(gyro_z) + self.body.head_yaw_rate_deg()
            # the camera's turn between this frame's capture and the one the moving detector compares it
            # with, from the state history at those moments: the gyro now is ~2 frames ahead of the image,
            # a large error on a gait that rocks the duck at ~3 Hz
            self._ftimes = (getattr(self, '_ftimes', []) + [t])[-8:]
            cam_turn = None
            if self.follow and len(self._ftimes) > 3:
                h1, h0 = self.body.camera_heading_at(t - self.lag), self.body.camera_heading_at(self._ftimes[-4] - self.lag)
                if h1 is not None and h0 is not None:
                    cam_turn = math.degrees(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)))
            # the moving detector takes the camera's turn from the gyro (an image estimate locks onto a large
            # target and cancels it) and the gait's bob from the image; DUCKOPHILA_IMAGE_SHIFT=1: both from the image
            extra = {'camera_turn_deg': None if os.environ.get('DUCKOPHILA_IMAGE_SHIFT') else cam_turn} if self.follow else {}
            if self.follow and cam_turn is not None and not os.environ.get('DUCKOPHILA_NO_FLOOR_FLOW'):
                # the whole field of the duck's own motion over the compared interval: its turn, and the
                # floor sliding under the camera as it walks (egoflow) -- so it can walk after a person
                e0 = self.body.ego_at((t + self._ftimes[-4]) / 2 - self.lag)
                if e0 is not None:
                    if getattr(self, '_egowarp', None) is None or self._egowarp.w != grey.shape[1]:
                        from egoflow import EgoWarp
                        self._egowarp = EgoWarp(grey.shape[1], grey.shape[0])
                    p0 = self.body.camera_pose_at(self._ftimes[-4] - self.lag)
                    p1 = self.body.camera_pose_at(t - self.lag)
                    mode = os.environ.get('DUCKOPHILA_FIELD', 'yaw')   # 'yaw': turn + walk only; 'pose': full poses; 'rate'
                    if mode == 'pose' and p0 is not None and p1 is not None:
                        extra['camera_field'] = self._egowarp.field_between(e0, p0[0], p1[0], p1[1] - p0[1])
                        Rr = p1[0].T @ p0[0]
                        self._rotdbg = [round(float(x), 4) for x in (Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1])]
                    else:
                        e_use = e0 if mode == 'rate' else (*e0[:5], None)    # 'yaw': no pitch/roll rate
                        extra['camera_field'] = self._egowarp.field(e_use, t - self._ftimes[-4], math.radians(cam_turn))
                    self._fdbg = {'v': [round(float(x), 3) for x in e0[4]], 'wpr': [round(float(x), 3) for x in e0[5]],
                                  'dt': round(t - self._ftimes[-4], 3), 'turn': round(cam_turn, 2),
                                  'fmed': [round(float(np.median(np.abs(extra['camera_field'][..., k]))), 2) for k in (0, 1)]}
            self.rates = self.enc.update(grey, t, head_yaw_rate_deg=w, ego=ego, **extra)
            if self.follow and os.environ.get('DUCKOPHILA_FOLLOW_REC') and len(getattr(self, '_ftimes', [])) > 3:
                # every frame with the robot state at its and the compared frame's capture, for offline checks
                import pickle as _pk
                rec = {'t': t, 't0': self._ftimes[-4], 'grey': grey, 'cam_turn': cam_turn,
                       'e0': self.body.ego_at((t + self._ftimes[-4]) / 2 - self.lag),
                       'p0': self.body.camera_pose_at(self._ftimes[-4] - self.lag), 'p1': self.body.camera_pose_at(t - self.lag)}
                with open(os.environ['DUCKOPHILA_FOLLOW_REC'], 'ab') as fh:
                    _pk.dump(rec, fh)
            if self.follow and os.environ.get('DUCKOPHILA_FOLLOW_DUMP') and n % 30 == 0:   # frames + raw motion, for debugging
                dd = getattr(self.enc.detect, 'det', self.enc.detect)
                if getattr(dd, 'mask', None) is not None:
                    import cv2 as _cv2
                    v = _cv2.cvtColor(grey, _cv2.COLOR_GRAY2BGR); v[dd.mask > 0] = (0, 0, 255)
                    _cv2.imwrite(os.path.join(os.environ['DUCKOPHILA_FOLLOW_DUMP'], f'{time.time():.2f}.jpg'), _cv2.resize(v, (180, 320)))
            if self.follow and os.environ.get('DUCKOPHILA_FOLLOW_LOG'):     # per frame, for debugging the detector
                d = self.enc.detect
                with open(os.environ['DUCKOPHILA_FOLLOW_LOG'], 'a') as fh:
                    fh.write(json.dumps({'t': t, 'target': self.enc.target, 'cam_yaw': cam_yaw, 'cam_turn': cam_turn,
                                         'shift': list(getattr(d, 'shift', (0, 0))),
                                         'mask': float(d.mask.mean()) if getattr(d, 'mask', None) is not None else None,
                                         'raw': float(d.det.mask.mean()) if getattr(getattr(d, 'det', None), 'mask', None) is not None else None,
                                         'xy': list((self.body.state.get('odom') or {}).get('position', [0, 0])[:2]),
                                         'run': getattr(d, 'run', None), 'shape': list(grey.shape), 'f': getattr(self, '_fdbg', None), 'rot': getattr(self, '_rotdbg', None)}) + '\n')
            self.grey = grey
            n += 1
            if n % 30 == 0:
                self.fps, n, t0 = 30 / (time.monotonic() - t0), 0, time.monotonic()


SEARCH_S = float(os.environ.get('DUCKOPHILA_SEARCH_S', 5.0))           # --detector population: look for a lost target this long
SEARCH_RATE = float(os.environ.get('DUCKOPHILA_SEARCH_RATE', 0.6))     # rad/s
SEARCH_MIN = float(os.environ.get('DUCKOPHILA_SEARCH_MIN', 0.05))       # rad/s of the brain's recent turn to search on
SEARCH_MAX_DEG = float(os.environ.get('DUCKOPHILA_SEARCH_MAX_DEG', 60.0))
KEEP_FAR_M = float(os.environ.get('DUCKOPHILA_KEEP_FAR_M', 0.8))
KEEP_NEAR_M = float(os.environ.get('DUCKOPHILA_KEEP_NEAR_M', 0.65))
KEEP_VX = float(os.environ.get('DUCKOPHILA_KEEP_VX', 0.15))


def target_range(f, floor, pursuit, cols_each_side: int = 1):
    """The nearest ToF return (m) in the columns toward the target followed, the floor left out, or None."""
    if pursuit is None:
        return None
    col = int(round((FOV_DEG / 2 - FOV_DEG / COLS / 2 - pursuit[0]) / (FOV_DEG / COLS)))
    cols = [c for c in range(col - cols_each_side, col + cols_each_side + 1) if 0 <= c < COLS]
    if not cols:
        return None
    d = np.where(np.isin(f['status'], (5, 9)), np.asarray(f['distance_mm'], float) / 1000.0, np.nan).reshape(ROWS, COLS)
    if floor is not None:
        d = np.where(np.asarray(floor, bool).reshape(ROWS, COLS), np.nan, d)
    v = d[:, cols]
    return float(np.nanmin(v)) if np.isfinite(v).any() else None


class Saccades:
    """Follow by saccades (`--saccades`), not by a continuous turn. Ours, after the fly.

    A fly tracks a moving object less by turning smoothly than by body saccades, each one fired when
    the visual evidence for the object being off to one side has added up (Mongeau & Frye 2017,
    Curr. Biol.: an integrate-and-fire account of object-tracking saccades in Drosophila). It suits
    this body: velstand ignores a small turn on the spot, and a stepping person gives DNa02 a weak,
    noisy left-right difference that a continuous turn passes straight on. So DNa02's left minus
    right (Hz) is integrated, leaking with `tau_s`; crossing `theta` (Hz s) fires a saccade toward that
    side, `rate` rad/s for `dur_s` (~20 degrees), and starts the integral over. During a saccade
    nothing is integrated: the camera's own turn would read as the world moving (flies, too, damp
    their vision during saccades). Between saccades the duck does not turn."""

    def __init__(self, readout: str = 'dna02'):
        e = os.environ.get
        self.theta = float(e('DUCKOPHILA_SACC_THETA', 24.0 if readout == 'composite' else 3.0))
        self.tau_s = float(e('DUCKOPHILA_SACC_TAU_S', 1.0))
        self.rate = float(e('DUCKOPHILA_SACC_RATE', 1.0))
        self.dur_s = float(e('DUCKOPHILA_SACC_S', 0.35))
        self.acc, self.until, self.dir, self.n = 0.0, -1e9, 0.0, 0

    def step(self, left_minus_right_hz: float, dt: float, now: float, seen: bool) -> float:
        if now < self.until:
            return self.dir * self.rate
        self.acc = self.acc * math.exp(-dt / self.tau_s) + left_minus_right_hz * dt
        if seen and abs(self.acc) > self.theta:
            self.dir, self.until, self.acc, self.n = math.copysign(1.0, self.acc), now + self.dur_s, 0.0, self.n + 1
            print(f'saccade {"left" if self.dir > 0 else "right"} (#{self.n})', flush=True)
            return self.dir * self.rate
        return 0.0


class Trace:
    """Every brain step, for `video/make_video.py`: wall time, which neurons spiked, the key
    descending neurons' rates, the depth frame, a camera thumbnail, startles and steering."""

    KEY = (N.GIANT_FIBER, 'DNp15', 'DNa02', 'LPLC2', 'LC10a')

    def __init__(self, path: Path, brain: FlyBrain):
        self.path, self.rows = path, []
        self.key = {f'{c}_{s[0].upper()}': [brain.index[f] for f in N.ids(c, s)] for c in self.KEY for s in ('left', 'right')}

    def add(self, counts, ms, frame, camera, note, head, steer, learn=None):
        """`learn`, with --learn: (petting now, the ball's learned value, the bar's) — None while naive."""
        thumb = camera.grey[::4, ::4].copy() if camera is not None and camera.grey is not None else None
        d = np.where(np.isin(frame['status'], (5, 9)), np.asarray(frame['distance_mm'], np.int16), 0)
        self.rows.append((time.time(), np.flatnonzero(counts).astype(np.int32), counts[np.flatnonzero(counts)].astype(np.int16),
                          {k: counts[v].sum() * 1000.0 / ms / len(v) for k, v in self.key.items()},
                          d, thumb, note or '', None if head is None else float(head[2]), steer,
                          learn or (False, None, None)))

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
            steer=np.array([np.nan if x[8] is None else x[8] for x in r]),
            petting=np.array([bool(x[9][0]) for x in r]),
            value_ball=np.array([np.nan if x[9][1] is None else x[9][1] for x in r]),
            value_bar=np.array([np.nan if x[9][2] is None else x[9][2] for x in r]))
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
            fl = getattr(camera.enc, 'flow_deg', None)
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
            if hasattr(camera.enc, 'mean_u'):
                rr.log('camera/mean_flow_deg_s', rr.Scalars(camera.enc.mean_u))
            tg = getattr(camera.enc, 'target', None)
            if tg is not None:
                rr.log('camera/target_bearing_deg', rr.Scalars(tg[0]))
        for name, ix in self.watch.items():
            rr.log(f'dn/{name}', rr.Scalars(counts[ix].sum() * 1000.0 / ms / len(ix)))
        if steer is not None:
            rr.log('steer/vyaw_rad_s', rr.Scalars(steer))
        if note:
            rr.log('events', rr.TextLog(note, level='WARN'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--duck', default='duck-a')
    ap.add_argument('--steer', choices=('head', 'body', 'dna02', 'follow'),
                    help='the optomotor reflex (step 2), or follow me (step 3); needs the camera')
    ap.add_argument('--brain', choices=('intact', 'shuffled', 'flipped'), default='intact',
                    help='a null brain for controls: wiring shuffled, or signs flipped (flysim.FlyBrain.null)')
    ap.add_argument('--no-startle', action='store_true', help='leave step 1 out: no flinch (the giant fiber is still read)')
    ap.add_argument('--no-viz', action='store_true')
    ap.add_argument('--record', type=Path, help='also save the view to this .rrd file (replay: rerun FILE.rrd)')
    ap.add_argument('--trace', type=Path, help='save every brain step (spikes, key neurons, depth, camera) to this .npz, for video/make_video.py')
    ap.add_argument('--dry', action='store_true', help='do not send the duck anything')
    ap.add_argument('--link', metavar='WS_URL', help="a duck on the network, over WebRTC: its signalling server, ws://<duck>:8443")
    ap.add_argument('--readout', choices=('dna02', 'composite'), default=None,
                    help='--steer follow: the turn from DNa02, or from the four DN pairs that follow a bearing best (READOUT); '
                         'default composite with --detector population, else dna02')
    ap.add_argument('--saccades', action='store_true', help='--steer follow: turn by saccades on integrated DNa02 (class Saccades)')
    ap.add_argument('--detector', choices=('white', 'moving', 'population'), default='white',
                    help="--steer follow's target detector: the sim's glowing target, or follow.MovingTarget, the real-world one")
    ap.add_argument('--learn', action='store_true', help='stage 4: follow only what it was petted for (with --steer follow; pet.py)')
    args = ap.parse_args()
    if args.readout is None:
        args.readout = 'composite' if args.detector == 'population' else 'dna02'

    print('loading the brain (FlyWire 783)…', flush=True)
    brain = FlyBrain(783, seed=0)
    brain.run(1)  # jit
    if args.brain != 'intact':
        brain = brain.null(args.brain, seed=1)
        print(f'NULL BRAIN: {args.brain} (a control: same neurons and weights, wired wrong)', flush=True)
    learner = None
    if args.learn:
        if args.steer != 'follow':
            ap.error('--learn needs --steer follow')
        from learning import Learner, MushroomBody, anosmic
        brain = anosmic(brain)
        learner = Learner(brain, MushroomBody(brain))
        serve_petting(learner)
        print(f'learning: antennal lobe silenced; petting on 127.0.0.1:{PET_PORT} (pet.py)', flush=True)
    enc = LoomingEncoder()
    gf = {s: [brain.index[f] for f in N.ids(N.GIANT_FIBER, s)] for s in ('left', 'right')}
    view = None if args.no_viz and not args.record else View(brain, live=not args.no_viz, record=args.record)
    if args.link:
        from link import RtcLink
        link = RtcLink(args.link)
        robot_at, tof_at, frame_at = link, link, link
        print(f'WebRTC session with {args.link}: camera rotated {link.rotate}°, frames dated on the robot clock'
              + ('' if link.clock_offset_ns is not None else ' — NOT: this mediad gives no clocks'), flush=True)
    else:
        robot_at, tof_at, frame_at = STATE / f'{args.duck}.sock', STATE / f'{args.duck}-tof.sock', STATE / f'{args.duck}-frame.sock'
    duck = None if args.dry else Duck(robot_at)
    body = Body(robot_at)
    camera = (Camera(frame_at, body, efference=args.steer == 'dna02', follow=args.steer == 'follow', detector=args.detector)
              if args.steer else None)
    steer_dn = 'DNa02' if args.steer in ('dna02', 'follow') else 'DNp15'
    pair = {s: [brain.index[f] for f in N.ids(steer_dn, s)] for s in ('left', 'right')}
    # --readout composite: the turn from the DN pairs whose left-right difference followed a target's
    # bearing best open loop (results/follow_dn.json, |slope| > 3 Hz per 10 deg), sign included
    readout = [(sg, [brain.index[f] for f in N.ids(n, 'left')], [brain.index[f] for f in N.ids(n, 'right')])
               for n, sg in READOUT[args.readout]] if args.steer == 'follow' else []
    args.readout_ix = readout
    args.lean_k = lean_k(args.brain) if args.readout != 'dna02' else 0.0

    trace = Trace(args.trace, brain) if args.trace else None
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))       # so `kill` still writes the trace
    try:
        loop(args, brain, enc, gf, view, duck, body, camera, steer_dn, pair, trace, learner, tof_at)
    finally:
        if trace:
            trace.save()


PET_PORT = 7897


def serve_petting(learner):
    """`pet.py`'s port: "pet <seconds>" -> the duck is being petted for that long."""
    import socketserver

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            for raw in self.rfile:
                try:
                    secs = float(raw.split()[1])
                    learner.pet(secs, time.monotonic())
                    print(f'{time.strftime("%H:%M:%S")}  PETTED for {secs:.1f} s', flush=True)
                    self.wfile.write(f'petting for {secs:.1f} s\n'.encode())
                except (IndexError, ValueError):
                    self.wfile.write(b'usage: pet <seconds>\n')

    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = S(('127.0.0.1', PET_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def loop(args, brain, enc, gf, view, duck, body, camera, steer_dn, pair, trace, learner=None, tof_at=None):
    head = None
    tof = connect(tof_at or STATE / f'{args.duck}-tof.sock')
    tof.send('tof.stream', request=True)
    last_t, last_startle, frames, busy = None, -1e9, 0, 0.0
    diff = 0.0                       # smoothed DNa02 R - L, Hz
    last_seen = -1e9                 # when the moving detector last had a target
    pursuit = None                   # (bearing deg, monotonic time) of the target last followed
    sacc = Saccades(args.readout)
    comp_diff = 0.0                  # --readout composite, smoothed
    lr_base = 0.0                    # --readout composite: the adapted-to lean
    goal = None                      # --readout composite: world heading (rad) where the target was last seen
    walking, last_range = False, None  # --detector population: keeping its distance
    brain_turn, search_from = 0.0, None  # --detector population: the brain's recent turn, and where a search began
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
        now = time.monotonic()
        kind = None
        if learner is not None:
            tg = camera.enc.target
            kind = tg[3] if tg else None
            rates = {**rates, **learner.rates(kind, now)}
        brain.stimulate(rates)
        counts = brain.run(ms)
        if learner is not None:
            learner.after(counts, ms, kind, now)
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
            # pursuit gate (ours): what the duck is following cannot be what it flees. A person's legs
            # at 0.6 m loom on the ToF as the duck steps after them, ~8 flinches a run, each one
            # breaking the following. A flinch whose looming peaks within PURSUIT_GATE_DEG of the
            # target followed in the last PURSUIT_MEMORY_S is held back; the giant fiber still fires
            # and is logged, and a ball from anywhere else still startles.
            zone_az = FOV_DEG / 2 - FOV_DEG / COLS / 2 - (zone % COLS) * FOV_DEG / COLS
            pursued = (args.detector != 'white' and pursuit is not None
                       and time.monotonic() - pursuit[1] < PURSUIT_MEMORY_S
                       and abs(zone_az - pursuit[0]) < PURSUIT_GATE_DEG)
            if pursued:
                note += '  | HELD BACK (pursuit gate)'   # the trace says so; make_video labels it
                print(f'{t:8.2f}s  ... held back: the looming is the target being followed '
                      f'(zone {zone_az:+.0f} deg, target {pursuit[0]:+.0f} deg)', flush=True)
            elif duck and not args.no_startle:
                if pursuit is not None and args.detector != 'white':
                    print(f'{t:8.2f}s  ... flinch: zone {zone_az:+.0f} deg, target {pursuit[0]:+.0f} deg '
                          f'{time.monotonic() - pursuit[1]:.1f} s ago', flush=True)
                duck.startle(zone)

        # step 2: the optomotor reflex
        vyaw = None
        if camera is not None:
            hz = {s: counts[ix].sum() * 1000.0 / ms / len(ix) for s, ix in pair.items()}
            a = 1.0 - math.exp(-ms / 1000.0 / STEER_TAU_S.get(args.steer, 0.15))
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
            elif args.steer == 'follow':
                if head is None and body.state.get('head'):
                    head = list(body.state['head'])
                    head[2] = 0.0                       # straight ahead: the camera looks where the body goes
                if duck and head is not None and time.monotonic() >= duck.flinching_until:
                    duck.head(head)
                    seen = camera.enc.target is not None
                    if args.detector in ('moving', 'population'):
                        # a stepping person is seen in bursts; walk-stop-walk at the brain's rate is a
                        # command velstand never takes its first step on. Keep walking FOLLOW_MEMORY_S
                        # after the target was last seen (ours; the glowing target is never intermittent)
                        if seen:
                            last_seen = now
                            pursuit = (camera.enc.target[0], time.monotonic())
                        seen = now - last_seen < FOLLOW_MEMORY_S
                    if learner is not None:
                        seen = seen and learner.approach(kind)
                    lr = sum(sg * (counts[l].sum() - counts[r].sum()) for sg, l, r in args.readout_ix) * 1000.0 / ms
                    if args.readout != 'dna02':
                        # the lean (ours, calibrated open loop on synthetic targets dead ahead): the readout
                        # leans left in proportion to how much LC10a is driven at all (+13 Hz with a person
                        # ahead; the two eyes' LC10a are not mirror images). Taken off, it cannot rebound.
                        lr -= args.lean_k * sum(camera.rates.get(f, 0.0) for f in camera.enc.ids)
                    if args.readout != 'dna02' and ADAPT_S > 0:
                        # adaptation (ours): the readout leans left whenever LC10a is driven at all
                        # (+13 Hz with a person dead ahead; the two eyes' LC10a are not mirror images),
                        # and a constant lean is a constant turn. What stays the same fades with ADAPT_S.
                        lr_base += (ms / 1000.0 / ADAPT_S) * (lr - lr_base)
                        lr -= lr_base
                    if args.saccades:
                        vyaw = sacc.step(lr, ms / 1000.0, now, seen)
                    elif args.readout != 'dna02':
                        comp_diff += (1.0 - math.exp(-ms / 1000.0 / STEER_TAU_S['follow'])) * (lr - comp_diff)
                        scale = COMPOSITE_SCALE_WALK if walking else COMPOSITE_SCALE   # the gait adds delay: less while walking
                        vyaw = float(np.clip(STEER_GAIN['follow'] * scale * comp_diff, -STEER_MAX, STEER_MAX))
                    if args.readout != 'dna02' and GOAL_MARGIN_DEG > 0:
                        # goal heading (ours, after the central complex's compass): where in the world the
                        # target was last seen. The duck may turn toward it, and past it by GOAL_MARGIN_DEG,
                        # but not away from it: a turn the brain keeps up after a fast turn of its own (its DNs
                        # fire on for a moment) otherwise carries it round, a spin, the target long gone.
                        yaw_now = body.yaw()
                        if yaw_now is not None:
                            if camera.enc.target is not None:
                                goal = yaw_now + math.radians(camera.enc.target[0])
                            if goal is not None:
                                rem = math.degrees(math.atan2(math.sin(goal - yaw_now), math.cos(goal - yaw_now)))
                                if vyaw * rem < 0 and abs(rem) > GOAL_MARGIN_DEG:
                                    vyaw = 0.0
                    # the real-world follow turns on the spot: walking at a person makes the floor flow through
                    # the lower field, where the feet are, and the intact brain follows its own flow round
                    if args.detector == 'population':
                        # keep its distance (ours): walk while the person, by the ToF zones toward the target,
                        # is further than KEEP_FAR_M, stop nearer than KEEP_NEAR_M (velstand will not take a
                        # first step much below 0.15 m/s, so it walks or stands). With the room's own motion
                        # taken off as a homography, walking no longer feeds the encoder the floor.
                        d = target_range(f, floor, pursuit)
                        if d is not None:
                            walking = d > KEEP_FAR_M or (walking and d > KEEP_NEAR_M)
                        vx = float(os.environ['DUCKOPHILA_BODY_VX']) if os.environ.get('DUCKOPHILA_BODY_VX') else (KEEP_VX if walking else 0.0)
                        last_range = d
                    else:
                        vx = BODY_VX
                    if os.environ.get('DUCKOPHILA_FOLLOW_FORCE'):     # verification only: walk regardless of the brain
                        fx, fy = (float(v) for v in os.environ['DUCKOPHILA_FOLLOW_FORCE'].split(','))
                        duck.turn(fy, fx)
                    elif not seen and args.detector == 'population' and now - last_seen < SEARCH_S \
                            and abs(brain_turn) > SEARCH_MIN and body.yaw() is not None:
                        # lost (ours, after the flies of Neuser et al. 2008, which keep heading for an object that
                        # has vanished): keep turning the way the *brain* was turning when it lost the target, on the
                        # spot, at most SEARCH_MAX_DEG, until it is in view again. The way is the brain's, not the
                        # encoder's: a rewired brain, which does not turn toward the target, has none to keep.
                        if search_from is None:
                            search_from = body.yaw()
                        gone = abs(math.degrees(math.atan2(math.sin(body.yaw() - search_from), math.cos(body.yaw() - search_from))))
                        duck.turn(math.copysign(SEARCH_RATE, brain_turn) if gone < SEARCH_MAX_DEG else 0.0, 0.0)
                    else:
                        if seen and vyaw is not None and args.detector == 'population':
                            brain_turn += (1.0 - math.exp(-ms / 1000.0 / 0.5)) * (vyaw - brain_turn)   # the brain's recent turn
                            search_from = None
                        duck.turn(vyaw if seen else 0.0, vx if seen else 0.0)
            elif duck:
                duck.turn(vyaw, BODY_VX)

        if view:
            view.frame(t, f['distance_mm'], f['status'], enc.inv_tau, counts, ms, note, floor, camera, vyaw)
        if trace:
            trace.add(counts, ms, f, camera, note, head, vyaw,
                      None if learner is None else (learner.petting(now), learner.learned('ball'), learner.learned('bar')))
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
                tg = getattr(camera.enc, 'target', None)
                seen = (f'target {tg[0]:+5.1f} deg' if tg else 'no target') if args.steer == 'follow' \
                    else f'flow {camera.enc.mean_u:+5.1f} deg/s'
                if args.steer == 'follow' and args.detector == 'population':
                    seen += f', range {"-" if last_range is None else f"{last_range:.2f} m"}, {"walking" if walking else "standing"}, vyaw {vyaw if vyaw is not None else 0:+.2f}'
                if learner is not None:
                    seen += ''.join(f', {k} {"naive" if learner.learned(k) is None else f"{learner.learned(k):+.1f} Hz"}'
                                    for k in sorted(learner.seen)) + (' PETTING' if learner.petting(now) else '')
                line += (f', camera {camera.fps:.0f} fps, {seen}, '
                         f'{steer_dn} R-L {diff:+5.1f} Hz -> {vyaw:+.2f} rad/s'
                         + (f', head yaw {head[2]:+.2f} rad (measured {body.state["joints"][body.head_yaw]:+.2f})'
                            if head is not None and body.head_yaw is not None else '') + rate)
            print(line, flush=True)


if __name__ == '__main__':
    main()
