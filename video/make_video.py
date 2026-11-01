"""Render a recorded run as a video: the duck in MuJoCo beside the fly brain that drives it.

    .venv/bin/python -I video/make_video.py demo/        # demo/{startle,arena,follow,learn}_{sim,trace}.npz -> demo/duckophila.mp4

The left half re-renders the simulator from `record_sim.py`'s samples — every joint, the trunk,
the ball and the drum — from a camera that follows the duck. The right half is the brain from
`bridge.py --trace`: every one of FlyWire's neurons at its own position, seen from the front, lit in
its class's colour when it spikes and fading over a few frames; below it the duck's camera, its
depth frame, and the neurons that matter for the act. A giant-fiber spike — a startle — rings the
giant fibers in red; in the learning act, petting is marked and each target's learned value (from the
mushroom body's output neurons, `learning.py`) is shown as the brain had it. The learning act plays
faster than real time, and says so. Nothing is drawn that the recording does not contain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import mujoco
import numpy as np

import neurons as N

HERE = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
FPS = 30
JOINT_NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
CLASS_BGR = {'optic': (200, 110, 70), 'central': (225, 225, 225), 'sensory': (120, 200, 90),
             'visual_projection': (250, 180, 80), 'descending': (60, 90, 255), 'ascending': (60, 200, 240),
             'motor': (200, 60, 255), 'visual_centrifugal': (255, 140, 120)}
INK, DIM, RED = (235, 235, 235), (140, 140, 140), (60, 60, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, xy, scale=0.8, color=INK, thick=1, box=True):
    """Text on a dark, half-transparent backing, so it reads over the sim and the brain alike."""
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = xy
    if box:
        x0, y0, x1, y1 = max(0, x - 8), max(0, y - th - 8), min(img.shape[1], x + tw + 8), min(img.shape[0], y + base + 6)
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


class Sim:
    def __init__(self, scene: Path, overhead: bool = False):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height=H, width=W // 2)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = np.array([self.m.jnt_qposadr[jid(n)] for n in JOINT_NAMES])
        free = int(np.flatnonzero(self.m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        self.free = self.m.jnt_qposadr[free]
        mid = lambda n: self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)]
        self.prop, self.drum = mid('duckophila_prop'), mid('duckophila_drum')
        self.target, self.bar = mid('duckophila_target'), mid('duckophila_bar')
        # the drum, see-through in the video only: from outside it is a wall, and the duck is inside
        drum_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'duckophila_drum')
        self.m.geom_rgba[self.m.geom_bodyid == drum_body, 3] = 0.45
        self.cam = mujoco.MjvCamera()
        if overhead:   # looking down into the arena
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.5, -52.0, 150.0
        else:
            self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.1, -20.0, 150.0
        self.look = None

    def frame(self, s, i):
        d = self.d
        d.qpos[self.qadr] = s['joints'][i]
        d.qpos[self.free:self.free + 3] = s['trunk'][i]
        d.qpos[self.free + 3:self.free + 7] = s['quat'][i]
        d.mocap_pos[self.prop] = s['prop'][i]
        d.mocap_pos[self.drum], d.mocap_quat[self.drum] = s['drum_pos'][i], s['drum_quat'][i]
        if 'target' in s:
            d.mocap_pos[self.target], d.mocap_pos[self.bar] = s['target'][i], s['bar'][i]
        mujoco.mj_forward(self.m, d)
        target = np.array(s['trunk'][i]) + [0, 0, 0.05]
        self.look = target if self.look is None else 0.92 * self.look + 0.08 * target   # a steady follow
        self.cam.lookat[:] = self.look
        self.r.update_scene(d, self.cam)
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)


class Brain:
    """The brain from the front: x across, y down (FlyWire voxels are 4 x 4 x 40 nm)."""

    def __init__(self, ids: np.ndarray, box: tuple[int, int, int, int]):
        t = N.table().reindex(ids)
        xy = t[['pos_x', 'pos_y']].to_numpy(float) * 0.004
        ok = ~np.isnan(xy).any(1)
        x0, y0, bw, bh = box
        lo, hi = np.nanpercentile(xy[ok], 0.5, 0), np.nanpercentile(xy[ok], 99.5, 0)
        k = min(bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1]))
        off = np.array([x0 + (bw - k * (hi[0] - lo[0])) / 2, y0 + (bh - k * (hi[1] - lo[1])) / 2])
        self.px = np.where(ok[:, None], (xy - lo) * k + off, -10).astype(int)
        self.col = np.array([CLASS_BGR.get(c, (180, 180, 180)) for c in t.super_class.fillna('')], np.float32)
        # the silhouette: every neuron, faint
        dens = np.zeros((H, W), np.float32)
        good = ok & (self.px[:, 0] >= 0) & (self.px[:, 0] < W) & (self.px[:, 1] >= 0) & (self.px[:, 1] < H)
        np.add.at(dens, (self.px[good, 1], self.px[good, 0]), 1.0)
        dens = cv2.GaussianBlur(dens, (0, 0), 1.2)
        self.bg = (np.clip(dens / np.percentile(dens[dens > 0], 98), 0, 1) * 70).astype(np.uint8)
        self.glow = np.zeros((H, W, 3), np.float32)
        self.good = good

    def draw(self, canvas, spikes):
        self.glow *= 0.80
        sp = spikes[self.good[spikes]]
        for i in sp:
            cv2.circle(self.glow, tuple(self.px[i]), 3, tuple(float(c) for c in self.col[i]), -1, cv2.LINE_AA)
        out = np.maximum(canvas.astype(np.float32), self.glow)
        canvas[:] = np.clip(out, 0, 255).astype(np.uint8)


ROWS = {
    'reflex': [('DNp01_L', 'giant fiber L', RED), ('DNp01_R', 'giant fiber R', (90, 90, 255)),
               ('DNp15_L', 'DNp15 L (head)', (80, 220, 255)), ('DNp15_R', 'DNp15 R (head)', (255, 220, 80))],
    'follow': [('LC10a_L', 'LC10a L (target)', (120, 200, 90)), ('LC10a_R', 'LC10a R (target)', (200, 230, 120)),
               ('DNa02_L', 'DNa02 L (turn left)', (80, 220, 255)), ('DNa02_R', 'DNa02 R (turn right)', (255, 220, 80))],
}
SCALE = {'DNp15': 150.0, 'DNa02': 60.0, 'LC10a': 30.0}


def trace_panel(canvas, tr, j, x0, y0, rows='reflex'):
    """The duck's camera, its depth frame, and the key descending neurons over the last 6 s."""
    cam = tr['camera'][j]
    if cam.size > 1:
        k = min(2.0, (H - y0 - 10) / cam.shape[0])                 # portrait: as tall as the panel allows
        c = cv2.resize(cam, (int(cam.shape[1] * k), int(cam.shape[0] * k)), interpolation=cv2.INTER_LINEAR)
        canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        text(canvas, "duck's camera", (x0, y0 - 8), 0.5, DIM, box=False)
        cx = x0 + c.shape[1] + 20
    else:
        cx = x0
    dep = tr['depth'][j].reshape(8, 8).astype(np.float32)
    v = np.where(dep > 0, 255 - np.clip(dep / 1500 * 255, 0, 255), 0).astype(np.uint8)
    dimg = cv2.applyColorMap(cv2.resize(v, (160, 160), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
    canvas[y0:y0 + 160, cx:cx + 160] = dimg
    text(canvas, 'depth sensor', (cx, y0 - 8), 0.5, DIM, box=False)
    # traces
    tx, tw, th = cx + 190, W - (cx + 190) - 30, 150
    names = list(tr['key_names'])
    t_now = tr['t'][j]
    lo = np.searchsorted(tr['t'], t_now - 6.0)
    ts = tr['t'][lo:j + 1]
    cv2.rectangle(canvas, (tx, y0), (tx + tw, y0 + th), (60, 60, 60), 1)
    for k, (key, label, col) in enumerate(ROWS[rows]):
        if key not in names or len(ts) < 2:
            continue
        if key.startswith('DNp01'):   # it fires a spike or two: ticks, not a rate
            for tt, v in zip(ts, tr['key'][lo:j + 1, names.index(key)]):
                if v > 0:
                    x = int(tx + (tt - (t_now - 6.0)) / 6.0 * tw)
                    cv2.line(canvas, (x, y0 + 4), (x, y0 + th - 4), col, 3, cv2.LINE_AA)
            text(canvas, label, (tx + 6 + k * (tw // 4), y0 + th + 22), 0.45, col, box=False)
            continue
        ysig = tr['key'][lo:j + 1, names.index(key)]
        ysig = np.convolve(ysig, np.ones(3) / 3, mode='same')     # a brain step is ~66 ms: smooth a little
        pts = np.stack([tx + (ts - (t_now - 6.0)) / 6.0 * tw, y0 + th - np.clip(ysig / SCALE.get(key[:-2], 150.0), 0, 1) * (th - 4)], 1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, col, 2, cv2.LINE_AA)
        text(canvas, label, (tx + 6 + k * (tw // 4), y0 + th + 22), 0.45, col, box=False)
    text(canvas, 'descending neurons, last 6 s', (tx, y0 - 8), 0.5, DIM, box=False)


def learn_panel(canvas, tr, j, T, phases):
    """The learning act: what is happening, whether the duck is being petted, what it has learned."""
    x0, y0 = W // 2 + 40, 770
    phase = [p for t, p in phases if t <= T]
    if phase:
        text(canvas, phase[-1], (x0, y0), 0.8, INK, 1)
    if 'petting' in tr and tr['petting'][j]:
        text(canvas, 'PETTED  ->  reward dopamine neurons (PAM)', (x0, y0 - 40), 0.85, (120, 200, 255), 2)
    for k, (key, label) in enumerate((('value_ball', 'ball'), ('value_bar', 'bar'))):
        if key not in tr:
            continue
        v = tr[key][j]
        s = f'{label}: naive' if np.isnan(v) else f'{label}: {v:+.1f} Hz' + ('  -> follows it' if v > 1.0 else '')
        text(canvas, s, (W - 470, y0 - 40 + 40 * k), 0.75, (120, 230, 120) if not np.isnan(v) and v > 1.0 else DIM)
    text(canvas, 'learned value (mushroom-body output)', (W - 470, y0 - 82), 0.5, DIM, box=False)


def render(sim_npz: Path, trace_npz: Path, caption: str, out: Path, scene: Path, overhead: bool = False,
           rows: str = 'reflex', speed: float = 1.0, phases: list | None = None):
    s, tr = np.load(sim_npz), np.load(trace_npz, allow_pickle=False)
    sim = Sim(scene, overhead)
    from flysim import DATA
    import pandas as pd
    ids = pd.read_csv(DATA[783][0], index_col=0).index.to_numpy()
    brain = Brain(ids, (W // 2 + 30, 90, W // 2 - 60, 640))
    brain.idx = {n: [int(np.flatnonzero(ids == f)[0]) for f in N.ids(n)] for n in ('DNp01', 'DNp15')}
    offs = np.concatenate([[0], np.cumsum(tr['spk_n'])])
    t0, t1 = max(s['t'][0], tr['t'][0]), min(s['t'][-1], tr['t'][-1])
    n = int((t1 - t0) * FPS / speed)
    tmp = out.with_suffix('.raw.mp4')
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    flash_until, last_j = -1.0, -1
    for f in range(n):
        T = t0 + f * speed / FPS
        i = min(int(np.searchsorted(s['t'], T)), len(s['t']) - 1)
        j = max(0, int(np.searchsorted(tr['t'], T)) - 1)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:, :W // 2] = sim.frame(s, i)
        canvas[:, W // 2:] = 12
        canvas[:, W // 2:, :] = np.maximum(canvas[:, W // 2:, :], brain.bg[:, W // 2:, None])
        spikes = np.concatenate([tr['spk'][offs[k]:offs[k + 1]] for k in range(last_j + 1, j + 1)]) if j > last_j else np.zeros(0, int)
        for k in range(last_j + 1, j + 1):
            if str(tr['note'][k]).startswith('STARTLE'):
                flash_until = T + 1.2
        last_j = j
        brain.draw(canvas, spikes.astype(int))
        if T < flash_until:
            for g in brain.idx['DNp01']:
                cv2.circle(canvas, tuple(brain.px[g]), 26, RED, 3, cv2.LINE_AA)
            text(canvas, 'giant fiber fires  ->  startle', (W // 2 + 40, 760), 0.9, RED, 2)
        if phases is not None:
            learn_panel(canvas, tr, j, T, phases)
        trace_panel(canvas, tr, j, W // 2 + 40, 840, rows)
        # titles
        text(canvas, 'microduck in MuJoCo  (the real robot daemons)', (30, 50), 0.9)
        text(canvas, "a fruit fly's brain: 138,639 neurons (FlyWire)", (W // 2 + 30, 50), 0.9)
        text(canvas, f'{T - t0:4.1f} s' + (f'  ({speed:g}x)' if speed != 1 else ''), (W // 2 - (170 if speed != 1 else 110), 50), 0.7, DIM)
        head, _, tail = caption.partition(':')
        text(canvas, head + (':' if tail else ''), (30, H - 76), 0.85)
        if tail:
            text(canvas, tail.strip(), (30, H - 34), 0.75)
        cv2.line(canvas, (W // 2, 0), (W // 2, H), (50, 50, 50), 2)
        vw.write(canvas)
        if f % 150 == 0:
            print(f'  {out.name}: {f}/{n}', flush=True)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '22', '-preset', 'medium', str(out)], check=True)
    tmp.unlink()


def title_card(out: Path, lines: list[tuple[str, float]], seconds: float = 3.5):
    img = np.full((H, W, 3), 12, np.uint8)
    y = H // 2 - 40 * len(lines)
    for s, scale in lines:
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, 2)
        text(img, s, ((W - tw) // 2, y), scale, INK, 2)
        y += int(90 * scale)
    tmp = out.with_suffix('.raw.mp4')
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for _ in range(int(seconds * FPS)):
        vw.write(img)
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tmp), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True)
    tmp.unlink()


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else 'demo')
    scene = HERE / 'sim' / '.scenes' / 'scene_backlash_prop.xml'
    parts = []
    title_card(d / 'v0.mp4', [('Duckophila', 2.0), ("a fruit fly's brain, driving a microduck", 1.0),
                              ('FlyWire connectome  -  Shiu et al. brain model  -  microduck by Pollen Robotics / Hugging Face', 0.6)])
    parts.append(d / 'v0.mp4')
    acts = [('startle', "Act 1 - a ball thrown at the face: the fly's looming detectors fire its escape neuron", False, 'reflex', 1.0),
            ('arena', 'Act 2 - a striped drum turns: the head follows it (DNp15), the walking body follows the head', True, 'reflex', 1.0),
            ('follow', "Act 3 - a small target on the floor: the fly's object detectors (LC10a) steer it through DNa02", True, 'follow', 1.0),
            ('learn', 'Act 4 - petted while it sees the ball, it follows the ball, not the bar: learning, mostly ours (see paper)', True, 'follow', 3.0)]
    import json
    for k, (name, cap, overhead, rows, speed) in enumerate(acts, 1):
        if (d / f'{name}_sim.npz').exists() and (d / f'{name}_trace.npz').exists():
            out = d / f'v{k}.mp4'
            ph = d / f'{name}_phases.json'
            phases = json.loads(ph.read_text()) if ph.exists() else None
            print(f'rendering {name}', flush=True)
            render(d / f'{name}_sim.npz', d / f'{name}_trace.npz', cap, out, scene, overhead, rows, speed, phases)
            parts.append(out)
    lst = d / 'parts.txt'
    lst.write_text(''.join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy',
                    str(d / 'duckophila.mp4')], check=True)
    for p in parts:
        p.unlink()
    lst.unlink()
    print(f'-> {d / "duckophila.mp4"}')


if __name__ == '__main__':
    main()
