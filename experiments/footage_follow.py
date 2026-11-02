"""Follow me on real footage, open loop: a phone video at duck height -> the follow encoder -> the brain.

    .venv/bin/python -I experiments/footage_follow.py VIDEO [--hfov 68] [--out DIR]
        -> DIR/<name>.json (per frame: what the detector picked, DNa02 L-R) and DIR/<name>_annotated.mp4

The phone stands in for the duck's camera standing still. Its view is wider (about 68 degrees across
for an iPhone's main camera in landscape video, `--hfov`), so the centre is cut to the duck's 45 degrees
and scaled to the sim camera's 360-pixel width; the height keeps what the phone has (~42 degrees, the
duck's camera sees more vertically). Then, per frame and exactly as `bridge.py --steer follow`:
`follow.py`'s detector -> LC10a rates -> 1/30 s of the brain -> DNa02 left minus right, smoothed with
the bridge's 0.4 s, and the turn it would command (+ = left).

The annotated video shows the duck's window, in red what the detector counts as target-bright, a cross
where it put the target, and a bar for the turn. Nothing here is published: footage stays local.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

import neurons as N
from flysim import FlyBrain
from follow import FollowEncoder

W = 360               # the sim camera's upright width
DUCK_HFOV = 45.0
FPS = 30.0
TAU_S, GAIN, MAX = 0.4, 0.05, 1.0     # bridge.py's follow smoothing, gain (rad/s per Hz), clip


def frames(path: Path, hfov: float):
    """Grey and colour duck-window frames, from ffmpeg (HEVC, rotation handled by ffmpeg)."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                            '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True).stdout
    w0, h0 = (int(x) for x in probe.strip().splitlines()[0].split(',')[:2])
    cw = int(round(w0 * math.tan(math.radians(DUCK_HFOV / 2)) / math.tan(math.radians(hfov / 2))))
    h = int(round(h0 * W / cw)) // 2 * 2
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'fps={FPS},crop={cw}:{h0},scale={W}:{h}',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        bgr = np.frombuffer(b, np.uint8).reshape(h, W, 3)
        yield bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', type=Path)
    ap.add_argument('--hfov', type=float, default=68.0, help="the phone's horizontal field of view, degrees")
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'footage')
    ap.add_argument('--brain', default='intact', choices=('intact', 'shuffled'))
    ap.add_argument('--detector', default='white', choices=('white', 'moving'), help="follow.py's sim detector, or the real-world one")
    ap.add_argument('--white', type=int, help="the detector's brightness threshold (default follow.py's, 250)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.video.stem + ('' if a.brain == 'intact' else f'_{a.brain}') + (f'_white{a.white}' if a.white else '') + ('_moving' if a.detector == 'moving' else '')

    brain = FlyBrain(783, seed=0)
    if a.brain == 'shuffled':
        brain = brain.null('shuffled', seed=1)
    brain.run(1)
    enc = FollowEncoder(detector=a.detector)
    if a.white:
        enc.p.white = a.white
    left = np.array([brain.index[f] for f in N.ids('DNa02', 'left')])
    right = np.array([brain.index[f] for f in N.ids('DNa02', 'right')])
    ms = 1000.0 / FPS
    k = 1.0 - math.exp(-ms / 1000.0 / TAU_S)

    rows, diff, writer, t0 = [], 0.0, None, time.time()
    for i, (bgr, grey) in enumerate(frames(a.video, a.hfov)):
        brain.stimulate(enc.update(grey))
        c = brain.run(ms)
        lr = (c[left].sum() - c[right].sum()) * 1000.0 / ms
        diff += k * (lr - diff)
        turn = float(np.clip(GAIN * diff, -MAX, MAX))        # + = left, as the bridge's vyaw
        tg = enc.target
        rows.append({'t': i / FPS, 'az': tg[0] if tg else None, 'el': tg[1] if tg else None,
                     'area': tg[2] if tg else 0, 'dna02_LminusR_hz': diff, 'turn_rad_s': turn,
                     'shift_px': list(getattr(enc.detect, 'shift', (0.0, 0.0)))})

        # annotated frame
        v = bgr.copy()
        if a.detector == 'white':
            v[grey >= enc.p.white] = (0, 0, 255)
        elif enc.detect.mask is not None:
            v[enc.detect.mask > 0] = (0.5 * v[enc.detect.mask > 0] + (0, 110, 255)).astype(np.uint8)
        h = v.shape[0]
        f = (W / 2) / math.tan(math.radians(DUCK_HFOV / 2))
        if tg:
            x = int(W / 2 - f * math.tan(math.radians(tg[0]))); y = int(h / 2 - f * math.tan(math.radians(tg[1])))
            cv2.drawMarker(v, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.rectangle(v, (0, h - 18), (W, h), (40, 40, 40), -1)
        bx = int(W / 2 - turn / MAX * W / 2)
        cv2.rectangle(v, (min(W // 2, bx), h - 14), (max(W // 2, bx), h - 4), (0, 200, 0), -1)
        cv2.putText(v, f'{i / FPS:5.1f}s  {"target" if tg else "no target"}  turn {"L" if turn > 0 else "R"} {abs(turn):.2f}',
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(a.out / f'{name}_annotated_raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, h))
        writer.write(v)
        if i % 300 == 0:
            print(f'{i / FPS:5.1f}s  {time.time() - t0:5.0f}s wall', flush=True)
    writer.release()
    raw = a.out / f'{name}_annotated_raw.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    str(a.out / f'{name}_annotated.mp4')], check=True)
    raw.unlink()

    seen = np.array([r['az'] is not None for r in rows])
    turns = np.array([r['turn_rad_s'] for r in rows])
    summary = {'video': a.video.name, 'brain': a.brain, 'hfov': a.hfov, 'frames': len(rows),
               'target_frames': float(seen.mean()), 'turning_frames': float((np.abs(turns) > 0.05).mean()),
               'turn_left_share': float((turns > 0.05).mean()), 'turn_right_share': float((turns < -0.05).mean())}
    (a.out / f'{name}.json').write_text(json.dumps({'summary': summary, 'rows': rows}))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
