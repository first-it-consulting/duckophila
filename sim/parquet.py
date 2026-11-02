"""A parquet floor texture for the twin: planks of random tone and grain, nothing that repeats.

    python parquet.py out.png [--size-m 4] [--px 2048] [--seed 0]

MuJoCo's ground is a checkerboard with bright edge lines, repeating every 20 cm. A periodic floor
fools anything that compares images by shifting them (one tile looks like the next), and its bright
lines shimmer into "moving things" when the duck rocks — neither is what a real room looks like. This
is a stand-in for a wooden floor: 7 x 50 cm planks in staggered rows, each its own tone with a
little grain along it, joined by slightly darker joints. One texture covers `--size-m` metres.
"""
import argparse

import numpy as np
from PIL import Image


def parquet(size_m=4.0, px=2048, seed=0, plank_w=0.07, plank_l=0.5):
    rng = np.random.default_rng(seed)
    m = px / size_m                                   # pixels per metre
    img = np.zeros((px, px, 3), np.float32)
    base = np.array([0.62, 0.46, 0.30])               # oak-ish
    y = np.arange(px)[:, None]; x = np.arange(px)[None, :]
    row = (y / (plank_w * m)).astype(int)
    off = rng.uniform(0, plank_l, size=row.max() + 2)
    along = x / m + off[row]
    plank = (along / plank_l).astype(int)
    pid = row * 10007 + plank
    tone = np.random.default_rng(seed + 1).uniform(0.75, 1.2, size=pid.max() + 1)[pid]
    grain = 0.02 * np.sin(2 * np.pi * (along * rng.uniform(25, 40) + 0.3 * np.sin(y / m * 60)))
    img[:] = base * (tone + grain)[..., None]
    joint = ((y % (plank_w * m)) < 1.5) | (((along % plank_l) * m) < 1.5)
    img[joint] *= 0.8
    img += rng.normal(0, 0.015, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--size-m', type=float, default=4.0)
    ap.add_argument('--px', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    Image.fromarray(parquet(a.size_m, a.px, a.seed)).save(a.out)
    print(a.out)
