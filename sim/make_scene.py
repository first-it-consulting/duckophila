"""Write a copy of a microduck_rl scene with three extra things in it: the prop, the drum, the target.

The prop is a mocap sphere (r = 5 cm, orange, no collisions) parked under the floor, where no ToF
ray reaches it. The drum is the classic fly-arena stimulus: a ring of 48 alternating black and
white panels, 0.5 m in radius and 0.6 m tall — one black/white pair every 15 degrees, in the
spatial period flies' optomotor response likes — also a mocap body, also parked below the floor.
The target is what the duck follows (`follow.py`): a small sphere (r = 4 cm) that lights itself
and so is the brightest thing in the grey sim camera — a stand-in for a detector's output, parked
too. `sitecustomize.py` moves all three. Asset directories are made absolute so the copy can
live here, outside the simulator's tree — the point being that `microduck_rl` stays untouched.

    python make_scene.py <scene.xml in microduck_rl> <out.xml>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

PROP = 'duckophila_prop'
RADIUS = 0.05
PARKED = (0.0, 0.0, -5.0)

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
spec = mujoco.MjSpec.from_file(str(src))
for attr in ('meshdir', 'texturedir'):
    d = getattr(spec, attr)
    setattr(spec, attr, str((src.parent / d).resolve()) if d else str(src.parent))
body = spec.worldbody.add_body(name=PROP, mocap=True, pos=list(PARKED))
body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[RADIUS, 0, 0],
              rgba=[1.0, 0.45, 0.05, 1.0], contype=0, conaffinity=0)
DRUM, DRUM_R, DRUM_H, PANELS = 'duckophila_drum', 0.5, 0.6, 48
drum = spec.worldbody.add_body(name=DRUM, mocap=True, pos=[0.0, 0.0, -5.0])
half_w = DRUM_R * np.tan(np.pi / PANELS)
for k in range(PANELS):
    a = 2 * np.pi * (k + 0.5) / PANELS
    drum.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, half_w, DRUM_H / 2],
                  pos=[DRUM_R * np.cos(a), DRUM_R * np.sin(a), DRUM_H / 2],
                  quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                  rgba=[0.95, 0.95, 0.95, 1] if k % 2 else [0.05, 0.05, 0.05, 1],
                  contype=0, conaffinity=0)
TARGET, TARGET_R = 'duckophila_target', 0.04
spec.add_material(name='duckophila_glow', rgba=[1, 1, 1, 1], emission=1.0)
target = spec.worldbody.add_body(name=TARGET, mocap=True, pos=[0.0, 0.0, -5.0])
target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TARGET_R, 0, 0], material='duckophila_glow',
                contype=0, conaffinity=0)
spec.compile()  # fail here, not in the simulator
out.write_text(spec.to_xml())
print(out)
