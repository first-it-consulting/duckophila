"""Check and print the visual-column orientation `retinotopy.py` derives from the connectome."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retinotopy as R

for side in ('left', 'right'):
    o = R.lplc2_offsets(side)
    print(f'{side} eye, LPLC2 T4/T5 input offsets from their centre (hex units):')
    for d, name in zip('abcd', ('front-to-back', 'back-to-front', 'up', 'down')):
        print(f'   {d} ({name:13s}) dX {o.loc[d, "dX"]:+.2f}  dY {o.loc[d, "dY"]:+.2f}')
    for hs in ('HSN', 'HSE', 'HSS'):
        x, y = R.input_centroid(hs, side)
        print(f'   {hs} input centroid  X {x:+.1f}  Y {y:+.1f}')
    print('   ->', R.orientation()[side])
d = R.directions()
t = d[d.type.str.match(r'T4[a-d]$')]
for side in ('left', 'right'):
    s = t[t.hemisphere == side]
    print(f'{side}: azimuth {s.az_deg.min():.0f}..{s.az_deg.max():.0f} deg, elevation {s.el_deg.min():.0f}..{s.el_deg.max():.0f} deg; '
          f'T4/T5 within the camera (|az| < 22.5 deg each side, |el| < 36 deg): '
          f'{((s.az_deg < 22.5) & (s.el_deg.abs() < 36)).sum() * 2} of {len(s) * 2}')
