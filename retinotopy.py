"""Where each columnar neuron looks: FlyWire's visual columns, turned into directions.

`data/column_assignment.csv.gz` is Codex's "Visual Neuron Columns" for FAFB 783 (Matsliah et al.
2024, FlyWire): 45k columnar neurons of 31 types, each on one of ~796 hexagonal columns per eye,
with hex coordinates p, q. It says which column, not which way that column points; that is
derived here, and every step is checked against the connectome rather than assumed.

**The lattice.** Nearest neighbours differ by (p+1, q+1) — straight up — or by one of p, q —
diagonally; x = (q - p)/2 steps are *second* neighbours. With neighbour spacing 1 that is
X = sqrt(3) x, Y = (p + q)/2: an eye ~29 columns wide and ~30 tall. (Reading x and y as a square
grid gives 17 x 52, an eye three times taller than wide.)

**Which way is front, which way is up — from the fly's own wiring** (`experiments/columns_orient.py`):
LPLC2 is a looming detector; each of its four dendrite arms takes the T4/T5 type whose preferred
direction points *away* from its centre. So its front-to-back inputs (T4a/T5a) sit on the back side
of it, back-to-front (b) on the front side, upward (c) above, downward (d) below. Averaged over all
LPLC2 of an eye, a sits at -X and b at +X, c at +Y and d at -Y — the same in both eyes. So +X is
front, +Y is up. Independently, HSN ("north") takes its input above HSS: +Y is up again. HSE, the
equatorial HS cell, takes it at the equator, which sets elevation 0.

**Angles.** 4.8 degrees per column (Drosophila's interommatidial angle, ~4.5-5). The eye's
frontmost column is put 10 degrees across the midline — flies see a little binocular strip. Both
are approximations; the shape of the eye is not modelled, the lattice is laid flat.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import os

import numpy as np
import pandas as pd

import neurons as N

COLUMNS = Path(__file__).resolve().parent / 'data' / 'column_assignment.csv.gz'
CONNECTIVITY = Path(__file__).resolve().parent / 'Drosophila_brain_model' / 'Connectivity_783.parquet'
DEG_PER_COLUMN = 4.8
FRONT_OVERLAP_DEG = 10.0


@lru_cache(maxsize=1)
def columns() -> pd.DataFrame:
    """Every columnar neuron: hemisphere, type, column_id, and the physical hex position X, Y."""
    c = pd.read_csv(COLUMNS).set_index('root_id')
    c['X'] = np.sqrt(3.0) * (c.q - c.p) / 2.0
    c['Y'] = (c.p + c.q) / 2.0
    return c


@lru_cache(maxsize=1)
def _t4_inputs() -> pd.DataFrame:
    c = columns()
    t45 = c.index[c.type.str.match(r'T[45][a-d]$')]
    con = pd.read_parquet(CONNECTIVITY, columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    e = con[con.Presynaptic_ID.isin(t45)].copy()
    e['dir'] = c.type.reindex(e.Presynaptic_ID).str[2].to_numpy()
    e['X'] = c.X.reindex(e.Presynaptic_ID).to_numpy()
    e['Y'] = c.Y.reindex(e.Presynaptic_ID).to_numpy()
    e['hemi'] = c.hemisphere.reindex(e.Presynaptic_ID).to_numpy()
    return e


def lplc2_offsets(side: str) -> pd.DataFrame:
    """Mean offset (dX, dY) of each direction's T4/T5 inputs from their LPLC2's own centre."""
    e = _t4_inputs()
    e = e[e.Postsynaptic_ID.isin(set(N.ids('LPLC2', side))) & (e.hemi == side)]
    w = e.Connectivity
    cen = e.assign(wx=e.X * w, wy=e.Y * w).groupby('Postsynaptic_ID')[['wx', 'wy', 'Connectivity']].sum()
    dx = e.X - (cen.wx / cen.Connectivity).reindex(e.Postsynaptic_ID).to_numpy()
    dy = e.Y - (cen.wy / cen.Connectivity).reindex(e.Postsynaptic_ID).to_numpy()
    g = pd.DataFrame({'dir': e.dir, 'dX': dx * w, 'dY': dy * w, 'w': w}).groupby('dir').sum()
    return g[['dX', 'dY']].div(g.w, axis=0)


def input_centroid(cell_type: str, side: str) -> tuple[float, float]:
    e = _t4_inputs()
    e = e[e.Postsynaptic_ID.isin(set(N.ids(cell_type, side))) & (e.hemi == side)]
    return float(np.average(e.X, weights=e.Connectivity)), float(np.average(e.Y, weights=e.Connectivity))


@lru_cache(maxsize=1)
def orientation() -> dict[str, dict[str, float]]:
    """Per eye: the sign of front along X and of up along Y, from LPLC2; the equator from HSE."""
    out = {}
    for side in ('left', 'right'):
        o = lplc2_offsets(side)
        front = np.sign(o.loc['b', 'dX'] - o.loc['a', 'dX'])
        up = np.sign(o.loc['c', 'dY'] - o.loc['d', 'dY'])
        hsn, hss = input_centroid('HSN', side)[1], input_centroid('HSS', side)[1]
        if np.sign(hsn - hss) != up:
            raise ValueError(f'{side} eye: LPLC2 and the HS cells disagree on which way is up')
        out[side] = {'front': float(front), 'up': float(up), 'equator_Y': input_centroid('HSE', side)[1]}
    if os.environ.get('DUCKOPHILA_EQUATOR') == 'shared':
        # One equator for both eyes. The column grid is the same in both (same range, same median Y);
        # HSE's input centroid, one cell per eye, puts the equators 2.2 columns (~10 deg) apart, and
        # the HSN/HSS midpoint 1.4 apart the other way — single-cell estimates, not two eyes that differ.
        # With separate equators the right eye's receptive fields sit ~10 deg lower than the left's,
        # so a target low ahead drives the left eye more and every LC10a readout leans left.
        eq = float(np.mean([out[s]['equator_Y'] for s in out]))
        for s in out:
            out[s]['equator_Y'] = eq
    return out


@lru_cache(maxsize=1)
def directions() -> pd.DataFrame:
    """Every columnar neuron's eye and viewing direction: `az_deg` toward the eye's own side
    (negative: across the midline), `el_deg` up."""
    c = columns().copy()
    o = orientation()
    az = np.empty(len(c))
    el = np.empty(len(c))
    for side in ('left', 'right'):
        m = (c.hemisphere == side).to_numpy()
        f = c.X.to_numpy()[m] * o[side]['front']               # larger = further front
        az[m] = -FRONT_OVERLAP_DEG + (f.max() - f) * DEG_PER_COLUMN
        el[m] = (c.Y.to_numpy()[m] - o[side]['equator_Y']) * o[side]['up'] * DEG_PER_COLUMN
    c['az_deg'], c['el_deg'] = az, el
    return c
