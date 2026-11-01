"""Cell types -> FlyWire 783 root ids, from the FlyWire annotations (Schlegel et al. 2024).

`data/flywire_783_annotations.tsv` is Supplemental_file1_neuron_annotations.tsv from
github.com/flyconnectome/flywire_annotations. Only meaningful with `FlyBrain(783)`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ANNOTATIONS = Path(__file__).resolve().parent / 'data' / 'flywire_783_annotations.tsv'

# The neurons the duck reads and writes, by role. Descending neurons (DNs) are the brain's output.
GIANT_FIBER = 'DNp01'          # escape takeoff -> Startle
LOOMING = ('LPLC2', 'LC4')     # looming detectors; the GF's two largest excitatory inputs
STEER = ('DNa01', 'DNa02')     # turning, per side
FORWARD = 'DNp09'              # "P9" (Bidaye et al. 2020): forward walking
BACKWARD = 'MDN'               # moonwalker: walk backwards


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    return pd.read_csv(ANNOTATIONS, sep='\t', dtype={'root_id': 'int64'}, low_memory=False).set_index('root_id')


def ids(cell_type: str, side: str | None = None) -> list[int]:
    """Root ids of a cell type, optionally one side ('left' / 'right')."""
    t = table()
    m = t.cell_type == cell_type
    if side:
        m &= t.side == side
    return [int(i) for i in t.index[m]]


def describe(root_id: int) -> str:
    r = table().loc[root_id]
    return f'{r.cell_type} ({r.side}, {r.top_nt})'
