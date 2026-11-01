import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

REFERENCE = ROOT / 'Drosophila_brain_model' / 'Connectivity_783.parquet'
ANNOTATIONS = ROOT / 'data' / 'flywire_783_annotations.tsv'
COLUMNS = ROOT / 'data' / 'column_assignment.csv.gz'

needs_connectome = pytest.mark.skipif(not REFERENCE.exists(), reason='the reference model is not cloned (README: Set it up)')
needs_annotations = pytest.mark.skipif(not ANNOTATIONS.exists(), reason='data/flywire_783_annotations.tsv is not downloaded')
needs_columns = pytest.mark.skipif(not COLUMNS.exists(), reason='data/column_assignment.csv.gz needs a Codex sign-in (README)')
