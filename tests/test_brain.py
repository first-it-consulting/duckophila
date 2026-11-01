"""The brain: exact against Brian2, and the controls are what they say."""
import numpy as np
import pytest

from conftest import ROOT, needs_annotations, needs_connectome


@needs_connectome
def test_replay_is_spike_for_spike_identical():
    """Brian2's own sugar-GRN spikes replayed into flysim: every downstream spike must match.
    Needs results/sugarR.parquet from `experiments/sugar_mn9.py` (CI makes it with 1 trial)."""
    ref = ROOT / 'results' / 'sugarR.parquet'
    if not ref.exists():
        pytest.skip('run experiments/sugar_mn9.py first')
    import subprocess, sys
    out = subprocess.run([sys.executable, '-I', str(ROOT / 'experiments' / 'replay.py')], capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith('spikes brian2'))
    n = [int(x) for x in line.replace('identical', '').split() if x.isdigit()]
    assert n[0] == n[1] == n[2] > 1000, line


@needs_connectome
def test_null_brains_keep_neurons_weights_and_degrees():
    from flysim import FlyBrain
    b = FlyBrain(783, seed=0)
    s = b.null('shuffled', seed=1)
    assert np.array_equal(s.indptr, b.indptr)                          # out-degree per neuron
    assert np.array_equal(np.sort(s.indices), np.sort(b.indices))      # in-degree per neuron
    assert np.array_equal(s.weights, b.weights)                        # every weight, on its source
    assert not np.array_equal(s.indices, b.indices)
    f = b.null('flipped')
    assert np.array_equal(f.weights, -b.weights)


@needs_connectome
@needs_annotations
def test_giant_fibre_answers_looming_detectors_and_not_rest():
    from flysim import FlyBrain
    import neurons as N
    b = FlyBrain(783, seed=0)
    gf = [b.index[f] for f in N.ids(N.GIANT_FIBER)]
    rest = b.copy(); c0 = rest.run(500)
    assert c0[gf].sum() == 0
    drive = b.copy(); drive.stimulate({f: 80 for t in N.LOOMING for f in N.ids(t, 'right')}); c1 = drive.run(500)
    assert c1[gf].sum() > 10
