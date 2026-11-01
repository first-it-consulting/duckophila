"""The hand-written translations behave as their docstrings say."""
import numpy as np

from conftest import needs_annotations


@needs_annotations
def test_an_edge_sliding_in_is_arrival_not_approach():
    """A zone whose range drops 50 cm in one frame saw something arrive; it must not drive."""
    from looming import LoomingEncoder
    enc = LoomingEncoder()
    st = np.full(64, 5)
    for k in range(6):
        enc.update(np.full(64, 1500), st, k / 15)
    d = np.full(64, 1500); d[20] = 1000                                 # an edge slides into zone 20
    rates = enc.update(d, st, 6 / 15)
    assert max(rates.values()) == 0.0


@needs_annotations
def test_fast_approach_drives_and_slow_does_not():
    from looming import LoomingEncoder
    st = np.full(64, 5)
    for speed, expect in ((1.5, True), (0.2, False)):
        enc = LoomingEncoder()
        peak = 0.0
        for k in range(12):
            d = np.full(64, 1500); d[27] = int((1.0 - speed * k / 15) * 1000)
            peak = max(peak, max(enc.update(d, st, k / 15).values()))
        assert (peak > 0) == expect, (speed, peak)
