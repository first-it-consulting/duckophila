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


def test_target_bearing_sign_and_scale():
    """A white blob left of centre is a target on the left; 10 degrees is ~76 px at 45 deg across 360."""
    from follow import bearing
    g = np.full((640, 360), 90, np.uint8)
    assert bearing(g) is None
    g[400:430, 180 - 76 - 15:180 - 76 + 15] = 255                        # 10 deg left, below centre
    g[100, :] = 255                                                       # a thin grid line: not a target
    az, el, _, kind = bearing(g)
    assert abs(az - 10.0) < 0.5 and el < 0 and kind == "ball"


@needs_annotations
def test_follow_rates_mirror_between_eyes():
    """From 10 degrees out, a target drives its own side's LC10a harder (column map needed). Nearer
    the midline it need not: the left eye's LC10a look further front than the right's."""
    from conftest import COLUMNS
    if not COLUMNS.exists():
        import pytest; pytest.skip('column map not downloaded')
    from follow import FollowEncoder
    enc = FollowEncoder()
    side = enc.rf.side.to_numpy()
    for az in (10.0, 20.0):
        a, b = (np.array(list(enc.rates_at(s, -10.0).values())) for s in (az, -az))
        assert a[side == 'left'].sum() > a[side == 'right'].sum()
        assert b[side == 'right'].sum() > b[side == 'left'].sum()


def test_target_kind_by_shape():
    from follow import bearing
    g = np.full((640, 360), 90, np.uint8)
    g[380:460, 170:190] = 255                                              # 20 wide, 80 tall: the bar
    assert bearing(g)[3] == 'bar'
