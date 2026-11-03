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


def _walking_blob(frames, x0, y0, w=40, h=50, step=3, bg=120, fg=20):
    """A dark blob walking sideways on a plain floor, `step` pixels a frame (0: standing)."""
    out = []
    for k in range(frames):
        g = np.full((640, 360), bg, np.uint8)
        x = x0 + k * step
        g[y0:y0 + h, x:x + w] = fg
        out.append(g)
    return out


def test_moving_target_ignores_still_and_finds_moving():
    from follow import MovingTarget
    still = MovingTarget()
    assert all(still(g, (0.0, 0.0)) is None for g in _walking_blob(20, 100, 450, step=0))
    mt = MovingTarget()
    hits = [mt(g, (0.0, 0.0)) for g in _walking_blob(30, 30, 450)]
    az = [h[0] for h in hits if h]
    assert az and all(a > 5 for a in az)                    # left of the centre: + azimuth


@needs_annotations
def test_population_drives_the_side_that_moves():
    """Motion on the left of the view drives the left eye's LC10a more than the right's, and
    the drive fades after the motion stops (column map needed)."""
    from conftest import COLUMNS
    if not COLUMNS.exists():
        import pytest; pytest.skip('column map not downloaded')
    from follow import FollowEncoder
    for x0, own in ((30, 'left'), (240, 'right')):
        enc = FollowEncoder(detector='population')
        for g in _walking_blob(20, x0, 450):
            enc.update(g, camera_turn_deg=0.0)        # the camera stands still, as the state history would say
        side = enc.rf.side.to_numpy(); r = enc.detect.rates()
        other = 'right' if own == 'left' else 'left'
        assert r[side == own].sum() > r[side == other].sum() > -1
        peak = r.sum()
        still = np.full((640, 360), 120, np.uint8)
        for _ in range(30):
            enc.update(still, camera_turn_deg=0.0)
        assert enc.detect.rates().sum() < 0.1 * peak         # 1 s later, mostly gone
