"""What needs no connectome: the encoders' rules and the geometry."""
import numpy as np

from geometry import quat_matrix, rotate, rotation_rate


def test_rotation_rate_recovers_a_known_turn():
    # 0.3 rad/s about z for 0.1 s
    a = 0.03
    r1 = quat_matrix([np.cos(a / 2), 0, 0, np.sin(a / 2)])
    w = rotation_rate(np.eye(3), r1, 0.1)
    assert np.allclose(w, [0, 0, 0.3], atol=1e-9)


def test_quat_matrix_matches_rotate():
    q = np.array([0.9, 0.1, -0.3, 0.2]); q /= np.linalg.norm(q)
    v = np.array([0.3, -1.0, 2.0])
    assert np.allclose(quat_matrix(q) @ v, rotate(q, v))
