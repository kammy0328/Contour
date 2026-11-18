"""Landmark outlier gate (j2751, LNGSHOT Bananas): a frame whose landmarks jump away while the head stays is dropped,
a real fast move (body nose moves with it) is kept, a tangled shape is dropped."""
import numpy as np

from contour.track import face as F


def _track(T=12, move=None):
    rng = np.random.default_rng(0)
    base = rng.uniform(-50, 50, (68, 2))
    z = np.zeros((1, T, 68, 2))
    nose = np.zeros((1, T, 2))
    for t in range(T):
        off = np.array([100.0, 100.0]) + (move(t) if move else 0)
        z[0, t] = base + off
        nose[0, t] = off
    ok = np.ones((1, T), bool)
    size = np.full((1, T), 100.0)
    return z, ok, size, nose


def test_jump_without_head_motion_is_dropped():
    z, ok, size, nose = _track()
    z[0, 6] += np.array([60.0, 0.0])            # 0.6 face widths, nose did not move
    out = F.gate_outliers(z, ok, size, nose)
    assert not out[0, 6]
    assert out[0, [4, 5, 7, 8]].all()


def test_fast_real_move_is_kept():
    z, ok, size, nose = _track(move=lambda t: np.array([40.0 * t, 0.0]))   # 0.4 face widths per frame
    out = F.gate_outliers(z, ok, size, nose)
    assert out.all()


def test_tangled_shape_is_dropped():
    z, ok, size, nose = _track()
    c = z[0, 5].mean(0)
    z[0, 5] = c + np.random.default_rng(1).uniform(-50, 50, (68, 2))      # same place, landmarks scrambled
    out = F.gate_outliers(z, ok, size, nose)
    assert not out[0, 5]
    assert out[0, [3, 4, 6, 7]].all()


def test_small_noise_is_kept():
    z, ok, size, nose = _track()
    z += np.random.default_rng(2).normal(0, 1.5, z.shape)
    out = F.gate_outliers(z, ok, size, nose)
    assert out.all()
