"""1.26.2 (LNGSHOT BEEP BEEP QA, 11/26, cut 068 t31): the tracker lost a 500 px face's landmarks on one frame (a hand passing in front)
and the whole retouch went off for that frame. A face missing (or its visibility dipping) for at most FACE_GAP_BRIDGE frames between
frames where it is known is taken from them."""
import inspect
from types import SimpleNamespace

import numpy as np

from contour.retouch import engine as RE


def _store(T=10, gap=(5,), vis=0.9):
    face = np.zeros((1, T, 68, 3), np.float32)
    for t in range(T):
        face[0, t, :, 0] = 100 + 10 * t
        face[0, t, :, 1] = 200
        face[0, t, :, 2] = 1
    for t in gap:
        face[0, t] = np.nan
    return SimpleNamespace(has_face=True, present=np.ones((1, T), bool), face=face, face_vis=np.full((1, T), vis, np.float32))


def test_a_one_frame_gap_is_bridged():
    st = _store()
    out = RE._face_bridge(st, 0, 5, 2)
    assert out is not None
    lm, w = out
    assert np.allclose(lm[:, 0], 150) and w == 1.0                # half way between frames 4 and 6


def test_a_two_frame_gap_is_bridged_a_longer_one_not():
    st = _store(gap=(5, 6))
    lm, w = RE._face_bridge(st, 0, 5, 2)
    assert np.allclose(lm[:, 0], 100 + 10 * (4 + 1 / 3 * 3))       # frames 4 and 7: a third of the way
    st3 = _store(gap=(4, 5, 6))
    assert RE._face_bridge(st3, 0, 5, 2) is None


def test_not_across_an_absence_or_the_cut_end():
    st = _store()
    st.present[0, 4] = False
    assert RE._face_bridge(st, 0, 5, 2) is None
    assert RE._face_bridge(_store(gap=(9,)), 0, 9, 2) is None      # the last frame: nothing after it


def test_the_weight_is_the_lower_of_the_two():
    st = _store()
    st.face_vis[0, 4] = 0.3
    _, w = RE._face_bridge(st, 0, 5, 2)
    assert abs(w - RE._vis_w(0.3)) < 1e-6


def test_prepare_uses_it():
    src = inspect.getsource(RE._prepare_raw)
    assert "_face_bridge(st, p, t, FACE_GAP_BRIDGE)" in src and "face, fw = bridge_[0], bridge_[1]" in src
