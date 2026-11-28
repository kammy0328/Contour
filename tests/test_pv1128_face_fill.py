"""The third check of the 11/28 renders: c038 f45..48, 률's profile in plain sight with no landmarks from the tracker (visibility 0) while both face models
had the face; the retouch went off for four frames between frames that had it (FACE_RAW_FILL)."""
import numpy as np

from contour.retouch import engine as RE


def _lm(cx, cy, w):
    ang = np.linspace(0, np.pi, 17)
    L = np.zeros((68, 3), np.float32)
    L[:17, 0] = cx - 0.5 * w * np.cos(ang)
    L[:17, 1] = cy + 0.6 * w * np.sin(ang)
    rng = np.random.default_rng(int(cx))
    L[17:, :2] = np.c_[cx + (rng.random(51) - 0.5) * 0.7 * w, cy + (rng.random(51) - 0.6) * 0.8 * w]
    L[:, 2] = 1.0
    return L


class _St:
    def __init__(self, n=12):
        self.has_face = True
        self.present = np.ones((1, n), bool)
        self.face = np.full((1, n, 68, 3), np.nan, np.float32)
        self.face_raw = np.full((1, n, 68, 3), np.nan, np.float32)
        self.face106_raw = np.full((1, n, 68, 3), np.nan, np.float32)
        self.face_vis = np.zeros((1, n), np.float32)


def _scene(gap=(4, 8)):
    st = _St()
    for t in range(12):
        L = _lm(500 + 4 * t, 300, 220)
        if not (gap[0] <= t < gap[1]):
            st.face[0, t] = L
            st.face_vis[0, t] = 0.6
        a = L.copy()
        a[:, 2] = 5.6
        b = L.copy()
        b[:, :2] += 3.0
        b[:, 2] = 0.72
        st.face_raw[0, t] = a
        st.face106_raw[0, t] = b
    return st


def test_a_face_dropped_for_four_frames_is_taken_from_the_two_face_models():
    st = _scene()
    for t in range(4, 8):
        got = RE._face_raw_fill(st, 0, t)
        assert got is not None, t
        face, ends = got
        assert ends == (3, 8)
        assert np.isfinite(face[:, :2]).all()
        assert abs(float(face[:17, 0].mean()) - (500 + 4 * t + 1.5)) < 2.0


def test_no_fill_when_the_models_disagree_or_are_unsure_or_the_gap_is_long_or_the_face_is_elsewhere():
    st = _scene()
    st.face106_raw[0, 5, :, :2] += 60.0                      # the face model somewhere else
    assert RE._face_raw_fill(st, 0, 5) is None
    st = _scene()
    st.face_raw[0, 5, :, 2] = 3.0                            # the whole-body model unsure
    assert RE._face_raw_fill(st, 0, 5) is None
    st = _scene(gap=(2, 9))                                  # seven frames out: not a dropout
    assert RE._face_raw_fill(st, 0, 5) is None
    st = _scene()
    st.face_raw[0, 5, :, :2] += 300.0                        # both models on another face, away from the line between the frames round it
    st.face106_raw[0, 5, :, :2] += 300.0
    assert RE._face_raw_fill(st, 0, 5) is None
    st = _scene()
    st.present[0, 8:] = False                                # no frame after it with the face
    assert RE._face_raw_fill(st, 0, 6) is None
    old = RE.FACE_RAW_FILL
    try:
        RE.FACE_RAW_FILL = None
        assert RE._face_raw_fill(_scene(), 0, 5) is None
    finally:
        RE.FACE_RAW_FILL = old
