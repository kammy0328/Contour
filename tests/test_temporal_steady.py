"""(1.27.1, LNGSHOT BEEP BEEP 11/27: c056 f21 the chin's stage on for one frame; c087 f32 a shadow's edge taken for pebbles on one frame) the retouch
held steady through time (retouch.temporal): a frame's retouch is the median of its own and its neighbours' carried by the flow."""
import numpy as np

from contour.retouch import temporal as TM


def _scene(n=5, H=120, W=160, shift=2):
    """a textured picture moving right by `shift` px a frame; the retouch: a soft spot that moves with it"""
    rng = np.random.default_rng(1)
    base = rng.uniform(200, 800, (H, W + n * shift + 20)).astype(np.float32)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    src, out = [], []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for t in range(n):
        y = base[:, t * shift:t * shift + W]
        y = np.ascontiguousarray(base[:, 20 - t * shift:20 - t * shift + W])
        spot = 30.0 * np.exp(-((xx - (60 + t * shift)) ** 2 + (yy - 60) ** 2) / (2 * 8.0 ** 2))
        s_ = [np.round(y).astype(np.uint16), np.full((H, W), 512, np.uint16), np.full((H, W), 512, np.uint16)]
        o_ = [np.round(y + spot).astype(np.uint16), s_[1].copy(), s_[2].copy()]
        src.append(s_)
        out.append(o_)
    return src, out


SUBS = [(1, 1), (1, 1), (1, 1)]


def test_a_retouch_all_three_frames_share_stays():
    src, out = _scene()
    got = TM.steady(src[1:4], out[1:4], SUBS, 1023.0)
    d_eng = out[2][0].astype(np.int32) - src[2][0].astype(np.int32)
    d_got = got[0].astype(np.int32) - src[2][0].astype(np.int32)
    assert np.abs(d_got - d_eng).max() <= 3, "what the neighbours do too is kept"


def test_a_one_frame_change_goes():
    src, out = _scene()
    out = [[p.copy() for p in o] for o in out]
    out[2][0][20:30, 100:140] += 40                       # a streak on the middle frame only
    got = TM.steady(src[1:4], out[1:4], SUBS, 1023.0)
    d = got[0].astype(np.int32) - src[2][0].astype(np.int32)
    assert np.abs(d[20:30, 104:136]).max() <= 4, "a change of one frame only is taken back"
    assert d[50:70, 55:70].max() > 20, "the retouch the three frames share is there"


def test_off_and_untouched_frames():
    src, out = _scene()
    keep = TM.STEADY
    try:
        TM.STEADY = False
        got = TM.steady(src[1:4], out[1:4], SUBS, 1023.0)
        assert all(np.array_equal(a, b) for a, b in zip(got, out[2]))
    finally:
        TM.STEADY = keep
    same = TM.steady(src[1:4], src[1:4], SUBS, 1023.0)
    assert all(np.array_equal(a, b) for a, b in zip(same, src[2])), "a frame no one retouched is written as it came"


def test_the_steadier_keeps_every_frame_in_order():
    src, out = _scene(6)
    st = TM.Steadier(SUBS, 1023.0)
    got = []
    for t in range(6):
        got += st.push(t, src[t], out[t])
    got += st.flush()
    assert [t for t, _ in got] == list(range(6))
    assert all(np.array_equal(a, b) for a, b in zip(got[0][1], out[0])) and all(np.array_equal(a, b) for a, b in zip(got[-1][1], out[-1]))
