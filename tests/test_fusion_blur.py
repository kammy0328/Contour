"""Motion blur and the fused joints (PC check 2026-11-19, Ballin + 3 MVs, 422 cuts): on blurred frames the fused
measurement lost 2.8 % of the joints the solved track sees, against 1.5 % for each pose model alone. The models'
scores already drop on a blurred frame, and the fusion lowered them again by the sharpness (down to 0.3 x), under
the 0.05 the solver keeps. blur_rule="agree" keeps the trust of a joint two or more models agree on closely."""
import numpy as np

from contour.track import solver as SV
from contour.track import skeleton as S
from contour.track.store import TrackStore


def _store(spread_px=0.0, score=0.3, blur=0.35, models=3):
    st = TrackStore(5, [1])
    st.present[:] = True
    base = np.array([[200 + 10 * (j % 3), 100 + 30 * j] for j in range(S.K)], np.float32)
    rng = np.random.default_rng(0)
    for m in range(models):
        a = np.zeros((1, 5, S.K, 3), np.float32)
        a[..., :2] = base + rng.normal(0, spread_px, (1, 5, S.K, 2)) if spread_px else base
        a[..., 2] = score
        st.raw[f"pose_m{m}"] = a
    st.blur[:] = 1.0
    st.blur[0, 2] = blur
    return st


def kept(st, rule):
    norm = {k: 1.0 for k in st.raw}
    SV.fuse_measurements(st, blur_rule=rule, norm=norm)
    return (st.meas[0, 2, :, 2] > 0.05).mean(), (st.meas[0, 0, :, 2] > 0.05).mean()


def test_agreeing_models_keep_their_joints_on_a_blurred_frame():
    # scores a blurred frame gives (under the usual 0.2): before, no measurement at all
    blur_frame, sharp_frame = kept(_store(score=0.15), "scaled")
    assert blur_frame == 0.0
    blur_frame, sharp_frame = kept(_store(score=0.15), "agree_low")
    assert blur_frame == 1.0
    # a lower score on a sharp frame is still not taken
    st = _store(score=0.15, blur=1.0)
    SV.fuse_measurements(st, blur_rule="agree_low", norm={k: 1.0 for k in st.raw})
    assert (st.meas[0, 2, :, 2] <= 0.05).all()
    # scores still over 0.2 but lowered by the sharpness twice (bone check, agreement): kept with "agree"
    a = _store(score=0.22, blur=0.3)
    SV.fuse_measurements(a, blur_rule="scaled", norm={k: 1.0 for k in a.raw})
    b = _store(score=0.22, blur=0.3)
    SV.fuse_measurements(b, blur_rule="agree", norm={k: 1.0 for k in b.raw})
    assert (b.meas[0, 2, :, 2] > a.meas[0, 2, :, 2] * 2).all()


def test_one_low_point_alone_is_not_a_measurement():
    st = _store(score=0.15, models=1)
    SV.fuse_measurements(st, blur_rule="agree_low", norm={k: 1.0 for k in st.raw})
    assert (st.meas[0, 2, :, 2] == 0).all()


def test_disagreeing_or_single_models_are_still_lowered():
    a = _store(spread_px=60.0, score=0.9)
    SV.fuse_measurements(a, blur_rule="agree", norm={k: 1.0 for k in a.raw})
    b = _store(spread_px=60.0, score=0.9)
    SV.fuse_measurements(b, blur_rule="none", norm={k: 1.0 for k in b.raw})
    assert (a.meas[0, 2, :, 2] < b.meas[0, 2, :, 2] - 1e-6).any()
    one = _store(models=1, score=0.9)
    SV.fuse_measurements(one, blur_rule="agree", norm={k: 1.0 for k in one.raw})
    two = _store(models=1, score=0.9)
    SV.fuse_measurements(two, blur_rule="none", norm={k: 1.0 for k in two.raw})
    assert (one.meas[0, 2, :, 2] < two.meas[0, 2, :, 2]).all()
    assert (one.flags[0, 2] & S.FLAG_BLUR).all()
