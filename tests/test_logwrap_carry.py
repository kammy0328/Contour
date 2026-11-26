"""The change of the view carried back into log code values (retouch.logwrap.display_delta_to_source): 1.26.3 evaluates the display
transform fewer times (13 a pixel at most instead of 16, and a pixel whose change is met is not stepped again: LNGSHOT, ARRI LogC4 under
ACES 2.0, the transform was 40 % of a render). What comes out is what 1.26.1's way gave, to a tenth of a 10-bit code value."""
import cv2
import numpy as np
import pytest

from contour.core.colour import ViewSettings, manager
from contour.retouch import logwrap as LW


def _old(cm, vs, src, disp0, disp1):
    """1.26.1's display_delta_to_source, as it was."""
    d = disp1 - disp0
    ch = np.max(np.abs(d), axis=-1) > LW._THRESH
    out = np.zeros_like(src)
    if not ch.any():
        return out
    ys, xs = np.nonzero(ch)
    p = src[ys, xs].astype(np.float64)
    target = disp1[ys, xs].astype(np.float64)
    cur = p.copy()
    best = p.copy()
    f0 = LW._eval(cm, vs, p.astype(np.float32)).astype(np.float64)
    best_err = np.linalg.norm(target - f0, axis=1)
    for _ in range(LW._STEPS):
        f0 = LW._eval(cm, vs, cur.astype(np.float32)).astype(np.float64)
        J = np.empty((cur.shape[0], 3, 3))
        for k in range(3):
            q = cur.copy()
            q[:, k] += LW._EPS
            J[:, :, k] = (LW._eval(cm, vs, q.astype(np.float32)).astype(np.float64) - f0) / LW._EPS
        r = target - f0
        JT = np.transpose(J, (0, 2, 1))
        A = JT @ J
        lam = 1e-3 * np.trace(A, axis1=1, axis2=2)[:, None, None] + 1e-9
        A = A + lam * np.eye(3)[None]
        step = np.linalg.solve(A, (JT @ r[:, :, None]))[:, :, 0]
        step = np.clip(step, -LW._STEP_MAX, LW._STEP_MAX)
        nxt = np.clip(cur + step, p - LW._TOTAL_MAX, p + LW._TOTAL_MAX)
        err = np.linalg.norm(target - LW._eval(cm, vs, nxt.astype(np.float32)).astype(np.float64), axis=1)
        better = err < best_err
        best[better] = nxt[better]
        best_err = np.where(better, err, best_err)
        cur = nxt
    out[ys, xs] = (best - p).astype(np.float32)
    return out


@pytest.fixture
def logc4():
    cm = manager()
    if not cm.available:
        pytest.skip("no OpenColorIO config")
    names = [c.getName() for c in cm.config.getColorSpaces()]
    if "ARRI LogC4" not in names:
        pytest.skip("no ARRI LogC4 in the config")
    return cm, ViewSettings(input_cs="ARRI LogC4")


def test_carry_back_as_before(logc4):
    cm, vs = logc4
    rng = np.random.default_rng(3)
    src = np.clip(rng.normal([0.45, 0.40, 0.37], 0.03, (160, 200, 3)), 0, 1).astype(np.float32)
    d0 = cm.to_display(src, vs)
    lift = np.zeros(d0.shape[:2], np.float32)
    lift[20:140, 30:170] = 0.01
    lift[50:56, 60:66] = 0.08                            # a filled spot: a bigger change
    lift = cv2.GaussianBlur(lift, (0, 0), 4)
    d1 = d0 + lift[..., None] * np.array([1.0, 0.95, 0.9], np.float32)
    calls = []
    real = cm.to_display

    def count(x, v):
        calls.append(x.size // 3)
        return real(x, v)
    cm.to_display = count
    try:
        a = _old(cm, vs, src, d0, d1)
        n_old = sum(calls)
        calls.clear()
        b = LW.display_delta_to_source(cm, vs, src, d0, d1)
        n_new = sum(calls)
    finally:
        cm.to_display = real
    assert np.abs(a - b).max() * 1023 < 0.1                        # under a tenth of a 10-bit code value
    assert n_new < 0.7 * n_old                                    # the point of it
    res_old = np.abs(cm.to_display(src + a, vs) - d1).max(-1)
    res_new = np.abs(cm.to_display(src + b, vs) - d1).max(-1)
    assert float(np.median(res_new[lift > 1e-3])) < 1e-4          # the change asked for is there
    assert (res_new <= res_old + 1e-4).all()                      # as well as 1.26.1 met it, everywhere


def test_nothing_asked_nothing_done(logc4):
    cm, vs = logc4
    src = np.full((8, 8, 3), 0.4, np.float32)
    d0 = cm.to_display(src, vs)
    assert not LW.display_delta_to_source(cm, vs, src, d0, d0.copy()).any()
