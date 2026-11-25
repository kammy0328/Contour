"""The armpit fold retouch learned from hand-retouched pairs (contour.learn.armpitfit) and its use in the engine (ARMPIT_LEARNED)."""
import cv2
import numpy as np

from contour.learn import armpitfit as AF


def _plate(seed=0, n=360):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    base = 0.45 + 0.05 * np.sin(xx / 60.0)
    folds = 0.03 * np.sin((xx + 0.6 * yy) / 4.0) * np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 70.0 ** 2))
    grain = rng.normal(0, 0.004, (n, n)).astype(np.float32)
    Y = base + folds + grain
    rgb = np.stack([Y * 1.05, Y, Y * 0.85], -1).astype(np.float32)
    return rgb, grain


def _hand(rgb, wf, sigma_rel=0.033, opacity=0.8):
    n = rgb.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    m = (np.hypot(xx - n / 2, yy - n / 2) < 110).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 6)[..., None]
    bl = cv2.GaussianBlur(rgb, (0, 0), sigma_rel * wf)
    return rgb + opacity * m * (bl - rgb)


def test_fit_blur_finds_the_hands_width_and_opacity():
    wf = 300.0
    pairs = []
    for s in range(3):
        b, _ = _plate(s)
        pairs.append((b, _hand(b, wf), wf))
    M = AF.fit_blur(pairs)
    assert M["kind"] == "blur"
    assert abs(M["sigma_rel"] - 0.033) < 0.008
    assert 0.65 < M["opacity"] < 0.95


def test_apply_evens_the_folds_and_keeps_the_grain():
    wf = 300.0
    b, grain = _plate(5)
    Y = AF.ycc(b)[..., 0]
    n = Y.shape[0]
    m = np.zeros_like(Y); m[90:270, 90:270] = 1.0
    M = dict(kind="blur", sigma_rel=0.033, opacity=0.85, grain_rel=0.006)
    Yo = AF.apply(Y, m, M, wf)
    sl = np.s_[130:230, 130:230]
    band = lambda z: cv2.GaussianBlur(z, (0, 0), 2.0) - cv2.GaussianBlur(z, (0, 0), 20.0)
    assert band(Yo)[sl].std() < 0.5 * band(Y)[sl].std()          # the folds evened
    fine = lambda z: z - cv2.GaussianBlur(z, (0, 0), 1.0)
    r = np.corrcoef(fine(Yo)[sl].ravel(), fine(Y)[sl].ravel())[0, 1]
    assert r > 0.97                                               # the grain is the plate's own
    out = np.ones_like(m, bool); out[60:300, 60:300] = False
    assert np.abs(Yo - Y)[out].max() < 1e-6                       # nothing outside the mask


def test_engine_learned_model_switch():
    from contour.retouch import engine as RE
    assert RE.ARMPIT_LEARNED is None                              # off unless set
    RE_old = RE.ARMPIT_LEARNED
    try:
        RE.ARMPIT_LEARNED = dict(kind="blur", sigma_rel=0.033, opacity=0.85, grain_rel=0.006)
        assert RE._armpit_learned_model()["opacity"] == 0.85
        Y = np.full((64, 64), 0.4, np.float32); Y[:, ::4] += 0.02
        Cb = np.zeros((32, 32), np.float32); Cr = np.zeros((32, 32), np.float32)
        m = np.ones((64, 64), np.float32)
        Yo, Cbo, Cro = RE._armpit_learned_apply(Y, Cb, Cr, m, 1.0, 300.0, RE._armpit_learned_model())
        assert Yo.shape == Y.shape and Cbo.shape == Cb.shape
        RE.ARMPIT_LEARNED = dict(kind="other")
        assert RE._armpit_learned_model() is None
    finally:
        RE.ARMPIT_LEARNED = RE_old
