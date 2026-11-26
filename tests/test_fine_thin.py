"""(1.27, LNGSHOT BEEP BEEP 11/27: "피부 보정 Bananas 때처럼", "납품본의 피부를 기준으로") 질감 유지 below its default thins the skin's finest texture
(engine.FINE_THIN, fitted on the delivered hand retouch of LNGSHOT BANANAS); at the default and above every picture is as before."""
from collections import defaultdict

import cv2
import numpy as np


def _plane(h=360, w=480, seed=3):
    """A soft shading with texture of every size on it (0..1)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.45 + 0.15 * np.sin(xx / 90.0) * np.cos(yy / 70.0)
    tex = sum(a * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), s) * s for s, a in ((0.6, 0.02), (1.5, 0.02), (4.0, 0.02), (12.0, 0.02)))
    return (base + tex).astype(np.float32)


def _band(y, s0, s1):
    a = y if s0 <= 0 else cv2.GaussianBlur(y, (0, 0), s0)
    return a - cv2.GaussianBlur(y, (0, 0), s1)


def test_amount_follows_the_slider():
    from contour.retouch import engine as RE
    assert RE.fine_thin_amount(0.45, 0.70) == 0.0, "the slider's default: nothing"
    assert RE.fine_thin_amount(1.00, 0.70) == 0.0
    assert abs(RE.fine_thin_amount(0.15, 0.70) - RE.FINE_THIN) < 1e-6, "질감 유지 15, 피부 표준: the fitted share"
    assert abs(RE.fine_thin_amount(0.30, 0.70) - 0.5 * RE.FINE_THIN) < 1e-6
    assert abs(RE.fine_thin_amount(0.00, 0.70) - 1.5 * RE.FINE_THIN) < 1e-6
    assert abs(RE.fine_thin_amount(0.15, 0.35) - 0.5 * RE.FINE_THIN) < 1e-6, "with the 피부 value"
    assert RE.fine_thin_amount(0.15, 0.0) == 0.0
    # a small face is thinned less (the hand's own way, FINE_THIN_WF): nothing on a face of a full shot, all of it from 600 px of a 2160-line picture
    assert RE.fine_thin_amount(0.15, 0.70, 100.0) == 0.0 and RE.fine_thin_amount(0.15, 0.70, 150.0) == 0.0
    assert abs(RE.fine_thin_amount(0.15, 0.70, 375.0) - 0.5 * RE.FINE_THIN) < 1e-6
    assert abs(RE.fine_thin_amount(0.15, 0.70, 600.0) - RE.FINE_THIN) < 1e-6 and abs(RE.fine_thin_amount(0.15, 0.70, 1600.0) - RE.FINE_THIN) < 1e-6
    keep = RE.FINE_THIN
    try:
        RE.FINE_THIN = 0.0
        assert RE.fine_thin_amount(0.0, 1.0) == 0.0
    finally:
        RE.FINE_THIN = keep


def test_the_finest_bands_are_thinned_and_the_wide_ones_stay():
    from contour.retouch import engine as RE
    y = _plane()
    h, w = y.shape
    m = np.zeros((h, w), np.float32)
    m[60:300, 80:400] = 1.0
    sc = h / 2160.0
    assert RE._thin_fine(y, y, m, 0.0) is y, "nothing asked: the plane itself"
    out = RE._thin_fine(y, y, m, 0.16, rows=2160)          # (the widths as on a 2160-line picture)
    inner = np.zeros((h, w), bool)
    inner[110:250, 130:350] = True
    fine = lambda z: _band(z, 0, 1.2)
    kept = float(np.sqrt((fine(out)[inner] ** 2).mean()) / np.sqrt((fine(y)[inner] ** 2).mean()))
    assert 0.82 <= kept <= 0.86, f"the band below 1.2 px: {kept:.3f} of it left (0.84 asked)"
    mid = lambda z: _band(z, 4.0, 5.6)
    kept_mid = float((mid(out)[inner] * mid(y)[inner]).sum() / (mid(y)[inner] ** 2).sum())
    assert 0.90 <= kept_mid <= 0.97, f"the band 4 to 5.6 px: {kept_mid:.3f} (0.92 to 0.95 fitted)"
    wide = lambda z: cv2.GaussianBlur(z, (0, 0), 30.0)
    assert float(np.abs(wide(out) - wide(y))[inner].max()) < 2e-4, "light and shade stay"
    outside = np.ones((h, w), bool)
    outside[60:300, 80:400] = False
    assert float(np.abs(out - y)[outside].max()) == 0.0, "nothing outside the mask"
    assert float(np.abs(out - y).max()) < 0.03


def test_the_widths_follow_the_pictures_height():
    """A piece of a 4K plane and the same picture at half the size: the same share of the same structure."""
    from contour.retouch import engine as RE
    y = _plane(360, 480)
    m = np.ones_like(y)
    a = RE._thin_fine(y, y, m, 0.16, rows=2160)
    b = RE._thin_fine(y, y, m, 0.16, rows=4320)
    # twice the height: the blurs are twice as wide, so more of this picture's (unscaled) texture counts as fine
    da, db = float(np.abs(a - y).mean()), float(np.abs(b - y).mean())
    assert db > da * 1.05, (da, db)


def _job(h, w, keep, m):
    from contour.retouch import engine as RE
    from contour.retouch import skin as SK
    v = defaultdict(float, skin=0.70)
    return RE.Job(p=0, pid=0, v=v, keep=keep, scope="face", grain_size=1.0, face=None, face_w=220.0, scale=900.0,
                  skin=SK.SkinJob(0, 0.70, keep, "face", None, 220.0, 900.0), masks={"skin": m}, wf_src=220.0)


def test_the_skin_stage_takes_it_from_the_slider():
    """Through the engine: 질감 유지 at its default gives what 1.26 gave (the thinning switched off changes nothing there); at 15 the finest band inside the
    skin's mask is thinner by the fitted share, and the picture outside the mask is the same."""
    from contour.retouch import engine as RE
    h, w = 360, 480
    y = _plane(h, w)
    rgb = np.clip(np.dstack([y * 1.10, y * 0.92, y * 0.80]) * 255.0, 0, 255).astype(np.uint8)
    m = np.zeros((h, w), np.float32)
    m[60:300, 80:400] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 6.0)
    run = lambda keep: RE.apply_rgb(rgb, RE.RetouchFrame(t=0, jobs=[_job(h, w, keep, m)], proxy_wh=(w, h), src_wh=(w, h)), 1.0).astype(np.float32)
    d45, d15 = run(0.45), run(0.15)
    was = RE.FINE_THIN
    try:
        RE.FINE_THIN = 0.0
        off45, off15 = run(0.45), run(0.15)
    finally:
        RE.FINE_THIN = was
    assert np.array_equal(d45, off45), "the default: as before"
    assert np.array_equal(off15, off45), "without the thinning 질감 유지 does nothing in this stage"
    # what the slider adds is the thinning's own term: -amount x mask x (luma - the mix of its blurs), on the picture's luma (the 8-bit rounding of the two
    # pictures lies on it: the slope of what came out on what is expected, over the mask)
    K = np.float32([0.2126, 0.7152, 0.0722])
    y0 = (rgb.astype(np.float32) @ K) / 255.0
    sc = h / 2160.0
    mix = sum(wt * cv2.GaussianBlur(y0, (0, 0), max(0.5, sg * sc)) for sg, wt in RE.FINE_THIN_MIX)
    want = -RE.fine_thin_amount(0.15, 0.70, 220.0 * 2160.0 / h) * m * (y0 - mix) * 255.0      # (a face 220 px wide on this picture: a large one)
    assert abs(RE.fine_thin_amount(0.15, 0.70, 220.0 * 2160.0 / h) - RE.FINE_THIN) < 1e-6
    got = (d15 - d45) @ K
    sel = m > 0.5
    slope = float((got[sel] * want[sel]).sum() / (want[sel] ** 2).sum())
    assert 0.85 <= slope <= 1.15, f"the change from 질감 유지 45 to 15 against the thinning's own term: slope {slope:.3f}"
    assert float(np.abs(got - want)[sel].mean()) < 0.6, "and nothing else (8-bit rounding of the two pictures)"
    far = np.zeros((h, w), bool)
    far[:30] = True
    far[-30:] = True
    assert np.array_equal(d15[far], d45[far])


def test_a_small_face_is_thinned_less_through_the_engine():
    """The same picture with a face of a full shot's size (wf_src): 질감 유지 15 does nothing there."""
    from contour.retouch import engine as RE
    from contour.retouch import skin as SK
    h, w = 360, 480
    y = _plane(h, w)
    rgb = np.clip(np.dstack([y * 1.10, y * 0.92, y * 0.80]) * 255.0, 0, 255).astype(np.uint8)
    m = cv2.GaussianBlur(np.pad(np.ones((240, 320), np.float32), ((60, 60), (80, 80))), (0, 0), 6.0)

    def run(keep, wf):
        v = defaultdict(float, skin=0.70)
        job = RE.Job(p=0, pid=0, v=v, keep=keep, scope="face", grain_size=1.0, face=None, face_w=wf, scale=900.0,
                     skin=SK.SkinJob(0, 0.70, keep, "face", None, wf, 900.0), masks={"skin": m}, wf_src=wf)
        return RE.apply_rgb(rgb, RE.RetouchFrame(t=0, jobs=[job], proxy_wh=(w, h), src_wh=(w, h)), 1.0).astype(np.float32)
    small = 20.0                       # 20 px on a 360-line picture = 120 px of a 2160-line one: under FINE_THIN_WF[0]
    assert np.array_equal(run(0.15, small), run(0.45, small)), "a face of a full shot: nothing is thinned"
    mid = 62.5                         # 375 px of a 2160-line picture: half
    K = np.float32([0.2126, 0.7152, 0.0722])
    d_mid = float(np.abs((run(0.0, mid) - run(0.45, mid)) @ K).mean())
    d_big = float(np.abs((run(0.0, 220.0) - run(0.45, 220.0)) @ K).mean())
    assert 0 < d_mid < d_big
