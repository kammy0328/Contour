"""(1.27, LNGSHOT BEEP BEEP 11/27, the user: "코 뾰루지") a pimple on the nose is removed: the red spot the blemish stage filled there is no longer taken
back by the nose's give-back of the last pass (engine.NOSE_PIMPLE); everything else a stage does to the nose is still given back, and nothing changes
inside the nostrils."""
from collections import defaultdict

import cv2
import numpy as np

from baseline_util import face68


def _blob(shape, c, sig):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2.0 * sig * sig)).astype(np.float32)


def _scene():
    from contour.retouch import engine as RE
    H = W = 520
    lm = face68(260.0, 150.0, 300.0)
    L = lm[:, :2]
    wf = 300.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Y0 = (0.50 + 0.04 * np.cos((xx - 260.0) / 120.0)).astype(np.float32)             # a face's soft shading
    Cb0 = np.full((H, W), -0.05, np.float32)
    Cr0 = np.full((H, W), 0.06, np.float32)
    bridge = (L[28] + L[29]) / 2.0                                                    # on the nose's bridge
    ns = L[31:36]
    up = (L[27] - L[33]) / np.linalg.norm(L[27] - L[33])
    nostril = ns.mean(0) + up * 0.08 * np.linalg.norm(ns[-1] - ns[0])                 # inside the nostrils' area
    pim, nos = _blob((H, W), bridge, 4.0), _blob((H, W), nostril, 4.0)
    Cr0 = Cr0 + 0.05 * (pim + nos)
    Y0 = Y0 - 0.02 * (pim + nos)
    # after the stages: both red spots filled (the skin's own level and colour), and another stage lifted the whole nose a little
    na = RE._nose_area(L.astype(np.float32), (H, W), wf)
    assert na is not None and na[int(bridge[1]), int(bridge[0])] > 0.9 and na[int(nostril[1]), int(nostril[0])] > 0.9
    Y = Y0 + 0.02 * (pim + nos) + 0.03 * na
    Cr = Cr0 - 0.05 * (pim + nos)
    Cb = Cb0.copy()
    v = defaultdict(float, blemish=0.4, skin=0.7)
    job = RE.Job(p=0, pid=0, v=v, keep=0.45, scope="face", grain_size=1.0, face=lm.astype(np.float32), face_w=wf, scale=1200.0,
                 masks={"skin": np.ones((H, W), np.float32)}, wf_src=wf)
    rf = RE.RetouchFrame(t=0, jobs=[job], proxy_wh=(W, H), src_wh=(W, H))
    spots = {id(job): np.clip(pim + nos, 0, 1)}
    at = lambda P, c: float(P[int(round(c[1])), int(round(c[0]))])
    return dict(Y=Y, Cb=Cb, Cr=Cr, Y0=Y0, Cb0=Cb0, Cr0=Cr0, rf=rf, spots=spots, bridge=bridge, nostril=nostril, na=na, at=at, job=job)


def _run(sc, spots):
    from contour.retouch import engine as RE
    M = lambda m, shape: m if m.shape == shape else cv2.resize(m.astype(np.float32), (shape[1], shape[0]))
    return RE._final_pass(sc["Y"].copy(), sc["Cb"].copy(), sc["Cr"].copy(), sc["Y0"], sc["Cb0"], sc["Cr0"], sc["rf"], 1.0, M, spots)


def test_a_red_spot_filled_on_the_nose_stays_filled():
    from contour.retouch import engine as RE
    assert RE.NOSE_PIMPLE == 1.0
    sc = _scene()
    at, b, n = sc["at"], sc["bridge"], sc["nostril"]
    Y, Cb, Cr = _run(sc, sc["spots"])
    assert abs(at(Cr, b) - 0.06) < 0.012, f"the pimple's red is gone: Cr {at(Cr, b):.3f} (the skin 0.060, the pimple {at(sc['Cr0'], b):.3f})"
    assert at(Y, b) > at(sc["Y0"], b) + 0.012, "and its dark is lifted"
    # inside the nostrils nothing a stage did is kept
    assert abs(at(Cr, n) - at(sc["Cr0"], n)) < 0.012, f"the nostril's own colour is back: {at(Cr, n):.3f} / {at(sc['Cr0'], n):.3f}"
    # what another stage did to the nose away from the pimple is given back as before
    far = (sc["na"] > 0.95) & (sc["spots"][id(sc["job"])] < 0.01)
    far &= cv2.erode(far.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    assert far.sum() > 200
    assert float(np.abs(Y - sc["Y0"])[far].max()) < 0.004, "the nose keeps its light and shade"


def test_without_the_setting_or_the_spots_the_nose_is_given_back_whole():
    from contour.retouch import engine as RE
    sc = _scene()
    at, b = sc["at"], sc["bridge"]
    Y, Cb, Cr = _run(sc, None)
    assert abs(at(Cr, b) - at(sc["Cr0"], b)) < 0.006, "as before 1.27: the nose's colour as it came in"
    was = RE.NOSE_PIMPLE
    try:
        RE.NOSE_PIMPLE = 0.0
        Y2, Cb2, Cr2 = _run(sc, sc["spots"])
    finally:
        RE.NOSE_PIMPLE = was
    assert np.array_equal(Cr2, Cr) and np.array_equal(Y2, Y)


def test_the_stage_tells_the_last_pass_where_it_filled():
    import inspect
    from contour.retouch import engine as RE
    src = inspect.getsource(RE._process_ycc)
    assert "red_filled_[id(job)]" in src and "_final_pass(Y, Cb, Cr, Y_in, Cb_in, Cr_in, rf, y_per_src, M, red_filled_)" in src
