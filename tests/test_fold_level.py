"""2611 cut 024: at 강하게 the soft nasolabial shading and the soft band across the neck stayed. _fold_level and
_neck_level even out that middle scale inside the band; fine texture and the broad shading stay, and a hair strand
in the band does not end up as a dark notch in lifted skin (2611 cut 011)."""
import numpy as np
import cv2

from contour.retouch import engine as RE

WF = 300.0


def _face(seed=0):
    rng = np.random.default_rng(seed)
    H = W = 240
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Y = 0.55 + 0.0004 * xx                                   # broad light from one side
    fold = 0.05 * np.exp(-((xx - 120) / 12.0) ** 2)          # soft shading band, a tenth of the face wide
    Y = Y - fold
    tex = 0.006 * rng.standard_normal((H, W)).astype(np.float32)
    band = np.exp(-((xx - 120) / 25.0) ** 2).astype(np.float32)
    band[:40] = 0
    band[-40:] = 0
    band = cv2.GaussianBlur(band, (0, 0), 4)
    return (Y + tex).astype(np.float32), tex, band, np.ones((H, W), np.float32)


def _depth(Y):
    s = cv2.GaussianBlur(Y, (0, 0), 3)[100:140]
    return float(np.mean(s[:, 80:90]) + np.mean(s[:, 150:160])) / 2 - float(np.mean(s[:, 115:125]))


def test_soft_fold_evened_texture_kept():
    Y, tex, band, skin = _face()
    out = RE._fold_level(Y, band, skin, None, WF, 1.0)
    assert _depth(Y) > 0.04
    assert _depth(out) < 0.4 * _depth(Y)
    fine = lambda a: a - cv2.GaussianBlur(a, (0, 0), 2.0)
    r = np.corrcoef(fine(out)[60:180, 100:140].ravel(), fine(Y)[60:180, 100:140].ravel())[0, 1]
    assert r > 0.9
    far = np.abs(out - Y)[:, :20]
    assert far.max() < 1e-4                                   # outside the band nothing changes


def test_off_changes_nothing():
    Y, _, band, skin = _face()
    assert np.array_equal(RE._fold_level(Y, band, skin, None, WF, 0.0), Y)


def test_strand_not_made_a_notch():
    Y, _, band, skin = _face(1)
    Y = Y.copy()
    Y[80:160, 118:121] -= 0.12                               # a dark strand along the fold
    edges = np.zeros_like(Y)
    edges[80:160, 117:122] = 1
    out = RE._fold_level(Y, band, skin, None, WF, 1.0, edges)
    # the skin right beside the strand is lifted hardly more than the strand: no notch
    beside = float(np.mean((out - Y)[100:140, 124:128]))
    on = float(np.mean((out - Y)[100:140, 118:121]))
    assert beside - on < 0.01


def test_neck_band_evened_and_jaw_held():
    Y, tex, _, _ = _face(2)
    yy = np.mgrid[0:240, 0:240][0].astype(np.float32)
    Y = (0.55 + tex - 0.05 * np.exp(-((yy - 120) / 12.0) ** 2)).astype(np.float32)   # a band across the neck
    m = np.zeros_like(Y)
    m[40:200, 40:200] = 1
    m = cv2.GaussianBlur(m, (0, 0), 4)
    out = RE._neck_level(Y, m, None, WF, 1.0)
    d0 = float(np.mean(Y[80:90, 80:160]) - np.mean(Y[115:125, 80:160]))
    d1 = float(np.mean(out[80:90, 80:160]) - np.mean(out[115:125, 80:160]))
    assert d1 < 0.4 * d0
    hold = np.zeros_like(Y)
    held = RE._neck_level(Y, m, hold, WF, 1.0)
    assert np.allclose(held, Y)
