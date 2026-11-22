"""A hair or a thin necklace chain lying over the neck is left alone by the neck line removal (2611 IWALY c050).

Two causes, one test each: skin._neck_lines_back handed any faint thin dark line below the chin back to the neck line
removal, a long straight one (a hair, a chain) too; and the neck's mask blurred the skin probability, which closed the
few pixels wide hole the strand has in it."""
import numpy as np
import cv2

from contour.retouch import skin as SK
from contour.retouch import engine as RE


def _lm(cx=320.0, top=150.0, w=160.0):
    """68 landmarks of a frontal face in a 640 picture: jaw 0..16 (width w), chin = 8, nose bridge 27 down to 30."""
    lm = np.zeros((68, 3), np.float64)
    a = np.linspace(np.pi, 0, 17)
    lm[:17, 0] = cx - np.cos(a) * w / 2
    lm[:17, 1] = top + np.sin(a) * w * 0.55
    lm[17:22, :2] = np.c_[np.linspace(cx - 60, cx - 20, 5), np.full(5, top - 40)]
    lm[22:27, :2] = np.c_[np.linspace(cx + 20, cx + 60, 5), np.full(5, top - 40)]
    lm[27:31, :2] = np.c_[np.full(4, cx), np.linspace(top - 25, top + 8, 4)]
    lm[31:36, :2] = np.c_[np.linspace(cx - 14, cx + 14, 5), np.full(5, top + 14)]
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    lm[48:60, :2] = np.c_[cx + 24 * np.cos(ang), top + 40 + 8 * np.sin(ang)]
    lm[60:68, :2] = np.c_[cx + 16 * np.cos(ang[::3].repeat(2)[:8]), top + 40 + 4 * np.sin(ang[::3].repeat(2)[:8])]
    lm[:, 2] = 1
    return lm


def _picture(lines):
    """A skin coloured picture with faint dark lines (each: two end points, how much darker, width) and the map of them."""
    h, w = 640, 640
    rng = np.random.RandomState(3)
    rgb = np.full((h, w, 3), (205.0, 150.0, 128.0), np.float32)
    rgb += rng.normal(0, 1.2, (h, w, 1)).astype(np.float32)
    out = np.zeros((h, w), np.float32)
    for (p0, p1, dark, wd) in lines:
        layer = np.zeros((h, w), np.float32)
        cv2.line(layer, p0, p1, 1.0, wd, cv2.LINE_AA)
        rgb *= (1.0 - dark * cv2.GaussianBlur(layer, (0, 0), 0.7))[..., None]
        mk = cv2.GaussianBlur(cv2.dilate(layer, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
        out = np.maximum(out, np.clip(mk * 1.5, 0, 1))
    return np.clip(rgb, 0, 255).astype(np.uint8), out


def _kept(res, out, p0, p1):
    m = np.zeros(out.shape, np.uint8)
    cv2.line(m, p0, p1, 1, 3)
    return float(res[m > 0].mean() / max(float(out[m > 0].mean()), 1e-6))


def test_a_long_straight_thin_line_below_the_chin_stays_a_strand_a_short_one_is_a_neck_line():
    lm = _lm()
    chain = ((250, 300), (318, 260))               # 0.5 face widths, thin, diagonal: a chain or a hair
    crease = ((285, 330), (310, 334))              # a short faint line across the neck: a neck line
    rgb, out = _picture([(chain[0], chain[1], 0.07, 2), (crease[0], crease[1], 0.07, 2)])
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    old = SK.NECK_STRAND_GUARD
    try:
        SK.NECK_STRAND_GUARD = False
        before = SK._neck_lines_back(out, rgb, lm, Wf)
        SK.NECK_STRAND_GUARD = True
        after = SK._neck_lines_back(out, rgb, lm, Wf)
    finally:
        SK.NECK_STRAND_GUARD = old
    assert _kept(before, out, *chain) < 0.1, "before: the chain was handed to the neck line removal"
    assert _kept(before, out, *crease) < 0.1
    assert _kept(after, out, *chain) > 0.9, "now: it stays a strand"
    assert _kept(after, out, *crease) < 0.1, "a neck line is still given back"


def test_guard_off_changes_nothing():
    lm = _lm()
    rgb, out = _picture([((250, 300), (318, 260), 0.07, 2), ((285, 330), (310, 334), 0.07, 2)])
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    old = SK.NECK_STRAND_GUARD
    try:
        SK.NECK_STRAND_GUARD = False
        a = SK._neck_lines_back(out.copy(), rgb, lm, Wf)
        b = SK._neck_lines_back(out.copy(), rgb, lm, Wf)
    finally:
        SK.NECK_STRAND_GUARD = old
    assert np.array_equal(a, b)


def test_neck_mask_keeps_a_strand_out_of_the_neck_even_when_the_blur_would_close_its_hole():
    h, w = 420, 360
    lm = np.zeros((68, 3), np.float32)
    for i in range(17):
        a = np.pi * i / 16
        lm[i, :2] = (180 - 100 * np.cos(a), 120 + 140 * np.sin(a))
    lm[17:27, :2] = [(100 + 18 * k, 90) for k in range(10)]
    lm[27:, 0] = 180
    lm[27:, 1] = np.linspace(110, 230, 41)
    protect = np.zeros((h, w), np.float32)
    cv2.line(protect, (150, 330), (230, 290), 1.0, 3, cv2.LINE_AA)
    protect = np.clip(cv2.GaussianBlur(protect, (0, 0), 1.0) * 1.5, 0, 1)
    prob = (1.0 - protect).astype(np.float32)
    parse = dict(neck=np.ones((h, w), np.float32), skin=np.zeros((h, w), np.float32), cover=np.ones((h, w), np.float32), trust=1.0)
    old = RE.NECK_PROTECT
    try:
        RE.NECK_PROTECT = 0.0
        m0 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse, protect=protect)
        RE.NECK_PROTECT = 1.0
        m1 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse, protect=protect)
        m2 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
    finally:
        RE.NECK_PROTECT = old
    assert float(m0[309:311, 189:191].mean()) > 0.3, "before: the blur closed the strand's hole"
    assert float(m1[309:311, 189:191].mean()) < 0.1, "now: the strand is out of the neck's mask"
    near = cv2.dilate((protect > 1e-3).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    assert float(m1[309:311, 189:191].mean()) < 0.1 * float(m0[309:311, 189:191].mean()) + 0.05
    assert float(np.abs(m1 - m0)[~near].max()) < 1e-4, "away from the strand nothing changes"
    assert float(m1[near].max()) > 0.5, "and the neck beside it is kept"
    assert float(np.abs(m2 - m0).max()) < 1e-6, "without a protect map (or with the weight 0) the mask is as it was"
