"""The second check of the 11/28 (client's 1st PV feedback) renders: hands at the face.

  - limb_cover: the parser's sure-skin map of the face (11/28) had the name of the hand code's own flag, so once a hand had been drawn, a limb of
    somebody else over the face failed the cover of the face as a whole (TypeError, caught by the caller: no cover at all, the face's own hand retouched).
  - LIMB_WRIST: a forearm ends flat at the wrist; its round cap as wide as the forearm lay over the cheek past the wrist (c021 f16).
  - LIMB_R_ANATOMY: a forearm in front of the body has the body's width in the person matte; the capsule is held to the limb's own width.
  - HAND_CORE / HAND_PARSE_TRIM: a hand drawn from its 21 points is its core (palm and the fingers' middle lines) and a margin; the margin goes back
    to the face where the face parser is sure of the face's skin (c024's hands on the cheeks, c026's at the mouth, c023's heart)."""
import numpy as np
import cv2

from contour.retouch import engine as RE
from contour.warp.body import SEGMENTS, SEG_INDEX
from contour.track import skeleton as S


def _face_lm(cx=300.0, cy=260.0, w=200.0):
    lm = np.zeros((68, 3), np.float32)
    for i in range(17):
        a = np.pi * i / 16
        lm[i, :2] = (cx - 0.5 * w * np.cos(a), cy + 0.7 * w * np.sin(a) - 0.05 * w)
    lm[17:27, :2] = [(cx - 0.4 * w + 0.08 * w * k, cy - 0.3 * w) for k in range(10)]
    lm[27:36, 0] = cx
    lm[27:36, 1] = np.linspace(cy - 0.25 * w, cy + 0.1 * w, 9)
    lm[36:48, :2] = [(cx + (-0.25 if k < 6 else 0.25) * w + 0.02 * w * (k % 6), cy - 0.15 * w) for k in range(12)]
    ang = np.linspace(0, 2 * np.pi, 21)[:-1]
    lm[48:68, :2] = np.c_[cx + 0.18 * w * np.cos(ang), cy + 0.3 * w + 0.06 * w * np.sin(ang)]
    return lm


class _Store:
    def __init__(self, n):
        self.present = np.zeros((n, 1), bool)
        self.final = np.full((n, 1, S.K, 3), np.nan, np.float32)
        self.vis = np.zeros((n, 1, S.K), np.uint8)
        self.widths = np.full((n, 1, len(SEGMENTS), 2), np.nan, np.float32)
        self.scale = np.full((n, 1), 400.0, np.float32)
        self.hand_raw = np.full((n, 1, 42, 3), np.nan, np.float32)
        self.hand = np.full((n, 1, 42, 3), np.nan, np.float32)
        self.has_hands = False
        self.pids = np.arange(n)


def _put_arm(st, q, el, wr, w, side="L"):
    ie, iw = (S.L_EL, S.L_WR) if side == "L" else (S.R_EL, S.R_WR)
    st.present[q, 0] = True
    st.final[q, 0, ie] = (el[0], el[1], 0.9)
    st.final[q, 0, iw] = (wr[0], wr[1], 0.9)
    st.vis[q, 0, ie] = st.vis[q, 0, iw] = 1
    st.widths[q, 0, SEG_INDEX["forearm_l" if side == "L" else "forearm_r"]] = (w, w)


def _put_hand(st, q, wrist, up=(0.0, -1.0), size=120.0, i0=0, conf=5.0):
    """21 points of an open hand, fingers along `up`"""
    up = np.asarray(up, np.float64) / np.hypot(*up)
    side = np.array([-up[1], up[0]])
    w = np.asarray(wrist, np.float64)
    pts = np.zeros((21, 2))
    pts[0] = w
    for k, (s_, l_) in enumerate(((-0.45, 0.7), (-0.15, 1.0), (0.05, 1.05), (0.25, 1.0), (0.42, 0.85))):
        base = w + up * (0.45 * size if k else 0.15 * size) + side * s_ * 0.5 * size
        for j in range(4):
            pts[1 + 4 * k + j] = base + up * (l_ * 0.5 * size) * (j / 3.0) + (side * -0.3 * size * (j / 3.0) if k == 0 else 0)
    st.present[q, 0] = True
    st.hand_raw[q, 0, i0:i0 + 21, :2] = pts
    st.hand_raw[q, 0, i0:i0 + 21, 2] = conf
    return pts


def test_a_limb_of_somebody_else_after_a_hand_leaves_the_cover_whole():
    h, w = 600, 700
    lm = _face_lm()
    st = _Store(2)
    # this person (0): a hand at the cheek, its forearm coming from below
    _put_arm(st, 0, (330.0, 560.0), (330.0, 420.0), 30.0, "L")
    _put_hand(st, 0, (330.0, 420.0), size=110.0)
    # somebody else (1): a forearm across the lower face, drawn after person 0's hand
    _put_arm(st, 1, (520.0, 300.0), (360.0, 330.0), 25.0, "L")
    parse = dict(skin=np.zeros((h, w), np.float32), trust=1.0)
    cv2.fillPoly(parse["skin"], [np.round(lm[:17, :2]).astype(np.int32)], 1.0)
    cov = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)       # raised a TypeError on 11/28 (the caller then took no cover)
    assert cov is not None and float(cov.max()) > 0.9, "the face's own hand is covered"


def test_a_forearm_ends_flat_at_the_wrist():
    h, w = 600, 700
    lm = _face_lm()
    st = _Store(1)
    _put_arm(st, 0, (200.0, 560.0), (250.0, 380.0), 40.0, "L")          # its wrist at the cheek's lower edge, no hand points
    st.final[0, 0, S.L_WR, 2] = 0.2                                      # (the hand from the forearm's continuation left out: the wrist not sure enough)
    old = (RE.LIMB_WRIST, RE.LIMB_R_ANATOMY)
    try:
        RE.LIMB_WRIST, RE.LIMB_R_ANATOMY = None, None
        c0 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
        RE.LIMB_WRIST = 0.65
        c1 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
    finally:
        RE.LIMB_WRIST, RE.LIMB_R_ANATOMY = old
    assert c0 is not None and c1 is not None
    # past the wrist along the forearm (the cheek): before covered by the round cap, now not
    d = np.array([50.0, -180.0]) / np.hypot(50.0, -180.0)
    px, py = np.round(np.array([250.0, 380.0]) + d * 30.0).astype(int)
    assert float(c0[py, px]) > 0.5 and float(c1[py, px]) < 0.2
    # the forearm itself still covered
    assert float(c1[470, 225]) > 0.9


def test_a_forearm_is_held_to_its_own_width():
    h, w = 600, 700
    lm = _face_lm()
    st = _Store(1)
    _put_arm(st, 0, (260.0, 560.0), (300.0, 360.0), 140.0, "L")          # the matte gave it the body's width
    old = RE.LIMB_R_ANATOMY
    try:
        RE.LIMB_R_ANATOMY = None
        c0 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
        RE.LIMB_R_ANATOMY = old
        c1 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
    finally:
        RE.LIMB_R_ANATOMY = old
    fb = float(SEGMENTS[SEG_INDEX["forearm_l"]].fallback) * float(np.hypot(40.0, 200.0))
    assert float((c0 > 0.5).sum()) > 1.4 * float((c1 > 0.5).sum())
    # across the forearm's middle, the covered width is about its anatomical one
    row = c1[460]
    assert abs(float((row > 0.5).sum()) - 2 * 0.9 * fb) < 0.35 * 2 * 0.9 * fb


def test_a_hands_margin_goes_back_to_the_face_where_the_parser_is_sure():
    h, w = 600, 700
    lm = _face_lm()
    st = _Store(1)
    pts = _put_hand(st, 0, (300.0, 400.0), up=(0.0, -1.0), size=170.0)
    _put_arm(st, 0, (300.0, 590.0), (300.0, 400.0), 30.0, "L")
    # the parser: the face's skin everywhere in the outline but on the fingers (a finger's width round their middle lines)
    skin = np.zeros((h, w), np.float32)
    cv2.fillPoly(skin, [np.round(lm[:17, :2]).astype(np.int32)], 1.0)
    fing = np.zeros((h, w), np.uint8)
    for k in range(5):
        for j in range(3):
            a, b = pts[1 + 4 * k + j], pts[2 + 4 * k + j]
            cv2.line(fing, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 255, 18)
    skin[fing > 0] = 0.0
    skin = cv2.GaussianBlur(skin, (0, 0), 2.0)
    parse = dict(skin=skin, trust=1.0)
    old = (RE.HAND_PARSE_TRIM, RE.HAND_CORE)
    try:
        RE.HAND_PARSE_TRIM = None
        c0 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)
        RE.HAND_PARSE_TRIM = old[0]
        c1 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)
    finally:
        RE.HAND_PARSE_TRIM, RE.HAND_CORE = old
    face = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face, [np.round(lm[:17, :2]).astype(np.int32)], 1)
    marg = (skin > 0.9) & (face > 0) & (c0 > 0.3)                      # the drawn hand's margin on the face's sure skin
    assert marg.sum() > 200
    assert float(c0[marg].mean()) > 2.5 * float(c1[marg].mean()), "the margin on the face the parser is sure of is given back"
    on_f = (fing > 0) & (face > 0)
    assert float(c1[on_f].mean()) > 0.85, "the fingers themselves stay covered"
