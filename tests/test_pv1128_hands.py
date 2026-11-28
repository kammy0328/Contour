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



def _thumb(st, q, start, step, conf, i0=0):
    """the hand's thumb (points 1..4) from `start`, `step` apart, all of this confidence"""
    for j in range(4):
        st.hand_raw[q, 0, i0 + 1 + j, :2] = np.asarray(start, np.float64) + np.asarray(step, np.float64) * j
        st.hand_raw[q, 0, i0 + 1 + j, 2] = conf


def _cover_at(st, lm, pts, held):
    old = RE.HAND_THUMB_HELD
    try:
        RE.HAND_THUMB_HELD = held
        cv_ = RE.limb_cover(st, 0, lm, (600, 700), 1.0, 1.0, 0, None)
    finally:
        RE.HAND_THUMB_HELD = old
    assert cv_ is not None
    return np.array([float(cv_[int(round(y)), int(round(x))]) for x, y in pts])


def _hand_on_cheek():
    """a hand on the face's left cheek, fingers up (the index finger at x 289, the little finger at 332), its forearm from below"""
    st = _Store(1)
    _put_arm(st, 0, (300.0, 590.0), (300.0, 430.0), 30.0, "L")
    pts = _put_hand(st, 0, (300.0, 430.0), up=(0.0, -1.0), size=150.0, conf=5.0)
    return st, pts


def test_a_thumb_out_on_the_face_alone_is_held_to_the_hand():
    """c024 f6: the right hand on the cheek, its fingers up; the model's thumb (less sure than the fingers) from the jaw up beside the mouth to the nose,
    parallel to the fingers and out on the face alone: the cheek there is not covered (it was, the client's circle at 48:20)"""
    lm = _face_lm()
    st, pts = _hand_on_cheek()
    _thumb(st, 0, (230.0, 400.0), (0.0, -35.0), 3.4)                    # 59 px beside the index finger, along it
    on_thumb = [(230.0, 330.0), (230.0, 300.0)]
    c_old = _cover_at(st, lm, on_thumb, None)
    c_new = _cover_at(st, lm, on_thumb, RE.HAND_THUMB_HELD)
    assert (c_old > 0.9).all(), "without the rule the thumb's strip is covered"
    assert (c_new < 0.1).all(), "the cheek under the model's lone thumb keeps its retouch"
    fing = [tuple(pts[1 + 4 * k + 2]) for k in range(1, 5)]
    assert (_cover_at(st, lm, fing, RE.HAND_THUMB_HELD) > 0.9).all(), "the fingers stay covered"


def test_a_thumb_the_model_is_sure_of_or_at_an_angle_stays():
    """the same thumb, but the model as sure of it as of the fingers (c023's heart, c021's at the chin), or at an angle to them (c022's under the lip): covered"""
    lm = _face_lm()
    st, _ = _hand_on_cheek()
    _thumb(st, 0, (230.0, 400.0), (0.0, -35.0), 4.6)                    # sure
    assert (_cover_at(st, lm, [(230.0, 330.0), (230.0, 300.0)], RE.HAND_THUMB_HELD) > 0.9).all()
    st2, _ = _hand_on_cheek()
    _thumb(st2, 0, (270.0, 420.0), (-30.0, -20.0), 3.4)                 # unsure, out at 56 degrees to the fingers
    assert (_cover_at(st2, lm, [(225.0, 390.0), (195.0, 370.0)], RE.HAND_THUMB_HELD) > 0.9).all()


def test_a_thumb_laid_against_the_index_finger_stays():
    """an unsure thumb along the fingers but against the index finger (the hand flat on the cheek, the thumb closed): still the hand"""
    lm = _face_lm()
    st, pts = _hand_on_cheek()
    xi = float(pts[5:9, 0].mean())
    _thumb(st, 0, (xi - 11.0, 395.0), (0.0, -30.0), 3.4)
    assert (_cover_at(st, lm, [(xi - 11.0, 350.0), (xi - 11.0, 320.0)], RE.HAND_THUMB_HELD) > 0.9).all()


def test_a_heart_thumb_leaving_the_palm_at_an_angle_stays():
    """c023 f10, the heart: the fingers bend in, so the middle finger's last bones run as the thumb does at the heart's foot, and the model was 0.82 as sure
    of the thumb as of the fingers; but the thumb leaves the palm's line (wrist to the middle knuckle) at about 40 degrees: it is the hand, covered"""
    lm = _face_lm()
    st = _Store(1)
    pts = np.array([(230, 390),
                    (260, 375), (297.6, 361.3), (335.2, 347.6), (372.8, 334),       # the thumb: 20 degrees up, along the heart's foot
                    (285, 265), (300, 235), (325, 220), (350, 215),                # index
                    (300, 260), (318, 228), (345, 213), (372, 210),                # middle: its last bones 35 degrees up
                    (312, 265), (330, 238), (352, 225), (375, 222),                # ring
                    (322, 275), (338, 253), (356, 242), (375, 238)], np.float32)   # little
    st.present[0, 0] = True
    st.hand_raw[0, 0, :21, :2] = pts
    st.hand_raw[0, 0, :21, 2] = 5.0
    st.hand_raw[0, 0, 1:5, 2] = 4.1                                         # 0.82 of the fingers
    on_thumb = [(316.0, 354.0), (345.0, 344.0)]
    assert RE._thumb_held(st.hand_raw[0, 0, :21, 2], pts) is False
    assert (_cover_at(st, lm, on_thumb, RE.HAND_THUMB_HELD) > 0.9).all(), "the heart's thumb is covered"
    assert (_cover_at(st, lm, on_thumb, RE.HAND_THUMB_HELD[:2]) < 0.1).all(), "(without the palm's angle the rule took it)"


def test_an_arm_of_somebody_farther_away_does_not_cover_the_face():
    """(LIMB_OTHER_FAR, the d18 tone check: c002 f56..105, the upper arm of the person standing behind 률 reached over his jaw and the parser's trust was too
    low there for LIMB_OTHER_SKIN to see it behind) somebody clearly smaller (farther from the camera) is behind the face; somebody as big or bigger is not."""
    h, w = 600, 700
    lm = _face_lm()
    covers = {}
    for other_scale in (250.0, 400.0, 520.0):
        st = _Store(2)
        st.scale[0, 0] = 400.0
        st.scale[1, 0] = other_scale
        _put_arm(st, 1, (520.0, 300.0), (360.0, 330.0), 25.0, "L")      # a forearm across the lower face
        cov = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
        covers[other_scale] = 0.0 if cov is None else float((np.asarray(cov) > 0.5).sum())
    assert covers[250.0] == 0.0, "a farther person's arm is behind the head"
    assert covers[400.0] > 500 and covers[520.0] > 500, "somebody as near (or nearer) can have an arm in front"
    old = RE.LIMB_OTHER_FAR
    try:
        RE.LIMB_OTHER_FAR = None
        st = _Store(2)
        st.scale[0, 0], st.scale[1, 0] = 400.0, 250.0
        _put_arm(st, 1, (520.0, 300.0), (360.0, 330.0), 25.0, "L")
        cov = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, None)
        assert cov is not None and float((np.asarray(cov) > 0.5).sum()) > 500, "off: as before"
    finally:
        RE.LIMB_OTHER_FAR = old
