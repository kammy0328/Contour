"""The places left open after the checks of the 11/28 renders (LNGSHOT BEEP BEEP), one rule each:

  - NECK_STRAND (c046 f5): a strand of hair hanging across the neck keeps its line; a crease of the neck and a fold between two differently lit
    sides do not count as one.
  - CHIN_PEBBLE_SLIP (c030 f41..43): landmarks that sit off the path of the frames before and after them have slid; the pebbled way of the chin's
    stages fades out on that frame. Job.slip is set by prepare() and, on a frame read from a saved analysis, by late().
  - LIMB_SAME_FACE (c023 f2): the hands of a track that holds this same face are this person's own.
  - HAND_TRUST_LOCAL (c024 f12..16): the parser's trust for the drawn hands' rim is also measured on the face the hands leave free."""
import numpy as np
import cv2

from contour.retouch import engine as RE
from contour.track import skeleton as S
from contour.warp.body import SEGMENTS

from test_pv1128_hands import _face_lm, _put_hand


# --------------------------------------------------------------------------- NECK_STRAND

def _neck(h=420, w=420):
    """A lit neck (0.6) with its mask, hair (0.08) above it"""
    Y = np.full((h, w), 0.60, np.float32)
    Y[:60] = 0.08
    m = np.zeros((h, w), np.float32)
    m[60:380, 60:360] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 6)
    return Y, m


def test_a_strand_hanging_from_the_hair_across_the_neck_is_found():
    Y, m = _neck()
    cv2.line(Y, (200, 40), (212, 300), 0.50, 3, cv2.LINE_AA)               # a strand: thin, 17 % darker, along the face's axis, out of the hair
    H = RE._neck_strands(Y, m, np.array([0.0, 1.0]), 200.0)
    assert H is not None
    assert float(H[150, 205]) > 0.5 and float(H[250, 210]) > 0.5, "along the strand"
    assert float(H[200, 120]) < 0.05 and float(H[200, 300]) < 0.05, "not beside it"


def test_a_crease_across_the_neck_is_no_strand():
    Y, m = _neck()
    cv2.line(Y, (90, 200), (330, 206), 0.50, 3, cv2.LINE_AA)               # a ring of the neck: across the axis
    assert RE._neck_strands(Y, m, np.array([0.0, 1.0]), 200.0) is None


def test_a_line_between_two_differently_lit_sides_is_no_strand():
    Y, m = _neck()
    Y[60:, 206:] = 0.42                                                    # the neck's shaded side: a fold's line stands at its edge
    cv2.line(Y, (205, 40), (205, 300), 0.39, 3, cv2.LINE_AA)               # (the sides differ by 1.5 times the line's own depth)
    assert RE._neck_strands(Y, m, np.array([0.0, 1.0]), 200.0) is None
    old = RE.NECK_STRAND_SIDES
    try:
        RE.NECK_STRAND_SIDES = None
        assert RE._neck_strands(Y, m, np.array([0.0, 1.0]), 200.0) is not None, "(without the sides' test it was taken)"
    finally:
        RE.NECK_STRAND_SIDES = old


def test_a_line_that_hangs_from_nothing_dark_is_no_strand():
    Y, m = _neck()
    cv2.line(Y, (205, 140), (212, 330), 0.50, 3, cv2.LINE_AA)              # starts in the middle of the neck: a tendon's shade, not hair
    assert RE._neck_strands(Y, m, np.array([0.0, 1.0]), 200.0) is None


# --------------------------------------------------------------------------- CHIN_PEBBLE_SLIP

class _Faces:
    has_face = True

    def __init__(self, paths):
        """paths: [per person: [(x, y) per frame or None]]: a face 200 wide round each point"""
        n, T = len(paths), len(paths[0])
        self.face = np.full((n, T, 68, 3), np.nan, np.float32)
        self.present = np.zeros((n, T), bool)
        for p, path in enumerate(paths):
            for t, c in enumerate(path):
                if c is None:
                    continue
                self.face[p, t] = _face_lm(c[0], c[1], 200.0)
                self.present[p, t] = True


def test_landmarks_on_a_steady_path_have_not_slid():
    st = _Faces([[(300.0 + 60.0 * t, 260.0 + 10.0 * t) for t in range(6)]])        # a third of the face's width a frame
    for t in range(1, 5):
        assert RE._face_slip(st, 0, t) < 0.01
    assert RE._face_slip(st, 0, 0) is None and RE._face_slip(st, 0, 5) is None, "no frame before / after: not known"


def test_landmarks_off_the_path_for_a_frame_have_slid():
    path = [(300.0 + 60.0 * t, 260.0) for t in range(6)]
    path[3] = (path[3][0] + 20.0, path[3][1] + 50.0)                               # down onto the collar: 54 px, 0.27 of the face width
    st = _Faces([path])
    assert abs(RE._face_slip(st, 0, 3) - 0.27) < 0.02
    assert abs(RE._face_slip(st, 0, 2) - 0.135) < 0.02, "the frames beside it see half of it"
    st.face[0, 4] = np.nan
    assert RE._face_slip(st, 0, 3) is None


def test_the_pebbled_way_fades_out_on_slid_landmarks():
    lm = _face_lm()
    v = {"chin_pebble": 1.0}
    full = RE._pebble_share(lm, v, 0.2)
    assert full > 0.99
    lo, hi = RE.CHIN_PEBBLE_SLIP
    assert RE._pebble_share(lm, v, 0.2, None) == full
    assert RE._pebble_share(lm, v, 0.2, lo - 0.02) == full
    mid = RE._pebble_share(lm, v, 0.2, 0.5 * (lo + hi))
    assert 0.4 < mid < 0.6
    assert RE._pebble_share(lm, v, 0.2, hi + 0.01) == 0.0
    old = RE.CHIN_PEBBLE_SLIP
    try:
        RE.CHIN_PEBBLE_SLIP = None
        assert RE._pebble_share(lm, v, 0.2, hi + 0.2) == full
    finally:
        RE.CHIN_PEBBLE_SLIP = old


def test_a_frame_read_from_a_saved_analysis_gets_its_slip():
    path = [(300.0 + 60.0 * t, 260.0) for t in range(6)]
    path[3] = (path[3][0], path[3][1] + 50.0)
    st = _Faces([path, [None] * 6])

    class _Tracker:
        store = st

    jobs = [RE.Job(0, 7, {}, 0.5, "face", 1.0, st.face[0, 3].copy(), 1.0, 400.0), RE.Job(1, 8, {}, 0.5, "face", 1.0, None, 0.0, 400.0)]
    rf = RE.RetouchFrame(3, jobs, (640, 360), (1280, 720))
    assert jobs[0].slip is None
    assert RE.late(rf, _Tracker()) is rf
    assert abs(jobs[0].slip - RE._face_slip(st, 0, 3)) < 1e-9 and jobs[0].slip > 0.2
    assert jobs[1].slip is None, "no face: none"
    assert RE.late(None, _Tracker()) is None


# --------------------------------------------------------------------------- hands

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
        self.has_face = True
        self.face = np.full((n, 1, 68, 3), np.nan, np.float32)
        self.pids = np.arange(n)


def _polygon_skin(lm, shape):
    sk = np.zeros(shape, np.float32)
    cv2.fillPoly(sk, [np.round(RE.SK.face_polygon(lm)).astype(np.int32)], 1.0)
    return sk


def test_the_hand_of_a_track_holding_the_same_face_is_this_persons_own():
    """c023 f2: the palm lies along the jaw, part of it over skin the parser is sure of; held by the other track of the same person it counted as
    somebody else's hand behind the head and the palm beside the jaw was left to the retouch"""
    h, w = 800, 800
    lm = _face_lm(400.0, 300.0, 300.0)
    st = _Store(2)
    st.present[:, 0] = True
    st.face[0, 0] = lm
    st.face[1, 0] = lm + np.float32([6.0, 4.0, 0.0])                       # the same face, a few px off
    pts = _put_hand(st, 1, (330.0, 700.0), size=300.0)                     # the other track's hand, from below up over the jaw
    parse = dict(skin=_polygon_skin(lm, (h, w)), trust=1.0)
    inside = parse["skin"] > 0.5
    palm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(palm, [np.round(pts[[0, 5, 9, 13, 17]]).astype(np.int32)], 1)
    below = (palm > 0) & ~inside                                           # the palm under the jaw: no face skin
    assert int(below.sum()) > 500
    cov = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)
    assert cov is not None and float(np.asarray(cov)[below].mean()) > 0.8, "the palm beside the jaw is covered"
    old = RE.LIMB_SAME_FACE
    try:
        RE.LIMB_SAME_FACE = False
        cov0 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)
        assert cov0 is None or float(np.asarray(cov0)[below].mean()) < 0.1, "(as somebody else's it lay behind the head)"
    finally:
        RE.LIMB_SAME_FACE = old
    # somebody else's face elsewhere is somebody else
    st.face[1, 0] = _face_lm(400.0 + 200.0, 300.0, 300.0)
    assert not RE._same_face(st, 0, 0, 1)
    cov1 = RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse)
    assert cov1 is None or float(np.asarray(cov1)[below].mean()) < 0.1


def test_the_drawn_hands_rim_goes_by_the_trust_of_the_face_they_leave_free():
    """c024 f12..16: both hands on the cheeks; the parser sees the face's skin right up to the fingers, but its trust (the share of the inner face it
    calls skin) is low because the hands lie on it"""
    h, w = 800, 800
    lm = _face_lm(400.0, 300.0, 300.0)
    st = _Store(1)
    pl = _put_hand(st, 0, (290.0, 640.0), size=300.0, i0=0)
    pr = _put_hand(st, 0, (510.0, 640.0), size=300.0, i0=21)
    # what the hands really cover: palm and fingers, 9 px thick (the drawn rim reaches about 20)
    real = np.zeros((h, w), np.uint8)
    for pts in (pl, pr):
        cv2.fillPoly(real, [np.round(pts[[0, 1, 2, 5, 9, 13, 17]]).astype(np.int32)], 1)
        for ch in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20), (0, 5), (0, 17)):
            for a, b in zip(ch[:-1], ch[1:]):
                cv2.line(real, tuple(int(round(x)) for x in pts[a]), tuple(int(round(x)) for x in pts[b]), 1, 18)
    skin = _polygon_skin(lm, (h, w)) * (1.0 - real)
    parse = dict(skin=skin, trust=0.15)
    cov = np.asarray(RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse))
    old = RE.HAND_TRUST_LOCAL
    try:
        RE.HAND_TRUST_LOCAL = None
        cov0 = np.asarray(RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse))
    finally:
        RE.HAND_TRUST_LOCAL = old
    rim = (cov0 > 0.5) & (skin > 0.99) & (cv2.dilate(real, np.ones((13, 13), np.uint8)) == 0)       # the face's skin in the drawn rim, clear of the hand
    assert int(rim.sum()) > 300
    assert float(cov[rim].mean()) < 0.15, "the cheek beside the fingers is the face's again"
    hand = cv2.erode(real, np.ones((9, 9), np.uint8)) > 0
    assert float(cov[hand].mean()) > 0.85 and float((cov0 - cov)[hand].mean()) < 0.03, "the hands stay covered (the rim's soft edge at the fingertips aside)"
    assert float((cov - cov0).max()) < 1e-4, "nothing is covered that was not"
    # a face the parser does not see where the hands leave it free: its trust stays low
    parse2 = dict(skin=skin * 0.3, trust=0.0)
    c2 = np.asarray(RE.limb_cover(st, 0, lm, (h, w), 1.0, 1.0, 0, parse2))
    assert float(c2[rim].mean()) > 0.5
