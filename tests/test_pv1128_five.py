"""The places left open after the checks of the 11/28 renders (LNGSHOT BEEP BEEP), one rule each:

  - NECK_STRAND (c046 f5): a strand of hair hanging across the neck keeps its line; a crease of the neck and a fold between two differently lit
    sides do not count as one.
  - CHIN_PEBBLE_SLIP (c030 f41..43): landmarks that sit off the path of the frames before and after them have slid; the pebbled way of the chin's
    stages fades out on that frame. Job.slip is set by prepare() and, on a frame read from a saved analysis, by late().
  - DOUBLE_SURER (c023 f2): a face held by two tracks takes the landmarks found on the frame.
  - PARSE_FILL_INSIDE (c024 f12..16): the parser's fill of the colour model's holes also goes by how much of its sure skin lies inside the face."""
import numpy as np
import cv2

from contour.retouch import engine as RE

from test_pv1128_hands import _face_lm


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


# --------------------------------------------------------------------------- PARSE_FILL_INSIDE

def _sunlit_cheek():
    """A face whose colour model leaves a cheek in the sun out, and a parser that sees its upper half only (hands on the cheeks, the mouth and the chin
    between them not seen)"""
    h, w = 600, 700
    lm = _face_lm(350.0, 280.0, 240.0)
    face = np.zeros((h, w), np.float32)
    cv2.fillPoly(face, [np.round(RE.SK.face_polygon(lm)).astype(np.int32)], 1.0)
    patch = np.zeros((h, w), np.float32)
    cv2.circle(patch, (425, 270), 22, 1.0, -1)
    upper = face.copy()
    upper[300:] = 0.0
    return h, w, lm, face, patch > 0.5, upper


def test_sure_skin_inside_the_face_is_trusted_however_little_of_the_face_it_is():
    h, w, lm, face, patch, upper = _sunlit_cheek()
    SK = RE.SK
    assert SK.parse_trust_inside(upper, lm, (h, w)) > 0.99, "the upper face alone: all of it inside"
    assert SK.parse_trust_inside(face, lm, (h, w)) > 0.99
    assert SK.parse_trust_inside(np.roll(upper, 150, axis=1), lm, (h, w)) < 0.01, "skin beside the face: the landmarks or the crop slid"
    few = np.zeros((h, w), np.float32)
    few[250:262, 340:352] = 1.0
    assert SK.parse_trust_inside(few, lm, (h, w)) == 0.0, "too little sure skin to tell"
    assert SK.parse_trust_inside(upper, lm, (h, w), rule=None) > 0.99 and SK.parse_trust_inside(None, lm, (h, w)) == 0.0


def test_the_parsers_fill_of_a_cheek_in_the_sun_holds_when_hands_take_its_share_of_the_face_down():
    """c024 f12..16: PARSE_FILL went by the parser's trust (0.15 with both hands on the cheeks) and the cheek in the sun beside the fingers, filled
    on the frames round them, was left for five frames"""
    h, w, lm, face, patch, upper = _sunlit_cheek()
    SK = RE.SK
    prob0 = face * (1.0 - patch.astype(np.float32))
    rgb = np.full((h, w, 3), 128, np.uint8)
    job = SK.SkinJob(0, 0.7, 0.5, "face", lm.copy(), 1.0, 400.0)

    hand = np.zeros((h, w), np.float32)                 # a hand laid on the other cheek and the jaw: a fifth of the face
    cv2.ellipse(hand, (275, 330), (50, 95), 20, 0, 360, 1.0, -1)
    assert 0.12 < float(hand[face > 0].mean()) < 0.35

    def mask(skin, trust, cover=hand):
        return SK.skin_mask(job, rgb, 1.0, 1.0, None, None, dict(skin=skin, trust=trust), None, cover, prob0)

    full = float(mask(face, 1.0)[patch].mean())
    assert full > 0.7, "a trusted parser fills the cheek"
    assert abs(float(mask(upper, 0.15)[patch].mean()) - full) < 0.02, "... and so does one that sees the upper face only, a hand on the face"
    assert float(mask(np.roll(upper, 150, axis=1), 0.15)[patch].mean()) < 0.25, "skin beside the face stays at its trust"
    assert float(mask(upper, 0.15, None)[patch].mean()) < 0.25, "no hand on the face: the trust fell for another reason, and holds"
    assert float(mask(upper, 0.15, hand * 0.2)[patch].mean()) < 0.25, "... nor a hand that takes less than PARSE_FILL_COVER of it"
    old = SK.PARSE_FILL_INSIDE
    try:
        SK.PARSE_FILL_INSIDE = None
        assert float(mask(upper, 0.15)[patch].mean()) < 0.25, "(as before: by the trust alone)"
    finally:
        SK.PARSE_FILL_INSIDE = old


# --------------------------------------------------------------------------- DOUBLE_SURER

def test_a_face_held_by_two_tracks_takes_the_landmarks_found_on_the_frame():
    """c023 f0..3: the face finder's face goes to one of the two tracks a frame; the other's landmarks are carried there (score 0)"""
    st = _Faces([[(300.0, 260.0)] * 3, [(308.0, 266.0)] * 3, [(700.0, 260.0)] * 3])      # the two jaw lines 10 px apart: 0.05 of the face width
    st.face[0, :, :, 2] = 0.8
    st.face[1, :, :, 2] = 0.8
    st.face[2, :, :, 2] = 0.9
    st.face[0, 1, :, 2] = 0.0                      # track 0's landmarks of frame 1 were carried there
    assert RE._double_face(st, 0, 0) is None, "found on the frame: its own"
    got = RE._double_face(st, 0, 1)
    assert got is not None and np.allclose(got, st.face[1, 1]), "carried: the other track's, found there"
    assert RE._double_face(st, 1, 1) is None
    st.face[1, 1, :, 2] = 0.3                      # the other track's are not sure either
    assert RE._double_face(st, 0, 1) is None
    st.face[1, 1, :, 2] = 0.8
    st.face[2, 1, :, 2] = 0.0
    assert RE._double_face(st, 2, 1) is None, "somebody else's face elsewhere is not this face"
    st.present[1, 1] = False
    assert RE._double_face(st, 0, 1) is None, "a track that is not there on the frame"
    st.present[1, 1] = True
    st.face[1, 1, :, :2] = st.face[0, 1, :, :2] + np.float32([3.0, 2.0])      # 3.6 px apart, 0.018: carried landmarks that sit where the found ones do
    assert RE._double_face(st, 0, 1) is None
    st.face[1, 1, :, :2] = st.face[0, 1, :, :2] + np.float32([8.0, 6.0])
    assert RE._double_face(st, 0, 1) is not None
    st.face[1, 1, :, :2] = st.face[0, 1, :, :2] + np.float32([36.0, 27.0])    # 45 px apart, 0.22: two faces crossing (c022 f11)
    assert RE._same_face(st, 1, 0, 1) and RE._double_face(st, 0, 1) is None
    st.face[1, 1, :, :2] = st.face[0, 1, :, :2] + np.float32([8.0, 6.0])
    old = RE.DOUBLE_SURER
    try:
        RE.DOUBLE_SURER = None
        assert RE._double_face(st, 0, 1) is None
    finally:
        RE.DOUBLE_SURER = old


# --------------------------------------------------------------------------- the forehead (11/29: 이마 보정 경계면, 이마 점, 뾰루지)

def _forehead(h=360, w=480, hairline=60, level=0.6):
    """A forehead lit evenly, hair above it; its mask, the skin below the hairline"""
    Y = np.full((h, w), level, np.float32)
    Y[:hairline] = 0.08
    Y = cv2.GaussianBlur(Y, (0, 0), 6)                     # the hairline's ramp
    m = np.zeros((h, w), np.float32)
    m[hairline + 4:h - 30, 30:w - 30] = 1.0
    return Y, cv2.GaussianBlur(m, (0, 0), 3)


def test_the_tone_softening_leaves_the_masks_border_and_a_patch_of_sun():
    wf = 400.0
    Y, m = _forehead()
    w = (m > 0.2).astype(np.float32)
    sel = RE._forehead_step_sel(Y, w, wf)
    assert float(sel[64:90].max()) < 0.05, "the hairline's ramp at the mask's border"
    Y2 = Y.copy()
    Y2[150:, 260:] += 0.25                                  # a patch of sun: a step of 40 %
    Y2 = cv2.GaussianBlur(Y2, (0, 0), 2)
    assert float(RE._forehead_step_sel(Y2, w, wf)[150:300, 230:290].max()) < 0.05
    Y3 = Y.copy()
    Y3[150:, :] -= 0.02                                     # a foundation's line: 3 %
    Y3 = cv2.GaussianBlur(Y3, (0, 0), 1.5)
    assert float(RE._forehead_step_sel(Y3, w, wf)[140:160, 150:330].max()) > 0.5, "a faint line in the middle is softened"
    old = (RE.FOREHEAD_STEP_EDGE, RE.FOREHEAD_STEP_HIGH)
    try:
        RE.FOREHEAD_STEP_EDGE, RE.FOREHEAD_STEP_HIGH = None, None
        assert float(RE._forehead_step_sel(Y2, w, wf)[150:300, 230:290].max()) > 0.3, "(before: the sun's rim was taken)"
    finally:
        RE.FOREHEAD_STEP_EDGE, RE.FOREHEAD_STEP_HIGH = old


def test_a_blot_on_the_forehead_is_filled_and_the_hairline_and_the_sun_stay():
    wf = 400.0
    Y, m = _forehead()
    Y[180:300, 280:] += 0.2                                  # a patch of sun, its shade beside it
    blot = np.zeros_like(Y)
    cv2.circle(blot, (180, 170), 9, 1.0, -1)
    Yb = Y - 0.03 * cv2.GaussianBlur(blot, (0, 0), 3)        # a brown mark 3 levels in 100 deep, 0.05 of the face wide
    out = RE._forehead_blot(Yb, m, wf, 1.0)
    assert float(Yb[170, 180] - Y[170, 180]) < -0.02
    assert abs(float(out[170, 180] - Y[170, 180])) < 0.006, "the blot is filled"
    assert float(np.abs(out - Yb)[50:75].max()) < 0.004, "the hairline is left"
    assert float(np.abs(out - Yb)[180:300, 270:290].max()) < 0.004, "the edge of the sun is left"
    Cb = np.zeros_like(Yb)
    Cb[blot > 0.5] = 0.02
    y2, cb2, cr2 = RE._forehead_blot(Yb, m, wf, 1.0, (Cb, Cb.copy(), 1.0))
    assert float(cb2[170, 180]) < 0.008, "and its colour"


def test_the_forehead_zone_reaches_past_the_brows_ends_whatever_their_order():
    lm = _face_lm(300.0, 300.0, 200.0)
    z = RE.forehead_zone(lm, (600, 600), 1.0, 200.0)
    ends = lm[[17, 26], 0]
    up_y = lm[17, 1] - 40
    assert float(z[int(up_y), int(ends.min()) - 12]) > 0.5 and float(z[int(up_y), int(ends.max()) + 12]) > 0.5
    lm2 = lm.copy()
    lm2[17:27] = lm2[[19, 17, 22, 18, 21, 20, 26, 23, 25, 24]]  # the brows' points out of order (a hand at the face)
    RE._cache_clear() if hasattr(RE, "_cache_clear") else None
    z2 = RE.forehead_zone(lm2, (600, 600), 1.0, 200.0)
    assert float(np.abs(z2 - z).max()) < 0.05
