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

    def mask(skin, trust):
        return SK.skin_mask(job, rgb, 1.0, 1.0, None, None, dict(skin=skin, trust=trust), None, None, prob0)

    full = float(mask(face, 1.0)[patch].mean())
    assert full > 0.7, "a trusted parser fills the cheek"
    assert abs(float(mask(upper, 0.15)[patch].mean()) - full) < 0.02, "... and so does one that sees the upper face only"
    assert float(mask(np.roll(upper, 150, axis=1), 0.15)[patch].mean()) < 0.25, "skin beside the face stays at its trust"
    old = SK.PARSE_FILL_INSIDE
    try:
        SK.PARSE_FILL_INSIDE = None
        assert float(mask(upper, 0.15)[patch].mean()) < 0.25, "(as before: by the trust alone)"
    finally:
        SK.PARSE_FILL_INSIDE = old


# --------------------------------------------------------------------------- DOUBLE_SURER

def test_a_face_held_by_two_tracks_takes_the_landmarks_found_on_the_frame():
    """c023 f0..3: the face finder's face goes to one of the two tracks a frame; the other's landmarks are carried there (score 0)"""
    st = _Faces([[(300.0, 260.0)] * 3, [(306.0, 264.0)] * 3, [(700.0, 260.0)] * 3])
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
    old = RE.DOUBLE_SURER
    try:
        st.present[1, 1] = True
        RE.DOUBLE_SURER = None
        assert RE._double_face(st, 0, 1) is None
    finally:
        RE.DOUBLE_SURER = old
