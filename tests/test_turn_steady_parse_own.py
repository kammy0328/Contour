"""(1.27.1) the face's turn held steady over frames (engine.TURN_STEADY), and a face in profile whose matte misses its lit cheek takes the parser's sure
skin inside its outline (skin.PARSE_OWN_TURN, PARSE_OWN_MATTE)."""
from types import SimpleNamespace

import numpy as np

from baseline_util import face68


def test_turn_steady_is_the_mean_round_the_frame():
    from contour.retouch import engine as EN
    T = 9
    face = np.full((1, T, 68, 3), np.nan, np.float32)
    for t in range(T):
        lm = face68(200.0, 50.0, 200.0).astype(np.float32)
        if t == 4:
            lm[30, 0] += 60.0                                 # one frame's nose swung far to a side
        face[0, t] = lm
    st = SimpleNamespace(face=face)
    own = EN.face_turn(face[0, 4])
    held = EN._turn_steady(st, 0, 4, 3)
    assert own > 0.2 and held < 0.6 * own
    assert EN._turn_steady(st, 0, 0, 0) == EN.face_turn(face[0, 0])
    face[0, 2] = np.nan
    assert EN._turn_steady(st, 0, 2, 3) is None


def test_the_turn_of_skin_is_the_engines():
    from contour.retouch import engine as EN
    from contour.retouch import skin as SK
    lm = face68(200.0, 50.0, 200.0).astype(np.float32)
    lm[30, 0] += 50.0
    assert abs(SK._turn(lm) - EN.face_turn(lm)) < 1e-9
    assert SK.PARSE_OWN_TURN is not None and 0.3 <= SK.PARSE_OWN_TURN <= 0.6
