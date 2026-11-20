"""Instruction 2026-11-20 §1-4: joints with low scores in motion blur against joints that are really hidden.

visibility.hold_blur keeps a body joint whose score drops for a few frames when the frames before and after and
the frame's own matte say the limb is there (this person's pixels on the path between the two sightings), and
leaves it hidden behind another person, behind a thing, out of the frame, across a cut, when the track jumps to
another person, when it is gone for long, or when nothing says motion. It is off by default (BLUR_HOLD): with it
off the visibility is what 1.22.2 gave. Whether it is turned on is decided on footage (how many blur-lost joints
come back against how many hidden joints appear), not by these cases."""
import numpy as np
import pytest

from contour.track import skeleton as S
from contour.track import visibility as VI
from contour.track.store import VIS_HIDDEN, VIS_VISIBLE, TrackStore

N, W, H = 30, 640, 360
MW, MH = 320, 180
J = S.L_WR


def _store(two=True):
    st = TrackStore(N, [3, 8] if two else [3])
    st.present[:] = True
    base = np.array([[320, 60], [330, 55], [310, 55], [340, 60], [300, 60], [360, 110], [280, 110], [380, 170],
                     [260, 170], [400, 230], [240, 230], [350, 240], [290, 240], [350, 300], [290, 300], [350, 350],
                     [290, 350]], np.float32)
    raw = np.zeros((len(st.pids), N, S.K, 3), np.float32)
    for p in range(len(st.pids)):
        dx = -120.0 if p == 0 else 160.0
        for t in range(N):
            raw[p, t, :, :2] = base + [dx, 0]
            raw[p, t, :, 2] = 0.9
    # person 0's left wrist swings: 6 px a frame
    for t in range(N):
        raw[0, t, J, :2] = [120 + 6.0 * t, 230]
    st.raw["pose_a"] = raw
    st.meas = raw.copy()
    st.scale[:] = 300.0
    st.blur[:] = 1.0
    return st


def _mattes(st, wrist_owner=None):
    """Alpha and owners at half size: a block per person around its joints; the wrist's path of person 0 owned by
    wrist_owner (a store index, 'nobody' or None for its own person)."""
    alpha = np.zeros((N, MH, MW), np.uint8)
    own = np.zeros((N, MH, MW), np.uint8)
    for p in range(len(st.pids)):
        xs = st.raw["pose_a"][p, 0, :, 0]
        x0, x1 = int(xs.min() / 2) - 6, int(xs.max() / 2) + 6
        alpha[:, 20:MH, max(0, x0):x1] = 255
        own[:, 20:MH, max(0, x0):x1] = st.pids[p] + 1
    for t in range(N):
        cx, cy = int((120 + 6.0 * t) / 2), 115
        who = st.pids[0] + 1 if wrist_owner is None else (0 if wrist_owner == "nobody" else st.pids[wrist_owner] + 1)
        own[t, cy - 8:cy + 9, cx - 8:cx + 9] = who
        alpha[t, cy - 8:cy + 9, cx - 8:cx + 9] = 255 if who else 0
    return alpha, own


def _vis(st, alpha, own, hold=True, **over):
    old = {k: getattr(VI, k) for k in over}
    for k, v in over.items():
        setattr(VI, k, v)
    try:
        return VI.measure(st, {"pose_a": 1.0}, (W, H), lambda t: alpha[t], (0.5, 0.5),
                          owner_at=(None if own is None else (lambda t: own[t])), hold=hold)
    finally:
        for k, v in old.items():
            setattr(VI, k, v)


def _dip(st, a=10, b=15, score=0.1):
    st.raw["pose_a"][0, a:b, J, 2] = score
    st.meas = st.raw["pose_a"].copy()


def test_off_by_default_and_then_as_1222():
    st = _store()
    _dip(st)
    alpha, own = _mattes(st)
    assert VI.BLUR_HOLD is False
    v = VI.measure(st, {"pose_a": 1.0}, (W, H), lambda t: alpha[t], (0.5, 0.5), owner_at=lambda t: own[t])
    assert (v[0, 10:15, J] == VIS_HIDDEN).all() and (v[0, :9, J] == VIS_VISIBLE).all()
    assert np.array_equal(v, _vis(st, alpha, own, hold=False))


def test_low_scores_in_motion_with_own_pixels_are_held():
    st = _store()
    _dip(st)
    alpha, own = _mattes(st)
    v = _vis(st, alpha, own)
    assert (v[0, :, J] == VIS_VISIBLE).all(), v[0, :, J]
    assert np.array_equal(v[1], _vis(st, alpha, own, hold=False)[1]), "nobody else is touched"
    rep = {}
    low = np.zeros(v.shape, bool)
    low[0, 10:15, J] = True
    VI.hold_blur(st, _vis(st, alpha, own, hold=False), low, (W, H), lambda t: alpha[t], (0.5, 0.5), lambda t: own[t], rep)
    assert rep == dict(held_runs=1, held_joints=5)


@pytest.mark.parametrize("owner,why", [(1, "others_pixels"), ("nobody", "not_own_pixels")])
def test_joint_behind_someone_or_something_stays_hidden(owner, why):
    """The same scores, the same sightings before and after; the picture says the spot is not this person's."""
    st = _store()
    _dip(st)
    alpha, own = _mattes(st)
    a2, o2 = _mattes(st, wrist_owner=owner)
    alpha[11:14], own[11:14] = a2[11:14], o2[11:14]          # hidden on three frames in the middle of the dip
    v = _vis(st, alpha, own)
    assert (v[0, 10:15, J] == VIS_HIDDEN).all(), "the run is held whole or not at all"
    rep = {}
    low = np.zeros(v.shape, bool)
    low[0, 10:15, J] = True
    VI.hold_blur(st, _vis(st, alpha, own, hold=False), low, (W, H), lambda t: alpha[t], (0.5, 0.5), lambda t: own[t], rep)
    assert rep.get(why) == 1 and "held_runs" not in rep


def test_track_jumping_to_another_person_is_not_a_blur_dip():
    st = _store()
    _dip(st)
    st.raw["pose_a"][0, 15:, J, :2] += [400, 0]               # after the dip the 'wrist' is on someone else
    st.meas = st.raw["pose_a"].copy()
    alpha, own = _mattes(st)
    alpha[:], own[:] = 255, st.pids[0] + 1                    # (even if the pixels said yes)
    v = _vis(st, alpha, own)
    assert (v[0, 10:15, J] == VIS_HIDDEN).all()


def test_leaving_the_frame_long_gaps_cuts_and_no_second_sighting():
    alpha_own = lambda st: _mattes(st)                         # noqa: E731
    # gone for longer than a blur dip lasts
    st = _store(); _dip(st, 8, 8 + VI.HOLD_MAX + 1)
    a, o = alpha_own(st)
    assert (_vis(st, a, o)[0, 8:8 + VI.HOLD_MAX + 1, J] == VIS_HIDDEN).all()
    # never seen again afterwards (it left, or the shot ends)
    st = _store(); _dip(st, 22, N)
    a, o = alpha_own(st)
    assert (_vis(st, a, o)[0, 23:, J] == VIS_HIDDEN).all()
    # a cut inside the stretch between the two sightings
    st = _store(); _dip(st)
    st.jump[12] = True
    a, o = alpha_own(st)
    assert (_vis(st, a, o)[0, 10:15, J] == VIS_HIDDEN).all()
    # the path between the sightings runs out of the picture
    st = _store(); _dip(st)
    st.raw["pose_a"][0, :, J, 0] += 470                       # at the right edge, crossing it during the dip
    st.meas = st.raw["pose_a"].copy()
    a, o = alpha_own(st)
    v = _vis(st, a, o)
    assert not (v[0, 10:15, J] == VIS_VISIBLE).any()


def test_a_still_limb_needs_a_blurred_picture():
    st = _store()
    st.raw["pose_a"][0, :, J, :2] = [200, 230]                # the wrist does not move
    _dip(st)
    alpha, own = _mattes(st)
    alpha[:, 105:125, 90:110], own[:, 105:125, 90:110] = 255, st.pids[0] + 1
    assert (_vis(st, alpha, own)[0, 10:15, J] == VIS_HIDDEN).all(), "nothing says motion: something is in front of it"
    st.blur[0, 11:14] = 0.4                                   # the person is smeared on those frames
    assert (_vis(st, alpha, own)[0, 10:15, J] == VIS_VISIBLE).all()


def test_estimate_far_from_the_path_is_not_the_limb():
    st = _store()
    _dip(st)
    st.raw["pose_a"][0, 12, J, :2] += [0, 120]                # the low-score guess is somewhere else
    st.meas = st.raw["pose_a"].copy()
    alpha, own = _mattes(st)
    assert (_vis(st, alpha, own)[0, 10:15, J] == VIS_HIDDEN).all()


def test_matte_without_owners_only_says_somebody_is_there():
    st = _store(two=False)
    _dip(st)
    alpha, own = _mattes(st)
    v = _vis(st, alpha, None)
    assert (v[0, 10:15, J] == VIS_VISIBLE).all()
    alpha[11:14] = 0
    assert (_vis(st, alpha, None)[0, 10:15, J] == VIS_HIDDEN).all()
