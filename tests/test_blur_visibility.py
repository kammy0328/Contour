"""Motion blur and visibility (2026-11-20, five projects): 32 % of body joints missing from the final track on
blurred frames against 21 % on sharp ones, 90 % of them judged hidden for the score alone (blur lowers every
model's score under SCORE_HIDE). A joint visible right before and right after a short blurred stretch is carried
through it; a long stretch, a sharp hidden frame (a real occlusion), a picture jump or the person leaving stops it."""
import numpy as np

from contour.track import visibility as VI
from contour.track.store import TrackStore, VIS_HIDDEN, VIS_VISIBLE


def _case(hidden, blurred, n=20):
    st = TrackStore(n, [1])
    st.present[:] = True
    st.blur[:] = 1.0
    st.blur[0, blurred] = 0.3
    vis = np.full((1, n, 17), VIS_VISIBLE, np.uint8)
    vis[0, hidden, 9] = VIS_HIDDEN
    return st, vis


def test_a_short_blurred_dip_is_carried_through():
    st, vis = _case(slice(8, 11), slice(8, 11))
    out = VI.bridge_blur(vis, st)
    assert (out[0, :, 9] == VIS_VISIBLE).all()
    assert (vis[0, 8:11, 9] == VIS_HIDDEN).all(), "the input is not changed"


def test_what_is_not_bridged():
    st, vis = _case(slice(5, 13), slice(5, 13))                  # 8 blurred frames: too long
    assert (VI.bridge_blur(vis, st)[0, 5:13, 9] == VIS_HIDDEN).all()
    st, vis = _case(slice(8, 12), slice(8, 11))                  # hidden on a sharp frame too: an occlusion
    assert (VI.bridge_blur(vis, st)[0, 8:12, 9] == VIS_HIDDEN).all()
    st, vis = _case(slice(8, 11), slice(8, 11))
    st.jump[9] = True                                            # a cut inside
    assert (VI.bridge_blur(vis, st)[0, 8:11, 9] == VIS_HIDDEN).all()
    st, vis = _case(slice(8, 11), slice(8, 11))
    vis[0, 11, 9] = VIS_HIDDEN                                   # not seen after
    assert (VI.bridge_blur(vis, st)[0, 8:11, 9] == VIS_HIDDEN).all()
    st, vis = _case(slice(0, 3), slice(0, 3))                    # at the start: nothing before
    assert (VI.bridge_blur(vis, st)[0, 0:3, 9] == VIS_HIDDEN).all()
