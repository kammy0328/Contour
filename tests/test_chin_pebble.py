"""(1.27, LNGSHOT BEEP BEEP 11/27, the user: "턱주름 ... 안된 거 확인되면 알아서 해결"; c046, c034, c084: the pebbles stayed with 턱주름 at 100) the slider 자갈턱 넓게
("chin_pebble", a person's own): the chin's stages work on the whole chin (the keep-outs under the lip and at the outline small, no edge gate, what is taken
from a point bounded) for the person it is switched on for. Off (0, the default) the stages are those of 1.26.2 to the last number; a face turned away keeps
them whatever the slider says."""
import cv2
import numpy as np

from baseline_util import face68


def _scene():
    """A face 400 px wide with a pebbled chin whose pebbles reach up under the lower lip (an open mouth) and lie in a hard light (steep edges)."""
    H = W = 520
    wf = 400.0
    lm = face68(260.0, 60.0, wf).astype(np.float32)
    top = 60.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Y = np.full((H, W), 0.55, np.float32)
    peb = np.zeros((H, W), np.float32)
    centres = [(260 + dx, top + 282 + dy) for dx in (-40, -20, 0, 20, 40) for dy in (-14, 6, 22)]
    for i, (cx, cy) in enumerate(centres):
        peb += (0.04 if i % 2 else -0.04) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5.0 ** 2))
    return dict(Y=(Y + peb).astype(np.float32), peb=peb, lm=lm, wf=wf, top=top, centres=centres, skin=np.ones((H, W), np.float32))


def _amp(P, centres):
    base = cv2.GaussianBlur(P, (0, 0), 20.0)
    return float(np.mean([abs(P[int(cy), int(cx)] - base[int(cy), int(cx)]) for cx, cy in centres]))


def _both(RE, sc, share):
    out = RE._final_chin(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0, share)
    return RE._chin_dimple(out, sc["lm"], sc["skin"], sc["wf"], 1.0, share)


def test_the_slider_is_there_and_off_by_default():
    from contour.core import params as P
    from contour.retouch import engine as RE
    d = next(x for x in P.PARAMS if x.key == "chin_pebble")
    assert d.default == 0.0 and d.group == "skin" and (d.lo, d.hi) == (0, 100)
    assert RE.CHIN_PEBBLE == 0.0, "no face gets the pebbled way unless its slider says so"


def test_the_share_is_the_persons_slider_and_none_on_a_face_turned_away():
    from contour.retouch import engine as RE
    lm = face68(260.0, 60.0, 400.0).astype(np.float32)
    assert RE._pebble_share(lm, {}) == 0.0 and RE._pebble_share(lm, None) == 0.0
    assert RE._pebble_share(lm, {"chin_pebble": 1.0}) == 1.0
    assert abs(RE._pebble_share(lm, {"chin_pebble": 0.4}) - 0.4) < 1e-6
    assert RE._pebble_share(None, {"chin_pebble": 1.0}) == 0.0
    turned = lm.copy()
    turned[30] = lm[2] + 0.05 * (lm[14] - lm[2])                        # the nose tip close to one side of the outline: a profile
    assert RE.face_turn(turned) > RE.CHIN_PEBBLE_YAW[1]
    assert RE._pebble_share(turned, {"chin_pebble": 1.0}) == 0.0


def test_off_the_stages_are_those_of_before():
    """share 0: the zone with its keep-outs and the edge gate, as 1.26.2 (the same code path: the numbers of the engine baselines do not move)."""
    from contour.retouch import engine as RE
    sc = _scene()
    z0 = RE.chin_zone(sc["Y"].shape, sc["lm"], sc["wf"])
    z1 = RE.chin_zone(sc["Y"].shape, sc["lm"], sc["wf"], pebble=True)
    assert float(z1.sum()) > 1.3 * float(z0.sum()), "the pebbled way's zone keeps less clear under the lip and at the outline"
    a = RE._final_chin(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0)
    b = RE._final_chin(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0, 0.0)
    assert np.array_equal(a, b)
    # under the lip (inside CHIN_KEEP_LIP) nothing is touched without the slider
    top = sc["top"]
    lip_y = int(float(sc["lm"][57, 1]))
    band = slice(lip_y + 4, lip_y + int(0.06 * sc["wf"]))
    assert float(np.abs(a - sc["Y"])[band, 200:320].max()) < 2e-3


def test_with_the_slider_the_pebbles_go_up_to_the_lip_and_in_hard_light():
    from contour.retouch import engine as RE
    sc = _scene()
    before = _amp(sc["Y"], sc["centres"])
    off = _amp(_both(RE, sc, 0.0), sc["centres"])
    on = _amp(_both(RE, sc, 1.0), sc["centres"])
    assert off > 0.65 * before, f"without the slider most of these pebbles stay ({before:.4f} -> {off:.4f}): they lie under the lip and have steep edges"
    assert on < 0.5 * before and on < 0.7 * off, f"with it they go ({before:.4f} -> {on:.4f}; without {off:.4f})"
    half = _amp(_both(RE, sc, 0.5), sc["centres"])
    assert on < half < off, (on, half, off)


def test_what_the_last_pass_takes_is_bounded():
    from contour.retouch import engine as RE
    sc = _scene()
    Y = sc["Y"].copy()
    top = sc["top"]
    yy, xx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]].astype(np.float32)
    Y -= (0.30 * np.exp(-((xx - 260) ** 2 + (yy - (top + 290)) ** 2) / (2 * 6.0 ** 2))).astype(np.float32)        # a thing far deeper than a pebble
    out = RE._final_chin(Y, sc["lm"], sc["skin"], sc["wf"], 1.0, 1.0)
    assert float(np.abs(out - Y).max()) <= RE.CHIN_PEBBLE_CAP + 1e-4
    was = RE.CHIN_PEBBLE_CAP
    try:
        RE.CHIN_PEBBLE_CAP = None
        free = RE._final_chin(Y, sc["lm"], sc["skin"], sc["wf"], 1.0, 1.0)
    finally:
        RE.CHIN_PEBBLE_CAP = was
    assert float(np.abs(free - Y).max()) > 2.0 * RE.CHIN_PEBBLE_CAP, "without the bound the deep thing would be filled"
