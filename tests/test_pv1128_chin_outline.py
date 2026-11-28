"""(11/28, the 1128 check of the delivered files: c084 f11..25, c030 f10..11 / f45) 자갈턱 넓게's levelling (_chin_level) worked on the chin's outline
itself: the shade under the jaw was lifted and the lit rim above it taken down by 15 to 30 levels, and with the landmarks moving from frame to frame the
outline flickered; a fingertip at the chin was levelled too. The levelling now fades out inside the outline (CHIN_LEVEL_EDGE_IN), its rim does not
reach past it (CHIN_LEVEL_RIM[1] 0) and is made without the hand cover (CHIN_RIM_MASK); the pebbles inside the chin are levelled as before."""
import cv2
import numpy as np

from baseline_util import face68


def _scene(finger=False):
    """A face 400 px wide on a dark ground (the shade under the jaw), lit skin inside the outline, pebbles on the chin; with a bright finger lying
    across the jaw's outline under the chin."""
    H = W = 560
    wf = 400.0
    lm = face68(280.0, 60.0, wf).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hull = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(lm[0:27, :2]).astype(np.int32)), 255)
    inside = cv2.GaussianBlur(hull.astype(np.float32) / 255.0, (0, 0), 2.0)
    Y = 0.18 + 0.42 * inside                      # the outline: a step from the lit chin to the shade under it
    peb = np.zeros((H, W), np.float32)
    cx8, cy8 = float(lm[8, 0]), float(lm[8, 1])
    centres = [(cx8 + dx, cy8 - 60 + dy) for dx in (-40, -20, 0, 20, 40) for dy in (-12, 6)]
    for i, (cx, cy) in enumerate(centres):
        peb += (0.04 if i % 2 else -0.04) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5.0 ** 2))
    Y = Y + peb * inside
    cover = np.zeros((H, W), np.float32)
    if finger:
        f = np.zeros((H, W), np.uint8)
        cv2.line(f, (int(cx8 - 150), int(cy8 + 40)), (int(cx8 - 30), int(cy8 - 18)), 255, 26)
        cover = f.astype(np.float32) / 255.0
        Y = np.where(f > 0, 0.85, Y).astype(np.float32)
        Y = cv2.GaussianBlur(Y, (0, 0), 1.0)
    skin = inside * (1.0 - cover)
    return dict(Y=Y.astype(np.float32), lm=lm, wf=wf, centres=centres, skin=skin.astype(np.float32), cover=cover, hull=hull, peb=peb)


def _amp(P, centres):
    base = cv2.GaussianBlur(P, (0, 0), 20.0)
    return float(np.mean([abs(P[int(cy), int(cx)] - base[int(cy), int(cx)]) for cx, cy in centres]))


def test_the_outline_under_the_chin_is_left_and_the_pebbles_inside_are_levelled():
    from contour.retouch import engine as RE
    sc = _scene()
    out = RE._chin_level(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0)
    d = np.abs(out - sc["Y"])
    # a band of 6 px either side of the outline round the chin's lowest part
    edge = cv2.morphologyEx(sc["hull"], cv2.MORPH_GRADIENT, np.ones((13, 13), np.uint8)) > 0
    y8 = int(sc["lm"][8, 1])
    chin_part = np.zeros_like(edge)
    chin_part[y8 - 60:y8 + 20, int(sc["lm"][8, 0]) - 110:int(sc["lm"][8, 0]) + 110] = True
    assert float(d[edge & chin_part].max()) < 0.012, "the chin's outline (the step into the shade under it) is not worked on"
    before = _amp(sc["Y"], sc["centres"])
    after = _amp(out, sc["centres"])
    assert after < 0.6 * before, f"the pebbles inside the chin are still levelled ({before:.4f} -> {after:.4f})"


def test_a_fingertip_across_the_jaw_is_not_levelled():
    from contour.retouch import engine as RE
    sc = _scene(finger=True)
    rim = RE._chin_rim_mask(sc["Y"].shape, sc["lm"], sc["cover"])
    assert rim is not None
    out = RE._chin_level(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0, rim=rim)
    on_finger = sc["cover"] > 0.5
    assert float(np.abs(out - sc["Y"])[on_finger].max()) < 0.01, "the finger is no chin"
    # the rim made from the landmarks alone (no masks: as an old dump) would have reached it
    rim0 = RE._chin_rim(sc["Y"].shape, sc["lm"], sc["wf"])
    assert float((rim0 * on_finger).max()) > 0.5


def test_no_rim_where_the_cover_took_it_all():
    from contour.retouch import engine as RE
    sc = _scene(finger=True)
    a = RE._chin_level(sc["Y"], sc["lm"], sc["skin"], sc["wf"], 1.0, rim=0.0)
    assert np.isfinite(a).all()
    assert float(np.abs(a - sc["Y"])[sc["cover"] > 0.5].max()) < 0.01
