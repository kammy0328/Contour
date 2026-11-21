"""1.23 body skin (instruction 2026-11-20 §2): 몸 피부 보정 and 몸 잡티 제거 work with no face and one shoulder in the
picture, only on the person's own bare skin, at the original's own pixels, steady through time. The scene is drawn
(tests/body_util.py): every pixel's kind is known, so each test says exactly what may change and what may not."""
import os

import cv2
import numpy as np
import pytest

import body_util as B


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """The analysed scene, made once (its files are only read by the tests)."""
    d = tmp_path_factory.mktemp("bodyscene")
    mp = pytest.MonkeyPatch()
    (d / "home").mkdir()
    mp.setenv("HOME", str(d / "home"))
    mp.setenv("USERPROFILE", str(d / "home"))
    import contour.core.settings as S
    S._settings = None
    try:
        prj, cut = B.make_project(d)
    finally:
        mp.undo()
        S._settings = None
    return prj, cut


@pytest.fixture
def body(scene, monkeypatch):
    """The scene opened, the parser stood in for by the scene's truth, every body value off."""
    from contour.core import frames as FR
    prj, cut = scene
    FR.pool().clear()
    calls = B.truth_parser(monkeypatch)
    for p in cut.persons:
        pp = B.body_on(cut, p.id)
        pp.body_marks = []
        pp.amounts["skin"].base = 0.0
    fr, tr = B.open_cut(prj, cut)
    yield cut, tr, calls
    fr.close()


def _kinds(tr, t):
    return B.kind(t, (tr.proxy.w, tr.proxy.h))


def _changed(a, b):
    return np.abs(b.astype(np.int16) - a.astype(np.int16)).max(-1)


def _inside(sel, r=3):
    return cv2.erode(sel.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0


def _fine(rgb, sel, lo=0.9, hi=2.2):
    """How much skin texture the picture has inside sel: the standard deviation of the detail between two widths
    (finer than lo is the picture's grain, which every strength keeps)."""
    y = B.luma(rgb)
    return float((cv2.GaussianBlur(y, (0, 0), lo) - cv2.GaussianBlur(y, (0, 0), hi))[sel].std())


def _plain_skin(t, r=10):
    """The person's bare skin of the original's frame t well inside its edge and away from every mark on it."""
    k = B.kind(t)
    sel = _inside(k == B.SKIN_K, r)
    yy, xx = np.mgrid[0:B.H, 0:B.W]
    L = lambda a, b, uv: B.local(a, b, uv, t)
    pts = [L(B.EL, B.WR, u) for u in B.PIMPLES + B.DARKS + [B.MOLE, B.BIG] + B.FRECKLES] + [L(B.HIP, B.KN, B.THIGH_SPOT)]
    for q in pts:
        sel &= np.hypot(xx - q[0], yy - q[1]) > 22
    return sel & ~(cv2.dilate(B.ink_mask(t).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)


def _export(cut, tr, t):
    """Frame t as an export makes it: the original's planes (4:4:4 here) through apply_planes."""
    from contour.retouch import engine as RE
    src = B.frame(t)
    f = src.astype(np.float32) / 255.0
    Y, Cb, Cr = RE.rgb_to_ycc(f)
    rf = RE.prepare(cut, tr, t, {})
    Y2, Cb2, Cr2 = RE.process_ycc(Y, Cb, Cr, rf, 1.0)
    out = np.clip(RE.ycc_to_rgb(Y2, Cb2, Cr2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return src, out



# ------------------------------------------------------------------------------------------------- where it goes
def test_off_by_default_changes_nothing(body):
    cut, tr, calls = body
    from contour.retouch import engine as RE
    assert RE.prepare(cut, tr, 2, {}) is None, "no body value on: nothing is prepared at all"
    assert calls == [], "and the body parser is not run"


def test_close_up_without_face_or_second_shoulder_is_retouched(body):
    """No head, one shoulder: the arm and the thigh are found from their own joints and the person's matte; the
    regions of before 1.23 give nothing here (they hang on both shoulders)."""
    cut, tr, _ = body
    from contour.retouch import engine as RE
    st = tr.store
    assert not st.has_face and not np.isfinite(st.final[0, 2, 6, :2]).all()
    assert RE.body_regions({"arm", "leg", "shoulder"}, st.final[0, 2], None, (tr.proxy.h, tr.proxy.w), 0.5, 0.5) is None
    B.body_on(cut, skin=90)
    a, b, rf = B.preview(cut, tr, 2)
    k = _kinds(tr, 2)
    m = rf.body_area(rf.jobs[0])
    skin = _inside(k == B.SKIN_K, 4)
    assert float(m[skin].mean()) > 0.9, "the bare skin is the area"
    assert rf.jobs[0].face is None and rf.jobs[0].body["ref"] > 60
    ch = _changed(a, b)
    assert int((ch[skin] > 0).sum()) > 100, "and it is smoothed"
    src, out = _export(cut, tr, 2)
    plain = _plain_skin(2)
    # a body of this size in a picture of this size leaves the smoothing a narrow band (between the picture's grain
    # and a few pores of this body); the share taken in it is the face's at the same size
    assert _fine(out, plain) < 0.95 * _fine(src, plain), "on the original its texture is thinner"


def test_only_the_persons_own_bare_skin_changes(body):
    """A top of nearly the skin's colour, a strap, hair, the background, the other person's arm across and what
    lies on the border between the two: not one pixel of them changes."""
    cut, tr, _ = body
    B.body_on(cut, skin=100, blemish=100, size=B.SPOT_SIZE)
    for t in (1, 4, 6):
        a, b, rf = B.preview(cut, tr, t)
        k = _kinds(tr, t)
        ch = _changed(a, b)
        for name, kk in (("the top", B.CLOTH), ("the strap", B.STRAP), ("the hair", B.HAIR), ("the background", B.BG),
                         ("the other person", B.OTHER_K), ("the cover", B.COVER)):
            sel = k == kk
            if sel.any():
                assert int((ch[sel] > 0).sum()) == 0, f"frame {t}: {name} changed ({int(ch[sel].max())} code values)"
        # at the border to the other person the owner is not sure: left as it is
        near = (cv2.dilate((k == B.OTHER_K).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (k == B.SKIN_K)
        assert int((ch[near] > 0).sum()) == 0, f"frame {t}: skin right at the other person's arm changed"
        assert int((ch > 0).sum()) > 100, f"frame {t}: nothing changed at all"


def test_the_other_person_is_theirs_when_they_are_switched_on(body):
    """Each person's body skin goes by their own switch: with only the other person on, only their arm changes."""
    cut, tr, _ = body
    cut.person(1).role = "target"
    try:
        B.body_on(cut, 1, skin=100)
        a, b, rf = B.preview(cut, tr, 2)
        k = _kinds(tr, 2)
        ch = _changed(a, b)
        assert int((ch[_inside(k == B.OTHER_K, 4)] > 0).sum()) > 100
        assert int((ch[k == B.SKIN_K] > 0).sum()) == 0
    finally:
        cut.person(1).role = "protect"
        B.body_on(cut, 1)


def test_regions_arm_or_leg(body):
    """부위별: the arm alone leaves the thigh, the leg alone leaves the arm, 전신 takes both."""
    cut, tr, _ = body
    k = _kinds(tr, 2)
    yy = np.mgrid[0:tr.proxy.h, 0:tr.proxy.w][0]
    arm = _inside((k == B.SKIN_K) & (yy < 0.56 * tr.proxy.h), 5)
    leg = _inside((k == B.SKIN_K) & (yy > 0.80 * tr.proxy.h), 5)
    got = {}
    for scope in ("arm", "leg", "all"):
        B.body_on(cut, skin=100, scope=scope)
        a, b, _rf = B.preview(cut, tr, 2)
        ch = _changed(a, b)
        got[scope] = (int((ch[arm] > 0).sum()), int((ch[leg] > 0).sum()))
    assert got["arm"][0] > 100 and got["arm"][1] == 0, got
    assert got["leg"][1] > 100 and got["leg"][0] == 0, got
    assert got["all"][0] > 100 and got["all"][1] > 100, got


def test_without_a_parser_and_without_a_face_nothing_is_touched(body, monkeypatch):
    """What is not surely skin stays: with no parser there is nothing to learn the skin's colour from in a close-up
    without a face, and a top of the skin's colour could not be told from skin. Nothing is retouched."""
    cut, tr, _ = body
    from contour.retouch import bodyskin as BS
    monkeypatch.setattr(BS, "_parse_cached", lambda *a, **k: None)
    B.body_on(cut, skin=100, blemish=100)
    a, b, rf = B.preview(cut, tr, 2)
    assert int(_changed(a, b).max()) == 0


def test_parser_missing_a_patch_of_skin_is_filled_by_colour_but_cloth_called_cloth_stays_out(body, monkeypatch):
    """The parser says what kind of thing a place is, the colour decides the pixel: a hole in the parser's skin (a
    knee it takes for background) is skin by its colour and by the skin round it; the top stays out although its
    colour is the skin's."""
    cut, tr, _ = body

    def wrong(t, k):
        k = k.copy()
        c = B.local(B.EL, B.WR, (0.5, 0.0), t) * 0.5
        yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]]
        hole = (np.hypot(xx - c[0], yy - c[1]) < 7) & (k == B.SKIN_K)
        k[hole] = B.BG
        return k
    B.truth_parser(monkeypatch, wrong)
    B.body_on(cut, skin=100)
    a, b, rf = B.preview(cut, tr, 2)
    k = _kinds(tr, 2)
    m = rf.body_area(rf.jobs[0])
    c = B.local(B.EL, B.WR, (0.5, 0.0), 2) * 0.5
    assert float(m[int(c[1]) - 2:int(c[1]) + 3, int(c[0]) - 2:int(c[0]) + 3].mean()) > 0.8, "the hole is skin"
    # (a pixel on the border of skin and top is partly each: a pixel inside the top is not skin at all)
    assert float(m[_inside(k == B.CLOTH, 1)].max()) < 0.05 and int((_changed(a, b)[k == B.CLOTH] > 0).sum()) == 0


# ------------------------------------------------------------------------------------------------- what the smoothing keeps
def test_tone_shading_and_outline_are_kept(body):
    """The smoothing thins the skin's fine texture; the skin's colour, the limb's own light and shade and the
    body's outline are as they were."""
    cut, tr, _ = body
    B.body_on(cut, skin=100, detail=0)
    a, b, rf = B.preview(cut, tr, 2)
    k = _kinds(tr, 2)
    skin = _inside(k == B.SKIN_K, 5)
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    assert np.abs(fa[skin].mean(0) - fb[skin].mean(0)).max() < 0.5, "the skin's colour"
    wa = cv2.GaussianBlur(B.luma(a), (0, 0), 6.0)
    wb = cv2.GaussianBlur(B.luma(b), (0, 0), 6.0)
    assert float(np.abs(wa - wb)[skin].max()) < 0.006, "light and shade (what is wider than the texture)"
    # the outline: the step from skin to what is beside it, measured across the skin's own edge
    edge = (cv2.dilate((k == B.SKIN_K).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~_inside(k == B.SKIN_K, 2)
    ga = np.hypot(cv2.Sobel(B.luma(a), cv2.CV_32F, 1, 0), cv2.Sobel(B.luma(a), cv2.CV_32F, 0, 1))
    gb = np.hypot(cv2.Sobel(B.luma(b), cv2.CV_32F, 1, 0), cv2.Sobel(B.luma(b), cv2.CV_32F, 0, 1))
    assert float(np.abs(ga - gb)[edge].max()) < 0.02, "the outline is not softened"
    assert int((_changed(a, b)[edge & ~skin] > 1).sum()) == 0, "nothing more than a rounding at the skin's edge"


def test_texture_kept_slider(body):
    """결 보존: at 100 more of the fine texture is left than at 0, and neither leaves more than the original had."""
    cut, tr, _ = body
    plain = _plain_skin(2)
    left = {}
    for keep in (0, 100):
        B.body_on(cut, skin=100, detail=keep)
        src, out = _export(cut, tr, 2)
        left[keep] = _fine(out, plain, 0.5, 1.2)
    base = _fine(src, plain, 0.5, 1.2)
    assert left[0] < left[100] <= base * 1.02, (left, base)
    assert left[100] - left[0] > 0.3 * (base - left[0]), (left, base)


def test_texture_kept_slider_on_a_body_large_in_the_picture():
    """The same on a close-up (a body whose face would be 480 px wide), where pores and blotches are pixels apart in
    size: 결 보존 0 thins both, 100 leaves the pores as they were and still evens the blotches out. Strength 0
    changes nothing."""
    from types import SimpleNamespace as NS
    from contour.retouch import engine as RE
    rng = np.random.default_rng(5)
    H = W = 640

    def grain(sig):
        n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sig)
        return n / float(n.std())
    Y = (0.60 + 0.008 * grain(0.9) + 0.008 * grain(5.0)).astype(np.float32)       # pores, blotches
    Cb = np.full((H, W), -0.05, np.float32)
    Cr = np.full((H, W), 0.06, np.float32)
    m = np.zeros((H, W), np.float32)
    m[30:-30, 30:-30] = 1.0
    sel = np.zeros((H, W), bool)
    sel[90:-90, 90:-90] = True

    def band(P, lo, hi):
        return float((cv2.GaussianBlur(P, (0, 0), lo) - cv2.GaussianBlur(P, (0, 0), hi))[sel].std())
    left = {}
    for keep in (0.0, 1.0):
        job = NS(v={"body_skin": 1.0}, body={"ref": 480.0}, body_keep=keep)
        Y1 = RE._body_smooth(Y, Cb, Cr, m, job, 1.0, 1.0)[0]
        left[keep] = (band(Y1, 0.6, 1.6) / band(Y, 0.6, 1.6), band(Y1, 3.5, 8.0) / band(Y, 3.5, 8.0))
        assert abs(float(Y1[sel].mean()) - float(Y[sel].mean())) < 2e-4, "the skin's level"
    assert left[0.0][0] < 0.82 and left[0.0][1] < 0.75, left
    assert left[1.0][0] > 0.92 and left[1.0][1] < 0.80, left
    job = NS(v={"body_skin": 0.0}, body={"ref": 480.0}, body_keep=0.6)
    assert RE._body_smooth(Y, Cb, Cr, m, job, 1.0, 1.0)[0] is Y


def test_things_on_the_skin_and_ink_are_not_smoothed(body):
    """A strap across the arm, hair over the shoulder and a drawing on the thigh are as sharp as they were."""
    cut, tr, _ = body
    B.body_on(cut, skin=100, detail=0)
    a, b, rf = B.preview(cut, tr, 2)
    ink = cv2.resize(B.ink_mask(2).astype(np.uint8), (tr.proxy.w, tr.proxy.h), interpolation=cv2.INTER_NEAREST) > 0
    ch = _changed(a, b)
    assert int(ch[ink].max()) <= 1, f"the ink changed by {int(ch[ink].max())}"
    k = _kinds(tr, 2)
    for kk in (B.STRAP, B.HAIR):
        near = cv2.dilate((k == kk).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        assert int(ch[near].max()) <= 1


# ------------------------------------------------------------------------------------------------- blemishes
def _mark(t, limb, uv, tr):
    a, b = (B.EL, B.WR) if limb == "fore" else (B.HIP, B.KN)
    return B.local(a, b, uv, t) * tr.proxy.sx


def test_small_spots_go_and_marks_of_the_person_stay(body):
    """몸 잡티 제거: the two pimples and the faint dark spot on the forearm and the pimple on the thigh are filled;
    the mole, the freckles, the patch larger than the largest spot and the ink stay as they are."""
    cut, tr, _ = body
    B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    a, b, rf = B.preview(cut, tr, 2)
    for uv in B.PIMPLES:
        c = _mark(2, "fore", uv, tr)
        assert B.contrast(b, c, 1.5, 5, red=True) < 0.45 * B.contrast(a, c, 1.5, 5, red=True), f"pimple {uv} stayed"
    c = _mark(2, "fore", B.DARKS[0], tr)
    assert B.contrast(b, c, 1.5, 5) < 0.5 * B.contrast(a, c, 1.5, 5), "the dark spot stayed"
    c = _mark(2, "thigh", B.THIGH_SPOT, tr)
    assert B.contrast(b, c, 1.5, 5, red=True) < 0.45 * B.contrast(a, c, 1.5, 5, red=True), "the thigh's pimple stayed"
    ch = _changed(a, b)

    def around(c, r):
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        return np.hypot(xx - c[0], yy - c[1]) <= r
    assert int(ch[around(_mark(2, "fore", B.MOLE, tr), 4)].max()) == 0, "the mole was touched"
    for uv in B.FRECKLES:
        assert int(ch[around(_mark(2, "fore", uv, tr), 2)].max()) == 0, f"freckle {uv} was touched"
    assert int(ch[around(_mark(2, "fore", B.BIG, tr), 3)].max()) <= 1, "the large patch was touched"
    ink = cv2.resize(B.ink_mask(2).astype(np.uint8), (tr.proxy.w, tr.proxy.h), interpolation=cv2.INTER_NEAREST) > 0
    assert int(ch[ink].max()) == 0, "the ink was touched"
    k = _kinds(tr, 2)
    assert int((ch[k != B.SKIN_K] > 0).sum()) == 0
    assert int((ch > 0).sum()) < 400, "only the spots change, not the skin round them"


def test_largest_spot_size_slider(body):
    """최대 잡티 크기: at a small size the spots of the scene (about 5 source px) are too large and stay; they go at
    the size the other tests use."""
    cut, tr, _ = body
    c = _mark(2, "fore", B.PIMPLES[0], tr)
    left = {}
    for size in (0, B.SPOT_SIZE):
        B.body_on(cut, blemish=100, size=size)
        a, b, _rf = B.preview(cut, tr, 2)
        left[size] = B.contrast(b, c, 1.5, 5, red=True) / B.contrast(a, c, 1.5, 5, red=True)
    assert left[0] > 0.9 and left[B.SPOT_SIZE] < 0.45, left


def test_spots_are_found_on_the_originals_pixels_in_tiles(body, monkeypatch):
    """The spots are looked for on the original where it has more pixels than the analysis picture (2.5 px there:
    hardly a spot), in overlapping pieces; small pieces give the same spots as one piece, each once."""
    cut, tr, _ = body
    from contour.retouch import bodyspots as BP
    from contour.retouch import engine as RE
    B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    raw = RE._prepare_raw(cut, tr, 2, {})
    d1 = raw.jobs[0].bdet
    assert d1["src"] and abs(d1["scale"] - 1.0) < 1e-6 and d1["tiles"] == 1
    monkeypatch.setattr(BP, "TILE", 256)
    monkeypatch.setattr(BP, "TILE_LAP", 64)
    d2 = RE._prepare_raw(cut, tr, 2, {}).jobs[0].bdet
    assert d2["tiles"] >= 4
    t1 = d1["cands"][BP.taken(d1)]
    t2 = d2["cands"][BP.taken(d2)]
    assert len(t1) == 4 and len(t2) == 4, (len(t1), len(t2))
    for c in t1:
        d = np.hypot(t2["x"] - c["x"], t2["y"] - c["y"])
        assert d.min() < 1.5, "a spot of the one piece is missing from the tiles"
    # on the analysis picture alone the small spots are not all found: the original's pixels are what finds them
    monkeypatch.setattr(RE, "DET_SRC", False)
    d3 = RE._prepare_raw(cut, tr, 2, {}).jobs[0].bdet
    assert not d3["src"] and int(BP.taken(d3).sum()) < 4


def test_a_spot_that_flickers_is_removed_on_every_frame(body):
    """The thigh's pimple is half as strong on one frame. Alone that frame would leave more of it; with its
    neighbours it is removed there as on the frames round it (the retouch does not blink)."""
    cut, tr, _ = body
    from contour.retouch import engine as RE
    B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    t = B.FLICKER_FRAME
    cache = {}
    left = []
    for s in (t - 1, t, t + 1):
        a, b, rf = B.preview(cut, tr, s, cache)
        c = _mark(s, "thigh", B.THIGH_SPOT, tr)
        left.append(abs(B.contrast(b, c, 1.5, 5, red=True)))
    assert max(left) < 0.012, f"what is left of the spot on the three frames: {left}"
    assert max(left) - min(left) < 0.006, f"the retouch changes from frame to frame: {left}"


def _det(spots, scale=1.0):
    """A frame's spots as the caches keep them: [(x, y, score, hold)] in source px, a 5 px patch each."""
    from contour.retouch import bodyspots as BP
    c = np.zeros(len(spots), BP.CAND)
    ps = []
    for i, (x, y, sc, hold) in enumerate(spots):
        c[i] = (x, y, 5.0, sc, 0.0, BP.KIND_DARK, hold, 0.5, x - 2, y - 2, 0, 5, 5)
        ps.append(np.full((5, 5), int(sc * 255), np.uint8))
    return BP.pack(c, ps, [np.zeros((5, 5), np.uint8) for _ in spots], scale)


def _still(same: float, size=64):
    """The motion of a frame onto a neighbour where nothing moved, and how far the neighbour has a say (0..1)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return dict(box=(0, 0, size, size), map=np.stack([xx, yy], -1), same=np.full((size, size), same, np.float32))


def test_a_spot_seen_on_one_frame_only_is_not_removed():
    """A spot is removed where a frame beside it shows the same spot at the same place. One that stands on its own
    frame's word alone is held: where the neighbours can say nothing (fast motion, blur: on real footage every
    such removal was gone again on the next frame, j2790) as where they say there is nothing. 1.23's first build
    removed it on that one frame, and the retouch blinked."""
    from contour.retouch import bodyspots as BP
    spot = (30.0, 30.0, 0.8, BP.HOLD_NONE)
    own = _det([spot])
    none = BP.empty(1.0)
    assert BP.taken(own).tolist() == [True]
    # no frame beside it at all (a cut of one frame): its own word stands
    assert BP.taken(BP.fuse(own, [], 100.0, 1.0, 1.0)).tolist() == [True]
    assert BP.HOLD_TEXT[BP.HOLD_ALONE] == "한 프레임에만 보임"
    # neighbours that see the place and have nothing there: not removed either
    out = BP.fuse(own, [(1.0, none, _still(1.0)), (1.0, none, _still(1.0))], 100.0, 1.0, 1.0)
    assert BP.taken(out).tolist() == [False]
    # one neighbour shows the same spot there: removed, on this frame and (seen from there) on that one
    there = _det([(30.6, 29.7, 0.7, BP.HOLD_NONE)])
    out = BP.fuse(own, [(1.0, there, _still(1.0)), (1.0, none, _still(0.0))], 100.0, 1.0, 1.0)
    assert BP.taken(out).tolist() == [True]
    back = BP.fuse(there, [(1.0, own, _still(1.0))], 100.0, 1.0, 1.0)
    assert BP.taken(back).tolist() == [True], "and so it does not blink between the two"
    # the neighbour's mark is one it keeps (a mole there): kept here too, as before
    mole = _det([(30.0, 30.0, 0.7, BP.HOLD_MOLE)])
    out = BP.fuse(own, [(1.0, mole, _still(1.0))], 100.0, 1.0, 1.0)
    assert BP.taken(out).tolist() == [False] and int(out["cands"]["hold"][0]) == BP.HOLD_MOLE
    # the picture's motion has nothing to say (fast movement), the joints carry the place to the neighbour:
    seen = lambda dx, dy, vis: (lambda x, y: (x + dx, y + dy, vis))
    #   the neighbour shows the same spot at that place of the body (a few px off: the joints are coarse): it stands
    moved = _det([(133.0, 28.0, 0.7, BP.HOLD_NONE)])
    out = BP.fuse(own, [(1.0, moved, _still(0.0), seen(100.0, 0.0, 1.0))], 100.0, 1.0, 1.0)
    assert BP.taken(out).tolist() == [True]
    #   the place is to be seen there and no such spot is: held
    out = BP.fuse(own, [(1.0, none, _still(0.0), seen(100.0, 0.0, 1.0))], 100.0, 1.0, 1.0)
    assert int(out["cands"]["hold"][0]) == BP.HOLD_ALONE
    far = _det([(160.0, 30.0, 0.7, BP.HOLD_NONE)])
    out = BP.fuse(own, [(1.0, far, _still(0.0), seen(100.0, 0.0, 1.0))], 100.0, 1.0, 1.0)
    assert int(out["cands"]["hold"][0]) == BP.HOLD_ALONE, "another spot far from that place does not confirm it"
    #   the place is covered or out of the picture on every frame beside this one: nobody can say, its own word stands
    out = BP.fuse(own, [(1.0, none, _still(0.0), seen(100.0, 0.0, 0.0)), (1.0, none, _still(0.0), lambda x, y: None)],
                  100.0, 1.0, 1.0)
    assert BP.taken(out).tolist() == [True]
    #   what the neighbour keeps at that place (ink) is kept here
    ink = _det([(131.0, 30.0, 0.7, BP.HOLD_THING)])
    out = BP.fuse(own, [(1.0, ink, _still(0.0), seen(100.0, 0.0, 1.0))], 100.0, 1.0, 1.0)
    assert int(out["cands"]["hold"][0]) == BP.HOLD_THING
    # the operator's own mark removes a spot that was held for standing alone
    alone = BP.fuse(own, [(1.0, none, _still(1.0))], 100.0, 1.0, 1.0)
    assert int(alone["cands"]["hold"][0]) == BP.HOLD_ALONE
    marked = BP.apply_marks(alone, [("remove", np.array([[30.0, 30.0]]), 4.0)], 1.0)
    assert BP.taken(marked).any(), "a hand mark is the person's word: the marked place is filled"


def test_nothing_is_filled_on_a_cover_and_nothing_is_carried_over_it(body):
    """A prop passes over the forearm on two frames. On those frames the spots under it are not filled (the cover's
    pixels do not change), and on the frames beside them the spots are filled as ever."""
    cut, tr, _ = body
    B.body_on(cut, blemish=100, skin=80, size=B.SPOT_SIZE)
    cache = {}
    for t in B.COVER_FRAMES:
        a, b, rf = B.preview(cut, tr, t, cache)
        k = _kinds(tr, t)
        cov = cv2.dilate((k == B.COVER).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        assert int(_changed(a, b)[cov].max()) == 0, f"frame {t}: the cover changed"
    a, b, rf = B.preview(cut, tr, B.COVER_FRAMES[0] - 1, cache)
    c = _mark(B.COVER_FRAMES[0] - 1, "fore", B.DARKS[0], tr)
    assert B.contrast(b, c, 1.5, 5) < 0.5 * B.contrast(a, c, 1.5, 5), "the frame before the cover lost its retouch"


def test_what_this_frame_rules_out_is_not_brought_back_by_the_frames_beside_it(body, monkeypatch):
    """§2 프레임 안정화: "매 프레임의 현재 피부 마스크로 제한". On one frame the parser calls a band across the forearm
    clothes (a sleeve of the skin's own colour slid over it). The frames beside it have skin there, and the area is
    steadied through time; this frame's area does not take the band from them and no pixel of it changes. (The
    steadying added what the neighbours had beside this frame's own area, whatever this frame said of it: on real
    4K footage hair lying over an arm on one frame was retouched by the frames around it.)"""
    cut, tr, _ = body
    T0 = 2

    def band(t, shape):
        f = shape[1] / float(B.W)
        c = B.local(B.EL, B.WR, (0.5, 0.0), t) * f
        u = (np.asarray(B.WR, np.float64) - np.asarray(B.EL, np.float64))
        u = u / np.hypot(*u)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        return np.abs((xx - c[0]) * u[0] + (yy - c[1]) * u[1]) < 9.0 * f / 0.5

    def wrong(t, k):
        if t != T0:
            return k
        k = k.copy()
        k[band(t, k.shape) & (k == B.SKIN_K)] = B.CLOTH
        return k
    B.truth_parser(monkeypatch, wrong)
    B.body_on(cut, skin=100, blemish=100, size=B.SPOT_SIZE)
    cache = {}
    for t in (T0 - 1, T0 + 1):
        a, b, rf = B.preview(cut, tr, t, cache)
        k = _kinds(tr, t)
        core = _inside(band(t, k.shape) & (k == B.SKIN_K), 3)
        assert float(rf.body_area(rf.jobs[0])[core].min()) > 0.8, f"frame {t}: the band is skin on the frames beside it"
    a, b, rf = B.preview(cut, tr, T0, cache)
    k = _kinds(tr, T0)
    bd = band(T0, k.shape) & (k == B.SKIN_K)
    core = _inside(bd, 3)
    m = rf.body_area(rf.jobs[0])
    assert int(core.sum()) > 100
    assert float(m[core].max()) < 0.05, f"the area on what this frame rules out: {float(m[core].max()):.2f}"
    assert int(_changed(a, b)[core].max()) == 0
    # and right at the band's rim: the area there is this frame's own soft edge, nothing the neighbours added
    from contour.retouch import engine as RE, bodyskin as BS
    own = BS.unpack_mask(RE._prepare_raw(cut, tr, T0, {}).jobs[0].body["mask"])
    rim = bd & ~core
    grown = float((m - own)[rim].max())
    assert grown < 0.03, f"the frames beside it added {grown:.2f} to the area on the band's rim"
    # the skin beside the band is retouched as ever on this frame
    beside = _inside((k == B.SKIN_K) & ~cv2.dilate(band(T0, k.shape).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool), 4)
    assert float(m[beside].mean()) > 0.9


def test_light_change_keeps_the_area_and_the_retouch(body):
    """One frame is a fifth brighter (a flash). The skin's colour is learned on each frame, so the area and the
    spots removed are the same as on the frames beside it."""
    cut, tr, _ = body
    B.body_on(cut, blemish=100, skin=80, size=B.SPOT_SIZE)
    cache = {}
    area = {}
    for t in (B.BRIGHT_FRAME - 1, B.BRIGHT_FRAME):
        a, b, rf = B.preview(cut, tr, t, cache)
        k = _kinds(tr, t)
        m = rf.body_area(rf.jobs[0])
        area[t] = float(m[_inside(k == B.SKIN_K, 4)].mean())
        c = _mark(t, "fore", B.PIMPLES[0], tr)
        assert B.contrast(b, c, 1.5, 5, red=True) < 0.5 * B.contrast(a, c, 1.5, 5, red=True), f"frame {t}"
    assert min(area.values()) > 0.9 and abs(area[B.BRIGHT_FRAME] - area[B.BRIGHT_FRAME - 1]) < 0.03, area


def test_nothing_is_carried_across_a_cut(tmp_path, monkeypatch):
    """Where the picture jumps (a cut inside the clip) the frames beyond it have no say: a frame next to the jump
    is retouched from its own side only."""
    from contour.retouch import engine as RE
    B.truth_parser(monkeypatch)
    prj, cut = B.make_project(tmp_path, jump_at=3)
    fr, tr = B.open_cut(prj, cut)
    try:
        B.body_on(cut, blemish=100, skin=80, size=B.SPOT_SIZE)
        st = tr.store
        assert not st.connected(3, 4) and st.connected(4, 5)
        rf = RE.prepare(cut, tr, 4, {})
        j = rf.jobs[0]
        nb = RE._body_neighbours(cut, tr, st, rf, j, {}, both=False)
        assert sorted(s for _w, s, _j, _m in nb) == [5, 6], "only frames of this side of the jump"
        assert RE._body_neighbours(cut, tr, st, rf, j, {}, both=True) == [], "the mask: both sides or neither"
    finally:
        fr.close()


def test_owner_change_cuts_the_area_at_once(body, monkeypatch):
    """ID 교체: from one frame on the pixel owners give the person's forearm to the other person. The area on that
    frame is what the owners say now; the frames before do not carry the forearm back in."""
    cut, tr, _ = body
    from contour.retouch import engine as RE
    labels0 = tr.labels
    t_sw = 4
    yy = np.mgrid[0:B.MH, 0:B.MW][0]

    def labels(t):
        lab = np.array(labels0(t))
        if t >= t_sw:
            k = B.kind(t, (B.MW, B.MH))
            mid = B.local(B.EL, B.WR, (0.0, 0.0), t)[1] / 4.0
            lab[(k == B.SKIN_K) & (yy > mid - 2) & (yy < 0.5 * B.MH)] = 1          # (stored index of the other person)
        return lab
    monkeypatch.setattr(tr, "labels", labels)
    B.body_on(cut, skin=100)
    a, b, rf = B.preview(cut, tr, t_sw)
    k = _kinds(tr, t_sw)
    ya = np.mgrid[0:tr.proxy.h, 0:tr.proxy.w][0]
    fore = _inside((k == B.SKIN_K) & (ya > B.local(B.EL, B.WR, (0.15, 0.0), t_sw)[1] * 0.5) & (ya < 0.5 * tr.proxy.h), 3)
    assert fore.sum() > 300
    m = rf.body_area(rf.jobs[0])
    assert float(m[fore].max()) < 0.15, "the forearm the owners gave away is still in the area"
    assert int((_changed(a, b)[fore] > 0).sum()) == 0


# ------------------------------------------------------------------------------------------------- marks by hand
def test_hand_marks_remove_and_exclude_follow_the_body(body):
    """A 'remove' mark on the mole (which the detection keeps) takes it away; an 'exclude' mark on a pimple keeps
    it. Both are made on one frame and lie on the same place of the forearm on another."""
    cut, tr, _ = body
    from contour.retouch import bodyskin as BS
    pp = B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    t0, t1 = 1, 5 if 5 not in B.COVER_FRAMES else 7
    mole0 = B.local(B.EL, B.WR, B.MOLE, t0)
    pim0 = B.local(B.EL, B.WR, B.PIMPLES[1], t0)
    pp.body_marks = [BS.anchor_mark(tr.store, 0, t0, dict(id="a", kind="remove", t=t0, pts=[list(mole0)], r=5.0)),
                     BS.anchor_mark(tr.store, 0, t0, dict(id="b", kind="exclude", t=t0, pts=[list(pim0)], r=6.0))]
    assert pp.body_marks[0]["part"] == "forearm_l" and pp.body_marks[1]["part"] == "forearm_l"
    for t in (t0, 7):
        got = BS.marks_at(tr.store, 0, t, pp.body_marks)
        assert np.hypot(*(got[0][1][0] - B.local(B.EL, B.WR, B.MOLE, t))) < 0.5, "the mark left the mole"
        a, b, rf = B.preview(cut, tr, t)
        c = B.local(B.EL, B.WR, B.MOLE, t) * tr.proxy.sx
        assert B.contrast(b, c, 1.5, 6) < 0.35 * B.contrast(a, c, 1.5, 6), f"frame {t}: the marked mole stayed"
        c = B.local(B.EL, B.WR, B.PIMPLES[1], t) * tr.proxy.sx
        assert B.contrast(b, c, 1.5, 5, red=True) > 0.9 * B.contrast(a, c, 1.5, 5, red=True), f"frame {t}: the excluded pimple went"
        c = B.local(B.EL, B.WR, B.PIMPLES[0], t) * tr.proxy.sx
        assert B.contrast(b, c, 1.5, 5, red=True) < 0.5 * B.contrast(a, c, 1.5, 5, red=True), "the other pimple still goes"
    pp.body_marks = []


def test_marks_do_nothing_while_the_tool_is_off(body):
    cut, tr, _ = body
    from contour.retouch import bodyskin as BS
    from contour.retouch import engine as RE
    pp = B.body_on(cut, blemish=0)
    pp.body_marks = [BS.anchor_mark(tr.store, 0, 1, dict(id="a", kind="remove", t=1, pts=[list(B.local(B.EL, B.WR, B.MOLE, 1))], r=5.0))]
    assert RE.prepare(cut, tr, 2, {}) is None
    pp.body_marks = []


# ------------------------------------------------------------------------------------------------- the picture's own pixels
def test_export_lays_the_area_at_the_originals_own_pixels(body):
    """The analysis picture is half the original. The exported frame changes only where the original's own pixels
    are the person's skin: not a band of enlarged mask over the background, the top or the strap."""
    cut, tr, _ = body
    B.body_on(cut, skin=100, blemish=100, size=B.SPOT_SIZE)
    src, out = _export(cut, tr, 2)
    k = B.kind(2)
    ch = _changed(src, out)
    assert int((ch[_inside(k == B.SKIN_K, 8)] > 0).sum()) > 2000
    for name, kk in (("the background", B.BG), ("the top", B.CLOTH), ("the strap", B.STRAP), ("the hair", B.HAIR),
                     ("the other person", B.OTHER_K)):
        sel = _inside(k == kk, 1)
        assert int((ch[sel] > 0).sum()) == 0, f"{name}: {int((ch[sel] > 0).sum())} px changed, up to {int(ch[sel].max())}"
    off = k != B.SKIN_K
    assert int((ch[off] > 1).sum()) == 0, "more than a rounding outside the skin's own pixels"


def test_preview_and_export_agree_in_place_and_strength(body):
    """The same spots go, at the same places and by the same share, on the analysis picture and on the original."""
    cut, tr, _ = body
    B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    a, b, rf = B.preview(cut, tr, 2)
    src, out = _export(cut, tr, 2)
    for limb, uv in [("fore", u) for u in B.PIMPLES] + [("thigh", B.THIGH_SPOT)]:
        cs = B.local(*((B.EL, B.WR) if limb == "fore" else (B.HIP, B.KN)), uv, 2)
        left_x = B.contrast(out, cs, 3.0, 10, red=True) / B.contrast(src, cs, 3.0, 10, red=True)
        left_p = B.contrast(b, cs * 0.5, 1.5, 5, red=True) / B.contrast(a, cs * 0.5, 1.5, 5, red=True)
        assert left_x < 0.45 and left_p < 0.45 and abs(left_x - left_p) < 0.3, (limb, uv, left_x, left_p)
    # the export made small is the preview: where one changes the other does
    dx = cv2.resize(_changed(src, out).astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    dp = _changed(a, b).astype(np.float32)
    assert float(np.corrcoef(cv2.GaussianBlur(dx, (0, 0), 2).ravel(), cv2.GaussianBlur(dp, (0, 0), 2).ravel())[0, 1]) > 0.8


# ------------------------------------------------------------------------------------------------- speed and memory
def test_strength_changes_reuse_the_area_and_the_spots(body, monkeypatch):
    """Dragging the strength: the area, the parser's answer, the spots and their motion are not made again."""
    cut, tr, calls = body
    from contour.retouch import bodyspots as BP
    from contour.retouch import bodyskin as BS
    from contour.retouch import engine as RE
    n = dict(detect=0, motion=0)
    d0, m0 = BP.detect, BS.motion_pair
    monkeypatch.setattr(BP, "detect", lambda *a, **k: (n.__setitem__("detect", n["detect"] + 1), d0(*a, **k))[1])
    monkeypatch.setattr(BS, "motion_pair", lambda *a, **k: (n.__setitem__("motion", n["motion"] + 1), m0(*a, **k))[1])
    tr.__dict__.pop("_bflow", None)
    pp = B.body_on(cut, skin=50, blemish=50, size=B.SPOT_SIZE)
    cache = {}
    B.preview(cut, tr, 2, cache)
    first = (len(calls), n["detect"], n["motion"])
    assert first[0] >= 5 and first[1] >= 5 and first[2] == 4
    for v in (60, 70, 85):
        pp.amounts["body_skin"].base = v
        pp.amounts["body_blemish"].base = v
        pp.body_detail = v
        a, b, rf = B.preview(cut, tr, 2, cache)
    assert (len(calls), n["detect"], n["motion"]) == first, "a strength change made masks or spots again"
    # the next frame makes one more frame's worth, not five
    B.preview(cut, tr, 3, cache)
    assert len(calls) - first[0] == 1 and n["detect"] - first[1] == 1
    # and the motion between two frames is measured once, both ways: of the four pairs frame 4 asks for, the two with
    # frames 2 and 3 are there
    m3 = n["motion"]
    B.preview(cut, tr, 4, cache)
    assert n["motion"] - m3 == 2


def test_caches_hold_pieces_and_lists_not_frame_sized_maps(body):
    """What is kept for a person and frame: the piece of the area that is not empty at half precision, and the
    spots as a short list. Nothing of the frame's size, and it is counted in the masks' memory budget."""
    cut, tr, _ = body
    from contour.retouch import engine as RE
    B.body_on(cut, skin=80, blemish=80, size=B.SPOT_SIZE)
    cache = {}
    a, b, rf = B.preview(cut, tr, 2, cache)
    j = rf.jobs[0]
    full = tr.proxy.w * tr.proxy.h * 4
    pm = j.body["mask"]
    assert pm["m"].dtype == np.float16 and pm["m"].nbytes < 0.5 * full
    assert "bskin" not in j.masks and not j.masks, "no frame-sized body map among the masks"
    det = j.bdet
    assert det["maps"]["S"].nbytes + det["maps"]["R"].nbytes + det["cands"].nbytes < 20000
    body_keys = [k for k in cache if isinstance(k, tuple) and str(k[2]).startswith(("bskin", "bdet"))]
    assert len(body_keys) >= 10
    counted = sum(RE._nbytes(cache[k]) for k in cache if k != "__bytes__")
    assert cache["__bytes__"] >= counted > 0, "the body's entries are in the byte count"
    assert counted < 12 * 0.5 * full


def test_job_without_its_spots_finds_them_on_the_plane(body):
    """A frame worked on by itself (no neighbours, as a single export frame or a test does): the spots are found
    on the picture being worked on."""
    cut, tr, _ = body
    from contour.retouch import engine as RE
    B.body_on(cut, blemish=100, size=B.SPOT_SIZE)
    raw = RE._prepare_raw(cut, tr, 2, {})
    raw.jobs[0].bdet = None
    src = B.frame(2)
    Y, Cb, Cr = RE.rgb_to_ycc(src.astype(np.float32) / 255.0)
    Y2, Cb2, Cr2 = RE.process_ycc(Y, Cb, Cr, raw, 1.0)
    out = np.clip(RE.ycc_to_rgb(Y2, Cb2, Cr2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cs = B.local(B.EL, B.WR, B.PIMPLES[0], 2)
    assert B.contrast(out, cs, 3.0, 10, red=True) < 0.5 * B.contrast(src, cs, 3.0, 10, red=True)


# ------------------------------------------------------------------------------------------------- the render itself
class _Recorder:
    """Stands in for a render's output file: keeps the planes handed to the encoder."""
    path = ""

    def __init__(self):
        self.frames = []

    def write(self, planes):
        self.frames.append([np.array(p) for p in planes])


def _render(prj, cut, tr, fmt="prores_422_hq"):
    from contour.core.project import OutputSettings
    from contour.warp.render import render_cut
    rec = _Recorder()
    rep = render_cut(prj, cut, tr, OutputSettings(format=fmt, out_dir=str(prj.path) + "_out", include_audio=False),
                     ext_writer=rec)
    assert rep.frames == B.N and not rep.cancelled
    return rec.frames


def test_a_render_of_the_cut_retouches_the_body_in_its_10_bit_planes(scene, body):
    """The whole way of an export (render_cut, ProRes 422 HQ: 10-bit planes, chroma at half width): with the body
    effects on, every frame's planes differ from the render without them on the person's skin and nowhere else,
    and the spot that is removed is removed on every frame it can be seen on."""
    prj, _ = scene
    cut, tr, _ = body
    plain = _render(prj, cut, tr)
    B.body_on(cut, skin=100, blemish=100, size=B.SPOT_SIZE)
    done = _render(prj, cut, tr)
    assert len(done) == len(plain) == B.N
    Y0 = np.stack([f[0] for f in plain]).astype(np.int32)
    Y1 = np.stack([f[0] for f in done]).astype(np.int32)
    assert Y0.shape == (B.N, B.H, B.W) and int(Y0.max()) > 255, "10-bit luma at the original's size"
    assert plain[0][1].shape == (B.H, B.W // 2), "chroma at half width"
    left = []
    for t in range(B.N):
        k = B.kind(t)
        d = np.abs(Y1[t] - Y0[t])
        assert int((d[_inside(k == B.SKIN_K, 8)] > 0).sum()) > 2000, f"frame {t}: the skin is not retouched"
        for name, kk in (("the background", B.BG), ("the top", B.CLOTH), ("the other person", B.OTHER_K), ("a cover", B.COVER)):
            sel = _inside(k == kk, 2)
            if sel.any():
                assert int(d[sel].max()) == 0, f"frame {t}, {name}: luma changed by {int(d[sel].max())} of 1023"
        dc = np.abs(done[t][1].astype(np.int32) - plain[t][1].astype(np.int32))
        kc = k[:, ::2]
        assert int(dc[_inside(kc == B.CLOTH, 2)].max()) == 0 and int(dc[_inside(kc == B.BG, 2)].max()) == 0
        if t not in B.COVER_FRAMES:
            c = B.local(B.EL, B.WR, B.DARKS[0], t)
            yy, xx = np.mgrid[0:B.H, 0:B.W]
            spot = np.hypot(xx - c[0], yy - c[1]) < 1.5
            ring = (np.hypot(xx - c[0], yy - c[1]) > 5) & (np.hypot(xx - c[0], yy - c[1]) < 8)
            before = float(Y0[t][ring].mean() - Y0[t][spot].mean())
            after = float(Y1[t][ring].mean() - Y1[t][spot].mean())
            left.append(after / before)
    assert max(left) < 0.5, f"what is left of the dark spot on each frame: {[round(v, 2) for v in left]}"


# ------------------------------------------------------------------------------------------------- what is not the body's
def _conf_scene():
    """A picture with an arm (the parser is sure it is skin), a face of the same colour and a prop of the same
    colour the parser is not sure about, all inside the person's own pixels; and the parser's maps of it."""
    from contour.retouch import bodyskin as BS
    h, w = 240, 320
    rng = np.random.default_rng(3)
    rgb = np.full((h, w, 3), 58, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    arm = (xx >= 30) & (xx < 120) & (yy >= 50) & (yy < 190)
    face = np.hypot(xx - 190, yy - 80) < 34
    prop = (xx >= 240) & (xx < 300) & (yy >= 130) & (yy < 210)
    skin = arm | face | prop
    rgb[skin] = np.clip(np.array((204, 152, 128)) + rng.normal(0, 3, (int(skin.sum()), 3)), 0, 255).astype(np.uint8)
    own = np.ones((h, w), np.float32)
    S = 256                             # the parser's own size
    def m(sel, v=255):
        return cv2.resize(np.where(sel, v, 0).astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA)
    parse = dict(box=(0, 0, w, h), skin=np.maximum(m(arm | face), m(prop, 77)), hair=m(np.zeros_like(arm)),
                 cloth=m(np.zeros_like(arm)), face=m(face))
    inner = lambda sel: cv2.erode(sel.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return rgb, own, parse, inner(arm), inner(face), inner(prop)


def test_a_face_the_parser_sees_is_never_body_skin(monkeypatch):
    """BANANAS 083 (j2793): a face among the person's own pixels with no landmarks and no head joints to place it
    (the person's own in profile or half hidden, or somebody's the matte gave them) was body skin: smoothed, and
    its features taken for blemishes. The parser's face class is the face's, whoever places the head."""
    from contour.retouch import bodyskin as BS
    rgb, own, parse, arm, face, prop = _conf_scene()
    conf, model, parsed, allow = BS.skin_conf(rgb, own, None, parse, None, None, 200.0)
    assert parsed and model is not None
    assert conf[arm].min() > 0.9, "the arm is body skin as before"
    assert conf[face].max() < 0.02 and allow[face].max() < 0.02, "the face is not, and no neighbour's skin is carried onto it"
    # the same parse without the face class told apart (a parser stood in for by the tests): as it was
    old = {k: v for k, v in parse.items() if k != "face"}
    assert BS.skin_conf(rgb, own, None, old, None, None, 200.0)[0][face].min() > 0.9
    monkeypatch.setattr(BS, "FACE_OUT", None)
    assert BS.skin_conf(rgb, own, None, parse, None, None, 200.0)[0][face].min() > 0.9, "1.23's first build: body skin"


def test_what_the_parser_is_not_sure_of_is_skin_only_beside_skin_it_found(monkeypatch):
    """2611 036 (j2790): a glowing light stick in a hand. The parser gave it a skin share of about a third, nowhere
    was it sure of skin, and the colour (the face's, in the same light) agreed: the stick became body skin. A weak
    answer is no longer skin 'found'; skin the parser misses still counts beside skin it is sure of."""
    from contour.retouch import bodyskin as BS
    rgb, own, parse, arm, face, prop = _conf_scene()
    prior = BS.fit_model(rgb, (arm | face).astype(np.float32))
    assert prior is not None
    # nothing the parser is sure of: only the prop's weak answer, and the face's colour to go by
    S = parse["skin"].shape[0]
    weak = dict(parse, skin=np.where(parse["skin"] > 100, 0, parse["skin"]).astype(np.uint8), face=np.zeros((S, S), np.uint8))
    conf = BS.skin_conf(rgb, own, None, weak, None, prior, 200.0)[0]
    assert conf[prop].max() < 0.02, "a prop of the skin's colour the parser is not sure of stays as it is"
    monkeypatch.setattr(BS, "SUPPORT_FROM", None)
    assert BS.skin_conf(rgb, own, None, weak, None, prior, 200.0)[0][prop].min() > 0.5, "1.23's first build: body skin"
    monkeypatch.undo()
    # skin the parser misses beside skin it found: the lower part of the arm is not in its skin map at all
    part = parse["skin"].copy()
    part[int(S * 120 / 240):, :int(S * 130 / 320)] = 0
    conf = BS.skin_conf(rgb, own, None, dict(parse, skin=part), None, None, 200.0)[0]
    yy, xx = np.mgrid[0:240, 0:320]
    just_below = arm & (yy >= 122) & (yy < 126)
    far_below = arm & (yy >= 170)
    assert conf[just_below].min() > 0.5, "beside skin the parser found, the colour is enough"
    assert conf[far_below].max() < 0.05, "far from it, it is not"


# ------------------------------------------------------------------------------------------------- clothing the parser takes for skin
def _fur_parser(monkeypatch):
    """The body parser calling the top (nearly the skin's colour) skin, as it calls a fur collar (BANANAS 056)."""
    return B.truth_parser(monkeypatch, wrong=lambda t, k: np.where(k == B.CLOTH, B.SKIN_K, k).astype(np.uint8))


def _garment_words(monkeypatch, seen):
    """SAM 3.1's garment words stood in for by the scene's truth (the top and the strap are clothing). seen: the
    frames they were asked for."""
    from contour.track import clothing as CL

    class Finder:
        stats = dict(frames=0)

    def one(gf, st, proxy, pres, t, frame, lab, rgb, mw, mh):
        seen.append(int(t))
        k = B.kind(int(t), (mw, mh))
        for p in pres:
            own = lab == p
            frame[own] = 1
            frame[own & np.isin(k, (B.CLOTH, B.STRAP))] = 4 + 1
        return frame
    monkeypatch.setattr(CL, "GarmentFinder", Finder)
    monkeypatch.setattr(CL, "_garment_frame", one)
    from contour.track import skincloth as SC
    monkeypatch.setattr(SC, "models_missing", lambda: [])


def _area(cut, tr, t):
    from contour.retouch import engine as RE
    rf = RE.prepare(cut, tr, t, {})
    j = next(j for j in rf.jobs if j.pid == 0)
    return rf.body_area(j)


def _drop_look(tr):
    import shutil
    from contour.track import skincloth as SC
    shutil.rmtree(SC._dir(tr), ignore_errors=True)
    try:
        SC._state_file(tr).unlink()
    except OSError:
        pass
    for k in ("_skincloth_state", "_skincloth_lru", "_steady_lru"):
        tr.__dict__.pop(k, None)


def test_clothing_the_parser_takes_for_skin_is_found_by_a_look_and_left_out(body, monkeypatch):
    """BANANAS 056, 118 (j2790): a fur collar and cuffs of the skin's colour. The body parser calls them skin, the
    colour learnt from them agrees, and they were retouched. The cut is looked at with SAM 3.1's garment words
    (track.skincloth); where they call clothing what the parser is sure is skin, the area leaves it out."""
    from contour.track import skincloth as SC
    cut, tr, _ = body
    _drop_look(tr)
    _fur_parser(monkeypatch)
    B.body_on(cut, skin=100, blemish=100, size=B.SPOT_SIZE)
    try:
        top = _inside(_kinds(tr, 2) == B.CLOTH, 4)
        skin = _inside(_kinds(tr, 2) == B.SKIN_K, 4)
        before = _area(cut, tr, 2)
        assert float(before[top].mean()) > 0.8, "the defect: the top is body skin"
        src, out = _export(cut, tr, 2)
        assert int((_changed(src, out)[_inside(B.kind(2) == B.CLOTH, 8)] > 0).sum()) > 2000, "and it is retouched"
        assert SC.needed(tr, cut) == [0] and SC.info(tr)["state"] == "none"
        seen = []
        _garment_words(monkeypatch, seen)
        said = []
        assert SC.run(tr, SC.needed(tr, cut), lambda f, m: said.append(f)) is True
        assert sorted(seen) == list(range(B.N)), "both looks (frames 0 and 7) found it: every frame between has its map"
        assert said and said[-1] == 1.0 and said == sorted(said)
        assert SC.needed(tr, cut) == [] and SC.info(tr)["state"] == "current" and SC.info(tr)["ranges"] == [[0, B.N]]
        after = _area(cut, tr, 2)                      # (the masks made before the look are not used again)
        assert float(after[top].max()) < 0.05, "the top is no longer body skin"
        assert float(after[skin].mean()) > 0.9 * float(before[skin].mean()), "the skin is as before"
        src, out = _export(cut, tr, 2)
        d = _changed(src, out)
        assert int(d[_inside(B.kind(2) == B.CLOTH, 4)].max()) == 0, "and nothing of it changes in the export"
        assert int((d[_inside(B.kind(2) == B.SKIN_K, 8)] > 0).sum()) > 2000
        # a second run has nothing to do; cancelled, the look in use stays
        seen.clear()
        assert SC.run(tr, [0], cancel=lambda: True) is False and SC.info(tr)["ranges"] == [[0, B.N]]
    finally:
        _drop_look(tr)


def test_a_cut_without_such_clothing_costs_its_looks_only(body, monkeypatch):
    """The parser and the garment words agree (the top is clothing to both): the looked-at frames get a map, no
    other frame does, and the area is what it was without a look, on the looked-at frames too."""
    from contour.track import skincloth as SC
    cut, tr, _ = body
    _drop_look(tr)
    B.body_on(cut, skin=100)
    try:
        before = {t: _area(cut, tr, t).copy() for t in (0, 3)}
        seen = []
        _garment_words(monkeypatch, seen)
        assert SC.run(tr, SC.needed(tr, cut)) is True
        assert sorted(seen) == [0, B.N - 1], "frames 0 and 7 (every 12th and the last)"
        inf = SC.info(tr)
        assert inf["state"] == "current" and inf["ranges"] == [] and inf["looks"] == 2
        assert SC.get(tr, 0) is None and SC.get(tr, 3) is None
        assert sorted(f.name for f in SC._dir(tr).glob("*.png")) == ["000000.png", "000007.png"]
        for t in (0, 3):
            assert np.array_equal(_area(cut, tr, t), before[t])
        assert SC.needed(tr, cut) == []
    finally:
        _drop_look(tr)


def test_a_look_is_of_its_analysis_its_owners_and_the_people_it_saw(body, monkeypatch):
    from contour.track import skincloth as SC
    cut, tr, _ = body
    _drop_look(tr)
    _fur_parser(monkeypatch)
    B.body_on(cut, skin=100)
    seen = []
    _garment_words(monkeypatch, seen)
    meta = tr.matte.live_path.parent / "seg_meta.json" if hasattr(tr.matte, "live_path") else None
    keep_meta = meta.read_text(encoding="utf-8") if meta is not None and meta.exists() else None
    try:
        assert SC.needed(tr, cut) == [0]
        assert SC.run(tr, [0]) is True and SC.get(tr, 3) is not None
        # nobody with a body value on: nothing to look at
        B.body_on(cut, skin=0)
        assert SC.needed(tr, cut) == []
        B.body_on(cut, skin=100)
        # in and out moved inwards: the same frames, the look stands
        keep = (cut.in_frame, cut.out_frame)
        cut.in_frame, cut.out_frame = 2, 6
        assert SC.needed(tr, cut) == [] and SC.get(tr, 3) is not None
        cut.in_frame, cut.out_frame = keep
        # another analysis: not this look's
        cut.analysis_version += 1
        assert SC.needed(tr, cut) == [0] and SC.get(tr, 3) is None and SC.info(tr)["state"] == "stale"
        cut.analysis_version -= 1
        assert SC.needed(tr, cut) == [] and SC.get(tr, 3) is not None
        # the person matte made again on frames 2..4 after the look: other owners there, the look is made again
        if meta is not None and keep_meta is not None:
            import json as _json
            import time as _time
            d = _json.loads(keep_meta)
            d["refreshed"] = [[2, 5, _time.time() + 5]]
            meta.write_text(_json.dumps(d), encoding="utf-8")
            tr.__dict__.pop("_seg_meta", None)
            if SC.stale_ranges(tr):
                assert SC.stale_ranges(tr) == [[2, 5]]
                assert SC.get(tr, 3) is None and SC.get(tr, 6) is not None and SC.needed(tr, cut) == [0]
    finally:
        if meta is not None and keep_meta is not None:
            meta.write_text(keep_meta, encoding="utf-8")
            tr.__dict__.pop("_seg_meta", None)
        _drop_look(tr)


def test_the_frames_round_a_look_that_found_clothing():
    from contour.track import skincloth as SC
    looks = [0, 12, 24, 36, 47]
    assert SC.ranges_of(looks, [False] * 5) == []
    assert SC.ranges_of(looks, [False, False, True, False, False]) == [[12, 37]]
    assert SC.ranges_of(looks, [True, False, False, False, False]) == [[0, 13]]
    assert SC.ranges_of(looks, [False, False, False, False, True]) == [[36, 48]]
    assert SC.ranges_of(looks, [True, False, True, False, False]) == [[0, 37]], "two that touch are one"
    assert SC.ranges_of([5], [True]) == [[5, 6]]
    # what counts as found: a share of the skin the parser is sure of, and of the picture
    px = 1920 * 1080
    assert not SC._dirty({0: [50000, 300]}, px), "an edge the two draw a little apart"
    assert not SC._dirty({0: [900, 600]}, px), "a few hundred pixels of a small figure"
    assert SC._dirty({0: [50000, 9000]}, px) and SC._dirty({0: [50000, 300], 3: [20000, 15000]}, px)


def test_an_export_looks_at_the_cut_first_and_says_when_it_could_not(body, tmp_path, monkeypatch):
    """A render started before the app looked at the cut (a body value just switched on, a batch render) does it
    itself; when SAM 3.1 is not on the computer the render still comes out, with the parser's word alone and a
    line that says so."""
    from contour.core.project import OutputSettings
    from contour.track import clothing as CL
    from contour.track import skincloth as SC
    from contour.warp.render import render_cut
    cut, tr, _ = body
    _drop_look(tr)
    _fur_parser(monkeypatch)
    B.body_on(cut, skin=100)

    class Rec:
        path = ""

        def __init__(self):
            self.frames = []

        def write(self, planes):
            self.frames.append([np.array(p) for p in planes[:3]])
    project = tr.project
    out = OutputSettings(format="prores_4444", out_dir=str(tmp_path), include_audio=False)
    try:
        # SAM 3.1 is not on this computer (as in every test): nothing is downloaded, the render says so
        def never():
            raise AssertionError("a render opened (and would download) the garment models")
        monkeypatch.setattr(CL, "GarmentFinder", never)
        assert SC.models_missing() == list(SC.MODELS)
        r0 = Rec()
        rep = render_cut(project, cut, tr, out, ext_writer=r0)
        assert len(r0.frames) == B.N and any("피부색 옷" in w and "SAM 3.1" in w for w in rep.warnings), rep.warnings
        assert SC.info(tr)["state"] == "none"
        with pytest.raises(RuntimeError, match="SAM 3.1"):
            SC.run(tr, [0])
        seen = []
        _garment_words(monkeypatch, seen)
        r1 = Rec()
        rep = render_cut(project, cut, tr, out, ext_writer=r1)
        assert not any("피부색 옷" in w for w in rep.warnings) and sorted(seen) == list(range(B.N))
        assert SC.info(tr)["ranges"] == [[0, B.N]]
        B.body_on(cut, skin=0)
        plain = Rec()
        render_cut(project, cut, tr, out, ext_writer=plain)
        top = _inside(B.kind(3) == B.CLOTH, 4)
        d0 = np.abs(r0.frames[3][0].astype(np.int32) - plain.frames[3][0].astype(np.int32))
        d1 = np.abs(r1.frames[3][0].astype(np.int32) - plain.frames[3][0].astype(np.int32))
        assert int((d0[top] > 0).sum()) > 2000, "without the look the top was retouched"
        assert int(d1[top].max()) == 0, "with it, not a pixel of it"
    finally:
        _drop_look(tr)
