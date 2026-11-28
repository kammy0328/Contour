"""(1.27.1, LNGSHOT BEEP BEEP BTv1 review 11/27, 1127_BEEP_피드백.pptx) the fixes of the review:

- 이마주름 (forehead_lines): the soft horizontal lines above the brows levelled along the face's up; broad shading and the grain stay;
- 오돌토돌 정리 (texture_level): relief between the grain and TEXTURE_WIDE levelled; off by default;
- the chin's levelling (with 자갈턱 넓게) reaches the chin's edge;
- the blemish removal leaves the dark spots of the strip of neck under the jaw (the jaw's own shade, c024 p2: a light patch there);
- a turned face's neck band lies towards the jaw's middle (c025, c034)."""
import cv2
import numpy as np

from baseline_util import face68


def test_the_new_sliders_are_there_and_off_by_default():
    from contour.core import params as P
    for key in ("forehead_lines", "texture_level"):
        d = next(x for x in P.PARAMS if x.key == key)
        assert d.default == 0.0 and d.group == "skin" and (d.lo, d.hi) == (0, 100)


def _forehead_scene():
    H = W = 520
    wf = 400.0
    lm = face68(260.0, 160.0, wf).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    brow_y = float(lm[17:27, 1].mean())
    Y = 0.55 + 0.0004 * (xx - 260)                           # broad shading across the forehead
    lines = np.zeros((H, W), np.float32)
    for k in range(4):
        y = brow_y - 40 - 25 * k
        lines -= 0.03 * np.exp(-0.5 * ((yy - y) / 3.0) ** 2) * (np.abs(xx - 260) < 120)
    rng = np.random.default_rng(1)
    grain = cv2.GaussianBlur(rng.normal(0, 0.01, (H, W)).astype(np.float32), (0, 0), 0.6)
    return dict(Y=(Y + lines + grain).astype(np.float32), lines=lines, grain=grain, lm=lm, wf=wf, base=Y.astype(np.float32))


def test_forehead_lines_are_levelled_and_the_shading_stays():
    from contour.retouch import engine as RE
    sc = _forehead_scene()
    old_ = RE.FOREHEAD_ZONE_WIDE
    RE.FOREHEAD_ZONE_WIDE = 0.0                              # (the zone of 11/27: the lines lie above the brows only)
    try:
        z = RE.forehead_zone(sc["lm"] + np.float32(1e-3), sc["Y"].shape, 1.0, sc["wf"])
    finally:
        RE.FOREHEAD_ZONE_WIDE = old_
    assert z is not None and z.max() > 0.9
    out = RE._forehead_level(sc["Y"], z, sc["lm"], 1.0, sc["wf"], 1.0)
    inside = z > 0.95
    # the lines' relief (what varies from line to gap; the levelled forehead may sit at the lines' mean)
    before = sc["lines"][inside].std()
    after = (out - sc["base"] - sc["grain"])[inside].std()
    assert after < 0.5 * before
    # the broad shading (a gradient across the forehead) is still there
    row = int(np.median(np.nonzero(inside)[0]))
    xs = np.nonzero(inside[row])[0]
    assert (out[row, xs[-1]] - out[row, xs[0]]) > 0.5 * (sc["base"][row, xs[-1]] - sc["base"][row, xs[0]])
    # nothing outside the zone
    assert np.abs(out - sc["Y"])[z < 1e-3].max() < 1e-6
    # off: nothing
    assert np.array_equal(RE._forehead_level(sc["Y"], z, sc["lm"], 1.0, sc["wf"], 0.0), sc["Y"])


def test_texture_level_takes_the_relief_and_leaves_the_grain():
    from contour.retouch import engine as RE
    H = W = 400
    wf = 400.0
    rng = np.random.default_rng(2)
    relief = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 4.0)
    relief *= 0.02 / relief.std()
    grain = rng.normal(0, 0.008, (H, W)).astype(np.float32)
    Y = (0.5 + relief + grain).astype(np.float32)
    m = np.zeros((H, W), np.float32)
    m[60:340, 60:340] = 1.0
    out = RE._level(Y, m, wf, 1.0, RE.TEXTURE_WIDE, RE.TEXTURE_FINE, RE.TEXTURE_CAP)
    c = (slice(120, 280), slice(120, 280))
    lo = lambda a: cv2.GaussianBlur(a, (0, 0), 2.0)
    assert lo(out - 0.5)[c].std() < 0.6 * lo(relief)[c].std()
    hi = lambda a: a - cv2.GaussianBlur(a, (0, 0), 1.0)
    assert hi(out)[c].std() > 0.8 * hi(Y)[c].std()
    assert np.abs(out - Y)[m == 0].max() < 1e-6


def test_the_strip_under_the_jaw_holds_and_the_cheek_does_not():
    from contour.retouch import engine as RE
    H = W = 640
    wf = 400.0
    lm = face68(320.0, 100.0, wf).astype(np.float32)
    h = RE.under_jaw_hold(lm, (H, W), 1.0, wf, RE.BLEMISH_JAW_BAND)
    assert h is not None
    chin = lm[8, :2]
    assert h[int(chin[1] + 0.08 * wf), int(chin[0])] < 0.1           # just under the chin: held
    assert h[int(chin[1] + 0.45 * wf), int(chin[0])] > 0.95          # lower down the neck: not
    cheek = (lm[3, :2] + lm[31, :2]) / 2
    assert h[int(cheek[1]), int(cheek[0])] > 0.95                    # inside the face, off the line: not


def test_a_turned_face_has_its_neck_band_towards_the_jaw_middle(monkeypatch):
    from contour.retouch import engine as RE
    H = W = 700
    wf = 400.0
    lm = face68(350.0, 80.0, wf).astype(np.float32)
    shape = (H, W)
    prob = np.ones(shape, np.float32)
    front = RE.neck_mask(lm, None, None, shape, 1.0, 1.0, prob, None)
    # the same face, turned: the nose and the chin to one side of the jaw's middle
    t = lm.copy()
    t[27:36, 0] += 0.32 * wf
    t[48:68, 0] += 0.16 * wf
    t[6:11, 0] += 0.12 * wf
    assert RE.face_turn(t) > RE.NECK_TURN[0] + 0.1
    turned = RE.neck_mask(t, None, None, shape, 1.0, 1.0, prob, None)
    monkeypatch.setattr(RE, "NECK_TURN", None)
    old = RE.neck_mask(t, None, None, shape, 1.0, 1.0, prob, None)
    mid = float(t[[3, 13], 0].mean())
    cx = lambda m: float((m.sum(0) * np.arange(W)).sum() / max(1e-6, m.sum()))
    assert abs(cx(turned) - mid) < abs(cx(old) - mid)
    assert front.sum() > 0
