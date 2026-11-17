"""1.19 flaw detection (retouch/detect.py) and what the engine does with it, on synthetic skin with grain:
- kinds: dark spots, red spots, raised bumps of the skin's colour (bright top), thin lines of several widths and
  directions, broad folds; and what is not a flaw (grain, a broad highlight, glitter, a fold beside the mouth, the
  foot of a shadow's edge, a dark strand: protected)
- the fill takes a found spot or bump away and leaves the pixel grain and the skin beside it alone
- scores held steady through time: grain found on one frame goes, a flaw missed on one frame comes back, a
  neighbour showing something else has no say, no neighbours: the frame's own scores
- the neck mask takes the side of the neck in shade where the parser is sure of neck, not thin things."""
import numpy as np
import cv2
import pytest


def _skin(h=360, w=360, seed=0, grain=0.012, level=0.55):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    Y = level + 0.06 * (xx / w - 0.5) + 0.03 * np.sin(yy / 70.0)          # soft shading
    Y = Y + cv2.GaussianBlur(rng.normal(0, grain, (h, w)).astype(np.float32), (0, 0), 0.7) * 1.6
    Cb = np.full((h, w), -0.03, np.float32) + rng.normal(0, 0.002, (h, w)).astype(np.float32)
    Cr = np.full((h, w), 0.05, np.float32) + rng.normal(0, 0.002, (h, w)).astype(np.float32)
    m = np.zeros((h, w), np.float32)
    m[20:h - 20, 20:w - 20] = 1.0
    return Y.astype(np.float32), Cb, Cr, m, xx, yy


def _g(xx, yy, cx, cy, s):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)).astype(np.float32)


WF = 300.0
SIZE = (0.012 + 0.05 * 0.5) * WF            # the default blemish size


def _peak(m, x, y, r=5):
    return float(m[y - r:y + r + 1, x - r:x + r + 1].max())


def test_spot_kinds_found_and_told_apart():
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin()
    Y = Y * (1 - 0.12 * _g(xx, yy, 90, 90, 2.5))                       # a dark spot
    Cr = Cr + 0.03 * _g(xx, yy, 250, 90, 3.0)                          # a red spot
    Y = Y * (1 + 0.09 * _g(xx, yy, 180, 180, 1.8) - 0.04 * _g(xx, yy, 183, 183, 2.2))   # a bump: bright top, dark side
    d = DT.spots(Y, Cr, m, SIZE)
    assert _peak(d["dark"], 90, 90) > 0.6
    assert _peak(d["red"], 250, 90) > 0.6
    assert _peak(d["bump"], 180, 180) > 0.6
    assert _peak(d["bump"], 90, 90) < 0.3 and _peak(d["red"], 90, 90) < 0.3
    assert _peak(d["dark"], 250, 90) < 0.3


def test_grain_highlight_glitter_and_folds_are_not_flaws():
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=3, grain=0.02)
    d0 = DT.spots(Y, Cr, m, SIZE)
    for k in ("dark", "red", "bump"):
        assert float((d0[k] > 0.5)[m > 0.5].mean()) < 0.004, f"grain taken for {k}"
    Y2 = Y * (1 + 0.10 * _g(xx, yy, 180, 120, 22.0))                    # a broad highlight (a forehead, a nose ridge)
    Y2 = Y2 * (1 + 0.6 * _g(xx, yy, 100, 250, 1.5))                     # glitter: far brighter than a bump
    yv = np.abs((xx - 240) * 0.7 + (yy - 200) * 0.7)                    # a fold: a valley running on
    Y2 = Y2 * (1 - 0.10 * np.exp(-(yv ** 2) / (2 * 4.0 ** 2)) * (np.abs((xx - 240) * 0.7 - (yy - 200) * 0.7) < 70))
    d = DT.spots(Y2.astype(np.float32), Cr, m, SIZE)
    assert _peak(d["bump"], 180, 120, 12) < 0.5, "a broad highlight is the face's own light"
    pr = DT.protect(Y2.astype(np.float32), Cb, Cr, m, WF)
    assert _peak(d["bump"] * (1 - pr), 100, 250) < 0.3, "glitter stays (too bright for a bump, and protected as a glint)"
    assert _peak(d["dark"], 240, 200, 6) < 0.35, "a fold is no spot"


@pytest.mark.parametrize("width,ang", [(1.2, 0.0), (2.4, 0.5), (4.5, -0.6), (2.4, 1.2)])
def test_neck_lines_of_several_widths_and_directions(width, ang):
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=5)
    ca, sa = np.cos(ang), np.sin(ang)
    u = (xx - 180) * ca + (yy - 180) * sa
    v = -(xx - 180) * sa + (yy - 180) * ca
    Y = Y * (1 - 0.08 * np.exp(-(v ** 2) / (2 * width ** 2)) * (np.abs(u) < 70))
    d = DT.lines(Y.astype(np.float32), m, WF, (0.0, 1.0))
    assert _peak(d["line"], 180, 180, 4) > 0.6, f"width {width} px at {ang} rad"


def test_shadow_edge_is_no_line_and_strand_is_protected():
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=7)
    Y = Y * (1 - 0.25 / (1 + np.exp((xx - 120) / 2.5)))                 # the dark side of a shadow, a soft step
    d = DT.lines(Y.astype(np.float32), m, WF, (0.0, 1.0))
    assert float(d["line"][40:320, 100:140].max()) < 0.35 and float(d["fold"][40:320, 100:140].max()) < 0.35
    Y2, Cb2, Cr2, m2, xx, yy = _skin(seed=8)
    strand = np.exp(-((yy - 200) ** 2) / (2 * 1.3 ** 2))                 # a strand of hair across: far darker, grey
    Y2 = Y2 * (1 - 0.7 * strand)
    Cb2, Cr2 = Cb2 * (1 - 0.9 * strand), Cr2 * (1 - 0.9 * strand)
    pr = DT.protect(Y2.astype(np.float32), Cb2, Cr2, m2, WF)
    assert float(pr[200, 60:300].min()) > 0.7


def test_fill_takes_spot_and_bump_keeps_grain_and_surroundings():
    from contour.retouch import detect as DT, engine as RE
    Y0, Cb, Cr, m, xx, yy = _skin(seed=11)
    spot = 0.12 * _g(xx, yy, 100, 100, 3.0)
    bump = 0.09 * _g(xx, yy, 220, 220, 2.0)
    Y = (Y0 * (1 - spot) * (1 + bump)).astype(np.float32)
    d = DT.spots(Y, Cr, m, SIZE)
    S = np.maximum(d["dark"], d["bump"])
    Y1, Cb1, Cr1 = RE._spot_fill(Y, Cb, Cr, m, S, d["red"], 1.0, SIZE, 1.0)
    sm = lambda a: cv2.GaussianBlur(a, (0, 0), 1.2)               # the flaw's own size, without the grain
    for (cx, cy), amp in (((100, 100), spot), ((220, 220), bump)):
        left = abs(float(sm(Y1 - Y0)[cy, cx])) / abs(float(sm(Y - Y0)[cy, cx]))
        assert left < 0.4, f"{left:.2f} of the flaw at {(cx, cy)} is left"
    far = S < 0.01
    assert float(np.abs(Y1 - Y)[far].max()) < 1e-3, "nothing changes where nothing was found"
    # grain: all of it stays beside the spot (where the score is low); in the sure core only the pixel-fine part,
    # and nothing is put there in its place
    fine = lambda a: a - cv2.GaussianBlur(a, (0, 0), 1.0)
    ring = (S > 0.02) & (S < 0.3)
    assert fine(Y1)[ring].std() > 0.9 * fine(Y)[ring].std()
    core = S > 0.85
    assert 0.1 * fine(Y)[core].std() < fine(Y1)[core].std() <= fine(Y)[core].std()


def _det(maps, wf=WF):
    h, w = next(iter(maps.values())).shape
    base = np.full((h // 4, w // 4), 128, np.uint8)
    return dict(box=(0.0, 0.0, float(w), float(h)), maps={k: (np.clip(v, 0, 1) * 255).astype(np.uint8) for k, v in maps.items()},
                thumb=base, wf=wf)


def test_scores_held_steady_through_time():
    from contour.retouch import engine as RE
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    real = _g(xx, yy, 60, 60, 4.0)                 # on every frame (moved by the face's motion)
    rng = np.random.default_rng(1)
    I = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    shift = np.array([[1, 0, 3], [0, 1, 0]], np.float32)       # the neighbour's picture is 3 px to the left of this one

    def frame(noise_at, has_real=True, dx=0):
        m = np.zeros((h, w), np.float32)
        if has_real:
            m = np.maximum(m, _g(xx, yy, 60 - dx, 60, 4.0))
        m = np.maximum(m, _g(xx, yy, noise_at[0], noise_at[1], 3.0))
        return _det({"dark": m, "protect": np.zeros((h, w), np.float32)})

    own = frame((150, 150))
    nb = [(0.6, frame((30, 160), dx=3), shift), (0.6, frame((170, 40), dx=3), shift)]
    f = RE._det_fuse(own, nb)["maps"]["dark"].astype(np.float32) / 255
    assert f[60, 60] > 0.8, "the flaw every frame has stays (carried by the motion)"
    assert f[150, 150] < 0.2, "grain found on this frame alone goes"
    assert f[160, 33] < 0.35 and f[40, 173] < 0.35, "a neighbour's grain is not brought here in full"
    # the grain hid the flaw on this frame: the neighbours bring it back
    miss = RE._det_fuse(frame((150, 150), has_real=False), nb)["maps"]["dark"].astype(np.float32) / 255
    assert miss[60, 60] > 0.45
    # a neighbour that shows something else there (a hand in front): no say, the frame's own score stands
    cover = frame((30, 160), has_real=False, dx=3)
    cover["thumb"] = np.full_like(cover["thumb"], 30)
    f2 = RE._det_fuse(own, [(0.6, cover, shift), (0.6, cover, shift)])["maps"]["dark"].astype(np.float32) / 255
    assert f2[60, 60] > 0.8 and f2[150, 150] > 0.8
    # no neighbours (a cut's edge, a failed alignment): unchanged
    assert RE._det_fuse(own, []) is own


def test_engine_takes_a_bump_only_with_detection():
    """A raised bump of the skin's colour: the blemish step before 1.19 (dark and red spots, one size) left it."""
    from contour.retouch import engine as RE
    Y0, Cb, Cr, m, xx, yy = _skin(seed=13)
    bump = 0.10 * _g(xx, yy, 180, 180, 2.0)
    Y = (Y0 * (1 + bump)).astype(np.float32)
    v = {k: 0.0 for k in RE.RETOUCH_KEYS}
    v["blemish"] = 1.0
    face = np.zeros((68, 3), np.float32)
    face[:, 0] = np.linspace(30, 30 + WF, 68)
    face[:, 1] = 180
    job = RE.Job(0, 1, v, 0.45, "face", 1.0, None, 1.0, 600.0, masks={"skinx": m}, wf_src=WF, blemish_size=0.5)
    rf = RE.RetouchFrame(0, [job], (360, 360), (360, 360))

    sm = lambda a: cv2.GaussianBlur(a, (0, 0), 1.2)

    def left(on):
        old = RE.DET_ON
        RE.DET_ON = on
        try:
            Y1, _, _ = RE.process_ycc(Y, Cb, Cr, rf, 1.0)
        finally:
            RE.DET_ON = old
        return abs(float(sm(Y1 - Y0)[180, 180])) / abs(float(sm(Y - Y0)[180, 180]))
    assert left(False) > 0.85, "before: the bump stays"
    assert left(True) < 0.4, "now: most of it is gone"


def test_neck_mask_takes_the_side_in_shade_not_thin_things():
    from contour.retouch import engine as RE
    h, w = 420, 360
    lm = np.zeros((68, 3), np.float32)
    # a face outline: jaw from (80,120) round the chin (180,260) to (280,120)
    for i in range(17):
        a = np.pi * i / 16
        lm[i, :2] = (180 - 100 * np.cos(a), 120 + 140 * np.sin(a))
    lm[17:27, :2] = [(100 + 18 * k, 90) for k in range(10)]
    lm[27, :2] = (180, 110)
    lm[27:, 0] = 180
    lm[27:, 1] = np.linspace(110, 230, 41)
    prob = np.ones((h, w), np.float32)
    prob[:, :170] = 0.0                              # the left half of the neck is in shade: not the colour model's skin
    prob[:, 228:234] = 0.0                           # a strand of hair down the right half
    parse = dict(neck=np.zeros((h, w), np.float32), skin=np.zeros((h, w), np.float32), cover=np.ones((h, w), np.float32),
                 trust=1.0)
    parse["neck"][250:400, 100:260] = 1.0
    old = RE.NECK_SHADE
    try:
        RE.NECK_SHADE = False
        m0 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
        RE.NECK_SHADE = True
        m1 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
    finally:
        RE.NECK_SHADE = old
    assert float(m0[300:340, 120:160].mean()) < 0.1, "before: the side in shade is no neck"
    assert float(m1[300:340, 120:160].mean()) > 0.7
    assert float(m1[300:340, 230:232].mean()) < 0.5, "the strand stays out"
