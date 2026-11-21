"""Knee tone (retouch.bodytone): blotches of the knee go to the leg's own colour, the grain, the sock and everything outside
the knee's area stay exactly."""
import cv2
import numpy as np
import pytest

from contour.retouch import bodytone as BT

H, W = 220, 160
CX, CY, HALF = 80, 110, 20.0


def scene(blotch=True, seed=3, amp=1.0):
    rng = np.random.default_rng(seed)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    leg = (np.abs(gx - CX) < HALF) & (gy < 175)
    Y = np.full((H, W), 0.40, np.float32)
    Cb = np.full((H, W), 0.0, np.float32)
    Cr = np.full((H, W), 0.0, np.float32)
    Y[leg] = 0.55 + 0.0006 * (gy[leg] - CY)                      # the leg's light along it
    Cb[leg], Cr[leg] = -0.05, 0.08
    Y[(gy >= 175) & (np.abs(gx - CX) < HALF + 2)] = 0.05            # the sock
    blob = np.zeros((H, W), np.float32)
    cv2.ellipse(blob, (CX - 4, CY + 4), (13, 17), 20, 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 3.0)
    red = np.zeros((H, W), np.float32)
    cv2.ellipse(red, (CX + 7, CY - 9), (7, 6), 0, 0, 360, 1.0, -1)
    red = cv2.GaussianBlur(red, (0, 0), 2.5)
    if blotch:
        Y -= 0.07 * amp * blob * leg
        Cb += 0.045 * amp * blob * leg
        Cr += (-0.035 * blob + 0.05 * red) * amp * leg
    Y += rng.normal(0, 0.012, (H, W)).astype(np.float32)             # grain
    Cb += rng.normal(0, 0.006, (H, W)).astype(np.float32)
    Cr += rng.normal(0, 0.006, (H, W)).astype(np.float32)
    skin = (cv2.GaussianBlur(leg.astype(np.float32), (0, 0), 0.8)).astype(np.float32)
    return Y, Cb, Cr, skin, blob * leg, red * leg


KNEE = [((float(CX), float(CY)), (0.0, 1.0), HALF)]


def lp(a, s=2.5):
    return cv2.GaussianBlur(a, (0, 0), s)


@pytest.fixture
def wide_limit(monkeypatch):
    """The limit on one pixel's change (BT.MAX_CHANGE, held in test_one_pixels_change_is_held_to_the_limit) is set wide: these tests are about WHERE
    the knee's colour goes (to the leg's), not how far one picture's pixel may move."""
    monkeypatch.setattr(BT, "MAX_CHANGE", {"luma": 0.035, "chroma": 0.04})


def test_the_limit_is_the_references_size():
    """The reference's knee moves by about 0.01 (log): the limit is of that order, not several times it (a knee evened, not repainted: it
    is what keeps a picture's wrong read of the leg from showing)."""
    assert 0.008 <= BT.MAX_CHANGE["luma"] <= 0.025 and 0.008 <= BT.MAX_CHANGE["chroma"] <= 0.03


def test_blotches_go_to_the_leg_colour(wide_limit):
    Y, Cb, Cr, skin, blob, red = scene()
    Y0, Cb0, Cr0, *_ = scene(blotch=False)
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    core = (BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF) < 0.6) & (skin > 0.9)
    for name, a, b, c in (("cb", Cb, Cbo, Cb0), ("cr", Cr, Cro, Cr0)):
        before = np.abs(lp(a - c)[core]).mean()
        after = np.abs(lp(b - c)[core]).mean()
        assert after < 0.3 * before, (name, before, after)       # (the small red patch is finer than the band the correction works in: a spot, for 몸 잡티)
    before = np.abs(lp(Y - Y0)[core]).mean()
    after = np.abs(lp(Yo - Y0)[core]).mean()
    assert after < 0.55 * before, (before, after)      # (a dark blotch keeps a quarter of itself, the knee's own shade: the reference keeps the knee's form; the brightness is read from further out than the colour: what is near may be the blotch's own)


def test_grain_is_kept_bit_for_bit_in_the_fine_band():
    Y, Cb, Cr, skin, *_ = scene()
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    s = max(2.0, BT.FINE * HALF)
    gy, gx = np.mgrid[0:H, 0:W]
    inside = (np.abs(gx - CX) < HALF - 5) & (gy < 165)          # (the leg's own edge is where the correction ends: not a grain matter)
    for a, b in ((Y, Yo), (Cb, Cbo), (Cr, Cro)):
        d = (a - cv2.GaussianBlur(a, (0, 0), s)) - (b - cv2.GaussianBlur(b, (0, 0), s))
        # the correction is a smooth field: what a blur takes from a picture is the same before and after, up to the
        # correction's own slow change over the blur's width
        assert np.abs(d[inside]).max() < 0.008, np.abs(d[inside]).max()      # (grain peaks at 0.03: the correction's own curvature)
    inner = (np.abs(gx - CX) < HALF - 3) & (np.abs(gy - CY) < 30)
    Y0, Cb0, Cr0, *_ = scene(blotch=False)                       # the same grain on a leg without blotches
    for a, b in ((Y0, Yo), (Cb0, Cbo), (Cr0, Cro)):
        assert 0.97 < (b - lp(b, 2.0))[inner].std() / (a - lp(a, 2.0))[inner].std() < 1.03


def test_nothing_outside_the_knees_area_and_the_sock_stay():
    Y, Cb, Cr, skin, *_ = scene()
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    e = BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF)
    far = e > 1.2
    for a, b in ((Y, Yo), (Cb, Cbo), (Cr, Cro)):
        assert np.array_equal(a[far], b[far])
    gy, gx = np.mgrid[0:H, 0:W]
    sock = (gy >= 175)
    assert np.array_equal(Y[sock], Yo[sock])


def test_off_and_half_strength():
    Y, Cb, Cr, skin, *_ = scene()
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 0.0)
    assert Yo is Y and Cbo is Cb and Cro is Cr
    Y0, Cb0, Cr0, *_ = scene(blotch=False)
    full = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    half = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 0.5)
    core = (BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF) < 0.6) & (skin > 0.9)
    d0 = np.abs(lp(Cb - Cb0)[core]).mean()
    d1 = np.abs(lp(full[1] - Cb0)[core]).mean()
    dh = np.abs(lp(half[1] - Cb0)[core]).mean()
    assert d1 < dh < d0


def test_chroma_at_half_the_size():
    Y, Cb, Cr, skin, *_ = scene()
    Cb2 = cv2.resize(Cb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    Cr2 = cv2.resize(Cr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    Yo, Cb2o, Cr2o = BT.knee_tone(Y, Cb2, Cr2, skin, KNEE, 1.0, cs=0.5)
    assert Cb2o.shape == Cb2.shape and Cr2o.shape == Cr2.shape
    assert np.abs(Cb2o - Cb2).max() > 0.01


def test_a_knee_without_skin_round_it_is_left_alone():
    Y, Cb, Cr, skin, *_ = scene()
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, np.zeros_like(skin), KNEE, 1.0)
    assert np.array_equal(Y, Yo) and np.array_equal(Cb, Cbo)


# ---------------------------------------------------------------- belly creases

def belly_scene(crease=True, seed=5):
    rng = np.random.default_rng(seed)
    h, w = 260, 200
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    trunk = np.abs(gx - 100) < 55
    Y = np.full((h, w), 0.35, np.float32)
    Y[trunk] = 0.58 + 0.00025 * (gy[trunk] - 130)
    if crease:
        for yy, amp in ((128, 0.07), (146, 0.045)):
            line = np.exp(-((gy - yy - 0.04 * (gx - 100)) ** 2) / (2 * 1.6 ** 2))     # a valley across the trunk
            Y -= amp * line * trunk
    Y += rng.normal(0, 0.006, (h, w)).astype(np.float32)
    Cb = np.full((h, w), -0.04, np.float32)
    Cr = np.full((h, w), 0.07, np.float32)
    skin = cv2.GaussianBlur(trunk.astype(np.float32), (0, 0), 0.8)
    return Y, Cb, Cr, skin


BELLY = dict(c=(100.0, 137.0), u=(0.0, 1.0), along=45.0, across=50.0, trunk=220.0)


def test_belly_creases_are_filled_and_nothing_else_moves():
    Y, Cb, Cr, skin = belly_scene()
    Y0, *_ = belly_scene(crease=False)
    Yo, Cbo, Cro = BT.belly_fold(Y, Cb, Cr, skin, BELLY, 1.0, 1.0)
    gy, gx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]]
    on = (np.abs(gx - 100) < 40) & (np.abs(gy - 137) < 28)
    before = np.abs(lp(Y - Y0, 1.2)[on]).mean()
    after = np.abs(lp(Yo - Y0, 1.2)[on]).mean()
    assert after < 0.4 * before, (before, after)
    far = BT.belly_zone(Y.shape, BELLY, 1.0) == 0
    assert np.array_equal(Y[far], Yo[far]) and Cb is Cbo and Cr is Cro


def test_a_belly_without_creases_stays_as_it_is():
    Y, Cb, Cr, skin = belly_scene(crease=False)
    Yo, *_ = BT.belly_fold(Y, Cb, Cr, skin, BELLY, 1.0, 1.0)
    gy, gx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]]
    on = (np.abs(gx - 100) < 40) & (np.abs(gy - 137) < 28)
    assert np.abs(Yo - Y)[on].mean() < 0.002, np.abs(Yo - Y)[on].mean()
    assert np.abs(lp(Yo - Y, 1.2)[on]).max() < 0.01


def test_belly_fold_off_and_without_geometry():
    Y, Cb, Cr, skin = belly_scene()
    for out in (BT.belly_fold(Y, Cb, Cr, skin, BELLY, 0.0, 1.0), BT.belly_fold(Y, Cb, Cr, skin, None, 1.0, 1.0)):
        assert out[0] is Y


def test_person_geometry_from_a_standing_pose():
    from contour.track import skeleton as S
    from contour.warp import body as B
    kp = np.full((17, 3), np.nan, np.float32)
    for j, xy in ((S.L_SH, (80, 60)), (S.R_SH, (120, 60)), (S.L_HIP, (88, 160)), (S.R_HIP, (112, 160)), (S.L_KN, (86, 230)),
                  (S.R_KN, (114, 230)), (S.L_AN, (84, 300)), (S.R_AN, (116, 300))):
        kp[j] = (xy[0], xy[1], 1.0)
    g = BT.person_geometry(kp, None)
    assert len(g["knees"]) == 2
    cx, cy, ax, ay, half = g["knees"][0]
    assert abs(cx - 86) < 1 and abs(cy - 230) < 1 and ay > 0.99 and half > 5
    b = g["belly"]
    # the waist runs from the neck to the pelvis: the belly lies a little over half way down it, the trunk runs down
    assert b is not None and b["u"][1] > 0.95 and 100 < b["c"][1] < 175 and b["along"] > 10


# ---- no body area (the parser found nothing): the leg's own strip, from the joints and the picture

def test_without_a_body_area_the_blotches_still_go_to_the_leg_colour(wide_limit):
    Y, Cb, Cr, skin, blob, red = scene()
    Y0, Cb0, Cr0, *_ = scene(blotch=False)
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, None, KNEE, 1.0)
    core = (BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF) < 0.6) & (skin > 0.9)
    for name, a, b, c in (("cb", Cb, Cbo, Cb0), ("cr", Cr, Cro, Cr0), ("y", Y, Yo, Y0)):
        before = np.abs(lp(a - c)[core]).mean()
        after = np.abs(lp(b - c)[core]).mean()
        assert after < (0.55 if name == "y" else 0.4) * before, (name, before, after)
    gy, gx = np.mgrid[0:H, 0:W]
    away = (np.abs(gx - CX) > HALF + 3) | (gy > 175)             # beside the leg, the sock, below it: as it was
    for a, b in ((Y, Yo), (Cb, Cbo), (Cr, Cro)):
        assert np.array_equal(a[away], b[away])


def test_without_a_body_area_cloth_is_left_alone():
    Y, Cb, Cr, skin, *_ = scene()
    for ymean, cb_, cr_ in ((0.08, 0.0, 0.0),            # black tights: dark
                            (0.5, 0.01, -0.02),          # a blue-grey trouser: bright, not warm
                            (0.5, 0.0, 0.0)):            # a grey one
        Y2 = np.where(skin > 0.5, ymean + (Y - 0.55) * 0.3, Y).astype(np.float32)
        Cb2 = np.where(skin > 0.5, cb_ + (Cb + 0.05) * 0.3, Cb).astype(np.float32)
        Cr2 = np.where(skin > 0.5, cr_ + (Cr - 0.08) * 0.3, Cr).astype(np.float32)
        out = BT.knee_tone(Y2, Cb2, Cr2, None, KNEE, 1.0)
        assert out[0] is Y2 and out[1] is Cb2 and out[2] is Cr2, (ymean, cb_, cr_)


def test_the_soft_top_edge_of_a_sock_is_not_pulled_up_to_the_legs_light():
    """The body skin's area takes the blur at a sock's edge for skin: the knee used to be lifted there to the leg's light, a
    bright crescent along the sock's top edge (a flat dim clip seen in Rec.709, the sock within the knee's reach)."""
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    leg = (np.abs(gx - CX) < HALF) & (gy < 200)
    ramp = np.clip((gy - 134) / 8.0, 0, 1)
    Y = np.full((H, W), 0.40, np.float32)
    Cb = np.zeros((H, W), np.float32)
    Cr = np.zeros((H, W), np.float32)
    Y[leg] = 0.55 * (1 - ramp[leg]) + 0.05 * ramp[leg]
    Cb[leg] = -0.05 * (1 - ramp[leg])
    Cr[leg] = 0.08 * (1 - ramp[leg])
    Y += np.random.default_rng(1).normal(0, 0.01, Y.shape).astype(np.float32)
    skin = cv2.GaussianBlur(leg.astype(np.float32), (0, 0), 0.8)
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    band = (gy >= 130) & (gy < 150) & (np.abs(gx - CX) < HALF - 4)
    assert float(np.abs(Yo - Y)[band].max()) < 0.01
    assert float(np.abs(Cbo - Cb)[band].max()) < 0.005


def test_one_pixels_change_is_held_to_the_limit():
    """A strong blotch is evened, but no pixel moves by more than the limit (the reference's knee moves by under 1 %)."""
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    leg = (np.abs(gx - CX) < HALF) & (gy < 200)
    Y = np.full((H, W), 0.40, np.float32)
    Y[leg] = 0.55
    Cb = np.zeros((H, W), np.float32)
    Cr = np.zeros((H, W), np.float32)
    blot = (gx - CX) ** 2 + (gy - CY) ** 2 < (0.5 * HALF) ** 2
    Y[blot] = 0.25
    Cr[blot] = 0.10
    skin = cv2.GaussianBlur(leg.astype(np.float32), (0, 0), 0.8)
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    assert float(np.abs(Yo - Y).max()) <= BT.MAX_CHANGE["luma"] + 1e-4
    assert float(np.abs(Cro - Cr).max()) <= BT.MAX_CHANGE["chroma"] + 1e-4
    assert float(np.abs(Yo - Y).max()) > 0.005         # still does something


# ---------------------------------------------------------------- 1.26: the knee tone's steadiness from one picture to the next

@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setattr(BT, "KNEE_V2", True)


def test_a_ring_of_a_few_pixels_leaves_the_knee_alone(monkeypatch):
    """A knee with hardly any of the leg's skin round it read the light of those few pixels as the leg's and changed the knee to it (a sliver of knee
    between cloth, 8 % lifted in one picture and none in the next)."""
    Y, Cb, Cr, skin, *_ = scene()
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    e = BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF)
    area_only = (e < 1.1).astype(np.float32) * skin
    ring_bit = ((np.abs(gx - CX) < 7) & (gy > 62) & (gy < 70)).astype(np.float32) * skin          # ~ 100 px of ring: enough to be read, hardly any of the leg
    only = np.maximum(area_only, ring_bit).astype(np.float32)
    monkeypatch.setattr(BT, "KNEE_V2", False)
    old = BT.knee_tone(Y, Cb, Cr, only, KNEE, 1.0)
    monkeypatch.setattr(BT, "KNEE_V2", True)
    new = BT.knee_tone(Y, Cb, Cr, only, KNEE, 1.0)
    assert float(np.abs(old[0] - Y).max()) > 0.004, "1.25 changed it (the case is real)"
    assert float(np.abs(new[0] - Y).max()) < 1e-3 and float(np.abs(new[2] - Cr).max()) < 1e-3
    full = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    assert float(np.abs(full[0] - Y).max()) > 0.004, "a knee with its leg round it is corrected as it was"


def test_two_knees_on_each_other_are_changed_once(v2):
    Y, Cb, Cr, skin, *_ = scene()
    one = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    two = BT.knee_tone(Y, Cb, Cr, skin, KNEE + KNEE, 1.0)
    for a, b in zip(one, two):
        assert np.allclose(a, b, atol=1e-6)
    assert float(np.abs(one[0] - Y).max()) <= BT.MAX_CHANGE["luma"] + 1e-4


def test_the_strength_is_given_by_the_gain(v2):
    Y, Cb, Cr, skin, *_ = scene()
    full = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0)
    half = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0, gains=[0.5])
    none = BT.knee_tone(Y, Cb, Cr, skin, KNEE, 1.0, gains=[0.0])
    assert none[0] is Y and none[1] is Cb and none[2] is Cr
    assert np.allclose(half[0] - Y, 0.5 * (full[0] - Y), atol=2e-4)


def test_the_envelope_fades_a_stretch_in_and_out_and_ignores_a_short_one():
    sup = {t: 1.0 for t in range(10, 20)}
    g = [BT.knee_envelope(sup, t, 0, 40) for t in range(0, 30)]
    assert max(g) == 1.0 and g[0] == 0.0 and g[29] == 0.0
    assert all(abs(g[i + 1] - g[i]) <= BT.ENV_STEP + 1e-9 for i in range(len(g) - 1))      # never a jump
    assert all(g[t] <= sup.get(t, 0.0) + 1e-9 for t in range(30))                            # never more than the picture's own support
    short = {20: 1.0, 21: 1.0}
    assert max(BT.knee_envelope(short, t, 0, 40) for t in range(0, 40)) <= BT.ENV_STEP + 1e-9   # two pictures: a fifth of a knee
    assert BT.knee_envelope({t: 1.0 for t in range(0, 10)}, 0, 0, 9) == 1.0       # the cut's own ends are not a fade
    assert BT.knee_envelope({}, 3, 0, 9) == 0.0


def test_a_leg_s_width_is_the_median_of_the_pictures_round_it(v2):
    from contour.track import skeleton as S
    from contour.warp import body as B
    n = 9
    kp = np.full((n, 17, 3), np.nan, np.float32)
    for j, xy in ((S.L_HIP, (88, 160)), (S.R_HIP, (112, 160)), (S.L_KN, (86, 230)), (S.R_KN, (114, 230)), (S.L_AN, (84, 300)), (S.R_AN, (116, 300)),
                  (S.L_SH, (80, 60)), (S.R_SH, (120, 60))):
        kp[:, j] = (xy[0], xy[1], 1.0)
    wd = np.full((n, len(B.SEG_INDEX), 2), np.nan, np.float32)
    meas = [10, 10, 40, 10, 10, 10, 41, 10, 10]
    for t in range(n):
        for nm in ("knee_tl", "knee_cl", "knee_tr", "knee_cr"):
            wd[t, B.SEG_INDEX[nm]] = meas[t]
    raw = BT.person_geometry(kp[2], wd[2])["knees"][0][4]
    g = BT.person_geometry_t(kp, wd, 2)
    assert raw == 40.0 and g["knees"][0][4] == 10.0 and g["sides"] == ["l", "r"]
    assert BT.person_geometry_t(kp, None, 4)["knees"][0][4] > 0           # no widths: the leg's own length gives one


def test_knee_support_follows_the_area_round_the_knee():
    h, w = 220, 160
    full = np.zeros((h, w), np.float32)
    full[:, CX - 20:CX + 20] = 1.0
    kn = [(float(CX), float(CY), 0.0, 1.0, HALF)]
    assert BT.knee_support(full, 1.0, kn)[0] > 0.95
    assert BT.knee_support(np.zeros((h, w), np.float32), 1.0, kn)[0] == 0.0
    sliver = np.zeros((h, w), np.float32)
    sliver[CY - 25:CY + 25, CX + 15:CX + 20] = 1.0
    assert BT.knee_support(sliver, 1.0, kn)[0] < 0.2


def test_the_look_ahead_hands_pictures_on_with_gains_that_never_jump(v2):
    import threading
    from types import SimpleNamespace as NS
    from contour.retouch import engine as RE
    from contour.retouch import bodyskin as BS
    n = 24
    lost = {10, 11, 12}                                                     # pictures the skin area is lost in (nothing of the leg left in it)
    px = NS(sx=1.0, sy=1.0)

    def prepare(t):
        m = np.zeros((220, 160), np.float32)
        if t in lost:
            m[0, 0] = 1.0
        else:
            m[:, CX - 20:CX + 20] = 1.0
        job = NS(pid=3, tone={"knees": [(float(CX), float(CY), 0.0, 1.0, HALF)], "sides": ["l"]}, body={"mask": BS.pack_mask(m)})
        return NS(jobs=[job])

    look = RE.KneeLook(prepare, lambda t: ("fi", t), threading.Lock(), 0, n, px).start()
    gains, order = [], []
    try:
        for t in range(n):
            rf, fi = look.get(t)
            assert fi == ("fi", t)
            gains.append(rf.jobs[0].tone["gain"][0])
            order.append(t)
    finally:
        look.stop()
    assert order == list(range(n))
    assert gains[10] == gains[11] == gains[12] == 0.0 and gains[0] == 1.0 and gains[n - 1] == 1.0
    assert all(abs(gains[i + 1] - gains[i]) <= BT.ENV_STEP + 1e-6 for i in range(n - 1))
    assert 0.0 < gains[9] < 1.0 and 0.0 < gains[13] < 1.0
