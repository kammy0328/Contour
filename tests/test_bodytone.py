"""Knee tone (retouch.bodytone): blotches of the knee go to the leg's own colour, the grain, the sock and everything outside
the knee's area stay exactly."""
import cv2
import numpy as np
import pytest

from contour.retouch import bodytone as BT

H, W = 220, 160
CX, CY, HALF = 80, 110, 20.0


def scene(blotch=True, seed=3):
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
        Y -= 0.07 * blob * leg
        Cb += 0.045 * blob * leg
        Cr += -0.035 * blob * leg + 0.05 * red * leg
    Y += rng.normal(0, 0.012, (H, W)).astype(np.float32)             # grain
    Cb += rng.normal(0, 0.006, (H, W)).astype(np.float32)
    Cr += rng.normal(0, 0.006, (H, W)).astype(np.float32)
    skin = (cv2.GaussianBlur(leg.astype(np.float32), (0, 0), 0.8)).astype(np.float32)
    return Y, Cb, Cr, skin, blob * leg, red * leg


KNEE = [((float(CX), float(CY)), (0.0, 1.0), HALF)]


def lp(a, s=2.5):
    return cv2.GaussianBlur(a, (0, 0), s)


def test_blotches_go_to_the_leg_colour():
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
    assert after < 0.3 * before, (before, after)


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

def test_without_a_body_area_the_blotches_still_go_to_the_leg_colour():
    Y, Cb, Cr, skin, blob, red = scene()
    Y0, Cb0, Cr0, *_ = scene(blotch=False)
    Yo, Cbo, Cro = BT.knee_tone(Y, Cb, Cr, None, KNEE, 1.0)
    core = (BT.knee_geometry((H, W), KNEE[0][0], KNEE[0][1], HALF) < 0.6) & (skin > 0.9)
    for name, a, b, c in (("cb", Cb, Cbo, Cb0), ("cr", Cr, Cro, Cr0), ("y", Y, Yo, Y0)):
        before = np.abs(lp(a - c)[core]).mean()
        after = np.abs(lp(b - c)[core]).mean()
        assert after < 0.4 * before, (name, before, after)
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
