"""다리 멍 정리 (contour.retouch.legbruise) on a drawn leg: a closed dark patch off the skin's hue goes, the leg's shading, a shade open to the
skin's edge (the gap between the thighs), the grain and everything off the leg stay."""
import numpy as np
import cv2

from contour.retouch import legbruise as LB


def _leg(seed=0):
    rng = np.random.RandomState(seed)
    h, w = 260, 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    skin = ((yy > 40) & (yy < 230)).astype(np.float32)               # a leg lying across the picture
    Y = np.where(skin > 0, 0.60 - 0.0006 * np.abs(yy - 135) ** 1.3, 0.15).astype(np.float32)   # rounder in the middle, darker to its sides
    Cb = np.where(skin > 0, -0.03, 0.0).astype(np.float32)
    Cr = np.where(skin > 0, 0.035, 0.0).astype(np.float32)
    bruise = np.exp(-(((xx - 150) / 18) ** 2 + ((yy - 120) / 14) ** 2)).astype(np.float32)
    Y -= 0.05 * bruise
    Cb += 0.012 * bruise
    Cr -= 0.010 * bruise
    # a shade open to the edge: a groove from the lower edge up into the leg
    gap = np.exp(-((xx - 300) / 5) ** 2).astype(np.float32) * (yy > 160) * skin
    Y -= 0.06 * gap
    Y += rng.normal(0, 0.004, (h, w)).astype(np.float32) * skin
    return Y, Cb, Cr, skin, bruise, gap


def test_leg_bruise_cleaned_and_kept():
    Y, Cb, Cr, skin, bruise, gap = _leg()
    Y0, Cb0, Cr0 = Y.copy(), Cb.copy(), Cr.copy()
    Yo, Cbo, Cro = LB.clean(Y.copy(), Cb.copy(), Cr.copy(), skin, 190.0, 1.0)
    core = bruise > 0.6
    ring = (bruise < 0.05) & (bruise > 0.005)
    before = float(Y0[ring].mean() - Y0[core].mean())
    after = float(Yo[ring].mean() - Yo[core].mean())
    assert before > 0.02 and abs(after) < 0.3 * before, (before, after)             # the bruise's dark filled
    assert float((Cbo - Cb0)[core].mean()) < -0.005                                  # and its blue taken out
    g = gap > 0.6
    assert float(np.abs(Yo - Y0)[g].mean()) < 0.005                                  # the shade open to the edge kept
    far = (bruise < 1e-3) & (gap < 1e-3)
    assert float(np.abs(Yo - Y0)[far & (skin > 0)].max()) < 0.01                      # the leg's own shading kept
    assert float(np.abs(Yo - Y0)[skin == 0].max()) == 0.0                            # nothing off the leg
    hp = lambda a: a - cv2.GaussianBlur(a, (0, 0), 1.0)
    assert float(hp(Yo)[core].std()) > 0.85 * float(hp(Y0)[core].std())             # the grain kept


def test_leg_planes_and_geometry():
    Y, Cb, Cr, skin, bruise, gap = _leg(1)
    legs = [(0.0, 135.0, 400.0, 135.0, 95.0)]
    Cb2 = cv2.resize(Cb, (200, 130), interpolation=cv2.INTER_AREA)
    Cr2 = cv2.resize(Cr, (200, 130), interpolation=cv2.INTER_AREA)
    Yo, Cbo, Cro = LB.clean_planes(Y.copy(), Cb2.copy(), Cr2.copy(), skin, legs, 1.0)
    assert Cbo.shape == Cb2.shape
    assert float(Yo[bruise > 0.6].mean() - Y[bruise > 0.6].mean()) > 0.02
    Yz, _, _ = LB.clean_planes(Y.copy(), Cb2.copy(), Cr2.copy(), skin, legs, 0.0)
    assert float(np.abs(Yz - Y).max()) == 0.0
    Yn, _, _ = LB.clean_planes(Y.copy(), Cb2.copy(), Cr2.copy(), skin, [], 1.0)
    assert float(np.abs(Yn - Y).max()) == 0.0
    a = LB.leg_area((260, 400), [(100.0, 50.0, 100.0, 200.0, 20.0)])
    assert a[120, 100] == 1 and a[120, 160] == 0


def test_leg_part():
    from contour.core.params import skin_basic_values, PARAM
    assert "leg_bruise" in PARAM
    v = skin_basic_values("std", [], {"leg": "std"})
    assert v["leg_bruise"] == 80.0 and v["knee_tone"] == 0.0
