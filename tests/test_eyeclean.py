"""흰자 정리 (contour.retouch.eyeclean) on a drawn eye: the dot, the vein and the red go, the lashes, the grain and everything outside the white
stay; the basic adjustment's parts (eye, knee, elbow: 끄기 / 약하게 / 표준 / 강하게 each) set their sliders and are saved with the project."""
import numpy as np
import cv2

from contour.retouch import eyeclean as EC


def _eye(seed=0):
    rng = np.random.RandomState(seed)
    h, w = 160, 300
    Y = np.full((h, w), 0.55, np.float32)                      # skin
    Cb = np.full((h, w), -0.02, np.float32)
    Cr = np.full((h, w), 0.03, np.float32)
    # the eye opening (an almond), the white in it, a little shade under the upper lid
    eye6 = np.array([(70, 85), (110, 58), (190, 58), (230, 85), (190, 108), (110, 108)], np.float64)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (150, 84), (80, 26), 0, 0, 360, 1, -1)
    white = m > 0
    yy, xx = np.mgrid[0:h, 0:w]
    Y[white] = 0.78 - 0.002 * np.clip(70 - yy[white], 0, None)
    Cb[white], Cr[white] = 0.0, 0.004
    # the iris
    iris = (xx - 120) ** 2 + (yy - 82) ** 2 < 22 ** 2
    Y[iris & white] = 0.12
    Cb[iris & white], Cr[iris & white] = 0.03, -0.02
    # lashes hanging from the upper lid over the white
    lashes = np.zeros((h, w), np.uint8)
    for x in range(150, 215, 9):
        cv2.line(lashes, (x, 60), (x + 4, 71), 1, 1)
    lashes = (lashes > 0) & white
    Y[lashes] = 0.20
    # a dark dot and a red vein on the white
    dot = (xx - 185) ** 2 + (yy - 90) ** 2 < 3 ** 2
    Y[dot] = 0.55
    vein = np.zeros((h, w), np.uint8)
    cv2.line(vein, (155, 95), (205, 80), 1, 1)
    vein = (vein > 0) & white & ~dot
    Y[vein] -= 0.06
    Cr[vein] += 0.03
    grain = rng.normal(0, 0.006, (h, w)).astype(np.float32)
    Y = Y + grain
    return Y, Cb, Cr, eye6, dict(white=white, iris=iris, lashes=lashes, dot=dot, vein=vein)


def test_eye_white_cleaned_and_kept():
    Y, Cb, Cr, eye6, r = _eye()
    Y0, Cb0, Cr0 = Y.copy(), Cb.copy(), Cr.copy()
    Yo, Cbo, Cro, info = EC.clean_eye(Y, Cb, Cr, eye6, 1.0)
    assert "skip" not in info
    ring = cv2.dilate(r["dot"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(r["dot"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    before = float(Y0[ring].mean() - Y0[r["dot"]].mean())
    after = float(Yo[ring].mean() - Yo[r["dot"]].mean())
    assert after < 0.3 * before, (before, after)                          # the dot filled
    assert float((Cro - Cr0)[r["vein"]].mean()) < -0.01                   # the vein's red taken down
    assert float(Yo[r["vein"]].mean() - Y0[r["vein"]].mean()) > 0.02      # and its dark filled
    assert abs(float((Yo - Y0)[r["lashes"]].mean())) < 0.02               # the lashes kept
    outside = ~cv2.dilate(r["white"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert float(np.abs(Yo - Y0)[outside].max()) == 0.0                   # nothing outside the white
    clean = r["white"] & ~r["iris"] & ~cv2.dilate((r["dot"] | r["vein"] | r["lashes"]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    hp = lambda a: a - cv2.GaussianBlur(a, (0, 0), 1.0)
    assert float(hp(Yo)[clean].std()) > 0.85 * float(hp(Y0)[clean].std())    # the grain kept


def test_eye_planes_subsampled_chroma():
    Y, Cb, Cr, eye6, r = _eye(1)
    L = np.full((68, 2), np.nan)
    L[36:42] = eye6
    Cb2 = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Cr2 = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    Yo, Cbo, Cro = EC.clean_planes(Y.copy(), Cb2.copy(), Cr2.copy(), L, 1.0)
    assert Cbo.shape == Cb2.shape and Yo.shape == Y.shape
    assert float(np.abs(Yo - Y).max()) > 0.05
    # an eye too small, or no strength: nothing
    Yz, _, _ = EC.clean_planes(Y.copy(), Cb.copy(), Cr.copy(), L * 0.05, 1.0)
    assert float(np.abs(Yz - Y).max()) == 0.0
    Yk, _, _ = EC.clean_planes(Y.copy(), Cb.copy(), Cr.copy(), L, 0.0)
    assert float(np.abs(Yk - Y).max()) == 0.0


def test_basic_parts():
    from contour.core.params import skin_basic_values, PARAM
    v_old = skin_basic_values("std", ["naso"])
    assert "eye_white" not in v_old and "knee_tone" not in v_old          # an adjustment of before 1.26 does not touch the parts
    v = skin_basic_values("std", ["naso"], {"eye": "strong"})
    assert v["eye_white"] == 100.0 and v["knee_tone"] == 0.0
    v2 = skin_basic_values("strong", [], {"eye": "light", "knee": "std", "elbow": "strong"})
    assert v2["eye_white"] == 60.0 and v2["knee_tone"] == 70.0             # each part its own step, apart from the skin's level
    assert ("elbow_tone" in v2) == ("elbow_tone" in PARAM)                # a part the app has no slider for is left out


def test_basic_parts_saved():
    from contour.core.project import PartParams
    p = PartParams()
    p.skin_basic = {"level": "std", "remove": ["naso"], "parts": {"eye": "strong", "knee": "light"}}
    q = PartParams.from_dict(p.to_dict())
    assert q.skin_basic.get("parts") == {"eye": "strong", "knee": "light"}
    p2 = PartParams()
    p2.skin_basic = {"level": "std", "remove": []}
    assert "parts" not in PartParams.from_dict(p2.to_dict()).skin_basic
