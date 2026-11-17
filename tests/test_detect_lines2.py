"""1.19 neck line detection, second round (found on real crops with j2720 and the local harness):
- a weak line in grain was thrown out by the "skin on both sides" test: the depth was replaced by the depth against
  the darker side, taken through the grain, so the texture term fell under its threshold. Now a gate on smooth sides
- a chain or a cord is one thing along its run: deep where it is in the light, faint elsewhere. The faint stretch
  alone read as a neck line (2611 cut 011: a necklace). A valley deep anywhere along its run is left alone everywhere
- lines in any direction but the neck's own (a hanging strand, a tendon's shade)
- the hollow under the lower lip and under the eyes is the face's own shade, not a dark spot (lngshot 058)
- the wide weighted mean on a reduced picture gives the same scores as the exact one."""
import numpy as np
import cv2

from test_detect import _skin, _peak, WF


def _line(Y, xx, yy, cx, cy, ang, half_len, w, depth):
    ca, sa = np.cos(ang), np.sin(ang)
    u, v = (xx - cx) * ca + (yy - cy) * sa, -(xx - cx) * sa + (yy - cy) * ca
    return Y * (1 - depth * np.exp(-0.5 * (v / w) ** 2) * np.exp(-0.5 * (u / half_len) ** 4))


def test_weak_line_in_textured_skin_is_found_along_its_length():
    """A 6 % line 1.3 px wide in skin with texture a few pixels across (pores, down): the share of its length found.
    With the depth taken against the darker side through the texture it was 0.47; with the gate, 0.84."""
    from contour.retouch import detect as DT
    shares = []
    for seed in range(6):
        Y, Cb, Cr, m, xx, yy = _skin(seed=seed)
        n = cv2.GaussianBlur(np.random.default_rng(100 + seed).normal(0, 1, Y.shape).astype(np.float32), (0, 0), 3.0)
        Y = _line(Y + 0.008 * n / n.std(), xx, yy, 180, 180, 0.2, 60, 1.3, 0.06)
        d = DT.lines(Y.astype(np.float32), m, WF)["line"]
        shares.append(np.mean([_peak(d, int(180 + u * np.cos(0.2)), int(180 + u * np.sin(0.2)), 2) > 0.5
                               for u in range(-45, 46, 3)]))
    assert np.mean(shares) > 0.7, np.mean(shares)


def test_chain_faint_along_part_of_its_run_is_left_alone():
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=2)
    # one thin dark thing across the neck: deep on the left (in the light), faint on the right
    depth = (0.07 + 0.38 / (1 + np.exp((xx - 150) / 12.0))).astype(np.float32)
    chain = Y * (1 - depth * np.exp(-0.5 * ((yy - 200 - 0.05 * (xx - 180)) / 1.2) ** 2) * (np.abs(xx - 180) < 140))
    d = DT.lines(chain, m, WF)
    assert float(d["line"][170:230, 60:320].max()) < 0.3, float(d["line"][170:230, 60:320].max())
    # the same faint line without the deep stretch is a neck line
    only = Y * (1 - 0.07 * np.exp(-0.5 * ((yy - 200 - 0.05 * (xx - 180)) / 1.2) ** 2) * (np.abs(xx - 240) < 70))
    assert _peak(DT.lines(only, m, WF)["line"], 240, 203, 5) > 0.5


def test_lines_down_the_neck_are_not_neck_lines():
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=4)
    axis = (0.0, 1.0)                                           # the neck runs down the picture
    for ang, want in ((0.0, True), (0.8, True), (-0.8, True), (np.pi / 2, False)):
        d = DT.lines(_line(Y, xx, yy, 180, 180, ang, 55, 1.4, 0.09), m, WF, axis=axis)
        p = _peak(d["line"], 180, 180, 4)
        assert (p > 0.5) == want, (ang, p)
    # without an axis (no face landmarks) every direction counts
    assert _peak(DT.lines(_line(Y, xx, yy, 180, 180, np.pi / 2, 55, 1.4, 0.09), m, WF)["line"], 180, 180, 4) > 0.5


def test_hollow_under_the_lip_and_eyes_is_the_faces_own_shade():
    from contour.retouch import detect as DT
    lm = np.full((68, 2), np.nan, np.float32)
    lm[27], lm[30], lm[8] = (180, 120), (180, 170), (180, 300)
    lm[31], lm[35], lm[48], lm[54], lm[57] = (160, 180), (200, 180), (145, 225), (215, 225), (180, 240)
    for i, p in zip((36, 39, 40, 41, 42, 45, 46, 47), ((110, 125), (150, 127), (138, 133), (122, 133), (210, 127), (250, 125), (238, 133), (222, 133))):
        lm[i] = p
    z = DT.fold_zone(lm, (360, 360), WF)
    assert z[262, 180] > 0.8, z[262, 180]                        # between the lower lip and the chin
    assert z[150, 130] > 0.8 and z[150, 230] > 0.8               # under the eyes
    assert z[60, 180] < 0.05 and z[200, 60] < 0.05               # the forehead, the cheek: not


def test_wide_mean_on_a_reduced_picture_agrees_with_the_exact_one(monkeypatch):
    from contour.retouch import detect as DT
    Y, Cb, Cr, m, xx, yy = _skin(seed=5)
    Y = Y * (1 - 0.12 * np.exp(-((xx - 120) ** 2 + (yy - 150) ** 2) / (2 * 3.0 ** 2)))      # a dark spot
    Y = _line(Y, xx, yy, 230, 240, 0.1, 50, 1.4, 0.09)
    fast = DT.detect(Y, Cb, Cr, m, m, WF, 0.037 * WF)
    monkeypatch.setattr(DT, "GM_WORK", 1e9)
    exact = DT.detect(Y, Cb, Cr, m, m, WF, 0.037 * WF)
    for k in ("dark", "red", "bump", "line", "fold"):
        a, b = fast[k] > 0.5, exact[k] > 0.5
        assert (a != b).sum() <= 0.1 * max(b.sum(), 1) + 12, (k, int((a != b).sum()), int(b.sum()))
    assert _peak(fast["dark"], 120, 150) > 0.5 and _peak(fast["line"], 230, 240) > 0.5


def test_dip_along_the_mouth_fold_goes_only_when_the_fold_is_to_go():
    """2611 cut 011 (j2721): the dip at the mouth's corner is the face's own shade, so the spot finder leaves it; but
    before 1.19 the blemish step took it, and with 팔자주름·입가주름 asked for the user wants it gone. Then it counts."""
    from contour.retouch import engine as RE
    from test_detect import _g
    Y0, Cb, Cr, m, xx, yy = _skin(seed=21)
    Y = (Y0 * (1 - 0.13 * _g(xx, yy, 121, 226, 5.5))).astype(np.float32)
    face = np.zeros((68, 3), np.float32)
    face[:, 0], face[:, 1] = 180, 120
    face[27, :2], face[30, :2], face[8, :2] = (180, 120), (180, 170), (180, 300)
    face[31, :2], face[35, :2], face[48, :2], face[54, :2], face[57, :2] = (160, 180), (200, 180), (145, 225), (215, 225), (180, 240)
    sm = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)

    def left(**opts):
        v = {k: 0.0 for k in RE.RETOUCH_KEYS}
        v["blemish"] = 1.0
        v.update(opts)
        job = RE.Job(0, 1, v, 0.45, "face", 1.0, face, 1.0, 600.0, masks={"skinx": m}, wf_src=WF, blemish_size=0.5)
        Y1, _, _ = RE.process_ycc(Y, Cb, Cr, RE.RetouchFrame(0, [job], (360, 360), (360, 360)), 1.0)
        return abs(float(sm(Y1 - Y0)[226, 121])) / abs(float(sm(Y - Y0)[226, 121]))
    assert left() > 0.8, left()
    assert left(nasolabial=0.8) < 0.5, left(nasolabial=0.8)
    assert left(mouth_lines=0.8) < 0.5, left(mouth_lines=0.8)
