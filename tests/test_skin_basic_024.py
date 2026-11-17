"""2611 cut 024 (strong + remove): the shine's white went dim and took the skin's colour. The basic adjustment no
longer sets the shine; people set by an older version are brought up to date when the project is read (values set
by hand stay), and the shine step keeps a highlight's colour."""
import numpy as np


def test_old_strong_basic_is_upgraded_hand_values_kept():
    from contour.core.project import PartParams
    from contour.core.params import SKIN_LEVEL_VALUES_V1
    old = {k: {"base": float(v)} for k, v in SKIN_LEVEL_VALUES_V1["strong"].items()}
    old["nasolabial"] = {"base": 60.0}
    old["mouth_lines"] = {"base": 60.0}             # switch off: a hand value, not the level's
    p = PartParams.from_dict({"amounts": old, "skin_basic": {"level": "strong", "remove": ["naso"]}})
    assert p.amount("shine", 0) == 0.0
    assert p.amount("skin", 0) == 90.0 and p.amount("redness", 0) == 30.0
    assert p.amount("neck_lines", 0) == 70.0 and p.amount("nasolabial", 0) == 80.0
    assert p.amount("mouth_lines", 0) == 60.0
    hand = dict(old, shine={"base": 25.0})
    p = PartParams.from_dict({"amounts": hand, "skin_basic": {"level": "strong", "remove": []}})
    assert p.amount("shine", 0) == 25.0, "a value set by hand is not touched"
    p = PartParams.from_dict({"amounts": old})                     # no basic adjustment: nothing changes
    assert p.amount("shine", 0) == 30.0


def test_shine_keeps_highlight_colour():
    from contour.retouch import engine as RE
    h = w = 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hot = np.exp(-(((xx - 150) ** 2 + (yy - 120) ** 2) / (2 * 14.0 ** 2)))
    Y = 0.55 + 0.3 * hot
    Cb = np.full((h, w), -0.03, np.float32) * (1 - 0.8 * hot)        # the highlight is whiter than the skin
    Cr = np.full((h, w), 0.05, np.float32) * (1 - 0.8 * hot)
    m = np.zeros((h, w), np.float32)
    m[30:270, 30:270] = 1
    Y1, Cb1, Cr1 = RE._shine(Y.astype(np.float32), Cb, Cr, m, m, 1.0, 240.0, 1.0)
    c = hot > 0.7
    assert np.allclose(Cb1[c], Cb[c]) and np.allclose(Cr1[c], Cr[c]), "the highlight keeps its colour"


def test_broad_fold_and_neck_ring_are_lifted():
    """A soft fold a tenth of the face wide (cut 024's nasolabial and neck rings): lifted, never darkened."""
    from contour.retouch import engine as RE
    h = w = 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wf = 300.0
    fold = np.exp(-((xx - 200) ** 2) / (2 * 12.0 ** 2))         # a vertical soft valley ~0.08 wf wide
    Y = (0.6 - 0.04 * fold).astype(np.float32)
    m = np.zeros((h, w), np.float32)
    m[60:340, 120:280] = 1
    Y1 = RE._fold_broad(Y, m, None, wf, 1.0, 0.12)
    assert (Y1 - Y).min() >= -1e-6
    assert float((Y1 - Y)[200, 200]) > 0.02, "the fold's depth is mostly filled"
    ring = np.exp(-((yy - 200) ** 2) / (2 * 8.0 ** 2))          # a horizontal ring
    Yn = (0.55 - 0.03 * ring).astype(np.float32)
    Y2 = RE._ring_fill(Yn, m, None, wf, 1.0, (0.0, 1.0))
    assert float((Y2 - Yn)[200, 200]) > 0.015 and (Y2 - Yn).min() >= -1e-6
