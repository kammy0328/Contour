"""1.25 무릎 얼룩 정리 / 뱃살 주름 제거 through the engine: where they go, and that nothing else moves. The scene is the body
skin tests' drawn one with a blotch laid on the thigh's end at the knee."""
import cv2
import numpy as np
import pytest

import body_util as B

BLOB_AT = (-18.0, -14.0)          # from the knee joint (source px): inside the knee zone


def with_blotch(orig):
    def frame(t, marks=True, tex=1.0):
        f = orig(t, marks, tex).astype(np.float32) / 255.0
        k = B.kind(t)
        s = B.shift(t)
        c = np.array([B.KN[0] + BLOB_AT[0] + s[0], B.KN[1] + BLOB_AT[1] + s[1]])
        yy, xx = np.mgrid[0:B.H, 0:B.W].astype(np.float32)
        blob = np.exp(-(((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * 14.0 ** 2))) * (k == B.SKIN_K)
        f[..., 0] -= 0.10 * blob          # a grey-purple, darker patch
        f[..., 1] -= 0.05 * blob
        f[..., 2] += 0.02 * blob
        return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
    return frame


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("kneescene")
    mp = pytest.MonkeyPatch()
    (d / "home").mkdir()
    mp.setenv("HOME", str(d / "home"))
    mp.setenv("USERPROFILE", str(d / "home"))
    import contour.core.settings as S
    S._settings = None
    mp.setattr(B, "frame", with_blotch(B.frame))
    try:
        prj, cut = B.make_project(d)
    finally:
        mp.undo()
        S._settings = None
    return prj, cut


@pytest.fixture
def body(scene, monkeypatch):
    from contour.core import frames as FR
    prj, cut = scene
    FR.pool().clear()
    B.truth_parser(monkeypatch)
    for p in cut.persons:
        pp = B.body_on(cut, p.id)
        pp.body_marks = []
        pp.amounts["skin"].base = 0.0
        pp.amounts["knee_tone"].base = 0.0
        pp.amounts["belly_fold"].base = 0.0
    fr, tr = B.open_cut(prj, cut)
    yield cut, tr
    fr.close()


def test_off_prepares_nothing(body):
    from contour.retouch import engine as RE
    cut, tr = body
    assert RE.prepare(cut, tr, 2, {}) is None


def test_knee_tone_alone_brings_the_area_and_evens_the_patch_only_there(body):
    from contour.core.project import Curve
    cut, tr = body
    pp = cut.params_for(0)
    pp.amounts["knee_tone"] = Curve(base=100.0)
    src, out, rf = B.preview(cut, tr, 2)
    assert rf is not None and rf.jobs and rf.jobs[0].tone and rf.jobs[0].tone["knees"]
    sx = tr.proxy.sx
    s = B.shift(2)
    c = np.array([B.KN[0] + BLOB_AT[0] + s[0], B.KN[1] + BLOB_AT[1] + s[1]]) * sx
    yy, xx = np.mgrid[0:src.shape[0], 0:src.shape[1]]
    near = np.hypot(xx - c[0], yy - c[1]) < 10 * sx
    ys, ye = B.luma(src), B.luma(out)
    ring = (np.hypot(xx - c[0], yy - c[1]) > 22 * sx) & (np.hypot(xx - c[0], yy - c[1]) < 30 * sx) & (B.kind_map(src) if hasattr(B, "kind_map") else True)
    dip_src = float(np.median(ys[ring]) - ys[near].mean())
    dip_out = float(np.median(ye[ring]) - ye[near].mean())
    assert dip_src > 0.02 and dip_out < 0.8 * dip_src, (dip_src, dip_out)
    knee = np.array(B.KN) * sx + s * sx
    far = np.hypot(xx - knee[0], yy - knee[1]) > 3.2 * rf.jobs[0].tone["knees"][0][4] * sx
    assert np.array_equal(src[far], out[far]), "nothing outside the knee's area moves"


def test_belly_fold_on_a_scene_without_a_trunk_changes_nothing(body):
    from contour.core.project import Curve
    cut, tr = body
    cut.params_for(0).amounts["belly_fold"] = Curve(base=100.0)
    src, out, rf = B.preview(cut, tr, 2)
    assert np.array_equal(src, out)
