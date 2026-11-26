"""1.26.3: the retouch's look kept per project (retouch/looks.py; the review of 11/26: a cut of a delivered project rendered again after an
update came out with another skin). The look 1.26.1 renders and shows the frozen baseline projects as DIAP 1.26.1 did (its golden at
v1.26.1, tests/data/*/golden_look_1261.npz); the newest look is the engine as it is (golden.npz, test_baseline_1222)."""
import json
from pathlib import Path

import numpy as np
import pytest

import baseline_util as BU
from contour.retouch import engine as E
from contour.retouch import looks as LK
from contour.retouch import skin as SK

DATA = Path(__file__).resolve().parent / "data"
PLANE_TOL = 2
PREVIEW_TOL = 1


@pytest.fixture(autouse=True)
def _newest_after():
    yield
    LK.use("")


def test_use_and_back():
    new = {k: getattr(E if k.startswith("engine.") else SK, k.split(".", 1)[1]) for k in LK.LOOKS["1.26.1"]}
    assert LK.use("1.26.1") == "1.26.1" and LK.current() == "1.26.1"
    assert E.FOLD_ZONE_OLD is True and E.FACE_GAP_BRIDGE == 0 and SK.FS_EDGE_GRAD is None and E.FINAL_NASO_SLIDER is False
    assert tuple(SK.FS_G) == (1.35, 0.70, 0.36, 0.33, 0.45, 0.80) and E.SKIN_FS_G is None
    assert LK.use("") == LK.LOOK and LK.use("no such look") == LK.LOOK
    for k, v in new.items():
        assert getattr(E if k.startswith("engine.") else SK, k.split(".", 1)[1]) == v, k


def test_every_setting_1262_changed_is_in_the_look():
    # the settings a look puts back: every one the 1.26.2 defaults changed (tools/look_diff.py found them), and only those
    for k in LK.LOOKS["1.26.1"]:
        mod, name = k.split(".", 1)
        assert hasattr(E if mod == "engine" else SK, name), k
    assert "engine.PREP_THREADS" not in LK.LOOKS["1.26.1"] and "engine.RETOUCH_KEYS" not in LK.LOOKS["1.26.1"]


def test_held_restores():
    prj = type("P", (), {"look": "1.26.1"})()
    with LK.held(prj):
        assert LK.current() == "1.26.1"
    assert LK.current() == LK.LOOK


def test_project_saves_its_look(tmp_path):
    from contour.core.project import Project
    prj = Project.create(str(tmp_path), "look_test")
    assert prj.look == LK.LOOK                       # a new project keeps the look it was made with
    prj.look = "1.26.1"
    prj.save()
    assert Project.load(prj.path).look == "1.26.1"
    d = json.loads((Path(prj.path) / "project.json").read_text(encoding="utf-8"))
    d.pop("look")
    (Path(prj.path) / "project.json").write_text(json.dumps(d), encoding="utf-8")
    old = Project.load(prj.path)
    assert old.look == "" and "look" not in old.to_dict()      # a project from before: nothing added by opening and saving it


@pytest.mark.parametrize("name,body", [("baseline_1222", False), ("baseline_1222_body", True)])
def test_look_1261_renders_what_1261_rendered(tmp_path, name, body):
    from contour.core.project import Project
    g = np.load(DATA / name / "golden_look_1261.npz")
    pdir, src = BU.copy_fixture(DATA / name, tmp_path)
    prj = Project.load(str(pdir))
    prj.look = "1.26.1"
    with BU.NoModels():
        planes = BU.render_planes(prj, prj.cuts[0])
        LK.use("1.26.1")
        try:
            prev = BU.preview_frames(prj, prj.cuts[0], [int(t) for t in g["frames"]])
        finally:
            LK.use("")
    d = np.abs(planes.astype(np.int32) - g["planes"].astype(np.int32))
    assert d.max() <= PLANE_TOL, f"render: max {d.max()} of 1023, mean {d.mean():.4f}"
    dp = np.abs(prev.astype(np.int32) - g["preview"].astype(np.int32))
    assert dp.max() <= PREVIEW_TOL, f"preview: max {dp.max()} of 255"
    assert LK.current() == LK.LOOK                      # the render held the project's look for its time only


def test_render_notes_a_changed_look(monkeypatch):
    from contour.warp import render as R
    monkeypatch.setattr(R, "_render_cut", lambda *a, **k: R.RenderReport())
    monkeypatch.setattr(R.RE, "prewarm_models", lambda cut, tr: {})
    cut = type("C", (), {"render_look": "1.26.1"})()
    prj = type("P", (), {"look": ""})()
    rep = R.render_cut(prj, cut, object(), None)
    assert any("1.26.1 보정으로 렌더" in w for w in rep.warnings) and cut.render_look == LK.LOOK
    rep = R.render_cut(prj, cut, object(), None)
    assert not rep.warnings
