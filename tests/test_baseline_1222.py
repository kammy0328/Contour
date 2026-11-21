"""Instruction 2026-11-20 §6-1: before anything else is changed, the opening and the output of existing projects are
held. tests/data/baseline_1222 is a small analysed project exactly as DIAP 1.22.2 (6a9f96e) saved it (a Windows
project: its original under C:\\..., tracks without a matte revision, a matte record without one), and what 1.22.2
showed and rendered from it (tools/make_baseline_fixture.py, run with the 1.22.2 code). The code of now must

- open it as it is: every value 1.22.2 saved read back the same, nothing in its folder written by opening it;
- show and render the same picture: bit for bit where numpy, OpenCV and the platform are the ones the fixture was
  made with, within a rounding step of the 10-bit planes elsewhere (other SIMD paths);
- open it in the window without a question, an error or an unsaved change, adding at most the files listed in
  baseline_util.ALLOWED_NEW;
- save it so that everything 1.22.2 wrote is still there."""
import json
import time
from pathlib import Path

import os
import numpy as np
import pytest

import baseline_util as BU

FIX = Path(__file__).resolve().parent / "data" / "baseline_1222"
PLANE_TOL = 2          # of 1023: a rounding step either way in the warp's resampling and the retouch's filters
PREVIEW_TOL = 1        # of 255


@pytest.fixture
def golden():
    with np.load(FIX / "golden.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


@pytest.fixture
def old_project(tmp_path):
    return BU.copy_fixture(FIX, tmp_path)


def _same_env(golden) -> bool:
    """Bit for bit only where the fixture's numbers can be expected again: the same libraries, and not on a hosted CI
    runner (its CPU differs from run to run, and a filter's last rounding step with it: the preview of the first
    GitHub run was within one step of the golden everywhere, not equal). The tolerances are checked everywhere."""
    if os.environ.get("GITHUB_ACTIONS"):
        return False
    return str(golden["env"]) == BU.env_key()


def test_exact_comparison_is_left_to_machines_that_can_give_it(monkeypatch):
    env = BU.env_key()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert _same_env({"env": env}) and not _same_env({"env": env + "x"})
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert not _same_env({"env": env})


def test_fixture_is_what_1222_wrote(golden):
    assert str(golden["version"]) == "1.22.2"
    cut_dir = next((FIX / "project" / "cuts").iterdir())
    with np.load(cut_dir / "tracks.npz") as d:
        assert "matte_rev" not in d.files, "tracks saved by 1.22.2 know no matte revision"
    meta = json.loads((cut_dir / "cache" / "seg_meta.json").read_text(encoding="utf-8"))
    assert "revision" not in meta and not (cut_dir / "cache" / "matte_prompts.npz").exists()
    assert BU.PLACEHOLDER in json.loads((FIX / "project" / "project.json").read_text(encoding="utf-8"))["cuts"][0]["source"]


def test_existing_project_reads_back_every_value(old_project, golden):
    from contour.core.project import Project
    pdir, src = old_project
    before = BU.digest(pdir)
    prj = Project.load(str(pdir))
    cut = prj.cuts[0]
    was = json.loads(str(golden["cut_json"]).replace(json.dumps(BU.PLACEHOLDER)[1:-1], json.dumps(str(src))[1:-1]))
    now = json.loads(json.dumps(cut.to_dict(), ensure_ascii=False))
    _holds(was, now, "cut")
    assert BU.digest(pdir) == before, "reading a project writes nothing"


def _holds(was, now, where: str) -> None:
    """Every value 1.22.2 saved is read back as it was. A later version may know more (a new slider at its
    default: 1.23's body skin values, off), never less and never another value."""
    if isinstance(was, dict):
        assert isinstance(now, dict), f"{where}: no longer a group of values"
        for k, v in was.items():
            assert k in now, f"{where}.{k}: a value 1.22.2 saved is no longer read"
            _holds(v, now[k], f"{where}.{k}")
    else:
        assert now == was, f"{where}: read back as {now!r}, 1.22.2 had {was!r}"


def test_new_values_of_a_later_version_are_off_in_an_old_project(old_project):
    """What a version after 1.22.2 added to a person's values starts switched off in a project made before it,
    and a project that never used it is written without it (1.23: body skin)."""
    from contour.core.project import Project
    from contour.core.params import BODY_SKIN_KEYS
    pdir, _src = old_project
    cut = Project.load(str(pdir)).cuts[0]
    for pp in cut.params.values():
        assert all(pp.amounts[k].base == 0.0 and not pp.amounts[k].keys for k in BODY_SKIN_KEYS)
        assert not pp.body_on("body_skin") and not pp.body_on("body_blemish")
        assert pp.body_scope == "all" and pp.body_marks == [] and pp.body_memo == {}
        d = pp.to_dict()
        assert not [k for k in d if k.startswith("body_") and k not in ("body_sculpt", "body_sides")], \
            "an unused body skin setting is not written into the project"


def test_existing_project_opens_in_a_session_without_writing(old_project):
    from contour.core.project import Project
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    from contour.ui.session import EditSession
    pdir, src = old_project
    before = BU.digest(pdir)
    prj = Project.load(str(pdir))
    s = EditSession()
    s.project = prj
    s.open_cut(prj.cuts[0])
    try:
        tr = s.tracker
        assert s.proxy is not None and s.proxy_ready() and tr is not None and tr.ready
        assert tr.recovered == "" and tr.store.matte_rev == 0 and MRV.revision(tr) == 0
        assert tr.has_seg() and (tr.matte.w, tr.matte.h) == (BU.MW, BU.MH)
        assert s.frame(0).shape == (BU.H, BU.W, 3) and s.result(2, final=True) is not None
        assert MR.stale(tr) == [], "nothing recorded with a 1.22.2 matte: nothing is called stale by itself"
        assert BU.digest(pdir) == before, "opening and looking writes nothing into the project"
    finally:
        s.open_cut(None)
    assert BU.digest(pdir) == before


def test_existing_project_renders_what_1222_rendered(old_project, golden):
    from contour.core.project import Project
    pdir, src = old_project
    before = BU.digest(pdir)
    prj = Project.load(str(pdir))
    with BU.NoModels():
        planes = BU.render_planes(prj, prj.cuts[0])
    g = golden["planes"]
    assert planes.shape == g.shape and planes.dtype == g.dtype
    d = np.abs(planes.astype(np.int32) - g.astype(np.int32))
    where = np.unravel_index(int(d.argmax()), d.shape)
    assert d.max() <= PLANE_TOL, f"max difference {d.max()} of 1023 at frame/plane/y/x {where}, mean {d.mean():.4f}"
    assert d.mean() <= 0.05, f"mean difference {d.mean():.4f}"
    if _same_env(golden):
        assert np.array_equal(planes, g), f"same libraries as the fixture: {int((d > 0).sum())} values differ"
    left = {k: v for k, v in BU.digest(pdir).items() if not k.startswith("out/")}
    assert left == before, "a render writes nothing into the project but its output"


def test_existing_project_previews_what_1222_showed(old_project, golden):
    from contour.core.project import Project
    pdir, src = old_project
    prj = Project.load(str(pdir))
    with BU.NoModels():
        prev = BU.preview_frames(prj, prj.cuts[0], [int(t) for t in golden["frames"]])
    g = golden["preview"]
    assert prev.shape == g.shape
    d = np.abs(prev.astype(np.int32) - g.astype(np.int32))
    assert d.max() <= PREVIEW_TOL, f"max difference {d.max()} of 255, mean {d.mean():.4f}"
    if _same_env(golden):
        assert np.array_equal(prev, g)
    # and it is a retouched, reshaped picture (the fixture would hold nothing otherwise)
    srcf = BU.frames_rgb()[[int(t) for t in golden["frames"]]]
    assert np.abs(g.astype(np.int32) - srcf.astype(np.int32)).max() > 40


def _pump(app, sec=0.3):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)


def test_window_opens_existing_project_as_before(qapp, old_project, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    pdir, src = old_project
    before = BU.digest(pdir)
    settings().ask_proxy = False
    asked = []
    for name in ("question", "warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: (asked.append((_n, a[1:3])),
                                                                                    QMessageBox.No)[1]))
    w = MW.MainWindow()
    errors = []
    monkeypatch.setattr(w, "show_error", lambda title, text: errors.append((title, text)))
    nm = BU.NoModels()
    nm.__enter__()                  # (a face layer is made for old cuts only where the parser model is installed)
    try:
        w.cache_worker.enabled = False
        _pump(qapp, 0.3)            # the window's own start-up first, as when a person opens a project
        w.open_project(str(pdir), wait=True)
        _pump(qapp, 0.5)
        s = w.session
        cut = s.cut
        assert cut is not None and cut.analyzed and s.tracker is not None and s.tracker.ready
        assert w.viewer.message == "" and not errors and not asked and w.job is None
        assert [p.id for p in cut.persons] == [0, 1] and w.sel_pid == 0
        # the first picture comes from the loader thread; the result of the frame from the preview thread
        t0 = time.time()
        while (w.viewer._qimg("orig") is None or w.viewer._qimg("result") is None
               or getattr(w.viewer, "_res_pending", False)) and time.time() - t0 < 20:
            _pump(qapp, 0.05)
        assert w.viewer._qimg("orig") is not None and not w.viewer._res_pending
        w.edit_worker.drain(10)
        _pump(qapp, 0.3)
        assert not s.dirty, "opening an existing project changes nothing that would have to be saved"
        now = BU.digest(pdir)
        added = sorted(k for k in now if k not in before)
        assert all(Path(k).name in BU.ALLOWED_NEW or k.startswith("timeline/thumbs/") for k in added), added
        assert all(now[k] == before[k] for k in before), [k for k in before if now.get(k) != before[k]]
    finally:
        nm.__exit__(None, None, None)
        w.refresher.cancel()
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()


def test_saving_keeps_everything_1222_wrote(old_project, golden):
    """The project saved by the code of now still holds every key and value 1.22.2 wrote (new keys may be added:
    an older build ignores them)."""
    from contour.core.project import Project
    from contour.track.store import TrackStore
    pdir, src = old_project
    was_prj = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    cut_dir = next((pdir / "cuts").iterdir())
    with np.load(cut_dir / "tracks.npz") as d:
        was_tracks = {k: d[k].copy() for k in d.files}
    prj = Project.load(str(pdir))
    prj.save()
    st = TrackStore.load(cut_dir / "tracks.npz")
    st.save(cut_dir / "tracks.npz")
    now_prj = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    for k, v in was_prj.items():
        if k == "modified":
            continue
        if k == "cuts":
            for c0, c1 in zip(v, now_prj["cuts"]):
                _holds(c0, c1, "cut")
        elif k == "learn":
            # the one value that is not carried over (instruction 2026-11-20 §4-4): 1.22.2 wrote true for every
            # project (its default, never a choice); learning is now off unless switched on by hand, and the file
            # says so for an older build that opens it too
            assert v is True and now_prj["learn"] is False and now_prj["learn_set"] is False
        else:
            assert now_prj[k] == v, k
    with np.load(cut_dir / "tracks.npz") as d:
        for k, v in was_tracks.items():
            assert k in d.files and np.array_equal(d[k], v, equal_nan=True), f"tracks.{k}"


# ------------------------------------------------------------------------------------------------- body regions chosen by hand
# Instruction 2026-11-20 §2: "기존 프로젝트의 명시적인 몸 영역 선택은 마이그레이션 후에도 같은 결과를 유지". The second
# fixture (tools/make_baseline_fixture.py --body) is the same person with bare shoulders, saved by 1.22.2 with the skin
# retouch's regions 어깨·쇄골, 팔, 손, 다리 chosen besides the face and neck. 1.23 adds the body skin effects beside
# them; the regions of before keep working as they did, and nothing is moved over to the new ones.
FIX_BODY = Path(__file__).resolve().parent / "data" / "baseline_1222_body"


@pytest.fixture
def golden_body():
    with np.load(FIX_BODY / "golden.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


@pytest.fixture
def old_body_project(tmp_path):
    return BU.copy_fixture(FIX_BODY, tmp_path)


def test_body_fixture_has_the_regions_and_they_did_something_in_1222(golden_body):
    assert str(golden_body["version"]) == "1.22.2"
    cut = json.loads(str(golden_body["cut_json"]))
    assert cut["params"]["0"]["skin_scope"] == BU.BODY_REGIONS
    d = np.abs(golden_body["preview"].astype(np.int32) - golden_body["preview_face"].astype(np.int32)).max(-1)
    assert int((d > 0).sum()) > 200 and int(d.max()) >= 10, "the fixture's body regions change nothing: it would hold nothing"
    # and where: on the bare skin below the face (it begins at row 87 of the frames), not on the face
    ys = np.nonzero(d[0] > 0)[0]
    assert float((ys >= 84).mean()) > 0.9 and int(ys.min()) > 70, "the difference of the two is below the face"


def test_explicit_body_regions_render_what_1222_rendered(old_body_project, golden_body):
    from contour.core.project import Project
    pdir, src = old_body_project
    prj = Project.load(str(pdir))
    pp = prj.cuts[0].params_for(0)
    assert pp.skin_scope == BU.BODY_REGIONS
    assert not pp.body_on("body_skin") and not pp.body_on("body_blemish") and pp.body_scope == "all", \
        "the regions of before are not turned into the new effects"
    with BU.NoModels():
        planes = BU.render_planes(prj, prj.cuts[0])
    g = golden_body["planes"]
    assert planes.shape == g.shape and planes.dtype == g.dtype
    d = np.abs(planes.astype(np.int32) - g.astype(np.int32))
    where = np.unravel_index(int(d.argmax()), d.shape)
    assert d.max() <= PLANE_TOL, f"max difference {d.max()} of 1023 at frame/plane/y/x {where}, mean {d.mean():.4f}"
    assert d.mean() <= 0.05, f"mean difference {d.mean():.4f}"
    if _same_env(golden_body):
        assert np.array_equal(planes, g), f"same libraries as the fixture: {int((d > 0).sum())} values differ"


def test_explicit_body_regions_preview_what_1222_showed(old_body_project, golden_body):
    from contour.core.project import Project
    pdir, src = old_body_project
    prj = Project.load(str(pdir))
    with BU.NoModels():
        prev = BU.preview_frames(prj, prj.cuts[0], [int(t) for t in golden_body["frames"]])
    g = golden_body["preview"]
    assert prev.shape == g.shape
    d = np.abs(prev.astype(np.int32) - g.astype(np.int32))
    assert d.max() <= PREVIEW_TOL, f"max difference {d.max()} of 255, mean {d.mean():.4f}"
    if _same_env(golden_body):
        assert np.array_equal(prev, g)


def test_explicit_body_regions_are_saved_as_they_were(old_body_project, golden_body):
    """Opened and saved by the code of now: the regions are the ones 1.22.2 wrote, and no body skin value was added
    for a project that does not use them."""
    from contour.core.project import Project
    pdir, src = old_body_project
    prj = Project.load(str(pdir))
    prj.save()
    now = json.loads((pdir / "project.json").read_text(encoding="utf-8"))["cuts"][0]["params"]["0"]
    assert now["skin_scope"] == BU.BODY_REGIONS
    assert not [k for k in now if k.startswith("body_") and k not in ("body_sculpt", "body_sides")]
    assert now["amounts"]["body_skin"]["base"] == 0 and now["amounts"]["body_blemish"]["base"] == 0
