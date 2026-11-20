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
    return str(golden["env"]) == BU.env_key()


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
    for k, v in was.items():
        assert k in now, f"{k}: a value 1.22.2 saved is no longer read"
        assert now[k] == v, f"{k}: read back as {now[k]!r}, 1.22.2 had {v!r}"
    assert BU.digest(pdir) == before, "reading a project writes nothing"


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
                for kk, vv in c0.items():
                    assert kk in c1 and c1[kk] == vv, f"cut.{kk}"
        else:
            assert now_prj[k] == v, k
    with np.load(cut_dir / "tracks.npz") as d:
        for k, v in was_tracks.items():
            assert k in d.files and np.array_equal(d[k], v, equal_nan=True), f"tracks.{k}"
