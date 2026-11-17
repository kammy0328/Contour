"""1.18 original-first: a project with no proxy at all opens its cuts on the originals at once (no proxy job, no
proxy file), keeps the analysis size of cuts analysed on a proxy, renders from the original, and a proxy made in
the background takes over without changing frames or coordinates. Old proxies (own and the bake's shared one)
are still used."""
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from conftest import make_video


def _pump(app, sec=0.3):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)


def _proxy_files(prj):
    out = []
    for c in prj.cuts:
        out += list(Path(prj.proxy_dir(c)).glob("proxy_*.u8"))
    return out


def test_session_opens_on_original(project):
    from contour.core.colour import manager
    from contour.core.frames import proxy_for
    from contour.ui.session import EditSession
    from contour.warp.render import _vs_for
    s = EditSession()
    s.project = project
    c = project.cuts[0]
    s.open_cut(c)
    assert s.proxy is not None and s.proxy.kind == "source" and s.tracker is not None
    assert s.proxy_ready()
    f = s.frame(3)
    assert f.shape == (s.proxy.h, s.proxy.w, 3)
    assert _proxy_files(project) == [], "opening a cut makes no proxy"
    ref = proxy_for(project, c)
    assert ref.build(manager(), _vs_for(c))
    assert np.array_equal(f, ref.frame(3))
    # a proxy in this view now exists: the next opening reads it
    s.open_cut(c)
    assert s.proxy.kind == "proxy"
    s.proxy.close()


def test_view_change_reopens_without_a_proxy(project):
    from contour.ui.session import EditSession
    s = EditSession()
    s.project = project
    c = project.cuts[0]
    s.open_cut(c)
    k0 = s.proxy.view_key
    c.lut_path = ""
    c.input_cs = "Linear Rec.709 (sRGB)" if c.input_cs != "Linear Rec.709 (sRGB)" else ""
    assert not s.proxy_ready() or s.proxy.view_key == s.view_settings().key()
    s.open_cut(c)
    assert s.proxy.kind == "source" and s.proxy_ready()
    assert s.proxy.view_key == s.view_settings().key()
    assert _proxy_files(project) == []


def test_analysis_size_kept_from_mattes(project):
    """A cut analysed on a 1920 proxy whose proxy was cleared, opened with the setting now 1280: still 1920."""
    from contour.core import frames as FR
    from contour.core.settings import settings
    c = project.cuts[0]
    w, h = 64, 48                                   # proxy size of this small source at any setting
    mw, mh = w // 2, h // 2
    (Path(project.cache_dir(c)) / f"matte_{mw}x{mh}.u8").write_bytes(b"\0" * (c.info.frames * mw * mh))
    settings().proxy_long_side = 32                 # would make 32x24
    assert FR.frame_size(project, c) == (w, h)
    assert FR.proxy_for(project, c).w == w
    fr = FR.frames_for(project, c)
    assert (fr.w, fr.h) == (w, h)
    c2 = project.cuts[1]                            # never analysed: the setting decides
    assert FR.frame_size(project, c2) == (32, 24)


def test_old_proxy_still_used_and_original_missing(project, tmp_path):
    from contour.core import frames as FR
    from contour.core.colour import manager
    from contour.warp.render import _vs_for
    c = project.cuts[0]
    px = FR.proxy_for(project, c)
    assert px.build(manager(), _vs_for(c))
    assert FR.frames_for(project, c).kind == "proxy"
    # the original goes missing, the colour view changes: the old proxy can still be looked at
    Path(c.info.path).rename(tmp_path / "moved.mp4")
    c.input_cs = "Linear Rec.709 (sRGB)"
    fr = FR.frames_for(project, c)
    assert fr is not None and fr.kind == "proxy" and not fr.is_valid(_vs_for(c).key())
    # a cut with neither
    c2 = project.cuts[1]
    Path(c2.info.path).rename(tmp_path / "moved2.mp4")
    assert FR.frames_for(project, c2) is None


def test_render_from_original_without_proxy(project, tmp_path):
    """The render queue's path: no proxy file, no copy of the original instead of a render."""
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    from contour.warp.render import render_cut
    c = project.cuts[0]
    c.output.out_dir = str(tmp_path / "out")
    c.output.format = "prores_422_hq"
    tr = CutTracker(project, c, frames_for(project, c))
    assert tr.proxy.kind == "source"
    rep = render_cut(project, c, tr, c.output)
    outs = list(Path(tmp_path / "out").rglob("*.mov"))
    assert outs and rep.frames == c.info.frames
    assert _proxy_files(project) == []


def test_timeline_retouch_does_not_copy_for_lack_of_proxy(project, monkeypatch):
    """Before 1.18 a shot with no proxy was put in the export as its original, with a warning. Now it is rendered."""
    from contour.warp import tlrender as TL
    called = {}
    monkeypatch.setattr(TL, "_copy", lambda *a, **k: 0)          # the frames outside the work range

    import contour.warp.render as R
    c = project.cuts[0]

    class Rep:
        frames, cancelled, warnings = c.work_range[1] - c.work_range[0], False, []

    def fake(prj, cut, tr, o, *a, **k):
        called["render"] = tr.proxy.kind
        return Rep()
    monkeypatch.setattr(R, "render_cut", fake)
    rep = TL.TimelineReport()
    out = type("O", (), {"fmt": "yuv422p10le"})()
    TL._retouch(project, out, c, lambda: False, lambda: None, rep, lambda m: None)
    assert called.get("render") == "source"
    assert not any("프록시" in w for w in getattr(rep, "warnings", []))


def test_background_proxy_swaps_in_same_frames(project):
    from contour.core.colour import manager
    from contour.core.frames import frames_for, proxy_for
    from contour.ui.session import EditSession
    s = EditSession()
    s.project = project
    c = project.cuts[0]
    s.open_cut(c)
    before = s.frame(4)
    tr = s.tracker
    assert proxy_for(project, c).build(manager(), s.view_settings(c))
    new = frames_for(project, c, s.view_settings(c))
    assert new.kind == "proxy"
    assert s.swap_frames(c.id, new)
    assert s.proxy is new and tr.proxy is new and s.tracker is tr
    assert np.array_equal(s.frame(4), before)
    assert not s.swap_frames("other", new)
    new.close()


@pytest.fixture
def win(qapp, project, monkeypatch):
    from contour.core.settings import settings
    settings().ask_proxy = False
    from contour.ui import main_window as MW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.errors = []
    monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append((title, text)))
    jobs = []
    orig = w.run_job
    monkeypatch.setattr(w, "run_job", lambda title, *a, **k: (jobs.append(title), orig(title, *a, **k)))
    w.jobs = jobs
    w.open_project(project.path, wait=True)
    _pump(qapp, 0.5)
    yield w
    w.proxy_builder.cancel()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_window_opens_cut_without_proxy_job(win, qapp, project):
    from contour.core.settings import settings
    assert win.session.cut is not None
    assert win.session.from_source()
    assert not any("프록시" in t for t in win.jobs), f"no proxy job: {win.jobs}"
    assert win.job is None
    other = next(c for c in win.session.project.cuts if c is not win.session.cut)
    win.open_cut(other.id)
    _pump(qapp, 0.3)
    assert win.session.cut is other and win.session.from_source() and win.job is None
    assert win.session.frame(2) is not None
    if settings().proxy_mode != "always":
        assert _proxy_files(win.session.project) == []


def test_window_always_mode_builds_in_background_and_swaps(win, qapp):
    from contour.core.settings import settings
    settings().proxy_mode = "always"
    try:
        c = win.session.cut
        other = next(x for x in win.session.project.cuts if x is not c)
        win.open_cut(other.id)
        t0 = time.time()
        while win.session.from_source() and time.time() - t0 < 20:
            _pump(qapp, 0.1)
        assert win.session.proxy.kind == "proxy", "the background proxy took over"
        assert win.job is None
    finally:
        settings().proxy_mode = "auto"


def test_window_missing_original_says_so(win, qapp, tmp_path):
    c = next(x for x in win.session.project.cuts if x is not win.session.cut)
    Path(c.info.path).rename(tmp_path / "gone.mp4")
    win.open_cut(c.id)
    _pump(qapp, 0.2)
    assert win.session.proxy is None
    assert "원본" in win.viewer.message
    assert win.job is None


def test_project_close_ends_readers(win, qapp):
    """Closing the project leaves no ffmpeg reader of its original running."""
    from contour.core import frames as FR
    s = win.session
    fr = s.proxy
    FR.pool().clear()
    fr.close()
    s._frames.clear()
    s.frame(0)                      # a reader stays open (frames after 0 are still to come)
    procs = [c.reader._proc for c in fr._cursors]
    assert procs
    assert win.close_project()
    for p in procs:
        assert p.poll() is not None


def test_background_proxy_without_space_changes_nothing(win, qapp, monkeypatch):
    """No room for a proxy: the cut keeps reading its original, no error box."""
    from contour.core import proxy as PX
    from contour.core.settings import settings

    def full(*a, **k):
        raise PX.DiskSpaceError("공간 부족")
    monkeypatch.setattr(PX, "ensure_space", full)
    settings().proxy_mode = "always"
    try:
        other = next(x for x in win.session.project.cuts if x is not win.session.cut)
        win.open_cut(other.id)
        t0 = time.time()
        while win.proxy_builder.busy and time.time() - t0 < 10:
            _pump(qapp, 0.1)
        _pump(qapp, 0.3)
        assert win.session.from_source() and win.session.frame(1) is not None
        assert not win.errors
    finally:
        settings().proxy_mode = "auto"


def test_reopen_same_cut_is_warm_and_fresh(project):
    """Reopening a cut finds its decoded frames again (no new decode) under a new epoch (no stale results)."""
    from contour.core import frames as FR
    from contour.ui.session import EditSession
    FR.pool().clear()
    s = EditSession()
    s.project = project
    c = project.cuts[0]
    s.open_cut(c)
    a = s.frame(2)
    e0 = s.epoch
    s.open_cut(c)
    assert s.epoch != e0
    d0 = s.proxy.decoded
    assert np.array_equal(s.frame(2), a)
    assert s.proxy.decoded == d0, "served from the pool"
    s.open_cut(None)


def test_viewer_original_frame_comes_from_loader_thread(qapp, project):
    """The viewer never decodes on the UI thread: the first call asks the loader, the picture comes back from it."""
    from contour.core import frames as FR
    from contour.ui.session import EditSession
    from contour.ui.viewer import Viewer
    FR.pool().clear()
    s = EditSession()
    s.project = project
    s.open_cut(project.cuts[0])
    v = Viewer(s)
    try:
        v.t = 3
        assert v._qimg("orig") is None, "nothing decoded on this thread"
        t0 = time.time()
        img = None
        while img is None and time.time() - t0 < 10:
            _pump(qapp, 0.05)
            img = v._img_cache.get(("orig", 3, project.cuts[0].id))
        assert img is not None and v._qimg("orig") is img
    finally:
        v.stop_workers()
        s.open_cut(None)



def test_auto_mode_wants_proxy_after_scrubbing(project):
    """'auto': a cut read from its original gets a proxy once the user keeps jumping around it (each jump restarts
    a decoder), not for a cut only played through."""
    from contour.core import frames as FR
    from contour.core.settings import settings
    settings().proxy_mode = "auto"
    fr = FR.frames_for(project, project.cuts[0])
    assert not FR.wants_proxy(fr, 24.0)
    fr.seeks = FR.SCRUB_SEEKS
    assert FR.wants_proxy(fr, 24.0)
    settings().proxy_mode = "off"
    assert not FR.wants_proxy(fr, 24.0)
    fr.close()
