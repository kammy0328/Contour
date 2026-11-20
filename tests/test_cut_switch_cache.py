"""Instruction 2026-11-20 §1-3: cut switching and proxies.

Until 1.22.2 the playback cache belonged to one opening of one cut: its keys carried the number of that opening,
so A -> B -> A rendered A again from nothing, and each cut prepared ahead for playback had 1.2 GB of its own.

- A -> B -> A finds A's finished frames again, all cuts within one memory budget;
- keys say what a result is made from (project, cut, source file and its version, view, settings, processing
  revisions); the number of an opening only drops what was started before a cut was changed;
- a job that finishes right after a switch (also A -> B -> A) is never kept or shown;
- the original replaced, a setting or the view changed, values or analysis files changed while the cut was not
  open: nothing kept for the old state is used;
- the proxy on disk and the frames kept in memory are checked each by their own state: a valid proxy opens without
  being made again; a cut without one can be scrubbed at once; making and failing are said without a dialog."""
import os
import shutil
import threading
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


def _session(project):
    from contour.ui.session import EditSession
    s = EditSession()
    s.project = project
    return s


def _fill(s, frames):
    for t in frames:
        assert s.play_render(t, near=frames[0])
    assert all(s.play_has(t) for t in frames)


# ---------------------------------------------------------------------------------------------- A -> B -> A
def test_a_b_a_finds_the_frames_again_under_a_new_generation(project):
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    ep0, key0 = s.epoch, s.result_key(2, True)
    _fill(s, [0, 1, 2, 3])
    pic = s.play_get(2)
    s.open_cut(b)
    assert s.epoch != ep0 and not s.play_has(2), "B has none of A's frames"
    _fill(s, [1, 2])
    assert s.result_key(2, True) != key0
    s.open_cut(a)
    assert s.epoch not in (ep0,) and s.result_key(2, True) == key0, "same state: same key, another generation"
    assert s.kept == 4 and sorted(s.play_frames()) == [0, 1, 2, 3]
    assert np.array_equal(s.play_get(2), pic)
    assert key0[-2] == (str(project.path), a.id) and key0[-1] == s.src_key
    # B's frames are still there as well, in the same store
    assert set(PC.cache().cuts()) == {(str(project.path), a.id), (str(project.path), b.id)}
    s.open_cut(b)
    assert sorted(s.play_frames()) == [1, 2]
    s.open_cut(None)


def test_one_budget_for_all_cuts(project):
    """The cut used longest ago gives up its frames first (the furthest from where it was looked at), then the
    open cut the ones furthest from where it is looked at; together never more than the limit."""
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    _fill(s, [0, 1, 2, 3, 4, 5])
    one = PC.cache().size(s.cut_key) // 6
    PC.cache().set_limit(int(one * 8.5))                # room for 8 frames of the two cuts together
    assert s.play_limit == int(one * 8.5)
    s.open_cut(b)
    for t in range(6):
        assert s.play_render(t, near=0)
        assert PC.cache().bytes <= PC.cache().limit
    assert sorted(s.play_frames()) == [0, 1, 2, 3, 4, 5], "the open cut keeps its frames"
    ka = (str(project.path), a.id)
    left = sorted(t for t, _ in PC.cache().frames(ka))
    assert 0 < len(left) < 6 and left == list(range(len(left))), f"A lost its far frames first: {left}"
    s.open_cut(a)
    assert sorted(s.play_frames()) == left
    s.open_cut(None)


def test_stale_frames_of_the_open_cut_leave_before_anything_else(project):
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(b)
    _fill(s, [0, 1])
    s.open_cut(a)
    _fill(s, [0, 1, 2])
    one = PC.cache().size(s.cut_key) // 3
    s.touch("ret")                                     # A's three frames are of old values now
    s.save()
    assert s.play_frames() == []
    PC.cache().set_limit(int(one * 5.5))
    _fill(s, [3, 4])
    assert PC.cache().bytes <= PC.cache().limit
    assert sorted(t for t, _ in PC.cache().frames((str(project.path), b.id))) == [0, 1], "B untouched"
    assert sorted(s.play_frames()) == [3, 4]
    s.open_cut(None)


# ---------------------------------------------------------------------------------------------- late jobs
def _blocked_render(s, t, monkeypatch):
    """Starts play_render(t) on a thread and holds it inside the retouch step (after its frame was read)."""
    from contour.ui.session import EditSession
    entered, release, out = threading.Event(), threading.Event(), {}
    orig = EditSession.retouch

    def held(self, tt):
        if self is s and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return orig(self, tt)
    monkeypatch.setattr(EditSession, "retouch", held)
    th = threading.Thread(target=lambda: out.setdefault("ok", s.play_render(t)), daemon=True)
    th.start()
    assert entered.wait(5)
    return th, release, out


def test_job_finishing_right_after_a_switch_is_dropped(project, monkeypatch):
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    th, release, out = _blocked_render(s, 3, monkeypatch)
    s.open_cut(b)                                      # the switch, while A's frame 3 is being made
    release.set()
    th.join(10)
    assert out["ok"] is False
    assert PC.cache().bytes == 0, "nothing kept: not for A, not for B"
    assert s.cached_result(3) is None and not s.play_has(3)
    assert not any(k[0] in ("ret_img", "res", "retouch", "field") for k in s._cache), s._cache.keys()
    # what B shows for frame 3 is B's own frame
    assert np.array_equal(s.result(3), s.frame(3))
    s.open_cut(None)


def test_job_of_an_earlier_opening_is_dropped_after_a_b_a(project, monkeypatch):
    """Back on A before the job started in A's first opening ends: its key is A's current key again, but part of
    it may have been read while B was open. The generation tells: nothing of it is kept."""
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    key = s.result_key(3, True)
    th, release, out = _blocked_render(s, 3, monkeypatch)
    s.open_cut(b)
    s.open_cut(a)
    assert s.result_key(3, True) == key, "the same state, so the same key"
    release.set()
    th.join(10)
    assert out["ok"] is False and PC.cache().bytes == 0 and s.cached_result(3) is None
    assert s.play_render(3) and s.play_has(3)          # asked again in this opening: made and kept
    s.open_cut(None)


def test_preview_result_of_an_earlier_opening_is_not_shown(qapp, project, monkeypatch):
    from contour.ui.viewer import Viewer
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    v = Viewer(s)
    try:
        payload = v._compute_preview((2, s.result_key(2)))
        assert payload is not None and payload[3] == s.epoch
        s.open_cut(b)
        s.open_cut(a)
        v._img_cache.clear()
        v._preview_ready(None, payload)                 # arrives now, made in the first opening
        assert not v._img_cache, "a picture of an earlier opening is not taken"
        fresh = v._compute_preview((2, s.result_key(2)))
        v._preview_ready(None, fresh)
        assert ("result", s.result_key(2)) in v._img_cache
    finally:
        v.stop_workers()
        s.open_cut(None)


# ---------------------------------------------------------------------------------------------- what a key is made from
def test_original_replaced_while_the_cut_was_not_open(project, tmp_path):
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    _fill(s, [0, 1, 2])
    old = s.play_get(1)
    key0, src0 = s.result_key(1, True), s.src_key
    s.open_cut(b)
    other = make_video(tmp_path / "other.mp4", n=6)
    data = bytearray(other.read_bytes())
    Path(a.info.path).write_bytes(bytes(data) + b"\0" * 64)     # another file under the same name
    os.utime(a.info.path, (time.time() + 5, time.time() + 5))
    s.open_cut(a)
    assert s.src_key != src0 and s.result_key(1, True) != key0
    assert s.kept == 0 and s.play_frames() == [] and not s.play_has(1)
    assert PC.cache().size(s.cut_key) == 0, "what was kept for the old file is given back"
    assert old is not None
    s.open_cut(None)


def test_view_and_model_settings_are_part_of_the_key(project):
    from contour.core.settings import settings
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    _fill(s, [0, 1])
    k0 = s.result_key(0, True)
    s.open_cut(a)                                      # opened again, nothing changed: found again
    assert s.kept == 2 and s.result_key(0, True) == k0
    a.input_cs = "Linear Rec.709 (sRGB)"               # another colour view
    s.open_cut(a)
    assert s.kept == 0 and s.result_key(0, True) != k0 and not s.play_has(0)
    _fill(s, [0, 1])
    k1 = s.result_key(0, True)
    settings().face_model = "mp" if settings().face_model != "mp" else "face_parse"
    s.open_cut(a)
    assert s.kept == 0 and s.result_key(0, True) not in (k0, k1)
    _fill(s, [0])
    k2 = s.result_key(0, True)
    settings().gpu = "cpu" if settings().gpu != "cpu" else "auto"
    s.open_cut(a)
    assert s.kept == 0 and s.result_key(0, True) != k2
    s.open_cut(None)


def test_values_or_analysis_files_changed_while_away(project):
    from contour.core import playcache as PC
    s = _session(project)
    a, b = project.cuts
    ka = (str(project.path), a.id)
    s.open_cut(a)
    _fill(s, [0, 1])
    s.open_cut(b)
    a.name = "renamed"                                 # a name is nothing a frame is made from
    a.status = dict(a.status or {}, user="good")
    s.open_cut(a)
    assert s.kept == 2
    s.open_cut(b)
    a.params_for(0).amounts["waist"].base = 12.0       # pasted onto it while it was not open
    s.open_cut(a)
    assert s.kept == 0 and PC.cache().size(ka) == 0
    _fill(s, [0, 1])
    s.open_cut(b)
    f = Path(project.cut_dir(a)) / "tracks.npz"        # an analysis (or a batch) wrote its tracks meanwhile
    f.write_bytes(b"x" * 10)
    s.open_cut(a)
    assert s.kept == 0
    _fill(s, [0, 1])
    s.open_cut(b)
    meta = Path(project.cache_dir(a)) / "seg_meta.json"   # the matte was made again (its revision file)
    meta.write_text('{"model": "m", "refreshed": [[0, 2, 1.0]]}', encoding="utf-8")
    s.open_cut(a)
    assert s.kept == 0
    s.open_cut(None)


def test_changes_in_memory_never_saved_take_their_frames_with_them(project):
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    s.touch("geo")
    _fill(s, [0, 1])
    assert s.dirty
    s.open_cut(b)                                      # left without saving (the window never does this)
    s.open_cut(a)
    assert s.kept == 0 and s.play_frames() == []
    s.touch("geo")
    _fill(s, [0, 1])
    s.save()
    s.open_cut(b)
    s.open_cut(a)
    assert s.kept == 2
    s.open_cut(None)


# ---------------------------------------------------------------------------------------------- proxy and memory apart
def test_proxy_on_disk_and_frames_in_memory_each_by_their_own_state(project):
    from contour.core import frames as FR
    from contour.core.colour import manager
    from contour.warp.render import _vs_for
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    assert s.from_source()
    _fill(s, [0, 1, 2])
    pic = s.play_get(1)
    # a proxy is finished in the background and takes over: same pixels, the frames kept stay good
    px = FR.proxy_for(project, a)
    assert px.build(manager(), _vs_for(a))
    px.close()
    made = (px.path.stat().st_mtime_ns, px.path.stat().st_size)
    new = FR.frames_for(project, a, s.view_settings(a))
    assert new.kind == "proxy" and s.swap_frames(a.id, new)
    assert sorted(s.play_frames()) == [0, 1, 2]
    # left and opened again, now on the proxy from the start: the proxy opens as it is, the frames are found
    s.open_cut(b)
    s.open_cut(a)
    assert s.proxy.kind == "proxy" and s.kept == 3 and np.array_equal(s.play_get(1), pic)
    assert (px.path.stat().st_mtime_ns, px.path.stat().st_size) == made, "a valid proxy is not made again"
    # the memory side given up (values changed), the proxy still valid: opened without being made again
    s.touch("ret")
    s.save()
    s.open_cut(b)
    s.open_cut(a)
    assert s.proxy.kind == "proxy" and s.play_frames() == []
    assert (px.path.stat().st_mtime_ns, px.path.stat().st_size) == made
    _fill(s, [0, 1])
    # the proxy gone from the disk, the memory side still good: read from the original, frames found
    s.open_cut(b)
    px.path.unlink()
    px.meta_path.unlink()
    s.open_cut(a)
    assert s.from_source() and s.kept == 2
    s.open_cut(None)


def test_proxy_of_another_view_shown_without_the_original_is_its_own_source(project, tmp_path):
    """The original missing and only an old proxy to look at: what is kept for it is not what the original in
    the current view would give."""
    from contour.core import frames as FR
    from contour.core.colour import manager
    from contour.warp.render import _vs_for
    s = _session(project)
    a, b = project.cuts
    px = FR.proxy_for(project, a)
    assert px.build(manager(), _vs_for(a))
    px.close()
    s.open_cut(a)
    assert s.proxy.kind == "proxy"
    src_with_original = s.src_key
    s.open_cut(b)
    Path(a.info.path).rename(tmp_path / "gone.mp4")
    s.open_cut(a)
    assert s.proxy is not None and s.proxy.kind == "proxy" and s.src_key != src_with_original
    s.open_cut(None)


# ---------------------------------------------------------------------------------------------- next cut prepared
def test_cut_prepared_for_playback_shares_the_store(qapp, project):
    """The next cut rendered ahead goes into the same store; taken over or opened by hand, its frames are there."""
    from contour.core import playcache as PC
    from contour.ui.workers import NextCutPrep
    s = _session(project)
    a, b = project.cuts
    s.open_cut(a)
    _fill(s, [0, 1])
    prep = NextCutPrep()
    prep.start(project, b, head=2)
    t0 = time.time()
    kb = (str(project.path), b.id)
    while len(PC.cache().frames(kb)) < 6 and time.time() - t0 < 15:
        time.sleep(0.02)
    assert len(PC.cache().frames(kb)) == 6
    sh = prep.take(b.id)
    assert sh is not None
    ep = s.epoch
    s.adopt(sh)
    assert s.cut is b and s.epoch != ep and sorted(s.play_frames()) == [0, 1, 2, 3, 4, 5]
    s.open_cut(a)
    assert s.kept == 2
    # prepared, then not taken (playback stopped) and opened by hand: the frames it finished are used
    PC.cache().drop(kb)
    PC.revisions().forget(kb)
    prep2 = NextCutPrep()
    prep2.start(project, b, head=2)
    t0 = time.time()
    while len(PC.cache().frames(kb)) < 6 and time.time() - t0 < 15:
        time.sleep(0.02)
    prep2.cancel()
    t0 = time.time()
    while any(t.name == "diap-nextcut" and t.is_alive() for t in threading.enumerate()) and time.time() - t0 < 5:
        time.sleep(0.02)
    prep2.cancel()
    s.open_cut(b)
    assert s.kept == 6
    s.open_cut(None)


# ---------------------------------------------------------------------------------------------- the window
@pytest.fixture
def win(qapp, project, monkeypatch):
    from contour.core.settings import settings
    settings().ask_proxy = False
    from contour.ui import main_window as MW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.errors = []
    w.toasts = []
    monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append((title, text)))
    orig_toast = w.toast
    monkeypatch.setattr(w, "toast", lambda text, kind="": (w.toasts.append((text, kind)), orig_toast(text, kind))[1])
    w.open_project(project.path, wait=True)
    _pump(qapp, 0.4)
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_window_a_b_a_keeps_the_frames(win, qapp):
    s = win.session
    a = s.cut
    b = next(c for c in s.project.cuts if c is not a)
    win.cache_worker.enabled = False
    _fill(s, [0, 1, 2])
    win.open_cut(b.id)
    _pump(qapp, 0.2)
    assert s.cut is b and not s.play_has(0)
    win.open_cut(a.id)
    _pump(qapp, 0.2)
    assert s.cut is a and s.kept == 3 and sorted(s.play_frames()) == [0, 1, 2]
    assert win.viewer._qimg("result") is not None, "the kept frame shows at once"


def test_cut_being_given_a_proxy_can_be_scrubbed_and_says_so(win, qapp, monkeypatch):
    """A cut whose proxy is still being made: every frame can be asked for at once, the state is a note beside
    the picture; a proxy that fails says so in the same place and with a toast, never with a dialog."""
    from contour.core.settings import settings
    hold, entered = threading.Event(), threading.Event()
    result = {"ok": True, "err": None}

    def build(project, cut, vs, stop):
        report = win.proxy_builder.reporter(str(project.path), cut.id)
        report(0.37)
        entered.set()
        while not hold.is_set() and not stop.is_set():
            time.sleep(0.01)
        if result["err"]:
            raise RuntimeError(result["err"])
        return False if stop.is_set() else result["ok"]
    monkeypatch.setattr(win.proxy_builder, "build", build)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: pytest.fail("a dialog")))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: pytest.fail("a dialog")))
    settings().proxy_mode = "always"
    try:
        other = next(x for x in win.session.project.cuts if x is not win.session.cut)
        result["err"] = "디스크에 쓸 수 없습니다 (테스트)"
        t0 = time.time()
        win.open_cut(other.id)
        assert time.time() - t0 < 2.0, "opening does not wait for the proxy"
        assert entered.wait(5)
        _pump(qapp, 0.2)
        assert win.proxy_builder.busy and win.session.from_source()
        for t in (5, 0, 3, 1):                               # anywhere, at once, while it is being made
            assert win.session.frame(t) is not None
        assert "프록시 만드는 중 37%" in win.px_lbl.text()
        hold.set()
        t0 = time.time()
        while win.proxy_builder.busy and time.time() - t0 < 5:
            _pump(qapp, 0.05)
        _pump(qapp, 0.2)
        assert "프록시 실패" in win.px_lbl.text() and "테스트" in win.px_lbl.toolTip()
        assert any("프록시를 만들지 못했습니다" in t for t, _ in win.toasts)
        assert not win.errors and win.session.from_source() and win.session.frame(2) is not None
        # another cut and back: the note is still this cut's
        first = next(x for x in win.session.project.cuts if x is not other)
        hold.clear()
        entered.clear()
        result["err"] = None
        win.open_cut(first.id)
        assert entered.wait(5)
        _pump(qapp, 0.2)
        assert "프록시 만드는 중" in win.px_lbl.text()
        win.open_cut(other.id)
        _pump(qapp, 0.2)
        assert "프록시 실패" in win.px_lbl.text()
    finally:
        hold.set()
        settings().proxy_mode = "auto"


def test_reopening_for_a_view_change_saves_what_is_in_memory_first(win, qapp, monkeypatch):
    """A LUT or a setting changed with joint edits not saved yet: the cut is opened again from its files, so they
    are written first (1.22.2 read the tracks file again and the edits were gone)."""
    s = win.session
    saved = []
    orig = s.save
    monkeypatch.setattr(s, "save", lambda: (saved.append(1), orig())[1])
    s.touch("trk")
    assert s.dirty
    assert win._reopen_cut()
    assert saved and not s.dirty
    n = len(saved)
    assert win._reopen_cut() and len(saved) == n, "nothing to save: not saved again"


def test_settings_dialog_reopens_when_a_model_setting_changed(win, qapp, monkeypatch):
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    s = win.session
    win.cache_worker.enabled = False
    _fill(s, [0, 1])
    ep = s.epoch

    class Dlg:
        check_update = False

        def __init__(self, parent):
            pass

        def exec(self):
            return True

        def apply(self):
            settings().gpu = "cpu" if settings().gpu != "cpu" else "auto"
            return False                                # the view did not change
    monkeypatch.setattr(MW, "SettingsDialog", Dlg)
    win.open_settings()
    _pump(qapp, 0.2)
    assert s.epoch != ep and s.play_frames() == [], "frames made with the other device are not shown"
