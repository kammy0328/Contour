"""1.22.1 project opening: the cut list and the timeline are usable as soon as project.json is read; thumbnails
come afterwards from one shared background cache. A slow thumbnail read never holds the opening, the cut list and
the timeline never decode the same JPG twice, a thumbnail of the project opened before never shows in the one
opened now, and opening again reads the saved thumbnails instead of making them again."""
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from conftest import make_video

N = 120


def _pump(app, sec=0.3):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)


def _settle(app, win, sec=30.0):
    t0 = time.time()
    tc = win.thumbcache
    while time.time() - t0 < sec:
        _pump(app, 0.1)
        if tc.idle():
            _pump(app, 0.2)                 # the views' queued requests and the last pictures' signals
            if tc.idle():
                return
    raise AssertionError("thumbnails never settled")


def _big_project(tmp_path, name, colour, n=N, ids=None, timeline=True):
    from contour.core import media as MD
    from contour.core.project import Project
    from contour.core.tlimport import save_thumb
    v = tmp_path / f"{name}.mp4"
    if not v.exists():
        make_video(v)
    prj = Project.create(str(tmp_path), name)
    info = MD.probe(str(v))
    for i in range(n):
        c = prj.add_cut(info, f"c{i:03d}")
        if ids is not None:
            c.id = ids[i]
        if timeline:
            c.tl = {"kind": "clip", "index": i, "start": i * 6, "end": i * 6 + 6}
    prj.save()
    if colour is not None:
        im = np.zeros((54, 96, 3), np.uint8)
        im[:] = colour
        for c in prj.cuts:
            save_thumb(prj, c.id, im)
    return prj


@pytest.fixture
def make_win(qapp, monkeypatch):
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    settings().ask_proxy = False
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    made = []

    def make():
        w = MW.MainWindow()
        w.errors = []
        monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append((title, text)))
        made.append(w)
        return w
    yield make
    for w in made:
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()


@pytest.fixture
def counted(monkeypatch):
    """load_thumb (the JPG read and decode) counted per cut, and slow."""
    from contour.core import tlimport as TI
    real = TI.load_thumb
    calls = {}
    delay = {"s": 0.0}

    def load(project, cut_id):
        # the cut list and timeline (UI thread before 1.22.1, the shared worker since); the project manager's
        # card picture (its own reader of one cut) is another page
        if threading.current_thread().name in ("MainThread", "diap-thumbs"):
            calls[cut_id] = calls.get(cut_id, 0) + 1
        if delay["s"]:
            time.sleep(delay["s"])
        return real(project, cut_id)
    monkeypatch.setattr(TI, "load_thumb", load)
    return calls, delay


def test_slow_thumbnails_never_hold_the_opening(make_win, qapp, tmp_path, counted):
    calls, delay = counted
    prj = _big_project(tmp_path, "Big", (200, 30, 30))
    delay["s"] = 0.08                       # 120 cuts: about 10 s of reading, were it done before opening
    win = make_win()
    t0 = time.perf_counter()
    win.open_project(prj.path, wait=True)
    took = time.perf_counter() - t0
    assert win.session.project is not None and win.session.cut is not None
    assert win.cutlist.list.count() == N and win.tlstrip.list.count() == N
    assert took < 4.0, f"opening waited for the thumbnails: {took:.2f}s"
    assert sum(calls.values()) < N, "thumbnails are not read before the project is shown"
    assert set(win.open_times) >= {"load", "list", "first_cut"}
    delay["s"] = 0.0
    _settle(qapp, win)
    assert len(win.thumbcache.images) == N


def test_both_views_share_one_read_per_jpg(make_win, qapp, tmp_path, counted):
    calls, _ = counted
    prj = _big_project(tmp_path, "Share", (30, 200, 30))
    win = make_win()
    win.open_project(prj.path, wait=True)
    _settle(qapp, win)
    win.tlstrip.refresh()
    win.cutlist.refresh(win.session.cut.id)
    _settle(qapp, win)
    assert len(calls) == N
    assert max(calls.values()) == 1, f"a JPG was read more than once: {max(calls.values())}"
    assert win.cutlist.thumbs is win.tlstrip.thumbs
    some = prj.cuts[5].id
    assert win.cutlist.thumbs[some] is win.tlstrip.thumbs[some]


def test_switching_project_mid_thumbnails_never_mixes(make_win, qapp, tmp_path, counted):
    calls, delay = counted
    a = _big_project(tmp_path, "A", (220, 0, 0))
    ids = [c.id for c in a.cuts]
    b = _big_project(tmp_path, "B", (0, 0, 220), ids=ids)      # the same cut ids: only the opening tells them apart
    delay["s"] = 0.03
    win = make_win()
    win.open_project(a.path, wait=True)
    _pump(qapp, 0.4)                        # A's thumbnails are coming in
    assert 0 < sum(calls.values()) < N
    win.open_project(b.path, wait=True)
    delay["s"] = 0.0
    _settle(qapp, win)
    imgs = win.thumbcache.images
    assert len(imgs) == N
    for cid, q in imgs.items():
        c = q.pixelColor(10, 10)
        assert c.blue() > 150 and c.red() < 60, f"{cid}: a thumbnail of the project opened before"


def test_closing_mid_thumbnails_leaves_nothing(make_win, qapp, tmp_path, counted):
    calls, delay = counted
    a = _big_project(tmp_path, "C", (220, 0, 0))
    delay["s"] = 0.03
    win = make_win()
    win.open_project(a.path, wait=True)
    _pump(qapp, 0.3)
    win.close_project()
    delay["s"] = 0.0
    _pump(qapp, 0.8)
    assert win.thumbcache.images == {}


def test_reopen_reads_what_the_first_opening_made(make_win, qapp, tmp_path, counted):
    """Plain cuts (no timeline JPG): the first opening makes the thumbnails of the cuts on screen from the original
    and saves them; opening again shows the same pictures from the saved JPGs without decoding the original."""
    calls, _ = counted
    prj = _big_project(tmp_path, "Plain", None, n=6, timeline=False)
    win = make_win()
    win.open_project(prj.path, wait=True)
    _settle(qapp, win)
    tc = win.thumbcache
    assert tc.made == 6 and len(tc.images) == 6
    first = {cid: np.array([q.pixelColor(x, 20).red() for x in (5, 40, 90)]) for cid, q in tc.images.items()}
    from contour.core.tlimport import thumbs_dir
    assert all((thumbs_dir(prj) / f"{c.id}.jpg").exists() for c in prj.cuts)
    win.close_project()
    win2 = make_win()
    win2.open_project(prj.path, wait=True)
    _settle(qapp, win2)
    tc2 = win2.thumbcache
    assert tc2.made == 0, "the saved thumbnails were made again"
    assert set(tc2.images) == set(first)
    for cid, q in tc2.images.items():
        again = np.array([q.pixelColor(x, 20).red() for x in (5, 40, 90)])
        assert np.abs(again - first[cid]).max() <= 8          # JPG round trip


def test_changed_colour_view_remakes_only_that_cut(make_win, qapp, tmp_path, counted):
    prj = _big_project(tmp_path, "View", None, n=4, timeline=False)
    win = make_win()
    win.open_project(prj.path, wait=True)
    _settle(qapp, win)
    tc = win.thumbcache
    assert tc.made == 4
    open_prj = win.session.project
    c = open_prj.cuts[2]
    c.input_cs = "ACEScct"                  # another colour view for this cut only
    tc.refresh(c.id)
    for other in open_prj.cuts:
        if other is not c:
            tc.refresh(other.id, front=False)
    _settle(qapp, win)
    assert tc.made == 5, "only the cut whose view changed was made again"


def test_no_thumbnail_reader_on_the_ui_thread_in_timeline_refresh(make_win, qapp, tmp_path, counted):
    calls, _ = counted
    prj = _big_project(tmp_path, "Strip", (10, 10, 10), n=20)
    win = make_win()
    win.open_project(prj.path, wait=True)
    _settle(qapp, win)
    before = sum(calls.values())
    win.thumbcache.drop()
    win.tlstrip.refresh()                   # used to read every JPG here, on the UI thread
    assert sum(calls.values()) == before
