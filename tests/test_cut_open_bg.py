"""1.26.1 (the user 11/26: "컷 넘기거나 다른 컷 선택해서 넘어가는데 약간 프리징"): a cut picked in the list is opened on a thread and
taken over when ready; a cut left without changes is not written again; the page error of a mapped file is gone
(frames read with plain reads)."""
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox


def _pump(app, sec=0.3):
    t0 = time.time()
    while time.time() - t0 < sec:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def win(qapp, project, monkeypatch):
    from contour.core.settings import settings
    settings().ask_proxy = False
    from contour.ui import main_window as MW
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.errors = []
    monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append((title, text)))
    w.open_project(project.path, wait=True)
    _pump(qapp, 0.5)
    yield w
    w.proxy_builder.cancel()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_a_picked_cut_opens_on_a_thread_and_the_window_goes_on(win, qapp, monkeypatch):
    from contour.ui import session as SS
    first = win.session.cut
    other = next(c for c in win.session.project.cuts if c is not first)
    gate = threading.Event()
    orig = SS.EditSession.open_cut
    threads = []

    def slow(self, cut):
        threads.append(threading.current_thread().name)
        if cut is not None and self is not win.session:
            gate.wait(5)                    # a network drive taking its time
        return orig(self, cut)
    monkeypatch.setattr(SS.EditSession, "open_cut", slow)
    t0 = time.perf_counter()
    win.cutlist.cutSelected.emit(other.id)
    assert time.perf_counter() - t0 < 0.5, "the click does not wait for the cut"
    _pump(qapp, 0.2)
    assert win.session.cut is first and "여는 중" in win.viewer.message
    assert not win.inspector.isEnabled()
    gate.set()
    for _ in range(100):
        _pump(qapp, 0.05)
        if win.session.cut is other:
            break
    assert win.session.cut is other and win.session.proxy is not None and win.session.tracker is not None
    assert "diap-opencut" in threads
    assert win.inspector.isEnabled() and "여는 중" not in win.viewer.message
    assert win.session.frame(2) is not None


def test_the_last_pick_wins(win, qapp):
    a = win.session.cut
    b = next(c for c in win.session.project.cuts if c is not a)
    win.open_cut(b.id, background=True)
    win.open_cut(a.id, background=True)          # back on the open cut before b came: b is dropped
    for _ in range(40):
        _pump(qapp, 0.05)
    assert win.session.cut is a and getattr(win, "_opening_cut", None) is None
    assert "여는 중" not in win.viewer.message


def test_a_cut_left_unchanged_is_not_saved_again(win, qapp, monkeypatch):
    a = win.session.cut
    b = next(c for c in win.session.project.cuts if c is not a)
    saves = []
    monkeypatch.setattr(win.session, "save", lambda: saves.append(1))
    win.session.dirty = False
    win.open_cut(b.id)
    assert saves == [] and win.session.cut is b
    win.session.dirty = True
    monkeypatch.setattr(win.session, "save", lambda: (saves.append(1), setattr(win.session, "dirty", False)))
    win.open_cut(a.id)
    assert saves == [1] and win.session.cut is a


def test_matte_store_reads_without_a_map_and_survives_a_bad_file(tmp_path):
    from contour.core.proxy import MatteStore
    m = MatteStore(tmp_path, 4, 8, 6)
    mm = m.create()
    mm[2] = 7
    mm.flush()
    del mm
    a = m.get(2)
    assert a.shape == (6, 8) and int(a.max()) == 7 and m.get(0).max() == 0
    m.close()
    with open(m.path, "r+b") as f:              # the file cut short under the reader (a network drive failing)
        f.truncate(8 * 6 * 2)
    m._mm = None
    assert m.get(3) is None                       # no crash, no picture
