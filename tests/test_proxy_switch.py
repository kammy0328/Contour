"""Cut switching never asks for the proxy size (it came back on every unanalysed cut when the dialog was closed),
never waits for a running proxy (start joined the old one for up to 5 s on the UI thread), runs only the latest
request after the running one ends, never lets an earlier cut's or closed project's proxy replace what is on
screen, and lets a proxy stopped by a switch be made later."""
import threading
import time
from types import SimpleNamespace as NS

import pytest

from contour.ui.workers import ProxyBuilder


class _VS:
    def __init__(self, k="v"):
        self.k = k

    def key(self):
        return self.k


def _prj(p="P"):
    return NS(path=p)


def _cut(i):
    return NS(id=i)


class SlowBuilder(ProxyBuilder):
    def __init__(self, seconds=0.6):
        super().__init__()
        self.seconds = seconds
        self.writing = set()
        self.overlap = False
        self.built = []

    def build(self, project, cut, vs, stop):
        k = (project.path, cut.id)
        if self.writing:
            self.overlap = True
        self.writing.add(k)
        try:
            t0 = time.time()
            while time.time() - t0 < self.seconds:
                if stop.is_set():
                    return False
                time.sleep(0.01)
            self.built.append(cut.id)
            return True
        finally:
            self.writing.discard(k)


def _wait(pb, s=5.0):
    t0 = time.time()
    while (pb.busy or pb._pending is not None) and time.time() - t0 < s:
        time.sleep(0.02)


def test_switch_never_waits_and_only_the_last_runs(qapp):
    pb = SlowBuilder(0.6)
    done = []
    pb.finished.connect(lambda p, c, ok, m: done.append((p, c, ok)))
    t0 = time.time()
    assert pb.start(_prj(), _cut("A"), _VS())
    assert pb.start(_prj(), _cut("B"), _VS())
    assert pb.start(_prj(), _cut("C"), _VS())
    assert time.time() - t0 < 0.2                       # no join on the UI thread
    _wait(pb)
    qapp.processEvents()
    assert pb.built == ["C"] and not pb.overlap
    assert [k[1] for k in pb.started] == ["A", "C"]      # B was replaced before it began
    assert ("P", "C", True) in done and all(c != "A" for _, c, _ in done)


def test_stopped_proxy_can_be_made_later(qapp):
    pb = SlowBuilder(0.4)
    pb.start(_prj(), _cut("A"), _VS())
    pb.start(_prj(), _cut("B"), _VS())
    _wait(pb)
    assert ("P", "A", "v") not in pb.tried and ("P", "B", "v") in pb.tried
    assert pb.start(_prj(), _cut("A"), _VS())           # asked again when A is opened again
    _wait(pb)
    assert pb.built == ["B", "A"]
    assert not pb.start(_prj(), _cut("A"), _VS())       # finished: not again


def test_close_drops_the_waiting_request(qapp):
    pb = SlowBuilder(0.4)
    pb.start(_prj(), _cut("A"), _VS())
    pb.start(_prj(), _cut("B"), _VS())
    t0 = time.time()
    pb.cancel()
    assert time.time() - t0 < 0.1
    _wait(pb)
    assert pb.built == []


def test_done_signal_of_another_cut_or_project_is_ignored():
    from contour.ui.main_window import MainWindow
    swaps = []
    s = NS(cut=NS(id="C"), project=NS(path="P"), swap_frames=lambda cid, new: swaps.append(cid) or True,
           view_settings=lambda c: None)
    win = NS(session=s, job=None, toast=lambda *a, **k: None)
    import contour.core.frames as FR
    orig = FR.frames_for
    FR.frames_for = lambda p, c, vs=None: NS(kind="proxy")
    try:
        MainWindow._bg_proxy_done(win, "P", "A", True, "")
        MainWindow._bg_proxy_done(win, "OLD", "C", True, "")
        assert swaps == []
        MainWindow._bg_proxy_done(win, "P", "C", True, "")
        assert swaps == ["C"]
    finally:
        FR.frames_for = orig


def test_no_proxy_question_on_cut_switch():
    import inspect
    from contour.ui.main_window import MainWindow
    assert "_ask_proxy_size" not in inspect.getsource(MainWindow.open_cut)
    assert not hasattr(MainWindow, "_ask_proxy_size")
