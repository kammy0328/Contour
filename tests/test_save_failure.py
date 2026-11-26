"""1. A failed save must not let the project be closed, the cut be switched or the app quit as if it had worked:
what was not written would be lost without a word."""
import time

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
    # dialogs answer by themselves: "do not discard" for every question, errors only recorded
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    w.errors = []
    monkeypatch.setattr(w, "show_error", lambda title, text: w.errors.append(title))
    w.open_project(project.path, wait=True)
    t0 = time.time()
    while w.job is not None and time.time() - t0 < 30:
        _pump(qapp, 0.1)
    _pump(qapp)
    yield w
    w.session.dirty = False
    w._discard_on_close = True
    monkeypatch.undo()
    w.save = lambda *a, **k: True
    w.close()


def _break_save(w, monkeypatch):
    def boom():
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(w.session, "save", boom)
    w.session.dirty = True


def test_cut_switch_stops_when_save_fails(win, monkeypatch):
    first = win.session.cut
    other = next(c for c in win.session.project.cuts if c is not first)
    _break_save(win, monkeypatch)
    win.open_cut(other.id)
    assert win.errors, "the failed save is shown"
    assert win.session.cut is first, "the cut must not change after a failed save"


def test_close_project_stops_when_save_fails(win, monkeypatch):
    prj = win.session.project
    _break_save(win, monkeypatch)
    win.close_project()
    assert win.session.project is prj, "the project must stay open after a failed save"


def test_app_close_is_refused_when_save_fails(win, monkeypatch, qapp):
    from PySide6.QtGui import QCloseEvent
    _break_save(win, monkeypatch)
    win.session.dirty = False          # no "unsaved changes" box: the final quiet save is what fails
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert not ev.isAccepted(), "quitting must stop when the last save fails"


# ------------------------------------------------------------------------------------------------ 1.27: closing
class _FakeJob:
    def __init__(self):
        self.calls = []

    def cancel(self):
        self.calls.append("cancel")

    def wait(self, ms=0):
        self.calls.append("wait")
        return True


def test_closing_takes_the_window_from_the_screen_before_it_waits_for_the_background(win, monkeypatch, qapp):
    """The waits for the frame cache, the proxies and the edit worker are up to 5 s each: the window stood there
    meanwhile without answering (the app's log, 11/26)."""
    from PySide6.QtGui import QCloseEvent
    win.show()
    _pump(qapp, 0.05)
    assert win.isVisible()
    seen = {}
    wait0 = win.cache_worker.wait
    monkeypatch.setattr(win.cache_worker, "wait", lambda *a: (seen.setdefault("cache", win.isVisible()), wait0(*a))[1])
    cancel0 = win.proxy_builder.cancel
    monkeypatch.setattr(win.proxy_builder, "cancel", lambda *a, **k: (seen.setdefault("proxy", win.isVisible()), cancel0(*a, **k))[1])
    win.session.dirty = False
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert ev.isAccepted()
    assert seen == {"proxy": False, "cache": False}, "the window is hidden before the waits"


def test_no_to_a_later_question_leaves_everything_running(win, monkeypatch, qapp):
    """Render running and batch running: "yes" to the render and "no" to the batch must not have cancelled the render."""
    from PySide6.QtGui import QCloseEvent
    win.show()
    _pump(qapp, 0.05)
    job = _FakeJob()
    monkeypatch.setattr(win.queue_page, "job", job, raising=False)
    monkeypatch.setattr(type(win.batch), "busy", property(lambda self: True))
    killed = []
    monkeypatch.setattr(win.batch, "kill", lambda: killed.append(1))
    asked = []

    def question(parent, title, *a, **k):
        asked.append(title)
        return QMessageBox.Yes if title == "렌더 중" else QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    win.session.dirty = False
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert asked == ["렌더 중", "일괄 분석 중"]
    assert not ev.isAccepted() and win.isVisible()
    assert job.calls == [] and killed == [], "nothing was stopped: the close was called off"
    # yes to both: stopped, and the window gone
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert ev.isAccepted() and job.calls == ["cancel", "wait"] and killed == [1]


def test_a_failed_last_save_keeps_the_window_on_the_screen(win, monkeypatch, qapp):
    from PySide6.QtGui import QCloseEvent
    win.show()
    _pump(qapp, 0.05)
    _break_save(win, monkeypatch)
    win.session.dirty = False
    ev = QCloseEvent()
    win.closeEvent(ev)
    assert not ev.isAccepted() and win.isVisible()
