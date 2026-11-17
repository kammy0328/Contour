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
