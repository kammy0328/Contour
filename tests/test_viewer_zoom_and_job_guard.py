"""1.16.2: (a) 비교 (side by side): the wheel zooms about the pointer in either pane, also when already zoomed in (the
left picture's rect then reaches into the right pane). (b) While a job runs (analysis...), clicks and keys outside
the progress panel do nothing (tabs, buttons, shortcuts)."""
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent


def _stop(v):
    """The viewer's worker threads end before the test does (a QThread still running at exit aborts Python)."""
    for w in (v.preview, v.skin_worker):
        w.stop()
        w.wait(3000)


class _PX:
    w, h, sx, sy, n = 1920, 1080, 1.0, 1.0, 10


def _wheel(v, pos, up=True):
    ev = QWheelEvent(QPointF(pos), QPointF(pos), QPoint(0, 0), QPoint(0, 120 if up else -120), Qt.NoButton,
                     Qt.NoModifier, Qt.NoScrollPhase, False)
    v.wheelEvent(ev)


@pytest.mark.parametrize("side", [0, 1])
def test_side_zoom_keeps_pixel_under_pointer(qapp, side):
    from contour.ui.session import EditSession
    from contour.ui.viewer import Viewer
    s = EditSession()
    s.proxy = _PX()
    v = Viewer(s)
    v.resize(1200, 500)
    v.mode = "side"
    cells = v._cells()
    pos = QPointF(cells[side].left() + cells[side].width() * 0.7, cells[side].top() + cells[side].height() * 0.4)
    for i in range(6):                      # zoom in a lot: the left picture now runs far into the right cell
        before = v.w_to_img(pos, v._panes()[side])
        _wheel(v, pos)
        after = v.w_to_img(pos, v._panes()[side])
        assert np.hypot(before[0] - after[0], before[1] - after[1]) < 2.0, f"step {i}: {before} -> {after}"
    _stop(v)


def test_job_guard_blocks_clicks_but_not_cancel(qapp, project, monkeypatch):
    from contour.core.settings import settings
    settings().ask_proxy = False
    from contour.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    try:
        g = w._job_guard
        press = lambda: QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.LeftButton,
                                    Qt.NoModifier)
        assert not g.eventFilter(w.inspector, press()), "no job: clicks go through"
        w.job = object()                    # a job is running
        assert g.eventFilter(w.inspector, press()), "the inspector (피부, 몸매 tabs) must not take clicks"
        assert g.eventFilter(w.cutlist, press())
        w.busy.start("분석 중", True)
        btns = [c for c in w.busy.findChildren(__import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton)]
        assert btns and not g.eventFilter(btns[0], press()), "the progress panel's cancel still works"
    finally:
        w.job = None
        w.busy.hide()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()


@pytest.mark.parametrize("side", [0, 1])
def test_side_zoom_pixel_under_pointer_painted(qapp, project, side):
    """The same, on the painted widget: the colour under the pointer stays the same while zooming in 비교."""
    from PySide6.QtWidgets import QApplication
    from contour.core.colour import manager
    from contour.ui.session import EditSession
    from contour.ui.viewer import Viewer
    s = EditSession()
    s.project = project
    cut = project.cuts[0]
    px = s.make_proxy(cut)
    px.build(manager(), s.view_settings(cut))
    s.open_cut(cut)
    v = Viewer(s)
    v.resize(1300, 700)
    v.mode = "side"
    v.show()
    qapp.processEvents()
    v._qimg_o = v._qimg
    v._qimg = lambda which: v._qimg_o("orig")          # both panes: the original (no analysis here)
    r = v._panes()[side]
    pos = v.img_to_w(40, 20, r)
    c0 = v.grab().toImage().pixelColor(int(pos.x()), int(pos.y())).red()
    for i in range(5):
        _wheel(v, pos)
        qapp.processEvents()
    c1 = v.grab().toImage().pixelColor(int(pos.x()), int(pos.y())).red()
    _stop(v)
    assert abs(c0 - c1) <= 12, f"colour under the pointer {c0} -> {c1} (zoom {v.zoom:.2f})"


def test_hover_highlight_cleared_when_pointer_is_outside(qapp):
    """A lost leave event left the person's matte highlighted with the pointer off the screen."""
    from PySide6.QtGui import QCursor
    from contour.ui.session import EditSession
    from contour.ui.viewer import Viewer
    s = EditSession()
    s.proxy = _PX()
    v = Viewer(s)
    v.resize(600, 400)
    v.show()
    QCursor.setPos(v.mapToGlobal(v.rect().bottomRight()) + QPoint(300, 300))   # far outside (offscreen: any point)
    v._hover_pid = 3
    assert not v._pointer_inside()
    v._hover_check()
    assert v._hover_pid is None
    v._hover_pid = 3
    v.hide()
    assert v._hover_pid is None
    _stop(v)
