"""The busy box must fit its content when the time-left line appears later (the cancel button was cut off)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QWidget


def test_busy_box_fits_content():
    app = QApplication.instance() or QApplication([])
    from contour.ui import widgets as W
    host = QWidget()
    host.resize(900, 500)
    host.show()
    b = W.BusyOverlay(host)
    b.start("분석 중", True, 0)          # no estimate: the time-left line is empty at first
    b._t0 -= 20                         # pace measurable now: the line appears
    b.progress(0.4, "SAM 인물 분할 10/11")
    app.processEvents()
    need = b.box.layout().sizeHint().height()
    assert b.box.height() >= need
    assert b.cancel.geometry().height() >= b.cancel.sizeHint().height()
    assert b.eta.geometry().bottom() <= b.box.height()
    b.progress(0.999, "끝")
    assert b.box.height() >= b.box.layout().sizeHint().height()
    b.hide()
    host.close()
