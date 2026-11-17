"""Cancel (2026-11-18): the panel must not bring the time left or "분석 중" back after cancel was pressed, the key-press
toast is gone, and a cancel inside the SAM matte stops the analysis (it was taken for a failure: the RVM matte was
built and the analysis went on)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget


def test_cancel_keeps_the_panel_saying_cancelling():
    app = QApplication.instance() or QApplication([])
    from contour.ui import widgets as W
    host = QWidget()
    host.resize(900, 500)
    host.show()
    b = W.BusyOverlay(host)
    b.start("분석 중", True, 600)
    b._t0 -= 20
    b.progress(0.3, "SAM 인물 분할 10/30")
    assert "남은 시간" in b.eta.text()
    b.cancelling()
    assert b.msg.text() == "취소하는 중…" and b.eta.text() == "" and not b.cancel.isEnabled()
    for f in (0.31, 0.5, 0.9):                       # the worker still sends progress until it notices
        b.progress(f, "SAM 인물 분할 11/30")
    app.processEvents()
    assert b.msg.text() == "취소하는 중…" and b.eta.text() == ""
    b.start("다시 분석", True, 0)                     # the next job starts clean
    b.progress(0.2, "인물·포즈 검출 5/30")
    assert b.msg.text().startswith("인물") and b.cancel.isEnabled()
    b.hide()
    host.close()


def test_no_toast_when_a_key_is_pressed_while_busy():
    from contour.ui import main_window as MW
    import inspect
    src = inspect.getsource(MW.InputBlocker.eventFilter) if hasattr(MW, "InputBlocker") else inspect.getsource(MW)
    assert "작업이 끝난 뒤 조작할 수 있습니다" not in src


def test_cancel_in_the_sam_matte_is_not_a_failure_to_fall_back_from(monkeypatch):
    from contour.track import analyze as AN, segment as SG

    class Seglab:
        removed = 0

        def remove(self):
            Seglab.removed += 1

    class Matte:
        def exists(self):
            return True

    class Tk:
        tracking = type("T", (), {"seg": "sam2"})()

    t = object.__new__(AN.CutTracker)
    t.seglab, t.matte, t.proxy, t.cut = Seglab(), Matte(), None, Tk()
    t._label_cache = {}
    monkeypatch.setattr("contour.track.infer.Matter", lambda: None)

    def boom(*a, **k):
        raise AN.Cancelled()
    monkeypatch.setattr(SG, "run", boom)
    with pytest.raises(AN.Cancelled):
        t._run_matte(0, 5, lambda f, m="": None, st=object(), accuracy="high")
    assert Seglab.removed == 0, "the RVM fallback ran after a cancel"


def test_segment_run_raises_cancelled_not_break():
    from contour.track import segment as SG, analyze as AN
    import inspect
    src = inspect.getsource(SG.run)
    assert "raise Cancelled()" in src and "                break" not in src
