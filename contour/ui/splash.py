"""Start-up window: the DIAP wordmark, a turning spinner and what is being loaded, until the main window is up.

The heavy modules (numpy, OpenCV, SciPy, OpenColorIO, the analysis and retouch code) are imported on a thread
while the window keeps turning; the Qt parts of the app are built on the main thread afterwards. ONNX Runtime is
left to the model code: it must come after the CUDA DLL folders are set up."""
from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from .theme import C, font

HEAVY = ("numpy", "cv2", "scipy.optimize", "PyOpenColorIO", "contour.core.colour",
         "contour.core.media", "contour.track.analyze", "contour.retouch.engine", "contour.warp.render")


class Splash(QWidget):
    W, H = 440, 250

    def __init__(self, version: str):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self.W, self.H)
        self.version = version
        self.msg = "시작하는 중"
        self._a = 0.0
        self._n = 0
        wm = Path(__file__).resolve().parents[2] / "assets" / "wordmark.png"
        self.wordmark = QPixmap(str(wm)) if wm.exists() else QPixmap()
        self._t = QTimer(self)
        self._t.setInterval(16)
        self._t.timeout.connect(self._tick)
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.move(g.center().x() - self.W // 2, g.center().y() - self.H // 2)

    # ---------------------------------------------------------------- drawing
    def showEvent(self, e):
        self._t.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._t.stop()
        super().hideEvent(e)

    def _tick(self):
        self._a = (self._a + 6.0) % 360.0
        self._n += 1
        self.update()

    def set_message(self, msg: str):
        self.msg = msg
        self.update()

    def paintEvent(self, e):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(C["line2"]), 1))
        p.setBrush(QColor(C["bg1"]))
        p.drawRoundedRect(r, 14, 14)
        # wordmark (white on transparent) or the name
        top = 58
        if not self.wordmark.isNull():
            h = 34
            pm = self.wordmark.scaledToHeight(h, Qt.SmoothTransformation)
            p.drawPixmap(int((self.W - pm.width()) / 2), top, pm)
        else:
            p.setPen(QColor(C["text"]))
            p.setFont(font(26, 700))
            p.drawText(QRectF(0, top, self.W, 40), Qt.AlignHCenter | Qt.AlignVCenter, "DIAP")
        p.setPen(QColor(C["text3"]))
        p.setFont(font(11, 400))
        p.drawText(QRectF(0, top + 42, self.W, 18), Qt.AlignHCenter | Qt.AlignVCenter, f"버전 {self.version}")
        # spinner: a thin arc that turns and breathes
        s, w = 26.0, 2.2
        sr = QRectF((self.W - s) / 2, 150, s, s)
        p.setPen(QPen(QColor(C["line2"]), w))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(sr)
        span = 70 + 160 * (0.5 - 0.5 * math.cos(self._n * 0.05))
        pen = QPen(QColor(C["accent"]), w)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(sr, int(-self._a * 16), int(-span * 16))
        p.setPen(QColor(C["text2"]))
        p.setFont(font(11, 400))
        p.drawText(QRectF(0, 192, self.W, 18), Qt.AlignHCenter | Qt.AlignVCenter, self.msg)
        p.end()

    # ---------------------------------------------------------------- loading
    def wait_imports(self, app: QApplication, modules=HEAVY):
        """Imports `modules` on a thread; the window keeps turning meanwhile."""
        state = {"i": 0}

        def work():
            for i, m in enumerate(modules):
                state["i"] = i
                try:
                    importlib.import_module(m)
                except Exception:
                    pass
            state["i"] = len(modules)

        th = threading.Thread(target=work, name="diap-preload", daemon=True)
        th.start()
        while th.is_alive():
            n = len(modules)
            self.set_message(f"구성 요소 불러오는 중 {min(state['i'] + 1, n)}/{n}")
            app.processEvents()
            time.sleep(0.012)
        app.processEvents()

    def finish(self, w: QWidget = None):
        """Fades out once the main window is up."""
        an = QPropertyAnimation(self, b"windowOpacity", self)
        an.setDuration(220)
        an.setStartValue(1.0)
        an.setEndValue(0.0)
        an.setEasingCurve(QEasingCurve.OutCubic)
        an.finished.connect(self.close)
        self._fade = an
        an.start()
