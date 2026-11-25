"""Start-up window: the DIAP wordmark, the loading line and what is being loaded, until the main window is up.

The heavy modules (numpy, OpenCV, SciPy, OpenColorIO, the analysis and retouch code) are imported on a thread
while the window keeps turning; the Qt parts of the app are built on the main thread afterwards. ONNX Runtime is
left to the model code: it must come after the CUDA DLL folders are set up."""
from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QApplication, QWidget

from . import loader as L
from .theme import C, font

FADE_IN_S = 0.45
HEAVY = ("numpy", "cv2", "scipy.optimize", "PyOpenColorIO", "contour.core.colour",
         "contour.core.media", "contour.track.analyze", "contour.retouch.engine", "contour.warp.render")


class Splash(QWidget):
    """1.26: a quieter card. A soft shadow, a dark surface that is a shade lighter at the top, a hairline edge and a
    faint highlight along the top, the wordmark, the loading line (loader.py) and the step under it, the version at
    the foot. It fades in (user 11/25: a fade, no rise, no maker's name), and the step's text fades from one to the next."""
    CW, CH = 480, 288        # the card
    M = 28                   # room for the shadow round it
    W, H = CW + 2 * M, CH + 2 * M

    def __init__(self, version: str):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self.W, self.H)
        self.version = version
        self.msg = "시작하는 중"
        self._prev = ""
        self._msg_t = 0.0
        self._t0 = time.perf_counter()
        wm = Path(__file__).resolve().parents[2] / "assets" / "wordmark.png"
        self.wordmark = QPixmap(str(wm)) if wm.exists() else QPixmap()
        self._t = QTimer(self)
        self._t.setInterval(16)
        self._t.timeout.connect(self.update)
        scr = QApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.move(g.center().x() - self.W // 2, g.center().y() - self.H // 2)

    # ---------------------------------------------------------------- drawing
    def showEvent(self, e):
        self._t0 = time.perf_counter()
        self._t.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._t.stop()
        super().hideEvent(e)

    def _tick(self):                    # (kept for callers of 1.25 and before)
        self.update()

    def set_message(self, msg: str):
        if msg != self.msg:
            self._prev, self.msg = self.msg, msg
            self._msg_t = time.perf_counter()
        self.update()

    def _shadow(self, p: QPainter, card: QRectF):
        """a soft shadow: rounded rects growing outwards, each a little fainter (no blur effect on a frameless window)"""
        p.setPen(Qt.NoPen)
        n = 14
        for i in range(n, 0, -1):
            k = i / n
            a = int(round(26 * (1 - k) ** 2))
            if a <= 0:
                continue
            g = 2 + k * (self.M - 6)
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(card.adjusted(-g, -g + 6, g, g + 6), 18 + g, 18 + g)

    def paintEvent(self, e):
        t = time.perf_counter() - self._t0
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        # it fades in (the whole card, shadow included; painted, so it works where a window's opacity cannot be set)
        p.setOpacity(1.0 - (1.0 - min(1.0, t / FADE_IN_S)) ** 3)
        card = QRectF(self.M, self.M, self.CW, self.CH)
        self._shadow(p, card)
        # surface
        path = QPainterPath()
        path.addRoundedRect(card, 16, 16)
        g = QLinearGradient(0, card.top(), 0, card.bottom())
        g.setColorAt(0.0, QColor("#222224"))
        g.setColorAt(1.0, QColor("#171718"))
        p.fillPath(path, g)
        # a faint light behind the wordmark
        rg = QRadialGradient(QPointF(card.center().x(), card.top() + 112), 260)
        a = QColor(C["accent"])
        a.setAlpha(10)
        rg.setColorAt(0.0, a)
        a2 = QColor(C["accent"])
        a2.setAlpha(0)
        rg.setColorAt(1.0, a2)
        p.save()
        p.setClipPath(path)
        p.fillRect(card, rg)
        p.restore()
        # edge and top highlight
        p.setPen(QPen(QColor(255, 255, 255, 20), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)
        hl = QLinearGradient(card.left() + 24, 0, card.right() - 24, 0)
        hl.setColorAt(0.0, QColor(255, 255, 255, 0))
        hl.setColorAt(0.5, QColor(255, 255, 255, 34))
        hl.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(QPen(hl, 1))
        p.drawLine(QPointF(card.left() + 24, card.top() + 1.5), QPointF(card.right() - 24, card.top() + 1.5))
        # wordmark
        cx = card.center().x()
        top = card.top() + 96
        if not self.wordmark.isNull():
            h = 30
            pm = self.wordmark.scaledToHeight(h * 2, Qt.SmoothTransformation)   # drawn at half size: sharp on 2x screens too
            w_ = pm.width() / 2
            p.drawPixmap(QRectF(cx - w_ / 2, top, w_, h), pm, QRectF(pm.rect()))
        else:
            p.setPen(QColor(C["text"]))
            p.setFont(font(28, 700))
            p.drawText(QRectF(card.left(), top - 4, self.CW, 38), Qt.AlignHCenter | Qt.AlignVCenter, "DIAP")
        # the line
        lr = QRectF(cx - 84, top + 66, 168, 2)
        L.paint_track(p, lr)
        L.paint_sweep(p, lr, t)
        # the step: the new one fades in over the old
        k = min(1.0, (time.perf_counter() - self._msg_t) / 0.22) if self._msg_t else 1.0
        p.setFont(font(11, 400, tabular=True))
        mr = QRectF(card.left(), top + 80, self.CW, 18)
        if k < 1.0 and self._prev:
            c0 = QColor(C["text3"])
            c0.setAlphaF(1.0 - k)
            p.setPen(c0)
            p.drawText(mr, Qt.AlignHCenter | Qt.AlignVCenter, self._prev)
        c1 = QColor(C["text3"])
        c1.setAlphaF(k)
        p.setPen(c1)
        p.drawText(mr, Qt.AlignHCenter | Qt.AlignVCenter, self.msg)
        # foot
        fc = QColor(C["text3"])
        fc.setAlpha(150)
        p.setPen(fc)
        fr = QRectF(card.left() + 22, card.bottom() - 34, self.CW - 44, 16)
        p.setFont(font(10, 400, tabular=True))
        p.drawText(fr, Qt.AlignRight | Qt.AlignVCenter, f"v{self.version}")
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
