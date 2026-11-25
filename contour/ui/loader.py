"""The app's loading look (1.26, the user's pick of four: "가는 선 위 빛"): a hairline track with a light that sweeps
across it, and the same line as a determinate progress bar with a soft sheen.

Everything moves by the clock, not by frames: when the app is busy (a model loading, the main window being built)
a late repaint lands where the light should be by then, instead of the animation slowing down and catching up.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

from .theme import C

SWEEP_S = 1.5            # one pass of the light
TRACK_ALPHA = 24         # the track: white at this alpha over the panel


def ease_in_out(x: float) -> float:
    """cubic ease in and out (0..1 -> 0..1)"""
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def sweep_span(t: float, width: float, period: float = SWEEP_S):
    """Where the light is at time t on a track of this width: (centre, length) in track pixels. It enters at the left,
    speeds through the middle, leaves at the right, and is a little longer while it is fast."""
    u = (t / period) % 1.0
    L = width * (0.30 + 0.18 * math.sin(math.pi * u))
    return -L / 2 + (width + L) * ease_in_out(u), L


def paint_track(p: QPainter, r: QRectF, alpha: int = TRACK_ALPHA) -> None:
    h = r.height()
    path = QPainterPath()
    path.addRoundedRect(r, h / 2, h / 2)
    p.fillPath(path, QColor(255, 255, 255, alpha))


def paint_sweep(p: QPainter, r: QRectF, t: float, color: str = "", hi: str = "", period: float = SWEEP_S) -> None:
    """The light on the track r at time t (seconds): a short stretch that fades out at both ends."""
    c, L = sweep_span(t, r.width(), period)
    x = r.left() + c
    g = QLinearGradient(x - L / 2, 0, x + L / 2, 0)
    a0 = QColor(color or C["accent"])
    a0.setAlpha(0)
    mid = QColor(hi or C["accent_hi"])
    g.setColorAt(0.0, a0)
    g.setColorAt(0.5, mid)
    g.setColorAt(1.0, a0)
    h = r.height()
    path = QPainterPath()
    path.addRoundedRect(r, h / 2, h / 2)
    p.save()
    p.setClipPath(path)
    p.fillRect(QRectF(x - L / 2, r.top(), L, h), g)
    p.restore()


def paint_progress(p: QPainter, r: QRectF, f: float, t: float) -> None:
    """A determinate line: the done part in the accent (a little lighter at its front), and a faint sheen that runs
    along it now and then so a slow step still looks alive."""
    f = max(0.0, min(1.0, f))
    h = r.height()
    w = max(h, r.width() * f) if f > 0 else 0.0
    if w <= 0:
        return
    done = QRectF(r.left(), r.top(), w, h)
    g = QLinearGradient(done.left(), 0, done.right(), 0)
    g.setColorAt(0.0, QColor(C["accent_lo"]))
    g.setColorAt(1.0, QColor(C["accent_hi"]))
    path = QPainterPath()
    path.addRoundedRect(done, h / 2, h / 2)
    p.fillPath(path, g)
    # the sheen: a soft white band crossing the done part every 2.4 s
    u = (t / 2.4) % 1.0
    if u < 0.55 and w > 24:
        L = max(24.0, w * 0.25)
        x = done.left() - L + (w + 2 * L) * ease_in_out(u / 0.55)
        s = QLinearGradient(x - L / 2, 0, x + L / 2, 0)
        s.setColorAt(0.0, QColor(255, 255, 255, 0))
        s.setColorAt(0.5, QColor(255, 255, 255, 70))
        s.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.save()
        p.setClipPath(path)
        p.fillRect(QRectF(x - L / 2, done.top(), L, h), s)
        p.restore()


class LineLoader(QWidget):
    """The line as a widget: indeterminate (the light sweeps) until a value is set, then the done part with a sheen.
    QProgressBar's few calls (setRange, setValue, value, reset) so it drops in where one was."""

    def __init__(self, height: int = 3, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._min, self._max = 0, 1000
        self._v = None                     # None: not known yet (the light sweeps)
        self._t0 = time.perf_counter()
        self._tm = QTimer(self)
        self._tm.setInterval(16)
        self._tm.timeout.connect(self.update)

    # QProgressBar's calls
    def setRange(self, a: int, b: int):
        self._min, self._max = int(a), max(int(a) + 1, int(b))

    def setValue(self, v: int):
        v = int(v)
        self._v = None if v <= self._min else v
        self.update()

    def value(self) -> int:
        return self._min if self._v is None else self._v

    def reset(self):
        self._v = None
        self.update()

    def setTextVisible(self, on: bool):
        pass

    def fraction(self):
        return None if self._v is None else (self._v - self._min) / float(self._max - self._min)

    def showEvent(self, e):
        self._tm.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._tm.stop()
        super().hideEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        paint_track(p, r)
        t = time.perf_counter() - self._t0
        f = self.fraction()
        if f is None:
            paint_sweep(p, r, t)
        else:
            paint_progress(p, r, f, t)
        p.end()
