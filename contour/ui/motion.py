"""Soft motion for every push and tool button in the app (1.26, the user: "버튼들도 누르면 부드럽게 애니메이션").

The style sheet changes a button's colour at once on hover and on press. Here those two are eased instead: a
light over the button that fades in on hover (120 ms) and out on leave (200 ms), a little more while it is held,
and on a click a soft round light that spreads from the pointer and fades (420 ms). It is a layer laid over the
button (no mouse, no focus), made the first time the pointer comes over the button, so a window full of buttons
costs nothing until they are used. The button's own colours, text and icon are untouched; a disabled button
gets nothing. A primary (accent) button darkens a little on press instead of lightening.

install(app) once after the theme (theme.apply calls it)."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPointF, QRectF, Qt, QTimer, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPainterPath, QRadialGradient
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton, QWidget

HOVER_IN_MS, HOVER_OUT_MS = 120, 200
PRESS_IN_MS, PRESS_OUT_MS = 70, 220
RIPPLE_S = 0.42
HOVER_A, PRESS_A, RIPPLE_A = 18, 14, 30         # white over the button at full hover / press, the ripple's centre


class _Layer(QWidget):
    def __init__(self, btn: QWidget):
        super().__init__(btn)
        self.btn = btn
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.hover = 0.0
        self.press = 0.0
        self.ripples = []                 # (point, start time)
        self._ah = QVariantAnimation(self)
        self._ah.valueChanged.connect(self._set_hover)
        self._ap = QVariantAnimation(self)
        self._ap.valueChanged.connect(self._set_press)
        self._rt = QTimer(self)
        self._rt.setInterval(16)
        self._rt.timeout.connect(self._ripple_tick)
        self.setGeometry(btn.rect())
        self.show()
        self.raise_()

    # ------------------------------------------------------------------ values
    def _set_hover(self, v):
        self.hover = float(v)
        self.update()

    def _set_press(self, v):
        self.press = float(v)
        self.update()

    def _go(self, an: QVariantAnimation, start: float, end: float, ms: int):
        an.stop()
        an.setStartValue(float(start))
        an.setEndValue(float(end))
        an.setDuration(max(1, int(ms * abs(end - start)) if end != start else 1))
        an.setEasingCurve(QEasingCurve.OutCubic)
        an.start()

    def enter(self):
        self._go(self._ah, self.hover, 1.0, HOVER_IN_MS)

    def leave(self):
        self._go(self._ah, self.hover, 0.0, HOVER_OUT_MS)
        if self.press > 0:
            self._go(self._ap, self.press, 0.0, PRESS_OUT_MS)

    def down(self, pos: QPointF):
        self._go(self._ap, self.press, 1.0, PRESS_IN_MS)
        self.ripples.append((QPointF(pos), time.perf_counter()))
        self.ripples = self.ripples[-3:]
        if not self._rt.isActive():
            self._rt.start()
        self.raise_()

    def up(self):
        self._go(self._ap, self.press, 0.0, PRESS_OUT_MS)

    def _ripple_tick(self):
        now = time.perf_counter()
        self.ripples = [r for r in self.ripples if now - r[1] < RIPPLE_S]
        if not self.ripples:
            self._rt.stop()
        self.update()

    # ------------------------------------------------------------------ drawing
    def _radius(self) -> float:
        r = self.btn.property("fxRadius")
        return float(r) if r is not None else 6.0

    def paintEvent(self, e):
        if not self.btn.isEnabled():
            return
        if self.hover <= 0.001 and self.press <= 0.001 and not self.ripples:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        rad = self._radius()
        path.addRoundedRect(r, rad, rad)
        primary = self.btn.property("kind") == "primary"
        a = HOVER_A * self.hover
        if a > 0.2:
            p.fillPath(path, QColor(255, 255, 255, int(round(a))))
        if self.press > 0.001:
            c = QColor(0, 0, 0, int(round(28 * self.press))) if primary else QColor(255, 255, 255, int(round(PRESS_A * self.press)))
            p.fillPath(path, c)
        now = time.perf_counter()
        if self.ripples:
            p.save()
            p.setClipPath(path)
            far = math.hypot(self.width(), self.height())
            for pt, t0 in self.ripples:
                u = min(1.0, (now - t0) / RIPPLE_S)
                k = 1 - (1 - u) ** 3                       # ease out
                rr = max(1.0, far * (0.15 + 0.85 * k))
                al = RIPPLE_A * (1 - u) ** 1.5
                g = QRadialGradient(pt, rr)
                g.setColorAt(0.0, QColor(255, 255, 255, int(round(al))))
                g.setColorAt(0.7, QColor(255, 255, 255, int(round(al * 0.45))))
                g.setColorAt(1.0, QColor(255, 255, 255, 0))
                p.fillRect(self.rect(), g)
            p.restore()
        p.end()


class ButtonMotion(QObject):
    """The application's event filter that gives each push / tool button its layer and drives it."""

    TYPES = (QPushButton, QToolButton)

    def _layer(self, w, make: bool = True):
        ly = getattr(w, "_fx_layer", None)
        if ly is None and make:
            try:
                ly = _Layer(w)
            except Exception:
                return None
            w._fx_layer = ly
        return ly

    def eventFilter(self, obj, ev):
        try:
            if not isinstance(obj, self.TYPES) or obj.property("noMotion"):
                return False
            t = ev.type()
            if t == QEvent.Enter:
                if obj.isEnabled():
                    ly = self._layer(obj)
                    if ly is not None:
                        ly.enter()
            elif t == QEvent.Leave:
                ly = self._layer(obj, False)
                if ly is not None:
                    ly.leave()
            elif t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                if obj.isEnabled():
                    ly = self._layer(obj)
                    if ly is not None:
                        ly.down(ev.position())
            elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                ly = self._layer(obj, False)
                if ly is not None:
                    ly.up()
            elif t == QEvent.Resize:
                ly = self._layer(obj, False)
                if ly is not None:
                    ly.setGeometry(obj.rect())
            elif t == QEvent.EnabledChange and not obj.isEnabled():
                ly = self._layer(obj, False)
                if ly is not None:
                    ly.hover = ly.press = 0.0
                    ly.ripples = []
                    ly.update()
        except Exception:
            pass
        return False


_MOTION = None


def install(app: QApplication) -> None:
    global _MOTION
    if _MOTION is None:
        _MOTION = ButtonMotion(app)
        app.installEventFilter(_MOTION)
