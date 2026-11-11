"""Custom controls: segmented switch, toggle, centre-zero slider, keyframe diamond, person chip, overlays."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import (QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer,
                            Signal, Property)
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractButton, QFrame, QHBoxLayout, QLabel, QLayout, QLineEdit, QProgressBar, QPushButton,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from . import icons as I
from .theme import C, font, qc


def hline() -> QFrame:
    f = QFrame()
    f.setObjectName("HLine")
    return f


def vline() -> QFrame:
    f = QFrame()
    f.setObjectName("VLine")
    return f


def label(text: str, kind: str = "", px: int = 12, weight: int = 400) -> QLabel:
    l = QLabel(text)
    if kind:
        l.setObjectName(kind)
    if px != 12 or weight != 400:
        l.setFont(font(px, weight))
    return l


def button(text: str, kind: str = "", icon_name: str = "", tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if icon_name:
        col = "#1A0E13" if kind == "primary" else C["text"]
        b.setIcon(I.icon(icon_name, 15, col, col))
        b.setIconSize(QSize(15, 15))
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


def tool(icon_name: str, tip: str = "", checkable: bool = False, size: int = 16) -> QToolButton:
    b = QToolButton()
    b.setIcon(I.icon(icon_name, size))
    b.setIconSize(QSize(size, size))
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setAutoRaise(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setFixedSize(size + 14, size + 14)
    return b


# --------------------------------------------------------------------------- segmented

class Segmented(QWidget):
    changed = Signal(str)

    def __init__(self, options: Sequence[Tuple[str, str]], current: str = "", parent=None, compact=False,
                 icons: Optional[dict] = None):
        super().__init__(parent)
        self.options = list(options)
        self.icons = icons or {}
        self.current = current or (self.options[0][0] if self.options else "")
        self._hover = -1
        self.compact = compact
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(font(12 if not compact else 11, 500))
        h = 26 if compact else 30
        self.setFixedHeight(h)
        fm = QFontMetrics(self.font())
        w = sum(fm.horizontalAdvance(t) + (22 if not compact else 16) + (18 if k in self.icons else 0)
                for k, t in self.options)
        self.setMinimumWidth(w + 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_current(self, key: str, emit: bool = False) -> None:
        if key != self.current:
            self.current = key
            self.update()
            if emit:
                self.changed.emit(key)

    def _rects(self) -> List[QRectF]:
        n = max(1, len(self.options))
        w = (self.width() - 4) / n
        return [QRectF(2 + i * w, 2, w, self.height() - 4) for i in range(n)]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc("line2"), 1))
        p.setBrush(qc("bg2"))
        p.drawRoundedRect(r, 7, 7)
        for i, (rect, (k, t)) in enumerate(zip(self._rects(), self.options)):
            sel = k == self.current
            if sel:
                p.setPen(Qt.NoPen)
                p.setBrush(qc("bg4"))
                p.drawRoundedRect(rect, 5, 5)
            elif i == self._hover and self.isEnabled():
                p.setPen(Qt.NoPen)
                p.setBrush(qc("bg3"))
                p.drawRoundedRect(rect, 5, 5)
            col = C["text"] if sel else C["text2"]
            if not self.isEnabled():
                col = C["text3"]
            p.setPen(QColor(col))
            if k in self.icons:
                fm = QFontMetrics(self.font())
                tw = fm.horizontalAdvance(t) + 18
                x = rect.center().x() - tw / 2
                pm = I.pixmap(self.icons[k], 13, col)
                p.drawPixmap(int(x), int(rect.center().y() - 6.5), pm)
                p.drawText(QRectF(x + 18, rect.top(), tw, rect.height()), Qt.AlignVCenter | Qt.AlignLeft, t)
            else:
                p.drawText(rect, Qt.AlignCenter, t)

    def mouseMoveEvent(self, e):
        hv = next((i for i, r in enumerate(self._rects()) if r.contains(e.position())), -1)
        if hv != self._hover:
            self._hover = hv
            self.update()

    def leaveEvent(self, e):
        self._hover = -1
        self.update()

    def mousePressEvent(self, e):
        for r, (k, _) in zip(self._rects(), self.options):
            if r.contains(e.position()):
                self.set_current(k, emit=True)
                break


# --------------------------------------------------------------------------- toggle

class Toggle(QAbstractButton):
    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(30, 18)
        self._pos = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(120)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._go)

    def _go(self, on):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def get_knob(self):
        return self._pos

    def set_knob(self, v):
        self._pos = v
        self.update()

    knob = Property(float, get_knob, set_knob)

    def setChecked(self, v: bool) -> None:  # noqa: N802
        super().setChecked(v)
        self._pos = 1.0 if v else 0.0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        on = qc("accent") if self.isEnabled() else qc("accent_soft")
        off = qc("bg4")
        t = self._pos
        col = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                     int(off.blue() + (on.blue() - off.blue()) * t))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.left() + 2 + (r.width() - d - 4) * t
        p.setBrush(QColor("#FFFFFF") if self.isEnabled() else qc("text3"))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))

    def sizeHint(self):
        return QSize(30, 18)


# --------------------------------------------------------------------------- centre-zero slider

class BipolarSlider(QWidget):
    valueChanged = Signal(float)
    editingFinished = Signal(float)

    def __init__(self, lo: float = -50, hi: float = 50, value: float = 0, step: float = 1, zero: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.step, self.zero = lo, hi, step, zero
        self._v = value
        self._drag = False
        self._hover = False
        self.setMinimumWidth(80)
        self.setFixedHeight(22)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def value(self) -> float:
        return self._v

    def setValue(self, v: float, emit: bool = False) -> None:  # noqa: N802
        v = max(self.lo, min(self.hi, round(v / self.step) * self.step))
        if abs(v - self._v) > 1e-9:
            self._v = v
            self.update()
            if emit:
                self.valueChanged.emit(v)

    def _x(self, v: float) -> float:
        m = 7
        return m + (v - self.lo) / (self.hi - self.lo) * (self.width() - 2 * m)

    def _v_at(self, x: float) -> float:
        m = 7
        return self.lo + (x - m) / max(1, self.width() - 2 * m) * (self.hi - self.lo)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cy = self.height() / 2
        x0, x1 = self._x(self.lo), self._x(self.hi)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("bg4"))
        p.drawRoundedRect(QRectF(x0, cy - 1.5, x1 - x0, 3), 1.5, 1.5)
        xz = self._x(self.zero)
        xv = self._x(self._v)
        if abs(self._v - self.zero) > 1e-9:
            p.setBrush(qc("accent") if self.isEnabled() else qc("text3"))
            p.drawRoundedRect(QRectF(min(xz, xv), cy - 1.5, abs(xv - xz), 3), 1.5, 1.5)
        # zero tick
        p.setBrush(qc("text3"))
        p.drawRect(QRectF(xz - 0.5, cy - 5, 1, 10))
        r = 6.5 if (self._hover or self._drag or self.hasFocus()) else 5.5
        p.setBrush(QColor("#FFFFFF") if self.isEnabled() else qc("text3"))
        p.drawEllipse(QPointF(xv, cy), r, r)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = True
            self.setValue(self._v_at(e.position().x()), emit=True)

    def mouseMoveEvent(self, e):
        if self._drag:
            fine = e.modifiers() & Qt.ShiftModifier
            v = self._v_at(e.position().x())
            if fine:
                v = self._v + (v - self._v) * 0.2
            self.setValue(v, emit=True)
        elif not self._hover:
            self._hover = True
            self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = False
            self.update()
            self.editingFinished.emit(self._v)

    def mouseDoubleClickEvent(self, e):
        self.setValue(self.zero, emit=True)
        self.editingFinished.emit(self._v)

    def wheelEvent(self, e):
        if not self.hasFocus():
            e.ignore()
            return
        d = self.step * (1 if e.angleDelta().y() > 0 else -1)
        self.setValue(self._v + d, emit=True)
        self.editingFinished.emit(self._v)

    def keyPressEvent(self, e):
        d = self.step * (5 if e.modifiers() & Qt.ShiftModifier else 1)
        if e.key() in (Qt.Key_Right, Qt.Key_Up):
            self.setValue(self._v + d, emit=True)
            self.editingFinished.emit(self._v)
        elif e.key() in (Qt.Key_Left, Qt.Key_Down):
            self.setValue(self._v - d, emit=True)
            self.editingFinished.emit(self._v)
        else:
            super().keyPressEvent(e)


# --------------------------------------------------------------------------- keyframe diamond

class KeyButton(QAbstractButton):
    """Diamond: empty = static value, outlined accent = animated, filled = key on this frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "none"      # none | animated | key
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("키프레임 추가/삭제 (이 프레임)")

    def set_state(self, s: str):
        if s != self.state:
            self.state = s
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = 5.0
        poly = QPolygonF([c + QPointF(0, -r), c + QPointF(r, 0), c + QPointF(0, r), c + QPointF(-r, 0)])
        if self.state == "key":
            p.setPen(QPen(qc("accent"), 1.4))
            p.setBrush(qc("accent"))
        elif self.state == "animated":
            p.setPen(QPen(qc("accent"), 1.4))
            p.setBrush(Qt.NoBrush)
        else:
            p.setPen(QPen(qc("text3") if not self.underMouse() else qc("text2"), 1.2))
            p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()


# --------------------------------------------------------------------------- parameter row

class ParamRow(QWidget):
    changed = Signal(str, float)          # part, value (live)
    committed = Signal(str, float)        # part, value (on release)
    keyToggled = Signal(str)

    def __init__(self, key: str, text: str, lo=-50, hi=50, step=1, unit="%", zero=0.0, parent=None):
        super().__init__(parent)
        self.key = key
        self.unit = unit
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)
        self.name = QLabel(text)
        self.name.setFixedWidth(52)
        self.slider = BipolarSlider(lo, hi, zero, step, zero)
        self.val = QLineEdit()
        self.val.setObjectName("ParamVal")
        self.val.setFixedWidth(52)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val.setFont(font(12, 500, tabular=True))
        self.val.setStyleSheet("QLineEdit { background: transparent; border: 1px solid transparent; padding: 2px 4px; }"
                               f"QLineEdit:hover {{ border-color: {C['line2']}; }}"
                               f"QLineEdit:focus {{ border-color: {C['accent']}; background: {C['bg2']}; }}")
        self.keyb = KeyButton()
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)
        lay.addWidget(self.keyb)
        self.slider.valueChanged.connect(self._live)
        self.slider.editingFinished.connect(lambda v: self.committed.emit(self.key, v))
        self.val.editingFinished.connect(self._typed)
        self.keyb.clicked.connect(lambda: self.keyToggled.emit(self.key))
        self._show(zero)

    def _fmt(self, v: float) -> str:
        if self.unit == "%":
            return f"{v:+.0f}%" if abs(v) > 1e-9 else "0%"
        if self.unit == "x":
            return f"{v:.2f}×"
        if self.unit == "pct1":
            return f"{v:.1f}%"
        if self.unit == "pct":
            return f"{v:.0f}%"
        if self.unit == "deg":
            return f"{v:+.0f}°" if abs(v) > 1e-9 else "0°"
        return f"{v:.0f}"

    def _show(self, v: float):
        if not self.val.hasFocus():
            self.val.setText(self._fmt(v))

    def _live(self, v: float):
        self._show(v)
        self.changed.emit(self.key, v)

    def _typed(self):
        t = self.val.text().replace("%", "").replace("×", "").replace("+", "").replace("x", "").replace("°", "").strip()
        try:
            v = float(t)
        except ValueError:
            self._show(self.slider.value())
            return
        self.slider.setValue(v)
        v = self.slider.value()
        self._show(v)
        self.changed.emit(self.key, v)
        self.committed.emit(self.key, v)
        self.val.clearFocus()

    def set_value(self, v: float):
        self.slider.setValue(v)
        self._show(self.slider.value())

    def set_key_state(self, s: str):
        self.keyb.set_state(s)


# --------------------------------------------------------------------------- person chip

class Chip(QAbstractButton):
    def __init__(self, text: str, color: str, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.text_ = text
        self.color = QColor(color)
        self.badge = ""
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(font(12, 500))
        self.setFixedHeight(28)
        self._fit()

    def _fit(self):
        fm = QFontMetrics(self.font())
        bw = fm.horizontalAdvance(self.badge) + 12 if self.badge else 0
        self.setFixedWidth(fm.horizontalAdvance(self.text_) + 34 + bw)

    def set_text(self, t: str, badge: str = ""):
        self.text_ = t
        self.badge = badge
        self._fit()
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        sel = self.isChecked()
        p.setPen(QPen(self.color if sel else qc("line2"), 1.2))
        p.setBrush(qc("bg3") if sel else (qc("bg2") if not self.underMouse() else qc("bg3")))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(Qt.NoPen)
        p.setBrush(self.color)
        p.drawEllipse(QPointF(14, r.center().y()), 4.5, 4.5)
        p.setPen(qc("text") if sel else qc("text2"))
        fm = QFontMetrics(self.font())
        tw = fm.horizontalAdvance(self.text_)
        p.drawText(QRectF(24, 0, tw + 4, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text_)
        if self.badge:
            p.setPen(qc("text3"))
            p.setFont(font(11, 500))
            p.drawText(QRectF(24 + tw + 8, 0, 60, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.badge)

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()


# --------------------------------------------------------------------------- section header

class Section(QWidget):
    def __init__(self, title: str, right: Optional[QWidget] = None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 4)
        t = QLabel(title)
        t.setObjectName("Section")
        lay.addWidget(t)
        lay.addStretch(1)
        if right is not None:
            lay.addWidget(right)


class GroupHeader(QAbstractButton):
    """Collapsible group title: chevron, title, a dot when something in the group is in use, hint."""

    def __init__(self, title: str, hint: str = "", expanded: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.hint = hint
        self.active = False
        self.setCheckable(True)
        self.setChecked(expanded)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(34)
        self.toggled.connect(lambda _: self.update())

    def set_active(self, on: bool) -> None:
        if on != self.active:
            self.active = on
            self.update()

    def sizeHint(self):
        return QSize(200, 34)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        if self.underMouse():
            p.fillRect(r, QColor(255, 255, 255, 8))
        p.setPen(QPen(qc("line"), 1))
        p.drawLine(QPointF(r.left(), r.top() + 0.5), QPointF(r.right(), r.top() + 0.5))
        # chevron
        c = QPointF(8, r.center().y() + 1)
        p.setPen(QPen(qc("text2"), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.isChecked():
            pts = [c + QPointF(-4, -2), c + QPointF(0, 2), c + QPointF(4, -2)]
        else:
            pts = [c + QPointF(-2, -4), c + QPointF(2, 0), c + QPointF(-2, 4)]
        p.drawPolyline(QPolygonF(pts))
        p.setFont(font(12, 650))
        p.setPen(qc("text"))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(self.title)
        p.drawText(QRectF(22, 0, tw + 4, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.title)
        if self.active:
            p.setPen(Qt.NoPen)
            p.setBrush(qc("accent"))
            p.drawEllipse(QPointF(22 + tw + 9, r.center().y()), 3, 3)
        if self.hint and self.isChecked():
            p.setFont(font(11))
            p.setPen(qc("text3"))
            p.drawText(r.adjusted(0, 0, -2, 0), Qt.AlignVCenter | Qt.AlignRight, self.hint)

    def enterEvent(self, e):
        self.update()

    def leaveEvent(self, e):
        self.update()


class FieldRow(QWidget):
    """label ......... control"""

    def __init__(self, text: str, w: QWidget, tip: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(10)
        l = QLabel(text)
        if tip:
            l.setToolTip(tip)
            w.setToolTip(tip)
        lay.addWidget(l)
        lay.addStretch(1)
        lay.addWidget(w)


# --------------------------------------------------------------------------- busy overlay

class BusyOverlay(QWidget):
    cancelRequested = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(18,18,18,0.82);")
        box = QWidget(self)
        box.setObjectName("BusyBox")
        box.setStyleSheet(f"QWidget#BusyBox {{ background: {C['bg2']}; border: 1px solid {C['line2']}; border-radius: 12px; }}"
                          "QLabel { background: transparent; }")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(22, 18, 22, 18)
        bl.setSpacing(10)
        self.title = QLabel("")
        self.title.setFont(font(14, 600))
        self.msg = QLabel("")
        self.msg.setObjectName("Muted")
        self.msg.setFont(font(12, 400, tabular=True))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedHeight(6)
        self.cancel = button("취소", "ghost")
        self.cancel.clicked.connect(self.cancelRequested.emit)
        row = QHBoxLayout()
        row.addWidget(self.msg, 1)
        row.addWidget(self.cancel)
        bl.addWidget(self.title)
        bl.addWidget(self.bar)
        bl.addLayout(row)
        box.setFixedWidth(420)
        self.box = box
        self.hide()

    def start(self, title: str, cancellable: bool = True):
        self.title.setText(title)
        self.msg.setText("준비 중")
        self.bar.setValue(0)
        self.cancel.setVisible(cancellable)
        self.cancel.setEnabled(True)
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()

    def progress(self, f: float, msg: str = ""):
        self.bar.setValue(int(max(0, min(1, f)) * 1000))
        if msg:
            self.msg.setText(msg)

    def resizeEvent(self, e):
        self.box.adjustSize()
        self.box.move((self.width() - self.box.width()) // 2, (self.height() - self.box.height()) // 2)


class Toast(QLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setStyleSheet(f"background: {C['bg3']}; border: 1px solid {C['line2']}; border-radius: 8px; padding: 8px 14px;")
        self.setFont(font(12, 500))
        self.hide()
        self._t = QTimer(self)
        self._t.setSingleShot(True)
        self._t.timeout.connect(self.hide)

    def show_msg(self, text: str, ms: int = 2600, kind: str = ""):
        col = {"warn": C["warn"], "error": C["error"], "ok": C["ok"]}.get(kind, C["text"])
        self.setStyleSheet(f"background: {C['bg3']}; color: {col}; border: 1px solid {C['line2']}; border-radius: 8px; padding: 8px 14px;")
        self.setText(text)
        self.adjustSize()
        par = self.parentWidget()
        self.move((par.width() - self.width()) // 2, par.height() - self.height() - 28)
        self.raise_()
        self.show()
        self._t.start(ms)


class StatTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Tile")
        self.setStyleSheet(f"QFrame#Tile {{ background: {C['bg1']}; border: 1px solid {C['line']}; border-radius: 10px; }}"
                           "QLabel { background: transparent; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(2)
        self.value = QLabel("-")
        self.value.setFont(font(26, 600, tabular=True))
        self.title = QLabel(title)
        self.title.setObjectName("Muted")
        self.sub = QLabel("")
        self.sub.setObjectName("Faint")
        self.sub.setFont(font(11))
        lay.addWidget(self.value)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)

    def set(self, value: str, sub: str = ""):
        self.value.setText(value)
        self.sub.setText(sub)


class FlowLayout(QLayout):
    """Left-to-right layout that wraps (person chips)."""

    def __init__(self, parent=None, hspace: int = 6, vspace: int = 6):
        super().__init__(parent)
        self._items = []
        self.h, self.v = hspace, vspace
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, w):  # noqa: N802
        return self._do(QRect(0, 0, w, 0), True)

    def setGeometry(self, r):  # noqa: N802
        super().setGeometry(r)
        self._do(r, False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _do(self, rect, test):
        x, y, line_h = rect.x(), rect.y(), 0
        for it in self._items:
            sz = it.sizeHint()
            nx = x + sz.width() + self.h
            if nx - self.h > rect.right() and line_h > 0:
                x = rect.x()
                y += line_h + self.v
                nx = x + sz.width() + self.h
                line_h = 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), sz))
            x = nx
            line_h = max(line_h, sz.height())
        return y + line_h - rect.y()
