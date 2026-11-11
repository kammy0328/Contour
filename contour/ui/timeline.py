"""Timeline: ruler, per-person tracking-confidence ribbons, keyframes, work range, playhead."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .theme import C, font, qc

LANE_H = 20
RULER_H = 24
LEFT = 92


class Timeline(QWidget):
    seek = Signal(int)
    rangeSelected = Signal(int, int)       # shift-drag selection (inclusive)
    interacted = Signal()
    keyMoved = Signal(str, int, int)       # param key ('*' = every key on that frame), from, to
    keyDeleted = Signal(str, int)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.t = 0
        self.sel_pid: Optional[int] = None
        self.selection: Optional[Tuple[int, int]] = None
        self._drag = None
        self.view0 = 0.0
        self.view1 = 1.0          # visible fraction of the clip
        self.setMouseTracking(True)
        self.setMinimumHeight(RULER_H + 3 * LANE_H + 16)
        self._hover_x = None
        self.params_open = False

    # ------------------------------------------------------------------ geometry
    @property
    def n(self) -> int:
        return self.s.proxy.n if self.s.proxy else (self.s.cut.info.frames if self.s.cut else 1)

    def _span(self) -> Tuple[float, float]:
        n = max(1, self.n)
        return self.view0 * n, self.view1 * n

    def x_of(self, f: float) -> float:
        a, b = self._span()
        return LEFT + (f - a) / max(1e-6, b - a) * (self.width() - LEFT - 10)

    def f_of(self, x: float) -> int:
        a, b = self._span()
        f = a + (x - LEFT) / max(1, self.width() - LEFT - 10) * (b - a)
        return int(np.clip(round(f - 0.5), 0, self.n - 1))

    def lanes(self) -> List[tuple]:
        """(kind, label, data)"""
        out = []
        cut = self.s.cut
        if cut is None:
            return out
        st = self.s.tracker.store if self.s.tracker else None
        for p in cut.persons:
            if p.role == "ignore":
                continue
            out.append(("person", p.name, p))
        out.append(("params", "보정 키", None))
        if self.params_open:
            for key, label in self._keyed():
                out.append(("pkey", label, key))
        return out

    def _params(self):
        cut = self.s.cut
        if cut is None or self.sel_pid is None:
            return None
        return cut.params.get(self.sel_pid)

    def _keyed(self):
        """(key, label) of every curve of the selected person that has keyframes."""
        from ..core.params import PARAM
        pp = self._params()
        if pp is None:
            return []
        out = []
        if pp.strength.keys:
            out.append(("__strength__", "전체 강도"))
        for k in pp.keyed_parts():
            out.append((k, PARAM[k].label))
        return out

    def _curve_keys(self, key: str):
        pp = self._params()
        if pp is None:
            return []
        if key == "__strength__":
            return sorted(pp.strength.keys)
        if key == "*":
            fr = set(pp.strength.keys)
            for c in pp.amounts.values():
                fr |= set(c.keys)
            return sorted(fr)
        c = pp.amounts.get(key)
        return sorted(c.keys) if c else []

    def relayout(self):
        n = len(self.lanes())
        self.setMinimumHeight(RULER_H + min(14, max(3, n)) * LANE_H + 16)
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(800, RULER_H + min(14, max(3, len(self.lanes()))) * LANE_H + 14)

    # ------------------------------------------------------------------ paint
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), qc("bg1"))
        p.setRenderHint(QPainter.Antialiasing, True)
        cut = self.s.cut
        if cut is None:
            return
        n = self.n
        W = self.width()
        x0, x1 = self.x_of(0), self.x_of(n)
        # work range shading
        a, b = cut.work_range
        p.fillRect(QRectF(LEFT, 0, W - LEFT, self.height()), qc("bg1"))
        if a > 0:
            p.fillRect(QRectF(x0, RULER_H, self.x_of(a) - x0, self.height() - RULER_H), QColor(0, 0, 0, 90))
        if b < n:
            p.fillRect(QRectF(self.x_of(b), RULER_H, x1 - self.x_of(b), self.height() - RULER_H), QColor(0, 0, 0, 90))
        self._paint_ruler(p)
        # frames whose finished result is ready for playback
        cached = self.s.play_frames() if hasattr(self.s, "play_frames") else []
        if cached:
            fr = np.zeros(n + 1, bool)
            fr[[c for c in cached if 0 <= c < n]] = True
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(88, 214, 141, 170))
            i = 0
            while i < n:
                if fr[i]:
                    j = i
                    while j < n and fr[j]:
                        j += 1
                    p.drawRect(QRectF(self.x_of(i), RULER_H - 3, max(1.0, self.x_of(j) - self.x_of(i)), 2))
                    i = j
                else:
                    i += 1
        lanes = self.lanes()
        st = self.s.tracker.store if self.s.tracker else None
        y = RULER_H + 4
        thr = cut.tracking.conf_threshold
        for kind, text, data in lanes:
            r = QRectF(LEFT, y, W - LEFT - 10, LANE_H - 4)
            sel = kind == "person" and data.id == self.sel_pid
            # label column
            p.setFont(font(11, 600 if sel else 500))
            p.setPen(qc("text") if sel else qc("text2"))
            if kind == "person":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(data.color))
                p.drawEllipse(QPointF(14, y + LANE_H / 2 - 2), 3.5, 3.5)
                p.setPen(qc("text") if sel else qc("text2"))
                p.drawText(QRectF(24, y, LEFT - 28, LANE_H - 4), Qt.AlignVCenter | Qt.AlignLeft,
                           text + ("" if data.role == "target" else " ·보호"))
            elif kind == "params":
                has = bool(self._keyed())
                c = QPointF(14, y + LANE_H / 2 - 2)
                p.setPen(QPen(qc("text2") if has else qc("text3"), 1.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                if self.params_open:
                    p.drawPolyline(QPolygonF([c + QPointF(-3.5, -1.5), c + QPointF(0, 2), c + QPointF(3.5, -1.5)]))
                else:
                    p.drawPolyline(QPolygonF([c + QPointF(-1.5, -3.5), c + QPointF(2, 0), c + QPointF(-1.5, 3.5)]))
                p.setPen(qc("text2"))
                p.drawText(QRectF(24, y, LEFT - 28, LANE_H - 4), Qt.AlignVCenter | Qt.AlignLeft, text)
            elif kind == "pkey":
                p.setFont(font(10, 500))
                p.setPen(qc("text3"))
                p.drawText(QRectF(24, y, LEFT - 26, LANE_H - 4), Qt.AlignVCenter | Qt.AlignLeft, text)
            else:
                p.drawText(QRectF(10, y, LEFT - 14, LANE_H - 4), Qt.AlignVCenter | Qt.AlignLeft, text)
            p.setPen(Qt.NoPen)
            p.setBrush(qc("bg2"))
            p.drawRoundedRect(r, 3, 3)
            if kind == "person" and st is not None and st.has(data.id):
                self._paint_conf(p, r, st, data, thr, sel)
            elif kind == "params":
                self._paint_param_keys(p, r)
            elif kind == "pkey":
                self._paint_keys(p, r, data, small=True)
            elif kind == "qc":
                self._paint_qc(p, r)
            y += LANE_H
        # selection
        if self.selection:
            sa, sb = self.selection
            p.fillRect(QRectF(self.x_of(sa), RULER_H, self.x_of(sb + 1) - self.x_of(sa), self.height() - RULER_H),
                       QColor(255, 255, 255, 22))
            p.setPen(QPen(QColor(255, 255, 255, 90), 1))
            p.drawLine(QPointF(self.x_of(sa), RULER_H), QPointF(self.x_of(sa), self.height()))
            p.drawLine(QPointF(self.x_of(sb + 1), RULER_H), QPointF(self.x_of(sb + 1), self.height()))
        # in/out marks
        for f, txt in ((a, "["), (b, "]")):
            if 0 < f < n or (f == 0 and a > 0):
                p.setPen(QPen(qc("text2"), 1.2))
                xx = self.x_of(f)
                p.drawLine(QPointF(xx, RULER_H - 6), QPointF(xx, self.height()))
        # playhead
        xp = self.x_of(self.t + 0.5)
        p.setPen(QPen(qc("accent"), 1.6))
        p.drawLine(QPointF(xp, 6), QPointF(xp, self.height()))
        p.setPen(Qt.NoPen)
        p.setBrush(qc("accent"))
        p.drawPolygon(QPolygonF([QPointF(xp - 5, 2), QPointF(xp + 5, 2), QPointF(xp, 9)]))

    def _paint_ruler(self, p: QPainter):
        n = self.n
        a, b = self._span()
        W = self.width() - LEFT - 10
        info = self.s.cut.info
        fps = max(1.0, info.fps)
        px_per_f = W / max(1e-6, b - a)
        steps = [1, 2, 5, 10, round(fps / 2), round(fps), round(fps * 2), round(fps * 5), round(fps * 10),
                 round(fps * 30), round(fps * 60)]
        step = next((s for s in steps if s > 0 and s * px_per_f >= 70), steps[-1])
        p.setFont(font(10, 500, tabular=True))
        f = int(np.floor(a / step) * step)
        while f <= b:
            if f >= 0:
                x = self.x_of(f)
                p.setPen(QPen(qc("line2"), 1))
                p.drawLine(QPointF(x, RULER_H - 7), QPointF(x, RULER_H))
                p.setPen(qc("text3"))
                p.drawText(QRectF(x + 3, 3, 120, 14), Qt.AlignLeft | Qt.AlignVCenter, info.tc_at(f) if info.timecode else str(f))
            f += step
        # current frame readout in the label column
        p.setFont(font(11, 600, tabular=True))
        p.setPen(qc("text"))
        p.drawText(QRectF(8, 2, LEFT - 12, RULER_H - 4), Qt.AlignVCenter | Qt.AlignLeft, f"{self.t}")

    def _paint_conf(self, p: QPainter, r: QRectF, st, person, thr: float, sel: bool):
        i = st.index(person.id)
        conf = st.conf[i]
        pres = st.present[i]
        n = len(conf)
        col = QColor(person.color)
        a, b = self._span()
        f0, f1 = max(0, int(a)), min(n, int(np.ceil(b)) + 1)
        W = r.width()
        cols = max(1, int(W))
        # sample per pixel column
        fr = np.linspace(f0, f1 - 1, cols).astype(int) if f1 > f0 else np.zeros(0, int)
        if len(fr) == 0:
            return
        vals = np.where(pres[fr], conf[fr], np.nan)
        path_ok = QPainterPath()
        h = r.height() - 2
        base = r.bottom() - 1
        fill = QColor(col.red(), col.green(), col.blue(), 150 if sel else 95)
        bad = QColor(C["error"])
        bad.setAlpha(200)
        for k in range(cols):
            v = vals[k]
            if not np.isfinite(v):
                continue
            x = r.left() + k * W / cols
            hh = max(1.5, h * float(np.clip(v, 0, 1)))
            p.fillRect(QRectF(x, base - hh, W / cols + 0.6, hh), fill if v >= thr else bad)
        # keyframes for this person
        keys = self.s.cut.joint_keys.get(person.id, {})
        kf = sorted({f for frs in keys.values() for f in frs})
        p.setPen(QPen(QColor(20, 20, 20), 1))
        p.setBrush(QColor(255, 255, 255, 230))
        for f in kf:
            if a <= f <= b:
                x = self.x_of(f + 0.5)
                cy = r.top() + 4
                p.drawPolygon(QPolygonF([QPointF(x, cy - 3.5), QPointF(x + 3.5, cy), QPointF(x, cy + 3.5),
                                         QPointF(x - 3.5, cy)]))
        # locked / confirmed ranges
        for ra, rb in self.s.cut.locked.get(person.id, []):
            p.fillRect(QRectF(self.x_of(ra), r.top(), self.x_of(rb + 1) - self.x_of(ra), 3), qc("text2"))
        for ra, rb in self.s.cut.confirmed.get(person.id, []):
            p.fillRect(QRectF(self.x_of(ra), r.bottom() - 2, self.x_of(rb + 1) - self.x_of(ra), 2), qc("ok"))
        # joints the user marked as not visible: a thin dashed line under the keyframes
        hid = [rg for rs in self.s.cut.hidden.get(person.id, {}).values() for rg in rs]
        if hid:
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 200), 1.2, Qt.DashLine))
            for ra, rb in hid:
                y = r.top() + 9
                p.drawLine(QPointF(self.x_of(ra), y), QPointF(self.x_of(rb + 1), y))

    def _paint_param_keys(self, p: QPainter, r: QRectF):
        self._paint_keys(p, r, "*")

    def _paint_keys(self, p: QPainter, r: QRectF, key: str, small: bool = False):
        cy = r.center().y()
        rr = 3.8 if small else 4.5
        drag = self._drag if (self._drag and self._drag[0] == "key" and self._drag[1] == key) else None
        for f in self._curve_keys(key):
            moving = drag is not None and drag[2] == f
            ff = drag[3] if moving else f
            x = self.x_of(ff + 0.5)
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.setBrush(QColor(255, 255, 255) if moving else qc("accent"))
            p.drawPolygon(QPolygonF([QPointF(x, cy - rr), QPointF(x + rr, cy), QPointF(x, cy + rr), QPointF(x - rr, cy)]))

    def _lane_at(self, y: float):
        yy = RULER_H + 4
        for kind, text, data in self.lanes():
            if yy <= y < yy + LANE_H:
                return kind, data
            yy += LANE_H
        return None, None

    def _key_at(self, key: str, x: float):
        for f in self._curve_keys(key):
            if abs(self.x_of(f + 0.5) - x) <= 6:
                return f
        return None

    def _paint_qc(self, p: QPainter, r: QRectF):
        for q in self.s.cut.qc:
            if q.status == "ignored":
                col = qc("text3")
            else:
                col = qc("error") if q.severity == "error" else qc("warn")
            x0 = self.x_of(q.start)
            x1 = max(x0 + 3, self.x_of(q.end + 1))
            p.fillRect(QRectF(x0, r.top() + 3, x1 - x0, r.height() - 6), col)

    # ------------------------------------------------------------------ interaction
    def set_frame(self, t: int):
        self.t = t
        self.update()

    def mousePressEvent(self, e):
        self.interacted.emit()
        x, y = e.position().x(), e.position().y()
        kind, data = self._lane_at(y)
        if x < LEFT:
            if kind == "params" and e.button() == Qt.LeftButton:
                self.params_open = not self.params_open
                self.relayout()
            return
        if kind in ("params", "pkey"):
            key = "*" if kind == "params" else data
            kf = self._key_at(key, x)
            if kf is not None:
                if e.button() == Qt.RightButton or (e.modifiers() & Qt.AltModifier):
                    self.keyDeleted.emit(key, kf)
                    return
                if e.button() == Qt.LeftButton:
                    self._drag = ("key", key, kf, kf)
                    self.update()
                    return
        if e.button() != Qt.LeftButton:
            return
        f = self.f_of(x)
        if e.modifiers() & Qt.ShiftModifier:
            self._drag = ("sel", f)
            self.selection = (f, f)
        else:
            self._drag = ("seek", f)
            self.seek.emit(f)
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag is None:
            kind, data = self._lane_at(e.position().y())
            over = kind in ("params", "pkey") and e.position().x() >= LEFT and \
                self._key_at("*" if kind == "params" else data, e.position().x()) is not None
            self.setCursor(Qt.SizeHorCursor if over else Qt.ArrowCursor)
            return
        f = self.f_of(e.position().x())
        if self._drag[0] == "key":
            self._drag = ("key", self._drag[1], self._drag[2], f)
            self.update()
            return
        if self._drag[0] == "seek":
            self.seek.emit(f)
        else:
            a = self._drag[1]
            self.selection = (min(a, f), max(a, f))
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag and self._drag[0] == "sel" and self.selection:
            self.rangeSelected.emit(*self.selection)
        if self._drag and self._drag[0] == "key":
            _, key, f0, f1 = self._drag
            self._drag = None
            if f1 != f0:
                self.keyMoved.emit(key, f0, f1)
            else:
                self.seek.emit(f0)
            self.update()
            return
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.selection = None
        self.update()

    def wheelEvent(self, e):
        if not (e.modifiers() & Qt.ControlModifier):
            # plain wheel steps frames
            self.seek.emit(int(np.clip(self.t + (-1 if e.angleDelta().y() > 0 else 1), 0, self.n - 1)))
            return
        x = e.position().x()
        n = max(1, self.n)
        fc = (self.f_of(x) + 0.5) / n
        span = self.view1 - self.view0
        f = 0.8 if e.angleDelta().y() > 0 else 1.25
        ns = float(np.clip(span * f, min(1.0, 30 / n), 1.0))
        rel = (fc - self.view0) / max(1e-6, span)
        self.view0 = float(np.clip(fc - rel * ns, 0, 1 - ns))
        self.view1 = self.view0 + ns
        self.update()
