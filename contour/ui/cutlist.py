"""Left panel: the project's cuts with thumbnails and status."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
                               QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from . import widgets as Wd
from .theme import C, font, qc
from .viewer import np_to_qimage

ROW_H = 76


def label_of(cut) -> str:
    """Cuts of an edited timeline are listed by their number only (001, 002, ...), not by file name."""
    if cut.tl:
        return f"{int(cut.tl.get('index', 0)) + 1:03d}"
    return cut.name


def listed(prj):
    """Cuts shown in the list. Once a timeline is loaded its cuts replace the whole source video, which is
    not listed (it would sit above 001 as one long clip)."""
    cuts = list(prj.cuts) if prj else []
    if any(c.tl for c in cuts):
        tl = [c for c in cuts if c.tl]
        base = (prj.timeline or {}).get("baked")
        from ..core.tlimport import _same_file
        rest = [c for c in cuts if not c.tl and not (base and _same_file(c.source, base))]
        return sorted(tl, key=lambda c: c.tl.get("index", 0)) + rest
    return cuts


class _Delegate(QStyledItemDelegate):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel

    def sizeHint(self, opt, idx):
        return QSize(200, ROW_H)

    def paint(self, p: QPainter, opt, idx):
        cut = idx.data(Qt.UserRole)
        if cut is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(opt.rect).adjusted(6, 3, -6, -3)
        sel = bool(opt.state & QStyle.State_Selected)
        hov = bool(opt.state & QStyle.State_MouseOver)
        if sel or hov:
            p.setPen(Qt.NoPen)
            p.setBrush(qc("bg3") if sel else qc("bg2"))
            p.drawRoundedRect(r, 8, 8)
        th = self.panel.thumbs.get(cut.id)
        tr = QRectF(r.left() + 6, r.top() + 6, 96, 54)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("bg0"))
        p.drawRoundedRect(tr, 5, 5)
        if th is not None:
            p.save()
            path_r = tr
            p.setClipRect(path_r)
            iw, ih = th.width(), th.height()
            s = min(tr.width() / iw, tr.height() / ih)
            dw, dh = iw * s, ih * s
            p.drawImage(QRectF(tr.center().x() - dw / 2, tr.center().y() - dh / 2, dw, dh), th)
            p.restore()
        draw_analysed(p, tr, cut)
        x = tr.right() + 10
        p.setFont(font(12, 600))
        p.setPen(qc("text"))
        fm = p.fontMetrics()
        ver = f"v{int(getattr(cut, 'version', 1) or 1)}"
        vw = QFontMetrics(font(10, 700)).horizontalAdvance(ver) + 12
        name = fm.elidedText(label_of(cut), Qt.ElideMiddle, int(r.right() - x - 10 - vw))
        p.drawText(QRectF(x, r.top() + 8, r.right() - x - vw - 6, 18), Qt.AlignLeft | Qt.AlignVCenter, name)
        vr = QRectF(r.right() - vw - 4, r.top() + 9, vw, 16)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("bg0"))
        p.drawRoundedRect(vr, 8, 8)
        p.setFont(font(10, 700))
        p.setPen(qc("accent"))
        p.drawText(vr, Qt.AlignCenter, ver)
        p.setFont(font(12, 600))
        p.setPen(qc("text"))
        info = cut.info
        dur = info.frames / max(1e-6, info.fps)
        meta = f"{info.width}×{info.height} · {int(dur // 60)}:{int(dur % 60):02d} · {info.frames}f"
        p.setFont(font(11, 400, tabular=True))
        p.setPen(qc("text3"))
        p.drawText(QRectF(x, r.top() + 27, r.right() - x, 16), Qt.AlignLeft | Qt.AlignVCenter, meta)
        # status pill
        status, col = self.panel.status_of(cut)
        p.setFont(font(10, 600))
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(status) + 14
        pr = QRectF(x, r.top() + 46, w, 18)
        c = QColor(col)
        bg = QColor(c.red(), c.green(), c.blue(), 40)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(pr, 9, 9)
        p.setPen(c)
        p.drawText(pr, Qt.AlignCenter, status)
        # tracking needs a look (automatic check): a small amber note after the pill
        from ..core import workstate as WS
        from ..track import status as ST
        if (cut.status or {}).get("failed") and not cut.analyzed:
            p.setPen(qc("error"))
            p.setFont(font(10, 600))
            p.drawText(QRectF(pr.right() + 6, pr.top(), r.right() - pr.right() - 6, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       "분석 실패")
        elif (ST.effective(cut) == "bad" and not (cut.status or {}).get("reviewed")
                and WS.state(cut) not in ("skip", "done")):
            p.setPen(qc("warn"))
            p.setFont(font(10, 500))
            p.drawText(QRectF(pr.right() + 6, pr.top(), r.right() - pr.right() - 6, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       "추적 확인")
        p.restore()


def draw_analysed(p: QPainter, tr: QRectF, cut, left: bool = False) -> None:
    """On a cut's thumbnail: a green check once analysed, a red ! when the last analysis failed."""
    st = cut.status or {}
    if cut.analyzed:
        col, mark = QColor("#30A46C"), "check"
    elif st.get("failed"):
        col, mark = QColor("#E5484D"), "fail"
    else:
        return
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    d = 15.0
    c = QRectF(tr.left() + 3, tr.bottom() - d - 6, d, d) if left else QRectF(tr.right() - d - 3, tr.top() + 3, d, d)
    p.setPen(QPen(QColor(0, 0, 0, 120), 1))
    p.setBrush(col)
    p.drawEllipse(c)
    pen = QPen(QColor("white"), 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    cx, cy = c.center().x(), c.center().y()
    if mark == "check":
        p.drawPolyline([QPointF(cx - 3.6, cy + 0.2), QPointF(cx - 1.0, cy + 2.8), QPointF(cx + 3.8, cy - 2.6)])
    else:
        p.drawLine(QPointF(cx, cy - 3.6), QPointF(cx, cy + 0.8))
        p.drawPoint(QPointF(cx, cy + 3.4))
    p.restore()


class CutList(QWidget):
    cutSelected = Signal(str)
    addRequested = Signal()
    action = Signal(str, str)        # action, cut id
    menuRequested = Signal(list, QPoint)   # selected cut ids, global position

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.thumbs: Dict[str, QImage] = {}
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        head = QHBoxLayout()
        t = QLabel("컷")
        t.setFont(font(13, 600))
        self.count = QLabel("")
        self.count.setObjectName("Faint")
        add = Wd.tool("plus", "컷 추가 (영상 파일·이미지 시퀀스)")
        add.clicked.connect(self.addRequested.emit)
        head.addWidget(t)
        head.addWidget(self.count)
        head.addStretch(1)
        head.addWidget(add)
        lay.addLayout(head)
        # whole project: analyse every cut not analysed yet / skin basic adjustment on every analysed cut
        allb = QHBoxLayout()
        allb.setSpacing(6)
        self.all_analyze = Wd.button("전체 분석", "", "scan",
                                     "분석 전인 컷을 모두 차례로 분석 (컷을 여러 개 선택하면 그 컷만)")
        self.all_analyze.clicked.connect(lambda: self.action.emit("analyze_all", ""))
        self.all_skin = Wd.button("전체 피부 보정", "", "wand-sparkles",
                                  "분석된 모든 컷의 보정 대상 인물에 피부 기본 조정을 적용")
        self.all_skin.clicked.connect(lambda: self.action.emit("skin_all", ""))
        allb.addWidget(self.all_analyze, 1)
        allb.addWidget(self.all_skin, 1)
        lay.addLayout(allb)
        from ..core.workstate import STATES
        self.filter = QComboBox()
        self.filter.addItem("전체", "")
        for k, lab in STATES:
            self.filter.addItem(lab, k)
        self.filter.addItem("분석 완료", "analyzed")
        self.filter.addItem("분석 실패", "failed")
        self.filter.currentIndexChanged.connect(lambda _: self.refresh(self._current_id()))
        lay.addWidget(self.filter)
        self.list = QListWidget()
        self.list.setItemDelegate(_Delegate(self))
        self.list.setStyleSheet("QListWidget { background: transparent; border: none; } QListWidget::item { border: none; }")
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)     # Shift: range, Ctrl: add
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.currentItemChanged.connect(self._changed)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list, 1)
        self.empty = QLabel("영상 파일을 끌어다 놓거나\n+ 버튼으로 컷을 추가하세요.\n\nProRes, H.264, MXF, EXR·DPX 시퀀스")
        self.empty.setObjectName("Faint")
        self.empty.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.empty, 1)
        self._block = False

    def status_of(self, cut):
        from ..core import workstate as WS
        k = WS.state(cut)
        return WS.LABEL[k], WS.COLOR[k]

    def _current_id(self) -> Optional[str]:
        it = self.list.currentItem()
        return it.data(Qt.UserRole).id if it is not None else None

    def selected_ids(self):
        return [it.data(Qt.UserRole).id for it in self.list.selectedItems()]

    def set_thumb(self, cut_id: str, rgb: Optional[np.ndarray]):
        if rgb is None:
            self.thumbs.pop(cut_id, None)
        else:
            self.thumbs[cut_id] = np_to_qimage(rgb).copy()
        self.list.viewport().update()

    def refresh(self, current_id: Optional[str] = None):
        self._block = True
        self.list.clear()
        prj = self.s.project
        cuts = listed(prj)
        allc = len(cuts)
        f = self.filter.currentData() if hasattr(self, "filter") else ""
        if f == "analyzed":
            cuts = [c for c in cuts if c.analyzed]
        elif f == "failed":
            cuts = [c for c in cuts if (c.status or {}).get("failed")]
        elif f:
            from ..core import workstate as WS
            cuts = [c for c in cuts if WS.state(c) == f]
        for c in cuts:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, c)
            it.setSizeHint(QSize(200, ROW_H))
            self.list.addItem(it)
            if c.id == current_id:
                self.list.setCurrentItem(it)
        self.count.setText((f"{len(cuts)}/{allc}" if f else f"{allc}") if allc else "")
        self.empty.setVisible(not allc)
        self.list.setVisible(bool(allc))
        self._block = False

    def update_rows(self):
        self.list.viewport().update()

    def _changed(self, cur, prev):
        if self._block or cur is None:
            return
        from PySide6.QtWidgets import QApplication
        if QApplication.keyboardModifiers() & (Qt.ShiftModifier | Qt.ControlModifier) and len(self.list.selectedItems()) > 1:
            return                       # building a multi-selection: stay on the open cut
        self.cutSelected.emit(cur.data(Qt.UserRole).id)

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        if not it.isSelected():
            self.list.clearSelection()
            it.setSelected(True)
        self.menuRequested.emit(self.selected_ids(), self.list.viewport().mapToGlobal(pos))
