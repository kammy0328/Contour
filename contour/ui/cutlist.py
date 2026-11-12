"""Left panel: the project's cuts with thumbnails and status."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
                               QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from . import widgets as Wd
from .theme import C, font, qc
from .viewer import np_to_qimage

ROW_H = 76


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
        x = tr.right() + 10
        p.setFont(font(12, 600))
        p.setPen(qc("text"))
        fm = p.fontMetrics()
        name = fm.elidedText(cut.name, Qt.ElideMiddle, int(r.right() - x - 6))
        p.drawText(QRectF(x, r.top() + 8, r.right() - x, 18), Qt.AlignLeft | Qt.AlignVCenter, name)
        info = cut.info
        dur = info.frames / max(1e-6, info.fps)
        meta = f"{info.width}×{info.height} · {int(dur // 60)}:{int(dur % 60):02d} · {info.frames}f · v{getattr(cut, 'version', 1)}"
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
        if (ST.effective(cut) == "bad" and not (cut.status or {}).get("reviewed")
                and WS.state(cut) not in ("skip", "done")):
            p.setPen(qc("warn"))
            p.setFont(font(10, 500))
            p.drawText(QRectF(pr.right() + 6, pr.top(), r.right() - pr.right() - 6, 18), Qt.AlignLeft | Qt.AlignVCenter,
                       "추적 확인")
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
        from ..core.workstate import STATES
        self.filter = QComboBox()
        self.filter.addItem("전체", "")
        for k, lab in STATES:
            self.filter.addItem(lab, k)
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
        cuts = prj.cuts if prj else []
        allc = len(cuts)
        f = self.filter.currentData() if hasattr(self, "filter") else ""
        if f:
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
