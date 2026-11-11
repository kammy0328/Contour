"""Left panel: the project's cuts with thumbnails and status."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
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
        p.restore()


class CutList(QWidget):
    cutSelected = Signal(str)
    addRequested = Signal()
    action = Signal(str, str)        # action, cut id

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
        self.list = QListWidget()
        self.list.setItemDelegate(_Delegate(self))
        self.list.setStyleSheet("QListWidget { background: transparent; border: none; } QListWidget::item { border: none; }")
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
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
        if cut.tl:
            from ..track import status as ST
            from .tlstrip import STATUS_COL
            k = ST.effective(cut)
            lab = ST.LABEL.get(k, k) + (" · 검수" if (cut.status or {}).get("reviewed") else "")
            return lab, STATUS_COL.get(k, C["text2"])
        open_err = [q for q in cut.qc if q.status == "open" and q.severity == "error"]
        open_warn = [q for q in cut.qc if q.status == "open"]
        if cut.stats.renders > 0 and not open_err:
            return "렌더 완료", C["ok"]
        if open_err:
            return f"QC 오류 {len(open_err)}", C["error"]
        if open_warn:
            return f"QC 확인 {len(open_warn)}", C["warn"]
        if cut.analyzed:
            n_t = len(cut.targets())
            return (f"분석 완료 · 대상 {n_t}명" if n_t else "분석 완료"), C["text2"]
        return "분석 전", C["text3"]

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
        for c in cuts:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, c)
            it.setSizeHint(QSize(200, ROW_H))
            self.list.addItem(it)
            if c.id == current_id:
                self.list.setCurrentItem(it)
        self.count.setText(f"{len(cuts)}" if cuts else "")
        self.empty.setVisible(not cuts)
        self.list.setVisible(bool(cuts))
        self._block = False

    def update_rows(self):
        self.list.viewport().update()

    def _changed(self, cur, prev):
        if self._block or cur is None:
            return
        self.cutSelected.emit(cur.data(Qt.UserRole).id)

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        cut = it.data(Qt.UserRole)
        m = QMenu(self)
        acts = {
            m.addAction("이름 바꾸기"): "rename",
            m.addAction("다른 컷에 설정 복사…"): "copy",
            m.addAction("렌더 큐에 추가"): "render",
            m.addAction("원본 파일 위치 열기"): "reveal",
        }
        m.addSeparator()
        acts[m.addAction("프로젝트에서 제거")] = "remove"
        a = m.exec(self.list.viewport().mapToGlobal(pos))
        if a in acts:
            self.action.emit(acts[a], cut.id)
