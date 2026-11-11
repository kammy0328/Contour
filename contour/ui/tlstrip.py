"""Timeline strip: every shot of an edited timeline as a thumbnail, in edit order, with its beauty status.

Click opens a shot; Up/Down (handled by the main window) go to the previous / next shot. Ctrl/Shift select
several for batch work (analyse, apply a preset, set status). Status chips at the top select all shots of
that status.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
                               QMenu, QPushButton, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from . import widgets as Wd
from .theme import C, font, qc
from .viewer import np_to_qimage
from ..track import status as ST

TW, TH = 132, 74
STATUS_COL = {"good": C["ok"], "bad": C["error"], "none": C["text3"], "small": C["warn"], "pending": C["text2"],
              "skip": "#7A6CF0"}
ORDER = ["good", "bad", "small", "none", "pending", "skip"]


class _Delegate(QStyledItemDelegate):
    def __init__(self, strip):
        super().__init__()
        self.strip = strip

    def sizeHint(self, opt, idx):
        return QSize(TW + 12, TH + 40)

    def paint(self, p: QPainter, opt, idx):
        cut = idx.data(Qt.UserRole)
        if cut is None:
            return
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(opt.rect).adjusted(4, 3, -4, -3)
        sel = bool(opt.state & QStyle.State_Selected)
        cur = self.strip.current_id == cut.id
        p.setPen(Qt.NoPen)
        p.setBrush(qc("bg3") if sel else (qc("bg2") if bool(opt.state & QStyle.State_MouseOver) else qc("bg1")))
        p.drawRoundedRect(r, 7, 7)
        tr = QRectF(r.left() + 4, r.top() + 4, TW, TH)
        p.setBrush(qc("bg0"))
        p.drawRoundedRect(tr, 4, 4)
        th = self.strip.thumbs.get(cut.id)
        if th is not None:
            p.save()
            p.setClipRect(tr)
            s = max(tr.width() / th.width(), tr.height() / th.height())
            dw, dh = th.width() * s, th.height() * s
            p.drawImage(QRectF(tr.center().x() - dw / 2, tr.center().y() - dh / 2, dw, dh), th)
            p.restore()
        st = ST.effective(cut)
        col = QColor(STATUS_COL.get(st, C["text2"]))
        # status bar under the picture, user override drawn solid, automatic drawn lighter
        bar = QRectF(tr.left(), tr.bottom() - 3, tr.width(), 3)
        c2 = QColor(col)
        if not (cut.status or {}).get("user"):
            c2.setAlpha(190)
        p.setBrush(c2)
        p.drawRect(bar)
        # running / queued
        run = self.strip.running.get(cut.id)
        if run is not None:
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRect(tr)
            p.setPen(qc("text"))
            p.setFont(font(10, 600))
            p.drawText(tr, Qt.AlignCenter, "대기" if run < 0 else f"분석 {run * 100:.0f}%")
            p.setPen(Qt.NoPen)
        # reviewed tick
        if (cut.status or {}).get("reviewed"):
            cr = QRectF(tr.right() - 20, tr.top() + 4, 16, 16)
            p.setBrush(qc("ok"))
            p.drawEllipse(cr)
            pen = QPen(QColor("#10251C"), 2)
            p.setPen(pen)
            p.drawLine(cr.left() + 4, cr.center().y(), cr.center().x() - 1, cr.bottom() - 4)
            p.drawLine(cr.center().x() - 1, cr.bottom() - 4, cr.right() - 3, cr.top() + 4)
            p.setPen(Qt.NoPen)
        # index
        p.setBrush(QColor(0, 0, 0, 160))
        ir = QRectF(tr.left() + 4, tr.top() + 4, 30, 16)
        p.drawRoundedRect(ir, 3, 3)
        p.setPen(qc("text"))
        p.setFont(font(9, 600, tabular=True))
        p.drawText(ir, Qt.AlignCenter, f"{cut.tl.get('index', 0) + 1:03d}")
        if cur:
            p.setPen(QPen(qc("accent"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(tr.adjusted(-1, -1, 1, 1), 5, 5)
        # label
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QRectF(r.left() + 6, tr.bottom() + 8, 6, 6))
        p.setPen(qc("text2") if not sel else qc("text"))
        p.setFont(font(10, 600))
        lab = ST.LABEL.get(st, st)
        p.drawText(QRectF(r.left() + 16, tr.bottom() + 3, TW - 10, 15), Qt.AlignLeft | Qt.AlignVCenter, lab)
        p.setPen(qc("text3"))
        p.setFont(font(9, 400, tabular=True))
        n = cut.info.frames
        fm = p.fontMetrics()
        sub = f"{n}f · " + (cut.tl.get("note") or (cut.status or {}).get("reason") or cut.tl.get("name", ""))
        p.drawText(QRectF(r.left() + 6, tr.bottom() + 18, TW, 14), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(sub, Qt.ElideRight, TW))
        p.restore()


class TimelineStrip(QWidget):
    cutSelected = Signal(str)
    action = Signal(str, list)          # action, cut ids

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.s = session
        self.thumbs: Dict[str, QImage] = {}
        self.running: Dict[str, float] = {}      # cut id -> progress (-1 = queued)
        self.current_id: Optional[str] = None
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 4)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel("타임라인")
        self.title.setFont(font(12, 600))
        head.addWidget(self.title)
        self.chips: Dict[str, QPushButton] = {}
        for k in ORDER:
            b = QPushButton("")
            b.setProperty("kind", "ghost")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip("이 상태인 컷 모두 선택")
            b.clicked.connect(lambda _=False, kk=k: self.select_status(kk))
            b.setStyleSheet(f"QPushButton {{ color: {STATUS_COL[k]}; padding: 1px 8px; }}")
            self.chips[k] = b
            head.addWidget(b)
        head.addStretch(1)
        self.run_lbl = QLabel("")
        self.run_lbl.setObjectName("Faint")
        head.addWidget(self.run_lbl)
        for key, text, ic, tip in (("analyze", "일괄 분석", "scan", "선택한 컷(없으면 분석 전 컷 전부)을 차례로 분석"),
                                   ("apply", "일괄 적용", "wand-sparkles", "선택한 컷(없으면 전부)의 보정 대상에 프리셋 적용"),
                                   ("review", "검수", "shield-check", "인식 불량·확인 필요 컷부터 하나씩 확인"),
                                   ("members", "인물", "users", "여러 컷에 나오는 같은 사람 (자동 연결, 이름, 설정 복사)"),
                                   ("export", "내보내기", "download", "전체 길이 / 컷별 / 리터칭 컷 + XML 출력"),
                                   ("info", "", "circle-alert", "편집 파일 정보와 경고")):
            b = Wd.button(text, "ghost" if key != "export" else "primary", ic, tip)
            b.clicked.connect(lambda _=False, kk=key: self.action.emit(kk, self.selected_ids()))
            head.addWidget(b)
            if key == "analyze":
                self.analyze_btn = b
        self.stop_btn = Wd.button("중지", "ghost", "square", "일괄 분석 중지 (지금 컷은 끝까지)")
        self.stop_btn.clicked.connect(lambda: self.action.emit("stop", []))
        self.stop_btn.hide()
        head.addWidget(self.stop_btn)
        lay.addLayout(head)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode)
        self.list.setFlow(QListView.LeftToRight)
        self.list.setWrapping(False)
        self.list.setMovement(QListView.Static)
        self.list.setResizeMode(QListView.Adjust)
        self.list.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setItemDelegate(_Delegate(self))
        self.list.setMouseTracking(True)
        self.list.setFixedHeight(TH + 58)
        self.list.setSpacing(0)
        self.list.setStyleSheet("QListWidget { background: transparent; border: none; } QListWidget::item { border: none; }")
        self.list.itemClicked.connect(self._clicked)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list)
        self._items: Dict[str, QListWidgetItem] = {}

    # ------------------------------------------------------------------ data
    def cuts(self) -> List:
        prj = self.s.project
        return prj.timeline_cuts() if prj else []

    def refresh(self):
        from ..core.tlimport import load_thumb
        prj = self.s.project
        cuts = self.cuts()
        self.setVisible(bool(cuts))
        if not cuts:
            return
        self.list.clear()
        self._items = {}
        for c in cuts:
            if c.id not in self.thumbs:
                rgb = load_thumb(prj, c.id)
                if rgb is not None:
                    self.thumbs[c.id] = np_to_qimage(np.ascontiguousarray(rgb)).copy()
            it = QListWidgetItem()
            it.setData(Qt.UserRole, c)
            it.setSizeHint(QSize(TW + 12, TH + 40))
            self.list.addItem(it)
            self._items[c.id] = it
        self.update_counts()

    def update_counts(self):
        cuts = self.cuts()
        cnt = {k: 0 for k in ORDER}
        for c in cuts:
            cnt[ST.effective(c)] = cnt.get(ST.effective(c), 0) + 1
        tl = (self.s.project.timeline or {}) if self.s.project else {}
        n = tl.get("frames", 0)
        fps = tl.get("fps", 24) or 24
        self.title.setText(f"타임라인 · {len(cuts)}컷 · {int(n / fps // 60)}:{int(n / fps % 60):02d}")
        for k, b in self.chips.items():
            b.setText(f"{ST.LABEL[k]} {cnt.get(k, 0)}")
            b.setVisible(cnt.get(k, 0) > 0)
        reviewed = sum(1 for c in cuts if (c.status or {}).get("reviewed"))
        self.chips["good"].setToolTip(f"이 상태인 컷 모두 선택 · 검수 완료 {reviewed}/{len(cuts)}")
        self.list.viewport().update()

    def set_current(self, cut_id: Optional[str]):
        self.current_id = cut_id
        it = self._items.get(cut_id or "")
        if it is not None:
            self.list.scrollToItem(it, QAbstractItemView.PositionAtCenter)
        self.list.viewport().update()

    def set_running(self, running: Dict[str, float], text: str = ""):
        self.running = dict(running)
        self.run_lbl.setText(text)
        busy = bool(running)
        self.stop_btn.setVisible(busy)
        self.analyze_btn.setEnabled(not busy)
        self.list.viewport().update()

    def selected_ids(self) -> List[str]:
        return [it.data(Qt.UserRole).id for it in self.list.selectedItems()]

    def select_status(self, k: str):
        self.list.clearSelection()
        for c in self.cuts():
            if ST.effective(c) == k:
                self._items[c.id].setSelected(True)

    def neighbour(self, d: int) -> Optional[str]:
        cuts = self.cuts()
        if not cuts:
            return None
        ids = [c.id for c in cuts]
        i = ids.index(self.current_id) if self.current_id in ids else (-1 if d > 0 else len(ids))
        j = i + d
        return ids[j] if 0 <= j < len(ids) else None

    # ------------------------------------------------------------------ interaction
    def _clicked(self, it):
        if len(self.list.selectedItems()) <= 1:
            self.cutSelected.emit(it.data(Qt.UserRole).id)

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is not None and not it.isSelected():
            self.list.clearSelection()
            it.setSelected(True)
        ids = self.selected_ids()
        if not ids:
            return
        m = QMenu(self)
        acts = {}
        sm = m.addMenu(f"상태 지정 ({len(ids)}컷)")
        for k in ("good", "bad", "none", "small", "skip"):
            acts[sm.addAction(ST.LABEL[k])] = f"status:{k}"
        sm.addSeparator()
        acts[sm.addAction("자동 판정으로 되돌리기")] = "status:"
        acts[m.addAction("검수 완료 표시")] = "reviewed:1"
        acts[m.addAction("검수 완료 해제")] = "reviewed:0"
        m.addSeparator()
        acts[m.addAction("선택 컷 분석 (다시 분석 포함)")] = "analyze_force"
        acts[m.addAction("선택 컷에 프리셋 적용…")] = "apply"
        acts[m.addAction("선택 컷 내보내기…")] = "export_sel"
        a = m.exec(self.list.viewport().mapToGlobal(pos))
        if a in acts:
            self.action.emit(acts[a], ids)
