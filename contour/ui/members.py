"""People across the timeline: members found automatically, with every cut they appear in.

Click a face to open that cut or to move it to another member; rename a member once and every cut uses the
name; copy one appearance's beauty settings to all of that member's appearances. Every correction made
here is also a lesson for identity self-learning.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
                               QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from . import widgets as Wd
from .theme import C, font
from .viewer import np_to_qimage
from ..track import xlink as XL

THUMB = 64


def _pix(rgb: Optional[np.ndarray], conf: float, color: str) -> QPixmap:
    pm = QPixmap(THUMB, THUMB)
    pm.fill(QColor(C["bg0"]))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    if rgb is not None:
        p.drawImage(pm.rect(), np_to_qimage(np.ascontiguousarray(rgb)))
    if conf < XL.CONF_LOW:
        p.setPen(QPen(QColor(C["warn"]), 3))
        p.drawRect(1, 1, THUMB - 3, THUMB - 3)
        p.setBrush(QColor(C["warn"]))
        p.setPen(Qt.NoPen)
        p.drawEllipse(THUMB - 15, 3, 12, 12)
        p.setPen(QColor("#1E1E1E"))
        p.setFont(font(9, 700))
        p.drawText(THUMB - 15, 3, 12, 12, Qt.AlignCenter, "?")
    p.end()
    return pm


class _Row(QWidget):
    def __init__(self, dlg, mid: str, m: dict):
        super().__init__()
        self.dlg, self.mid = dlg, mid
        prj = dlg.project
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 6)
        v.setSpacing(6)
        h = QHBoxLayout()
        h.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {m['color']}; border-radius: 5px;")
        h.addWidget(dot)
        name = QLabel(m["name"])
        name.setFont(font(13, 600))
        h.addWidget(name)
        people = [(c, int(p)) for c, p in m["people"] if prj.cut(c) is not None]
        low = 0
        for c, p in people:
            q = prj.cut(c).person(p)
            if q is not None and q.member_conf < XL.CONF_LOW:
                low += 1
        sub = QLabel(f"{m['cuts']}컷" + (f" · 확인 필요 {low}" if low else ""))
        sub.setObjectName("Faint")
        sub.setStyleSheet(f"color: {C['warn'] if low else C['text3']};")
        h.addWidget(sub)
        h.addStretch(1)
        for text, fn, tip in (("이름", self._rename, "이 멤버의 이름 (모든 컷에 적용)"),
                              ("합치기", self._merge, "다른 멤버와 같은 사람"),
                              ("설정 복사", self._copy, "지금 열린 컷의 이 멤버 보정값을 이 멤버의 모든 컷에")):
            b = QPushButton(text)
            b.setProperty("kind", "ghost")
            b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b)
        v.addLayout(h)
        grid = Wd.FlowLayout()
        wrap = QWidget()
        wrap.setLayout(grid)
        for c, p in people[:60]:
            cut = prj.cut(c)
            q = cut.person(p)
            tb = QToolButton()
            tb.setAutoRaise(True)
            tb.setIconSize(QSize(THUMB, THUMB))
            tb.setIcon(QIcon(_pix(dlg.thumb(cut, p), q.member_conf if q else 1.0, m["color"])))
            tb.setToolTip(f"{cut.name} · {q.name if q else p} · 연결 확신 {(q.member_conf if q else 1) * 100:.0f}%")
            tb.setText(f"{cut.tl.get('index', 0) + 1:03d}" if cut.tl else (cut.name if len(cut.name) <= 9 else "…" + cut.name[-8:]))
            tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            tb.setStyleSheet(f"QToolButton {{ color: {C['text3']}; font-size: 10px; }}")
            tb.clicked.connect(lambda _=False, cc=c, pp=p, b=tb: self._person_menu(cc, pp, b))
            grid.addWidget(tb)
        v.addWidget(wrap)

    def _rename(self):
        m = self.dlg.project.members[self.mid]
        t, ok = QInputDialog.getText(self, "멤버 이름", "이름 (모든 컷의 이 인물에 적용)", text=m["name"])
        if ok and t.strip():
            XL.rename(self.dlg.project, self.mid, t.strip())
            self.dlg.changed()

    def _merge(self):
        prj = self.dlg.project
        menu = QMenu(self)
        acts = {}
        for mid, m in sorted(prj.members.items(), key=lambda kv: kv[1]["name"]):
            if mid != self.mid:
                acts[menu.addAction(f"{m['name']}  ({m['cuts']}컷)")] = mid
        a = menu.exec(self.mapToGlobal(self.rect().center()))
        if a in acts:
            XL.merge(prj, acts[a], self.mid)
            self.dlg.changed()

    def _copy(self):
        prj = self.dlg.project
        cur = self.dlg.main.session.cut
        src = None
        if cur is not None:
            src = next(((cur.id, int(p)) for c, p in prj.members[self.mid]["people"] if c == cur.id), None)
        if src is None:
            QMessageBox.information(self, "설정 복사", "이 멤버가 나오는 컷을 열고, 그 컷에서 보정값을 맞춘 뒤 누르세요.")
            return
        n = XL.copy_settings(prj, src[0], src[1])
        self.dlg.main.session.dirty = True
        self.dlg.main.toast(f"{prj.members[self.mid]['name']}: 피부·메이크업 보정값을 {n}개 컷에 복사했습니다", "ok")

    def _person_menu(self, cut_id: str, pid: int, btn):
        prj = self.dlg.project
        menu = QMenu(self)
        a_open = menu.addAction("이 컷 열기")
        a_ok = menu.addAction("이 멤버가 맞음 (확정)")
        mv = menu.addMenu("다른 멤버로 옮기기")
        acts = {}
        for mid, m in sorted(prj.members.items(), key=lambda kv: kv[1]["name"]):
            if mid != self.mid:
                acts[mv.addAction(f"{m['name']}  ({m['cuts']}컷)")] = mid
        a_new = menu.addAction("따로 떼기 (새 멤버)")
        a = menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        if a is None:
            return
        if a == a_open:
            self.dlg.main.open_cut(cut_id)
            self.dlg.main.cutlist.refresh(cut_id)
            return
        if a == a_ok:
            XL.assign(prj, cut_id, pid, self.mid)
        elif a == a_new:
            XL.assign(prj, cut_id, pid, None)
        elif a in acts:
            XL.assign(prj, cut_id, pid, acts[a])
        self.dlg.changed()


class MembersDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.project = main.session.project
        self.setWindowTitle("인물 연결")
        self.resize(860, 720)
        self._thumbs: Dict[tuple, Optional[np.ndarray]] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        head = QHBoxLayout()
        t = QLabel("인물 연결")
        t.setFont(font(17, 600))
        head.addWidget(t)
        self.sum = QLabel("")
        self.sum.setObjectName("Muted")
        head.addWidget(self.sum)
        head.addStretch(1)
        self.only_low = QCheckBox("확인 필요만")
        self.only_low.toggled.connect(self.rebuild)
        head.addWidget(self.only_low)
        self.show_extra = QCheckBox("한 컷에만 나온 인물도")
        self.show_extra.toggled.connect(self.rebuild)
        head.addWidget(self.show_extra)
        again = Wd.button("자동 연결 다시", "ghost", "refresh-cw", "직접 정한 연결은 그대로 두고 나머지를 다시 연결")
        again.clicked.connect(self.relink)
        head.addWidget(again)
        v.addLayout(head)
        hint = QLabel("같은 사람을 여러 컷에 걸쳐 자동으로 묶었습니다. 얼굴을 누르면 그 컷을 열거나 다른 멤버로 옮길 수 있고, "
                      "고친 내용은 그대로 유지되며 인물 재인식 학습에도 쓰입니다. 노란 테두리는 연결이 확실하지 않은 곳입니다.")
        hint.setWordWrap(True)
        hint.setObjectName("Faint")
        v.addWidget(hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        v.addWidget(self.scroll, 1)
        self.rebuild()

    def thumb(self, cut, pid):
        k = (cut.id, pid)
        if k not in self._thumbs:
            try:
                self._thumbs[k] = XL.face_thumb(self.project, cut, pid, THUMB)
            except Exception:
                self._thumbs[k] = None
        return self._thumbs[k]

    def rebuild(self, *_):
        prj = self.project
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(2)
        items = sorted(prj.members.items(), key=lambda kv: (-kv[1]["cuts"], -len(kv[1]["people"]), kv[1]["name"]))
        shown = extra = 0
        for mid, m in items:
            if m["cuts"] <= 1 and not self.show_extra.isChecked() and not m.get("user_named"):
                extra += 1
                continue
            if self.only_low.isChecked():
                lows = [1 for c, p in m["people"] if prj.cut(c) and prj.cut(c).person(int(p))
                        and prj.cut(c).person(int(p)).member_conf < XL.CONF_LOW]
                if not lows:
                    continue
            lay.addWidget(_Row(self, mid, m))
            lay.addWidget(Wd.hline())
            shown += 1
        if not shown:
            e = QLabel("표시할 멤버가 없습니다. 컷을 분석하면 자동으로 연결됩니다.")
            e.setObjectName("Faint")
            lay.addWidget(e)
        lay.addStretch(1)
        self.scroll.setWidget(inner)
        multi = sum(1 for m in prj.members.values() if m["cuts"] > 1)
        self.sum.setText(f"여러 컷 멤버 {multi}명 · 한 컷에만 나온 인물 {extra}명")

    def relink(self):
        r = XL.link(self.project)
        self.changed(toast=f"다시 연결: 여러 컷 멤버 {r['multi']}명 · 확인 필요 {r['low']}명")

    def changed(self, toast: str = ""):
        self.main.session.dirty = True
        self.main.save()
        self.main.tlstrip.update_counts()
        self.main._refresh_all()
        if toast:
            self.main.toast(toast, "ok")
        self.rebuild()
