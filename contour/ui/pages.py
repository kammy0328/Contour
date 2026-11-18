"""Render queue, dashboard and model pages."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QDesktopServices, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import widgets as Wd
from ..core.settings import settings
from ..stats.tracker import aggregate, export_csv, log_event
from .theme import C, font, qc
from .workers import Job


def _table(headers: List[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setShowGrid(False)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    t.verticalHeader().setDefaultSectionSize(34)
    t.setFocusPolicy(Qt.NoFocus)
    t.setWordWrap(False)
    return t


def _item(text, align=Qt.AlignLeft, color: Optional[str] = None, data=None) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(align | Qt.AlignVCenter)
    if color:
        it.setForeground(QColor(color))
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


def _page_header(title: str, sub: str = "") -> QWidget:
    w = QWidget()
    l = QVBoxLayout(w)
    l.setContentsMargins(0, 0, 0, 6)
    l.setSpacing(2)
    t = QLabel(title)
    t.setFont(font(20, 600))
    l.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setObjectName("Muted")
        s.setWordWrap(True)
        l.addWidget(s)
    return w


# =========================================================================== render queue

@dataclass
class QueueItem:
    id: str
    project_path: str
    project_name: str
    cut_id: str
    cut_name: str
    output: dict
    frames: int
    fmt_label: str
    status: str = "wait"          # wait | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    out_path: str = ""
    added: float = field(default_factory=time.time)
    seconds: float = 0.0
    kind: str = "cut"             # cut | timeline
    export: dict = field(default_factory=dict)   # timeline export settings (warp/tlrender.TimelineExport)
    cut_no: str = ""              # timeline number (001) or the cut's place in the project
    version: int = 0
    src_cs: str = ""              # the source colour space as the cut reads it


STATUS_TEXT = {"wait": "대기", "running": "렌더 중", "done": "완료", "failed": "실패", "cancelled": "취소됨"}


def source_cs_label(prj, cut) -> str:
    """The colour space the cut's source is read in: the input colour space set on the cut (or the project),
    else what the file says."""
    cs = (getattr(cut, "input_cs", "") or getattr(prj, "input_cs", "") or "").strip()
    if cs and cs.lower() not in ("passthrough", "none"):
        return cs
    inf = cut.info
    trc = (inf.color_trc or "").lower()
    prim = (inf.color_primaries or "").lower()
    if inf.is_float:
        return "Linear (float)"
    if trc == "smpte2084":
        return "PQ (Rec.2100)"
    if trc == "arib-std-b67":
        return "HLG (Rec.2100)"
    if "2020" in prim:
        return "Rec.2020"
    if "p3" in prim or "smpte43" in prim:
        return "P3"
    return "Rec.709"


def cut_number(prj, cut) -> str:
    if cut.tl and "index" in cut.tl:
        return f"{int(cut.tl['index']) + 1:03d}"
    try:
        return f"{prj.cuts.index(cut) + 1:03d}"
    except ValueError:
        return ""


class _QueueTable(QTableWidget):
    """Clicks on 코덱 / 출력 경로 of a row that is part of a multi-row selection keep the selection
    (Ctrl+A, then click a path: every selected row gets the folder)."""
    cellAction = Signal(int, int)        # row, column

    ACT_COLS = ()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not (e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)):
            ix = self.indexAt(e.position().toPoint())
            if ix.isValid() and ix.column() in self.ACT_COLS:
                rows = {i.row() for i in self.selectionModel().selectedRows()}
                if ix.row() not in rows:
                    self.selectRow(ix.row())
                self.cellAction.emit(ix.row(), ix.column())
                return
        super().mousePressEvent(e)


class RenderQueuePage(QWidget):
    jobFinished = Signal(object, object)       # QueueItem, RenderReport | None
    previewed = Signal(str, int, object)       # item id, frame, rgb (from the render thread)
    C_THUMB, C_NO, C_CUT, C_VER, C_FMT, C_CS, C_PATH, C_STATE, C_PROG = range(9)

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.items: List[QueueItem] = []
        self.job: Optional[Job] = None
        self.current: Optional[QueueItem] = None
        self._prj_cache: dict = {}
        self._thumbs: dict = {}                 # item id -> QPixmap (source frame; the result while it renders)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(12)
        lay.addWidget(_page_header("렌더", "컷을 쌓아 두고 한 번에 렌더합니다. 앱을 다시 켜도 대기 중인 작업은 남아 있습니다. "
                                   "코덱이나 출력 경로를 누르면 바꿀 수 있고, Ctrl+A로 모두 고른 뒤 누르면 한꺼번에 바뀝니다."))
        bar = QHBoxLayout()
        self.start = Wd.button("렌더 시작", "primary", "play")
        self.start.clicked.connect(self.start_queue)
        self.stop = Wd.button("현재 작업 취소", "", "x")
        self.stop.clicked.connect(self.cancel_current)
        self.up = Wd.button("", "ghost", "arrow-up", "위로")
        self.down = Wd.button("", "ghost", "arrow-down", "아래로")
        self.rm = Wd.button("목록에서 제거", "ghost", "trash-2")
        self.clear_done = Wd.button("완료 항목 정리", "ghost", "check")
        self.open_dir = Wd.button("출력 폴더 열기", "ghost", "folder-open")
        self.up.clicked.connect(lambda: self._move(-1))
        self.down.clicked.connect(lambda: self._move(1))
        self.rm.clicked.connect(self._remove)
        self.clear_done.clicked.connect(self._clear_done)
        self.open_dir.clicked.connect(self._open_dir)
        self.train_after = QCheckBox("큐가 끝나면 누적 모델 학습")
        self.train_after.setChecked(True)
        self.train_after.hide()                 # learning is automatic; no switch
        self.bake_all = Wd.button("전체 컷 일괄 렌더 (베이크)", "", "film",
                                  "편집 타임라인 전체를 베이크 영상과 같은 길이의 한 파일로 출력합니다")
        self.bake_all.clicked.connect(lambda: self.main._tl_action("export_full", []))
        for w in (self.start, self.stop, self.bake_all, self.up, self.down, self.rm, self.clear_done,
                  self.open_dir):
            bar.addWidget(w)
        bar.addStretch(1)
        lay.addLayout(bar)
        lay.addWidget(self._now_panel())
        self.table = _QueueTable(0, 9)
        self.table.ACT_COLS = (self.C_FMT, self.C_PATH)
        self.table.setHorizontalHeaderLabels(["", "번호", "컷", "버전", "코덱", "원본 색공간", "출력 경로", "상태", "진행"])
        t = self.table
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setShowGrid(False)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setDefaultSectionSize(52)
        t.setIconSize(QSize(80, 45))
        t.setWordWrap(False)
        t.setFocusPolicy(Qt.StrongFocus)
        t.setMouseTracking(True)
        for i, w in enumerate((92, 52, 190, 50, 130, 150, 360, 70)):
            t.setColumnWidth(i, w)
        t.cellAction.connect(self._cell_action)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._context)
        lay.addWidget(t, 1)
        self.empty = Wd.label("렌더 목록이 비어 있습니다. 편집 화면의 출력 탭에서 '렌더에 추가'를 누르세요.", "Faint")
        lay.addWidget(self.empty)
        from PySide6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence.SelectAll, self)
        sc.activated.connect(self.table.selectAll)
        self.previewed.connect(self._on_preview)
        self.load()
        self.refresh()

    # ------------------------------------------------------------------ now rendering
    def _now_panel(self) -> QWidget:
        f = QFrame()
        f.setObjectName("Card")
        f.setStyleSheet(f"QFrame#Card {{ background: {C['bg2']}; border: 1px solid {C['line']}; border-radius: 10px; }}")
        h = QHBoxLayout(f)
        h.setContentsMargins(14, 12, 14, 12)
        h.setSpacing(16)
        self.now_img = QLabel()
        self.now_img.setFixedSize(320, 180)
        self.now_img.setAlignment(Qt.AlignCenter)
        self.now_img.setStyleSheet(f"background: {C['bg0'] if 'bg0' in C else '#000'}; border-radius: 6px;")
        h.addWidget(self.now_img)
        v = QVBoxLayout()
        v.setSpacing(5)
        self.now_title = QLabel("렌더 중인 작업 없음")
        self.now_title.setFont(font(15, 600))
        self.now_fmt = Wd.label("", "Faint", 12)
        self.now_path = Wd.label("", "Faint", 12)
        self.now_path.setWordWrap(True)
        self.now_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.now_bar = QProgressBar()
        self.now_bar.setRange(0, 1000)
        self.now_bar.setTextVisible(False)
        self.now_bar.setFixedHeight(8)
        self.now_msg = Wd.label("", "Faint", 12)
        for w in (self.now_title, self.now_fmt, self.now_path):
            v.addWidget(w)
        v.addStretch(1)
        v.addWidget(self.now_bar)
        v.addWidget(self.now_msg)
        h.addLayout(v, 1)
        return f

    def _fill_now(self):
        it = self.current
        if it is None:
            self.now_title.setText("렌더 중인 작업 없음")
            n = sum(1 for i in self.items if i.status == "wait")
            self.now_fmt.setText(f"대기 {n}개" if n else "")
            self.now_path.setText("")
            self.now_bar.setValue(0)
            self.now_msg.setText("")
            self.now_img.setPixmap(QPixmap())
            return
        no = f"{it.cut_no} · " if it.cut_no else ""
        ver = f" · v{it.version}" if it.version else ""
        self.now_title.setText(f"{no}{it.cut_name}{ver}")
        self.now_fmt.setText(f"{it.fmt_label}  ·  원본 {it.src_cs or '-'}  ·  {it.frames}프레임  ·  {it.project_name}")
        self.now_path.setText(self._out_label(it)[0])
        self.now_bar.setValue(int(it.progress * 1000))
        self.now_msg.setText(it.message)
        pm = self._thumbs.get(it.id)
        if pm is not None:
            self.now_img.setPixmap(pm.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_preview(self, iid: str, frame: int, rgb):
        from .viewer import np_to_qimage
        try:
            pm = QPixmap.fromImage(np_to_qimage(np.ascontiguousarray(rgb)))
        except Exception:
            return
        self._thumbs[iid] = pm
        if self.current is not None and self.current.id == iid:
            self.now_img.setPixmap(pm.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            r = self.items.index(self.current) if self.current in self.items else -1
            if r >= 0:
                it0 = self.table.item(r, self.C_THUMB)
                if it0 is not None:
                    it0.setIcon(QIcon(pm.scaled(80, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    # ------------------------------------------------------------------ project lookups (cached)
    def _project(self, path: str):
        from ..core.project import Project
        try:
            f = Path(path) / "project.json" if Path(path).is_dir() else Path(path)
            mt = f.stat().st_mtime
        except OSError:
            return None
        hit = self._prj_cache.get(path)
        if hit and hit[0] == mt:
            return hit[1]
        try:
            prj = Project.load(path)
        except Exception:
            return None
        self._prj_cache[path] = (mt, prj)
        return prj

    def _fill_info(self, it: QueueItem):
        """Number, version and source colour space for an item queued before they were kept."""
        if it.kind != "cut" or (it.cut_no and it.version and it.src_cs):
            return
        prj = self._project(it.project_path)
        cut = prj.cut(it.cut_id) if prj else None
        if cut is None:
            return
        it.cut_no = it.cut_no or cut_number(prj, cut)
        it.version = it.version or int(getattr(cut, "version", 1) or 1)
        it.src_cs = it.src_cs or source_cs_label(prj, cut)

    def _out_label(self, it: QueueItem):
        """(the file the render writes, whether a file of that name is there already)"""
        if it.out_path and it.status == "done":
            return it.out_path, False
        if it.kind == "timeline":
            d = it.export.get("out_dir") or str(Path(it.project_path).parent / "DIAP_Renders")
            return d, False
        from ..core.project import OutputSettings
        from ..warp.render import output_base
        prj = self._project(it.project_path)
        cut = prj.cut(it.cut_id) if prj else None
        if cut is None:
            return "(프로젝트에서 컷을 찾을 수 없음)", False
        base = output_base(prj, cut, OutputSettings.from_dict(it.output))
        if it.output.get("format") == "exr_half":
            return str(base) + "\\ (EXR 시퀀스)", base.exists()
        f = Path(str(base) + ".mov")
        return str(f), f.exists()

    def _thumb(self, it: QueueItem):
        pm = self._thumbs.get(it.id)
        if pm is not None or it.kind != "cut":
            return pm
        prj = self._project(it.project_path)
        cut = prj.cut(it.cut_id) if prj else None
        if cut is None:
            return None
        try:
            from ..core.frames import frames_for
            from .viewer import np_to_qimage
            px = frames_for(prj, cut)
            if px is None or px.kind != "proxy":
                return None                 # painted on the UI thread: never a decode of the original here
            a, b = cut.work_range
            th = px.thumbnail((a + b) // 2, 160)
            px.close()
            pm = QPixmap.fromImage(np_to_qimage(np.ascontiguousarray(th)))
        except Exception:
            return None
        self._thumbs[it.id] = pm
        return pm

    # ------------------------------------------------------------------ persistence
    def _file(self) -> Path:
        return settings().data_path / "render_queue.json"

    def load(self):
        f = self._file()
        if f.exists():
            try:
                for d in json.loads(f.read_text(encoding="utf-8")):
                    d = {k: v for k, v in d.items() if k in QueueItem.__dataclass_fields__}
                    it = QueueItem(**d)
                    if it.status == "running":
                        it.status = "wait"
                        it.progress = 0
                    self.items.append(it)
            except Exception:
                pass

    def save(self):
        try:
            self._file().write_text(json.dumps([asdict(i) for i in self.items], ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        except OSError:
            pass

    # ------------------------------------------------------------------ queue ops
    def add(self, project, cut) -> QueueItem:
        from ..core.project import OUTPUT_FORMATS
        a, b = cut.work_range
        fmt = dict(OUTPUT_FORMATS).get(cut.output.format, cut.output.format)
        it = QueueItem(id=uuid.uuid4().hex[:8], project_path=project.path, project_name=project.name, cut_id=cut.id,
                       cut_name=cut.name, output=cut.output.to_dict(), frames=b - a, fmt_label=fmt,
                       cut_no=cut_number(project, cut), version=int(getattr(cut, "version", 1) or 1),
                       src_cs=source_cs_label(project, cut))
        self.items.append(it)
        self.save()
        self.refresh()
        return it

    def add_timeline(self, project, exp) -> QueueItem:
        from ..warp.tlrender import FORMATS, plan
        rows = plan(project, exp)
        n = sum(r["cut"].info.frames for r in rows if r["action"] != "skip")
        mode = {"full": "전체 길이", "each": "컷별", "replace": "리터칭 컷 + XML"}[exp.mode]
        it = QueueItem(id=uuid.uuid4().hex[:8], project_path=project.path, project_name=project.name, cut_id="",
                       cut_name=f"타임라인 · {mode}", output={"format": exp.format}, frames=n,
                       fmt_label=dict(FORMATS).get(exp.format, exp.format), kind="timeline", export=exp.to_dict())
        self.items.append(it)
        self.save()
        self.refresh()
        return it

    def _sel(self) -> Optional[int]:
        r = self.table.currentRow()
        return r if 0 <= r < len(self.items) else None

    def _sel_rows(self) -> List[int]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [r for r in rows if 0 <= r < len(self.items)]

    def _move(self, d):
        r = self._sel()
        if r is None:
            return
        j = r + d
        if 0 <= j < len(self.items):
            self.items[r], self.items[j] = self.items[j], self.items[r]
            self.save()
            self.refresh()
            self.table.selectRow(j)

    def _remove(self):
        rows = [r for r in self._sel_rows() if self.items[r].status != "running"]
        if not rows:
            return
        for r in reversed(rows):
            self._thumbs.pop(self.items[r].id, None)
            self.items.pop(r)
        self.save()
        self.refresh()

    def _clear_done(self):
        self.items = [i for i in self.items if i.status not in ("done", "cancelled")]
        self.save()
        self.refresh()

    def _open_dir(self):
        r = self._sel()
        path = None
        if r is not None:
            p_, _ = self._out_label(self.items[r])
            path = Path(p_.split(" (")[0].rstrip("\\")).parent
        elif self.items:
            o = self.items[-1].output.get("out_dir")
            path = Path(o) if o else Path(self.items[-1].project_path).parent / "DIAP_Renders"
        if path and path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        elif path:
            self.main.toast("아직 출력 폴더가 없습니다 (렌더할 때 만들어집니다)")

    # ------------------------------------------------------------------ path / codec
    def _editable(self, rows: List[int]):
        ok = [self.items[r] for r in rows if self.items[r].status in ("wait", "failed", "cancelled")]
        return ok, len(rows) - len(ok)

    def _cell_action(self, row: int, col: int):
        rows = self._sel_rows() or [row]
        if row not in rows:
            rows = [row]
        if col == self.C_PATH:
            self._change_path(rows)
        elif col == self.C_FMT:
            self._change_codec(rows)

    def _change_path(self, rows: List[int]):
        items, skipped = self._editable(rows)
        if not items:
            self.main.toast("렌더 중이거나 완료된 항목은 경로를 바꿀 수 없습니다", "warn")
            return
        first = items[0]
        start = (first.export.get("out_dir") if first.kind == "timeline" else first.output.get("out_dir")) or \
            str(Path(first.project_path).parent / "DIAP_Renders")
        title = "출력 폴더" + (f" ({len(items)}개 항목)" if len(items) > 1 else "")
        d = QFileDialog.getExistingDirectory(self, title, start)
        if not d:
            return
        for it in items:
            if it.kind == "timeline":
                it.export["out_dir"] = d
            else:
                it.output["out_dir"] = d
        self.save()
        self.refresh()
        msg = f"{len(items)}개 항목의 출력 폴더를 바꿨습니다 (파일 이름은 그대로)"
        if skipped:
            msg += f" · 렌더 중이거나 완료된 {skipped}개는 그대로"
        self.main.toast(msg, "ok")

    def _change_codec(self, rows: List[int]):
        from PySide6.QtWidgets import QMenu
        from ..core.project import OUTPUT_FORMATS
        items, skipped = self._editable(rows)
        items = [i for i in items if i.kind == "cut"]
        if not items:
            self.main.toast("렌더 중이거나 완료된 항목(또는 타임라인 출력)은 코덱을 바꿀 수 없습니다", "warn")
            return
        m = QMenu(self)
        cur = items[0].output.get("format", "")
        acts = {}
        for k, lab in OUTPUT_FORMATS:
            a = m.addAction(lab)
            a.setCheckable(True)
            a.setChecked(k == cur and len(items) == 1)
            acts[a] = (k, lab)
        act = m.exec(QCursor.pos())
        if act is None:
            return
        k, lab = acts[act]
        for it in items:
            it.output["format"] = k
            it.fmt_label = lab
        self.save()
        self.refresh()
        msg = f"{len(items)}개 항목의 코덱을 {lab}(으)로 바꿨습니다"
        if skipped:
            msg += f" · 렌더 중이거나 완료된 {skipped}개는 그대로"
        self.main.toast(msg, "ok")

    def _context(self, pos):
        from PySide6.QtWidgets import QMenu
        ix = self.table.indexAt(pos)
        if not ix.isValid():
            return
        rows = self._sel_rows()
        if ix.row() not in rows:
            self.table.selectRow(ix.row())
            rows = [ix.row()]
        m = QMenu(self)
        a_path = m.addAction("출력 폴더 바꾸기…")
        a_fmt = m.addAction("코덱 바꾸기…")
        a_open = m.addAction("출력 폴더 열기")
        m.addSeparator()
        a_rm = m.addAction("목록에서 제거")
        act = m.exec(self.table.viewport().mapToGlobal(pos))
        if act == a_path:
            self._change_path(rows)
        elif act == a_fmt:
            self._change_codec(rows)
        elif act == a_open:
            self._open_dir()
        elif act == a_rm:
            self._remove()

    def refresh(self):
        sel = self._sel_rows() if self.table.rowCount() == len(self.items) else []
        self.table.setRowCount(len(self.items))
        for r, it in enumerate(self.items):
            self._fill_info(it)
            col = {"done": C["ok"], "failed": C["error"], "running": C["accent"], "cancelled": C["text3"]}.get(it.status)
            ti = _item("")
            pm = self._thumb(it)
            if pm is not None:
                ti.setIcon(QIcon(pm.scaled(80, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.table.setItem(r, self.C_THUMB, ti)
            self.table.setItem(r, self.C_NO, _item(it.cut_no or ("TL" if it.kind == "timeline" else ""), color=C["text2"]))
            name = _item(it.cut_name)
            name.setToolTip(f"{it.cut_name}\n{it.project_name} · {it.frames}프레임")
            self.table.setItem(r, self.C_CUT, name)
            self.table.setItem(r, self.C_VER, _item(f"v{it.version}" if it.version else "", color=C["text2"]))
            fmt = _item(it.fmt_label + "  ▾", color=C["text"])
            fmt.setToolTip("눌러서 코덱 바꾸기 (여러 줄을 고른 상태면 모두)")
            self.table.setItem(r, self.C_FMT, fmt)
            self.table.setItem(r, self.C_CS, _item(it.src_cs or "-", color=C["text2"]))
            path, exists = self._out_label(it)
            pi = _item(("⚠ " if exists else "") + path, color=C["warn"] if exists else C["text2"])
            pi.setToolTip(path + ("\n같은 이름의 파일이 이미 있어 이름 뒤에 번호가 붙어 저장됩니다" if exists else "")
                          + "\n눌러서 출력 폴더 바꾸기 (파일 이름은 그대로)")
            self.table.setItem(r, self.C_PATH, pi)
            st = _item(STATUS_TEXT[it.status], color=col)
            if it.message:
                st.setToolTip(it.message)
            self.table.setItem(r, self.C_STATE, st)
            pb = QProgressBar()
            pb.setRange(0, 1000)
            pb.setValue(int(it.progress * 1000))
            pb.setFixedHeight(6)
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(8, 0, 8, 0)
            l.addWidget(pb)
            self.table.setCellWidget(r, self.C_PROG, w)
        if sel:
            from PySide6.QtCore import QItemSelectionModel
            sm = self.table.selectionModel()
            for r in sel:
                if r < len(self.items):
                    sm.select(self.table.model().index(r, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        busy = self.job is not None
        self.start.setEnabled(not busy and any(i.status == "wait" for i in self.items))
        self.stop.setEnabled(busy)
        self.empty.setVisible(not self.items)
        self._fill_now()
        self.main.update_queue_badge(sum(1 for i in self.items if i.status in ("wait", "running")))

    def _update_row(self, it: QueueItem):
        r = self.items.index(it) if it in self.items else -1
        if r < 0:
            return
        w = self.table.cellWidget(r, self.C_PROG)
        if w is not None:
            pb = w.findChild(QProgressBar)
            if pb:
                pb.setValue(int(it.progress * 1000))
        st = self.table.item(r, self.C_STATE)
        if st is not None:
            st.setToolTip(it.message)
        if it is self.current:
            self.now_bar.setValue(int(it.progress * 1000))
            self.now_msg.setText(it.message)

    # ------------------------------------------------------------------ running
    def start_queue(self):
        if self.job is not None:
            return
        nxt = next((i for i in self.items if i.status == "wait"), None)
        if nxt is None:
            self.refresh()
            if self.train_after.isChecked():
                self.main.models_page.train(auto=True)
            return
        if self.main.before_render(nxt) is False:
            nxt.status = "failed"
            nxt.message = "프로젝트를 저장하지 못해 렌더하지 않았습니다 (저장 후 다시 시작하세요)"
            self.refresh()
            return
        self.current = nxt
        nxt.status = "running"
        nxt.progress = 0
        nxt.message = "준비 중"

        def work(progress, cancelled):
            from ..core.project import OutputSettings, Project
            from ..core.proxy import Proxy
            from ..track.analyze import CutTracker
            from ..warp.render import render_cut
            prj = Project.load(nxt.project_path)
            if nxt.kind == "timeline":
                from ..warp.tlrender import TimelineExport, render_timeline
                return render_timeline(prj, TimelineExport.from_dict(nxt.export), progress, cancelled)
            cut = prj.cut(nxt.cut_id)
            if cut is None:
                raise RuntimeError("프로젝트에서 컷을 찾을 수 없습니다.")
            from ..core.frames import frames_for, source_present
            if not source_present(cut.info):
                raise RuntimeError("원본 파일을 찾을 수 없습니다: " + str(cut.source or cut.info.path))
            px = frames_for(prj, cut)              # the masks' frames: its proxy, else the original
            tr = CutTracker(prj, cut, px)
            out = OutputSettings.from_dict(nxt.output)
            return render_cut(prj, cut, tr, out, progress, cancelled,
                              preview=lambda t_, rgb: self.previewed.emit(nxt.id, t_, rgb))

        self.job = Job(work, self)
        self.job.progressed.connect(self._prog)
        self.job.succeeded.connect(self._done)
        self.job.failed.connect(self._failed)
        self.job.start()
        self.refresh()

    def _prog(self, f, msg):
        if self.current:
            self.current.progress = f
            if msg:
                self.current.message = msg
            self._update_row(self.current)

    def _done(self, rep):
        it = self.current
        self.job = None
        self.current = None
        if rep.cancelled:
            it.status = "cancelled"
            it.message = "취소됨"
            kept = [p for p in getattr(rep, "paths", []) if p]
            if it.kind == "timeline" and kept:
                # each/replace: the shots finished before the cancel stay; the one being written was removed
                it.message = f"취소됨 · 완료된 파일 {len(kept)}개는 남김"
        else:
            it.status = "done"
            it.progress = 1.0
            it.out_path = rep.out_path
            it.seconds = rep.seconds
            warn = f" · {rep.warnings[0]}" if rep.warnings else ""
            acc = f" · {rep.accel}" if getattr(rep, "accel", "") else ""
            it.message = f"{Path(rep.out_path).name if rep.out_path else '출력 없음'} · {rep.seconds:.0f}초{warn}{acc}"
            if it.kind == "timeline":
                it.message = (f"{len(rep.paths)}개 파일 · 리터칭 {rep.retouched} · 원본 {rep.original} · 투명 "
                              f"{rep.transparent} · 제외 {rep.skipped} · {rep.seconds:.0f}초{warn}")
            log_event("render", it.project_name, it.cut_name, frames=rep.frames, seconds=rep.seconds,
                      format=it.output.get("format"))
        self.save()
        self.jobFinished.emit(it, rep)
        self.refresh()
        QTimer.singleShot(200, self.start_queue)

    def _failed(self, err):
        it = self.current
        self.job = None
        self.current = None
        if err == "__cancelled__":
            it.status = "cancelled"
            it.message = "취소됨"
        else:
            it.status = "failed"
            lines = [l for l in err.splitlines() if l.strip()]
            it.message = (lines[-1] if lines else "알 수 없는 오류")[:200]
            # never block the queue with a dialog: the next item starts right away (overnight renders)
            _log_error(f"render failed: {it.project_name} / {it.cut_name}\n{err}")
            self.main.toast(f"렌더 실패: {it.cut_name} · {it.message}  (자세한 내용은 로그)", kind="error")
        self.save()
        self.refresh()
        QTimer.singleShot(200, self.start_queue)

    def cancel_current(self):
        if self.job:
            self.job.cancel()


def _log_error(text: str) -> None:
    try:
        import time as _t
        from ..core.settings import settings as _settings
        d = _settings().data_path / "logs"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "contour.log", "a", encoding="utf-8") as f:
            f.write(f"[{_t.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
    except Exception:
        pass


# =========================================================================== charts

class BarChart(QWidget):
    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.data: List[tuple] = []
        self.setMinimumHeight(220)

    def set_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc("line"), 1))
        p.setBrush(qc("bg1"))
        p.drawRoundedRect(r, 10, 10)
        p.setFont(font(12, 600))
        p.setPen(qc("text"))
        p.drawText(QRectF(16, 12, r.width() - 32, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        if not self.data:
            p.setPen(qc("text3"))
            p.setFont(font(12))
            p.drawText(r, Qt.AlignCenter, "아직 기록이 없습니다")
            return
        data = self.data[:10]
        vmax = max(v for _, v in data) or 1
        top = 44
        rowh = min(24, (r.height() - top - 12) / len(data))
        lw = 120
        for i, (lab, v) in enumerate(data):
            y = top + i * rowh
            p.setFont(font(11, 500))
            p.setPen(qc("text2"))
            fm = p.fontMetrics()
            p.drawText(QRectF(16, y, lw - 8, rowh - 4), Qt.AlignVCenter | Qt.AlignLeft,
                       fm.elidedText(str(lab), Qt.ElideRight, lw - 10))
            bw = (r.width() - lw - 90) * v / vmax
            p.setPen(Qt.NoPen)
            p.setBrush(qc("accent") if i == 0 else qc("bg4"))
            p.drawRoundedRect(QRectF(16 + lw, y + 4, max(2, bw), rowh - 10), 3, 3)
            p.setPen(qc("text"))
            p.setFont(font(11, 500, tabular=True))
            p.drawText(QRectF(16 + lw + bw + 8, y, 80, rowh - 4), Qt.AlignVCenter | Qt.AlignLeft, f"{v:.0f}{self.unit}")


class LineChart(QWidget):
    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.points: List[tuple] = []     # (label, value, active)
        self.setMinimumHeight(220)

    def set_points(self, pts):
        self.points = pts
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc("line"), 1))
        p.setBrush(qc("bg1"))
        p.drawRoundedRect(r, 10, 10)
        p.setFont(font(12, 600))
        p.setPen(qc("text"))
        p.drawText(QRectF(16, 12, r.width() - 32, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        pts = [x for x in self.points if x[1] is not None]
        if len(pts) < 1:
            p.setPen(qc("text3"))
            p.setFont(font(12))
            p.drawText(r, Qt.AlignCenter, "벤치마크를 등록하고 모델을 학습하면 정확도 추이가 보입니다")
            return
        vals = [v for _, v, _ in pts]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.25 or max(0.1, hi * 0.2)
        lo, hi = max(0, lo - pad), hi + pad
        L, R_, T, B = 48, r.width() - 20, 48, r.height() - 34
        p.setFont(font(10, 500, tabular=True))
        for k in range(3):
            v = lo + (hi - lo) * k / 2
            y = B - (v - lo) / (hi - lo) * (B - T)
            p.setPen(QPen(qc("line"), 1))
            p.drawLine(QPointF(L, y), QPointF(R_, y))
            p.setPen(qc("text3"))
            p.drawText(QRectF(4, y - 8, L - 8, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:.2f}{self.unit}")
        n = len(pts)
        xs = [L + (R_ - L) * (i / max(1, n - 1)) if n > 1 else (L + R_) / 2 for i in range(n)]
        ys = [B - (v - lo) / (hi - lo) * (B - T) for v in vals]
        path = QPainterPath(QPointF(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)
        p.setPen(QPen(qc("text2"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        for (lab, v, act), x, y in zip(pts, xs, ys):
            p.setPen(QPen(qc("bg1"), 2))
            p.setBrush(qc("accent") if act else qc("text2"))
            p.drawEllipse(QPointF(x, y), 5 if act else 3.5, 5 if act else 3.5)
            p.setPen(qc("text3") if not act else qc("text"))
            p.drawText(QRectF(x - 30, B + 6, 60, 16), Qt.AlignCenter, lab)


# =========================================================================== dashboard

class DashboardPage(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        sa.setWidget(inner)
        outer.addWidget(sa)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)
        head = QHBoxLayout()
        head.addWidget(_page_header("통계", "작업 시간, 수동 수정, 자동 추적 성공률을 프로젝트·컷 단위로 모읍니다. 정산과 견적 근거로 CSV를 내보낼 수 있습니다."), 1)
        b = Wd.button("CSV 내보내기", "", "download")
        b.clicked.connect(self._csv)
        r = Wd.button("새로고침", "ghost", "refresh-cw")
        r.clicked.connect(self.refresh)
        head.addWidget(r, 0, Qt.AlignTop)
        head.addWidget(b, 0, Qt.AlignTop)
        lay.addLayout(head)
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.t_cuts = Wd.StatTile("작업한 컷")
        self.t_time = Wd.StatTile("컷당 평균 작업 시간")
        self.t_auto = Wd.StatTile("자동 추적 성공률")
        self.t_corr = Wd.StatTile("100프레임당 수동 수정")
        self.t_model = Wd.StatTile("사용 중인 모델")
        for t in (self.t_cuts, self.t_time, self.t_auto, self.t_corr, self.t_model):
            tiles.addWidget(t)
        lay.addLayout(tiles)
        charts = QHBoxLayout()
        charts.setSpacing(12)
        self.c_proj = BarChart("프로젝트별 작업 시간", "분")
        self.c_model = LineChart("모델 버전별 벤치마크 오차 (낮을수록 정확)", "%")
        charts.addWidget(self.c_proj, 1)
        charts.addWidget(self.c_model, 1)
        lay.addLayout(charts)
        lay.addWidget(Wd.label("컷별 기록", "", 13, 600))
        self.table = _table(["프로젝트", "컷", "작업 시간", "수동 수정", "프레임", "자동 추적 성공률", "렌더", "마지막 작업"])
        for i, w in enumerate((170, 200, 90, 90, 80, 120, 60)):
            self.table.setColumnWidth(i, w)
        self.table.setMinimumHeight(300)
        lay.addWidget(self.table)
        lay.addStretch(1)

    def refresh(self):
        from ..learn import benchmark as BM
        from ..learn.refiner import active_version, list_versions
        self.main.flush_activity()
        ag = aggregate()
        self.t_cuts.set(f"{ag['total_cuts']}", f"총 {ag['total_frames']:,} 프레임 · 렌더 {ag['total_renders']}회")
        m = ag["avg_cut_minutes"]
        self.t_time.set(f"{m:.0f}분" if m is not None else "-", f"전체 {ag['total_active'] / 3600:.1f}시간")
        a = ag["auto_ok"]
        self.t_auto.set(f"{a * 100:.0f}%" if a is not None else "-", "사람 손을 거치기 전, 신뢰도 기준을 넘은 프레임 비율")
        c = ag["corrections_per_100"]
        self.t_corr.set(f"{c:.1f}" if c is not None else "-", "적을수록 자동 추적이 잘 맞는다는 뜻")
        av = active_version()
        vers = list_versions()
        cur = next((v for v in vers if v["version"] == av), None)
        sub = ""
        if cur and cur.get("bench_score") is not None:
            sub = f"벤치마크 오차 {cur['bench_score']:.2f}%"
        self.t_model.set(f"v{av:03d}" if av else "기본", sub or (f"학습 버전 {len(vers)}개" if vers else "아직 학습 전"))
        self.c_proj.set_data([(p["project"], p["active"] / 60.0) for p in sorted(ag["projects"], key=lambda x: -x["active"])])
        # benchmark trend: latest score per version
        hist = BM.history()
        best = {}
        for h in hist:
            s = BM.summary(h["results"])
            if s is not None:
                best[h["version"]] = s
        pts = [(f"v{v:03d}" if v else "기본", best[v], v == av) for v in sorted(best)]
        self.c_model.set_points(pts)
        rows = ag["cuts"]
        self.table.setRowCount(len(rows))
        for r, c in enumerate(rows):
            self.table.setItem(r, 0, _item(c["project"], color=C["text2"]))
            self.table.setItem(r, 1, _item(c["cut"]))
            self.table.setItem(r, 2, _item(f"{c['active'] / 60:.0f}분", Qt.AlignRight))
            self.table.setItem(r, 3, _item(c["corrections"], Qt.AlignRight))
            self.table.setItem(r, 4, _item(c["frames"], Qt.AlignRight))
            self.table.setItem(r, 5, _item("-" if c["auto_ok"] is None else f"{c['auto_ok'] * 100:.0f}%", Qt.AlignRight))
            self.table.setItem(r, 6, _item(c["renders"], Qt.AlignRight))
            self.table.setItem(r, 7, _item(time.strftime("%Y-%m-%d %H:%M", time.localtime(c["last"])), color=C["text2"]))

    def _csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV로 내보내기", str(Path.home() / f"DIAP_통계_{time.strftime('%Y%m%d')}.csv"),
                                              "CSV (*.csv)")
        if path:
            export_csv(path)
            self.main.toast(f"저장했습니다: {Path(path).name}", kind="ok")


# =========================================================================== models

class ModelsPage(QWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.job: Optional[Job] = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        sa.setWidget(inner)
        outer.addWidget(sa)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)
        lay.addWidget(_page_header(
            "누적 학습",
            "작업하면서 고친 관절과 확정한 구간이 학습 데이터로 쌓입니다. 새 모델은 따로 떼어 둔 데이터와 벤치마크 컷에서 "
            "현재 모델보다 나을 때만 적용되고, 언제든 이전 버전으로 되돌릴 수 있습니다. 데이터는 이 컴퓨터 밖으로 나가지 않습니다."))
        top = QHBoxLayout()
        top.setSpacing(12)
        self.t_active = Wd.StatTile("사용 중인 모델")
        self.t_data = Wd.StatTile("학습 데이터")
        self.t_bench = Wd.StatTile("벤치마크 컷")
        for t in (self.t_active, self.t_data, self.t_bench):
            top.addWidget(t)
        lay.addLayout(top)
        bar = QHBoxLayout()
        self.train_btn = Wd.button("새 모델 학습", "primary", "brain")
        self.train_btn.clicked.connect(lambda: self.train(False))
        self.eval_btn = Wd.button("벤치마크 다시 측정", "", "gauge")
        self.eval_btn.clicked.connect(self.evaluate)
        self.collect = QCheckBox("작업하면서 학습 데이터 모으기")
        self.collect.toggled.connect(self._collect)
        self.selfl = QCheckBox("분석한 영상으로 자가학습")
        self.selfl.setToolTip("분석할 때마다 확실하게 추적된 관절을 스스로 정답으로 삼아 학습 데이터에 넣고,\n"
                              "새 영상이 충분히 쌓이면 자동으로 새 모델을 학습합니다.\n"
                              "새 모델은 플래시·색 조명·모션블러를 씌운 화면과 직접 고친 관절에서\n"
                              "모두 더 정확할 때만 적용됩니다.")
        self.selfl.toggled.connect(self._selfl)
        self.status = Wd.label("", "Muted")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(220)
        self.bar.hide()
        bar.addWidget(self.train_btn)
        bar.addWidget(self.eval_btn)
        bar.addWidget(self.bar)
        bar.addWidget(self.status, 1)
        bar.addWidget(self.selfl)
        bar.addWidget(self.collect)
        lay.addLayout(bar)
        lay.addWidget(Wd.label("버전", "", 13, 600))
        self.versions = _table(["버전", "만든 날", "샘플", "검증 오차 (전→후)", "벤치마크 오차", "결정", ""])
        for i, w in enumerate((70, 140, 70, 160, 120, 120)):
            self.versions.setColumnWidth(i, w)
        self.versions.setMinimumHeight(200)
        lay.addWidget(self.versions)
        row = QHBoxLayout()
        self.activate = Wd.button("선택한 버전 사용", "", "check")
        self.activate.clicked.connect(self._activate)
        self.base = Wd.button("기본 모델로 되돌리기", "ghost", "refresh-cw")
        self.base.clicked.connect(lambda: self._set_active(0))
        self.delete = Wd.button("선택한 버전 삭제", "ghost", "trash-2")
        self.delete.clicked.connect(self._delete)
        row.addWidget(self.activate)
        row.addWidget(self.base)
        row.addStretch(1)
        row.addWidget(self.delete)
        lay.addLayout(row)
        two = QHBoxLayout()
        two.setSpacing(16)
        left = QVBoxLayout()
        left.addWidget(Wd.label("학습 데이터 (프로젝트별)", "", 13, 600))
        self.data = _table(["프로젝트", "컷", "샘플"])
        self.data.setColumnWidth(0, 180)
        self.data.setColumnWidth(1, 180)
        self.data.setMinimumHeight(200)
        left.addWidget(self.data)
        ex = Wd.button("선택한 프로젝트 데이터 빼기", "ghost", "trash-2")
        ex.clicked.connect(self._exclude)
        left.addWidget(ex, 0, Qt.AlignLeft)
        right = QVBoxLayout()
        right.addWidget(Wd.label("벤치마크 컷", "", 13, 600))
        self.bench = _table(["컷", "프로젝트", "프레임", "관절 키"])
        self.bench.setColumnWidth(0, 180)
        self.bench.setColumnWidth(1, 160)
        self.bench.setMinimumHeight(200)
        right.addWidget(self.bench)
        rb = Wd.button("선택한 벤치마크 제거", "ghost", "trash-2")
        rb.clicked.connect(self._remove_bench)
        right.addWidget(rb, 0, Qt.AlignLeft)
        lw, rw = QWidget(), QWidget()
        lw.setLayout(left)
        rw.setLayout(right)
        two.addWidget(lw, 1)
        two.addWidget(rw, 1)
        lay.addLayout(two)
        lay.addStretch(1)

    def refresh(self):
        from ..learn import benchmark as BM
        from ..learn import refiner as RF
        av = RF.active_version()
        vers = RF.list_versions()
        cur = next((v for v in vers if v["version"] == av), None)
        self.t_active.set(f"v{av:03d}" if av else "기본",
                          (f"벤치마크 오차 {cur['bench_score']:.2f}%" if cur and cur.get("bench_score") is not None
                           else "포즈 앙상블만 사용 중" if not av else ""))
        ds = RF.dataset_summary()
        self.t_data.set(f"{ds['samples']:,}", f"{ds['cuts']}개 컷 · {ds['projects']}개 프로젝트 · 자가학습 {ds.get('pseudo', 0):,}")
        benches = BM.list_benchmarks()
        self.t_bench.set(f"{len(benches)}", "새 모델을 평가하는 고정 기준")
        try:
            from ..learn import fusion as FU
            fs = FU.status()
            self.selfl.setText("분석한 영상으로 자가학습" + (
                f" · 모델 신뢰도 학습 적용 중 ({fs['cuts']}개 영상, {fs['gain'] * 100:+.1f}%)" if fs else ""))
        except Exception:
            pass
        self.collect.blockSignals(True)
        self.collect.setChecked(settings().collect_learning_data)
        self.collect.blockSignals(False)
        self.selfl.blockSignals(True)
        self.selfl.setChecked(bool(getattr(settings(), "self_learning", True)))
        self.selfl.blockSignals(False)
        torch_ok = RF.torch_available()
        busy = self.job is not None
        self.train_btn.setEnabled(torch_ok and ds["samples"] >= 40 and not busy)
        self.eval_btn.setEnabled(bool(benches) and not busy)
        if not busy:
            if not torch_ok:
                self.status.setText("학습에는 PyTorch가 필요합니다. install.bat을 다시 실행해 '학습 기능'을 설치하세요.")
            elif ds["samples"] < 40:
                self.status.setText(f"학습 데이터가 {ds['samples']}개입니다. 40개 이상 모이면 학습할 수 있습니다.")
            else:
                self.status.setText("")
        self.versions.setRowCount(len(vers))
        for r, v in enumerate(reversed(vers)):
            act = v["version"] == av
            self.versions.setItem(r, 0, _item(f"v{v['version']:03d}", color=C["accent"] if act else None, data=v["version"]))
            self.versions.setItem(r, 1, _item(time.strftime("%Y-%m-%d %H:%M", time.localtime(v.get("created", 0))), color=C["text2"]))
            self.versions.setItem(r, 2, _item(v.get("samples", ""), Qt.AlignRight))
            vb, vm = v.get("val_baseline"), v.get("val_model")
            self.versions.setItem(r, 3, _item(f"{vb * 100:.1f}% → {vm * 100:.1f}%" if vb and vm is not None else "-"))
            bs = v.get("bench_score")
            self.versions.setItem(r, 4, _item(f"{bs:.2f}%" if bs is not None else "-", Qt.AlignRight))
            dec = {"activated": "자동 적용", "kept": "보관 (개선 없음)"}.get(v.get("decision", ""), "")
            self.versions.setItem(r, 5, _item(dec, color=C["text2"]))
            self.versions.setItem(r, 6, _item("사용 중" if act else "", color=C["accent"]))
        rows = ds["rows"]
        self.data.setRowCount(len(rows))
        for r, d in enumerate(rows):
            self.data.setItem(r, 0, _item(d["project"], data=d["project"]))
            self.data.setItem(r, 1, _item(d["cut"], color=C["text2"]))
            self.data.setItem(r, 2, _item(d["n"], Qt.AlignRight))
        self.bench.setRowCount(len(benches))
        for r, b in enumerate(benches):
            self.bench.setItem(r, 0, _item(b["cut"], data=b["id"]))
            self.bench.setItem(r, 1, _item(b["project"], color=C["text2"]))
            self.bench.setItem(r, 2, _item(b["frames"], Qt.AlignRight))
            self.bench.setItem(r, 3, _item(b.get("keys", 0), Qt.AlignRight))

    def _collect(self, v):
        s = settings()
        s.collect_learning_data = bool(v)
        s.save()

    def _selfl(self, v):
        s = settings()
        s.self_learning = bool(v)
        s.save()

    def maybe_self_train(self):
        """After an analysis: train a new version on its own once enough new footage has been seen."""
        from ..learn import refiner as RF
        if not getattr(settings(), "self_learning", True) or self.job is not None or not RF.torch_available():
            return False
        try:
            n, k = RF.new_auto_since_last()
            if RF.dataset_summary()["samples"] < 40 or (n < 1500 and k < 5):
                return False
        except Exception:
            return False
        self.main.toast(f"새 영상 {k}개로 자가학습을 시작합니다 (백그라운드, 계속 작업해도 됩니다)")
        self.train(auto=True)
        return True

    def _sel_version(self) -> Optional[int]:
        r = self.versions.currentRow()
        it = self.versions.item(r, 0) if r >= 0 else None
        return it.data(Qt.UserRole) if it else None

    def _set_active(self, v):
        from ..learn import refiner as RF
        RF.set_active(v)
        self.main.toast(f"{'v%03d' % v if v else '기본 모델'}을(를) 사용합니다. 다음 분석·재계산부터 적용됩니다.", kind="ok")
        self.refresh()

    def _activate(self):
        v = self._sel_version()
        if v is not None:
            self._set_active(v)

    def _delete(self):
        from ..learn import refiner as RF
        v = self._sel_version()
        if v is None:
            return
        if QMessageBox.question(self, "버전 삭제", f"v{v:03d}을(를) 삭제할까요? 되돌릴 수 없습니다.") == QMessageBox.Yes:
            RF.delete_version(v)
            self.refresh()

    def _exclude(self):
        from ..learn import refiner as RF
        r = self.data.currentRow()
        it = self.data.item(r, 0) if r >= 0 else None
        if it is None:
            return
        name = it.data(Qt.UserRole)
        if QMessageBox.question(self, "학습 데이터 빼기",
                                f"'{name}' 프로젝트의 학습 데이터를 지웁니다. 이미 학습된 모델에는 영향이 없습니다.") == QMessageBox.Yes:
            RF.remove_project_samples(name)
            self.refresh()

    def _remove_bench(self):
        from ..learn import benchmark as BM
        r = self.bench.currentRow()
        it = self.bench.item(r, 0) if r >= 0 else None
        if it is None:
            return
        BM.remove(it.data(Qt.UserRole))
        self.refresh()

    def _run(self, fn, title, done):
        self.bar.show()
        self.bar.setValue(0)
        self.status.setText(title)
        self.job = Job(fn, self)
        self.job.progressed.connect(lambda f, m: (self.bar.setValue(int(f * 1000)), self.status.setText(m or title)))
        self.job.succeeded.connect(lambda res: self._finish(res, done))
        self.job.failed.connect(self._fail)
        self.job.start()
        self.refresh()

    def _finish(self, res, done):
        self.job = None
        self.bar.hide()
        done(res)
        self.refresh()

    def _fail(self, err):
        self.job = None
        self.bar.hide()
        self.status.setText("")
        if err != "__cancelled__":
            self.main.show_error("학습 실패", err)
        self.refresh()

    def train(self, auto: bool = False):
        from ..learn import manager as MG
        from ..learn import refiner as RF
        if self.job is not None or not RF.torch_available():
            return
        if RF.dataset_summary()["samples"] < 40:
            return
        self.main.collect_learning_all()

        def done(meta):
            if meta.get("decision") == "activated":
                msg = f"v{meta['version']:03d}이(가) 더 정확해서 적용했습니다."
            else:
                msg = f"v{meta['version']:03d}을(를) 학습했지만 현재 모델보다 낫지 않아 보관만 했습니다."
            self.status.setText(msg)
            self.main.toast(msg, kind="ok" if meta.get("decision") == "activated" else "")

        self._run(lambda prog, cancel: MG.train_and_compare(prog, cancel), "학습 중", done)

    def evaluate(self):
        from ..learn import benchmark as BM
        from ..learn import refiner as RF

        def work(prog, cancel):
            av = RF.active_version()
            res = BM.evaluate(av, prog)
            score = BM.summary(res)
            if av:
                RF.update_meta(av, bench=res, bench_score=score)
            return score

        self._run(work, "벤치마크 측정 중",
                  lambda s: self.status.setText(f"현재 모델 벤치마크 오차 {s:.2f}% (인물 키 대비)" if s is not None else ""))
