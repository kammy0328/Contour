"""Main window and controller."""
from __future__ import annotations

import copy
import threading
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QObject, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPushButton, QSplitter, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from . import icons as I
from . import widgets as Wd
from .cutlist import CutList
from .dialogs import CopyDialog, MergeDialog, SettingsDialog, TextPrompt, error_box
from .inspector import Inspector
from .pages import DashboardPage, ModelsPage, RenderQueuePage
from .session import EditSession
from .theme import C, font
from .timeline import Timeline
from .viewer import Viewer
from .workers import Job, SerialWorker
from ..core import media as M
from ..core.colour import manager
from ..core.copyset import CopyOptions, apply_copy, suggest_mapping
from ..core.project import PART_KEYS, PERSON_COLORS, Project, safe_name
from ..core.params import BODY_SKIN_KEYS
from ..core.frames import SCRUB_SEEKS
from ..core.settings import APP_NAME, APP_VERSION, settings
from ..stats.tracker import ActivityClock, log_event
from ..track import skeleton as S

VIDEO_FILTER = "영상·시퀀스 (*.mov *.mp4 *.mxf *.mkv *.avi *.m4v *.webm *.mts *.exr *.dpx *.tif *.tiff *.png);;모든 파일 (*)"


class _ActivityFilter(QObject):
    def __init__(self, win):
        super().__init__()
        self.win = win

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.MouseButtonPress, QEvent.KeyPress, QEvent.Wheel):
            self.win.activity.touch()
        if ev.type() == QEvent.ShortcutOverride and isinstance(obj, QLineEdit):
            # Ctrl+Z / redo go to the project, not to a value box (only a name field being typed keeps it)
            if ev.matches(QKeySequence.Undo) or ev.matches(QKeySequence.Redo) or \
                    (ev.key() == Qt.Key_Y and ev.modifiers() & Qt.ControlModifier):
                if obj.objectName() == "ParamVal" or not obj.isModified():
                    return True
        return False


def _load_project(path: str, long_side: int, progress):
    """The project (project.json), read off the main thread. 1.22.1: nothing else. It used to check every cut's
    proxy and read every thumbnail before returning, so a big project stayed locked until all were read; the
    thumbnails now come in the background once the project is shown (ui/thumbs.py)."""
    prj = Project.load(path)
    progress(1.0, f"프로젝트 읽음 · 컷 {len(prj.cuts)}개")
    return prj


class _JobGuard(QObject):
    """While a job runs in this window (analysis, proxy, merge...) or a project is being opened, clicks, keys and
    shortcuts anywhere in the window except the progress panel (its cancel button) are swallowed: the job works on
    the open cut, and switching tabs or pressing buttons meanwhile changed it under the job's feet."""
    BLOCK = (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease, QEvent.KeyPress,
             QEvent.KeyRelease, QEvent.Shortcut, QEvent.Wheel, QEvent.ContextMenu, QEvent.Drop)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._told = 0.0

    def busy(self) -> bool:
        w = self.win
        return getattr(w, "job", None) is not None or getattr(w, "_opening", None) is not None

    def eventFilter(self, obj, ev):
        if ev.type() not in self.BLOCK or not self.busy():
            return False
        w = self.win
        if isinstance(obj, (QShortcut, QAction)):
            return QApplication.activeModalWidget() is None
        if not isinstance(obj, QWidget):
            return False
        if obj.window() is not w:
            return False                    # dialogs, message boxes, the batch window: their own business
        busy = getattr(w, "busy", None)
        if busy is not None and (obj is busy or busy.isAncestorOf(obj)):
            return False                    # the progress panel itself (cancel)
        if ev.type() == QEvent.Wheel and getattr(w, "viewer", None) is not None and \
                (obj is w.viewer or w.viewer.isAncestorOf(obj)):
            return False                    # looking around in the picture changes nothing
        return True                         # blocked without a message: the progress panel says what is going on


class _CenterFilter(QObject):
    """Keeps a child widget in the middle of its parent (horizontally and vertically)."""
    def __init__(self, parent, child):
        super().__init__(parent)
        self.child = child
        self.place()

    def place(self):
        bar = self.parent()
        c = self.child
        w, h = c.width(), c.height()
        c.setGeometry((bar.width() - w) // 2, (bar.height() - h) // 2, w, h)
        c.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Resize, QEvent.Show):
            self.place()
        return False


class _ProjectOpen(QThread):
    progressed = Signal(float, str)
    loaded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, parent, path: str, long_side: int):
        super().__init__(parent)
        self.path, self.long_side = path, long_side

    def run(self):
        try:
            t0 = time.perf_counter()
            prj = _load_project(self.path, self.long_side, lambda f, m: self.progressed.emit(f, m))
            prj._open_times = {"load": time.perf_counter() - t0}
            self.loaded.emit(prj, None)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


def resolve_task(session, pid: int, t: int = 0, frame_end: Optional[int] = None, every: bool = False):
    """A joint re-solve for the edit worker, bound to the cut open now. If the user has gone to another cut by the
    time it runs, it still re-solves (and saves) that cut's own tracker, never the one open then."""
    tr = session.tracker

    def run():
        if tr is None or tr.store is None:
            return None
        from ..track import matte_refresh as MR
        before = MR.prompt_state(tr)
        rng = tr.resolve_all() if every else tr.resolve_local(pid, t, frame_end=frame_end)
        marks = []
        try:
            # the frames whose person-matte prompts the re-solve changed (§4: matted again in the background)
            marks = MR.changed(tr, before, None if every or rng is None else rng)
        except Exception:
            traceback.print_exc()
        if session.tracker is tr:
            session.touch()
        else:
            tr.save()                       # its cut was saved and closed before this ran
        cut = getattr(tr, "cut", None)
        return dict(cut=getattr(cut, "id", None), rng=rng, marks=marks)
    return run


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from .thumbs import ThumbCache
        # one thumbnail per cut for the cut list and the timeline strip, filled in the background (1.22.1)
        self.thumbcache = ThumbCache(self)
        self.setWindowTitle(APP_NAME)
        self.session = EditSession()
        self.frame = 0
        self.sel_pid: Optional[int] = None
        self.selection: Optional[Tuple[int, int]] = None
        self.tool = "edit"
        self.accuracy = "max"                  # always the most precise analysis (no choice in the UI)
        self.activity = ActivityClock()
        self.job: Optional[Job] = None
        self.undo_stack: List[dict] = []
        self.redo_stack: List[dict] = []
        self.edit_worker = SerialWorker(self)
        self.edit_worker.done.connect(self._edit_done)
        self.edit_worker.error.connect(lambda e: self.show_error("재계산 오류", e))
        self.edit_worker.busy.connect(self._edit_busy)
        self._build()
        self._shortcuts()
        # the person matte follows the edits by itself (§4)
        from .matte_refresh import MatteRefresher, TRACK_TAGS
        self.refresher = MatteRefresher(self)
        self.refresher.changed.connect(self._matte_state_changed)
        self.edit_worker.submitted.connect(
            lambda tag: self.refresher.hold() if isinstance(tag, tuple) and tag and tag[0] in TRACK_TAGS else None)
        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.PreciseTimer)
        self.play_timer.timeout.connect(self._tick)
        self._play_clock = (0.0, 0, 24.0)          # start time, frames shown since, fps
        self._undo_by_cut: dict = {}               # cut id -> (undo, redo) of the cuts left during this session
        # results computed ahead for smooth playback
        from .workers import CacheWorker
        self.cache_worker = CacheWorker(self.session, self)
        from .workers import NextCutPrep, ProxyBuilder
        self.next_prep = NextCutPrep()
        self.proxy_builder = ProxyBuilder(self)
        self.proxy_builder.finished.connect(self._bg_proxy_done)
        self.proxy_builder.progressed.connect(self._bg_proxy_progress)
        self._px_making = None                     # ((project path, cut id), percent) of the proxy being made
        self._px_failed: dict = {}                 # (project path, cut id) -> why its proxy could not be made
        self.cache_worker.progressed.connect(lambda: self.timeline.update())
        self.cache_timer = QTimer(self)
        self.cache_timer.timeout.connect(lambda: self.cache_worker.kick(self.frame) if self.session.cut else None)
        self.cache_timer.start(1500)
        self.autosave = QTimer(self)
        self.autosave.timeout.connect(self._autosave)
        self.apply_autosave_settings()
        # Windows shutting down / logging off: save now, nothing to ask
        try:
            QApplication.instance().commitDataRequest.connect(self._emergency_save)
            QApplication.instance().setFallbackSessionManagementEnabled(False)
        except Exception:
            pass
        self._filter = _ActivityFilter(self)
        QApplication.instance().installEventFilter(self._filter)
        self.setAcceptDrops(True)
        self.resize(1600, 960)
        QTimer.singleShot(50, self._startup)

    # ================================================================== layout
    def _build(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._topbar())
        self.pages = QStackedWidget()
        v.addWidget(self.pages, 1)
        # ---- edit page
        edit = QWidget()
        h = QHBoxLayout(edit)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.cutlist = CutList(self.session)
        self.cutlist.setMinimumWidth(250)
        self.cutlist.setMaximumWidth(340)
        self.cutlist.cutSelected.connect(self.open_cut)
        self.cutlist.addRequested.connect(self.add_cuts)
        self.cutlist.action.connect(self._cut_action)
        self.cutlist.menuRequested.connect(self._cut_menu)
        self.cutlist.stateRequested.connect(self._state_menu)
        split.addWidget(self.cutlist)
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self._viewer_bar())
        self.viewer = Viewer(self.session)
        self.viewer.hover_pick = bool(settings().hover_pick)
        self.viewer.jointDragged.connect(self._joint_drag)
        self.viewer.jointCommitted.connect(self._joint_commit)
        self.viewer.jointKeyDeleted.connect(self._joint_delete)
        self.viewer.personClicked.connect(self.select_person)
        self.viewer.contextAction.connect(self._viewer_context)
        self.viewer.personAction.connect(self._person_action)
        self.viewer.editStarted.connect(lambda: self.push_undo())
        self.viewer.sculptEdited.connect(self._sculpt_edit)
        self.viewer.bodyEdited.connect(self._body_edit)
        self.viewer.bodyMarked.connect(self._body_mark)
        self.viewer.bodyMarkDeleted.connect(self._body_mark_delete)
        cv.addWidget(self.viewer, 1)
        cv.addWidget(self._transport())
        vs = QSplitter(Qt.Vertical)
        self.timeline = Timeline(self.session)
        self.timeline.seek.connect(self.set_frame)
        self.timeline.rangeSelected.connect(self._range_selected)
        self.timeline.keyMoved.connect(self._key_moved)
        self.timeline.keyDeleted.connect(self._key_deleted)
        tl_wrap = QWidget()
        tl_wrap.setObjectName("Panel")
        tl_wrap.setAttribute(Qt.WA_StyledBackground, True)
        tl = QVBoxLayout(tl_wrap)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.timeline)
        cv.addWidget(Wd.hline())
        from .tlstrip import TimelineStrip
        from .batchrun import BatchRunner
        self.tlstrip = TimelineStrip(self.session)
        self.tlstrip.cutSelected.connect(self.open_cut)
        self.tlstrip.action.connect(self._tl_action)
        self.tlstrip.menuRequested.connect(self._cut_menu)
        self.tlstrip.stateRequested.connect(self._state_menu)
        self.tlstrip.versionPicked.connect(self.switch_version)
        self.tlstrip.hide()
        tc = self.thumbcache
        tc.vs_for = self.session.view_settings
        self.cutlist.thumbs = self.tlstrip.thumbs = tc.images      # one decoded picture per cut for both views
        self.cutlist.thumbsWanted.connect(self._want_thumbs)
        self.tlstrip.thumbsWanted.connect(self._want_thumbs)
        tc.ready.connect(lambda _cid: (self.cutlist.update_rows(), self.tlstrip.list.viewport().update()))
        cv.addWidget(self.tlstrip)
        cv.addWidget(tl_wrap)
        self.tl_wrap = tl_wrap
        self.split = split
        self.batch = BatchRunner(self)
        self.batch.before_start = self._batch_before
        self.batch.progressed.connect(self._batch_progress)
        self.batch.finishedCut.connect(self._batch_cut_done)
        self.batch.finishedAll.connect(self._batch_all_done)
        self.batch.warned.connect(self._batch_warned)
        self.batch.skipped.connect(self._batch_skipped)
        self.batch.scanProgress.connect(lambda f, m: self.tlstrip.set_running(dict(self.tlstrip.running), m))
        self._batch_preset = None
        self._batch_dlg = None
        self._batch_skin = None
        self._review_pos = -1
        self._last_export = {}
        split.addWidget(center)
        self.inspector = Inspector(self)
        split.addWidget(self.inspector)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([280, 1000, 360])
        h.addWidget(split)
        self.center = center
        self.busy = Wd.BusyOverlay(center)
        self._job_guard = _JobGuard(self)
        QApplication.instance().installEventFilter(self._job_guard)
        self.busy.cancelRequested.connect(self._cancel_job)
        self.pages.addWidget(edit)
        # ---- other pages
        self.queue_page = RenderQueuePage(self)
        self.queue_page.jobFinished.connect(self._render_finished)
        self.dash_page = DashboardPage(self)
        self.models_page = ModelsPage(self)
        self.pages.addWidget(self.queue_page)
        self.pages.addWidget(self.dash_page)
        self.pages.addWidget(self.models_page)
        from .projectmgr import ProjectsPage
        self.projects_page = ProjectsPage(self)
        self.pages.addWidget(self.projects_page)
        self.toast_w = Wd.Toast(root)

    def _topbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(48)
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 0, 12, 0)
        h.setSpacing(10)
        mark = QLabel("DIAP")
        mark.setFont(font(15, 700))
        mark.setStyleSheet(f"color: {C['text']}; letter-spacing: 0.2px;")
        h.addWidget(mark)
        self.proj_btn = QPushButton("프로젝트 없음")
        self.proj_btn.setProperty("kind", "ghost")
        self.proj_btn.setIcon(I.icon("chevron-down", 13))
        self.proj_btn.setLayoutDirection(Qt.RightToLeft)
        self.proj_btn.setCursor(Qt.PointingHandCursor)
        self.proj_btn.clicked.connect(self._project_menu)
        h.addWidget(self.proj_btn)
        h.addStretch(1)
        # statistics and cumulative learning run on their own in the background; no tabs for them
        self.page_tabs = Wd.Segmented([("edit", "편집"), ("queue", "렌더")])
        self.page_tabs.setFixedWidth(200)
        self.page_tabs.setFixedHeight(30)
        self.page_tabs.changed.connect(self._page)
        # the tabs sit in the exact middle of the bar, whatever the widths on either side do (project name, save
        # state, device): not in the layout, placed on every resize
        self.page_tabs.setParent(bar)
        self._bar_center = _CenterFilter(bar, self.page_tabs)
        bar.installEventFilter(self._bar_center)
        self.saved_lbl = Wd.SaveIndicator()
        h.addWidget(self.saved_lbl)
        self.dev_lbl = QLabel("")
        self.dev_lbl.setObjectName("Faint")
        h.addWidget(self.dev_lbl)
        gear = Wd.tool("settings", "설정")
        gear.clicked.connect(self.open_settings)
        h.addWidget(gear)
        self._top_items = [self.proj_btn, self.page_tabs, self.saved_lbl, self.dev_lbl, gear]   # hidden on 프로젝트
        return bar

    def _viewer_bar(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        w.setAttribute(Qt.WA_StyledBackground, True)
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(4)
        self.tool_btns = {}
        # 1.22.2: the protection drawing tools (line, ellipse, rectangle) were removed with the 보호 tab
        for key, ic, tip in (("edit", "mouse-pointer-2", "관절 편집·선택  V"),):
            b = Wd.tool(ic, tip, checkable=True)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_btns[key] = b
            h.addWidget(b)
        self.tool_btns["edit"].setChecked(True)
        h.addWidget(Wd.vline())
        self.compare = Wd.Segmented([("result", "결과"), ("orig", "원본"), ("side", "비교"), ("wipe", "와이프")], compact=True)
        self.compare.setFixedWidth(260)
        self.compare.setToolTip("\\ 키: 결과/원본 빠르게 전환 (A/B)")
        self.compare.changed.connect(self._compare)
        h.addSpacing(6)
        h.addWidget(self.compare)
        h.addSpacing(6)
        h.addWidget(Wd.vline())
        self.ov = {}
        for key, ic, tip, on in (("skel", "bone", "스켈레톤  S", True),
                                 ("matte", "layers", "인물 매트  M", False),
                                 ("skin", "wand-sparkles", "피부 보정 영역  K", False)):
            b = Wd.tool(ic, tip, checkable=True)
            b.setChecked(on)
            b.toggled.connect(lambda v, k=key: self._overlay(k, v))
            self.ov[key] = b
            h.addWidget(b)
        self.hover_btn = Wd.tool("mouse-pointer-click", "마우스를 올린 인물 강조·선택 켜기/끄기", checkable=True)
        self.hover_btn.setChecked(bool(settings().hover_pick))
        self.hover_btn.toggled.connect(self._hover_pick)
        h.addWidget(self.hover_btn)
        h.addStretch(1)
        self.px_lbl = QLabel("")                 # where the open cut's frames come from; a proxy being made or failed
        self.px_lbl.setObjectName("Faint")
        h.addWidget(self.px_lbl)
        self.cs_lbl = QLabel("")
        self.cs_lbl.setObjectName("Faint")
        h.addWidget(self.cs_lbl)
        fit = Wd.tool("maximize", "화면에 맞추기  Z")
        fit.clicked.connect(lambda: self.viewer.fit_view())
        h.addWidget(fit)
        foc = Wd.tool("columns-2", "오른쪽 패널과 미리보기만  Shift+F")
        foc.clicked.connect(self.toggle_focus)
        h.addWidget(foc)
        full = Wd.tool("expand", "전체 화면  Ctrl+F")
        full.clicked.connect(self.toggle_fullscreen)
        h.addWidget(full)
        return w

    def _transport(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        w.setAttribute(Qt.WA_StyledBackground, True)
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 6, 12, 6)
        h.setSpacing(2)
        self.tc = QLabel("00:00:00:00")
        self.tc.setFont(font(15, 600, tabular=True))
        self.tc.setMinimumWidth(118)
        h.addWidget(self.tc)
        self.fr_lbl = QLabel("")
        self.fr_lbl.setObjectName("Faint")
        self.fr_lbl.setFont(font(11, 500, tabular=True))
        h.addWidget(self.fr_lbl)
        h.addStretch(1)
        # the play buttons sit in the exact middle of the bar (the timecode and frame count on the left change width
        # while playing; in the layout they pushed the buttons sideways)
        grp = QWidget(w)
        gl = QHBoxLayout(grp)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(2)
        for ic, tip, fn in (("skip-back", "처음  Home", lambda: self.set_frame(self.session.cut.work_range[0] if self.session.cut else 0)),
                            ("step-back", "이전 프레임  ←", lambda: self.step(-1)),
                            ("play", "재생/정지  Space", self.toggle_play),
                            ("step-forward", "다음 프레임  →", lambda: self.step(1)),
                            ("skip-forward", "끝  End", lambda: self.set_frame((self.session.cut.work_range[1] - 1) if self.session.cut else 0))):
            b = Wd.tool(ic, tip, size=17)
            b.clicked.connect(fn)
            gl.addWidget(b)
            if ic == "play":
                self.play_btn = b
        gl.addSpacing(8)
        self.loop_btn = Wd.tool("repeat", "구간 반복: 켜면 이 컷만 반복 재생, 끄면 다음 컷으로 이어서 재생", checkable=True, size=16)
        self.loop_btn.setChecked(bool(getattr(settings(), "play_loop", False)))
        self.loop_btn.toggled.connect(self._loop_toggled)
        gl.addWidget(self.loop_btn)
        grp.adjustSize()
        grp.setFixedSize(grp.sizeHint())
        self._play_center = _CenterFilter(w, grp)
        w.installEventFilter(self._play_center)
        # one pill: ‹ 확인 필요 ›
        pill = QWidget()
        pill.setObjectName("Pill")
        pill.setAttribute(Qt.WA_StyledBackground, True)
        pill.setStyleSheet(f"QWidget#Pill {{ background: {C['bg2']}; border: 1px solid {C['line2']}; border-radius: 13px; }}"
                           "QToolButton { border: none; padding: 2px; }")
        pl = QHBoxLayout(pill)
        pl.setContentsMargins(4, 0, 4, 0)
        pl.setSpacing(2)
        prev_i = Wd.tool("chevron-left", "이전 확인 필요 구간 (추적 신뢰도 낮음)", size=14)
        next_i = Wd.tool("chevron-right", "다음 확인 필요 구간 (추적 신뢰도 낮음)", size=14)
        lab_i = Wd.label("확인 필요", "Muted", 11, 500)
        prev_i.clicked.connect(lambda: self.next_issue(-1))
        next_i.clicked.connect(lambda: self.next_issue(1))
        pl.addWidget(prev_i)
        pl.addWidget(lab_i)
        pl.addWidget(next_i)
        pill.setFixedHeight(26)
        h.addWidget(pill)
        self.edit_busy = QLabel("")
        self.edit_busy.setObjectName("Faint")
        self.edit_busy.setMinimumWidth(90)
        h.addWidget(self.edit_busy)
        return w

    def _shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.WindowShortcut)
            s.activated.connect(fn)

        sc("Space", self.toggle_play)
        sc("Left", lambda: self.step(-1))
        sc("Right", lambda: self.step(1))
        sc("Shift+Left", lambda: self.step(-10))
        sc("Shift+Right", lambda: self.step(10))
        sc("Home", lambda: self.set_frame(self.session.cut.work_range[0] if self.session.cut else 0))
        sc("End", lambda: self.set_frame((self.session.cut.work_range[1] - 1) if self.session.cut else 0))
        sc("I", lambda: self.set_in_out("in"))
        sc("O", lambda: self.set_in_out("out"))
        sc("\\", self._ab)
        sc("S", lambda: self.ov["skel"].toggle())
        sc("M", lambda: self.ov["matte"].toggle())
        sc("K", lambda: self.ov["skin"].toggle())
        sc("V", lambda: self.set_tool("edit"))
        sc("Z", lambda: self.viewer.fit_view())
        sc("Ctrl+F", self.toggle_fullscreen)
        sc("Shift+F", self.toggle_focus)
        sc("Ctrl+C", self.copy_cut_settings)
        sc("Ctrl+V", self.paste_cut_settings)
        sc("Alt+V", self.paste_cut_settings_pick)
        sc("Ctrl+S", self.save)
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+Shift+Z", self.redo)
        sc("Ctrl+Y", self.redo)
        sc("Escape", lambda: self.toggle_fullscreen() if self.isFullScreen() else self.set_tool("edit"))
        sc("Up", lambda: self.step_cut(-1))
        sc("Down", lambda: self.step_cut(1))

    # ================================================================== startup / project
    def _startup(self):
        s = settings()
        M.configure_ffmpeg()                 # the ffmpeg that ships with DIAP
        manager()                            # the built-in ACES Studio config
        try:
            from ..track.models import device_label
            self.dev_lbl.setText(f"추론: {device_label()}")
        except Exception:
            self.dev_lbl.setText("추론: CPU")
        if not M._tool_paths.get("ffmpeg"):
            self.show_error("ffmpeg가 필요합니다",
                            r"ffmpeg를 찾지 못했습니다. 설정에서 ffmpeg 폴더(예: C:\ffmpeg\bin)를 지정하세요.")
        if s.auto_update_check and not os.environ.get("CONTOUR_NO_UPDATE"):
            from . import update as UP
            QTimer.singleShot(3000, lambda: UP.check(self))
        last = s.last_project
        self._refresh_all()
        self.viewer.message = "왼쪽 위 프로젝트 메뉴에서 새 프로젝트를 만들거나 여세요. 영상을 끌어다 놓아도 됩니다"
        self.viewer.update()
        if s.open_last_on_start and last and (Path(last) / "project.json").exists():
            self.open_project(last)
        else:
            self.show_page("projects")             # the project manager comes first

    def _project_menu(self):
        m = QMenu(self)
        a_mgr = m.addAction(I.icon("layers", 14), "프로젝트 관리…")
        a_new = m.addAction(I.icon("plus", 14), "새 프로젝트…")
        a_open = m.addAction(I.icon("folder-open", 14), "프로젝트 열기…")
        rec = m.addMenu("최근 프로젝트")
        for p in settings().recent_projects:
            if (Path(p) / "project.json").exists():
                a = rec.addAction(Path(p).name)
                a.setData(p)
        m.addSeparator()
        a_save = m.addAction(I.icon("save", 14), "저장  Ctrl+S")
        a_save.setEnabled(self.session.project is not None)
        a_learn = m.addAction("이 프로젝트에서 학습 데이터 수집·자동 학습")
        a_learn.setToolTip("켠 프로젝트에서만 분석한 컷의 학습용 이미지·샘플을 이 컴퓨터에 모으고 자동 학습을 시작합니다 "
                           "(기본 꺼짐). 꺼도 편집·미리보기·렌더와 이미 학습된 모델은 그대로 씁니다. 어디로도 보내지 않습니다")
        a_learn.setCheckable(True)
        a_learn.setChecked(bool(self.session.project and self.session.project.learn))
        a_learn.setEnabled(self.session.project is not None)
        a_tl = m.addAction("타임라인 불러오기…")
        a_tl.setEnabled(self.session.project is not None)
        a_tlx = m.addAction("타임라인 내보내기…")
        a_tlx.setEnabled(self.session.project is not None and bool(self.session.project.timeline_cuts()))
        a_review = m.addAction("AI 검수 리포트 만들기")
        a_review.setToolTip("저장된 추적·매트·QC로 컷별 분석·캐시 상태, 실패 원인, 시간, 설정, 요약 지표를 report.md와 "
                            "review.json으로 만듭니다. 영상·크롭·미디어 절대경로는 넣지 않고, 모델을 다시 돌리지 않습니다")
        a_review.setEnabled(self.session.project is not None)
        a_review_img = m.addAction("AI 검수 리포트 만들기 (화면 그림 포함)")
        a_review_img.setToolTip("위 리포트에, 문제가 큰 프레임의 화면 그림(추적 뼈대·인물 영역 표시)을 더합니다. "
                                "영상 프레임이 리포트 폴더에 들어가므로 밖으로 내보낼 때 주의하세요")
        a_review_img.setEnabled(self.session.project is not None)
        a_relink = m.addAction("원본 다시 연결…")
        a_relink.setToolTip("원본 영상·LUT의 위치가 바뀌었을 때(다른 컴퓨터, 다른 드라이브, Windows ↔ Mac) 폴더를 골라 같은 이름의 "
                            "파일을 찾아 연결합니다. 크기·프레임 속도·프레임 수가 같은 파일만 연결하고, 분석 결과와 보정값은 그대로 둡니다")
        a_relink.setEnabled(self.session.project is not None)
        a_reveal = m.addAction("프로젝트 폴더 열기")
        a_reveal.setEnabled(self.session.project is not None)
        act = m.exec(self.proj_btn.mapToGlobal(self.proj_btn.rect().bottomLeft()))
        if act is None:
            return
        if act == a_mgr:
            self.show_page("projects")
        elif act == a_new:
            self.new_project()
        elif act == a_open:
            d = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 (.dpprj) 선택", str(Path.home()))
            if d:
                self.open_project(d)
                self.show_page("edit")
        elif act == a_save:
            self.save()
        elif act == a_learn:
            self.session.project.set_learn(act.isChecked())
            self.session.dirty = True
            self.toast("이 프로젝트에서 학습 데이터 수집·자동 학습을 켰습니다 (이 컴퓨터에만 저장)" if act.isChecked() else
                       "이 프로젝트에서 학습 데이터 수집·자동 학습을 껐습니다", "ok")
        elif act == a_tl:
            self.import_timeline()
        elif act == a_tlx:
            self._tl_action("export", self.tlstrip.selected_ids())
        elif act == a_review:
            self.make_ai_review()
        elif act == a_review_img:
            self.make_ai_review(images=True)
        elif act == a_relink:
            self.relink_media()
        elif act == a_reveal:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.session.project.path))
        elif act.data():
            self.open_project(act.data())
            self.show_page("edit")

    def new_project(self):
        name = TextPrompt.ask(self, "새 프로젝트", "프로젝트 이름 (공백은 _로 바뀝니다)", "", "다음")
        if not name:
            return
        d = QFileDialog.getExistingDirectory(self, "프로젝트를 만들 폴더", str(Path.home()))
        if not d:
            return
        if not self.close_project():
            return
        prj = Project.create(d, name)
        self.open_project(prj.path, wait=True)
        self.show_page("edit")

    def open_project(self, path: str, wait: bool = False):
        """Reads the project and the cut thumbnails on a thread (the window keeps turning) and then opens it.
        wait: in one go on this thread (a project just made, which callers go on to fill)."""
        if getattr(self, "_opening", None) is not None:
            return
        if self.job is not None:
            self.toast("진행 중인 작업이 끝난 뒤 프로젝트를 열 수 있습니다.", kind="warn")
            return
        self._offer_recovery(path)
        if wait:
            try:
                t0 = time.perf_counter()
                prj = _load_project(path, settings().proxy_long_side, lambda f, m: None)
                prj._open_times = {"load": time.perf_counter() - t0}
            except Exception as e:
                self.show_error("프로젝트를 열 수 없습니다", f"{e}\n{traceback.format_exc()}")
                return
            self._project_loaded(prj)
            return
        self._set_loading(True)
        th = _ProjectOpen(self, path, settings().proxy_long_side)
        th.progressed.connect(self.busy.progress)
        th.loaded.connect(self._project_opened)
        th.failed.connect(self._project_failed)
        self._opening = th
        th.start()

    def _set_loading(self, on: bool, hide: bool = True):
        for w_ in (self.cutlist, self.inspector, self.tlstrip, self.timeline, self.proj_btn):
            w_.setEnabled(not on)
        if on:
            self.busy.start("프로젝트 여는 중", cancellable=False)
            self.busy.progress(0.0, "프로젝트 읽는 중")
        elif hide:
            self.busy.hide()

    def _project_failed(self, err: str):
        self._opening = None
        self._set_loading(False)
        self.show_error("프로젝트를 열 수 없습니다", err)

    def _project_opened(self, prj, _unused=None):
        self.busy.progress(1.0, "첫 컷 여는 중")
        QApplication.processEvents()
        try:
            self._project_loaded(prj)
        finally:
            self._opening = None
            self._set_loading(False, hide=self.job is None)     # a job the cut started keeps the overlay

    def _project_loaded(self, prj, thumbs: dict = None):
        t0 = time.perf_counter()
        if not self.close_project():
            self.toast("지금 프로젝트를 저장하지 못해 새 프로젝트를 열지 않았습니다", kind="error")
            return
        self.session.project = prj
        self._undo_by_cut.clear()
        self.undo_stack, self.redo_stack = [], []
        s = settings()
        s.add_recent(prj.path)
        s.save()
        self.proj_btn.setText(prj.name)
        self.viewer.message = ""
        self.thumbcache.reset(prj)                  # a new opening: what the last one's thumbnails bring is dropped
        self.cutlist.refresh()
        self.tlstrip.refresh()
        times = dict(getattr(prj, "_open_times", None) or {})
        times["list"] = time.perf_counter() - t0
        first = self._first_listed(prj)
        if first is not None:
            self.open_cut(first.id)
            self.cutlist.refresh(first.id)
        else:
            self._refresh_all()
            self.viewer.update()
        times["first_cut"] = time.perf_counter() - t0
        times.update({f"cut_{k}": v for k, v in (getattr(self.session, "open_times", None) or {}).items()})
        self.open_times = times                     # seconds: project.json, list usable, first cut opened
        self._tell_legacy(prj)
        self._tell_learning(prj)
        self._tell_missing(prj)

    def relink_media(self, folder: str = ""):
        """원본 다시 연결: the media this computer does not find, looked for by name under a folder and connected
        when it is the same footage (core/relink.py). The analysis and the edits stay with the cuts."""
        from ..core import relink as RL
        prj = self.session.project
        if prj is None:
            return
        if self.job is not None or self.batch.busy:
            self.toast("진행 중인 작업이 끝난 뒤 원본을 다시 연결할 수 있습니다.", kind="warn")
            return
        if not RL.missing(prj):
            self.toast("연결할 것이 없습니다 (모든 원본이 제자리에 있습니다)", "ok")
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "원본이 있는 폴더", str(Path(prj.path).parent))
        if not folder:
            return
        if not self.save():
            self.toast("저장하지 못해 원본을 연결하지 않았습니다", kind="error")
            return

        def work(progress, cancelled):
            return RL.relink(prj, folder, progress=progress, cancel=cancelled)

        def done(res):
            if res["linked"]:
                self.session.dirty = True
                self.save()
                self.cutlist.refresh(self.session.cut.id if self.session.cut else None)
                self.tlstrip.refresh()
                if self.session.cut is not None:
                    self._reopen_cut()
            self.toast(RL.summary(res), "ok" if res["linked"] and not (res["different"] or res["not_found"]) else "warn")
        self.run_job("원본 찾는 중", work, done)

    def _tell_missing(self, prj):
        """On opening: how many cuts have no original here, and where to connect them (once per opening)."""
        try:
            from ..core import relink as RL
            miss = [m for m in RL.missing(prj) if m["kind"] == "source"]
            if miss:
                n = sum(len(m["cuts"]) for m in miss)
                self.toast(f"원본을 찾을 수 없는 컷 {n}개가 있습니다. 프로젝트 메뉴의 '원본 다시 연결'에서 파일이 있는 폴더를 "
                           "고르면 다시 이어집니다 (분석 결과와 보정값은 그대로입니다).", kind="warn")
        except Exception:
            traceback.print_exc()

    def _tell_learning(self, prj):
        """Once per project made before 1.24 whose file said (or left out, which meant) that it is in the learning
        data: learning from a project now starts only where it is switched on by hand, and this one never was."""
        try:
            if not getattr(prj, "learn_before", False) or prj.learn_set:
                return
            key = str(prj.created or prj.path)
            s_ = settings()
            if key in (s_.learn_notified or []):
                return
            self.toast("1.24부터 학습 데이터 수집과 자동 학습은 직접 켠 프로젝트에서만 합니다. 이 프로젝트는 꺼져 있습니다 "
                       "(프로젝트 메뉴에서 켤 수 있습니다). 편집·미리보기·렌더와 이미 학습된 모델은 그대로 씁니다.")
            s_.learn_notified = (list(s_.learn_notified or []) + [key])[-500:]
            s_.save()
        except Exception:
            traceback.print_exc()

    def _tell_legacy(self, prj):
        """Once per project: what an older project had that 1.22.2 no longer applies (the separate mode, drawn
        protection lines and shapes, other protection settings). The project is not changed by opening it."""
        from ..warp.field import legacy_protection
        try:
            key = str(prj.created or prj.path)
            s_ = settings()
            if key in (s_.legacy_notified or []):
                return
            found = {}
            for c in prj.cuts:
                for what in legacy_protection(c):
                    found.setdefault(what, []).append(c.name)
            if not found:
                return
            parts = ", ".join(f"{k} ({len(v)}컷)" for k, v in found.items())
            self.toast(f"이 프로젝트의 {parts}은(는) 더 이상 적용되지 않고 일반 워프로 보입니다. "
                       "보정하지 않는 인물은 계속 자동으로 보호되고, 프로젝트 파일은 바뀌지 않았습니다.", kind="warn")
            s_.legacy_notified = list(s_.legacy_notified or []) + [key]
            s_.save()
        except Exception:
            traceback.print_exc()

    # ---- recovery: the autosave writes a recovery copy; project.json changes only when saved
    @staticmethod
    def _recovery_paths(folder: Path):
        folder = Path(folder)
        return folder / "autosave" / "recovery.json", list((folder / "cuts").glob("*/tracks.recovery.npz"))

    def _clear_recovery(self, folder) -> None:
        rec, trs = self._recovery_paths(folder)
        for f in [rec] + trs:
            try:
                f.unlink()
            except OSError:
                pass

    def _offer_recovery(self, path: str) -> None:
        """Work that was autosaved but never saved (the app ended without saving): offered back once."""
        p = Path(path)
        if p.is_file():
            p = p.parent
        rec, _ = self._recovery_paths(p)
        if not rec.exists():
            return
        try:
            d = json.loads(rec.read_text(encoding="utf-8"))
            t = float(d.get("meta", {}).get("time", 0))
            data = d["project"]
        except Exception:
            self._clear_recovery(p)
            return
        pj = p / "project.json"
        if pj.exists() and pj.stat().st_mtime >= t:
            self._clear_recovery(p)                 # saved after the recovery copy was made
            return
        m = QMessageBox(self)
        m.setWindowTitle("DIAP")
        m.setIcon(QMessageBox.Question)
        m.setText("저장하지 않고 끝난 작업이 있습니다.")
        m.setInformativeText(f"{time.strftime('%m월 %d일 %H:%M', time.localtime(t))}에 자동 저장된 복구본으로 "
                             "되돌릴까요? 버리면 마지막으로 저장한 상태로 엽니다.")
        b_rec = m.addButton("복구", QMessageBox.AcceptRole)
        m.addButton("버리기", QMessageBox.DestructiveRole)
        m.setDefaultButton(b_rec)
        m.exec()
        if m.clickedButton() is b_rec:
            try:
                if pj.exists():
                    (p / "project.before_recovery.json").write_bytes(pj.read_bytes())
                tmp = pj.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(pj)
                cid = d.get("meta", {}).get("cut")
                if cid:
                    tr = p / "cuts" / str(cid) / "tracks.recovery.npz"
                    if tr.exists():
                        dst = tr.with_name("tracks.npz")
                        if dst.exists():
                            dst.replace(tr.with_name("tracks.before_recovery.npz"))
                        tr.replace(dst)
            except OSError as e:
                self.show_error("복구하지 못했습니다", str(e))
        self._clear_recovery(p)

    def close_project(self) -> bool:
        if self.session.project is not None and not self._save_or_ask("프로젝트를 닫기"):
            return False
        self.activity.set_cut("", None)
        self.refresher.cancel()
        if self.batch.busy:
            self.batch.kill()
        self.proxy_builder.cancel()
        self.next_prep.cancel()
        self.thumbcache.reset(None)
        self.session.project = None
        self._undo_by_cut.clear()
        self.session.open_cut(None)
        self.tlstrip.refresh()
        self.proj_btn.setText("프로젝트 없음")
        return True

    def save(self, quiet: bool = False) -> bool:
        """False when the project could not be written: callers that would drop what is in memory (closing the
        project, going to another cut, quitting) must stop then."""
        if self.session.project is None:
            return True
        if not quiet:
            self.saved_lbl.start("저장 중")
        self.activity.flush()
        # joint re-solves still queued are part of what is saved (and must not write the tracks while they are saved):
        # one that has not finished means no save now (callers that would drop unsaved work stop on False)
        if not self.edit_worker.drain(30.0):
            self.saved_lbl.done("저장 보류")
            self.show_error("저장하지 않았습니다", "관절 다시 계산이 30초 안에 끝나지 않아 저장하지 않았습니다. "
                            "계산이 끝난 뒤(아래 '다시 계산 중' 표시가 사라지면) 다시 저장하세요.")
            return False
        try:
            self.session.save()
        except Exception as e:
            self.saved_lbl.done("저장 실패")
            self.show_error("저장 실패", f"{e}\n{traceback.format_exc()}")
            return False
        try:
            self._clear_recovery(Path(self.session.project.path))
            self._collect_learning()
        except Exception:
            traceback.print_exc()
        self.saved_lbl.done(f"저장됨 {time.strftime('%H:%M')}")
        return True

    def _save_or_ask(self, what: str) -> bool:
        """Saves; if that fails, asks whether to go on anyway (losing what was not saved). Default: no."""
        if self.save():
            return True
        return QMessageBox.question(
            self, "저장 실패",
            f"프로젝트를 저장하지 못했습니다. 저장하지 않은 변경 사항을 버리고 {what}할까요?\n"
            "(아니요: 그대로 두고 공간·권한·네트워크 연결을 확인한 뒤 다시 저장하세요)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def apply_autosave_settings(self):
        s = settings()
        if getattr(s, "autosave", True):
            self.autosave.start(max(1, int(getattr(s, "autosave_minutes", 5))) * 60_000)
        else:
            self.autosave.stop()

    def _autosave(self):
        """A recovery copy of unsaved work in <project>/autosave (recovery.json, the open cut's tracks next to its
        own) and the last N of them as project_<time>.json. project.json itself changes only when the project is
        saved, so closing without saving really leaves it as it was saved."""
        if not (self.session.dirty and self.job is None and self.session.project is not None):
            return
        prj = self.session.project
        cut = self.session.cut
        try:
            d = Path(prj.path) / "autosave"
            d.mkdir(exist_ok=True)
            data = {"meta": {"time": time.time(), "cut": cut.id if cut is not None else None, "version": APP_VERSION},
                    "project": prj.to_dict()}
            tmp = d / "recovery.json.tmp"
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(d / "recovery.json")
            (d / f"project_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
                json.dumps(data["project"], ensure_ascii=False, indent=1), encoding="utf-8")
            keep = max(1, int(getattr(settings(), "autosave_keep", 10)))
            old = sorted(d.glob("project_*.json"))
            for x in old[:-keep]:
                x.unlink()
            tr = self.session.tracker
            if tr is not None and tr.store is not None and cut is not None:
                tr.store.save(prj.cut_dir(cut) / "tracks.recovery.npz")
            self.saved_lbl.done(f"복구본 자동 저장 {time.strftime('%H:%M')}")
        except Exception:
            pass

    def _emergency_save(self, *_):
        try:
            if self.session.project is not None and self.session.dirty:
                self.session.save()
        except Exception:
            pass

    def _learn_allowed(self) -> bool:
        """Learning was switched on by hand for the open project (see learn.allowed)."""
        from ..learn import allowed
        return allowed(self.session.project)

    def _collect_learning(self, force: bool = False):
        s = self.session
        if s.cut is None or s.tracker is None or s.tracker.store is None:
            return
        sig = json.dumps([s.cut.joint_keys, s.cut.confirmed, s.project.learn, s.cut.learn,
                          settings().collect_learning_data], sort_keys=True, default=str)
        if not force and getattr(self, "_learn_sig", {}).get(s.cut.id) == sig:
            return
        try:
            from ..learn.refiner import collect_cut
            collect_cut(s.project, s.cut, s.tracker)
            self._learn_sig = getattr(self, "_learn_sig", {})
            self._learn_sig[s.cut.id] = sig
        except Exception:
            pass

    def collect_learning_all(self):
        self._collect_learning(force=True)

    def flush_activity(self):
        self.activity.flush()

    # ================================================================== cuts
    def import_edit_xml(self, path: str, quiet: bool = False) -> bool:
        """Edit points of the editor's sequence: tracking restarts exactly at those cuts and nowhere else."""
        from ..core import editxml as EX
        if self.session.project is None:
            return False
        try:
            done = EX.apply_to_project(self.session.project, path)
        except Exception as e:
            if quiet:
                return False
            self.show_error("XML을 읽을 수 없습니다", str(e))
            return False
        if not done:
            if not quiet:
                self.toast("이 XML이 쓰는 영상 파일과 이름이 같은 컷이 없습니다", "warn")
            return False
        self.session.dirty = True
        redo = [n for n, _ in done if (c := next((c for c in self.session.project.cuts if c.name == n), None)) and c.analyzed]
        msg = ", ".join(f"{n} ({len(p)}곳)" for n, p in done)
        self.toast(f"컷 경계 적용: {msg}" + (" · 이미 분석한 컷은 다시 분석해야 반영됩니다" if redo else ""))
        return True

    def add_cuts(self, paths: Optional[List[str]] = None):
        if self.session.project is None:
            self.new_project()
            if self.session.project is None:
                return
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "컷 추가", str(Path.home()), VIDEO_FILTER)
        if not paths:
            return
        from ..learn import style as SL
        paths, pairs, orphans = SL.pair_paths(list(paths))
        # a hand result dropped on its own: attach it to the plate already in the project
        attached = []
        for a in list(orphans):
            k = SL._norm(SL.split_after(Path(a).name) or "")
            c = next((c for c in self.session.project.cuts if SL._norm(Path(c.source).stem) == k), None)
            if c is not None:
                c.retouch_ref = a
                attached.append(c)
                orphans.remove(a)
        paths = paths + orphans
        added = []
        seen_seq = set()
        for p in paths:
            try:
                info = M.probe(p)
            except Exception as e:
                self.show_error("불러올 수 없는 파일", f"{Path(p).name}: {e}")
                continue
            if info.kind == "sequence":
                if info.seq_pattern in seen_seq:
                    continue
                seen_seq.add(info.seq_pattern)
            name = Path(info.seq_pattern).stem.split("%")[0].rstrip("._-") if info.kind == "sequence" else Path(p).stem
            cut = self.session.project.add_cut(info, name)
            # colour space from the file name and the camera tags (changed by hand in the 출력 tab)
            cut.input_cs, cut.cs_auto = manager().detect_input(info, name)
            if p in pairs:
                cut.retouch_ref = pairs[p]
            added.append(cut)
        paired = [c for c in added if c.retouch_ref] + attached
        if paired:
            names = ", ".join(c.name for c in paired)
            self.toast(f"수작업 결과 짝: {names} · 분석이 끝나면 리터칭 스타일을 자동으로 학습합니다", "ok")
        if not added:
            if attached:
                self.session.project.save()
                for c in attached:
                    if c.analyzed:
                        self.learn_style(c)
                        break
            return
        self.session.project.save()
        self.cutlist.refresh(added[0].id)
        self.open_cut(added[0].id)

    def open_cut(self, cid: str):
        prj = self.session.project
        if prj is None:
            return
        cut = prj.cut(cid)
        if cut is None or (self.session.cut is not None and self.session.cut.id == cid and self.session.proxy):
            return
        if self.job is not None:
            self.toast("진행 중인 작업이 끝난 뒤 컷을 바꿀 수 있습니다.", kind="warn")
            self.cutlist.refresh(self.session.cut.id if self.session.cut else None)
            return
        if self.batch.current == cid:
            self.toast("일괄 분석 중인 컷입니다. 분석이 끝나면 열 수 있습니다.", kind="warn")
            return
        self.stop_play()
        if self.session.cut is not None and not self.save():
            # the open cut's tracks and edits are written with the project: never dropped unsaved
            self.toast("저장하지 못해 컷을 바꾸지 않았습니다", kind="error")
            self.cutlist.refresh(self.session.cut.id)
            return
        self._switch_undo(cut.id)
        self.refresher.cancel()
        # 1.22.1: no proxy-size question here any more (it came back on every unanalysed cut when closed); the size
        # is 설정 > 프록시 해상도, and the proxy is made in the background by 설정 > 프록시 만들기
        self.session.open_cut(cut)
        self.viewer._img_cache.clear()            # cached pictures are per frame number, not per cut
        if hasattr(self.viewer, "_last_res"):
            self.viewer._last_res = None
        self.viewer._skin_last = None
        self.tlstrip.set_current(cut.id)
        if len(self.cutlist.selected_ids()) <= 1:
            self.cutlist.show_current(cut.id)       # the open cut in the middle of the list, as in the timeline
        self.activity.set_cut(prj.name, cut)
        self.selection = None
        self.timeline.selection = None
        self.viewer.selection = None
        self.timeline.view0, self.timeline.view1 = 0.0, 1.0
        self.frame = cut.work_range[0]
        targets = cut.targets()
        self.sel_pid = targets[0].id if targets else (cut.persons[0].id if cut.persons else None)
        self.viewer.fit_view()
        self.build_proxy()

    def _want_thumbs(self, on_screen: list, every: list):
        """A view was filled or scrolled: the rows on screen first (made when missing), then every saved JPG."""
        self.thumbcache.want(on_screen, make=True, front=True)
        self.thumbcache.want(every)

    def _after_open(self):
        cut = self.session.cut
        if self.session.proxy is not None:
            # its thumbnail through the shared queue (first in line): remade only when the saved one was made
            # from another original, colour view or frame
            self.thumbcache.refresh(cut.id)
        if not cut.analyzed and cut.tl and (cut.status or {}).get("auto") == "none":
            self.viewer.message = f"리터칭 불필요 컷 · {(cut.status or {}).get('reason', '')}"
        elif not cut.analyzed and self.batch.busy and cut.id in self.batch.queue:
            self.viewer.message = "일괄 분석 대기 중인 컷입니다"
        elif not cut.analyzed:
            self.viewer.message = "추적 탭에서 '분석 시작'을 누르세요"
            self.inspector.show_tab("track")
        else:
            self.viewer.message = ""
            if self.inspector.tabs.current == "track":
                self.inspector.show_tab("reshape")
            st = self.session.tracker.store if self.session.tracker else None
            tr = self.session.tracker
            if tr is not None and getattr(tr, "widths_stale", False):
                # body segments were added in this version: measure their widths in the background
                tr.widths_stale = False
                self.edit_worker.submit(("widths", -1), lambda: (tr.update_widths(), tr.save()))
            blocked = self._version_note(cut)      # analysed with package versions that are not on this computer
            if st is not None and not (st.has_vis and st.has_face):
                self.toast("이전 버전에서 분석한 컷입니다. 가려진 관절 판정과 얼굴·피부 보정을 쓰려면 다시 분석하세요.",
                           kind="warn")
            elif tr is not None and st is not None and st.has_face and not blocked:
                self._ensure_facelayer(tr)
            self.refresher.kick()                 # marks left from the last time it was open
        self._refresh_all()

    def _ensure_facelayer(self, tr):
        """Cuts analysed before 1.9 have no face layer (parsing, steady skin colour): build it once, in the
        background; the masks use it from then on."""
        from ..track import facelayer as FL, faceparse as FP
        if not FP.available() or FL.path_for(tr).exists() or getattr(tr, "_facelayer_building", False):
            return
        tr._facelayer_building = True

        def work():
            try:
                FL.build(tr)
            finally:
                tr._facelayer_building = False
            tr.__dict__.pop("_steady_lru", None)
            self.session._mask_cache = {}
        self.edit_worker.submit(("facelayer", -1), work)

    def build_proxy(self):
        """After a cut was (re)opened (session.open_cut): go on with its frames. Since 1.18 nothing waits for a proxy:
        the frames come from the proxy when there is one in this view, else straight from the original; a proxy is
        only made in the background (설정 > 프록시 만들기)."""
        s = self.session
        cut = s.cut
        self.viewer._img_cache.clear()
        if cut is None:
            return
        from ..core.frames import source_present
        if s.proxy is None:
            self.viewer.message = ("원본 파일을 찾을 수 없습니다: " + str(cut.source or cut.info.path)
                                   + "\n파일을 원래 자리에 두거나, 프로젝트 메뉴의 '원본 다시 연결'에서 파일이 있는 폴더를 "
                                     "고르세요.")
            self.viewer.update()
            self._refresh_all()
            return
        self.viewer._last_orig = None
        # the first picture is decoded by the viewer's loader thread (a file that cannot be read says so there)
        if not s.proxy_ready() and not source_present(cut.info):
            self.toast("원본이 없어 예전 프록시로 보여 줍니다. 바뀐 색공간은 원본이 있어야 반영되고, 분석·렌더도 원본이 필요합니다.",
                       "warn")
        self._after_open()
        self._maybe_bg_proxy(first=True)
        if getattr(self, "_reanalyse_after_proxy", None) == cut.id:
            self._reanalyse_after_proxy = None
            QTimer.singleShot(0, self.run_analysis)

    def _reopen_cut(self) -> bool:
        """The open cut again from its files (another colour view, LUT or setting). What was changed in memory and
        not saved yet is saved first: opening reads the tracks from the file again (1.22.2 dropped joint edits
        made since the last save here)."""
        cut = self.session.cut
        if cut is None:
            return False
        if self.session.dirty and not self.save():
            self.toast("저장하지 못해 화면을 다시 만들지 않았습니다. 저장한 뒤 컷을 다시 열면 반영됩니다", kind="error")
            return False
        self.session.open_cut(cut)
        self.build_proxy()
        return True

    def _maybe_bg_proxy(self, first: bool = False):
        """설정 > 프록시 만들기: 'always' makes the open cut's proxy in the background; 'auto' only once reading the
        original turned out slower than it plays."""
        from ..core.frames import proxy_mode, wants_proxy
        s = self.session
        if s.cut is None or s.project is None or not s.from_source() or self.job is not None:
            return
        if proxy_mode() == "always" or wants_proxy(s.proxy, s.cut.info.fps):
            self.proxy_builder.start(s.project, s.cut, s.view_settings(s.cut))

    def _proxy_note(self) -> None:
        """Beside the picture, never in the way (1.22.3): what the open cut's frames are read from, a proxy being
        made in the background (with how far it is) or one that could not be made (the reason in the tooltip)."""
        lbl = getattr(self, "px_lbl", None)
        if lbl is None:
            return
        s = self.session
        if s.cut is None or s.proxy is None or s.project is None:
            lbl.setText("")
            lbl.setToolTip("")
            return
        ck = (str(s.project.path), s.cut.id)
        making = getattr(self, "_px_making", None)
        failed = getattr(self, "_px_failed", {})
        if not s.from_source():
            text, tip = "프록시", "이 컷의 프록시 파일에서 읽습니다."
        elif ck in failed:
            text = "프록시 실패 · 원본으로 계속"
            tip = "프록시를 만들지 못했습니다. 원본을 바로 읽어 계속 작업할 수 있습니다.\n" + failed[ck][:600]
        elif making is not None and making[0] == ck and self.proxy_builder.busy:
            text = f"프록시 만드는 중 {making[1]}%"
            tip = "뒤에서 프록시를 만드는 중입니다. 기다리지 않고 원본으로 바로 탐색·재생할 수 있습니다."
        else:
            text = "원본에서 바로 읽는 중"
            tip = "프록시 없이 원본을 그때그때 읽습니다. 탐색·재생·분석 모두 바로 할 수 있습니다."
        lbl.setText(text + "   ")
        lbl.setToolTip(tip)

    def _bg_proxy_progress(self, prj_path: str, cut_id: str, f: float):
        self._px_making = ((prj_path, cut_id), int(round(100 * max(0.0, min(1.0, float(f))))))
        s = self.session
        if s.cut is not None and s.cut.id == cut_id and s.project is not None and str(s.project.path) == prj_path:
            self._proxy_note()

    def _bg_proxy_done(self, prj_path: str, cut_id: str, ok: bool, msg: str):
        s = self.session
        # a proxy of an earlier cut or of a project since closed never replaces what is on screen now
        here = not (s.cut is None or s.cut.id != cut_id or s.project is None or str(s.project.path) != prj_path)
        full = hasattr(self, "px_lbl")
        if full:
            self._px_making = None
        if not ok:
            if msg:
                print("background proxy:", msg, flush=True)
            if full:
                # said once, beside the picture, and kept there for this cut; it goes on from its original
                # (1.22.2 only printed it)
                self._px_failed[(prj_path, cut_id)] = msg or "원인을 알 수 없습니다"
                if here:
                    self._proxy_note()
                    self.toast("프록시를 만들지 못했습니다. 원본으로 계속 볼 수 있습니다"
                               + (": " + msg.splitlines()[0][:120] if msg else ""), "warn")
            return
        if full:
            self._px_failed.pop((prj_path, cut_id), None)
        if not here:
            return
        from ..core.frames import frames_for
        new = frames_for(s.project, s.cut, s.view_settings(s.cut))
        if getattr(new, "kind", "") == "proxy" and self.job is None and s.swap_frames(cut_id, new):
            self.toast("프록시가 준비되어 이 컷을 더 빠르게 읽습니다", "ok")
        if full:
            self._proxy_note()

    # ================================================================== cut menu (cut list and timeline)
    def _sel_cut_ids(self) -> list:
        ids = self.cutlist.selected_ids() if self.cutlist.isVisible() else []
        if not ids and self.tlstrip.isVisible():
            ids = self.tlstrip.selected_ids()
        if not ids and self.session.cut is not None:
            ids = [self.session.cut.id]
        return ids

    def _state_menu(self, ids: list, gpos):
        """A click on a cut's status label: set the work state of that cut (or of every selected cut)."""
        from ..core import workstate as WS
        prj = self.session.project
        cuts = [c for c in (prj.cut(i) for i in ids) if c is not None] if prj else []
        if not cuts:
            return
        n = len(cuts)
        cur = {WS.state(c) for c in cuts}
        m = QMenu(self)
        acts = {}
        for key, name in (("todo", "작업 전"), ("doing", "작업 중"), ("skip", "작업 불필요"), ("done", "완료")):
            a = m.addAction(name + (f" ({n}컷)" if n > 1 else ""))
            a.setCheckable(True)
            a.setChecked(cur == {key})
            acts[a] = key
        m.addSeparator()
        acts[m.addAction("자동 (분석·보정 상태에 따름)")] = ""
        a = m.exec(gpos)
        if a not in acts:
            return
        key = acts[a]
        self.push_undo()
        for c in cuts:
            WS.set_state(c, key)
        self.session.dirty = True
        self._after_status_change()
        self.cutlist.update_rows()
        self.toast(f"{n}컷: {WS.LABEL[key]}" if key else f"{n}컷: 작업 상태를 자동으로 되돌렸습니다", "ok")

    def _cut_menu(self, ids: list, gpos):
        from ..core import workstate as WS
        prj = self.session.project
        if prj is None or not ids:
            return
        cuts = [c for c in (prj.cut(i) for i in ids) if c is not None]
        if not cuts:
            return
        n = len(cuts)
        tag = f" ({n}컷)" if n > 1 else ""
        m = QMenu(self)
        acts = {}
        states = {WS.state(c) for c in cuts}
        if states != {"done"}:
            acts[m.addAction(I.icon("check", 14, C["ok"]), "작업 완료로 표시" + tag)] = "work:done"
        if "done" in states:
            acts[m.addAction("작업 중으로 되돌리기" + tag)] = "work:"
        if states != {"skip"}:
            acts[m.addAction("작업 불필요로 지정" + tag)] = "work:skip"
        if "skip" in states:
            acts[m.addAction("작업 필요로 되돌리기" + tag)] = "work:"
        m.addSeparator()
        c0 = cuts[0]
        if n == 1:
            acts[m.addAction(f"버전 올리기 (v{c0.version} → v{c0.version + 1})")] = "ver:up"
            older = sorted(int(v) for v in c0.versions if int(v) < c0.version)
            a = m.addAction(f"이전 버전으로 (v{older[-1]})" if older else "이전 버전 없음")
            a.setEnabled(bool(older))
            acts[a] = "ver:back"
        else:
            acts[m.addAction(f"버전 올리기{tag}")] = "ver:up"
        m.addSeparator()
        acts[m.addAction("보정값 복사  Ctrl+C")] = "copy"
        a = m.addAction("보정값 붙여넣기  Ctrl+V" + tag)
        a.setEnabled(getattr(self, "_copied", None) is not None)
        acts[a] = "paste"
        a = m.addAction("골라서 붙여넣기…  Alt+V" + tag)
        a.setEnabled(getattr(self, "_copied", None) is not None)
        acts[a] = "paste_pick"
        m.addSeparator()
        acts[m.addAction("선택 컷 분석" + tag)] = "analyze"
        acts[m.addAction("렌더에 추가" + tag)] = "render"
        if any(c.tl for c in cuts):
            acts[m.addAction("검수 완료 표시" + tag)] = "reviewed:1"
        if n == 1:
            m.addSeparator()
            acts[m.addAction("이름 바꾸기")] = "rename"
            acts[m.addAction("원본 파일 위치 열기")] = "reveal"
            if not c0.tl:
                acts[m.addAction("프로젝트에서 제거")] = "remove"
        a = m.exec(gpos)
        if a not in acts:
            return
        act = acts[a]
        if act.startswith("work:"):
            self.push_undo()
            for c in cuts:
                WS.set_state(c, act[5:])
            self.session.dirty = True
            self._after_status_change()
            self.toast(f"{n}컷: {WS.LABEL.get(act[5:], '작업 상태 자동')}" if act[5:] else f"{n}컷: 작업 상태를 되돌렸습니다", "ok")
        elif act == "ver:up":
            for c in cuts:
                WS.version_up(c)
            self.session.dirty = True
            self._after_status_change()
            self.toast("버전을 올렸습니다: " + ", ".join(f"{c.name} v{c.version}" for c in cuts[:4])
                       + (" …" if n > 4 else "") + " (작업 내용은 그대로 이어집니다)", "ok")
        elif act == "ver:back":
            v = WS.version_back(c0)
            self.session.dirty = True
            if self.session.cut is c0:
                self.session.touch("all")
                self.viewer._img_cache.clear()
            self._after_status_change()
            self._refresh_all()
            self.toast(f"{c0.name}: v{v}로 돌아갔습니다", "ok")
        elif act == "copy":
            self.copy_cut_settings(c0.id)
        elif act == "paste":
            self.paste_cut_settings([c.id for c in cuts])
        elif act == "paste_pick":
            self.paste_cut_settings_pick([c.id for c in cuts])
        elif act == "analyze":
            if any(c.tl for c in cuts):
                self._tl_action("analyze_force", [c.id for c in cuts])
            elif n == 1:
                if self.session.cut is None or self.session.cut.id != c0.id:
                    self.open_cut(c0.id)
                self.run_analysis()
        elif act == "render":
            self.save()
            for c in cuts:
                self.queue_page.add(prj, c)
            self.toast(f"{n}컷을 렌더 목록에 추가했습니다.")
        elif act.startswith("reviewed:"):
            self._tl_action(act, [c.id for c in cuts])
        else:
            self._cut_action(act, c0.id)

    def _after_status_change(self):
        cid = self.session.cut.id if self.session.cut else None
        self.cutlist.refresh(cid)
        self.tlstrip.refresh()
        self.tlstrip.set_current(cid)

    # ---- copy / paste of retouch values between cuts
    def copy_cut_settings(self, cid: Optional[str] = None):
        prj = self.session.project
        if prj is None:
            return
        cid = cid or (self._sel_cut_ids() or [None])[0]
        c = prj.cut(cid) if cid else None
        if c is None:
            return
        self._copied = (c.id, self.frame if (self.session.cut is c) else c.work_range[0])
        self.toast(f"보정값 복사: {c.name}  (Ctrl+V 붙여넣기 · Alt+V 골라서)", "ok")

    def _map_people(self, src, dst) -> dict:
        """dst person -> src person: the same member when the cuts are linked, else by appearance, else by size."""
        from ..core.copyset import suggest_mapping
        out = {}
        by_member = {getattr(q, "member", ""): q.id for q in src.persons if getattr(q, "member", "")}
        for q in dst.persons:
            mem = getattr(q, "member", "")
            if mem and mem in by_member:
                out[q.id] = by_member[mem]
        rest = [q for q in dst.persons if q.id not in out]
        if rest:
            try:
                sug = suggest_mapping(src, dst)
            except Exception:
                sug = {}
            used = set(out.values())
            for q in rest:
                sp = sug.get(q.id)
                if sp is not None and sp not in used:
                    out[q.id] = sp
                    used.add(sp)
        # still unmatched: pair the remaining people in order (a single-person cut always gets the source's
        # main person)
        left_src = [q.id for q in src.persons if q.id not in set(out.values())] or [q.id for q in src.persons][:1]
        for q in dst.persons:
            if q.id not in out and left_src:
                out[q.id] = left_src.pop(0) if len(left_src) > 1 else left_src[0]
        return out

    def _paste(self, ids: list, keys: Optional[set]):
        from ..core.copyset import CopyOptions, apply_copy
        prj = self.session.project
        cp = getattr(self, "_copied", None)
        if prj is None or cp is None:
            self.toast("먼저 Ctrl+C로 보정값을 복사하세요", "warn")
            return
        src = prj.cut(cp[0])
        if src is None:
            return
        dsts = [c for c in (prj.cut(i) for i in ids) if c is not None and c.id != src.id]
        if not dsts:
            return
        self.push_undo()
        n_people = 0
        for dst in dsts:
            mapping = self._map_people(src, dst)
            if keys is None:
                apply_copy(src, dst, mapping, cp[1], CopyOptions(amounts=True, widths=True, roles=False))
            else:
                for dpid, spid in mapping.items():
                    sp = src.params_for(spid).copy_static(cp[1])
                    dp = dst.params_for(dpid)
                    for k in keys:
                        if k in sp.amounts:
                            dp.amounts[k] = sp.amounts[k]
                        elif k == "__strength__":
                            dp.strength = sp.strength
                        elif hasattr(sp, k):
                            setattr(dp, k, getattr(sp, k))
                    from ..core.params import RETOUCH_KEYS, SKIN_BASIC_KEYS
                    if any(k in RETOUCH_KEYS and k not in BODY_SKIN_KEYS for k in keys):
                        dp.skin_basic = dict(sp.skin_basic) if set(SKIN_BASIC_KEYS) <= set(keys) else {}
            n_people += len(mapping)
            if not dst.analyzed:
                dst.pending_copy = dict(src=src.id, frame=cp[1], keys=sorted(keys) if keys else None)
        self.session.dirty = True
        if self.session.cut is not None and any(d is self.session.cut for d in dsts):
            self.session.touch("all")
            self.viewer._img_cache.clear()
        self._after_status_change()
        self._refresh_all()
        self.toast(f"{src.name}의 보정값을 {len(dsts)}컷에 붙여넣었습니다" + (f" ({len(keys)}개 항목)" if keys else ""), "ok")

    def paste_cut_settings(self, ids: Optional[list] = None):
        self._paste(ids or self._sel_cut_ids(), None)

    def paste_cut_settings_pick(self, ids: Optional[list] = None):
        from .dialogs import PastePickDialog
        prj = self.session.project
        cp = getattr(self, "_copied", None)
        if prj is None or cp is None or prj.cut(cp[0]) is None:
            self.toast("먼저 Ctrl+C로 보정값을 복사하세요", "warn")
            return
        dlg = PastePickDialog(self, prj.cut(cp[0]), cp[1])
        if not dlg.items:
            self.toast("복사한 컷에서 바뀐 보정값이 없습니다", "warn")
            return
        if dlg.exec() == QDialog.Accepted and dlg.selected():
            self._paste(ids or self._sel_cut_ids(), set(dlg.selected()))

    # ---- view layouts
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if getattr(self, "_was_max", False):
                self.showMaximized()
        else:
            self._was_max = self.isMaximized()
            self.showFullScreen()

    def toggle_focus(self):
        """Only the preview and the adjustment panel (cut list, timeline strip and timeline hidden)."""
        on = not getattr(self, "_focus", False)
        self._focus = on
        self.cutlist.setVisible(not on)
        self.tl_wrap.setVisible(not on)
        if on:
            self._strip_was = self.tlstrip.isVisible()
            self.tlstrip.setVisible(False)
        else:
            self.tlstrip.setVisible(getattr(self, "_strip_was", False))
        QTimer.singleShot(0, self.viewer.fit_view)

    def _cut_action(self, action: str, cid: str):
        if action == "analyze_all":
            self.analyze_all()
            return
        if action == "skin_all":
            self.skin_all()
            return
        prj = self.session.project
        if prj is None:
            return
        cut = prj.cut(cid)
        if cut is None:
            return
        if action == "rename":
            t = TextPrompt.ask(self, "이름 바꾸기", "컷 이름", cut.name)
            if t:
                cut.name = safe_name(t)
                self.session.dirty = True
                self.cutlist.refresh(self.session.cut.id if self.session.cut else None)
        elif action == "copy":
            if self.session.cut is None or self.session.cut.id != cid:
                self.open_cut(cid)
            self.copy_settings_dialog()
        elif action == "render":
            self.save()
            self.queue_page.add(prj, cut)
            self.toast(f"'{cut.name}'을(를) 렌더 목록에 추가했습니다.")
        elif action == "reveal":
            p = Path(cut.source if cut.info.kind == "video" else cut.info.seq_pattern).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
        elif action == "remove":
            if QMessageBox.question(self, "컷 제거", f"'{cut.name}'을(를) 프로젝트에서 제거할까요? 원본 파일은 지워지지 않습니다.") != QMessageBox.Yes:
                return
            was_current = self.session.cut is not None and self.session.cut.id == cid
            prj.cuts = [c for c in prj.cuts if c.id != cid]
            prj.save()
            if was_current:
                self.session.open_cut(None)
            self.cutlist.refresh()
            first = self._first_listed(prj)
            if first is not None and was_current:
                self.open_cut(first.id)
                self.cutlist.refresh(first.id)
            self._refresh_all()

    # ================================================================== jobs
    def run_job(self, title: str, fn, done, cancellable: bool = True, est_s: float = 0.0):
        if self.job is not None:
            self.toast("다른 작업이 진행 중입니다.", kind="warn")
            return
        self.stop_play()
        if getattr(self, "refresher", None) is not None:
            self.refresher.before_job()
        self.busy.start(title, cancellable, est_s)
        self.job = Job(fn, self)
        self.job.progressed.connect(self.busy.progress)

        def ok(res):
            self.job = None
            self.busy.hide()
            try:
                done(res)
            except Exception as e:
                self.show_error("오류", f"{e}\n{traceback.format_exc()}")
            self._gpu_notes()

        def fail(err):
            self.job = None
            self.busy.hide()
            self._gpu_notes()
            if err != "__cancelled__" and ("GpuMemoryError" in str(err) or "out of memory" in str(err).lower()):
                from ..track.models import GPU_OOM_TEXT
                self.show_error(title + " 실패", GPU_OOM_TEXT + "\n\n" + str(err))
                return
            if err == "__cancelled__":
                self.toast("취소했습니다.")
                pass
            elif "VersionNeeded" in str(err):
                self._versions_needed(str(err))
            else:
                self.show_error(title + " 실패", err)

        self.job.succeeded.connect(ok)
        self.job.failed.connect(fail)
        # the panel is painted before the worker starts: loading the models (DLLs, CUDA) holds the interpreter for
        # long stretches, and the panel came up seconds late (or after the work was well under way)
        self.busy.repaint()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        # started right here, not from a timer: a timer outlived the window in the tests and started a job for a
        # window already gone (a crash)
        self.job.start(QThread.LowPriority)

    def _versions_needed(self, text: str):
        """Work on the open cut needs model package versions that are not on this computer: nothing is switched to
        another version by itself; installing that version or analysing the whole cut again is asked."""
        from .modelsdlg import ask_versions
        got = ask_versions(self, text)
        if got == "models":
            self.open_models()
        elif got == "reanalyse":
            self.run_analysis()

    def open_models(self):
        from .modelsdlg import ModelsDialog
        d = ModelsDialog(self)
        d.exec()
        if d.changed:
            self._models_changed()

    def _models_changed(self):
        """A model version was put in use or taken out: sessions made with the old files are dropped (the next
        work opens the ones in use) and the open cut says again what it needs."""
        try:
            self.refresher.models.clear()
        except Exception:
            pass
        if self.session.cut is not None:
            self._version_note(self.session.cut, again=True)
            self.inspector.refresh()

    def _version_note(self, cut, again: bool = False) -> str:
        """Said once per cut and state: the package versions this cut was analysed with are not here."""
        from .modelsdlg import needed_text
        try:
            txt = needed_text(cut) if cut is not None and cut.analyzed else ""
        except Exception:
            txt = ""
        seen = getattr(self, "_version_noted", None)
        if seen is None:
            seen = self._version_noted = {}
        if txt and (again or seen.get(cut.id) != txt):
            self.toast(txt, kind="warn")
        if cut is not None:
            seen[cut.id] = txt
        return txt

    def _batch_warned(self, cid: str, text: str):
        seen = getattr(self, "_batch_warn_seen", set())
        if text in seen:
            return
        seen.add(text)
        self._batch_warn_seen = seen
        self.toast(text, kind="warn")

    def _gpu_notes(self):
        """Models that went to the CPU because the GPU had no room: say so once (an analysis then takes far longer)."""
        try:
            from ..track.models import take_cpu_fallbacks
            fb = take_cpu_fallbacks()
        except Exception:
            return
        try:
            from ..track.models import device_label
            self.dev_lbl.setText(f"추론: {device_label()}")     # by what the models really ran on
        except Exception:
            pass
        if fb:
            names = ", ".join(sorted({n for n, _ in fb}))[:120]
            why = "GPU 메모리가 부족해" if any(w == "GPU 메모리 부족" for _, w in fb) else "GPU에서 열지 못해"
            self.toast(f"{why} 일부 모델을 CPU로 실행했습니다 ({names}). 많이 느려질 수 있습니다", kind="warn")

    def _cancel_job(self):
        if self.job:
            self.job.cancel()
            self.busy.cancelling()

    def set_accuracy(self, v: str):
        self.accuracy = v
        s = settings()
        s.accuracy = v
        s.save()

    def run_analysis(self):
        s = self.session
        cut = s.cut
        if cut is None or s.tracker is None:
            return
        if not s.proxy_ready():
            # only an old proxy in another colour view is left (the original is missing): not analysed in that view
            self.toast("원본 파일이 없어 분석할 수 없습니다. 원본을 원래 자리에 두세요.", "error")
            return
        if cut.analyzed and cut.joint_keys:
            if QMessageBox.question(self, "다시 분석",
                                    "다시 분석해도 인물 이름·역할·보정값·관절 키는 같은 인물에 그대로 유지됩니다. 진행할까요?") != QMessageBox.Yes:
                return
        tr = s.tracker
        acc = self.accuracy

        prj = s.project
        # 분석 시작 on the open cut is the user's own call: an earlier quick look's 'nobody, skipped' is forgotten
        from ..track.prescan import clear_skip
        cdir = Path(prj.path) / "cuts" / cut.id
        if clear_skip(cut, cdir if cdir.is_dir() else None):
            s.dirty = True

        def work(progress, cancelled):
            res = tr.analyze(progress, cancelled, accuracy=acc)
            self._fusion_res = None
            try:
                from ..learn.refiner import collect_auto
                from ..learn import fusion as FU
                progress(1.0, "자가학습 데이터 모으는 중")
                collect_auto(prj, cut, tr)
                if FU.collect(prj, cut, tr):
                    self._fusion_res = FU.refit()
            except Exception:
                traceback.print_exc()
            return res

        from .batchdlg import DEFAULT_SPF
        a_, b_ = cut.work_range
        nfr = max(1, b_ - a_)
        t_start = time.time()
        spf0 = float(getattr(settings(), "analysis_spf", 0.0) or 0.0)

        def done(_):
            if nfr >= 24:                                   # learn this PC's pace for the next estimate
                st = settings()
                spf = (time.time() - t_start) / nfr
                st.analysis_spf = round(spf if not st.analysis_spf else 0.5 * st.analysis_spf + 0.5 * spf, 3)
                st.save()
            cut.stats.analysis_runs += 1
            cut.stats.analysis_seconds = round(time.time() - t_start, 1)
            ok = tr.auto_ok_rate(cut.tracking.conf_threshold)
            cut.stats.auto_ok_rate = ok
            log_event("analysis", s.project.name, cut.name, frames=cut.work_range[1] - cut.work_range[0], auto_ok=ok,
                      persons=len(cut.persons))
            self._apply_pending_copy(cut)
            if cut.tl:
                from ..track import status as ST
                from ..learn import identity as ID
                ST.evaluate(cut, tr)
                ID.collect(s.project, cut)
                self.tlstrip.update_counts()
                QTimer.singleShot(1000, lambda: self._relink(quiet=True))
            targets = cut.targets()
            self.sel_pid = targets[0].id if targets else (cut.persons[0].id if cut.persons else None)
            s.touch()
            self.save()
            self.viewer.message = ""
            self.inspector.show_tab("reshape")
            self._refresh_all()
            self.toast(f"분석 완료 · 인물 {len(cut.persons)}명 · 자동 추적 성공률 {ok * 100:.0f}%", kind="ok")
            from ..learn import allowed as learn_allowed
            if cut.retouch_ref and Path(cut.retouch_ref).exists() and learn_allowed(s.project, cut):
                QTimer.singleShot(1500, lambda: self.learn_style(cut))
            fr = getattr(self, "_fusion_res", None)
            if fr and fr.get("active"):
                log_event("self_learning", s.project.name, cut.name, fusion_gain=fr.get("gain"), cuts=fr.get("cuts"))
            if learn_allowed(s.project, cut):
                # automatic learning starts only from a project where it was switched on
                QTimer.singleShot(4000, self.models_page.maybe_self_train)

        self.run_job("분석 중", work, done, est_s=nfr * (spf0 or DEFAULT_SPF))

    # ================================================================== edited timeline
    @staticmethod
    def _is_after(p: str) -> bool:
        from ..learn.style import split_after
        return split_after(Path(p).name) is not None

    def import_timeline(self, edit: str = "", baked: str = "", after: str = ""):
        from .tldialogs import TimelineImportDialog
        from ..core.tlimport import import_timeline
        if self.session.project is None:
            self.new_project()
            if self.session.project is None:
                return
        prj = self.session.project
        dlg = TimelineImportDialog(self, edit, baked, bool(prj.timeline_cuts()))
        if dlg.exec() != QDialog.Accepted:
            return
        edit, baked, after, replace, auto = dlg.values()
        if not (edit and Path(edit).exists() and baked and Path(baked).exists()):
            self.toast("편집 파일과 베이크 영상을 모두 골라 주세요", "warn")
            return
        # the bake itself, loaded before as one plain cut (dropped first): the timeline takes its place
        same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        plain = [c for c in prj.cuts if not c.tl and c.info.kind == "video" and same(c.source, baked)
                 and not c.analyzed]
        if self.session.cut is not None and ((self.session.cut.tl and replace) or self.session.cut in plain):
            if not self.save():
                return
            self.session.open_cut(None)

        def work(progress, cancelled):
            return import_timeline(prj, edit, baked, progress, cancelled, replace=replace)

        def done(res):
            if plain:
                ids = {c.id for c in plain}
                prj.cuts = [c for c in prj.cuts if c.id not in ids]
            cuts = prj.timeline_cuts()
            if after and Path(after).exists():
                for c in cuts:
                    if c.tl.get("kind") == "clip":
                        c.retouch_ref = after
            prj.save()
            self.thumbcache.drop([c.id for c in cuts])     # the import wrote their JPGs again
            self.tlstrip.refresh()
            self.cutlist.refresh()
            al = res.get("align", {})
            self.toast(f"타임라인 {len(cuts)}컷 · 경계 일치 {al.get('matched', 0)}/{al.get('total', 0)}"
                       + (f" · 경고 {len(res.get('warnings', []))}개 (ⓘ에서 확인)" if res.get("warnings") else ""), "ok")
            if res.get("warnings"):
                QTimer.singleShot(300, lambda: self._tl_action("info", []))
            if cuts:
                first = next((c for c in cuts if c.tl.get("kind") == "clip"), cuts[0])
                self.open_cut(first.id)
            if auto:
                QTimer.singleShot(800, lambda: self._tl_action("analyze", []))

        self.run_job("편집 타임라인 불러오는 중", work, done)

    def step_cut(self, d: int):
        if self.session.project is None:
            return
        if self.session.project.timeline_cuts():
            nxt = self.tlstrip.neighbour(d)
        else:
            ids = [c.id for c in self.session.project.cuts]
            cur = self.session.cut.id if self.session.cut else None
            i = ids.index(cur) if cur in ids else -1
            j = i + d
            nxt = ids[j] if 0 <= j < len(ids) else None
        if nxt:
            self.open_cut(nxt)
            self.cutlist.refresh(nxt)

    def _tl_action(self, action: str, ids: list):
        from ..track import status as ST
        prj = self.session.project
        if prj is None:
            return
        cuts = prj.timeline_cuts()
        sel = [c for c in cuts if c.id in set(ids)]
        if action == "analyze":
            self.analyze_all(ids)
        elif action == "analyze_force":
            self._start_batch([c for c in (prj.cut(i) for i in ids) if c is not None], "선택 컷 분석")
        elif action == "stop":
            self.batch.stop()
            self.toast("지금 분석 중인 컷이 끝나면 멈춥니다")
        elif action == "apply":
            from .tldialogs import BatchApplyDialog
            from ..core import presets as P
            tgt = [c for c in (sel or cuts) if c.analyzed]
            if not tgt:
                self.toast("분석된 컷이 없습니다", "warn")
                return
            dlg = BatchApplyDialog(self, len(tgt))
            if dlg.exec() != QDialog.Accepted:
                return
            preset, all_faces = dlg.values()
            if preset is None:
                return
            n = 0
            for c in tgt:
                if all_faces:
                    for r in (c.status or {}).get("metrics", {}).get("persons", []):
                        if r.get("face_px", 0) >= ST.SMALL_FACE and r.get("face_share", 0) >= 0.2:
                            q = c.person(r["pid"])
                            if q is not None and q.role != "ignore":
                                q.role = "target"
                for q in c.targets():
                    P.apply(c.params_for(q.id), preset)
                    n += 1
            self.session.dirty = True
            self.save()
            if self.session.cut is not None and self.session.cut in tgt:
                self.session.touch()
                self._refresh_all()
            self.tlstrip.update_counts()
            self.toast(f"'{preset['name']}' 적용: {len(tgt)}컷 · 인물 {n}명", "ok")
        elif action == "members":
            from .members import MembersDialog
            if not prj.members:
                self._relink(quiet=True)
            MembersDialog(self).exec()
        elif action.startswith("status:"):
            from ..learn import statuscal as SC
            v = action.split(":", 1)[1]
            learn = self._learn_allowed()
            for c in sel:
                ST.set_user(c, v or None)
                if v and learn:
                    SC.record(c, v)
            if v and learn:
                SC.refit()
            self.session.dirty = True
            self.tlstrip.update_counts()
        elif action.startswith("reviewed:"):
            from ..learn import statuscal as SC
            for c in sel:
                ST.set_reviewed(c, action.endswith("1"))
                if action.endswith("1") and not (c.status or {}).get("user") and self._learn_allowed():
                    SC.record(c, (c.status or {}).get("auto", ""))      # checked and left as it was: a confirmation
            self.session.dirty = True
            self.tlstrip.update_counts()
        elif action == "review":
            order = {"bad": 0, "small": 1, "good": 2, "pending": 3}
            rest = [c for c in cuts if not (c.status or {}).get("reviewed") and ST.effective(c) in order]
            if not rest:
                self.toast("검수할 컷이 없습니다 (모두 검수 완료 또는 리터칭 불필요)", "ok")
                return
            rest.sort(key=lambda c: (order[ST.effective(c)], c.tl.get("index", 0)))
            cur = self.session.cut
            if cur is not None and cur.tl and not (cur.status or {}).get("reviewed") and cur in rest:
                ST.set_reviewed(cur, True)
                if self._learn_allowed():
                    from ..learn import statuscal as SC
                    SC.record(cur, (cur.status or {}).get("user") or (cur.status or {}).get("auto", ""))
                    SC.refit()
                self.session.dirty = True
                rest = [c for c in rest if c is not cur]
                self.tlstrip.update_counts()
            if rest:
                nxt = rest[0]
                self.open_cut(nxt.id)
                self.cutlist.refresh(nxt.id)
                self.toast(f"검수 {nxt.name}: {ST.LABEL[ST.effective(nxt)]} · {ST.reason(nxt)} "
                           f"· 확인 후 다시 '검수'를 누르면 검수 완료로 표시하고 다음 컷으로 갑니다 (남은 {len(rest)}컷)")
            else:
                self.toast("모든 컷 검수 완료", "ok")
        elif action in ("export", "export_sel", "export_full"):
            from .tldialogs import TimelineExportDialog
            if not cuts:
                self.toast("타임라인이 없습니다. 편집 타임라인을 먼저 불러오세요.", "warn")
                return
            if self.batch.busy:
                self.toast("일괄 분석이 끝난 뒤 내보내는 것이 안전합니다 (분석 중인 컷은 원본으로 들어갑니다)", "warn")
            self.save()
            last = dict(self._last_export or {})
            if action == "export_full":
                last["mode"] = "full"
            dlg = TimelineExportDialog(self, prj, [c.id for c in sel] if (sel and action == "export_sel") else None, last)
            if dlg.exec() != QDialog.Accepted:
                return
            exp = dlg.export()
            self._last_export = exp.to_dict()
            self.queue_page.add_timeline(prj, exp)
            self.toast("렌더 목록에 추가했습니다. 렌더 화면에서 시작하세요.", "ok")
        elif action == "info":
            tl = prj.timeline or {}
            al = tl.get("align", {})
            lines = [f"편집 파일: {Path(tl.get('edit', '')).name} ({tl.get('fmt', '')})",
                     f"베이크 영상: {Path(tl.get('baked', '')).name} · {tl.get('frames', 0)}프레임 · {tl.get('fps', 0):.3f} fps",
                     f"시퀀스 시작: {tl.get('start_tc', '')}{' (드롭프레임)' if tl.get('drop') else ''} · 영상 위치 보정 {tl.get('offset', 0):+d}프레임",
                     f"편집 경계와 화면 전환 일치: {al.get('matched', 0)}/{al.get('total', 0)}"]
            if al.get("unmatched"):
                lines.append("화면 전환이 없는 편집 경계(같은 화면으로 이어지는 컷이거나 편집 파일 불일치): 시퀀스 프레임 "
                             + ", ".join(str(x) for x in al["unmatched"][:12]))
            notes = [f"{c.name}: {c.tl.get('note')}" for c in cuts if c.tl.get("note")]
            if notes:
                lines.append("특이 컷: " + " / ".join(notes[:12]))
            if tl.get("warnings"):
                lines.append("")
                lines += [f"· {w}" for w in tl["warnings"]]
            QMessageBox.information(self, "편집 타임라인", "\n".join(lines))

    # ================================================================== whole project: 전체 분석 / 전체 피부 보정
    @staticmethod
    def _clip_cuts(prj, ids=None) -> list:
        cuts = [c for c in prj.cuts if c.tl.get("kind", "clip") == "clip"]
        if ids:
            sel = [c for c in cuts if c.id in set(ids)]
            return sel or cuts
        return cuts

    def _busy_with(self) -> str:
        if self.job is not None:
            return "진행 중인 작업이 끝난 뒤 할 수 있습니다."
        if self.batch.busy:
            return "일괄 분석이 진행 중입니다."
        if getattr(self, "_opening", None) is not None:
            return "프로젝트를 여는 중입니다."
        return ""

    def analyze_all(self, ids=None):
        """Every cut not analysed yet (or the cuts selected in the cut list / timeline), one after another, with
        the app covered by the progress window (pause, go on, cancel)."""
        prj = self.session.project
        if prj is None:
            self.toast("먼저 프로젝트를 여세요", "warn")
            return
        why = self._busy_with()
        if why:
            self.toast(why, "warn")
            return
        if ids is None:
            ids = [i for i in self.cutlist.selected_ids()] if len(self.cutlist.selected_ids()) > 1 else []
        pool = self._clip_cuts(prj, ids)
        picked = bool(ids) and len(pool) < len(self._clip_cuts(prj))
        # cuts picked one by one are analysed even when the quick look skipped them before (1.22.2)
        todo = [c for c in pool if not c.analyzed and (picked or (c.status or {}).get("auto") != "none")]
        if not todo:
            self.toast("분석할 컷이 없습니다 (모두 분석했거나 인물 없음)")
            return
        from .batchdlg import AnalyzeAllDialog
        frames = sum(int(c.info.frames) for c in todo)
        title = "선택 컷 분석" if ids else "전체 분석"
        dlg = AnalyzeAllDialog(self, len(todo), frames, sum(1 for c in pool if c.analyzed),
                               float(getattr(settings(), "analysis_spf", 0.0) or 0.0), title)
        if dlg.exec() != QDialog.Accepted:
            return
        self._batch_skin = dlg.skin()
        self._batch_preset = None
        self._batch_dlg = None
        self._batch_skin = None
        self._start_batch(todo, title, prescan=not picked)

    def _start_batch(self, todo: list, title: str = "전체 분석", prescan: bool = False, force=None):
        """force (default: when no quick look is asked for): the user picked these cuts, they are analysed whatever
        a quick look said before or would say now."""
        prj = self.session.project
        if prj is None or not todo:
            return
        force = (not prescan) if force is None else bool(force)
        if force:
            # before the project is saved for the analysis process: an earlier 'nobody, skipped' is forgotten
            from ..track.prescan import clear_skip
            root = Path(prj.path) / "cuts"
            for c in todo:
                clear_skip(c, root / c.id if (root / c.id).is_dir() else None)
        if not self.save():                     # the analysis process reads the project from disk
            return
        # 1.18: no proxy first. A bake proxy that exists is still shared by the shots; else each shot's analysis
        # reads its stretch of the original
        from ..core.tlimport import link_bake_proxy
        link_bake_proxy(prj)
        from .batchdlg import BatchProgress
        run = {c.id: -1.0 for c in todo}
        self.tlstrip.set_running(run, f"분석 대기 {len(todo)}컷")
        dlg = BatchProgress(self, title, [c.id for c in todo], {c.id: c.name for c in todo},
                            {c.id: int(c.info.frames) for c in todo},
                            float(getattr(settings(), "analysis_spf", 0.0) or 0.0))
        dlg.pauseToggled.connect(self._batch_pause)
        dlg.cancelNow.connect(self.batch.kill)
        dlg.cancelAfter.connect(self.batch.stop)
        self._batch_dlg = dlg
        dlg.open()
        try:
            # the analysis runs in its own process on the same GPU: what this window holds there (the preview's face
            # parser) is let go, it comes back by itself when it is needed
            from ..track.models import free_gpu_memory
            free_gpu_memory()
        except Exception:
            pass
        self.batch.start(prj.path, [c.id for c in todo], self.accuracy, True, prescan=prescan, force=force)

    def _batch_skipped(self, cid: str, info: str):
        """The quick look found nobody in this cut: not analysed, with the reason (다시 분석 takes it in again)."""
        prj = self.session.project
        cut = prj.cut(cid) if prj else None
        if cut is None:
            return
        try:
            r = json.loads(info)
        except Exception:
            r = {}
        figs = {k: r[k] for k in ("looked", "read", "best_score", "tallest", "seconds") if k in r}
        cut.status = dict(cut.status or {}, auto="none", reason=r.get("reason") or "인물 없음 (빠른 검사)",
                          prescan=True, prescan_info=figs)
        run = dict(self.tlstrip.running)
        run.pop(cid, None)
        self.tlstrip.set_running(run, self.tlstrip.run_lbl.text())
        dlg = getattr(self, "_batch_dlg", None)
        if dlg is not None:
            dlg.cut_done(cid, False)
        self.session.dirty = True
        try:
            prj.save()
        except Exception:
            traceback.print_exc()
        self.tlstrip.refresh()
        self.cutlist.refresh(self.session.cut.id if self.session.cut else None)

    def _batch_pause(self, on: bool):
        dlg = getattr(self, "_batch_dlg", None)
        if on:
            if not self.batch.pause() and dlg is not None:
                dlg.set_paused(False)
                self.toast("지금은 멈출 수 없습니다 (컷이 시작되는 중). 잠시 뒤 다시 누르세요.", "warn")
        else:
            self.batch.resume()

    def skin_all(self):
        """The skin basic adjustment on the targets of every analysed cut."""
        prj = self.session.project
        if prj is None:
            self.toast("먼저 프로젝트를 여세요", "warn")
            return
        why = self._busy_with()
        if why:
            self.toast(why, "warn")
            return
        cuts = self._clip_cuts(prj)
        done = [c for c in cuts if c.analyzed]
        if not done:
            self.toast("분석된 컷이 없습니다. 먼저 전체 분석을 하세요.", "warn")
            return

        def hand(pp) -> bool:
            return bool(pp.retouch_active()) and not pp.skin_basic

        people = [(c, q) for c in done for q in c.targets()]
        n_hand = sum(1 for c, q in people if hand(c.params_for(q.id)))
        from .batchdlg import SkinAllDialog, BatchProgress
        dlg = SkinAllDialog(self, len(done), len(people), n_hand, len(cuts) - len(done))
        if dlg.exec() != QDialog.Accepted:
            return
        level, remove, skip_hand = dlg.values()
        from ..core.params import skin_basic_values
        from ..core.project import Curve
        vals = skin_basic_values(level, remove)
        self.save()
        cur = self.session.cut
        if cur is not None and cur in done:
            self.push_undo()
        pg = BatchProgress(self, "전체 피부 보정", [c.id for c in done], {c.id: c.name for c in done},
                           {c.id: 1 for c in done}, 0.01, can_pause=False)
        stop = {"now": False}
        pg.cancelNow.connect(lambda: stop.__setitem__("now", True))
        pg.cancelAfter.connect(lambda: stop.__setitem__("now", True))
        pg.open()
        n = skipped = 0
        for c in done:
            if stop["now"]:
                break
            pg.started(c.id)
            QApplication.processEvents()
            for q in c.targets():
                pp = c.params_for(q.id)
                if skip_hand and hand(pp):
                    skipped += 1
                    continue
                for k, v in vals.items():
                    pp.amounts[k] = Curve(base=float(v))
                pp.skin_basic = {"level": level, "remove": list(remove)}
                n += 1
            pg.cut_done(c.id, False)
        pg.finish()
        self.session.dirty = True
        self.save()
        if cur is not None:
            self.session.touch("ret")
        self._refresh_all()
        self.viewer.update()
        self.toast(f"전체 피부 보정: {n}명 적용" + (f" · 손으로 조정한 {skipped}명 건너뜀" if skipped else ""), "ok")

    def _batch_before(self, cid: str) -> bool:
        prj = self.session.project
        cut = prj.cut(cid) if prj else None
        if cut is None:
            return False
        # the batch process reads the project from disk and writes its result over the cut: unsaved work first
        if not self.save():
            try:
                self.batch.stop()               # no more cuts (each would fail the same way, one error box each)
            except Exception:
                pass
            try:
                self.toast("저장하지 못해 일괄 분석을 멈췄습니다", kind="error")
            except Exception:
                pass
            return False
        if self.session.cut is not None and self.session.cut.id == cid:
            # the open cut holds its proxy files: the batch process could not rewrite them
            self.session.open_cut(None)
            self._reopen_after = cid
            self.viewer.message = "일괄 분석 중인 컷입니다"
            self.viewer.update()
        return True

    def _batch_progress(self, cid: str, f: float, msg: str):
        run = dict(self.tlstrip.running)
        run[cid] = f
        left = sum(1 for v in run.values() if v < 0)
        cut = self.session.project.cut(cid) if self.session.project else None
        self.tlstrip.set_running(run, f"분석 중 {cut.name if cut else ''} {f * 100:.0f}% · {msg} · 대기 {left}")
        dlg = getattr(self, "_batch_dlg", None)
        if dlg is not None:
            dlg.progress(cid, f, msg)

    def _batch_cut_done(self, cid: str, ok: bool, err: str):
        from ..core.project import Cut
        from ..batch import result_path
        prj = self.session.project
        run = dict(self.tlstrip.running)
        run.pop(cid, None)
        self.tlstrip.set_running(run, self.tlstrip.run_lbl.text())
        self.drop_undo(cid)
        cut = prj.cut(cid) if prj else None
        if cut is None:
            return
        dlg = getattr(self, "_batch_dlg", None)
        from .batchrun import CANCELLED
        if err == CANCELLED:
            # stopped in the middle: the half-done analysis is dropped, the cut stays as it was
            try:
                result_path(prj, cut).unlink()
            except OSError:
                pass
            if dlg is not None:
                dlg.drop(cid)
            if getattr(self, "_reopen_after", None) == cid:
                self._reopen_after = None
                self.open_cut(cid)
            return
        f = result_path(prj, cut)
        if dlg is not None and (not ok or not f.exists()):
            dlg.cut_done(cid, False)
        if not ok or not f.exists():
            s = dict(cut.status or {})
            s.update(auto="bad", reason=f"분석 실패: {err or '알 수 없는 오류'}"[:120], failed=True)
            cut.status = s
            self.session.dirty = True
            self.tlstrip.update_counts()
            return
        try:
            import json as _json
            d = _json.loads(f.read_text(encoding="utf-8"))
            new = Cut.from_dict(d["cut"])
            if dlg is not None:
                dlg.cut_done(cid, bool(d.get("analyzed")))
        except Exception as e:
            if dlg is not None:
                dlg.cut_done(cid, False)
            self.toast(f"{cut.name}: 분석 결과를 읽지 못했습니다 ({e})", "error")
            return
        # the user may have changed status or review marks meanwhile: keep those
        keep = {k: v for k, v in (cut.status or {}).items() if k in ("user", "reviewed")}
        new.status = dict(new.status or {}, **keep)
        i = prj.cuts.index(cut)
        prj.cuts[i] = new
        new.status.pop("failed", None)
        if self._batch_preset:
            from ..core import presets as P
            for q in new.targets():
                P.apply(new.params_for(q.id), self._batch_preset)
        if getattr(self, "_batch_skin", None) and d.get("analyzed"):
            from ..core.params import skin_basic_values
            from ..core.project import Curve
            lv, rm = self._batch_skin
            vals = skin_basic_values(lv, rm)
            for q in new.targets():
                pp = new.params_for(q.id)
                for k, v in vals.items():
                    pp.amounts[k] = Curve(base=float(v))
                pp.skin_basic = {"level": lv, "remove": list(rm)}
        try:
            f.unlink()
        except OSError:
            pass
        prj.save()
        self.tlstrip.refresh()
        self.tlstrip.set_running(run, self.tlstrip.run_lbl.text())
        self.cutlist.refresh(self.session.cut.id if self.session.cut else None)
        if getattr(self, "_reopen_after", None) == cid:
            self._reopen_after = None
            self.open_cut(cid)
        if d.get("analyzed"):
            log_event("analysis", prj.name, new.name, frames=new.info.frames, auto_ok=new.stats.auto_ok_rate,
                      persons=len(new.persons), batch=True)

    def _relink(self, quiet: bool = False) -> Optional[dict]:
        """Identity self-learning on the newest cuts, then people linked across the timeline."""
        prj = self.session.project
        if prj is None:
            return None
        try:
            from ..learn import identity as ID
            from ..track import xlink as XL
            if self._learn_allowed():
                ID.refit()
            r = XL.link(prj)
        except Exception as e:
            traceback.print_exc()
            if not quiet:
                self.toast(f"인물 연결 실패: {e}", "error")
            return None
        self.session.dirty = True
        self.save()
        self._refresh_all()
        if not quiet:
            self.toast(f"인물 연결: 여러 컷에 나오는 멤버 {r['multi']}명" + (f" · 확인 필요 {r['low']}명 (인물 버튼)" if r["low"] else ""), "ok")
        return r

    def _batch_all_done(self, done: int, failed: int):
        dlg = getattr(self, "_batch_dlg", None)
        self._batch_dlg = None
        self._batch_skin = None
        if dlg is not None:
            if dlg.meas_frames >= 24 and dlg.meas_sec > 0:
                st = settings()
                spf = dlg.meas_sec / dlg.meas_frames
                st.analysis_spf = round(spf if not st.analysis_spf else 0.5 * st.analysis_spf + 0.5 * spf, 3)
                st.save()
            dlg.finish()
        self.cutlist.refresh(self.session.cut.id if self.session.cut else None)
        self.tlstrip.set_running({}, "")
        self.tlstrip.update_counts()
        self.toast(f"일괄 분석 끝: {done}컷" + (f" · 실패 {failed}컷" if failed else ""), "ok" if not failed else "warn")
        if done:
            QTimer.singleShot(2500, lambda: self._relink())
        if self._learn_allowed():
            QTimer.singleShot(4000, self.models_page.maybe_self_train)

    # ================================================================== retouch style learning
    def learn_style(self, cut=None, after: str = ""):
        """Compares this plate's retouch with the retoucher's own result and updates the preset 학습된_스타일."""
        from ..learn import style as SL
        s = self.session
        prj = s.project
        cut = cut or s.cut
        if prj is None or cut is None:
            return
        after = after or cut.retouch_ref
        if not after or not Path(after).exists():
            self.toast("수작업 결과 파일을 찾을 수 없습니다", "warn")
            return
        if not cut.analyzed:
            cut.retouch_ref = after
            s.dirty = True
            self.toast("분석이 끝나면 수작업 결과로 스타일을 학습합니다", "ok")
            return
        cut.retouch_ref = after
        if s.cut is cut and s.tracker is not None:
            tr = s.tracker
        else:
            from ..core.frames import frames_for
            from ..track.analyze import CutTracker
            fr = frames_for(prj, cut)
            if fr is None:
                self.toast("원본 파일을 찾을 수 없습니다", "error")
                return
            tr = CutTracker(prj, cut, fr)

        def work(progress, cancelled):
            sample = SL.fit_pair(cut, tr, after, progress, cancelled)
            return sample, SL.add_sample(sample)

        def done(res):
            sample, agg = res
            cut.style_learned = sample
            if Path(after).suffix.lower() in SL.IMAGE_EXT:
                # a retouched still of this very cut: retouch the cut the same way, on the faces that were edited
                ids = {int(pid) for pid, _ in sample.get("frames", [])}
                for q in cut.persons:
                    if q.id in ids:
                        q.role = "target"
                        pp = cut.params_for(q.id)
                        for k in pp.amounts:
                            if k in SL.LEARNED_KEYS:
                                pp.amounts[k].base = float(sample["values"].get(k, 0.0))
                        pp.skin_detail = float(sample.get("detail", pp.skin_detail))
                        pp.skin_basic = {}
                if s.cut is cut:
                    s.touch()
                    self._refresh_all()
            s.dirty = True
            self.save()
            v = sample["values"]
            parts = [f"피부결 {v.get('skin', 0):.0f}", f"질감 유지 {sample['detail']:.0f}"]
            for k, t in (("blemish", "잡티"), ("redness", "홍조"), ("dark_circle", "다크서클"), ("nasolabial", "팔자"),
                         ("shine", "번들"), ("grain", "그레인")):
                if v.get(k, 0) > 0:
                    parts.append(f"{t} {v[k]:.0f}")
            gain = 100 * (sample["err_before"] - sample["err_after"]) / max(1e-6, sample["err_before"])
            tail = f" · 누적 {agg['n']}개 → 프리셋 '{SL.PRESET_NAME}' 갱신" if agg else " · 이 영상은 보정 안 함이 가장 가까워 프리셋에 반영하지 않음"
            self.toast(f"스타일 학습 {cut.name}: {' · '.join(parts)} · 수작업과 차이 {gain:.0f}% 줄임{tail}", "ok")

        self.run_job(f"수작업 결과로 스타일 학습 · {cut.name}", work, done)

    def _apply_pending_copy(self, cut):
        pc = cut.pending_copy
        if not pc:
            return
        src = self.session.project.cut(pc.get("src", ""))
        if src is not None:
            if "opt" in pc:
                mapping = suggest_mapping(src, cut)
                apply_copy(src, cut, mapping, int(pc.get("frame", 0)), CopyOptions(**pc.get("opt", {})))
            else:
                # Ctrl+V / Alt+V onto a cut that was not analysed yet: same paste now that its people are known
                cut.pending_copy = {}
                keep = getattr(self, "_copied", None)
                self._copied = (src.id, int(pc.get("frame", 0)))
                try:
                    self._paste([cut.id], set(pc["keys"]) if pc.get("keys") else None)
                finally:
                    self._copied = keep
        cut.pending_copy = {}

    def rebuild_matte(self):
        s = self.session
        if s.tracker is None or s.tracker.store is None:
            self.toast("먼저 분석을 실행하세요.", kind="warn")
            return
        cut = s.cut
        tr = s.tracker
        acc = self.accuracy
        prev = cut.tracking.seg
        cut.tracking.seg = "sam2"

        def work(progress, cancelled):
            tr.rebuild_matte(progress, cancelled, accuracy=acc)
            return tr.has_seg()

        def done(ok):
            if not ok:
                cut.tracking.seg = prev
            s.touch()
            self.viewer._img_cache.clear()
            self.save()
            self._refresh_all()
            model = (tr.seg_meta() or {}).get("model") or "SAM"
            self.toast(f"{model}(으)로 컷 전체의 인물 매트를 다시 만들고 몸 윤곽 점을 다시 쟀습니다." if ok else
                       "SAM 모델을 쓸 수 없어 RVM 매트를 유지했습니다. 로그를 확인하세요.", kind="ok" if ok else "warn")
            self.refresher.changed.emit(cut.id)

        self.run_job("인물 매트 강제 갱신 중", work, done)

    def range_reanalyse(self):
        """구간 다시 분석 (§3): one person, the selected frames, compared before it is applied."""
        from ..track import rangefix as RF
        from .rangefix_dialog import CompareDialog, RangeDialog
        s = self.session
        cut, tr = s.cut, s.tracker
        if cut is None or tr is None or tr.store is None:
            self.toast("먼저 분석을 실행하세요.", kind="warn")
            return
        if not self.selection or self.sel_pid is None:
            self.toast("인물을 고르고 타임라인에서 Shift+드래그로 구간을 선택하세요.", kind="warn")
            return
        a, b = int(self.selection[0]), int(self.selection[1]) + 1
        pid = self.sel_pid
        d = RangeDialog(self, cut, pid, a, b)
        if not d.exec():
            return
        if d.item() != "face":
            return
        if not getattr(tr.store, "has_face", False):
            self.toast("얼굴 점이 없는 분석입니다. 다시 분석한 뒤 쓰세요.", kind="warn")
            return

        def work(progress, cancelled):
            cand = RF.candidate(tr, pid, a, b, progress, cancelled)
            if cand is None:
                return None
            return cand, RF.compare(tr, cand)

        def done(res):
            if res is None:
                self.toast("이 구간에는 다시 계산할 얼굴이 없습니다 (작거나 가려진 얼굴).", kind="warn")
                return
            if s.tracker is not tr:
                return                                  # the cut was left meanwhile: nothing applied
            cand, cmp = res
            dlg = CompareDialog(self, tr, cut, cand, cmp)
            if not dlg.exec():
                self.toast("적용하지 않았습니다. 지금 결과를 그대로 씁니다.")
                return

            def apply_work(progress, cancelled):
                RF.apply(tr, cand)
                return True

            def applied(_):
                s._mask_cache = {}
                s.touch("ret")
                s.touch("geo")
                self.viewer._img_cache.clear()
                s.dirty = True
                self._refresh_all()
                self.toast(f"{cand['a']}~{cand['b'] - 1} 프레임의 얼굴·목 마스크를 바꿨습니다.", kind="ok")
            self.run_job("구간 적용 중", apply_work, applied, cancellable=False)
        self.run_job("구간 다시 분석 중 (얼굴·목)", work, done)

    def make_ai_review(self, open_folder: bool = True, images: bool = False):
        """AI 검수 리포트: read from the saved project, so it is saved first; nothing is analysed again. Without
        pictures of the footage unless asked for (instruction 2026-11-20 §4-4)."""
        prj = self.session.project
        if prj is None:
            return
        if not self.save():
            self.toast("저장하지 못해 검수 자료를 만들지 않았습니다", kind="error")
            return
        path = prj.path

        def work(progress, cancelled):
            from .. import review as RV
            return str(RV.build(path, images=images, progress=progress, cancel=cancelled))

        def done(out):
            self.last_review = out
            try:
                shown = Path(out).relative_to(path).as_posix()
            except ValueError:
                shown = out
            self.toast(f"AI 검수 리포트를 만들었습니다: {shown} (report.md, review.json"
                       + (", img: 영상 프레임 포함)" if images else "; 영상·크롭·미디어 경로 없음)"), kind="ok")
            if open_folder:
                QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        self.run_job("AI 검수 리포트 만드는 중", work, done)

    def _matte_state_changed(self, cid: str):
        cut = self.session.cut
        if cut is not None and cut.id == cid:
            self.inspector.refresh("track")
        self.cutlist.update_rows()

    def refresh_mattes_now(self, cut_ids, then=None, project=None):
        """Matte the marked frames of these cuts now, as a job (a render that needs the current matte waits for
        it). then(ok, why) once it is over. project: another project than the open one (loaded for a render)."""
        from ..track import matte_refresh as MR
        from .matte_refresh import clear, pending
        from ..track import clothing as CL
        s = self.session
        prj = project if project is not None else s.project
        other = prj is not s.project
        cuts = [c for c in (prj.cut(i) for i in cut_ids)
                if c is not None and (pending(c) or CL.stale_of_cut(prj, c))] if prj is not None else []
        if not cuts:
            if then:
                then(True, "")
            return
        models = self.refresher.models
        fails: dict = {}
        if any(c is not s.cut for c in cuts):
            self.next_prep.cancel()             # a next cut opened ahead would keep its old widths

        def work(progress, cancelled):
            from ..core.frames import frames_for
            from ..track.analyze import CutTracker
            done = {}
            for i, c in enumerate(cuts):
                if cancelled():
                    break
                sub = (lambda f, m="", i=i: progress((i + f) / len(cuts), f"{c.name}: 인물 매트 갱신 {m}"))
                own = not other and s.cut is c and s.tracker is not None
                tr = s.tracker if own else None
                fr = None
                try:
                    if tr is None:
                        fr = frames_for(prj, c)
                        if fr is None:
                            fails[c.id] = "원본을 찾을 수 없음"
                            continue
                        tr = CutTracker(prj, c, fr)
                    if tr.store is None or not tr.has_seg():
                        done[c.id] = pending(c)        # nothing a SAM matte would follow here
                        continue
                    marks = pending(c)
                    if marks:
                        MR.refresh_now(tr, marks, sub, cancelled, models)
                    done[c.id] = marks
                    # clothing left out of use where the owners changed (it could not be made again with the
                    # matte, or the cut is from 1.22.2): made again there, so the render keeps the clothes
                    if CL.stale_ranges(tr):
                        try:
                            CL.refresh_stale(tr, sub, cancelled, models)
                        except Exception as e:
                            fails[c.id] = f"옷 영역: {e}"[:200]
                except Exception as e:
                    if type(e).__name__ == "Cancelled":
                        break
                    fails[c.id] = f"{e}"[:200]
                finally:
                    if fr is not None:
                        try:
                            fr.close()
                        except Exception:
                            pass
            return done

        def finish(done):
            for c in cuts:
                if c.id in done:
                    clear(c, done[c.id])
            if other:
                try:
                    prj.save()
                except Exception as e:
                    fails["_save"] = f"프로젝트 저장 실패: {e}"
            else:
                s.dirty = True
                s.touch()
                self.viewer._img_cache.clear()
                self.save()
                self._refresh_all()
            why = "; ".join(f"{prj.cut(k).name}: {v}" if prj.cut(k) is not None else v for k, v in fails.items())
            ok = not fails and all(c.id in done for c in cuts)
            if then:
                then(ok, why or ("취소했습니다" if not ok else ""))

        def failed_then(*_):
            if then:
                then(False, "인물 매트 갱신을 취소했거나 실패했습니다")
        self.run_job("인물 매트 갱신 중", work, finish)
        if self.job is not None and then:
            self.job.failed.connect(failed_then)

    def _needs_matte(self, prj, cut) -> bool:
        """Before a render: frames an edit marked are not matted again yet, or clothing kept in the cut is out of
        use where its owners changed."""
        from .matte_refresh import pending
        from ..track import clothing as CL
        return bool(cut is not None and (pending(cut) or CL.stale_of_cut(prj, cut)))

    def _ask_matte(self, names: list, adding: bool) -> str:
        """'refresh' / 'old' / 'cancel': a render of cuts whose matte is not caught up with the edits yet."""
        box = QMessageBox(self)
        box.setWindowTitle("인물 매트 갱신 전")
        shown = ", ".join(names[:4]) + (f" 외 {len(names) - 4}컷" if len(names) > 4 else "")
        box.setText(f"{shown}: 편집한 구간의 인물 매트(또는 그 구간의 옷 영역)가 아직 다시 만들어지지 않았습니다.\n"
                    "렌더는 저장된 매트를 씁니다. 매트를 갱신한 뒤 진행할까요?")
        b_ref = box.addButton("매트 갱신 후 " + ("추가" if adding else "렌더"), QMessageBox.AcceptRole)
        b_old = box.addButton("이전 매트로 " + ("추가" if adding else "렌더"), QMessageBox.DestructiveRole)
        box.addButton("취소", QMessageBox.RejectRole)
        box.setDefaultButton(b_ref)
        box.exec()
        c = box.clickedButton()
        return "refresh" if c is b_ref else "old" if c is b_old else "cancel"

    def relearn_clip(self):
        s = self.session
        if s.tracker is None or s.tracker.store is None:
            return
        tr = s.tracker
        self.push_undo()

        def work(progress, cancelled):
            tr.resolve_all(progress)
            return True

        def done(_):
            s.touch()
            self._refresh_all()
            self.toast("수정 내용을 반영해 전체 구간을 다시 계산했습니다.", kind="ok")

        self.run_job("수정 내용으로 다시 계산", work, done, cancellable=False)

    def run_qc(self):
        s = self.session
        if s.tracker is None or s.tracker.store is None:
            self.toast("먼저 분석을 실행하세요.", kind="warn")
            return
        cut = s.cut
        tr = s.tracker
        from ..qc.checks import run_qc

        def work(progress, cancelled):
            return run_qc(cut, tr, progress, cancelled)

        def done(issues):
            cut.qc = issues
            cut._qc_ran = True
            s.dirty = True
            self._refresh_all()
            n_err = sum(1 for q in issues if q.severity == "error" and q.status == "open")
            n = sum(1 for q in issues if q.status == "open")
            self.toast("문제가 없습니다." if not n else f"확인할 항목 {n}개 (오류 {n_err})", kind="ok" if not n else "warn")

        self.inspector.show_tab("qc")
        self.run_job("QC 검사 중", work, done)

    def register_benchmark(self):
        s = self.session
        if s.tracker is None or s.tracker.store is None:
            return
        if QMessageBox.question(self, "벤치마크 등록",
                                "이 컷의 현재 추적 결과를 '정답'으로 저장합니다. 관절을 충분히 확인한 컷만 등록하세요. 진행할까요?") != QMessageBox.Yes:
            return
        from ..learn import benchmark as BM
        self.save()

        def work(progress, cancelled):
            return BM.register(s.project, s.cut, s.tracker, progress)

        def done(bid):
            s.dirty = True
            self.save()
            self.toast("벤치마크로 등록했습니다. 누적 학습 화면에서 모델 평가에 쓰입니다.", kind="ok")

        self.run_job("벤치마크 저장", work, done)

    # ================================================================== frame / playback
    def set_frame(self, t: int):
        s = self.session
        if s.proxy is None:
            return
        t = int(np.clip(t, 0, s.proxy.n - 1))
        if t != self.frame and not self.play_timer.isActive():
            s.last_seek = time.time()
            if s.from_source() and getattr(s.proxy, "seeks", 0) >= SCRUB_SEEKS:
                self._maybe_bg_proxy()          # jumping around a cut read from its original: a proxy helps
        self.frame = t
        self.viewer.set_frame(t)
        self.timeline.set_frame(t)
        info = s.cut.info
        self.tc.setText(info.tc_at(t) if info.timecode else M.frames_to_tc(t, info.fps))
        a, b = s.cut.work_range
        self.fr_lbl.setText(f"  {t} / {s.proxy.n - 1}")
        if not self.play_timer.isActive():
            self.inspector.refresh("reshape")

    def step(self, d: int):
        if self.session.proxy is None:
            return
        self.stop_play()
        self.set_frame(self.frame + d)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.stop_play()
        elif self.session.proxy is not None:
            fps = self.session.cut.info.fps
            self.viewer.playing = True
            self.cache_worker.kick(self.frame)
            self._clock_start(fps)
            self.play_timer.start(max(1, int(round(1000 / max(1.0, float(fps or 24.0))))))
            self.play_btn.setIcon(I.icon("pause", 17))
            self._prep_next_cut()

    def _clock_start(self, fps) -> None:
        self._play_clock = (time.perf_counter(), 0, max(1.0, float(fps or 24.0)))

    def _clock_next(self) -> None:
        """The next frame on time: 23.976 fps is 41.7 ms a frame, a fixed whole-ms timer ran 2 % fast. A frame that
        came late does not make the next ones hurry (no burst to catch up, the clock starts again)."""
        t0, n, fps = self._play_clock
        n += 1
        now = time.perf_counter()
        due = t0 + (n + 1) / fps
        if due - now < -2.0 / fps:
            t0, n = now, 0
            due = now + 1.0 / fps
        self._play_clock = (t0, n, fps)
        if self.play_timer.isActive():
            self.play_timer.setInterval(max(1, int(round((due - now) * 1000))))

    # ---- playback into the next cut
    def _loop_toggled(self, on: bool):
        st = settings()
        st.play_loop = bool(on)
        st.save()
        if on:
            self.next_prep.cancel()
        elif self.play_timer.isActive():
            self._prep_next_cut()

    def _next_play_cut(self):
        from .cutlist import listed
        prj, cur = self.session.project, self.session.cut
        if prj is None or cur is None:
            return None
        order = [c for c in listed(prj) if c.tl.get("kind", "clip") == "clip"]
        ids = [c.id for c in order]
        if cur.id not in ids:
            return None
        i = ids.index(cur.id)
        return order[i + 1] if i + 1 < len(order) else None

    def _prep_next_cut(self):
        if self.loop_btn.isChecked():
            return
        nxt = self._next_play_cut()
        if nxt is not None:
            fps = max(1.0, float(nxt.info.fps or 24.0))
            self.next_prep.start(self.session.project, nxt, head=int(fps * 2))

    def _play_into_next(self) -> bool:
        """At the end of the cut with 구간 반복 off: the prepared next cut takes over and playback goes on.
        False: nothing to go on to (repeat this cut). None: the next cut is still being opened (hold the last frame)."""
        nxt = self._next_play_cut()
        if nxt is None or self.job is not None or self.batch.current == nxt.id:
            return False
        sh = self.next_prep.take(nxt.id)
        if sh is None:
            if self.next_prep.failed or self.next_prep.cut_id != nxt.id:
                if self.next_prep.cut_id != nxt.id:
                    self._prep_next_cut()
                    return None
                self.toast("다음 컷을 열지 못해 이 컷을 반복합니다", "warn")
                return False
            return None
        prj = self.session.project
        if self.session.dirty and not self.save():
            self.stop_play()                    # not into the next cut with this one unsaved
            return True
        self._switch_undo(nxt.id)
        self.session.adopt(sh)
        cut = self.session.cut
        self.viewer._img_cache.clear()
        self.viewer._last_res = None
        self.viewer._skin_last = None
        self.tlstrip.set_current(cut.id)
        self.cutlist.show_current(cut.id)
        self.activity.set_cut(prj.name, cut)
        self.selection = None
        self.timeline.selection = None
        self.viewer.selection = None
        self.timeline.view0, self.timeline.view1 = 0.0, 1.0
        targets = cut.targets()
        self.sel_pid = targets[0].id if targets else (cut.persons[0].id if cut.persons else None)
        self.viewer.message = ""
        self.frame = cut.work_range[0]
        fps = max(1.0, float(cut.info.fps or 24.0))
        self._clock_start(fps)
        self.play_timer.setInterval(max(1, int(round(1000 / fps))))
        self._refresh_all(skip_inspector=True)
        QTimer.singleShot(250, lambda: self.inspector.refresh())     # the inspector after the first frames show
        self.cache_worker.kick(self.frame)
        self._prep_next_cut()
        return True

    def stop_play(self):
        self.viewer.playing = False
        self.next_prep.cancel()
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setIcon(I.icon("play", 17))
            self.inspector.refresh("reshape")
            self._maybe_bg_proxy()              # playback measured how fast the original reads

    def _tick(self):
        a, b = self.session.cut.work_range
        nf = self.frame + 1
        if nf >= b:
            if not self.loop_btn.isChecked():
                if self._next_play_cut() is None:
                    self.stop_play()            # the last cut, 구간 반복 off: playback ends here
                    return
                r = self._play_into_next()
                if r:
                    return
                if r is None:
                    return                      # the next cut is still opening: stay on the last frame
            nf = a
        self.set_frame(nf)
        self._clock_next()

    def set_in_out(self, which: str):
        cut = self.session.cut
        if cut is None:
            return
        old = cut.work_range
        if which == "in":
            cut.in_frame = self.frame
            if cut.out_frame > 0 and cut.out_frame <= self.frame:
                cut.out_frame = -1
        else:
            cut.out_frame = self.frame + 1
            if cut.in_frame >= cut.out_frame:
                cut.in_frame = 0
        self.session.dirty = True
        self.timeline.update()
        self.inspector.refresh("output")
        msg = f"작업 구간 {cut.work_range[0]}~{cut.work_range[1] - 1}  (분석 전에 정하면 분석 시간이 줄어듭니다)"
        tr = self.session.tracker
        if cut.analyzed and tr is not None and tr.store is not None:
            from ..track import matte_refresh as MR
            try:
                to_matte, unseen = MR.extension(tr, old, cut.work_range)
            except Exception:
                to_matte, unseen = [], []
            self.refresher.mark_ranges(cut, to_matte)
            if unseen:
                n = sum(b - a for a, b in unseen)
                msg = (f"작업 구간 {cut.work_range[0]}~{cut.work_range[1] - 1}: 늘어난 {n}프레임에는 추적된 인물이 "
                       "없습니다. 이 구간을 분석하지 않았다면 '분석 시작'으로 다시 분석하세요")
        self.toast(msg)

    def next_issue(self, d: int):
        s = self.session
        cut = s.cut
        if cut is None or s.tracker is None or s.tracker.store is None:
            return
        st = s.tracker.store
        thr = cut.tracking.conf_threshold
        bad = np.zeros(st.n, bool)
        for p in cut.persons:
            if p.role == "ignore" or not st.has(p.id):
                continue
            i = st.index(p.id)
            bad |= st.present[i] & (st.conf[i] < thr)
        for q in cut.qc:
            if q.status == "open":
                bad[q.start:q.end + 1] = True
        # jump to the start of the next run
        starts = np.nonzero(bad & ~np.r_[False, bad[:-1]])[0]
        if not len(starts):
            self.toast("확인이 필요한 구간이 없습니다.", kind="ok")
            return
        if d > 0:
            nxt = starts[starts > self.frame]
            f = int(nxt[0]) if len(nxt) else int(starts[0])
        else:
            prv = starts[starts < self.frame]
            f = int(prv[-1]) if len(prv) else int(starts[-1])
        self.stop_play()
        self.set_frame(f)

    def jump_to(self, f: int, pid: int = -1):
        self.stop_play()
        if pid is not None and pid >= 0:
            self.sel_pid = pid
            self.viewer.sel_pid = pid
        self.set_frame(f)
        self._refresh_all(skip_inspector=True)

    # ================================================================== viewer options
    def set_tool(self, tool: str):
        point_tools = ("sculpt", "bodypts")
        side_tools = point_tools + ("bodymark",)        # tools started from the right panel (the edit tool stays lit)
        was_points = self.viewer.tool in side_tools
        self.tool = tool
        self.viewer.tool = tool
        for k, b in self.tool_btns.items():
            b.setChecked(k == tool or (k == "edit" and tool in side_tools))
        self.viewer.setCursor(Qt.CrossCursor if tool not in ("edit",) + point_tools else Qt.ArrowCursor)
        if tool in point_tools:
            self.ov["skel"].setChecked(True)
        self.viewer.update()
        if was_points or tool in side_tools:
            self.inspector.refresh("reshape")

    def _compare(self, mode: str):
        if (mode == "side") != (self.viewer.mode == "side"):
            self.viewer.fit_view()          # one picture <-> two panes: the layout changes, so fit again
        self.viewer.mode = mode
        self.viewer.ab_flip = False
        self.viewer.update()

    def _ab(self):
        if self.viewer.mode in ("result", "orig"):
            self.compare.set_current("orig" if self.compare.current == "result" else "result", emit=True)
        else:
            self.compare.set_current("result", emit=True)

    def _hover_pick(self, v):
        settings().hover_pick = bool(v)
        try:
            settings().save()
        except Exception:
            pass
        self.viewer.hover_pick = bool(v)
        if not v:
            self.viewer.clear_hover()
        self.toast("호버 인물 강조 " + ("켜짐" if v else "꺼짐 (클릭으로 선택은 그대로 됩니다)"))

    def _overlay(self, k, v):
        if k == "skel":
            self.viewer.show_skel = v
        elif k == "heat":
            self.viewer.show_heat = v
        elif k == "matte":
            self.viewer.show_matte = v
        elif k == "skin":
            self.viewer.show_skin = v
            if v and self.session.cut is not None and not any(
                    pp.skin_active() for pp in self.session.cut.params.values()):
                self.toast("피부결 보정이나 몸 피부 보정을 켜면 적용 영역이 표시됩니다. 눈·눈썹·입술은 항상 빠집니다.")
        self.viewer.update()

    # ================================================================== people
    @staticmethod
    def _first_listed(prj):
        from .cutlist import listed
        lst = listed(prj)
        return lst[0] if lst else None

    def switch_version(self, cut_id: str, v: int):
        from ..core import workstate as WS
        """Timeline strip: open another version (v1, v2, ...) of a cut."""
        prj = self.session.project
        c = prj.cut(cut_id) if prj else None
        if c is None or int(v) == int(c.version):
            return
        if not self.save():                     # the version being left keeps what was not written
            return
        WS.version_switch(c, v)
        self.session.dirty = True
        if self.session.cut is c:
            self.session.touch("all")
            self.viewer._img_cache.clear()
        self._after_status_change()
        self._refresh_all()
        self.cutlist.update_rows()
        self.toast(f"{c.name}: v{c.version}으로 바꿨습니다", "ok")

    def select_person(self, pid: int):
        """Shift+click adds the person to the selection: they get the selected person's values, and every
        change from then on goes to all selected people."""
        cut = self.session.cut
        sel = set(getattr(self, "sel_pids", set()) or set())
        if QApplication.keyboardModifiers() & Qt.ShiftModifier and self.sel_pid is not None and cut is not None:
            if pid in sel and pid != self.sel_pid:
                sel.discard(pid)
                self.sel_pids = sel
                self.inspector.refresh()
                return
            if pid != self.sel_pid:
                self.push_undo()
                src = cut.params_for(self.sel_pid).copy_static(self.frame)
                dst = cut.params_for(pid)
                for k in ("amounts", "strength", "skin_detail", "skin_scope", "grain_size", "blemish_size", "feather"):
                    setattr(dst, k, getattr(src, k) if k != "amounts" else {kk: c for kk, c in src.amounts.items()})
                dst.copy_body_from(src)
                q = cut.person(pid)
                if q is not None and q.role != "target":
                    q.role = "target"
                sel |= {self.sel_pid, pid}
                self.sel_pids = sel
                self.session.touch("all")
                self.viewer.update()
                self.inspector.refresh()
                self.toast(f"{len(sel)}명에게 같은 보정값을 적용합니다 (Shift+클릭으로 빼기)", "ok")
                return
        self.sel_pids = {pid}
        self.sel_pid = pid
        self.viewer.sel_pid = pid
        self.timeline.sel_pid = pid
        self.viewer.update()
        self.timeline.update()
        self.inspector.refresh()

    def set_role(self, role: str):
        cut = self.session.cut
        p = cut.person(self.sel_pid) if cut else None
        if p is None:
            return
        self.push_undo()
        p.role = role
        self.session.touch()
        self._refresh_all()

    def delete_person(self):
        s = self.session
        cut = s.cut
        p = cut.person(self.sel_pid) if cut else None
        if p is None or s.tracker is None:
            return
        if QMessageBox.question(self, "인물 삭제",
                                f"'{p.name}'을(를) 이 컷에서 지울까요?\n잘못 잡힌 사람이나 물체를 정리할 때 씁니다. "
                                "다시 분석해도 돌아오지 않습니다.") != QMessageBox.Yes:
            return
        self.undo_stack.clear()
        self.redo_stack.clear()
        pid = p.id
        s.tracker.delete_person(pid)
        rest = cut.targets() or cut.persons
        self.sel_pid = rest[0].id if rest else None
        self.viewer.sel_pid = self.timeline.sel_pid = self.sel_pid
        s.touch()
        self.save()
        self._refresh_all()
        self.toast(f"{p.name}을(를) 지웠습니다.")

    # ---- face sculpt (dragging dense face points)
    def set_sculpt_mode(self, on: bool):
        if on:
            st = self.session.tracker.store if self.session.tracker else None
            if st is None or not st.has_face:
                self.toast("얼굴 랜드마크가 없습니다. 추적 탭에서 다시 분석하세요.", kind="warn")
                self.inspector.refresh("reshape")
                return
            self.set_tool("sculpt")
            self.toast("얼굴 점을 끌어 모양을 만드세요. 오른쪽 클릭: 그 점 되돌리기. Esc: 끝내기")
        else:
            self.set_tool("edit")

    def clear_sculpt(self):
        pp = self._params()
        if pp is None or not pp.sculpt:
            return
        self.push_undo()
        pp.sculpt = {}
        self.session.touch("geo")
        self._refresh_all()

    def _sculpt_edit(self, idx: int, uv, committed: bool):
        pp = self._params()
        if pp is None:
            return
        if uv is None:
            pp.sculpt.pop(int(idx), None)
        else:
            pp.sculpt[int(idx)] = [float(uv[0]), float(uv[1])]
        if pp.amount("sculpt", self.frame) < 1e-3 and pp.sculpt:
            pp.amounts["sculpt"].set_at(self.frame, 100.0)
        self.session.interactive = not committed
        self.session.touch("geo")
        self.viewer.update()
        if committed:
            self.inspector.refresh("reshape")

    # ---- clothes kept as they are
    def set_cloth(self, mode: str):
        pp = self._params()
        s = self.session
        if pp is None or s.tracker is None or mode == getattr(pp, "cloth", "include"):
            return
        if mode != "include" and not self._cloth_licence_ok():
            self.inspector.refresh("reshape")
            return
        self.push_undo()
        pp.cloth = mode
        s.touch("geo")
        self.viewer._img_cache.clear()
        self.inspector.refresh("reshape")
        if mode != "include":
            self._ensure_cloth()

    def _cloth_licence_ok(self) -> bool:
        """The clothing model is SegFormer fine-tuned for clothes; SegFormer's licence (NVIDIA) allows
        non-commercial use only. Said once, before the model is first downloaded."""
        from ..core.settings import settings
        from ..track import clothing as CL
        if CL.cloth_model() != "parse_clothes":
            return True                    # SAM 3.1 garment words: shipped, commercial use allowed, nothing to ask
        st = settings()
        if st.cloth_model_ok:
            return True
        r = QMessageBox.question(
            self, "옷 영역 분석 모델",
            "옷 영역은 SegFormer-B2 옷 분할 모델(약 110MB, 처음 한 번 다운로드)로 찾습니다.\n\n"
            "이 모델의 바탕인 SegFormer는 NVIDIA 라이선스로 비상업 연구·평가 용도로만 쓸 수 있습니다. "
            "상업 작업 결과물에 쓰기 전에 사내에서 라이선스를 확인하세요.\n\n계속할까요?")
        if r != QMessageBox.Yes:
            return False
        st.cloth_model_ok = True
        st.save()
        return True

    def _ensure_cloth(self):
        """Parse the clothes of every person who keeps them, once per analysis (in the background)."""
        from ..track import clothing as CL
        s = self.session
        cut, tr = s.cut, s.tracker
        if tr is None or tr.store is None or self.job is not None:
            return
        need = [tr.store.index(p.id) for p in cut.persons
                if p.role == "target" and tr.store.has(p.id) and getattr(cut.params_for(p.id), "cloth", "include") != "include"
                and not CL.ready_for(tr, p.id)]
        if not need:
            if CL.stale_of_cut(s.project, cut):
                # parsed, but out of use where the person matte was made again since (1.22.3): only those frames
                def again(progress, cancelled):
                    return CL.refresh_stale(tr, progress, cancelled, self.refresher.models)

                def again_done(rs):
                    s.touch("geo")
                    self.viewer._img_cache.clear()
                    self._refresh_all()
                    if rs:
                        self.toast(f"인물 매트가 바뀐 {len(rs)}곳의 옷 영역을 다시 만들었습니다.", kind="ok")
                self.run_job("옷 영역 갱신 중", again, again_done)
            return
        # all people to be reshaped at once, so switching someone else later is instant
        pids = sorted(set(need) | {tr.store.index(p.id) for p in cut.targets() if tr.store.has(p.id)})

        def work(progress, cancelled):
            return CL.run(tr, pids, progress, cancelled)

        def done(ok):
            s.touch("geo")
            self.viewer._img_cache.clear()
            self._refresh_all()
            if ok:
                self.toast("옷 영역 분석을 마쳤습니다. 고른 옷은 몸 보정에서 그대로 남습니다.", kind="ok")

        self.run_job("옷 영역 분석 중", work, done)

    # ---- masks made again for this cut only (1.22.1)
    def recompute_face_masks(self):
        """The face and neck masks of the open cut, made again with the face model of the settings. Joints, people,
        retouch values and keys stay; the clothing masks stay. The new masks are used only once complete."""
        from ..track import facelayer as FL
        s = self.session
        tr = s.tracker
        if tr is None or tr.store is None or not getattr(tr.store, "has_face", False):
            self.toast("분석한 얼굴이 없는 컷입니다.", kind="warn")
            return
        if self.job is not None:
            self.toast("다른 작업이 진행 중입니다.", kind="warn")
            return

        def work(progress, cancelled):
            tr._facelayer_building = True
            try:
                return FL.build(tr, progress, cancelled) is not None
            finally:
                tr._facelayer_building = False

        def done(ok):
            tr.__dict__.pop("_steady_lru", None)
            s._mask_cache = {}
            s.touch("ret")
            self.viewer._img_cache.clear()
            self._refresh_all()
            if ok:
                inf = FL.info(tr)
                self.toast(f"얼굴·목 마스크를 다시 만들었습니다 ({inf['label']})", kind="ok")
            else:
                self.toast("다시 만들 얼굴이 없습니다 (작거나 가려진 얼굴만 있는 컷).", kind="warn")

        self.run_job("얼굴·목 마스크 다시 계산 중", work, done)

    def recompute_cloth_masks(self):
        """The clothing masks of the open cut, made again with the clothing model of the settings, for every person
        (the protection mode can change later without another run). Face and neck masks stay."""
        from ..track import clothing as CL
        s = self.session
        cut, tr = s.cut, s.tracker
        if tr is None or tr.store is None:
            return
        if self.job is not None:
            self.toast("다른 작업이 진행 중입니다.", kind="warn")
            return
        if not self._cloth_licence_ok():
            return
        pids = sorted({tr.store.index(p.id) for p in cut.persons if tr.store.has(p.id)})
        if not pids:
            return

        def work(progress, cancelled):
            return CL.run(tr, pids, progress, cancelled)

        def done(ok):
            tr.__dict__.pop("_cloth_state", None)
            s.touch("geo")
            self.viewer._img_cache.clear()
            self._refresh_all()
            if ok:
                inf = CL.info(tr)
                self.toast(f"의상 마스크를 다시 만들었습니다 ({inf['label']})", kind="ok")

        self.run_job("의상 마스크 다시 계산 중", work, done)

    # ---- head parts (hair / face) for precise head edits
    def ensure_parts(self, force: bool = False):
        """Find hair and face of every person with head edits (in the background)."""
        from ..track import parts as PT
        s = self.session
        cut, tr = s.cut, s.tracker
        if tr is None or tr.store is None or self.job is not None:
            return
        heads = [p for p in cut.targets() if tr.store.has(p.id)
                 and any(cut.params_for(p.id)._active(k) for k in ("head", "crown", "hair_volume"))]
        if self.sel_pid is not None and tr.store.has(self.sel_pid) and force:
            heads += [p for p in cut.persons if p.id == self.sel_pid and p not in heads]
        need = [p for p in heads if force or not PT.covers(tr, p.id)]
        if not need:
            return
        pids = sorted({tr.store.index(p.id) for p in heads})

        def work(progress, cancelled):
            return PT.run(tr, pids, progress, cancelled)

        def done(ok):
            s.touch("geo")
            self.viewer._img_cache.clear()
            self._refresh_all()
            if ok:
                self.toast("머리 부위를 인식했습니다. 헤어 볼륨은 머리카락에만 들어갑니다.", kind="ok")

        self.run_job("머리 부위 인식 중", work, done)

    # ---- presets
    def preset_menu(self, anchor):
        from PySide6.QtWidgets import QMenu
        from ..core import presets as PS
        cut = self.session.cut
        m = QMenu(self)
        items = PS.list_presets()
        if not items:
            a = m.addAction("저장된 프리셋이 없습니다. '현재 값 저장'으로 만드세요.")
            a.setEnabled(False)
        for d in items:
            name = d.get("name", "")
            sub = m.addMenu(name)
            sub.setToolTipsVisible(True)
            info = sub.addAction(PS.summary(d) or "빈 프리셋")
            info.setEnabled(False)
            sub.addSeparator()
            sub.addAction("이 인물에 적용", lambda d=d: self.preset_apply(d, "person"))
            if cut is not None and len(cut.targets()) > 1:
                sub.addAction("이 컷의 보정 인물 모두에 적용", lambda d=d: self.preset_apply(d, "cut"))
            sub.addSeparator()
            sub.addAction("삭제", lambda n=name: self.preset_delete(n))
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def preset_apply(self, d: dict, scope: str):
        from ..core import presets as PS
        cut = self.session.cut
        if cut is None:
            return
        if scope == "person":
            if self.sel_pid is None:
                return
            pids = [self.sel_pid]
        else:
            pids = [p.id for p in cut.targets()]
        self.push_undo()
        for pid in pids:
            PS.apply(cut.params_for(pid), d)
        keeps = [pid for pid in pids if getattr(cut.params_for(pid), "cloth", "include") != "include"]
        if keeps:
            if self._cloth_licence_ok():
                self._ensure_cloth()
            else:
                for pid in keeps:
                    cut.params_for(pid).cloth = "include"
        self.session.interactive = False
        self.session.touch("all")
        self._refresh_all()
        self.toast(f"프리셋 '{d.get('name', '')}'을(를) {'이 인물' if scope == 'person' else f'보정 인물 {len(pids)}명'}에 적용했습니다.",
                   kind="ok")
        self.ensure_skin_cloth(quiet=True)          # a preset with a body value on (when no other job was started)

    def preset_save_dialog(self):
        from ..core import presets as PS
        from .dialogs import PresetDialog
        pp = self._params()
        if pp is None:
            self.toast("먼저 인물을 고르세요.", kind="warn")
            return
        dlg = PresetDialog(self, pp, self.frame)
        if dlg.exec() != QDialog.Accepted:
            return
        name, groups, points = dlg.result_values()
        if not name:
            self.toast("프리셋 이름을 입력하세요.", kind="warn")
            return
        if not groups and not points:
            self.toast("담을 그룹을 하나 이상 고르세요.", kind="warn")
            return
        if any(x.get("name") == name for x in PS.list_presets()):
            if QMessageBox.question(self, "프리셋 저장", f"'{name}' 프리셋이 이미 있습니다. 덮어쓸까요?") != QMessageBox.Yes:
                return
        try:
            PS.save(PS.make(pp, self.frame, name, groups, points))
        except Exception as e:
            self.show_error("프리셋 저장 오류", str(e))
            return
        self.toast(f"프리셋 '{name}'을(를) 저장했습니다.", kind="ok")

    def preset_delete(self, name: str):
        from ..core import presets as PS
        if QMessageBox.question(self, "프리셋 삭제", f"'{name}' 프리셋을 지울까요?") != QMessageBox.Yes:
            return
        PS.delete(name)
        self.toast(f"프리셋 '{name}'을(를) 지웠습니다.")

    # ---- body points (dragging the measured body outline)
    def set_body_mode(self, on: bool):
        if on:
            st = self.session.tracker.store if self.session.tracker else None
            if st is None or not getattr(st, "has_contour", False):
                self.toast("몸 윤곽 점이 아직 없습니다. 잠시 후 다시 누르거나 추적 탭에서 다시 분석하세요.", kind="warn")
                self.inspector.refresh("reshape")
                return
            self.set_tool("bodypts")
            self.toast("몸 윤곽 점을 끌어 모양을 만드세요. 주변 점도 부드럽게 따라옵니다 (Shift: 한 점만). "
                       "오른쪽 클릭: 그 점 되돌리기. Esc: 끝내기")
        else:
            self.set_tool("edit")

    def clear_body_sculpt(self):
        pp = self._params()
        if pp is None or not pp.body_sculpt:
            return
        self.push_undo()
        pp.body_sculpt = {}
        pp.body_sides = "person"
        self.session.touch("geo")
        self._refresh_all()

    def _body_edit(self, changes, committed: bool):
        pp = self._params()
        if pp is None:
            return
        for key, v in changes.items():
            if v is None:
                pp.body_sculpt.pop(key, None)
            else:
                pp.body_sculpt[key] = float(v)
        if not pp.body_sculpt:
            pp.body_sides = "person"
        if pp.amount("body_sculpt", self.frame) < 1e-3 and pp.body_sculpt:
            pp.amounts["body_sculpt"].set_at(self.frame, 100.0)
        self.session.interactive = not committed
        self.session.touch("geo")
        self.viewer.update()
        if committed:
            self.session.dirty = True
            self.inspector.refresh("reshape")

    def rename_person(self, name: str):
        cut = self.session.cut
        p = cut.person(self.sel_pid) if cut else None
        if p is None or not name.strip() or name == p.name:
            return
        p.name = name.strip()
        self.session.dirty = True
        self._refresh_all()

    def merge_persons_dialog(self):
        s = self.session
        if s.cut is None or len(s.cut.persons) < 2:
            self.toast("합칠 인물이 두 명 이상이어야 합니다.", kind="warn")
            return
        d = MergeDialog(self, s.cut)
        if d.exec():
            a, b = d.a.currentData(), d.b.currentData()
            if a == b:
                return
            tr = s.tracker
            cut = s.cut

            def work(progress, cancelled):
                from ..track import matte_refresh as MR
                before = MR.prompt_state(tr)
                tr.merge_persons(a, b)
                return MR.changed(tr, before)

            def done(marks):
                self.refresher.mark_ranges(cut, marks)
                self.sel_pid = a
                self.undo_stack.clear()
                self.redo_stack.clear()
                s.touch()
                self.save()
                self._refresh_all()

            self.run_job("인물 합치는 중", work, done, cancellable=False)

    def identity_dialog(self):
        from .dialogs import IdentityDialog
        s = self.session
        if s.cut is None or s.tracker is None or s.tracker.store is None or self.sel_pid is None:
            self.toast("먼저 분석한 컷에서 인물을 고르세요.", kind="warn")
            return
        pid = self.sel_pid
        d = IdentityDialog(self, s.cut, pid, int(self.frame), self.selection)
        if not d.exec():
            return
        mode, other, t0, t1 = d.values()
        if mode == "swap" and other is None:
            return
        tr = s.tracker
        cut = s.cut
        result = {}

        def work(progress, cancelled):
            from ..track import matte_refresh as MR
            before = MR.prompt_state(tr)
            if mode == "swap":
                tr.swap_persons(pid, other, t0, t1)
            else:
                result["new"] = tr.split_person(pid, t0, t1)
            # the owners move with the ids, nobody moves: usually nothing to matte again
            return MR.changed(tr, before, (t0, t1))

        def done(marks):
            self.refresher.mark_ranges(cut, marks)
            # tracking data is not in the undo history: older snapshots would no longer match the people
            self.undo_stack.clear()
            self.redo_stack.clear()
            s.touch()
            self.save()
            self._refresh_all()
            if mode == "swap":
                nm = s.cut.person(other).name if s.cut.person(other) else other
                self.toast(f"{t0}–{t1 - 1} 프레임에서 두 인물을 맞바꿨습니다 ({nm}). 되돌리려면 같은 구간을 한 번 더 맞바꾸세요.", kind="ok")
            else:
                self.toast(f"{t0}–{t1 - 1} 프레임을 새 인물로 떼어 냈습니다. 같은 사람이면 '인물 합치기'로 이어 붙이세요.",
                           kind="ok")

        self.run_job("인물 번호 바로잡는 중", work, done, cancellable=False)

    # ================================================================== reshape params

    def _params(self):
        cut = self.session.cut
        if cut is None or self.sel_pid is None:
            return None
        return cut.params_for(self.sel_pid)

    def _others(self) -> list:
        """Params of the other people selected with Shift+click (they follow every change)."""
        cut = self.session.cut
        sel = getattr(self, "sel_pids", set()) or set()
        if cut is None:
            return []
        return [cut.params_for(p) for p in sel if p != self.sel_pid and cut.person(p) is not None]

    @staticmethod
    def _kind_of(part: str) -> str:
        from ..core.params import PARAM
        d = PARAM.get(part)
        return "ret" if d is not None and d.kind in ("skin", "color") else "geo"

    def param_changed(self, part: str, v: float, committed: bool):
        pp = self._params()
        if pp is None:
            return
        if committed:
            self.push_undo(coalesce=f"param:{self.sel_pid}:{part}")
        c = pp.strength if part == "__strength__" else pp.amounts[part]
        c.set_at(self.frame, v)
        for op in self._others():
            (op.strength if part == "__strength__" else op.amounts[part]).set_at(self.frame, v)
        if part != "__strength__" and self._kind_of(part) == "ret" and part not in BODY_SKIN_KEYS:
            self._skin_hand_set(pp)
        self.session.interactive = not committed
        self.session.touch(self._kind_of(part))
        self.viewer.update()
        if committed:
            self.timeline.update()
            self.inspector.refresh("reshape")

    def toggle_param_key(self, part: str):
        pp = self._params()
        if pp is None:
            return
        self.push_undo()
        c = pp.strength if part == "__strength__" else pp.amounts[part]
        if c.has_key(self.frame):
            c.remove_key(self.frame)
        else:
            c.add_key(self.frame)
        if part != "__strength__" and self._kind_of(part) == "ret" and part not in BODY_SKIN_KEYS:
            self._skin_hand_set(pp)
        self.session.touch(self._kind_of(part))
        self._refresh_all()

    def width_changed(self, part: str, v: float, live: bool = False):
        pp = self._params()
        if pp is None:
            return
        if not live:
            self.push_undo(coalesce=f"width:{self.sel_pid}:{part}")
        pp.widths[part] = float(v)
        self.session.interactive = live
        self.session.touch("geo")
        self.viewer.update()

    def set_feather(self, v: float, committed: bool):
        pp = self._params()
        if pp is None:
            return
        if committed:
            self.push_undo(coalesce=f"feather:{self.sel_pid}")
        pp.feather = float(v)
        for op in self._others():
            op.feather = float(v)
        self.session.interactive = not committed
        self.session.touch("all" if committed else "geo")
        self.viewer.update()

    def set_skin(self, what: str, v, committed: bool):
        pp = self._params()
        if pp is None:
            return
        if committed:
            self.push_undo(coalesce=f"skin:{self.sel_pid}:{what}")
        self._skin_hand_set(pp)
        for q in [pp] + self._others():
            if what == "detail":
                q.skin_detail = float(v)
            elif what == "grain_size":
                q.grain_size = float(v)
            elif what == "blemish_size":
                q.blemish_size = float(v)
            elif what == "scope":
                q.skin_scope = v
        self.session.interactive = not committed
        self.session.touch("ret")
        self.viewer.update()
        if committed:
            self.inspector.refresh("reshape")

    # ---- body skin (1.23): 몸 피부 보정, 몸 잡티 제거
    def set_body_on(self, key: str, on: bool):
        """Switch a body skin effect of the selected people on or off (off keeps its values to come back to)."""
        pp = self._params()
        if pp is None:
            self.inspector.refresh("reshape")
            return
        self.push_undo()
        for q in [pp] + self._others():
            q.set_body_on(key, bool(on))
        if not on and key == "body_blemish" and self.viewer.tool == "bodymark":
            self.set_tool("edit")
        self.session.interactive = False
        self.session.touch("ret")
        self.viewer.update()
        self.timeline.update()
        self.inspector.refresh("reshape")
        if on:
            self.ensure_skin_cloth(quiet=True)

    def ensure_skin_cloth(self, quiet: bool = False):
        """The open cut looked at for clothing the body parser takes for skin (track.skincloth), for the people
        with a body value on, in the background job panel. quiet: nothing is said when there is nothing to do or
        another job is running (the export looks at the cut itself when this has not been done)."""
        from ..track import skincloth as SC
        s = self.session
        cut, tr = s.cut, s.tracker
        if tr is None or tr.store is None or cut is None:
            return
        if self.job is not None:
            if not quiet:
                self.toast("다른 작업이 진행 중입니다.", kind="warn")
            return
        try:
            need = SC.needed(tr, cut)
        except Exception:
            need = []
        if not need:
            if not quiet:
                self.toast("피부색 옷 확인이 이미 되어 있습니다.", kind="ok")
            return
        miss = SC.models_missing()
        if miss:
            if not quiet or not getattr(self, "_skin_cloth_told", False):
                self._skin_cloth_told = True
                self.toast("피부색 옷 확인에는 SAM 3.1 모델이 필요합니다 (설정 > 모델 관리). 없으면 몸 파서의 구분만으로 "
                           "보정합니다.", kind="warn")
            return

        def work(progress, cancelled):
            return SC.run(tr, need, progress, cancelled, getattr(getattr(self, "refresher", None), "models", None))

        def done(ok):
            s.touch("ret")
            self.viewer._img_cache.clear()
            self._refresh_all()
            if ok:
                inf = SC.info(tr)
                n = int(inf.get("frames", 0)) if inf.get("ranges") else 0
                self.toast("피부색 옷 확인 끝: " + (f"{len(inf['ranges'])}개 구간에서 옷·머리카락을 몸 피부에서 제외합니다."
                                                   if n else "피부로 잘못 볼 옷·머리카락이 없습니다."), kind="ok")
        self.run_job("피부색 옷 확인 중", work, done)

    def set_body(self, what: str, v, committed: bool):
        """The body skin settings that are not sliders with keys: detail (결 보존), blemish_size (최대 잡티 크기),
        scope (전신 or regions)."""
        pp = self._params()
        if pp is None:
            return
        if committed:
            self.push_undo(coalesce=f"body:{self.sel_pid}:{what}")
        from ..core.project import norm_body_scope
        for q in [pp] + self._others():
            if what == "detail":
                q.body_detail = float(v)
            elif what == "blemish_size":
                q.body_blemish_size = float(v)
            elif what == "scope":
                q.body_scope = norm_body_scope(v)
        self.session.interactive = not committed
        self.session.touch("ret")
        self.viewer.update()
        if committed:
            self.inspector.refresh("reshape")

    def set_body_mark_mode(self, kind: str):
        """The body spot marking tool: 'remove' (this spot goes) or 'exclude' (nothing is removed here); '' ends it."""
        if not kind:
            if self.viewer.tool == "bodymark":
                self.set_tool("edit")
            return
        pp = self._params()
        tr = self.session.tracker
        if pp is None or tr is None or tr.store is None:
            self.toast("먼저 분석된 컷에서 인물을 선택하세요.", kind="warn")
            self.inspector.refresh("reshape")
            return
        if not pp.body_on("body_blemish"):
            self.toast("몸 잡티 제거를 먼저 켜세요.", kind="warn")
            self.inspector.refresh("reshape")
            return
        self.viewer.mark_kind = kind
        self.set_tool("bodymark")
        if not self.ov["skin"].isChecked():
            self.ov["skin"].setChecked(True)             # the marks and what the detection found are shown on the area
        self.toast("잡티를 찍거나 끌어서 표시하세요 (지금: " + ("제거" if kind == "remove" else "자동 제거 제외")
                   + "). 표시는 몸을 따라 컷 전체에 적용됩니다. 오른쪽 클릭: 표시 지우기 · Esc: 끝내기")

    def set_body_mark_radius(self, r: float):
        self.viewer.mark_r = float(max(2.0, min(60.0, r)))
        self.viewer.update()

    def _body_mark(self, pts, r: float):
        """The viewer drew a mark for the selected person on this frame (points and radius in source px)."""
        pp = self._params()
        tr = self.session.tracker
        if pp is None or tr is None or tr.store is None or not tr.store.has(self.sel_pid) or not pts:
            return
        import uuid
        from ..retouch import bodyskin as BS
        self.push_undo()
        kind = getattr(self.viewer, "mark_kind", "remove")
        m = BS.anchor_mark(tr.store, tr.store.index(self.sel_pid), int(self.frame),
                           dict(id=uuid.uuid4().hex[:8], kind=kind, t=int(self.frame),
                                pts=[[float(x), float(y)] for x, y in pts], r=float(r)))
        pp.body_marks = list(pp.body_marks) + [m]
        self.session.interactive = False
        self.session.touch("ret")
        self.viewer.update()
        self.inspector.refresh("reshape")
        if not m.get("part"):
            self.toast("가까운 몸 부위를 찾지 못해 이 표시는 화면의 같은 자리에 고정됩니다.", kind="warn")

    def _body_mark_delete(self, mark_id: str):
        pp = self._params()
        if pp is None or not any(m.get("id") == mark_id for m in pp.body_marks):
            return
        self.push_undo()
        pp.body_marks = [m for m in pp.body_marks if m.get("id") != mark_id]
        self.session.touch("ret")
        self.viewer.update()
        self.inspector.refresh("reshape")

    def clear_body_marks(self):
        pp = self._params()
        if pp is None or not pp.body_marks:
            return
        self.push_undo()
        pp.body_marks = []
        self.session.touch("ret")
        self.viewer.update()
        self.inspector.refresh("reshape")

    def _curves_for(self, key: str):
        pp = self._params()
        if pp is None:
            return []
        if key == "__strength__":
            return [("__strength__", pp.strength)]
        if key == "*":
            return [("__strength__", pp.strength)] + list(pp.amounts.items())
        return [(key, pp.amounts[key])] if key in pp.amounts else []

    def _key_moved(self, key: str, f0: int, f1: int):
        curves = [(k, c) for k, c in self._curves_for(key) if f0 in c.keys]
        if not curves:
            return
        self.push_undo()
        kinds = set()
        for k, c in curves:
            c.keys[int(f1)] = c.keys.pop(int(f0))
            kinds.add(self._kind_of(k))
        for k in kinds:
            self.session.touch(k)
        self.toast(f"키프레임 {f0} → {f1}")
        self._refresh_all()

    def _key_deleted(self, key: str, f: int):
        curves = [(k, c) for k, c in self._curves_for(key) if f in c.keys]
        if not curves:
            return
        self.push_undo()
        for k, c in curves:
            c.remove_key(f)
            self.session.touch(self._kind_of(k))
        self._refresh_all()

    def reset_params(self, part: str = "all"):
        """part: skin (skin and colour retouch), body (every shape change) or all."""
        pp = self._params()
        if pp is None:
            return
        self.push_undo()
        from ..core.project import PartParams
        from ..core.params import RETOUCH_KEYS
        fresh = PartParams()
        if part == "skin":
            for k in RETOUCH_KEYS:
                pp.amounts[k] = fresh.amounts[k]
            pp.skin_detail, pp.skin_scope = fresh.skin_detail, fresh.skin_scope
            pp.grain_size, pp.blemish_size = fresh.grain_size, fresh.blemish_size
            pp.skin_basic = {}
            pp.copy_body_from(fresh, marks=True)          # body skin: off, as a new person has it; the hand marks go too
            self.session.touch("ret")
        elif part == "body":
            for k in RETOUCH_KEYS:
                fresh.amounts[k] = pp.amounts[k]
            fresh.skin_detail, fresh.skin_scope = pp.skin_detail, pp.skin_scope
            fresh.grain_size, fresh.blemish_size = pp.grain_size, pp.blemish_size
            fresh.skin_basic = pp.skin_basic
            fresh.copy_body_from(pp, marks=True)
            self.session.cut.params[self.sel_pid] = fresh
            self.session.touch("geo")
        else:
            self.session.cut.params[self.sel_pid] = fresh
            self.session.touch()
        self._refresh_all()

    # ---- skin: basic adjustment (a preset of the detail sliders)
    def _skin_hand_set(self, pp):
        """A skin or colour value changed by hand: the basic adjustment no longer describes the sliders."""
        for q in [pp] + self._others():
            if getattr(q, "skin_basic", None):
                q.skin_basic = {}

    def apply_skin_basic(self, level: str = "", switch=None, toggle=None):
        """Sets the detail sliders from the basic adjustment (level light | std | strong, switch (name, remove), toggle (part, on): the eye
        whites, the knees, the elbows). With no basic adjustment chosen yet, a switch starts from 표준 with every switch on 보호."""
        pp = self._params()
        if pp is None:
            self.inspector.refresh("reshape")
            return
        from ..core.params import skin_basic_values
        from ..core.project import Curve
        cur = dict(pp.skin_basic) if pp.skin_basic else {"level": "std", "remove": []}
        started = not pp.skin_basic
        if level:
            cur["level"] = level
        rm = set(cur.get("remove", []))
        if switch is not None:
            (rm.add if switch[1] else rm.discard)(switch[0])
        cur["remove"] = sorted(rm)
        if toggle is not None:
            ons = set(cur.get("on") or [])
            (ons.add if toggle[1] else ons.discard)(toggle[0])
            cur["on"] = sorted(ons)
        vals = skin_basic_values(cur["level"], cur["remove"], cur.get("on"))
        targets = [pp] + self._others()
        if any(q.amounts[k].keys for q in targets for k in vals):
            m = QMessageBox(self)
            m.setWindowTitle("기본 조정")
            m.setIcon(QMessageBox.Question)
            m.setText("키프레임이 있는 피부 보정 항목이 있습니다.")
            m.setInformativeText("키프레임을 지우고 컷 전체에 같은 값으로 적용할까요?")
            b_ok = m.addButton("키프레임 지우고 적용", QMessageBox.AcceptRole)
            m.addButton("취소", QMessageBox.RejectRole)
            m.setDefaultButton(b_ok)
            m.exec()
            if m.clickedButton() is not b_ok:
                self.inspector.refresh("reshape")
                return
        self.push_undo()
        for q in targets:
            for k, v in vals.items():
                q.amounts[k] = Curve(base=float(v))
            q.skin_basic = {"level": cur["level"], "remove": list(cur["remove"])}
            if cur.get("on") is not None:
                q.skin_basic["on"] = list(cur["on"])
        self.session.touch("ret")
        self.viewer.update()
        self.timeline.update()
        self.inspector.refresh("reshape")
        if started and (switch is not None or toggle is not None) and not level:
            self.toast("강도는 표준으로 적용했습니다", "ok")

    def copy_settings_dialog(self):
        s = self.session
        if s.cut is None or len(s.project.cuts) < 2:
            self.toast("복사할 다른 컷이 없습니다.", kind="warn")
            return
        d = CopyDialog(self, s.project, s.cut, self.frame)
        if d.exec():
            n = d.apply()
            s.dirty = True
            self.save()
            self.cutlist.update_rows()
            self.toast(f"{n}개 컷에 설정을 복사했습니다.", kind="ok")

    # ================================================================== tracking settings & ranges
    def set_tracking(self, attr: str, v):
        cut = self.session.cut
        if cut is None:
            return
        ts = cut.tracking
        if attr == "model_m":
            if v and "pose_rtmpose_m" not in ts.models:
                ts.models.append("pose_rtmpose_m")
            if not v and "pose_rtmpose_m" in ts.models:
                ts.models.remove("pose_rtmpose_m")
            self.toast("모델 구성은 다음 분석부터 적용됩니다.")
        else:
            setattr(ts, attr, v)
        self.session.dirty = True
        if attr in ("smoothing", "conf_threshold", "adapt", "refiner") and self.session.tracker and self.session.tracker.store is not None:
            self.relearn_clip()
        elif attr in ("face", "second_pass", "point_tracker", "detail", "flip_tta", "reid", "members", "video_track"):
            self.toast("다음 분석부터 적용됩니다.")
        elif attr == "seg":
            self.toast("다음 분석부터 적용됩니다. 지금 컷에 바로 쓰려면 고급 > '인물 매트 강제 갱신'을 누르세요."
                       if v == "sam2" else "다음 분석부터 RVM 매트를 씁니다.")
        self.inspector.refresh("track")

    def _range_selected(self, a: int, b: int):
        self.selection = (a, b)
        self.viewer.selection = (a, b)
        self.inspector.refresh("track")

    def _mark_range(self, kind: str):
        cut = self.session.cut
        if cut is None or self.selection is None or self.sel_pid is None:
            self.toast("타임라인에서 Shift+드래그로 구간을 고르고, 인물을 선택하세요.", kind="warn")
            return
        self.push_undo()
        a, b = self.selection
        target = cut.confirmed if kind == "confirm" else cut.locked
        rs = target.setdefault(self.sel_pid, [])
        rs.append([a, b])
        rs.sort()
        merged = []
        for r in rs:
            if merged and r[0] <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], r[1])
            else:
                merged.append(list(r))
        target[self.sel_pid] = merged
        self.session.dirty = True
        self.timeline.update()
        self.toast(("확정" if kind == "confirm" else "잠금") + f" {a}~{b}")

    def confirm_selection(self):
        self._mark_range("confirm")

    def lock_selection(self):
        self._mark_range("lock")

    def clear_selection_marks(self):
        cut = self.session.cut
        if cut is None or self.selection is None or self.sel_pid is None:
            return
        self.push_undo()
        a, b = self.selection
        for target in (cut.confirmed, cut.locked):
            out = []
            for r0, r1 in target.get(self.sel_pid, []):
                if r1 < a or r0 > b:
                    out.append([r0, r1])
                else:
                    if r0 < a:
                        out.append([r0, a - 1])
                    if r1 > b:
                        out.append([b + 1, r1])
            target[self.sel_pid] = out
        self.session.dirty = True
        self.timeline.update()

    # ================================================================== joint editing
    def _joint_drag(self, pid, j, x, y):
        if self.session.tracker is None:
            return
        self.session.set_joint_live(pid, j, self.frame, (x, y))
        self.viewer.update()

    def _joint_commit(self, pid, j, x, y):
        s = self.session
        self.push_undo()
        t = self.frame
        if s.tracker is not None and s.tracker.hidden_at(pid, j, t):
            s.tracker.set_hidden(pid, [j], t, t, False)     # placing a joint means it is visible here
        s.commit_joint(pid, j, self.frame, (x, y))
        s.tracker.store.flags[s.tracker.store.index(pid), t, j] |= S.FLAG_USER
        self.edit_worker.submit(("resolve", pid), resolve_task(s, pid, t))
        self.timeline.update()
        self.inspector.refresh("track")

    def _joint_delete(self, pid, j):
        s = self.session
        self.push_undo()
        if s.delete_joint_key(pid, j, self.frame):
            t = self.frame
            self.edit_worker.submit(("resolve", pid), resolve_task(s, pid, t))
            self.toast(f"{S.JOINT_NAMES_KO[j]} 키를 지웠습니다.")

    def _person_action(self, action: str, pid: int):
        """Right click on a person in the preview: rename, 보정 대상 on / off, delete."""
        cut = self.session.cut
        q = cut.person(pid) if cut else None
        if q is None:
            return
        self.select_person(pid)
        if action == "role":
            self.set_role("protect" if q.role == "target" else "target")
        elif action == "delete":
            self.sel_pids = set()
            self.delete_person()
        elif action == "rename":
            prj = self.session.project
            mid = q.member if (q.member and prj is not None and q.member in prj.members) else ""
            ncuts = len({c for c, _ in prj.members[mid]["people"]}) if mid else 1
            dlg = QDialog(self)
            dlg.setWindowTitle("이름 변경")
            v = QVBoxLayout(dlg)
            v.setContentsMargins(18, 16, 18, 14)
            ed = QLineEdit(q.name)
            ed.selectAll()
            v.addWidget(QLabel("이름"))
            v.addWidget(ed)
            from PySide6.QtWidgets import QRadioButton
            r_all = QRadioButton(f"같은 사람으로 연결된 모든 컷 ({ncuts}컷)")
            r_one = QRadioButton("이 컷만")
            (r_all if mid and ncuts > 1 else r_one).setChecked(True)
            r_all.setEnabled(bool(mid) and ncuts > 1)
            if not mid:
                r_all.setToolTip("다른 컷의 같은 사람과 아직 연결되지 않았습니다 (인물 버튼에서 연결)")
            v.addWidget(r_all)
            v.addWidget(r_one)
            h = QHBoxLayout()
            h.addStretch(1)
            b_no = QPushButton("취소")
            b_ok = QPushButton("변경")
            b_ok.setDefault(True)
            b_no.clicked.connect(dlg.reject)
            b_ok.clicked.connect(dlg.accept)
            h.addWidget(b_no)
            h.addWidget(b_ok)
            v.addLayout(h)
            if dlg.exec() != QDialog.Accepted or not ed.text().strip():
                return
            name = ed.text().strip()
            if r_all.isChecked() and mid:
                from ..track import xlink as XL
                XL.rename(prj, mid, name)
                self.session.dirty = True
                self.cutlist.refresh(cut.id)
                self.tlstrip.refresh()
                self._refresh_all()
            else:
                self.rename_person(name)
        self.viewer.update()

    def _viewer_context(self, action, pid, j):
        s = self.session
        if action.startswith("hide") or action == "unhide":
            self._hide_joint(action, pid, j)
            return
        if action == "revive":
            g = s.tracker.ghost(pid, self.frame) if s.tracker else {}
            if j in g:
                self._joint_commit(pid, j, float(g[j][0]), float(g[j][1]))
            return
        if action == "clear_joint":
            self.push_undo()
            frames = list(s.cut.joint_keys.get(pid, {}).get(j, {}).keys())
            s.cut.joint_keys.get(pid, {}).pop(j, None)
            s.touch()
            for f in frames[:1]:
                self.edit_worker.submit(("resolve", pid), resolve_task(s, pid, every=True))

    def _hide_joint(self, action: str, pid: int, j: int):
        s = self.session
        tr = s.tracker
        if tr is None or tr.store is None:
            return
        t = self.frame
        joints = [j] + (S.DESCENDANTS.get(j, []) if action.endswith("_chain") else [])
        if action == "unhide":
            # give the whole user range around this frame back to the automatic decision
            rngs = [r for r in s.cut.hidden.get(pid, {}).get(j, []) if r[0] <= t <= r[1]]
            if not rngs:
                return
            a, b = rngs[0]
            hide = False
        elif action.startswith("hide_sel"):
            if self.selection is None:
                self.toast("타임라인에서 Shift+드래그로 구간을 먼저 고르세요.", kind="warn")
                return
            a, b = self.selection
            hide = True
        else:
            a = b = t
            hide = True
        self.push_undo()
        tr.set_hidden(pid, joints, a, b, hide)
        names = "·".join(S.JOINT_NAMES_KO[x] for x in joints)
        rng = f"{a}" if a == b else f"{a}~{b}"
        self.toast(f"{names}: {rng} 프레임 " + ("안 보이는 관절로 지정" if hide else "자동 판단으로 되돌림"))
        s.touch()
        self.edit_worker.submit(("resolve", pid), resolve_task(s, pid, a, frame_end=b))
        self.viewer.update()

    def _edit_done(self, tag, res):
        kind = tag[0] if isinstance(tag, tuple) and tag else None
        if kind in ("resolve", "resolve_h") and isinstance(res, dict) and res.get("marks"):
            prj = self.session.project
            cut = prj.cut(res["cut"]) if prj is not None and res.get("cut") else None
            if cut is not None:
                self.refresher.mark_ranges(cut, res["marks"])
        elif kind == "matte":
            self.refresher.committed(res)
            if not res or res.get("error"):
                return                      # nothing was written (dropped, or taken back): nothing to redo
            self.viewer._img_cache.clear()
        elif kind == "matte_check":
            # only a comparison (the tracking with what the matte was made from): nothing a frame is made from
            # changed, so nothing is redone
            self.refresher.checked(res)
            return
        self.session.touch()
        self.viewer.update()
        self.timeline.update()
        self.cutlist.update_rows()

    def _edit_busy(self, on: bool):
        self.edit_busy.setText("다시 계산 중…" if on else "")

    # ================================================================== QC / output
    def qc_changed(self):
        self.session.dirty = True
        self._refresh_all()

    def set_output(self, attr, v):
        cut = self.session.cut
        if cut is None:
            return
        setattr(cut.output, attr, v)
        self.session.dirty = True

    def browse_output_dir(self):
        cut = self.session.cut
        if cut is None:
            return
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", cut.output.out_dir or str(Path(self.session.project.path).parent))
        if d:
            cut.output.out_dir = d
            s = settings()
            s.output_dir = d
            s.save()
            self.session.dirty = True
            self.inspector.refresh("output")

    def set_input_cs(self, cs: str):
        """Chosen by hand in the 출력 tab: for the open cut, and for every other cut selected in the cut list."""
        cut = self.session.cut
        if cut is None:
            return
        prj = self.session.project

        def logish(x):
            return "log" in (x or "").lower()

        # tracking and mattes were made on the picture as it was shown then: from a log picture shown flat (or a
        # normal one shown as log) they are less sure. Worth analysing again when log/not log changes.
        def seen_as(c):
            return c.cs_analysed if c.cs_analysed is not None else c.input_cs

        stale = [c for c in (prj.cut(i) for i in self.cutlist.selected_ids()) if c is not None and c is not cut
                 and c.analyzed and logish(seen_as(c)) != logish(cs)]
        others = [c for c in (prj.cut(i) for i in self.cutlist.selected_ids()) if c is not None and c is not cut]
        for c in others:
            c.input_cs, c.cs_auto = cs, ""        # their proxies follow when they are opened next
            self.thumbcache.refresh(c.id, front=False)
        if others:
            msg = f"{len(others) + 1}컷의 색공간을 {manager().short_name(cs)}(으)로 바꿨습니다"
            if stale:
                msg += f". 이미 분석된 {len(stale)}컷은 다시 분석하면 추적·매트가 더 정확합니다"
            self.toast(msg, "ok")
        self.cutlist.update_rows()
        if cs == (cut.input_cs or ""):
            cut.cs_auto = ""
            self.session.project.save()
            self.inspector.refresh("output")
            return
        was = seen_as(cut)
        again = cut.analyzed and logish(was) != logish(cs)
        cut.input_cs, cut.cs_auto = cs, ""
        self.session.project.save()
        self._reanalyse_after_proxy = None
        if again and self.job is None:
            r = QMessageBox.question(
                self, "색공간 변경",
                f"이 컷은 {'log' if logish(was) else 'log가 아닌'} 화면으로 분석되어 추적·인물 매트가 덜 정확할 수 있습니다.\n"
                "바뀐 색공간으로 다시 분석할까요? (보정 값은 그대로 남습니다)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if r == QMessageBox.Yes:
                self._reanalyse_after_proxy = cut.id
        self._reopen_cut()

    def choose_lut(self):
        cut = self.session.cut
        if cut is None:
            return
        if cut.lut_path:
            m = QMenu(self)
            a1 = m.addAction("다른 LUT 고르기…")
            a2 = m.addAction("LUT 끄기")
            act = m.exec(self.cursor().pos())
            if act == a2:
                cut.lut_path = ""
                self.session.project.save()
                self._reopen_cut()
                return
            if act != a1:
                return
        f, _ = QFileDialog.getOpenFileName(self, "뷰 LUT", "", "LUT (*.cube *.3dl *.csp *.spi3d *.spi1d *.lut)")
        if f:
            cut.lut_path = f
            self.session.project.save()
            self._reopen_cut()

    def enqueue_render(self):
        s = self.session
        cut = s.cut
        if cut is None:
            return
        if not cut.analyzed:
            self.toast("분석 후에 렌더할 수 있습니다.", kind="warn")
            return
        if not cut.targets() or not any(cut.params_for(p.id).any_active() for p in cut.targets()):
            if QMessageBox.question(self, "보정 없음", "보정값이 없어 원본과 같은 결과가 나옵니다. 그래도 추가할까요?") != QMessageBox.Yes:
                return
        errs = [q for q in cut.qc if q.severity == "error" and q.status == "open"]
        if errs and QMessageBox.question(self, "QC 오류", f"QC 오류 {len(errs)}개가 남아 있습니다. 그래도 렌더 목록에 추가할까요?") != QMessageBox.Yes:
            return
        from .matte_refresh import pending

        def add(matte=""):
            self.save()
            it = self.queue_page.add(s.project, cut)
            if matte:
                it.matte = matte
                self.queue_page.save()
            self.toast("렌더 목록에 추가했습니다. 렌더 화면에서 시작하세요.", kind="ok")
        if self._needs_matte(s.project, cut) and s.tracker is not None and s.tracker.has_seg():
            # the render reads the saved matte: edits whose frames are not matted again yet would not be in it
            ch = self._ask_matte([cut.name], adding=True)
            if ch == "cancel":
                return
            if ch == "old":
                add("old")
                return

            def then(ok, why):
                if ok:
                    add()
                elif QMessageBox.question(self, "인물 매트 갱신 실패",
                                          f"인물 매트를 갱신하지 못했습니다: {why or '알 수 없는 이유'}\n"
                                          "이전 매트로 렌더 목록에 추가할까요?") == QMessageBox.Yes:
                    add("old")
            self.refresh_mattes_now([cut.id], then)
            return
        add()

    def before_render(self, item) -> Optional[bool]:
        """The render reads the project from disk: False when the open project could not be saved (the render would
        use an older state than the one on screen). None: not yet, the person matte of an edited cut is made
        again first (the queue is started again after it), or the user stopped the queue."""
        from .matte_refresh import pending
        prj = self.session.project
        mine = prj is not None and prj.path == item.project_path
        if getattr(item, "matte", "") != "old":
            src = prj if mine else None
            if src is None:
                try:
                    src = Project.load(item.project_path)
                except Exception:
                    src = None
            if src is not None:
                if getattr(item, "kind", "cut") == "timeline":
                    cuts = [c for c in src.cuts if self._needs_matte(src, c)]
                else:
                    c = src.cut(item.cut_id)
                    cuts = [c] if self._needs_matte(src, c) else []
                if cuts:
                    ch = self._ask_matte([c.name for c in cuts], adding=False)
                    if ch == "cancel":
                        self.toast("렌더를 멈췄습니다. 렌더 목록의 항목은 그대로 대기합니다.")
                        return None
                    if ch == "old":
                        item.matte = "old"
                    else:
                        def then(ok, why):
                            if not ok:
                                if QMessageBox.question(self, "인물 매트 갱신 실패",
                                                        f"인물 매트를 갱신하지 못했습니다: {why or '알 수 없는 이유'}\n"
                                                        "이전 매트로 렌더할까요?") != QMessageBox.Yes:
                                    return
                                item.matte = "old"
                            self.queue_page.start_queue()
                        self.refresh_mattes_now([c.id for c in cuts], then, project=src)
                        return None
        if mine:
            return self.save()
        return True

    def _render_finished(self, item, rep):
        if rep is None or getattr(rep, "cancelled", True):
            return
        if getattr(item, "kind", "cut") == "timeline":
            self.toast(f"타임라인 출력 완료: {item.message}", kind="ok")
            return
        prj = self.session.project
        if prj is not None and prj.path == item.project_path:
            cut = prj.cut(item.cut_id)
        else:
            try:
                prj = Project.load(item.project_path)
                cut = prj.cut(item.cut_id)
            except Exception:
                return
        if cut is None:
            return
        cut.stats.renders += 1
        cut.stats.render_seconds += rep.seconds
        cut.stats.last_output = rep.out_path
        if prj is self.session.project:
            self.session.dirty = True
            self.cutlist.update_rows()
        else:
            prj.save()
        self.toast(f"렌더 완료: {Path(rep.out_path).name}", kind="ok")

    def update_queue_badge(self, n: int):
        self.page_tabs.options[1] = ("queue", f"렌더 {n}" if n else "렌더")
        self.page_tabs.update()

    # ================================================================== undo
    def _snapshot(self) -> dict:
        cut = self.session.cut
        return dict(cut_id=cut.id, params={k: v.to_dict() for k, v in cut.params.items()},
                    keys=copy.deepcopy(cut.joint_keys), protect=cut.protect.to_dict(), mode=cut.mode,
                    persons=[p.to_dict() for p in cut.persons], locked=copy.deepcopy(cut.locked),
                    confirmed=copy.deepcopy(cut.confirmed), hidden=copy.deepcopy(cut.hidden))

    def _switch_undo(self, new_id: str) -> None:
        """Each cut keeps its own undo history while the project is open (going to another cut, by hand or by
        playing on into it, and back again keeps what can be undone there)."""
        cur = self.session.cut
        if cur is not None and cur.id != new_id and (self.undo_stack or self.redo_stack):
            self._undo_by_cut[cur.id] = (self.undo_stack, self.redo_stack)
            while len(self._undo_by_cut) > 30:
                self._undo_by_cut.pop(next(iter(self._undo_by_cut)))
        u, r = self._undo_by_cut.pop(new_id, ([], []))
        self.undo_stack, self.redo_stack = list(u), list(r)

    def drop_undo(self, cut_id: str) -> None:
        """The cut's tracking was made again (analysis): its old snapshots no longer fit its people."""
        self._undo_by_cut.pop(cut_id, None)

    def push_undo(self, coalesce: str = ""):
        if self.session.cut is None:
            return
        if coalesce and self.undo_stack and self.undo_stack[-1].get("_co") == coalesce and \
                time.time() - self.undo_stack[-1].get("_t", 0) < 1.5:
            self.undo_stack[-1]["_t"] = time.time()
            return
        snap = self._snapshot()
        snap["_co"] = coalesce
        snap["_t"] = time.time()
        self.undo_stack.append(snap)
        del self.undo_stack[:-60]
        self.redo_stack.clear()

    def _restore(self, snap: dict):
        from ..core.project import PartParams, Person, ProtectSettings
        cut = self.session.cut
        if cut is None or snap["cut_id"] != cut.id:
            return
        old_keys = cut.joint_keys
        old_hidden = cut.hidden
        cut.hidden = snap.get("hidden", cut.hidden)
        cut.params = {int(k): PartParams.from_dict(v) for k, v in snap["params"].items()}
        cut.joint_keys = snap["keys"]
        cut.protect = ProtectSettings.from_dict(snap["protect"])
        cut.mode = snap["mode"]
        cut.persons = [Person.from_dict(p) for p in snap["persons"]]
        cut.locked = snap["locked"]
        cut.confirmed = snap["confirmed"]
        # re-solve people whose keys changed
        s = self.session
        if s.tracker is not None and s.tracker.store is not None:
            for pid in set(old_keys) | set(cut.joint_keys):
                a = {(j, f) for j, fr in old_keys.get(pid, {}).items() for f in fr}
                b = {(j, f) for j, fr in cut.joint_keys.get(pid, {}).items() for f in fr}
                changed = a ^ b
                changed |= {(j, f) for (j, f) in a & b if old_keys[pid][j][f] != cut.joint_keys[pid][j][f]}
                for f in sorted({f for _, f in changed}):
                    self.edit_worker.submit(("resolve", pid), resolve_task(s, pid, f))
            for pid in set(old_hidden) | set(cut.hidden):
                if old_hidden.get(pid) != cut.hidden.get(pid):
                    rs = [r for jd in (old_hidden.get(pid, {}), cut.hidden.get(pid, {})) for v in jd.values() for r in v]
                    if rs:
                        a0, b0 = min(r[0] for r in rs), max(r[1] for r in rs)
                        self.edit_worker.submit(("resolve_h", pid),
                                                resolve_task(s, pid, a0, frame_end=b0))
        s.touch()
        self._refresh_all()

    def undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        self.redo_stack.append(self._snapshot())
        self._restore(snap)
        self.toast("실행 취소")

    def redo(self):
        if not self.redo_stack:
            return
        snap = self.redo_stack.pop()
        self.undo_stack.append(self._snapshot())
        self._restore(snap)
        self.toast("다시 실행")

    # ================================================================== misc
    def show_page(self, key: str):
        """edit | queue | projects (the top tabs follow)."""
        self.page_tabs.set_current(key if key in ("edit", "queue") else "")
        self._page(key)

    def _page(self, key: str):
        idx = {"edit": 0, "queue": 1, "dash": 2, "models": 3, "projects": 4}[key]
        if key != "edit":
            self.stop_play()
            self.save()
        self.pages.setCurrentIndex(idx)
        for w_ in getattr(self, "_top_items", []):
            w_.setVisible(key != "projects")
        if key == "projects":
            self.projects_page.reload()
        if key == "dash":
            self.dash_page.refresh()
        elif key == "models":
            self.models_page.refresh()
        elif key == "queue":
            self.queue_page.refresh()

    def _refresh_all(self, skip_inspector: bool = False):
        s = self.session
        cut = s.cut
        self.viewer.sel_pid = self.sel_pid
        self.timeline.sel_pid = self.sel_pid
        self.viewer.update()
        self.timeline.relayout()
        self.timeline.update()
        if cut is not None and s.proxy is not None:
            self.set_frame(self.frame)
            cs = cut.input_cs or "변환 없음"
            self.cs_lbl.setText(f"입력 {cs}" + (" + LUT" if cut.lut_path else ""))
        else:
            self.cs_lbl.setText("")
        self._proxy_note()
        if not skip_inspector:
            self.inspector.refresh()
        self.cutlist.update_rows()

    def toast(self, text: str, kind: str = ""):
        self.toast_w.show_msg(text, kind=kind)

    def show_error(self, title: str, err: str):
        error_box(self, title, err)

    def open_settings(self):
        d = SettingsDialog(self)
        ok = d.exec()
        if getattr(d, "models_changed", False):
            self._models_changed()
        if ok:
            from ..core import playcache as PC
            before = PC.setting_values()
            changed = d.apply() or PC.setting_values() != before     # the view, or a setting results are made with
            PC.cache().set_limit(PC.limit_bytes())
            if d.check_update:
                from . import update as UP
                UP.check(self, manual=True)
            try:
                from ..track.models import device_label
                self.dev_lbl.setText(f"추론: {device_label()}")
            except Exception:
                pass
            if changed and self.session.cut is not None:
                self._reopen_cut()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        from ..core import projects as PJ
        prjs = [PJ.norm(p) for p in paths if Path(p).is_dir() and PJ.is_project_dir(p)]
        if prjs:
            # project folders (.dpprj / .contour): into the project manager (opened when dropped on the editor)
            for p in prjs:
                PJ.remember(p)
            if self.pages.currentIndex() == 4 or len(prjs) > 1:
                self.show_page("projects")
                self.toast(f"프로젝트 {len(prjs)}개를 목록에 추가했습니다", "ok")
            else:
                self.open_project(prjs[0])
            paths = [p for p in paths if PJ.norm(p) not in prjs]
            if not paths:
                return
        xmls = [p for p in paths if p.lower().endswith((".xml", ".fcpxml", ".edl"))]
        paths = [p for p in paths if not p.lower().endswith((".xml", ".fcpxml", ".edl"))]
        if xmls and (len(paths) >= 1 or not xmls[0].lower().endswith(".xml")):
            # an editing file with its baked video: one cut per shot
            vids = [p for p in paths if not self._is_after(p)]
            afters = [p for p in paths if self._is_after(p)]
            self.import_timeline(xmls[0], vids[0] if vids else "", afters[0] if afters else "")
            return
        if paths:
            self.add_cuts(paths)
        for x in xmls:
            # the baked video was dropped first (it became one plain cut) and the sequence XML after it:
            # that is a timeline, split it. Only an XML of the plates' own names keeps the old edit-point use.
            baked = self._baked_for(x)
            if baked is not None:
                self.import_timeline(x, baked.source, baked.retouch_ref or "")
                return
            if not self.import_edit_xml(x, quiet=True):
                self.import_timeline(x, "", "")
                return

    def _baked_for(self, xml: str):
        """A plain (non-timeline) video cut in the project that is this sequence's bake: same length."""
        from ..core.timeline import read_timeline
        prj = self.session.project
        if prj is None:
            return None
        try:
            tl = read_timeline(xml)
        except Exception:
            return None
        if sum(1 for s in tl.shots if s.kind == "clip") < 2 or tl.duration <= 0:
            return None
        dur = tl.duration
        best, bd = None, None
        for c in prj.cuts:
            if c.tl or c.info.kind != "video":
                continue
            d = abs(int(c.info.frames) - dur)
            if d <= max(3, int(0.01 * dur)) and (bd is None or d < bd):
                best, bd = c, d
        return best

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.busy.isVisible():
            self.busy.setGeometry(self.center.rect())

    def closeEvent(self, e):
        self._close_saved = False
        if self.session.project is not None and self.session.dirty and not getattr(self, "_close_decided", False):
            m = QMessageBox(self)
            m.setWindowTitle("DIAP")
            m.setIcon(QMessageBox.Question)
            m.setText("저장하지 않은 변경 사항이 있습니다.")
            m.setInformativeText("저장하고 닫을까요?")
            b_save = m.addButton("저장하고 닫기", QMessageBox.AcceptRole)
            b_no = m.addButton("저장 안 하고 닫기", QMessageBox.DestructiveRole)
            m.addButton("취소", QMessageBox.RejectRole)
            m.setDefaultButton(b_save)
            m.exec()
            if m.clickedButton() is b_save:
                if not self.save():
                    e.ignore()                  # not saved: the window stays open
                    return
                self._close_saved = True
            elif m.clickedButton() is b_no:
                self.session.dirty = False
                self._discard_on_close = True
                self._clear_recovery(Path(self.session.project.path))
            else:
                e.ignore()
                return
        if not getattr(self, "_discard_on_close", False) and not self._close_saved and self.job is None:
            # saved before anything is stopped: a failed save leaves the window (and a running render) as they were
            if not self._save_or_ask("종료"):
                e.ignore()
                return
            self._close_saved = True
        if self.queue_page.job is not None:
            if QMessageBox.question(self, "렌더 중", "렌더가 진행 중입니다. 종료하면 현재 렌더는 취소됩니다. 종료할까요?") != QMessageBox.Yes:
                e.ignore()
                return
            self.queue_page.job.cancel()
            self.queue_page.job.wait(5000)
        try:
            self.proxy_builder.cancel(wait=5.0)
            self.thumbcache.stop()
        except Exception:
            pass
        if self.batch.busy:
            if QMessageBox.question(self, "일괄 분석 중", "일괄 분석이 진행 중입니다. 종료하면 지금 컷의 분석은 버려집니다. 종료할까요?") != QMessageBox.Yes:
                e.ignore()
                return
            self.batch.kill()
        self.refresher.enabled = False           # a running matte refresh is dropped: its marks are saved with the cut
        self.refresher.cancel()
        if self.job is not None:
            self.job.cancel()
            self.job.wait(5000)
        if getattr(self, "_opening", None) is not None:
            self._opening.wait(15000)
        self.projects_page.shutdown()
        self.stop_play()
        if not getattr(self, "_discard_on_close", False) and not self._close_saved \
                and not self._save_or_ask("종료"):
            e.ignore()
            return
        self.activity.flush()
        self.edit_worker.stop()
        self.edit_worker.wait(3000)
        self.viewer.stop_workers(3000)
        self.cache_timer.stop()
        self.cache_worker.stop()
        self.cache_worker.wait(5000)
        e.accept()
