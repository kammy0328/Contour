"""프로젝트 관리: the first page after start-up. Every known project as a card (thumbnail, name, client, cuts,
when it was changed, size), search and sort, double click to open, right click for 열기 / 이름 변경 /
탐색기에서 보기 / 프록시 캐시 정리 / 삭제 (a confirmation first: out of the list only, or the folder to the
recycle bin)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtCore import QFile
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QStyle, QStyledItemDelegate,
                               QVBoxLayout, QWidget)

from . import widgets as Wd
from ..core import projects as PJ
from .theme import C, font, qc

CW, CH = 280, 258          # card size
TH = 150                   # thumbnail height
ROLE = Qt.UserRole + 1


class _Loader(QThread):
    """Thumbnails and folder sizes off the main thread (projects may sit on a NAS)."""
    got = Signal(str, object, object)          # path, rgb thumbnail or None, sizes dict or None

    def __init__(self, parent, paths: List[str]):
        super().__init__(parent)
        self.paths = list(paths)
        self.stop = False

    def run(self):
        from ..core.project import Project
        from ..core.proxy import Proxy
        from ..core.settings import settings
        for p in self.paths:
            if self.stop:
                return
            th = None
            try:
                if PJ.is_project_dir(p):
                    prj = Project.load(p)
                    for c in prj.cuts[:6]:
                        if (c.tl or {}).get("kind", "clip") != "clip":
                            continue
                        px = Proxy(prj.cut_dir(c) / "cache", c.info, settings().proxy_long_side)
                        if px.has_frames():
                            a, b = c.work_range
                            th = px.thumbnail((a + b) // 2, 360)
                            px.close()
                            break
                        if c.tl:
                            from ..core.tlimport import load_thumb
                            th = load_thumb(prj, c.id)
                            if th is not None:
                                break
            except Exception:
                th = None
            sz = None
            try:
                if PJ.is_project_dir(p):
                    sz = PJ.sizes(p)
            except Exception:
                pass
            self.got.emit(p, th, sz)


class _CardDelegate(QStyledItemDelegate):
    def __init__(self, page):
        super().__init__(page)
        self.page = page

    def sizeHint(self, opt, ix):
        return QSize(CW, CH)

    def paint(self, p: QPainter, opt, ix):
        d: Dict = ix.data(ROLE) or {}
        path = d.get("path", "")
        r = QRectF(opt.rect).adjusted(6, 6, -6, -6)
        sel = bool(opt.state & QStyle.State_Selected)
        hov = bool(opt.state & QStyle.State_MouseOver)
        cur = path == self.page.current_path()
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(qc("accent") if sel else (qc("line2") if hov else qc("line")), 2 if sel else 1))
        p.setBrush(qc("bg2") if (sel or hov) else qc("bg1"))
        p.drawRoundedRect(r, 10, 10)
        tr = QRectF(r.left() + 8, r.top() + 8, r.width() - 16, TH)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0))
        p.drawRoundedRect(tr, 6, 6)
        pm: Optional[QPixmap] = self.page.thumbs.get(path)
        missing = not d.get("exists", False)
        if pm is not None and not pm.isNull():
            s = pm.size().scaled(int(tr.width()), int(tr.height()), Qt.KeepAspectRatio)
            dst = QRectF(tr.center().x() - s.width() / 2, tr.center().y() - s.height() / 2, s.width(), s.height())
            p.drawPixmap(dst.toRect(), pm)
        else:
            p.setPen(qc("text3"))
            p.setFont(font(11, 500))
            p.drawText(tr, Qt.AlignCenter, "찾을 수 없음" if missing else ("손상됨" if d.get("broken") else "미리보기 없음"))
        if cur:
            b = QRectF(tr.left() + 6, tr.top() + 6, 58, 18)
            p.setPen(Qt.NoPen)
            p.setBrush(qc("accent"))
            p.drawRoundedRect(b, 4, 4)
            p.setPen(QColor("white"))
            p.setFont(font(10, 600))
            p.drawText(b, Qt.AlignCenter, "열려 있음")
        y = tr.bottom() + 8
        p.setPen(qc("text3") if missing else qc("text"))
        p.setFont(font(13, 600))
        fm = p.fontMetrics()
        p.drawText(QRectF(tr.left(), y, tr.width(), 20), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(d.get("name", ""), Qt.ElideRight, int(tr.width())))
        y += 21
        p.setFont(font(11, 400))
        fm = p.fontMetrics()
        p.setPen(qc("text2"))
        if missing:
            line1 = "드라이브가 연결되지 않았거나 폴더가 옮겨졌습니다"
        else:
            bits = [f"컷 {d.get('cuts', 0)}"]
            if d.get("done"):
                bits.append(f"완료 {d['done']}")
            if d.get("client"):
                bits.append(d["client"])
            line1 = " · ".join(bits)
        p.drawText(QRectF(tr.left(), y, tr.width(), 17), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(line1, Qt.ElideRight, int(tr.width())))
        y += 17
        p.setPen(qc("text3"))
        sz = self.page.sizes.get(path)
        line2 = PJ.when(d.get("modified", 0))
        if sz:
            line2 += f" · {PJ.human(sz['total'])}"
            if sz.get("proxy"):
                line2 += f" · 프록시 {PJ.human(sz['proxy'])}"
        p.drawText(QRectF(tr.left(), y, tr.width(), 17), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(line2.strip(" ·"), Qt.ElideRight, int(tr.width())))
        y += 17
        p.drawText(QRectF(tr.left(), y, tr.width(), 17), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(path, Qt.ElideMiddle, int(tr.width())))
        p.restore()


class ProjectsPage(QWidget):
    """main: the MainWindow (open_project, new_project, close_project, session, queue_page, toast)."""

    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        self.thumbs: Dict[str, QPixmap] = {}
        self.sizes: Dict[str, Dict] = {}
        self.infos: Dict[str, Dict] = {}
        self._loader: Optional[_Loader] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(32, 26, 32, 20)
        lay.setSpacing(12)
        head = QHBoxLayout()
        t = QLabel("프로젝트")
        t.setFont(font(22, 600))
        head.addWidget(t)
        head.addStretch(1)
        self.new_b = Wd.button("새 프로젝트", "primary", "plus")
        self.new_b.clicked.connect(self.main.new_project)
        self.open_b = Wd.button("폴더에서 열기", "", "folder-open", "프로젝트 폴더(.dpprj, 예전 .contour)를 골라 엽니다")
        self.open_b.clicked.connect(self._open_folder)
        self.find_b = Wd.button("", "ghost", "search", "검색 (Ctrl+F)")
        self.find_b.setCheckable(True)
        self.find_b.toggled.connect(self._toggle_search)
        head.addWidget(self.find_b)
        head.addWidget(self.new_b)
        head.addWidget(self.open_b)
        lay.addLayout(head)
        sub = Wd.label("더블클릭으로 열고, 우클릭으로 이름 변경·삭제·프록시 캐시 정리를 할 수 있습니다. "
                       "프로젝트 폴더(.dpprj)를 끌어다 놓으면 목록에 추가됩니다.", "Faint", 12)
        lay.addWidget(sub)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("이름, 클라이언트, 경로 검색")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refresh_list)
        self.sort = QComboBox()
        for k, lab in (("modified", "최근 수정순"), ("name", "이름순"), ("created", "만든 날짜순")):
            self.sort.addItem(lab, k)
        from ..core.settings import settings as _st
        i_ = self.sort.findData(getattr(_st(), "pm_sort", "modified") or "modified")
        self.sort.setCurrentIndex(max(0, i_))
        self.sort.currentIndexChanged.connect(self._sort_changed)
        self.count = Wd.label("", "Faint", 12)
        self.search.setVisible(False)
        bar.addWidget(self.search, 1)
        bar.addStretch(1)
        bar.addWidget(self.sort)
        bar.addWidget(self.count)
        lay.addLayout(bar)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setUniformItemSizes(True)
        self.list.setGridSize(QSize(CW + 4, CH + 4))
        self.list.setSpacing(0)
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setItemDelegate(_CardDelegate(self))
        self.list.setStyleSheet("QListWidget { background: transparent; border: none; }")
        self.list.itemDoubleClicked.connect(lambda it: self._open(it.data(ROLE)["path"]))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list, 1)
        self.empty = Wd.label("아직 프로젝트가 없습니다. '새 프로젝트'로 만들거나 '폴더에서 열기'로 기존 프로젝트를 여세요. "
                              "영상을 창에 끌어다 놓아도 됩니다.", "Faint", 13)
        self.empty.setWordWrap(True)
        lay.addWidget(self.empty)

    def _toggle_search(self, on: bool):
        self.search.setVisible(on)
        if on:
            self.search.setFocus()
        else:
            self.search.clear()

    def keyPressEvent(self, e):
        from PySide6.QtGui import QKeySequence
        if e.matches(QKeySequence.Find):
            self.find_b.setChecked(True)
            return
        if e.key() == Qt.Key_Escape and self.find_b.isChecked():
            self.find_b.setChecked(False)
            return
        super().keyPressEvent(e)

    def _sort_changed(self, *_):
        from ..core.settings import settings as _st
        s = _st()
        s.pm_sort = self.sort.currentData() or "modified"
        s.save()
        self.refresh_list()

    # ------------------------------------------------------------------ data
    def current_path(self) -> str:
        prj = self.main.session.project
        return PJ.norm(prj.path) if prj is not None else ""

    def reload(self):
        """Reads every known project's project.json (fast) and starts the thumbnail and size loader."""
        self.infos = {p: PJ.info(p) for p in PJ.known()}
        self.refresh_list()
        if self._loader is not None:
            self._loader.stop = True
            self._loader.wait(2000)
        todo = [p for p, d in self.infos.items() if d.get("exists") and p not in self.thumbs]
        todo += [p for p, d in self.infos.items() if d.get("exists") and p in self.thumbs]   # sizes again
        if todo:
            self._loader = _Loader(self, todo)
            self._loader.got.connect(self._got)
            self._loader.start()

    def _got(self, path, th, sz):
        if th is not None:
            from .viewer import np_to_qimage
            self.thumbs[path] = QPixmap.fromImage(np_to_qimage(np.ascontiguousarray(th)))
        if sz is not None:
            self.sizes[path] = sz
        self.list.viewport().update()

    def refresh_list(self):
        q = self.search.text().strip().lower()
        key = self.sort.currentData() or "modified"
        rows = list(self.infos.values())
        if q:
            rows = [d for d in rows if q in (d.get("name", "") + " " + d.get("client", "") + " " + d["path"]).lower()]
        if key == "name":
            rows.sort(key=lambda d: d.get("name", "").lower())
        else:
            rows.sort(key=lambda d: (bool(d.get("exists")), d.get(key, 0) or 0), reverse=True)
        sel = self._sel_path()
        self.list.clear()
        for d in rows:
            it = QListWidgetItem()
            it.setData(ROLE, d)
            it.setSizeHint(QSize(CW, CH))
            it.setToolTip(d["path"])
            self.list.addItem(it)
            if d["path"] == sel:
                it.setSelected(True)
        n = len(self.infos)
        self.count.setText(f"{len(rows)} / {n}개" if q else f"{n}개")
        self.empty.setVisible(n == 0)
        self.list.setVisible(n > 0)

    def _sel_path(self) -> str:
        it = self.list.currentItem()
        return it.data(ROLE)["path"] if it is not None else ""

    # ------------------------------------------------------------------ actions
    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 (.dpprj) 선택", str(Path.home()))
        if not d:
            return
        if not PJ.is_project_dir(d):
            QMessageBox.warning(self, "프로젝트 열기", "이 폴더에는 프로젝트(project.json)가 없습니다.")
            return
        self._open(d)

    def _open(self, path: str):
        if not PJ.is_project_dir(path):
            self.main.toast("프로젝트를 찾을 수 없습니다", "warn")
            return
        if path == self.current_path():
            self.main.show_page("edit")
            return
        PJ.remember(path)
        self.main.open_project(path)
        self.main.show_page("edit")

    def _blocked(self, path: str, what: str) -> bool:
        """Rendering, analysing or queued work in this project: no rename or delete."""
        qp = self.main.queue_page
        n = sum(1 for i in qp.items if PJ.norm(i.project_path) == path and i.status in ("wait", "running"))
        if n:
            QMessageBox.information(self, what, f"렌더 목록에 이 프로젝트의 대기·렌더 중 항목이 {n}개 있습니다.\n"
                                                "렌더가 끝나거나 목록에서 뺀 뒤 다시 하세요.")
            return True
        if path == self.current_path() and (self.main.batch.busy or self.main.job is not None):
            QMessageBox.information(self, what, "이 프로젝트에서 작업이 진행 중입니다. 끝난 뒤 다시 하세요.")
            return True
        return False

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        self.list.setCurrentItem(it)
        d = it.data(ROLE)
        path = d["path"]
        m = QMenu(self)
        if not d.get("exists"):
            a_find = m.addAction("다시 찾기…")
            a_drop = m.addAction("목록에서 빼기")
            act = m.exec(self.list.viewport().mapToGlobal(pos))
            if act == a_find:
                nd = QFileDialog.getExistingDirectory(self, f"'{d.get('name')}' 프로젝트 폴더 찾기", str(Path(path).parent))
                if nd and PJ.is_project_dir(nd):
                    PJ.forget(path)
                    PJ.remember(nd)
                    self.reload()
                elif nd:
                    QMessageBox.warning(self, "다시 찾기", "이 폴더에는 프로젝트(project.json)가 없습니다.")
            elif act == a_drop:
                PJ.forget(path)
                self.reload()
            return
        a_open = m.addAction("열기")
        a_ren = m.addAction("이름 변경…")
        a_show = m.addAction("탐색기에서 보기")
        m.addSeparator()
        sz = self.sizes.get(path, {})
        a_cache = m.addAction("프록시 캐시 정리…" + (f"  ({PJ.human(sz['proxy'])})" if sz.get("proxy") else ""))
        a_cache.setEnabled(bool(sz.get("proxy")) or path not in self.sizes)
        m.addSeparator()
        a_del = m.addAction("삭제…")
        act = m.exec(self.list.viewport().mapToGlobal(pos))
        if act == a_open:
            self._open(path)
        elif act == a_ren:
            self._rename(d)
        elif act == a_show:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
        elif act == a_cache:
            self._clear_cache(d)
        elif act == a_del:
            self._delete(d)

    def _release_if_open(self, path: str, what: str) -> bool:
        """The open project is closed (saved) first; False if the user says no."""
        if path != self.current_path():
            return True
        r = QMessageBox.question(self, what, "지금 열려 있는 프로젝트입니다. 저장하고 닫은 뒤 진행할까요?")
        if r != QMessageBox.Yes:
            return False
        # a failed save keeps the project open: renaming or deleting its folder now would lose what was not written
        return bool(self.main.close_project())

    def _rename(self, d: Dict):
        from .dialogs import TextPrompt
        path = d["path"]
        if self._blocked(path, "이름 변경"):
            return
        name = TextPrompt.ask(self, "이름 변경", "새 프로젝트 이름 (폴더 이름도 함께 바뀝니다, 공백은 _로)",
                              d.get("name", ""), "변경")
        if not name or name.strip() == d.get("name"):
            return
        if not self._release_if_open(path, "이름 변경"):
            return
        try:
            new = PJ.rename(path, name)
        except Exception as e:
            QMessageBox.warning(self, "이름 변경", f"이름을 바꾸지 못했습니다.\n{e}\n\n"
                                "다른 프로그램(탐색기, 영상 플레이어 등)이 이 폴더의 파일을 쓰고 있으면 닫고 다시 하세요.")
            return
        # render list entries of this project follow the folder
        qp = self.main.queue_page
        nm = PJ.info(new).get("name", name)
        for it in qp.items:
            if PJ.norm(it.project_path) == path:
                it.project_path = new
                it.project_name = nm
        qp._prj_cache.pop(path, None)
        qp.save()
        qp.refresh()
        self.thumbs[new] = self.thumbs.pop(path, None) or QPixmap()
        if path in self.sizes:
            self.sizes[new] = self.sizes.pop(path)
        self.reload()
        self.main.toast(f"이름을 '{nm}'(으)로 바꿨습니다 · {Path(new).name}", "ok")

    def _clear_cache(self, d: Dict):
        path = d["path"]
        sz = self.sizes.get(path) or PJ.sizes(path)
        if path == self.current_path():
            QMessageBox.information(self, "프록시 캐시 정리", "열려 있는 프로젝트는 정리할 수 없습니다. 다른 프로젝트를 열거나 닫은 뒤 하세요.")
            return
        if self._blocked(path, "프록시 캐시 정리"):
            return
        r = QMessageBox.question(
            self, "프록시 캐시 정리",
            f"'{d.get('name')}'의 프록시 {PJ.human(sz.get('proxy', 0))}를 지울까요?\n\n"
            "분석 결과, 매트, 보정 값은 그대로 남습니다. 컷을 다시 열면 이 PC의 캐시 폴더에 프록시를 새로 만듭니다"
            "(컷마다 시간이 걸립니다).")
        if r != QMessageBox.Yes:
            return
        freed = PJ.clear_proxies(path)
        self.sizes.pop(path, None)
        self.thumbs.pop(path, None)
        self.reload()
        self.main.toast(f"{PJ.human(freed)}를 비웠습니다", "ok")

    def _delete(self, d: Dict):
        path = d["path"]
        if self._blocked(path, "삭제"):
            return
        sz = self.sizes.get(path)
        m = QMessageBox(self)
        m.setWindowTitle("프로젝트 삭제")
        m.setIcon(QMessageBox.Warning)
        m.setText(f"'{d.get('name')}' 프로젝트를 삭제할까요?")
        info = f"컷 {d.get('cuts', 0)}개"
        if sz:
            info += f" · 폴더 {PJ.human(sz['total'])}"
        m.setInformativeText(
            f"{info}\n{path}\n\n"
            "휴지통으로: 프로젝트 폴더(분석 결과, 프록시, 보정 값)를 휴지통으로 보냅니다. 휴지통에서 되살릴 수 있습니다.\n"
            "목록에서만 빼기: 폴더는 그대로 두고 이 목록에서만 뺍니다.\n\n"
            "원본 영상과 렌더 결과(DIAP_Renders)는 어느 쪽이든 지워지지 않습니다.")
        b_trash = m.addButton("휴지통으로", QMessageBox.DestructiveRole)
        b_list = m.addButton("목록에서만 빼기", QMessageBox.AcceptRole)
        b_no = m.addButton("취소", QMessageBox.RejectRole)
        m.setDefaultButton(b_no)
        m.exec()
        c = m.clickedButton()
        if c is b_list:
            if path == self.current_path() and not self._release_if_open(path, "목록에서 빼기"):
                return
            PJ.forget(path)
            self.reload()
            self.main.toast("목록에서 뺐습니다 (폴더는 그대로)", "ok")
        elif c is b_trash:
            if not self._release_if_open(path, "삭제"):
                return
            lc = PJ.local_cache(path)            # this computer's proxies of the project (read before it moves)
            ok = QFile.moveToTrash(path)
            if isinstance(ok, tuple):
                ok = ok[0]
            if not ok:
                QMessageBox.warning(self, "삭제", "휴지통으로 보내지 못했습니다. 다른 프로그램이 폴더의 파일을 쓰고 있거나, "
                                    "네트워크 드라이브처럼 휴지통이 없는 위치일 수 있습니다.\n폴더는 그대로 있습니다.")
                return
            if lc is not None:
                import shutil
                shutil.rmtree(lc, ignore_errors=True)
            PJ.forget(path)
            self.thumbs.pop(path, None)
            self.sizes.pop(path, None)
            self.reload()
            self.main.toast(f"'{d.get('name')}'을(를) 휴지통으로 보냈습니다", "ok")

    def shutdown(self):
        if self._loader is not None:
            self._loader.stop = True
            self._loader.wait(3000)
