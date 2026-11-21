"""모델 관리: what each kind of work needs and has, the model packages on this computer, bringing a package in
from a folder or a zip (a computer that is never online), checking the installed files, going back to a version.

Only model files come in through here (a signed list and the files it names); nothing is sent (instruction
2026-11-20 §4)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from . import widgets as Wd
from ..core.settings import settings
from ..track import modelpkg as MP
from ..track import models as M
from .workers import Job

SOURCE_KO = {"package": "패키지", "app": "앱이 받은 파일", "none": "없음"}


def _mb(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n else ""


def needed_text(cut) -> str:
    """For a cut analysed with package versions that are not on this computer: what is needed and the choice
    ('' otherwise). Editing, preview and render of what is saved go on either way."""
    miss = [n for n in MP.needs(getattr(cut, "model_versions", None) or {}) if not n["installed"]]
    if not miss:
        return ""
    return (str(MP.VersionNeeded(miss)) + " 그때까지 이 컷의 매트 갱신·구간 다시 분석·옷과 얼굴 영역 다시 만들기는 쉬고, "
            "보정값 편집·미리보기·렌더는 저장된 분석 결과로 그대로 됩니다.")


def ask_versions(parent, text: str) -> str:
    """The explicit choice when work on a cut needs versions that are not here: 'models' (open 모델 관리 to
    install them), 'reanalyse' (the whole cut again with the versions in use), '' (leave it)."""
    m = QMessageBox(parent)
    m.setIcon(QMessageBox.Warning)
    m.setWindowTitle("다른 모델 버전으로 분석된 컷")
    m.setText(text.strip().splitlines()[0])
    m.setInformativeText("설치된 다른 버전으로 자동으로 바꾸지 않습니다. 한 컷의 결과가 두 버전으로 섞이지 않게, "
                         "그 버전을 설치하거나 컷 전체를 지금 버전으로 다시 분석하는 것 중에서 고르세요.")
    b_models = m.addButton("모델 관리 열기", QMessageBox.ActionRole)
    b_again = m.addButton("컷 전체 다시 분석", QMessageBox.DestructiveRole)
    m.addButton("닫기", QMessageBox.RejectRole)
    m.exec()
    got = m.clickedButton()
    return "models" if got is b_models else "reanalyse" if got is b_again else ""


class ModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("모델 관리")
        self.resize(760, 620)
        self.job: Optional[Job] = None
        self.changed = False                      # a version was put in use or taken out: open cuts are read again
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(8)
        v.addWidget(Wd.label("작업별 상태", "", 13, 600))
        self.tasks = QLabel("")
        self.tasks.setWordWrap(True)
        self.tasks.setTextFormat(Qt.RichText)
        v.addWidget(self.tasks)
        always = Wd.label(M.ALWAYS, "Faint", 11)
        always.setWordWrap(True)
        v.addWidget(always)
        v.addWidget(Wd.label("이 컴퓨터의 모델", "", 13, 600))
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["모델", "있는 곳", "사용 중인 버전", "설치된 다른 버전", "크기"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3, 4):
            self.tree.header().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._buttons)
        v.addWidget(self.tree, 1)
        h = QHBoxLayout()
        self.b_dir = Wd.button("패키지 폴더 가져오기…", "", "",
                               "모델 패키지가 든 폴더(manifest.json, manifest.sig와 모델 파일)를 골라 설치합니다. "
                               "인터넷 없이, 받을 때와 같은 서명·크기·SHA-256 검사를 거칩니다")
        self.b_zip = Wd.button("패키지 파일(zip) 가져오기…", "", "", "모델 패키지 zip 파일 하나를 골라 같은 검사를 거쳐 설치합니다")
        self.b_check = Wd.button("파일 검사", "", "", "고른 모델(없으면 전부)의 설치된 파일을 목록의 크기·SHA-256과 다시 맞춰 봅니다")
        self.b_use = Wd.button("다른 버전 사용…", "", "", "고른 모델의 설치된 버전 중 하나를 사용하게 합니다 (이전 버전으로 돌아가기)")
        self.b_del = Wd.button("쓰지 않는 버전 지우기", "", "", "고른 모델의 사용 중이 아닌 설치 버전을 지웁니다")
        self.b_dir.clicked.connect(lambda: self._pick(True))
        self.b_zip.clicked.connect(lambda: self._pick(False))
        self.b_check.clicked.connect(self._check)
        self.b_use.clicked.connect(self._use)
        self.b_del.clicked.connect(self._remove)
        for b in (self.b_dir, self.b_zip, self.b_check, self.b_use, self.b_del):
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        # the address of a model list (no storage provider is decided: empty unless set by hand)
        r = QHBoxLayout()
        self.addr = QLineEdit(getattr(settings(), "model_registry", "") or "")
        self.addr.setPlaceholderText("모델 배포 주소 (https://… 또는 폴더). 비워 두면 온라인으로 확인하지 않습니다")
        self.b_net = Wd.button("목록 확인·받기", "", "", "이 주소의 서명된 모델 목록에서 이 버전의 DIAP에 맞는 최신 패키지를 받습니다. "
                                                        "받는 것은 목록과 모델 파일뿐이고, 보내는 것은 없습니다")
        self.b_net.clicked.connect(self._online)
        r.addWidget(self.addr, 1)
        r.addWidget(self.b_net)
        v.addLayout(r)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.msg = Wd.label("", "Faint", 11)
        self.msg.setWordWrap(True)
        v.addWidget(self.msg)
        note = Wd.label("", "Faint", 11)
        note.setWordWrap(True)
        if not MP.TRUSTED:
            note.setText("이 버전의 DIAP에는 모델 배포 키가 아직 없어 패키지를 설치할 수 없습니다. 모델은 지금처럼 필요할 때 앱이 "
                         "직접 받습니다. 영상·프레임·크롭·마스크·프로젝트 파일은 어디로도 보내지 않습니다.")
        else:
            note.setText("받는 것은 서명된 모델 목록과 그 목록의 모델 파일뿐입니다. 영상·프레임·크롭·마스크·프로젝트 파일은 "
                         "어디로도 보내지 않습니다. 진행 중인 분석은 시작할 때의 버전으로 끝납니다.")
        v.addWidget(note)
        hb = QHBoxLayout()
        hb.addStretch(1)
        self.b_close = Wd.button("닫기", "primary")
        self.b_close.clicked.connect(self.accept)
        hb.addWidget(self.b_close)
        v.addLayout(hb)
        self.refresh()

    # ------------------------------------------------------------------ showing
    def refresh(self):
        rows = []
        for a in M.availability():
            if a["ok"]:
                rows.append(f"<b>{a['label']}</b>: 사용 가능")
            else:
                names = ", ".join(f"{m['label']}" + (f" ({m['size_mb']} MB)" if m["size_mb"] else "") for m in a["missing"])
                rows.append(f"<b>{a['label']}</b>: 모델 없음 ({names}). {a['limit']}")
        self.tasks.setText("<br>".join(rows))
        keep = self._key()
        self.tree.clear()
        for r in MP.listing():
            others = [x for x in r["versions"] if x != r["version"]]
            it = QTreeWidgetItem([r["label"], SOURCE_KO[r["source"]], r["version"] or ("-" if r["source"] != "package" else ""),
                                  ", ".join(others), _mb(r["size"]) if r["source"] != "none" else f"({_mb(r['size'])})"])
            it.setData(0, Qt.UserRole, r)
            it.setToolTip(0, r["key"] + (f" · 진행 중인 작업이 쓰는 버전 {', '.join(r['pinned'])}" if r["pinned"] else ""))
            self.tree.addTopLevelItem(it)
            if r["key"] == keep:
                self.tree.setCurrentItem(it)
        self._buttons()

    def _row(self) -> Optional[dict]:
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it is not None else None

    def _key(self) -> str:
        r = self._row()
        return r["key"] if r else ""

    def _buttons(self):
        r = self._row()
        idle = self.job is None
        can = bool(MP.TRUSTED)
        self.b_dir.setEnabled(idle and can)
        self.b_zip.setEnabled(idle and can)
        self.b_net.setEnabled(idle and can)
        self.addr.setEnabled(idle and can)
        self.b_check.setEnabled(idle and any(x["source"] == "package" for x in MP.listing()))
        self.b_use.setEnabled(idle and bool(r) and len(r["versions"]) > 1)
        self.b_del.setEnabled(idle and bool(r) and any(x != r["version"] for x in r["versions"]))

    # ------------------------------------------------------------------ work in the background
    def _run(self, text: str, fn, done):
        if self.job is not None:
            return
        self.msg.setText(text)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.job = Job(fn, self)
        self.job.progressed.connect(lambda f, t: (self.bar.setValue(int(f * 1000)), self.msg.setText(t) if t else None))

        def end(res=None, err=None):
            self.job = None
            self.bar.setVisible(False)
            if err is not None:
                self.msg.setText("취소했습니다." if err == "__cancelled__" else err.strip().splitlines()[0])
            else:
                done(res)
            self.refresh()
        self.job.succeeded.connect(lambda res: end(res))
        self.job.failed.connect(lambda err: end(None, err))
        self._buttons()
        self.job.start()

    def _installed(self, res: dict):
        parts = []
        if res["installed"]:
            parts.append("설치: " + ", ".join(f"{M.MODELS.get(k, {}).get('label', k)} {v}" for k, v in res["installed"]))
            self.changed = True
        if res["kept"]:
            parts.append(f"이미 있음 {len(res['kept'])}개")
        if res["failed"]:
            parts.append("실패: " + " / ".join(f"{t}" for t in res["failed"].values()))
        if not parts and res.get("absent"):
            parts.append(f"목록에 있는 패키지 {len(res['absent'])}개의 파일이 여기에 없습니다")
        self.msg.setText(" · ".join(parts) or "가져올 패키지가 없습니다.")

    def bring_in(self, where: str):
        """Installs from a folder, a zip or an address with the full checks (also what the tests call)."""
        def work(progress, cancelled):
            try:
                return MP.bring_in(where, progress=progress, cancel=cancelled)
            except MP.PackageError as e:
                if e.why == "cancelled":
                    from ..track.analyze import Cancelled
                    raise Cancelled()
                raise RuntimeError(str(e)) from e
        self._run("모델 패키지 확인 중…", work, self._installed)

    def _pick(self, folder: bool):
        if folder:
            p = QFileDialog.getExistingDirectory(self, "모델 패키지 폴더", str(Path.home()))
        else:
            p, _ = QFileDialog.getOpenFileName(self, "모델 패키지 파일", str(Path.home()), "모델 패키지 (*.zip)")
        if p:
            self.bring_in(p)

    def _online(self):
        s = settings()
        a = self.addr.text().strip()
        if getattr(s, "model_registry", "") != a:
            s.model_registry = a
            s.save()
        if a:
            self.bring_in(a)

    def _check(self):
        r = self._row()
        keys = [r["key"]] if r and r["source"] == "package" else [x["key"] for x in MP.listing() if x["source"] == "package"]

        def work(progress, cancelled):
            st, bad = MP.store(), []
            for i, k in enumerate(keys):
                if cancelled():
                    break
                progress(i / max(1, len(keys)), f"{M.MODELS.get(k, {}).get('label', k)} 검사 중")
                for ver in st.versions(k):
                    if not st.verify(k, ver):
                        now = st.set_aside(k, ver)
                        bad.append((k, ver, now))
            return bad

        def done(bad):
            if not bad:
                self.msg.setText(f"설치된 패키지 {len(keys)}개의 파일이 목록과 같습니다.")
                return
            self.changed = True
            self.msg.setText("손상된 버전을 사용에서 뺐습니다 (다시 설치하세요): " + ", ".join(
                f"{M.MODELS.get(k, {}).get('label', k)} {v}" + (f" → 지금 {now}" if now else " → 사용 중인 버전 없음")
                for k, v, now in bad))
        self._run("파일 검사 중…", work, done)

    def _use(self):
        r = self._row()
        if not r:
            return
        v, ok = QInputDialog.getItem(self, "다른 버전 사용", f"{r['label']}: 사용할 버전 (지금 {r['version'] or '없음'})",
                                     r["versions"], max(0, r["versions"].index(r["version"]) if r["version"] in r["versions"] else 0),
                                     False)
        if ok and v and v != r["version"]:
            self.use_version(r["key"], v)

    def use_version(self, key: str, version: str):
        try:
            MP.store().activate(key, version)
            self.changed = True
            self.msg.setText(f"{M.MODELS.get(key, {}).get('label', key)}: 이제 {version}을(를) 씁니다. 이미 분석한 컷은 "
                             "분석한 버전을 그대로 씁니다.")
        except MP.PackageError as e:
            self.msg.setText(str(e))
        self.refresh()

    def _remove(self):
        r = self._row()
        if not r:
            return
        old = [x for x in r["versions"] if x != r["version"]]
        if QMessageBox.question(self, "쓰지 않는 버전 지우기",
                                f"{r['label']}의 {', '.join(old)}을(를) 지울까요?\n\n이 버전으로 분석한 컷이 있으면, 그 컷의 "
                                "매트 갱신·구간 다시 분석을 할 때 이 버전을 다시 설치하거나 컷 전체를 다시 분석해야 합니다 "
                                "(편집·미리보기·렌더는 그대로 됩니다).") != QMessageBox.Yes:
            return
        self.remove_versions(r["key"], old)

    def remove_versions(self, key: str, versions):
        kept = [v for v in versions if not MP.store().remove(key, v)]
        self.msg.setText("지웠습니다." if not kept else f"진행 중인 작업이 쓰고 있어 남겼습니다: {', '.join(kept)}")
        self.refresh()

    def reject(self):
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job.wait(8000)
        super().reject()

    def accept(self):
        if self.job is not None and self.job.isRunning():
            self.job.cancel()
            self.job.wait(8000)
        super().accept()
