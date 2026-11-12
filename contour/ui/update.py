"""Update check, offer and install. Nothing changes until the user chooses "지금 업데이트"."""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QTextBrowser, QVBoxLayout)

from ..core import updater as U
from ..core.settings import APP_VERSION, settings
from .theme import font


class _Check(QThread):
    found = Signal(object)
    failed = Signal(str)

    def __init__(self, parent, ignore_skip: bool):
        super().__init__(parent)
        self.ignore_skip = ignore_skip

    def run(self):
        try:
            r = U.latest()
            if r is not None and U.is_newer(r.version) and (self.ignore_skip or settings().skip_version != r.version):
                self.found.emit(r)
            else:
                self.found.emit(None)
        except Exception as e:
            self.failed.emit(str(e))


class _Download(QThread):
    progressed = Signal(float, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, parent, rel):
        super().__init__(parent)
        self.rel = rel
        self.stop = False

    def run(self):
        try:
            p = U.download(self.rel, lambda f, t: self.progressed.emit(f, t), lambda: self.stop)
            self.done.emit(str(p))
        except Exception as e:
            self.failed.emit(str(e))


class UpdateDialog(QDialog):
    def __init__(self, main, rel):
        super().__init__(main)
        self.main, self.rel = main, rel
        self.dl = None
        self.setWindowTitle("업데이트")
        self.resize(560, 460)
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 20, 22, 18)
        v.setSpacing(10)
        t = QLabel(f"DIAP {rel.version}")
        t.setFont(font(18, 600))
        v.addWidget(t)
        s = QLabel(f"지금 버전 {APP_VERSION} · 설치 파일 {rel.size / 1e6:.0f} MB · 프로젝트와 학습 데이터는 그대로 남습니다")
        s.setObjectName("Muted")
        s.setWordWrap(True)
        v.addWidget(s)
        notes = QTextBrowser()
        notes.setMarkdown(rel.notes or "변경 내역 없음")
        notes.setOpenExternalLinks(True)
        v.addWidget(notes, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.msg = QLabel("")
        self.msg.setObjectName("Faint")
        v.addWidget(self.msg)
        h = QHBoxLayout()
        self.skip = QPushButton("이 버전 건너뛰기")
        self.skip.setProperty("kind", "ghost")
        self.skip.clicked.connect(self._skip)
        h.addWidget(self.skip)
        h.addStretch(1)
        self.later = QPushButton("나중에")
        self.later.setProperty("kind", "ghost")
        self.later.clicked.connect(self.reject)
        h.addWidget(self.later)
        self.go = QPushButton("지금 업데이트")
        self.go.setDefault(True)
        self.go.clicked.connect(self._go)
        h.addWidget(self.go)
        v.addLayout(h)

    def _skip(self):
        s = settings()
        s.skip_version = self.rel.version
        s.save()
        self.reject()

    def _go(self):
        if U.installed_dir() is None:
            QMessageBox.information(self, "업데이트", "설치판이 아닌 소스 실행이라 자동 설치를 할 수 없습니다. "
                                    "릴리스 페이지에서 설치 파일을 받아 설치하세요.")
            return
        if self.main.queue_page.job is not None or self.main.batch.busy:
            QMessageBox.information(self, "업데이트", "렌더나 일괄 분석이 끝난 뒤에 업데이트하세요.")
            return
        self.go.setEnabled(False)
        self.skip.setEnabled(False)
        self.bar.setVisible(True)
        self.dl = _Download(self, self.rel)
        self.dl.progressed.connect(lambda f, t: (self.bar.setValue(int(f * 1000)), self.msg.setText(t)))
        self.dl.done.connect(self._install)
        self.dl.failed.connect(self._fail)
        self.dl.start()

    def reject(self):
        if self.dl is not None and self.dl.isRunning():
            self.dl.stop = True
            self.dl.wait(5000)
        super().reject()

    def _fail(self, e):
        self.msg.setText(f"업데이트를 받지 못했습니다: {e}")
        self.go.setEnabled(True)
        self.skip.setEnabled(True)

    def _install(self, path):
        self.msg.setText("설치 파일 확인 완료 (SHA-256). 저장하고 DIAP을 닫은 뒤 설치합니다. 설치 중 Windows 권한 확인 창이 뜨면 '예'를 누르세요.")
        self.main.save()
        U.launch_installer(__import__("pathlib").Path(path))
        self.accept()
        self.main.close()


def check(main, manual: bool = False):
    """Background check; offers the update when there is one. manual: also says when there is none."""
    th = _Check(main, ignore_skip=manual)

    def found(rel):
        if rel is not None:
            UpdateDialog(main, rel).exec()
        elif manual:
            main.toast(f"최신 버전입니다 ({APP_VERSION})", "ok")

    def failed(e):
        if manual:
            main.toast(f"업데이트를 확인하지 못했습니다: {e}", "warn")

    th.found.connect(found)
    th.failed.connect(failed)
    main._update_thread = th
    th.start()
