"""Start-up password dialog. The app does not open without it."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from ..core import gate
from ..core.settings import APP_VERSION
from .theme import C, font

MAX_TRIES = 5


class GateDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contour")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedWidth(360)
        self.tries = 0
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 26, 28, 22)
        v.setSpacing(10)
        t = QLabel("Contour")
        t.setFont(font(20, 600))
        v.addWidget(t)
        s = QLabel(f"버전 {APP_VERSION} · 비밀번호를 입력하세요")
        s.setObjectName("Muted")
        v.addWidget(s)
        v.addSpacing(6)
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.Password)
        self.edit.setPlaceholderText("비밀번호")
        self.edit.returnPressed.connect(self._try)
        v.addWidget(self.edit)
        self.msg = QLabel("")
        self.msg.setStyleSheet(f"color: {C['warn']};")
        v.addWidget(self.msg)
        h = QHBoxLayout()
        h.addStretch(1)
        q = QPushButton("종료")
        q.setProperty("kind", "ghost")
        q.clicked.connect(self.reject)
        h.addWidget(q)
        ok = QPushButton("열기")
        ok.setDefault(True)
        ok.clicked.connect(self._try)
        h.addWidget(ok)
        v.addLayout(h)
        self.edit.setFocus()

    def _try(self):
        if gate.check(self.edit.text()):
            self.accept()
            return
        self.tries += 1
        self.edit.clear()
        if self.tries >= MAX_TRIES:
            self.reject()
            return
        self.msg.setText(f"비밀번호가 맞지 않습니다 ({self.tries}/{MAX_TRIES})")


def ask() -> bool:
    return GateDialog().exec() == QDialog.Accepted
