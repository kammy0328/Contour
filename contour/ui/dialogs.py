"""Dialogs: settings, copy settings between cuts, merge people, new project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from . import widgets as Wd
from ..core.colour import BUILTIN_STUDIO, manager
from ..core.copyset import CopyOptions, apply_copy, suggest_mapping
from ..core.media import configure_ffmpeg
from ..core.settings import default_data_dir, settings
from .theme import C, font


def _buttons(dlg: QDialog, ok_text: str = "확인") -> QDialogButtonBox:
    bb = QDialogButtonBox()
    ok = bb.addButton(ok_text, QDialogButtonBox.AcceptRole)
    ok.setProperty("kind", "primary")
    cancel = bb.addButton("취소", QDialogButtonBox.RejectRole)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    return bb


def error_box(parent, title: str, err: str):
    first = err.strip().splitlines()[0] if err.strip() else title
    m = QMessageBox(parent)
    m.setIcon(QMessageBox.Warning)
    m.setWindowTitle(title)
    m.setText(first)
    if "\n" in err.strip():
        m.setDetailedText(err)
    m.exec()


class TextPrompt(QDialog):
    def __init__(self, parent, title: str, label: str, text: str = "", ok: str = "확인"):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.addWidget(QLabel(label))
        self.edit = QLineEdit(text)
        self.edit.selectAll()
        lay.addWidget(self.edit)
        lay.addWidget(_buttons(self, ok))
        self.setMinimumWidth(380)

    @staticmethod
    def ask(parent, title, label, text="", ok="확인") -> Optional[str]:
        d = TextPrompt(parent, title, label, text, ok)
        if d.exec() == QDialog.Accepted and d.edit.text().strip():
            return d.edit.text().strip()
        return None


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        s = settings()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.ff = QLineEdit(s.ffmpeg_dir)
        self.ff.setPlaceholderText(r"자동 (PATH 또는 C:\ffmpeg\bin)")
        ffb = Wd.button("찾기", "ghost")
        ffb.clicked.connect(lambda: self._pick_dir(self.ff))
        form.addRow("ffmpeg 폴더", self._row(self.ff, ffb))
        self.data = QLineEdit(s.data_dir)
        self.data.setPlaceholderText(str(default_data_dir()))
        db = Wd.button("찾기", "ghost")
        db.clicked.connect(lambda: self._pick_dir(self.data))
        form.addRow("데이터 폴더", self._row(self.data, db))
        self.ocio = QLineEdit(s.ocio_config)
        self.ocio.setPlaceholderText("내장 ACES Studio 설정 (카메라 로그 포함)")
        ob = Wd.button("찾기", "ghost")
        ob.clicked.connect(self._pick_ocio)
        form.addRow("OCIO 설정", self._row(self.ocio, ob))
        cm = manager(s.ocio_config)
        self.display = QComboBox()
        self.view = QComboBox()
        for d in cm.displays():
            self.display.addItem(d)
        self.display.setCurrentText(s.display or cm.default_display())
        self.display.currentTextChanged.connect(self._fill_views)
        self._fill_views(self.display.currentText())
        self.view.setCurrentText(s.view or cm.default_view(self.display.currentText()))
        form.addRow("디스플레이", self.display)
        form.addRow("뷰 변환", self.view)
        self.proxy = QComboBox()
        for v, t in ((1280, "1280 (빠름)"), (1920, "1920 (권장)"), (2560, "2560 (정밀)")):
            self.proxy.addItem(t, v)
        self.proxy.setCurrentIndex(max(0, self.proxy.findData(s.proxy_long_side)))
        form.addRow("프록시 해상도", self.proxy)
        self.ask_proxy = QCheckBox("영상을 불러올 때마다 프록시 해상도 묻기")
        self.ask_proxy.setChecked(bool(getattr(s, "ask_proxy", True)))
        form.addRow("", self.ask_proxy)
        # auto save
        self.autosave = QCheckBox("자동 저장")
        self.autosave.setChecked(bool(getattr(s, "autosave", True)))
        self.as_min = QSpinBox()
        self.as_min.setRange(1, 60)
        self.as_min.setSuffix(" 분마다")
        self.as_min.setValue(int(getattr(s, "autosave_minutes", 5)))
        self.as_keep = QSpinBox()
        self.as_keep.setRange(1, 50)
        self.as_keep.setSuffix(" 개 보관")
        self.as_keep.setValue(int(getattr(s, "autosave_keep", 10)))
        asw = QWidget()
        asl = QHBoxLayout(asw)
        asl.setContentsMargins(0, 0, 0, 0)
        asl.addWidget(self.autosave)
        asl.addWidget(self.as_min)
        asl.addWidget(self.as_keep)
        asl.addStretch(1)
        self.autosave.toggled.connect(self.as_min.setEnabled)
        self.autosave.toggled.connect(self.as_keep.setEnabled)
        self.as_min.setEnabled(self.autosave.isChecked())
        self.as_keep.setEnabled(self.autosave.isChecked())
        form.addRow("자동 저장", asw)
        self.upd = QCheckBox("시작할 때 새 버전 확인")
        self.upd.setChecked(bool(getattr(s, "auto_update_check", True)))
        self.upd.setToolTip("새 버전이 있으면 시작할 때 알려 줍니다. 설치는 '지금 업데이트'를 눌러야만 합니다.\n"
                            "끄더라도 프로젝트 메뉴 > 업데이트 확인으로 언제든 확인할 수 있습니다.")
        form.addRow("업데이트", self.upd)
        self.open_last = QCheckBox("시작할 때 마지막 프로젝트 바로 열기")
        self.open_last.setChecked(bool(getattr(s, "open_last_on_start", False)))
        self.open_last.setToolTip("끄면(기본) 시작할 때 프로젝트 관리 화면이 먼저 뜹니다")
        form.addRow("시작", self.open_last)
        self.gpu = QComboBox()
        for v, t in (("auto", "자동"), ("cuda", "CUDA (NVIDIA)"), ("dml", "DirectML"), ("cpu", "CPU")):
            self.gpu.addItem(t, v)
        self.gpu.setCurrentIndex(max(0, self.gpu.findData(s.gpu)))
        form.addRow("추론 장치", self.gpu)
        lay.addLayout(form)
        from ..track.models import gpu_report
        rep = Wd.label(gpu_report(), "Faint", 11)
        rep.setWordWrap(True)
        lay.addWidget(rep)
        note = Wd.label("데이터 폴더에는 모델, 학습 데이터, 프리셋이 저장됩니다. 업데이트해도 그대로 유지됩니다. "
                        "자동 저장은 복구용으로 각 프로젝트 폴더의 autosave에 저장되고, 프로젝트 파일은 저장할 때만 바뀝니다.", "Faint", 11)
        note.setWordWrap(True)
        lay.addWidget(note)
        from ..core.settings import APP_NAME, APP_VERSION
        ver = Wd.label(f"{APP_NAME} 버전 {APP_VERSION}", "Muted", 12)
        lay.addWidget(ver)
        lay.addWidget(_buttons(self, "저장"))
        self.setMinimumWidth(560)

    def _row(self, a, b):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(a, 1)
        l.addWidget(b)
        return w

    def _fill_views(self, d):
        cm = manager()
        cur = self.view.currentText()
        self.view.clear()
        for v in cm.views(d):
            self.view.addItem(v)
        if cur:
            self.view.setCurrentText(cur)

    def _pick_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "폴더 선택", edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def _pick_ocio(self):
        f, _ = QFileDialog.getOpenFileName(self, "OCIO 설정 파일", "", "OCIO (*.ocio *.ocioz)")
        if f:
            self.ocio.setText(f)

    def apply(self) -> bool:
        s = settings()
        changed_view = (s.display != self.display.currentText() or s.view != self.view.currentText()
                        or s.ocio_config != self.ocio.text().strip())
        s.ffmpeg_dir = self.ff.text().strip()
        s.data_dir = self.data.text().strip()
        s.ocio_config = self.ocio.text().strip()
        s.display = self.display.currentText()
        s.view = self.view.currentText()
        s.proxy_long_side = int(self.proxy.currentData())
        s.ask_proxy = self.ask_proxy.isChecked()
        s.autosave = self.autosave.isChecked()
        s.autosave_minutes = int(self.as_min.value())
        s.autosave_keep = int(self.as_keep.value())
        s.auto_update_check = self.upd.isChecked()
        s.open_last_on_start = self.open_last.isChecked()
        if s.gpu != self.gpu.currentData():
            s.gpu = self.gpu.currentData()
            from ..track.models import reset_providers
            reset_providers()
        s.save()
        configure_ffmpeg(s.ffmpeg_dir or None)
        manager(s.ocio_config or BUILTIN_STUDIO)
        return changed_view


class CopyDialog(QDialog):
    def __init__(self, parent, project, src_cut, frame: int):
        super().__init__(parent)
        self.setWindowTitle("다른 컷에 설정 복사")
        self.project = project
        self.src = src_cut
        self.frame = frame
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)
        lay.addWidget(Wd.label(f"'{src_cut.name}'의 {frame}프레임 값을 복사합니다. 컷마다 구도가 달라 관절 키와 보호 선 위치는 복사하지 않습니다.",
                               "Muted"))
        opts = QGridLayout()
        self.o = {}
        for i, (k, t, d) in enumerate((("amounts", "부위별 보정값", True), ("widths", "범위 배율", True),
                                       ("roles", "인물 역할·이름", True), ("mode", "워프/분리 방식", True),
                                       ("protect", "보호 설정", True), ("tracking", "추적 설정", False),
                                       ("output", "출력 설정", False))):
            cb = QCheckBox(t)
            cb.setChecked(d)
            self.o[k] = cb
            opts.addWidget(cb, i // 2, i % 2)
        lay.addLayout(opts)
        lay.addWidget(Wd.label("대상 컷과 인물 연결 (생김새로 자동 추정)", "Section"))
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        inner = QWidget()
        self.grid = QVBoxLayout(inner)
        self.grid.setContentsMargins(0, 0, 0, 0)
        sa.setWidget(inner)
        sa.setMinimumHeight(240)
        lay.addWidget(sa, 1)
        self.targets: Dict[str, tuple] = {}
        for c in project.cuts:
            if c.id == src_cut.id:
                continue
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 6, 0, 6)
            cb = QCheckBox(c.name)
            cb.setFont(font(12, 600))
            bl.addWidget(cb)
            maps = {}
            if c.persons and src_cut.persons:
                sug = suggest_mapping(src_cut, c)
                for p in c.persons:
                    row = QHBoxLayout()
                    row.setContentsMargins(22, 0, 0, 0)
                    row.addWidget(QLabel(p.name))
                    row.addStretch(1)
                    combo = QComboBox()
                    combo.addItem("연결 안 함", None)
                    for sp in src_cut.persons:
                        combo.addItem(f"← {sp.name}", sp.id)
                    if sug.get(p.id) is not None:
                        combo.setCurrentIndex(combo.findData(sug[p.id]))
                    row.addWidget(combo)
                    maps[p.id] = combo
                    w = QWidget()
                    w.setLayout(row)
                    bl.addWidget(w)
            else:
                bl.addWidget(Wd.label("    아직 분석 전입니다. 분석이 끝나면 인물 설정이 자동으로 연결됩니다.", "Faint", 11))
            self.grid.addWidget(box)
            self.targets[c.id] = (cb, maps)
        self.grid.addStretch(1)
        lay.addWidget(_buttons(self, "복사"))
        self.setMinimumSize(560, 520)

    def apply(self) -> int:
        opt = CopyOptions(**{k: cb.isChecked() for k, cb in self.o.items()})
        n = 0
        for cid, (cb, maps) in self.targets.items():
            if not cb.isChecked():
                continue
            dst = self.project.cut(cid)
            if maps:
                mapping = {pid: combo.currentData() for pid, combo in maps.items()}
                apply_copy(self.src, dst, mapping, self.frame, opt)
            else:
                apply_copy(self.src, dst, {}, self.frame, opt)
                dst.pending_copy = dict(src=self.src.id, frame=self.frame, opt=opt.__dict__)
            n += 1
        return n


class MergeDialog(QDialog):
    def __init__(self, parent, cut):
        super().__init__(parent)
        self.setWindowTitle("인물 합치기")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.addWidget(Wd.label("같은 사람이 두 명으로 나뉘었을 때 합칩니다. 두 번째 인물의 구간이 첫 번째 인물로 들어갑니다.", "Muted"))
        self.a = QComboBox()
        self.b = QComboBox()
        for p in cut.persons:
            self.a.addItem(p.name, p.id)
            self.b.addItem(p.name, p.id)
        if self.b.count() > 1:
            self.b.setCurrentIndex(1)
        form = QFormLayout()
        form.addRow("남길 인물", self.a)
        form.addRow("합칠 인물", self.b)
        lay.addLayout(form)
        lay.addWidget(_buttons(self, "합치기"))
        self.setMinimumWidth(420)


class IdentityDialog(QDialog):
    """Fix a mix-up of who is who in a group dance: swap two people from here, or split off a wrong part."""

    def __init__(self, parent, cut, pid: int, frame: int, selection=None):
        super().__init__(parent)
        self.setWindowTitle("인물 번호 바로잡기")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(12)
        me = cut.person(pid)
        info = Wd.label(f"'{me.name if me else pid}'의 추적이 다른 사람으로 넘어간 구간을 바로잡습니다. 관절 수정, "
                        "숨김 표시, 인물 분할이 함께 옮겨지고 보정 값은 인물 이름에 그대로 남습니다.", "Muted")
        info.setWordWrap(True)
        lay.addWidget(info)
        others = [p for p in cut.persons if p.id != pid]
        opts = ([("swap", "두 인물 맞바꾸기")] if others else []) + [("split", "새 인물로 떼어 내기")]
        self.mode = Wd.Segmented(opts)
        lay.addWidget(self.mode)
        self.hint = Wd.label("", "Faint", 11)
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.other = QComboBox()
        for p in others:
            self.other.addItem(p.name, p.id)
        self.rng = QComboBox()
        a, b = cut.work_range
        self.rng.addItem(f"이 프레임({frame})부터 끝까지", (int(frame), int(b)))
        if selection:
            s0, s1 = selection
            self.rng.addItem(f"타임라인에서 고른 구간 ({s0}–{s1})", (int(s0), int(s1) + 1))
            self.rng.setCurrentIndex(1)
        form = QFormLayout()
        self.other_lbl = QLabel("맞바꿀 인물")
        form.addRow(self.other_lbl, self.other)
        form.addRow("구간", self.rng)
        lay.addLayout(form)
        self.mode.changed.connect(self._mode)
        self._mode(self.mode.current)
        lay.addWidget(_buttons(self, "바로잡기"))
        self.setMinimumWidth(480)

    def _mode(self, m: str):
        sw = m == "swap"
        self.other.setVisible(sw)
        self.other_lbl.setVisible(sw)
        self.hint.setText("두 사람이 엇갈린 뒤 서로의 번호를 달고 가는 경우. 이 구간에서 두 인물의 추적을 통째로 맞바꿉니다."
                          if sw else "추적이 엉뚱한 사람(추적되지 않던 사람)을 따라간 경우. 이 구간을 새 인물로 떼어 내고, "
                          "같은 사람의 다른 조각이 있으면 '인물 합치기'로 이으세요.")

    def values(self):
        t0, t1 = self.rng.currentData()
        return self.mode.current, self.other.currentData(), t0, t1


class PresetDialog(QDialog):
    """Save the selected person's values as a named preset (choose which groups go in)."""

    def __init__(self, parent, pp, frame: float):
        super().__init__(parent)
        from ..core import presets as PS
        self.setWindowTitle("프리셋 저장")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)
        lay.addWidget(Wd.label("지금 인물의 값을 이름을 붙여 저장합니다. 다른 인물·컷·프로젝트에서 불러올 수 있습니다.", "Muted"))
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.setInsertPolicy(QComboBox.NoInsert)
        for d in PS.list_presets():
            self.name.addItem(d.get("name", ""))
        self.name.setCurrentText("")
        self.name.lineEdit().setPlaceholderText("예: 아이돌_기본_슬림")
        form = QFormLayout()
        form.addRow("이름", self.name)
        lay.addLayout(form)
        lay.addWidget(Wd.label("담을 그룹 (불러올 때 이 그룹만 바뀝니다)", "", 12, 600))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        active = set(PS.active_groups(pp))
        self.checks = {}
        for i, (g, title) in enumerate(PS.GROUP_TITLE.items()):
            cb = QCheckBox(title)
            cb.setChecked(g in active)
            self.checks[g] = cb
            grid.addWidget(cb, i // 3, i % 3)
        lay.addLayout(grid)
        self.points = QCheckBox("얼굴·몸 점 편집도 담기")
        self.points.setChecked(False)
        self.points.setEnabled(bool(pp.sculpt or pp.body_sculpt))
        self.points.setToolTip("점 편집은 인물 고유의 모양을 고친 것이라 기본으로는 빼 둡니다")
        lay.addWidget(self.points)
        lay.addWidget(_buttons(self, "저장"))
        self.setMinimumWidth(440)

    def result_values(self):
        name = self.name.currentText().strip()
        groups = [g for g, cb in self.checks.items() if cb.isChecked()]
        return name, groups, self.points.isChecked()


class PastePickDialog(QDialog):
    """Alt+V: which of the copied cut's values to paste. Only values that were changed are listed."""

    EXTRA = [("skin_detail", "질감 유지", 45.0), ("grain_size", "그레인 크기", 1.0), ("blemish_size", "잡티 크기", 50.0),
             ("skin_scope", "보정 부위", "face,neck"), ("cloth", "옷 그대로", "include"), ("feather", "경계 부드러움", 50.0)]

    def __init__(self, parent, src_cut, frame: int):
        super().__init__(parent)
        from ..core.params import PARAMS, default as param_default
        self.setWindowTitle("골라서 붙여넣기")
        self.items: Dict[str, QCheckBox] = {}
        changed: Dict[str, str] = {}
        for pid, pp in src_cut.params.items():
            for d in PARAMS:
                c = pp.amounts.get(d.key)
                if c is None:
                    continue
                v = c.value(frame)
                if abs(v - param_default(d.key)) > 1e-6 or c.keys:
                    changed[d.key] = d.label
            if abs(pp.strength.value(frame) - 100.0) > 1e-6:
                changed["__strength__"] = "전체 강도"
            for k, lab, dflt in self.EXTRA:
                if getattr(pp, k, dflt) != dflt:
                    changed[k] = lab
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 14)
        lay.setSpacing(8)
        lay.addWidget(Wd.label(f"{src_cut.name}에서 바뀐 보정값", "", 14, 600))
        lay.addWidget(Wd.label("붙여넣을 항목을 고르세요. 인물은 같은 멤버끼리 자동으로 맞춥니다.", "Faint", 11))
        row = QHBoxLayout()
        all_b = Wd.button("전체 선택", "ghost")
        none_b = Wd.button("전체 선택 해제", "ghost")
        all_b.clicked.connect(lambda: [cb.setChecked(True) for cb in self.items.values()])
        none_b.clicked.connect(lambda: [cb.setChecked(False) for cb in self.items.values()])
        row.addWidget(all_b)
        row.addWidget(none_b)
        row.addStretch(1)
        lay.addLayout(row)
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(6)
        for i, (k, lab) in enumerate(changed.items()):
            cb = QCheckBox(lab)
            cb.setChecked(True)
            g.addWidget(cb, i // 3, i % 3)
            self.items[k] = cb
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setWidget(box)
        sc.setMinimumHeight(min(320, 40 + 30 * ((len(changed) + 2) // 3)))
        lay.addWidget(sc, 1)
        lay.addWidget(_buttons(self, "붙여넣기"))
        self.setMinimumWidth(460)

    def selected(self) -> List[str]:
        return [k for k, cb in self.items.items() if cb.isChecked()]


class ProxySizeDialog(QDialog):
    """Proxy resolution for new footage: the preview and every analysis work on it."""

    SIZES = [(1280, "1280", "가장 빠름 · 디스크 적게"), (1920, "1920", "권장"), (2560, "2560", "정밀 · 느림 · 디스크 많이")]

    def __init__(self, parent, current: int):
        super().__init__(parent)
        self.setWindowTitle("프록시 해상도")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 14)
        lay.setSpacing(10)
        lay.addWidget(Wd.label("프록시 해상도", "", 15, 600))
        lay.addWidget(Wd.label("미리보기와 분석이 이 해상도로 동작합니다. 렌더는 항상 원본 해상도입니다.", "Faint", 11))
        self.seg = Wd.Segmented([(str(v), f"{t}  {d}") for v, t, d in self.SIZES], compact=True)
        cur = current if current in [v for v, _, _ in self.SIZES] else 1920
        self.seg.set_current(str(cur))
        lay.addWidget(self.seg)
        self.na = QCheckBox("다음부터 묻지 않기 (설정에서 바꿀 수 있습니다)")
        lay.addWidget(self.na)
        lay.addWidget(_buttons(self, "만들기"))
        self.setMinimumWidth(520)

    def size(self) -> int:
        return int(self.seg.current)

    def dont_ask(self) -> bool:
        return self.na.isChecked()
