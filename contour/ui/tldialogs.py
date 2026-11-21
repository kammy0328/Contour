"""Dialogs for edited timelines: import (editing file + baked video), batch apply, export."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from ..track import status as ST
from .theme import C, font

EDIT_FILTER = "편집 파일 (*.xml *.fcpxml *.edl);;모든 파일 (*)"
VIDEO_FILTER = "베이크 영상 (*.mov *.mp4 *.mxf *.m4v *.avi *.mkv);;모든 파일 (*)"


def _bb(dlg, ok_text):
    bb = QDialogButtonBox()
    ok = bb.addButton(ok_text, QDialogButtonBox.AcceptRole)
    ok.setProperty("kind", "primary")
    bb.addButton("취소", QDialogButtonBox.RejectRole)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    return bb


def _path_row(edit: QLineEdit, title: str, flt: str, parent, folder=False) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    b = QPushButton("찾기…")

    def pick():
        start = edit.text() or ""
        if folder:
            p = QFileDialog.getExistingDirectory(parent, title, start)
        else:
            p, _ = QFileDialog.getOpenFileName(parent, title, str(Path(start).parent) if start else "", flt)
        if p:
            edit.setText(p)

    b.clicked.connect(pick)
    h.addWidget(b)
    return w


class TimelineImportDialog(QDialog):
    def __init__(self, parent, edit_path: str = "", baked_path: str = "", has_timeline: bool = False):
        super().__init__(parent)
        self.setWindowTitle("타임라인 불러오기")
        self.setMinimumWidth(620)
        v = QVBoxLayout(self)
        t = QLabel("편집 파일(XML·FCPXML·EDL)과 그 시퀀스를 통째로 뽑은 베이크 영상을 고르세요.\n"
                   "샷마다 컷이 하나씩 생기고, 영상과 편집 경계가 맞는지 자동으로 확인합니다.")
        t.setWordWrap(True)
        t.setObjectName("Muted")
        v.addWidget(t)
        f = QFormLayout()
        self.edit = QLineEdit(edit_path)
        self.baked = QLineEdit(baked_path)
        f.addRow("편집 파일", _path_row(self.edit, "편집 파일", EDIT_FILTER, self))
        f.addRow("베이크 영상", _path_row(self.baked, "베이크 영상", VIDEO_FILTER, self))
        self.after = QLineEdit("")          # (no hand-result input here any more; kept for values())
        v.addLayout(f)
        self.replace = QCheckBox("이미 있는 타임라인 컷을 바꾸기")
        self.replace.setChecked(True)
        self.replace.setVisible(has_timeline)
        v.addWidget(self.replace)
        self.auto = QCheckBox("불러온 뒤 바로 일괄 분석 시작 (인물 없는 컷은 빠르게 건너뛰고, 나머지는 최고 정밀)")
        self.auto.setChecked(True)
        v.addWidget(self.auto)
        v.addWidget(_bb(self, "불러오기"))

    def values(self):
        return (self.edit.text().strip(), self.baked.text().strip(), self.after.text().strip(),
                self.replace.isChecked(), self.auto.isChecked())


class TimelineExportDialog(QDialog):
    def __init__(self, parent, project, selected: Optional[List[str]] = None, last: Optional[dict] = None):
        from ..warp.tlrender import FORMATS, TimelineExport
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("타임라인 내보내기")
        self.setMinimumWidth(640)
        exp = TimelineExport.from_dict(last or {})
        cuts = project.timeline_cuts()
        v = QVBoxLayout(self)
        # mode
        v.addWidget(self._h("출력 방식"))
        self.modes = QButtonGroup(self)
        for k, text, tip in (("full", "전체 길이 한 파일", "베이크 영상과 같은 길이·타임코드·오디오"),
                             ("each", "컷별 개별 파일", "컷마다 한 파일, 컷 이름과 컷의 타임코드"),
                             ("replace", "리터칭 컷만 + 교체용 XML", "편집 프로그램에서 원본(V1) 위 V2에 그대로 올라감")):
            rb = QRadioButton(f"{text}  ·  {tip}")
            rb.setProperty("mode", k)
            self.modes.addButton(rb)
            v.addWidget(rb)
            if k == exp.mode:
                rb.setChecked(True)
        # format
        v.addWidget(self._h("포맷"))
        g = QGridLayout()
        self.fmt = QComboBox()
        for k, t in FORMATS:
            self.fmt.addItem(t, k)
        self.fmt.setCurrentIndex(max(0, [k for k, _ in FORMATS].index(exp.format) if exp.format in dict(FORMATS) else 0))
        g.addWidget(QLabel("코덱"), 0, 0)
        g.addWidget(self.fmt, 0, 1)
        self.enc = QComboBox()
        import sys as _sys
        if _sys.platform == "darwin":
            # a Mac has no NVENC; its hardware encoder is offered by name only (the choice is the person's until
            # its pictures were compared with x265's on a real Mac)
            encs = (("auto", "자동 (x265)"), ("videotoolbox", "VideoToolbox (하드웨어, 빠름)"),
                    ("x265", "x265 (CPU, 같은 용량에서 더 깨끗함)"))
        else:
            encs = (("auto", "자동 (NVENC 있으면 사용)"), ("nvenc", "NVENC (GPU, 빠름)"),
                    ("x265", "x265 (CPU, 같은 용량에서 더 깨끗함)"))
        for k, t in encs:
            self.enc.addItem(t, k)
        self.enc.setCurrentIndex(max(0, self.enc.findData(exp.encoder)))
        self.q = QComboBox()
        for k, t in (("normal", "보통"), ("high", "고화질"), ("max", "최고 화질")):
            self.q.addItem(t, k)
        self.q.setCurrentIndex(["normal", "high", "max"].index(exp.quality) if exp.quality in ("normal", "high", "max") else 1)
        self.enc_lbl, self.q_lbl = QLabel("H.265 인코더"), QLabel("H.265 화질")
        g.addWidget(self.enc_lbl, 1, 0)
        g.addWidget(self.enc, 1, 1)
        g.addWidget(self.q_lbl, 2, 0)
        g.addWidget(self.q, 2, 1)
        v.addLayout(g)
        # include
        v.addWidget(self._h("내보낼 컷"))
        cnt = {}
        for c in cuts:
            cnt[ST.effective(c)] = cnt.get(ST.effective(c), 0) + 1
        self.inc = {}
        row = QGridLayout()
        defaults = dict(good=True, bad=True, small=True, none=True, pending=True, skip=False)
        for i, k in enumerate(("good", "bad", "small", "none", "pending", "skip")):
            cb = QCheckBox(f"{ST.LABEL[k]} {cnt.get(k, 0)}")
            cb.setChecked(bool(exp.include.get(k, defaults[k])))
            cb.setEnabled(cnt.get(k, 0) > 0)
            self.inc[k] = cb
            row.addWidget(cb, i // 3, i % 3)
        v.addLayout(row)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setObjectName("Muted")
        v.addWidget(self.note)
        self.sel_only = QCheckBox(f"선택한 컷만 ({len(selected or [])}개)")
        self.sel_only.setVisible(bool(selected))
        self.sel_only.setChecked(bool(selected))
        self.selected = list(selected or [])
        v.addWidget(self.sel_only)
        # where
        v.addWidget(self._h("저장"))
        f = QFormLayout()
        self.dir = QLineEdit(exp.out_dir or str(Path(project.path).parent / "DIAP_Renders"))
        f.addRow("폴더", _path_row(self.dir, "출력 폴더", "", self, folder=True))
        self.name = QLineEdit(exp.name or (project.timeline or {}).get("name") or project.name)
        f.addRow("이름", self.name)
        self.audio = QCheckBox("오디오 포함 (베이크 영상의 오디오)")
        self.audio.setChecked(exp.audio)
        f.addRow("", self.audio)
        v.addLayout(f)
        self.sum = QLabel("")
        self.sum.setObjectName("Faint")
        v.addWidget(self.sum)
        v.addWidget(_bb(self, "렌더에 추가"))
        for w in (self.fmt,):
            w.currentIndexChanged.connect(self._upd)
        for b in self.modes.buttons():
            b.toggled.connect(self._upd)
        for cb in self.inc.values():
            cb.toggled.connect(self._upd)
        self.sel_only.toggled.connect(self._upd)
        self._upd()

    def _h(self, t):
        l = QLabel(t)
        l.setFont(font(12, 600))
        l.setContentsMargins(0, 8, 0, 0)
        return l

    def export(self):
        from ..warp.tlrender import TimelineExport
        mode = next((b.property("mode") for b in self.modes.buttons() if b.isChecked()), "full")
        return TimelineExport(mode=mode, format=self.fmt.currentData(), encoder=self.enc.currentData(),
                              quality=self.q.currentData(), include={k: cb.isChecked() for k, cb in self.inc.items()},
                              out_dir=self.dir.text().strip(), name=self.name.text().strip(), audio=self.audio.isChecked(),
                              selected=self.selected if self.sel_only.isChecked() else [])

    def _upd(self, *_):
        from ..warp.tlrender import plan
        exp = self.export()
        hevc = exp.format == "hevc_mp4"
        for w in (self.enc, self.q, self.enc_lbl, self.q_lbl):
            w.setVisible(hevc)
        if exp.mode == "full":
            self.note.setText("빼는 컷은 " + ("투명(알파 0)으로 들어갑니다." if exp.format == "prores_4444_xq"
                                            else "원본 그대로 들어갑니다 (알파가 없는 포맷). 길이와 타임코드는 베이크 영상과 같습니다."))
        elif exp.mode == "each":
            self.note.setText("빼는 컷은 파일을 만들지 않습니다. 리터칭할 수 없는 컷(인물 없음 등)은 원본 그대로 만듭니다.")
        else:
            self.note.setText("리터칭한 컷만 파일로 만들고, 편집 프로그램에서 베이크 영상(V1) 위 V2의 제자리에 놓이는 XML을 함께 만듭니다.")
        rows = plan(self.project, exp)
        n = {}
        for r in rows:
            n[r["action"]] = n.get(r["action"], 0) + 1
        self.sum.setText(f"리터칭 {n.get('retouch', 0)} · 원본 {n.get('original', 0)} · 투명 {n.get('alpha', 0)} · "
                         f"제외 {n.get('skip', 0)}")


class BatchApplyDialog(QDialog):
    """A preset onto the beauty targets of many cuts (and, optionally, every person whose face is seen)."""

    def __init__(self, parent, n_cuts: int):
        from ..core import presets as P
        super().__init__(parent)
        self.setWindowTitle("일괄 적용")
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        v.addWidget(QLabel(f"{n_cuts}개 컷에 적용합니다 (분석된 컷만)."))
        self.preset = QComboBox()
        self.presets = P.list_presets()
        for d in self.presets:
            self.preset.addItem(f"{d['name']}  ({P.summary(d)})", d["name"])
        i = next((k for k, d in enumerate(self.presets) if d["name"] == "학습된_스타일"), None)
        if i is None:
            i = 0
        self.preset.setCurrentIndex(i)
        v.addWidget(self.preset)
        self.all_faces = QCheckBox("얼굴이 잘 보이는 인물은 모두 보정 (군무)")
        self.all_faces.setChecked(True)
        v.addWidget(self.all_faces)
        v.addWidget(_bb(self, "적용"))

    def values(self):
        d = next((x for x in self.presets if x["name"] == self.preset.currentData()), None)
        return d, self.all_faces.isChecked()
