"""Whole-project work: 전체 분석 and 전체 피부 보정.

AnalyzeAllDialog   what will be analysed, how long it may take, an optional skin basic adjustment for each finished cut
SkinAllDialog      the skin basic adjustment for every analysed cut (and whether hand-set people are left alone)
BatchProgress      the window that covers the app while it runs: progress bar, n / N cuts, time left and end time,
                   일시정지 / 계속 / 취소. The rest of the app takes no input meanwhile (window modal).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from . import widgets as Wd

DEFAULT_SPF = 2.0        # seconds per frame of a whole-cut analysis before this PC has measured its own


def fmt_dur(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, m = divmod(sec // 60, 60)
    if h:
        return f"{h}시간 {m}분"
    if m:
        return f"{m}분"
    return f"{sec}초"


def fmt_end(sec: float) -> str:
    t = time.localtime(time.time() + max(0.0, sec))
    today = time.localtime()
    day = "" if t.tm_yday == today.tm_yday else "내일 "
    return day + time.strftime("%H:%M", t)


class SkinBasicPicker(QWidget):
    """Level (light / std / strong) and 보호 / 제거 for the mouth lines, nasolabial folds and spots."""
    def __init__(self, level: str = "std", remove=()):
        super().__init__()
        from ..core.params import SKIN_LEVELS, SKIN_REMOVE
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.level = Wd.Segmented(SKIN_LEVELS, current=level)
        v.addWidget(Wd.FieldRow("보정 강도", self.level, "표준은 엔진의 기준 보정 정도입니다"))
        self.sw: Dict[str, Wd.Segmented] = {}
        rm = set(remove)
        for sw, key, lab in SKIN_REMOVE:
            seg = Wd.Segmented([("keep", "보호"), ("remove", "제거")], current="remove" if sw in rm else "keep",
                               compact=True)
            seg.setFixedWidth(132)
            v.addWidget(Wd.FieldRow(lab, seg, ""))
            self.sw[sw] = seg

    def value(self):
        return self.level.current or "std", sorted(k for k, s in self.sw.items() if s.current == "remove")


class _Dialog(QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(20, 18, 20, 16)
        self.v.setSpacing(10)
        self.setMinimumWidth(420)

    def buttons(self, ok_text: str):
        h = QHBoxLayout()
        h.addStretch(1)
        no = QPushButton("취소")
        no.clicked.connect(self.reject)
        ok = QPushButton(ok_text)
        ok.setObjectName("Primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        h.addWidget(no)
        h.addWidget(ok)
        self.v.addSpacing(6)
        self.v.addLayout(h)


class AnalyzeAllDialog(_Dialog):
    def __init__(self, parent, n_cuts: int, frames: int, n_done: int, spf: float, title: str = "전체 분석"):
        super().__init__(parent, title)
        est = frames * (spf or DEFAULT_SPF)
        self.v.addWidget(Wd.label(f"분석할 컷 {n_cuts}개 · {frames:,}프레임", "", 14))
        sub = f"예상 소요 약 {fmt_dur(est)} · 끝나는 시각 약 {fmt_end(est)}"
        if not spf:
            sub += "\n(이 PC에서 처음 재는 속도라 첫 컷이 끝나면 다시 계산합니다)"
        if n_done:
            sub += f"\n이미 분석한 {n_done}컷은 그대로 둡니다."
        lb = Wd.label(sub, "Faint", 12)
        lb.setWordWrap(True)
        self.v.addWidget(lb)
        self.v.addWidget(Wd.hline())
        self.skin_on = QCheckBox("분석이 끝난 컷에 피부 기본 조정 적용")
        self.skin_on.setToolTip("컷마다 분석이 끝나는 대로 보정 대상 인물에 아래 값을 넣습니다")
        self.v.addWidget(self.skin_on)
        self.picker = SkinBasicPicker()
        self.picker.setEnabled(False)
        self.skin_on.toggled.connect(self.picker.setEnabled)
        self.v.addWidget(self.picker)
        note = Wd.label("분석 중에는 다른 조작을 할 수 없습니다. 일시정지, 계속, 취소는 진행 창에서 합니다.", "Faint", 11)
        note.setWordWrap(True)
        self.v.addWidget(note)
        self.buttons("분석 시작")

    def skin(self):
        return self.picker.value() if self.skin_on.isChecked() else None


class SkinAllDialog(_Dialog):
    def __init__(self, parent, n_cuts: int, n_people: int, n_hand: int, n_pre: int):
        super().__init__(parent, "전체 피부 보정")
        self.v.addWidget(Wd.label(f"분석된 컷 {n_cuts}개 · 보정 대상 {n_people}명", "", 14))
        if n_pre:
            lb = Wd.label(f"분석 전인 {n_pre}컷은 건너뜁니다.", "Faint", 12)
            self.v.addWidget(lb)
        self.picker = SkinBasicPicker()
        self.v.addWidget(self.picker)
        self.skip_hand = QCheckBox(f"피부 값을 손으로 조정한 인물은 건너뛰기 ({n_hand}명)")
        self.skip_hand.setChecked(True)
        self.skip_hand.setToolTip("세부 조정에서 직접 맞춘 인물은 그대로 둡니다.\n"
                                  "기본 조정으로만 맞춘 인물과 아직 값이 없는 인물에는 적용합니다.")
        self.v.addWidget(self.skip_hand)
        note = Wd.label("키프레임이 있는 피부 항목은 키프레임을 지우고 컷 전체 값으로 넣습니다. 실행 취소(Ctrl+Z)는 "
                        "지금 열린 컷에만 됩니다.", "Faint", 11)
        note.setWordWrap(True)
        self.v.addWidget(note)
        self.buttons("적용")

    def values(self):
        lv, rm = self.picker.value()
        return lv, rm, self.skip_hand.isChecked()


class BatchProgress(QDialog):
    """Covers the app while a whole-project job runs. frames: per cut id (time estimate by frames)."""
    pauseToggled = Signal(bool)
    cancelNow = Signal()
    cancelAfter = Signal()

    def __init__(self, parent, title: str, order: List[str], names: Dict[str, str], frames: Dict[str, int],
                 spf: float = 0.0, can_pause: bool = True):
        super().__init__(parent, Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(480)
        self.order, self.names, self.frames = list(order), names, frames
        self.spf0 = spf or DEFAULT_SPF
        self.done_ids: List[str] = []
        self.cur: Optional[str] = None
        self.cur_f = 0.0
        self.msg = ""
        self.t_active = 0.0           # seconds spent running (pauses not counted)
        self._t_last = time.time()
        self.paused = False
        self.meas_frames = 0          # frames of cuts really analysed (not 'no one in it' quick checks)
        self.meas_sec = 0.0
        self._cut_t0 = 0.0
        self.finished_ok = False
        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(8)
        self.head = Wd.label(title, "", 15)
        v.addWidget(self.head)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        v.addWidget(self.bar)
        self.l_cut = Wd.label("", "", 12)
        self.l_time = Wd.label("", "Faint", 12)
        self.l_msg = Wd.label("", "Faint", 11)
        for w in (self.l_cut, self.l_time, self.l_msg):
            w.setWordWrap(True)
            v.addWidget(w)
        h = QHBoxLayout()
        h.addStretch(1)
        self.b_pause = QPushButton("일시정지")
        self.b_pause.clicked.connect(self._toggle_pause)
        self.b_pause.setVisible(can_pause)
        self.b_cancel = QPushButton("취소")
        self.b_cancel.clicked.connect(self._cancel)
        h.addWidget(self.b_pause)
        h.addWidget(self.b_cancel)
        v.addSpacing(4)
        v.addLayout(h)
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update)
        self._tick.start()
        self._update()

    # ------------------------------------------------------------ fed by the runner
    def started(self, cid: str):
        self.cur, self.cur_f, self.msg = cid, 0.0, "시작"
        self._cut_t0 = self.t_active
        self._update()

    def progress(self, cid: str, f: float, msg: str):
        if cid != self.cur:
            self.started(cid)
        self.cur_f, self.msg = max(0.0, min(1.0, f)), msg
        self._update()

    def cut_done(self, cid: str, analysed: bool):
        if cid not in self.done_ids:
            self.done_ids.append(cid)
        if analysed:
            self.meas_frames += int(self.frames.get(cid, 0))
            self.meas_sec += max(0.0, self.t_active - self._cut_t0)
        self.cur, self.cur_f = None, 0.0
        self._update()

    def drop(self, cid: str):
        """A cut that will not run (cancelled before it started)."""
        if cid in self.order and cid not in self.done_ids:
            self.order.remove(cid)
        self._update()

    # ------------------------------------------------------------ estimate
    def spf(self) -> float:
        if self.meas_frames >= 24 and self.meas_sec > 0:
            return self.meas_sec / self.meas_frames
        return self.spf0

    def _update(self):
        now = time.time()
        if not self.paused:
            self.t_active += now - self._t_last
        self._t_last = now
        n = max(1, len(self.order))
        fr_all = sum(int(self.frames.get(c, 0)) for c in self.order) or 1
        fr_done = sum(int(self.frames.get(c, 0)) for c in self.done_ids)
        fr_cur = int(self.frames.get(self.cur, 0)) * self.cur_f if self.cur else 0
        f = min(1.0, (fr_done + fr_cur) / fr_all)
        self.bar.setValue(int(f * 1000))
        k = len(self.done_ids) + (1 if self.cur else 0)
        cur = f" · {self.names.get(self.cur, '')} {self.cur_f * 100:.0f}%" if self.cur else ""
        self.l_cut.setText(f"{min(k, n)} / {n}컷{cur}   ({f * 100:.0f}%)")
        left = max(0.0, fr_all - fr_done - fr_cur) * self.spf()
        if self.paused:
            self.l_time.setText(f"일시정지됨 · 남은 시간 약 {fmt_dur(left)} (계속을 누른 때부터)")
        else:
            self.l_time.setText(f"지난 시간 {fmt_dur(self.t_active)} · 남은 시간 약 {fmt_dur(left)} · "
                                f"끝나는 시각 약 {fmt_end(left)}")
        self.l_msg.setText(self.msg or "")

    # ------------------------------------------------------------ buttons
    def _toggle_pause(self):
        self._update()
        self.paused = not self.paused
        self.b_pause.setText("계속" if self.paused else "일시정지")
        self.pauseToggled.emit(self.paused)
        self._update()

    def set_paused(self, on: bool):
        """The runner could not pause (nothing running yet): back to running."""
        self.paused = on
        self.b_pause.setText("계속" if on else "일시정지")
        self._update()

    def _cancel(self):
        m = QMessageBox(self)
        m.setWindowTitle(self.windowTitle())
        m.setIcon(QMessageBox.Question)
        m.setText("작업을 멈출까요?")
        m.setInformativeText("끝난 컷은 모두 분석 완료로 남습니다.\n바로 멈추면 지금 컷의 진행 중인 분석은 버리고 "
                             "그 컷은 분석 전으로 둡니다.")
        b_after = m.addButton("지금 컷까지 마치고 멈춤", QMessageBox.AcceptRole)
        b_now = m.addButton("바로 멈춤", QMessageBox.DestructiveRole)
        m.addButton("계속 진행", QMessageBox.RejectRole)
        m.exec()
        c = m.clickedButton()
        if c is b_now:
            self.b_cancel.setEnabled(False)
            self.b_pause.setEnabled(False)
            self.msg = "멈추는 중"
            self.cancelNow.emit()
        elif c is b_after:
            self.b_cancel.setEnabled(False)
            self.b_pause.setEnabled(False)
            self.order = self.done_ids + ([self.cur] if self.cur else [])
            self.msg = "지금 컷이 끝나면 멈춥니다"
            if self.paused:
                self.set_paused(False)
                self.pauseToggled.emit(False)
            self.cancelAfter.emit()
        self._update()

    def finish(self):
        self.finished_ok = True
        self._tick.stop()
        self.accept()

    def reject(self):          # Esc / close: ask like 취소
        if self.finished_ok:
            super().reject()
        else:
            self._cancel()

    def closeEvent(self, e):
        if self.finished_ok:
            super().closeEvent(e)
        else:
            e.ignore()
            self._cancel()
