"""'구간 다시 분석' (instruction 2026-11-19 §3): pick what to make again for the selected person and range, then
look at the result in use and the new one on the same frames before applying it."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QRadioButton, QSlider,
                               QVBoxLayout)

from . import widgets as Wd
from ..track import rangefix as RF
from ..track import faceparse as FP


class RangeDialog(QDialog):
    """What to analyse again for one person over the selected frames."""

    def __init__(self, parent, cut, pid: int, a: int, b: int):
        super().__init__(parent)
        self.setWindowTitle("구간 다시 분석")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(10)
        p = cut.person(pid)
        info = Wd.label(f"'{p.name if p else pid}'의 {a}~{b - 1} 프레임 ({b - a}f)만 지금 모델과 설정으로 다시 계산합니다. "
                        "결과를 지금 것과 비교한 뒤 적용하며, 취소하면 아무것도 바뀌지 않습니다.", "Muted")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.group = QButtonGroup(self)
        self.buttons = {}
        for key, text in RF.ITEMS:
            ready = key in RF.READY
            rb = QRadioButton(text + ("" if ready else "  (준비 중)"))
            rb.setEnabled(ready)
            self.group.addButton(rb)
            self.buttons[key] = rb
            lay.addWidget(rb)
        self.buttons["face"].setChecked(True)
        hint = Wd.label("얼굴·목: 이 인물의 얼굴·목 마스크와 피부색 기준. 마스크를 고르게 하려고 앞뒤 몇 프레임을 함께 "
                        "읽지만 적용은 고른 구간에만 합니다.", "Faint", 11)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        bb = QDialogButtonBox()
        ok = bb.addButton("다시 계산", QDialogButtonBox.AcceptRole)
        ok.setProperty("kind", "primary")
        bb.addButton("취소", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.setMinimumWidth(460)

    def item(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), "face")


def _outline(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8)
    return (cv2.dilate(m8, np.ones((3, 3), np.uint8)) - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0


def compose(rgb: np.ndarray, old, new, mode: str) -> np.ndarray:
    """The frame with the masks in use (red: skin, orange: neck) and the new ones (green: skin, cyan: neck)."""
    img = rgb.astype(np.float32)
    layers = []
    if mode in ("old", "both") and old[0] is not None:
        layers.append((old, ((235, 70, 70), (245, 160, 60)), mode == "old"))
    if mode in ("new", "both") and new[0] is not None:
        layers.append((new, ((80, 220, 120), (70, 200, 230)), mode == "new"))
    for (sk, nk), (cs, cn), fill in layers:
        for m, c in ((sk, cs), (nk, cn)):
            if m is None:
                continue
            if fill:
                img[m] = 0.7 * img[m] + 0.3 * np.array(c, np.float32)
            img[_outline(m)] = c
    return np.clip(img, 0, 255).astype(np.uint8)


class CompareDialog(QDialog):
    """The masks in use and the new ones on the same frame; apply or cancel."""

    def __init__(self, parent, tracker, cut, cand: dict, cmp: dict):
        super().__init__(parent)
        self.setWindowTitle("구간 다시 분석: 비교")
        self.tr, self.cand, self.cmp = tracker, cand, cmp
        self.cur = RF.current(tracker, cand["pid"], cand["a"], cand["b"])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 12)
        lay.setSpacing(8)
        ms = cmp.get("mean_skin")
        head = Wd.label(f"{cand['a']}~{cand['b'] - 1} 프레임 · 바뀐 프레임 {cmp['changed']}/{cmp['frames']}"
                        + (f" · 피부 영역 평균 겹침 {ms:.2f}" if ms is not None else "")
                        + (f" · 모델 {FP.label(cand['model'])}" if cand.get("model") else ""), "", 12, 600)
        lay.addWidget(head)
        self.view = Wd.Segmented([("both", "겹쳐 보기"), ("old", "지금 결과"), ("new", "새 결과")], current="both")
        self.view.changed.connect(lambda _: self._show())
        lay.addWidget(self.view)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignCenter)
        self.pic.setMinimumSize(560, 360)
        lay.addWidget(self.pic, 1)
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(cand["a"], cand["b"] - 1)
        worst = min(cmp["rows"], key=lambda r: 2.0 if r["skin"] is None else r["skin"])["t"] if cmp["rows"] else cand["a"]
        self.slider.setValue(int(worst))
        self.slider.valueChanged.connect(lambda _: self._show())
        self.frame_lbl = Wd.label("", "Faint", 11)
        row.addWidget(self.slider, 1)
        row.addWidget(self.frame_lbl)
        lay.addLayout(row)
        legend = Wd.label("빨강·주황 테두리: 지금 쓰는 피부·목  /  초록·하늘 테두리: 새로 계산한 피부·목", "Faint", 11)
        lay.addWidget(legend)
        eff = Wd.label(RF.effects(cut, cand), "Muted", 11)
        eff.setWordWrap(True)
        lay.addWidget(eff)
        bb = QDialogButtonBox()
        ok = bb.addButton("적용", QDialogButtonBox.AcceptRole)
        ok.setProperty("kind", "primary")
        bb.addButton("취소", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(760, 640)
        self._show()

    def image(self, t: int, mode: str) -> np.ndarray:
        px = self.tr.proxy
        rgb = np.asarray(px.frame(int(t)))
        H, W = rgb.shape[:2]
        k = int(t) - self.cand["a"]
        new = RF._image_maps(self.cand, k, (H, W))
        old = RF._image_maps(self.cur, k, (H, W)) if self.cur is not None and k < len(self.cur["have"]) else (None, None)
        out = compose(rgb, old, new, mode)
        # close on the face: the union of both masks, with room around it
        ms = [m for m in (old + new) if m is not None and m.any()]
        if ms:
            u = np.any(ms, 0)
            ys, xs = np.nonzero(u)
            cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
            r = 0.85 * max(xs.max() - xs.min(), ys.max() - ys.min(), 40)
            x0, x1 = int(max(0, cx - r)), int(min(W, cx + r))
            y0, y1 = int(max(0, cy - r)), int(min(H, cy + r))
            out = out[y0:y1, x0:x1]
        return out

    def _show(self):
        t = self.slider.value()
        img = np.ascontiguousarray(self.image(t, self.view.current))
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.pic.setPixmap(QPixmap.fromImage(q).scaled(self.pic.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        r = next((r for r in self.cmp["rows"] if r["t"] == t), None)
        s = "" if r is None or r["skin"] is None else f" · 겹침 {r['skin']:.2f}"
        self.frame_lbl.setText(f"프레임 {t}{s}")
