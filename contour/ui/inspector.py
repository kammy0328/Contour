"""Right panel: 피부 · 몸매 · 추적 · 보호 · 출력."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QScrollArea,
                               QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from . import icons as I
from . import widgets as Wd
from ..core.project import OUTPUT_FORMATS
from ..qc.checks import KIND_TEXT
from ..track import skeleton as S
from .theme import C, font, qc


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    inner.setObjectName("Panel")
    inner.setAttribute(Qt.WA_StyledBackground, True)
    return sa


class _Tab(QWidget):
    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(16, 6, 16, 16)
        self.lay.setSpacing(4)

    def add(self, w):
        self.lay.addWidget(w)
        return w


# =========================================================================== 보정

class ReshapeTab(_Tab):
    """피부 (part="skin": basic and detail adjustment of the skin and colour retouch) or 몸매 (part="body": every
    shape change). Both start with the people of the cut and the selected person's role and presets."""
    _expanded = {"skin": True, "color": False}  # remembered while the app runs
    _view = "basic"                             # skin tab: basic or detail, remembered while the app runs
    SKIN_GROUPS = ("skin", "color")

    def __init__(self, ctl, part: str = "body"):
        super().__init__(ctl)
        from ..core.params import GROUPS, PARAMS, WIDTH_KEYS, PARAM
        self.part = part
        self.chips_box = QWidget()
        self.chips = Wd.FlowLayout(self.chips_box)
        self.add(Wd.Section("인물"))
        self.add(self.chips_box)
        self.no_people = Wd.label("분석을 실행하면 인물이 여기에 나타납니다.", "Faint")
        self.add(self.no_people)
        # selected person
        self.person_box = QWidget()
        pl = QVBoxLayout(self.person_box)
        pl.setContentsMargins(0, 6, 0, 0)
        pl.setSpacing(6)
        self.role = Wd.Segmented([("target", "보정"), ("protect", "보호")], compact=True)
        self.role.setToolTip("보정: 이 인물에 보정을 적용합니다\n보호: 보정하지 않고, 다른 사람의 변형도 닿지 않게 지킵니다")
        self.role.changed.connect(self.ctl.set_role)
        pl.addWidget(self.role)
        self.add(self.person_box)
        pr = QHBoxLayout()
        pr.setContentsMargins(0, 2, 0, 0)
        self.preset_btn = Wd.button("프리셋 불러오기", "", "layers", "저장해 둔 값 묶음을 이 인물이나 컷 전체에 적용")
        self.preset_btn.clicked.connect(lambda: self.ctl.preset_menu(self.preset_btn))
        self.preset_save = Wd.button("현재 값 저장", "ghost", "plus", "이 인물의 값을 이름을 붙여 프리셋으로 저장")
        self.preset_save.clicked.connect(self.ctl.preset_save_dialog)
        pr.addWidget(self.preset_btn)
        pr.addWidget(self.preset_save)
        pr.addStretch(1)
        pw = QWidget()
        pw.setLayout(pr)
        pl.addWidget(pw)
        # widgets only one of the two tabs has
        self.mode = self.strength = self.feather = None
        self.skin_detail = self.grain_size = self.blemish_size = self.skin_scope = None
        self.sculpt_btn = self.sculpt_clear = self.body_btn = self.body_clear = None
        self.face_note = self.parts_btn = self.parts_info = self.cloth_keep = None
        self.level = None
        self.switches: Dict[str, Wd.Segmented] = {}
        self.wrows: Dict[str, Wd.ParamRow] = {}
        add = self.add
        if part == "body":
            self.add(Wd.Section("방식"))
            self.mode = Wd.Segmented([("warp", "워프"), ("separate", "분리 + 배경 복원")])
            self.mode.changed.connect(self.ctl.set_mode)
            self.mode.setToolTip("워프: 주변 배경까지 자연스럽게 함께 변형 (기본)\n분리: 인물만 변형하고 드러난 배경은 이웃 프레임에서 복원")
            self.add(self.mode)
            # overall strength
            self.strength = Wd.ParamRow("strength", "전체 강도", 0, 150, 1, "pct", 100)
            self.strength.name.setToolTip("몸·얼굴·머리 모양 변화 전체에 곱해지는 강도 (피부·색 보정은 제외)")
            self.strength.changed.connect(lambda k, v: self.ctl.param_changed("__strength__", v, False))
            self.strength.committed.connect(lambda k, v: self.ctl.param_changed("__strength__", v, True))
            self.strength.keyToggled.connect(lambda k: self.ctl.toggle_param_key("__strength__"))
            self.add(self.strength)
            self.feather = Wd.ParamRow("feather", "경계 부드러움", 0, 100, 1, "pct", 50)
            self.feather.name.setToolTip("모양 보정이 주변으로 사라지는 폭. 높을수록 경계가 부드럽고 넓게 퍼지며, "
                                         "낮을수록 보정한 부위에만 좁게 들어갑니다 (얼굴·머리·몸·피부 영역 모두)")
            self.feather.changed.connect(lambda k, v: self.ctl.set_feather(v, False))
            self.feather.committed.connect(lambda k, v: self.ctl.set_feather(v, True))
            self.add(self.feather)
            self.lay.addSpacing(6)
        else:
            self.add(Wd.Section("피부 보정"))
            self.view = Wd.Segmented([("basic", "기본 조정"), ("detail", "세부 조정")], current=self._view)
            self.view.changed.connect(self._show_view)
            self.add(self.view)
            self.basic_box = self._basic_box()
            self.add(self.basic_box)
            self.detail_box = QWidget()
            dl = QVBoxLayout(self.detail_box)
            dl.setContentsMargins(0, 6, 0, 0)
            dl.setSpacing(4)
            self.add(self.detail_box)
            add = dl.addWidget
        # groups
        self.rows: Dict[str, Wd.ParamRow] = {}
        self.groups: Dict[str, tuple] = {}
        unit_map = {"%": "%", "pct": "pct", "deg": "deg"}
        for gkey, title, hint in GROUPS:
            if (gkey in self.SKIN_GROUPS) != (part == "skin"):
                continue
            head = Wd.GroupHeader(title, hint, expanded=self._expanded.get(gkey, False))
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 8)
            bl.setSpacing(2)
            for d in [x for x in PARAMS if x.group == gkey]:
                r = Wd.ParamRow(d.key, d.label, d.lo, d.hi, 1, unit_map.get(d.unit, "%"), d.default)
                r.name.setToolTip(d.tip)
                r.name.setFixedWidth(64)
                r.changed.connect(lambda k, v: self.ctl.param_changed(k, v, False))
                r.committed.connect(lambda k, v: self.ctl.param_changed(k, v, True))
                r.keyToggled.connect(self.ctl.toggle_param_key)
                self.rows[d.key] = r
                bl.addWidget(r)
                if d.key == "sculpt":
                    bl.addWidget(self._sculpt_row())
                if d.key == "body_sculpt":
                    bl.addWidget(self._body_row())
            if gkey == "body":
                bl.addWidget(self._cloth_row())
                bl.addWidget(self._widths_box(WIDTH_KEYS, PARAM))
            if gkey == "face":
                self.face_note = Wd.label("", "Faint", 11)
                self.face_note.setWordWrap(True)
                bl.addWidget(self.face_note)
            if gkey == "skin":
                bl.addWidget(self._skin_extras())
            if gkey == "head":
                bl.addWidget(self._parts_row())
            box.setVisible(head.isChecked())
            head.toggled.connect(lambda on, b=box, k=gkey: (b.setVisible(on), self._expanded.__setitem__(k, on)))
            self.groups[gkey] = (head, box)
            add(head)
            add(box)
        btns = QHBoxLayout()
        reset = Wd.button("초기화", "ghost", "refresh-cw",
                          "이 인물의 피부·색 보정 값을 처음으로" if part == "skin" else "이 인물의 몸매·얼굴형 값을 처음으로")
        reset.clicked.connect(lambda: self.ctl.reset_params(part))
        copy = Wd.button("다른 컷에 복사", "", "copy")
        copy.clicked.connect(self.ctl.copy_settings_dialog)
        btns.addWidget(reset)
        btns.addStretch(1)
        btns.addWidget(copy)
        w = QWidget()
        w.setLayout(btns)
        self.add(Wd.hline())
        self.add(w)
        self.lay.addStretch(1)
        self._chips: Dict[int, Wd.Chip] = {}
        # colour grain only means something once there is grain
        if "grain" in self.rows and "grain_color" in self.rows:
            self.rows["grain"].changed.connect(lambda k, v: self._grain_gate(v))
        if self.strength is not None:
            self.strength.name.setFixedWidth(64)
        if self.feather is not None:
            self.feather.name.setFixedWidth(64)
            self._retain(self.feather)
        if part == "skin":
            self._show_view(self._view)

    # ---- skin: basic adjustment
    def _basic_box(self):
        from ..core.params import SKIN_LEVELS, SKIN_REMOVE
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 4)
        v.setSpacing(6)
        v.addWidget(Wd.label("보정 강도", "", 12))
        self.level = Wd.Segmented(SKIN_LEVELS)
        self.level.set_current("")
        self.level.setToolTip("피부결과 번들거림·다크서클·홍조·목주름·목 닭살을 한 번에.\n"
                              "표준은 엔진의 기준 보정 정도입니다")
        self.level.changed.connect(lambda k: self.ctl.apply_skin_basic(level=k))
        v.addWidget(self.level)
        self.level_info = Wd.label("", "Faint", 11)
        self.level_info.setWordWrap(True)
        v.addWidget(self.level_info)
        v.addSpacing(4)
        tips = {"mouth": "입꼬리 옆 주름", "naso": "코 옆에서 입가로 내려오는 주름",
                "acne": "여드름·트러블·점처럼 주변보다 어둡거나 붉은 작은 점 (세부 조정의 잡티)"}
        for sw, key, lab in SKIN_REMOVE:
            seg = Wd.Segmented([("keep", "보호"), ("remove", "제거")], compact=True)
            seg.set_current("")
            seg.setFixedWidth(132)
            seg.changed.connect(lambda k, sw=sw: self.ctl.apply_skin_basic(switch=(sw, k == "remove")))
            v.addWidget(Wd.FieldRow(lab, seg, tips.get(sw, "")))
            self.switches[sw] = seg
        self.basic_note = Wd.label("고른 값은 세부 조정 슬라이더에 그대로 들어갑니다. 세부 조정에서 값을 바꾸면 "
                                   "여기 선택이 해제됩니다.", "Faint", 11)
        self.basic_note.setWordWrap(True)
        v.addSpacing(2)
        v.addWidget(self.basic_note)
        return w

    def _show_view(self, key: str):
        ReshapeTab._view = key
        self.view.set_current(key)
        self.basic_box.setVisible(key == "basic")
        self.detail_box.setVisible(key == "detail")

    def _fill_basic(self, pp, enabled: bool):
        from ..core.params import skin_basic_values, PARAM
        sb = (getattr(pp, "skin_basic", None) or {}) if pp is not None else {}
        self.level.set_current(sb.get("level", "") if sb else "")
        rm = set(sb.get("remove", [])) if sb else set()
        for sw, seg in self.switches.items():
            seg.set_current(("remove" if sw in rm else "keep") if sb else "")
        for w_ in [self.level] + list(self.switches.values()):
            w_.setEnabled(enabled)
        if sb:
            vals = skin_basic_values(sb["level"], sb.get("remove", []))
            self.level_info.setText(" · ".join(f"{PARAM[k].label} {v:.0f}" for k, v in vals.items() if v > 0))
        elif pp is not None and pp.retouch_active():
            self.level_info.setText("세부 조정에서 직접 맞춘 값입니다")
        else:
            self.level_info.setText("")
        self.level_info.setVisible(bool(self.level_info.text()))

    # ---- group extras
    def _retain(self, r):
        sp_ = r.keyb.sizePolicy()
        sp_.setRetainSizeWhenHidden(True)
        r.keyb.setSizePolicy(sp_)
        r.keyb.hide()

    def _cloth_row(self):
        """옷 그대로: on/off, then one checkbox per garment actually found on this person."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 6, 0, 2)
        v.setSpacing(4)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        self.cloth_keep = Wd.Toggle()
        self.cloth_keep.toggled.connect(self._cloth_toggled)
        h.addWidget(Wd.FieldRow("옷 그대로", self.cloth_keep,
                                "켜면 옷은 그대로 두고 드러난 팔·다리·목 같은 피부만 바꿉니다. "
                                "이 인물에게서 찾은 옷 종류만 아래에 나오고, 고른 것만 그대로 둡니다 "
                                "(얼굴·머리 보정은 영향 없음)"), 1)
        v.addLayout(h)
        self.garment_box = QWidget()
        gl = QHBoxLayout(self.garment_box)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(10)
        from ..track.clothing import GARMENTS
        self.garment_cb: Dict[str, QCheckBox] = {}
        for k, lab, _ in GARMENTS:
            cb = QCheckBox(lab)
            cb.toggled.connect(lambda on, kk=k: self._garment_toggled())
            gl.addWidget(cb)
            self.garment_cb[k] = cb
        gl.addStretch(1)
        v.addWidget(self.garment_box)
        self.cloth_info = Wd.label("", "Faint", 11)
        self.cloth_info.setWordWrap(True)
        v.addWidget(self.cloth_info)
        return w

    def _cloth_toggled(self, on: bool):
        if not on:
            self.ctl.set_cloth("include")
            return
        found = [k for k, cb in self.garment_cb.items() if cb.isVisible()]
        self.ctl.set_cloth("keep:" + ",".join(found or [k for k in self.garment_cb]))

    def _garment_toggled(self):
        sel = [k for k, cb in self.garment_cb.items() if cb.isChecked() and not cb.isHidden()]
        if not self.cloth_keep.isChecked():
            return
        self.ctl.set_cloth("keep:" + ",".join(sel) if sel else "include")

    def _fill_cloth(self, pp, person, s):
        from ..track import clothing as CL
        mode = getattr(pp, "cloth", "include")
        keep = CL.keep_set(mode)
        on = mode != "include"
        self.cloth_keep.blockSignals(True)
        self.cloth_keep.setChecked(on)
        self.cloth_keep.blockSignals(False)
        found = None
        ready = False
        if s.tracker is not None and s.tracker.store is not None and s.tracker.store.has(person.id):
            ready = CL.ready_for(s.tracker, person.id)
            if ready:
                found = CL.detected(s.tracker, s.tracker.store.index(person.id))
        for k, cb in self.garment_cb.items():
            cb.blockSignals(True)
            cb.setChecked(k in keep or (k == "dress" and "dress_low" in keep))
            cb.setVisible(on and found is not None and k in found)
            cb.blockSignals(False)
        self.garment_box.setVisible(on and bool(found))
        info = ""
        if on and not ready:
            info = "옷 영역을 분석하는 중입니다. 끝나면 이 인물에게서 찾은 옷이 나옵니다."
        elif on and found is not None and not found:
            info = "이 인물에게서 그대로 둘 옷을 찾지 못했습니다."
        elif on:
            info = "M 키(매트 보기)로 그대로 두는 옷 영역을 확인할 수 있습니다."
        self.cloth_info.setText(info)
        self.cloth_info.setVisible(bool(info))

    def _parts_row(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 6, 0, 2)
        h.setSpacing(6)
        self.parts_info = Wd.label("", "Faint", 11)
        self.parts_info.setWordWrap(True)
        self.parts_btn = Wd.button("머리 부위 인식", "", "scan")
        self.parts_btn.setToolTip("머리카락과 얼굴을 프레임마다 따로 찾아 둡니다 (SAM 2).\n"
                                  "헤어 볼륨은 머리카락에만 들어가고 얼굴·귀는 그대로 남으며,\n"
                                  "머리 크기·헤어 볼륨의 머리 윤곽에 옆에 든 손이나 팔이 섞이지 않습니다.")
        self.parts_btn.clicked.connect(lambda: self.ctl.ensure_parts(force=True))
        h.addWidget(self.parts_info, 1)
        h.addWidget(self.parts_btn)
        return w

    def _widths_box(self, keys, PARAM):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.width_toggle = QToolButton()
        self.width_toggle.setText("범위 조절")
        self.width_toggle.setCheckable(True)
        self.width_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.width_toggle.setIcon(I.icon("chevron-right", 14))
        self.width_toggle.toggled.connect(self._toggle_widths)
        self.width_toggle.setStyleSheet(f"QToolButton {{ color: {C['text2']}; padding: 4px 2px; }}")
        v.addWidget(self.width_toggle)
        self.width_box = QWidget()
        wl = QVBoxLayout(self.width_box)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(2)
        self.wrows: Dict[str, Wd.ParamRow] = {}
        for key in keys:
            r = Wd.ParamRow(key, PARAM[key].label, 0.5, 2.0, 0.05, "x", 1.0)
            r.name.setFixedWidth(64)
            self._retain(r)
            r.committed.connect(self.ctl.width_changed)
            r.changed.connect(lambda k, v: self.ctl.width_changed(k, v, live=True))
            self.wrows[key] = r
            wl.addWidget(r)
        hint = Wd.label("살이 아닌 옷까지 같이 줄어들면 범위를 줄이세요.", "Faint", 11)
        hint.setWordWrap(True)
        wl.addWidget(hint)
        self.width_box.hide()
        v.addWidget(self.width_box)
        return w

    def _sculpt_row(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 2, 0, 4)
        self.sculpt_btn = Wd.button("점 편집", "", "move", "얼굴 점을 직접 끌어 모양을 만듭니다. 한 프레임에서 옮기면 모든 프레임에 얼굴을 따라 적용됩니다")
        self.sculpt_btn.setCheckable(True)
        self.sculpt_btn.toggled.connect(self.ctl.set_sculpt_mode)
        self.sculpt_clear = Wd.button("점 초기화", "ghost", "refresh-cw")
        self.sculpt_clear.clicked.connect(self.ctl.clear_sculpt)
        self.sculpt_info = Wd.label("", "Faint", 11)
        h.addWidget(self.sculpt_btn)
        h.addWidget(self.sculpt_clear)
        h.addWidget(self.sculpt_info, 1)
        return w

    def _body_row(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 2, 0, 4)
        self.body_btn = Wd.button("몸 점 편집", "", "move",
                                  "팔·몸통·다리 윤곽을 따라 추적된 점을 직접 끌어 모양을 만듭니다. "
                                  "한 프레임에서 옮기면 모든 프레임에 몸을 따라 적용됩니다")
        self.body_btn.setCheckable(True)
        self.body_btn.toggled.connect(self.ctl.set_body_mode)
        self.body_clear = Wd.button("점 초기화", "ghost", "refresh-cw")
        self.body_clear.clicked.connect(self.ctl.clear_body_sculpt)
        self.body_info = Wd.label("", "Faint", 11)
        h.addWidget(self.body_btn)
        h.addWidget(self.body_clear)
        h.addWidget(self.body_info, 1)
        return w

    def _skin_extras(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(4)
        self.skin_detail = Wd.ParamRow("skin_detail", "질감 유지", 0, 100, 1, "pct", 45)
        self.skin_detail.name.setFixedWidth(64)
        self._retain(self.skin_detail)
        self.skin_detail.name.setToolTip("피부결 보정에서 모공·잔결을 남기는 정도. 높을수록 자연스럽습니다")
        self.skin_detail.changed.connect(lambda k, val: self.ctl.set_skin("detail", val, False))
        self.skin_detail.committed.connect(lambda k, val: self.ctl.set_skin("detail", val, True))
        v.addWidget(self.skin_detail)
        self.grain_size = Wd.ParamRow("grain_size", "그레인 크기", 0.5, 3.0, 0.1, "x", 1.0)
        self.grain_size.name.setFixedWidth(64)
        self._retain(self.grain_size)
        self.grain_size.name.setToolTip("그레인 입자 크기 (1.0 = 1080p에서 1px)")
        self.grain_size.changed.connect(lambda k, val: self.ctl.set_skin("grain_size", val, False))
        self.grain_size.committed.connect(lambda k, val: self.ctl.set_skin("grain_size", val, True))
        v.addWidget(self.grain_size)
        self.blemish_size = Wd.ParamRow("blemish_size", "잡티 크기", 0, 100, 1, "pct", 50)
        self.blemish_size.name.setFixedWidth(64)
        self._retain(self.blemish_size)
        self.blemish_size.name.setToolTip("지울 잡티의 최대 크기. 높이면 큰 트러블과 점까지, 낮추면 작은 점만 지웁니다")
        self.blemish_size.changed.connect(lambda k, val: self.ctl.set_skin("blemish_size", val, False))
        self.blemish_size.committed.connect(lambda k, val: self.ctl.set_skin("blemish_size", val, True))
        v.addWidget(self.blemish_size)
        from ..core.project import SKIN_REGIONS
        self.skin_scope = QWidget()
        self.skin_scope.setToolTip("피부결·피부톤·그레인이 적용되는 부위 (여러 개 선택, 최소 1개)\n"
                                   "옷과 머리카락은 어느 부위에서도 제외됩니다")
        gl = QGridLayout(self.skin_scope)
        gl.setContentsMargins(0, 2, 0, 2)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(4)
        self.scope_boxes: Dict[str, QCheckBox] = {}
        for i, (k, lab) in enumerate(SKIN_REGIONS):
            cb = QCheckBox(lab)
            cb.toggled.connect(lambda on, kk=k: self._scope_toggled(kk, on))
            gl.addWidget(cb, i // 3, i % 3)
            self.scope_boxes[k] = cb
        v.addWidget(Wd.label("보정 부위", "", 12))
        v.addWidget(self.skin_scope)
        v.addWidget(Wd.label("K: 피부·색 보정 영역 보기 (변형 결과를 따라 움직입니다)", "Faint", 11))
        return w

    def _grain_gate(self, grain: float):
        r = self.rows.get("grain_color")
        if r is not None:
            on = self.rows["grain"].isEnabled() and abs(float(grain)) > 1e-6
            r.setEnabled(on)
            r.setToolTip("" if on else "그레인을 먼저 올리면 쓸 수 있습니다")

    def _scope_toggled(self, key: str, on: bool):
        sel = [k for k, cb in self.scope_boxes.items() if cb.isChecked()]
        if not sel:                                   # never nothing: keep the one just unticked
            cb = self.scope_boxes[key]
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            return
        self.ctl.set_skin("scope", ",".join(sel), True)

    def _set_scope(self, scope: str):
        from ..core.project import scope_set
        on = scope_set(scope)
        for k, cb in self.scope_boxes.items():
            cb.blockSignals(True)
            cb.setChecked(k in on)
            cb.blockSignals(False)

    def _toggle_widths(self, on):
        self.width_box.setVisible(on)
        self.width_toggle.setIcon(I.icon("chevron-down" if on else "chevron-right", 14))

    def _chip_delete(self, pid: int):
        self.ctl.sel_pid = pid
        self.ctl.sel_pids = set()
        self.ctl.delete_person()

    def _chip_rename(self, pid: int, name: str):
        self.ctl.sel_pid = pid
        self.ctl.rename_person(name)

    def refresh(self):
        from ..core.params import GROUPS, PARAMS
        s = self.ctl.session
        cut = s.cut
        t = self.ctl.frame
        while self.chips.count():
            it = self.chips.takeAt(0)
            w = it.widget() if it is not None else None
            if w:
                w.setParent(None)
                w.deleteLater()
        self._chips.clear()
        persons = [p for p in (cut.persons if cut else [])]
        for i, p in enumerate(persons):
            badge = {"target": "", "protect": "보호", "ignore": "보호"}.get(p.role, "")
            ch = Wd.Chip(p.name, p.color)
            ch.set_text(p.name, badge)
            ch.setChecked(p.id == self.ctl.sel_pid or p.id in (getattr(self.ctl, "sel_pids", set()) or set()))
            ch.clicked.connect(lambda _=False, pid=p.id: self.ctl.select_person(pid))
            ch.setToolTip("더블클릭: 이름 수정 · 마우스를 올리면 오른쪽에 삭제 버튼")
            ch.deleteRequested.connect(lambda pid=p.id: self._chip_delete(pid))
            ch.renamed.connect(lambda t, pid=p.id: self._chip_rename(pid, t))
            self.chips.addWidget(ch)
            self._chips[p.id] = ch
        self.no_people.setVisible(not persons)
        person = cut.person(self.ctl.sel_pid) if (cut and self.ctl.sel_pid is not None) else None
        self.person_box.setVisible(person is not None)
        enabled = person is not None and person.role == "target"
        if person is not None:
            self.role.set_current("protect" if person.role == "ignore" else person.role)
        if cut and self.mode is not None:
            self.mode.set_current(cut.mode)
        ws = list(self.rows.values()) + list(self.wrows.values()) + \
            [w_ for w_ in (self.strength, self.feather, self.skin_detail, self.grain_size, self.blemish_size,
                           self.skin_scope, self.sculpt_btn, self.sculpt_clear, self.body_btn, self.body_clear)
             if w_ is not None]
        for w in ws:
            w.setEnabled(enabled)
        pp = cut.params_for(person.id) if person is not None else None
        for k, r in self.rows.items():
            if pp is not None:
                c = pp.amounts[k]
                r.set_value(c.value(t))
                r.set_key_state("key" if c.has_key(t) else ("animated" if c.is_animated() else "none"))
            else:
                r.set_value(0)
                r.set_key_state("none")
        if self.part == "skin":
            self._fill_basic(pp, enabled)
        if pp is not None:
            if self.strength is not None:
                self.strength.set_value(pp.strength.value(t))
                self.strength.set_key_state("key" if pp.strength.has_key(t) else
                                            ("animated" if pp.strength.is_animated() else "none"))
            if self.feather is not None:
                self.feather.set_value(pp.feather)
            if self.skin_detail is not None:
                self.skin_detail.set_value(pp.skin_detail)
                self.grain_size.set_value(pp.grain_size)
                self.blemish_size.set_value(pp.blemish_size)
                self._set_scope(pp.skin_scope)
            if self.cloth_keep is not None:
                self._fill_cloth(pp, person, s)
            if self.parts_btn is not None:
                head_on = any(pp._active(k) for k in ("head", "crown", "hair_volume"))
                pinfo = ""
                ready = False
                if s.tracker is not None and s.tracker.store is not None:
                    from ..track import parts as PT
                    ready = PT.covers(s.tracker, person.id)
                    if ready:
                        pinfo = "머리 부위 인식됨: 헤어 볼륨은 머리카락에만, 얼굴과 귀는 그대로"
                    elif head_on:
                        pinfo = "머리 부위를 인식하면 헤어 볼륨이 머리카락에만 들어갑니다"
                self.parts_info.setText(pinfo)
                self.parts_btn.setText("다시 인식" if ready else "머리 부위 인식")
                self.parts_btn.setVisible(head_on or ready)
                self.parts_info.setVisible(bool(pinfo))
            for k, r in self.wrows.items():
                r.set_value(pp.widths.get(k, 1.0))
            if self.sculpt_btn is not None:
                n = len(pp.sculpt)
                self.sculpt_info.setText(f"옮긴 점 {n}개" if n else "")
            if self.body_btn is not None:
                n = len(pp.body_sculpt)
                self.body_info.setText(f"옮긴 점 {n}개" if n else "")
            for gkey, (head, box) in self.groups.items():
                head.set_active(any(pp._active(d.key) for d in PARAMS if d.group == gkey))
            if "grain" in pp.amounts and "grain" in self.rows:
                self._grain_gate(pp.amounts["grain"].value(t))
        else:
            for head, box in self.groups.values():
                head.set_active(False)
        if self.sculpt_btn is not None:
            self.sculpt_btn.blockSignals(True)
            self.sculpt_btn.setChecked(self.ctl.viewer.tool == "sculpt")
            self.sculpt_btn.blockSignals(False)
        if self.body_btn is not None:
            self.body_btn.blockSignals(True)
            self.body_btn.setChecked(self.ctl.viewer.tool == "bodypts")
            self.body_btn.blockSignals(False)
        if self.face_note is not None:
            # face landmarks are needed for the face, head and skin tools
            st_ = s.tracker.store if s.tracker else None
            note = ""
            if cut and cut.analyzed and st_ is not None and not st_.has_face:
                note = "얼굴·머리·피부 보정은 얼굴 랜드마크가 필요합니다. 추적 탭에서 다시 분석하세요."
            elif person is not None and st_ is not None and st_.has(person.id):
                p_ = st_.index(person.id)
                if st_.present[p_, t] and st_.face_vis[p_, t] < 0.15:
                    note = "이 프레임에서는 얼굴이 잘 보이지 않아 얼굴 보정이 약해집니다."
            self.face_note.setText(note)
            self.face_note.setVisible(bool(note))


# =========================================================================== 추적

class TrackTab(_Tab):
    def __init__(self, ctl):
        super().__init__(ctl)
        self.status = Wd.label("", "", 12, 500)
        self.status.setWordWrap(True)
        self.sub = Wd.label("", "Faint", 11)
        self.sub.setWordWrap(True)
        self.add(Wd.Section("분석"))
        self.add(self.status)
        self.add(self.sub)
        self.analyze_btn = Wd.button("분석 시작", "primary", "scan")
        self.analyze_btn.clicked.connect(self.ctl.run_analysis)
        self.add(self.analyze_btn)
        self.add(Wd.Section("정확도"))
        self.t_face = Wd.Toggle()
        self.t_flow = Wd.Toggle()
        self.t_pass2 = Wd.Toggle()
        self.t_m = Wd.Toggle()
        self.t_detail = Wd.Toggle()
        self.t_flip = Wd.Toggle()
        self.t_reid = Wd.Toggle()
        self.t_vt = Wd.Toggle()
        self.t_vt.toggled.connect(lambda v: self.ctl.set_tracking("video_track", "auto" if v else "off"))
        self.add(Wd.FieldRow("비디오 추적 (SAM 2, 실험)", self.t_vt,
                             "실험 기능. 사람마다 SAM 2 비디오 메모리로 자기 픽셀을 기억해 프레임마다 다시 찾습니다. "
                             "같은 의상의 군무에서 엇갈림 뒤 번호가 덜 바뀌는 대신, 얼굴만 크게 나오는 클로즈업에서는 "
                             "기존 방식이 더 안정적일 수 있습니다. GPU(CUDA·DirectML)가 있을 때만 쓰고, 처음 한 번 "
                             "모델(약 190MB)을 받습니다"))
        for text, tg, attr, tip in (
                ("사람 재인식 (군무)", self.t_reid, "reid",
                 "몸 재인식(OSNet)과 얼굴 인식(SFace)으로 사람마다 생김새를 기억해, 교차·대형 변경·화면 이탈 뒤에도 "
                 "같은 사람에게 같은 번호를 붙입니다. 다른 컷으로 설정을 복사할 때 멤버 매칭에도 씁니다"),
                ("얼굴 트래킹 융합", self.t_face, "face", "얼굴 랜드마크로 머리·목 관절을 보정합니다"),
                ("얼굴·손 정밀 모델", self.t_detail, "detail",
                 "얼굴 106점, 손 21점 전용 모델을 얼굴·손 크롭에 따로 돌려 세부 위치를 잡습니다"),
                ("포인트 트래커 결합", self.t_flow, "point_tracker", "광류 추적과 포즈 추정을 양방향으로 융합합니다"),
                ("2차 정밀 추정", self.t_pass2, "second_pass", "1차 결과로 인물 영역을 다시 잡아 한 번 더 추정합니다"),
                ("좌우 반전 추정", self.t_flip, "flip_tta",
                 "모든 모델을 좌우 반전한 영상에도 돌려 평균합니다. 분석 시간 약 2배, 관절 떨림이 줄어듭니다 (최고 정밀은 항상 켜짐)"),
                ("보조 모델 추가 (RTMPose-m)", self.t_m, "model_m", "앙상블에 해상도가 다른 모델을 더해 튀는 프레임을 줄입니다")):
            tg.toggled.connect(lambda v, a=attr: self.ctl.set_tracking(a, v))
            self.add(Wd.FieldRow(text, tg, tip))
        # person matte
        self.t_sam = Wd.Toggle()
        self.t_sam.toggled.connect(lambda v: self.ctl.set_tracking("seg", "sam2" if v else "rvm"))
        self.add(Wd.FieldRow("SAM 2 인물 분할", self.t_sam,
                             "추적한 인물마다 SAM 2로 정확한 실루엣을 따로 만듭니다. 몸 윤곽 점, 다른 인물 보호, "
                             "분리 모드, 머리 윤곽이 모두 이 매트를 씁니다. 빠르게 모드에서는 RVM만 씁니다"))
        self.seg_info = Wd.label("", "Faint", 11)
        self.seg_info.setWordWrap(True)
        self.add(self.seg_info)
        self.rematte = Wd.button("인물 매트만 다시 만들기", "", "layers",
                                 "지금의 추적(수정한 관절 포함)으로 SAM 2 매트를 다시 만들고 몸 폭·윤곽 점을 다시 잽니다. "
                                 "다시 분석할 필요가 없습니다")
        self.rematte.clicked.connect(self.ctl.rebuild_matte)
        self.add(self.rematte)
        self.smooth = Wd.ParamRow("smooth", "스무딩", 0, 100, 1, "", 0)
        self.smooth.keyb.hide()
        self.smooth.name.setFixedWidth(64)
        self.smooth.committed.connect(lambda k, v: self.ctl.set_tracking("smoothing", v / 100.0))
        self.smooth.setToolTip("카메라 흔들림은 그대로 따라가고, 몸 기준의 떨림만 줄입니다")
        self.add(self.smooth)
        # learning (clip adaptation, cumulative model, re-solving from corrections) is always on and automatic
        # ranges
        self.add(Wd.Section("구간 (타임라인에서 Shift+드래그로 선택)"))
        self.sel_label = Wd.label("선택된 구간 없음", "Faint", 11)
        self.add(self.sel_label)
        g = QGridLayout()
        g.setSpacing(6)
        b1 = Wd.button("확정", "", "check", "이 구간의 추적이 정확하다고 표시 (학습 데이터와 클립 학습에 사용)")
        b2 = Wd.button("잠금", "", "lock", "다시 계산해도 이 구간은 바뀌지 않습니다")
        b3 = Wd.button("해제", "ghost", "lock-open", "선택 구간의 확정·잠금을 해제")
        b1.clicked.connect(self.ctl.confirm_selection)
        b2.clicked.connect(self.ctl.lock_selection)
        b3.clicked.connect(self.ctl.clear_selection_marks)
        g.addWidget(b1, 0, 0)
        g.addWidget(b2, 0, 1)
        g.addWidget(b3, 0, 2)
        w = QWidget()
        w.setLayout(g)
        self.add(w)
        self.add(Wd.Section("인물 정리"))
        m = Wd.button("인물 합치기…", "", "users")
        m.clicked.connect(self.ctl.merge_persons_dialog)
        fx = Wd.button("번호 바로잡기…", "", "git-compare-arrows",
                       "군무에서 두 사람의 번호가 서로 바뀌었거나, 추적이 엉뚱한 사람을 따라간 구간을 바로잡습니다")
        fx.clicked.connect(self.ctl.identity_dialog)
        g2 = QGridLayout()
        g2.setContentsMargins(0, 0, 0, 0)
        g2.addWidget(m, 0, 0)
        g2.addWidget(fx, 0, 1)
        w2 = QWidget()
        w2.setLayout(g2)
        self.add(w2)
        tips = Wd.label("관절 드래그: 위치 수정 (키프레임) · Alt+클릭: 키 삭제\n"
                        "우클릭: 안 보이는 관절로 지정 (가려짐·화면 밖)\n"
                        "점선 고리: 안 보인다고 판단한 관절, 드래그하면 다시 사용", "Faint", 11)
        tips.setWordWrap(True)
        self.add(Wd.Section("관절 편집"))
        self.add(tips)
        self.lay.addStretch(1)

    def refresh(self):
        from ..learn.refiner import active_version
        s = self.ctl.session
        cut = s.cut
        if cut is None:
            return
        ts = cut.tracking
        st = s.tracker.store if s.tracker else None
        if cut.analyzed and st is not None:
            ok = s.tracker.auto_ok_rate(ts.conf_threshold)
            n_low = int(((st.conf < ts.conf_threshold) & st.present).any(0).sum())
            self.status.setText(f"인물 {len(cut.persons)}명 · 자동 추적 성공률 {ok * 100:.0f}%")
            self.sub.setText(f"확인이 필요한 프레임 {n_low}개 · 수정한 관절 {cut.stats.corrections}회")
            self.analyze_btn.setText("다시 분석")
            self.analyze_btn.setProperty("kind", "")
            self.analyze_btn.setIcon(I.icon("refresh-cw", 15, C["text"], C["text"]))
        else:
            self.status.setText("아직 분석하지 않았습니다.")
            self.sub.setText("인물 검출, 포즈 앙상블, 얼굴 융합, 매트, 카메라 움직임을 한 번에 계산합니다.")
            self.analyze_btn.setText("분석 시작")
            self.analyze_btn.setProperty("kind", "primary")
            self.analyze_btn.setIcon(I.icon("scan", 15, "#FFFFFF", "#FFFFFF"))
        self.analyze_btn.style().unpolish(self.analyze_btn)
        self.analyze_btn.style().polish(self.analyze_btn)
        for tg, v in ((self.t_reid, getattr(ts, "reid", True)), (self.t_face, ts.face), (self.t_detail, ts.detail),
                      (self.t_flip, ts.flip_tta),
                      (self.t_flow, ts.point_tracker), (self.t_pass2, ts.second_pass),
                      (self.t_m, "pose_rtmpose_m" in ts.models)):
            tg.blockSignals(True)
            tg.setChecked(bool(v))
            tg.blockSignals(False)
        self.t_vt.blockSignals(True)
        self.t_vt.setChecked(getattr(ts, "video_track", "off") != "off")
        self.t_vt.blockSignals(False)
        self.t_sam.blockSignals(True)
        self.t_sam.setChecked(getattr(ts, "seg", "sam2") == "sam2")
        self.t_sam.blockSignals(False)
        has_seg = bool(s.tracker is not None and s.tracker.has_seg())
        self.seg_info.setText("이 컷의 매트: SAM 2" if has_seg else
                              ("이 컷의 매트: RVM (SAM 2로 바꾸려면 아래 버튼)" if cut.analyzed else ""))
        self.rematte.setEnabled(bool(cut.analyzed and st is not None))
        self.smooth.set_value(ts.smoothing * 100)
        sel = self.ctl.selection
        self.sel_label.setText(f"선택: {sel[0]}~{sel[1]} 프레임 ({sel[1] - sel[0] + 1}f)" if sel else "선택된 구간 없음")


# =========================================================================== 보호

class ProtectTab(_Tab):
    def __init__(self, ctl):
        super().__init__(ctl)
        self.add(Wd.Section("인물"))
        self.t_others = Wd.Toggle()
        self.t_others.toggled.connect(lambda v: self.ctl.set_protect("others", v))
        self.add(Wd.FieldRow("보정하지 않는 인물 보호", self.t_others,
                             "다른 멤버·엑스트라·소품 매트에는 변형이 닿지 않게 합니다"))
        self.falloff = Wd.ParamRow("falloff", "전환 폭", 0.5, 10, 0.5, "pct1", 0.5)
        self.falloff.keyb.hide()
        self.falloff.val.setFixedWidth(46)
        self.falloff.committed.connect(lambda k, v: self.ctl.set_protect("falloff_pct", v))
        self.falloff.setToolTip("보호 영역에서 변형이 0이 되기까지의 거리 (화면 높이 %)")
        self.add(self.falloff)
        self.reach = Wd.ParamRow("reach", "감쇠 반경", 1.3, 4.0, 0.1, "x", 1.9)
        self.reach.keyb.hide()
        self.reach.committed.connect(lambda k, v: self.ctl.set_protect("reach", v))
        self.reach.setToolTip("몸 바깥으로 변형이 퍼지는 거리. 작으면 배경 영향이 줄고, 크면 더 부드럽습니다")
        self.add(self.reach)
        self.t_border = Wd.Toggle()
        self.t_border.toggled.connect(lambda v: self.ctl.set_protect("border", v))
        self.add(Wd.FieldRow("화면 가장자리 고정", self.t_border))
        self.add(Wd.Section("배경 직선 보호"))
        hint = Wd.label("문틀, 기둥, 수평선처럼 휘면 티 나는 선을 따라 그리세요. "
                        "카메라가 움직이면 자동으로 따라갑니다. 다른 프레임에서 끝점을 옮기면 위치 키가 추가됩니다.", "Faint", 11)
        hint.setWordWrap(True)
        self.add(hint)
        row = QHBoxLayout()
        self.b_line = Wd.button("직선 그리기  L", "", "ruler")
        self.b_line.setCheckable(True)
        self.b_line.clicked.connect(lambda on: self.ctl.set_tool("line" if on else "edit"))
        row.addWidget(self.b_line)
        w = QWidget()
        w.setLayout(row)
        self.add(w)
        self.lines = QListWidget()
        self.lines.setFixedHeight(104)
        self.lines.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lines.customContextMenuRequested.connect(lambda p: self._item_menu(self.lines, p, "line"))
        self.add(self.lines)
        self.add(Wd.Section("보호 영역 (마이크, 소품, 손)"))
        row2 = QHBoxLayout()
        self.b_ell = Wd.button("타원", "", "circle-dashed")
        self.b_rect = Wd.button("사각형", "", "square-dashed")
        for b, k in ((self.b_ell, "ellipse"), (self.b_rect, "rect")):
            b.setCheckable(True)
            b.clicked.connect(lambda on, kk=k: self.ctl.set_tool(kk if on else "edit"))
            row2.addWidget(b)
        w2 = QWidget()
        w2.setLayout(row2)
        self.add(w2)
        self.shapes = QListWidget()
        self.shapes.setFixedHeight(90)
        self.shapes.setContextMenuPolicy(Qt.CustomContextMenu)
        self.shapes.customContextMenuRequested.connect(lambda p: self._item_menu(self.shapes, p, "shape"))
        self.add(self.shapes)
        self.lay.addStretch(1)

    def _item_menu(self, lst, pos, kind):
        it = lst.itemAt(pos)
        if it is None:
            return
        oid = it.data(Qt.UserRole)
        m = QMenu(self)
        a1 = m.addAction("켜기/끄기")
        a2 = m.addAction("이 프레임의 위치 키 삭제")
        m.addSeparator()
        a3 = m.addAction("삭제")
        a = m.exec(lst.viewport().mapToGlobal(pos))
        if a == a1:
            self.ctl.protect_item("toggle", kind, oid)
        elif a == a2:
            self.ctl.protect_item("delkey", kind, oid)
        elif a == a3:
            self.ctl.protect_item("delete", kind, oid)

    def refresh(self):
        cut = self.ctl.session.cut
        if cut is None:
            return
        pr = cut.protect
        for tg, v in ((self.t_others, pr.others), (self.t_border, pr.border)):
            tg.blockSignals(True)
            tg.setChecked(v)
            tg.blockSignals(False)
        self.falloff.set_value(pr.falloff_pct)
        self.reach.set_value(pr.reach)
        tool = self.ctl.tool
        self.b_line.setChecked(tool == "line")
        self.b_ell.setChecked(tool == "ellipse")
        self.b_rect.setChecked(tool == "rect")
        self.lines.clear()
        for i, ln in enumerate(pr.lines):
            it = QListWidgetItem(f"직선 {i + 1}  ·  위치 키 {len(ln.keys)}개" + ("" if ln.enabled else "  (꺼짐)"))
            it.setData(Qt.UserRole, ln.id)
            self.lines.addItem(it)
        self.shapes.clear()
        for i, sh in enumerate(pr.shapes):
            k = "타원" if sh.kind == "ellipse" else "사각형"
            it = QListWidgetItem(f"{k} {i + 1}  ·  위치 키 {len(sh.keys)}개" + ("" if sh.enabled else "  (꺼짐)"))
            it.setData(Qt.UserRole, sh.id)
            self.shapes.addItem(it)


# =========================================================================== QC

class QCTab(_Tab):
    def __init__(self, ctl):
        super().__init__(ctl)
        self.add(Wd.Section("렌더 전 자동 검사"))
        self.summary = Wd.label("아직 검사하지 않았습니다.", "Muted")
        self.summary.setWordWrap(True)
        self.add(self.summary)
        self.run = Wd.button("QC 실행", "primary", "shield-check")
        self.run.clicked.connect(self.ctl.run_qc)
        self.add(self.run)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.itemClicked.connect(self._jump)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        self.list.setMinimumHeight(260)
        self.list.setStyleSheet("QListWidget::item { padding: 8px 6px; border-bottom: 1px solid #2B2B2B; }")
        self.add(self.list)
        hint = Wd.label("항목을 누르면 해당 프레임으로 이동합니다. 오른쪽 클릭으로 '무시' 처리할 수 있습니다.", "Faint", 11)
        hint.setWordWrap(True)
        self.add(hint)
        self.lay.addStretch(1)

    def _jump(self, it):
        q = it.data(Qt.UserRole)
        self.ctl.jump_to(q.start, q.person)

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        q = it.data(Qt.UserRole)
        m = QMenu(self)
        a = m.addAction("무시" if q.status != "ignored" else "다시 표시")
        if m.exec(self.list.viewport().mapToGlobal(pos)) == a:
            q.status = "open" if q.status == "ignored" else "ignored"
            self.ctl.qc_changed()

    def refresh(self):
        cut = self.ctl.session.cut
        if cut is None:
            return
        self.list.clear()
        errs = sum(1 for q in cut.qc if q.severity == "error" and q.status == "open")
        warns = sum(1 for q in cut.qc if q.severity == "warn" and q.status == "open")
        ign = sum(1 for q in cut.qc if q.status == "ignored")
        if cut.qc or getattr(cut, "_qc_ran", False):
            if not errs and not warns:
                self.summary.setText("문제가 발견되지 않았습니다." + (f" (무시 {ign})" if ign else ""))
            else:
                self.summary.setText(f"오류 {errs} · 확인 {warns}" + (f" · 무시 {ign}" if ign else ""))
        for q in cut.qc:
            col = C["text3"] if q.status == "ignored" else (C["error"] if q.severity == "error" else C["warn"])
            rng = f"{q.start}" if q.start == q.end else f"{q.start}~{q.end}"
            it = QListWidgetItem(f"{KIND_TEXT.get(q.kind, q.kind)}  ·  {rng}\n{q.message}")
            it.setForeground(QColor(col) if q.status == "ignored" else QColor(C["text"]))
            it.setIcon(I.icon("triangle-alert" if q.severity == "warn" else "circle-alert", 14, col, col))
            it.setData(Qt.UserRole, q)
            self.list.addItem(it)


# =========================================================================== 출력

class OutputTab(_Tab):
    def __init__(self, ctl):
        super().__init__(ctl)
        self.add(Wd.Section("소스"))
        self.src = Wd.label("", "Faint", 11)
        self.src.setWordWrap(True)
        self.add(self.src)
        self.cs = QComboBox()
        self.cs.setMinimumWidth(180)
        self.cs.currentIndexChanged.connect(self._cs)
        self.add(Wd.FieldRow("입력 색공간", self.cs, "뷰어와 분석에 쓰는 표시 변환입니다. 렌더 결과의 픽셀 값은 바꾸지 않습니다."))
        self.lut = Wd.button("뷰 LUT 없음", "ghost")
        self.lut.clicked.connect(self.ctl.choose_lut)
        self.add(Wd.FieldRow("뷰 LUT", self.lut))
        self.range_lbl = Wd.label("", "Faint", 11)
        self.add(Wd.FieldRow("작업 구간", self.range_lbl, "I / O 키로 현재 프레임을 시작·끝으로 지정합니다"))
        self.add(Wd.Section("출력"))
        self.fmt = QComboBox()
        for k, t in OUTPUT_FORMATS:
            self.fmt.addItem(t, k)
        self.fmt.currentIndexChanged.connect(lambda i: self.ctl.set_output("format", self.fmt.itemData(i)))
        self.add(Wd.FieldRow("포맷", self.fmt))
        self.ocs = QComboBox()
        self.ocs.currentIndexChanged.connect(lambda i: self.ctl.set_output("output_cs", self.ocs.itemData(i) or ""))
        self.add(Wd.FieldRow("색공간", self.ocs, "원본 유지: 보정된 부분 외에는 원본과 같은 코드값으로 저장합니다 (권장)"))
        self.dir = Wd.button("", "ghost", "folder")
        self.dir.clicked.connect(self.ctl.browse_output_dir)
        self.add(Wd.FieldRow("폴더", self.dir))
        self.name = QLineEdit()
        self.name.setMinimumWidth(170)
        self.name.editingFinished.connect(lambda: self.ctl.set_output("name_pattern", self.name.text()))
        self.add(Wd.FieldRow("파일 이름", self.name, "{cut} 컷 이름 · {project} 프로젝트 · {date} 날짜. 같은 이름이 있으면 _v02처럼 번호가 붙습니다"))
        self.t_audio = Wd.Toggle()
        self.t_audio.toggled.connect(lambda v: self.ctl.set_output("include_audio", v))
        self.add(Wd.FieldRow("오디오 포함", self.t_audio))
        self.add(Wd.Section("매트 출력"))
        self.t_mp = Wd.Toggle()
        self.t_mp.toggled.connect(lambda v: self.ctl.set_output("matte_person", v))
        self.add(Wd.FieldRow("보정 인물 매트", self.t_mp, "보정된 모양에 맞춘 인물 알파 (별도 파일)"))
        self.t_mw = Wd.Toggle()
        self.t_mw.toggled.connect(lambda v: self.ctl.set_output("matte_warp", v))
        self.add(Wd.FieldRow("변형 영역 마스크", self.t_mw, "픽셀이 움직인 곳 (Resolve·Nuke 후처리용)"))
        self.add(Wd.Section(""))
        self.go = Wd.button("렌더 큐에 추가", "primary", "list-video")
        self.go.clicked.connect(self.ctl.enqueue_render)
        self.add(self.go)
        # edited timeline: how the whole timeline (or the chosen shots) is written out
        self.tl_box = QWidget()
        tv = QVBoxLayout(self.tl_box)
        tv.setContentsMargins(0, 8, 0, 0)
        tv.setSpacing(6)
        tv.addWidget(Wd.Section("타임라인 출력"))
        tv.addWidget(Wd.label("전체 길이 한 파일 · 컷별 개별 파일 · 리터칭 컷만 + 교체용 XML 중에서 고릅니다. "
                              "작업 불필요 컷과 추적 확인 컷을 넣을지도 여기서 정합니다.", "Faint", 11))
        row = QHBoxLayout()
        b_all = Wd.button("타임라인 내보내기…", "", "download")
        b_all.clicked.connect(lambda: self.ctl._tl_action("export", []))
        b_sel = Wd.button("선택한 컷만…", "ghost", "download", "컷 리스트나 타임라인에서 고른 컷만 내보냅니다")
        b_sel.clicked.connect(lambda: self.ctl._tl_action("export_sel", self.ctl._sel_cut_ids()))
        row.addWidget(b_all)
        row.addWidget(b_sel)
        row.addStretch(1)
        tv.addLayout(row)
        self.add(self.tl_box)
        self.note = Wd.label("", "Faint", 11)
        self.note.setWordWrap(True)
        self.add(self.note)
        self.lay.addStretch(1)
        self._filling = False

    def _cs(self, i):
        if not self._filling:
            self.ctl.set_input_cs(self.cs.itemData(i) or "")

    def refresh(self):
        from ..core.colour import PASSTHROUGH, manager
        s = self.ctl.session
        cut = s.cut
        if cut is None:
            return
        self._filling = True
        info = cut.info
        tc = f" · TC {info.timecode}" if info.timecode else ""
        self.src.setText(f"{info.codec} {info.pix_fmt} · {info.width}×{info.height} · {info.fps:.3f} fps{tc}\n"
                         f"{info.color_primaries or '?'} / {info.color_trc or '?'} / {info.color_matrix or '?'} · {info.color_range}")
        cm = manager()
        if self.cs.count() == 0:
            self.cs.addItem("변환 없음 (코드값 그대로)", "")
            for name, fam in cm.colorspaces():
                if fam.startswith("Display"):
                    continue
                self.cs.addItem(name, name)
            self.ocs.addItem("원본 유지", "")
            for name, fam in cm.colorspaces():
                self.ocs.addItem(name, name)
        idx = max(0, self.cs.findData(cut.input_cs or ""))
        self.cs.setCurrentIndex(idx)
        self.lut.setText(cut.lut_path.split("/")[-1].split("\\")[-1] if cut.lut_path else "뷰 LUT 없음")
        a, b = cut.work_range
        self.range_lbl.setText(f"{a} ~ {b - 1}  ({b - a}f)")
        o = cut.output
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(o.format)))
        self.ocs.setCurrentIndex(max(0, self.ocs.findData(o.output_cs or "")))
        self.ocs.setEnabled(bool(cut.input_cs))
        self.dir.setText(o.out_dir or "프로젝트 옆 DIAP_Renders")
        self.name.setText(o.name_pattern)
        for tg, v in ((self.t_audio, o.include_audio), (self.t_mp, o.matte_person), (self.t_mw, o.matte_warp)):
            tg.blockSignals(True)
            tg.setChecked(v)
            tg.blockSignals(False)
        self.t_audio.setEnabled(info.has_audio)
        self.note.setText("렌더는 원본 해상도·비트뎁스·타임코드를 유지합니다. 파일 이름 끝에 컷 버전(_v1)이 붙습니다.")
        self.tl_box.setVisible(bool(s.project is not None and s.project.timeline_cuts()))
        self._filling = False


# =========================================================================== container

class Inspector(QWidget):
    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.setSpacing(0)
        self.tabs = Wd.Segmented([("skin", "피부"), ("body", "몸매"), ("track", "추적"), ("protect", "보호"),
                                  ("output", "출력")], compact=True)
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(14, 0, 14, 8)
        wl.addWidget(self.tabs)
        lay.addWidget(wrap)
        lay.addWidget(Wd.hline())
        self.stack = QStackedWidget()
        self.pages = {
            "skin": ReshapeTab(ctl, "skin"), "body": ReshapeTab(ctl, "body"), "track": TrackTab(ctl),
            "protect": ProtectTab(ctl), "output": OutputTab(ctl),
        }
        self._edit_tab = "skin"            # "reshape" (from the window) means the last used of 피부 and 몸매
        for k, pg in self.pages.items():
            self.stack.addWidget(_scroll(pg))
        lay.addWidget(self.stack, 1)
        self.tabs.changed.connect(self.show_tab)
        self.setMinimumWidth(330)
        self.setMaximumWidth(420)

    def show_tab(self, key: str):
        keys = list(self.pages)
        if key == "reshape":
            key = self._edit_tab
        if key not in self.pages:
            return
        if key in ("skin", "body"):
            self._edit_tab = key
        self.tabs.set_current(key)
        self.stack.setCurrentIndex(keys.index(key))
        self.pages[key].refresh()

    def refresh(self, which: Optional[str] = None):
        cur = self.tabs.current
        if which == "reshape" and cur in ("skin", "body"):
            which = cur
        if which is None or which == cur:
            self.pages[cur].refresh()
