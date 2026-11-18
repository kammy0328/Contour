"""Clothing areas per person, for 'keep the clothes as they are' in the body reshaping.

A human parsing model (SegFormer-B2 fine-tuned for clothes, 18 classes) labels every pixel of each tracked
person's crop: skin parts (face, arms, legs), hair, and each kind of clothing. The class map is stored per frame
at matte resolution (class + 1, 0 = nobody) next to the matte; the person owning a pixel comes from the person
labels, so overlapping people never lend each other their clothes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .exposure import analysis_view

CLASSES = ["배경", "모자", "머리카락", "선글라스", "상의", "치마", "바지", "원피스", "벨트", "왼 신발", "오른 신발",
           "얼굴", "왼 다리", "오른 다리", "왼 팔", "오른 팔", "가방", "스카프"]
SKIRT, DRESS = 5, 7
ALL_CLOTHES = {1, 4, 5, 6, 7, 8, 9, 10, 16, 17}
# "옷 그대로": which garments stay as they are. Only garments found on the person are offered.
GARMENTS = [("top", "상의", {4, 17}), ("bottom", "바지", {6, 8}), ("skirt", "치마", {5}), ("dress", "원피스", {7})]
ACCESSORIES = {1, 9, 10, 16}           # hat, shoes, bag: kept whenever clothes are kept
MODES = ("include", "skirt", "all")    # older projects; now "keep:top,skirt,..."


def keep_set(mode: str) -> set:
    """Garment keys kept by a cloth mode ('include' -> none)."""
    mode = str(mode or "include")
    if mode == "include":
        return set()
    if mode == "all":
        return {k for k, _, _ in GARMENTS}
    if mode == "skirt":
        return {"skirt", "dress_low"}
    if mode.startswith("keep:"):
        return {k for k in mode[5:].split(",") if k}
    return set()


def detected(tracker, p: int, n: int = 12) -> list:
    """Garment keys seen on this person (more than 1 % of the person's pixels in the sampled frames)."""
    if not tracker.cloth.exists():
        return []
    st = tracker.store
    ts = [t for t in np.linspace(0, st.n - 1, n).astype(int) if st.present[p, t]]
    tot = 0
    cnt = {k: 0 for k, _, _ in GARMENTS}
    for t in ts:
        cm = tracker.cloth.get(int(t))
        lab = tracker.labels(int(t))
        if cm is None or lab is None:
            continue
        own = lab == p
        tot += int(own.sum())
        cls = cm.astype(np.int16) - 1
        for k, _, ids in GARMENTS:
            cnt[k] += int((own & np.isin(cls, list(ids))).sum())
    if tot <= 0:
        return []
    return [k for k, _, _ in GARMENTS if cnt[k] > 0.01 * tot]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Parser:
    SIZE = 512

    def __init__(self, key: str = "parse_clothes"):
        from . import unified as U
        self.key = key
        self.unified = U.runtime() if (key == "parse_clothes" and U.active("cloth")) else None
        self.sess = None
        if self.unified is None:
            self._load_own()

    def _load_own(self):
        from . import models as M
        self.sess = M.session(M.ensure_model(self.key))
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        """RGB uint8 crop -> class map (h, w) uint8."""
        h, w = crop.shape[:2]
        c512 = cv2.resize(crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        lg = None
        if self.unified is not None:
            try:
                lg = self.unified.cloth_logits(c512)
            except Exception as e:
                from . import unified as U
                U.fail("cloth", e)
                self.unified = None
                self._load_own()
        if lg is None:
            x = ((c512.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
            lg = self.sess.run(None, {self.inp: x})[0][0]             # (18, 128, 128)
        up = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in lg])
        return up.argmax(0).astype(np.uint8)


def _crop_box(st, p: int, t: int, sx: float, sy: float, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    b = st.boxes[p, t]
    kp = st.final[p, t]
    pts = []
    if np.isfinite(b).all():
        pts += [b[:2], b[2:]]
    ok = np.isfinite(kp[:, :2]).all(1)
    if ok.any():
        pts += list(kp[ok, :2])
    if len(pts) < 2:
        return None
    P = np.array(pts, np.float64) * [sx, sy]
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - 0.10 * w)), min(W, int(np.ceil(x1 + 0.10 * w)))
    y0, y1 = max(0, int(y0 - 0.12 * h)), min(H, int(np.ceil(y1 + 0.06 * h)))
    if x1 - x0 < 16 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def run(tracker, persons: List[int], progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> bool:
    """Parse the given people (store indices) on every frame of the work range into the clothing store."""
    st = tracker.store
    proxy = tracker.proxy
    a, b = tracker.cut.work_range
    parser = Parser()
    mw, mh = tracker.matte.w, tracker.matte.h
    W, H = proxy.w, proxy.h
    kx, ky = mw / W, mh / H
    mm = tracker.cloth.open_rw()
    try:
        for t in range(a, b):
            if cancel and cancel():
                return False
            pres = [p for p in persons if st.present[p, t]]
            if not pres:
                continue
            rgb = analysis_view(np.asarray(proxy.frame(t)))
            lab = tracker.labels(t)
            frame = np.array(mm[t])
            for p in pres:
                if lab is not None:
                    frame[lab == p] = 0
                box = _crop_box(st, p, t, proxy.sx, proxy.sy, W, H)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                cm = parser(rgb[y0:y1, x0:x1])
                mx0, my0 = int(x0 * kx), int(y0 * ky)
                mx1, my1 = max(mx0 + 1, int(round(x1 * kx))), max(my0 + 1, int(round(y1 * ky)))
                small = cv2.resize(cm, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
                own = (lab[my0:my1, mx0:mx1] == p) if lab is not None else (small > 0)
                reg = frame[my0:my1, mx0:mx1]
                reg[own] = small[own] + 1
            mm[t] = frame
            if progress:
                progress((t - a + 1) / max(1, b - a), f"옷 영역 분석 {t - a + 1}/{b - a}")
    finally:
        mm.flush()
        del mm
        tracker.cloth.close()
    _write_state(tracker, persons)
    return True


def _state_path(tracker) -> Path:
    return Path(tracker.cloth.path).with_suffix(".json")


def _write_state(tracker, persons: List[int]) -> None:
    st = tracker.store
    try:
        old = json.loads(_state_path(tracker).read_text(encoding="utf-8"))
    except Exception:
        old = {}
    pids = sorted(set(old.get("pids", [])) | {int(st.pids[p]) for p in persons}) \
        if old.get("analysis") == tracker.cut.analysis_version else sorted(int(st.pids[p]) for p in persons)
    _state_path(tracker).write_text(json.dumps(dict(analysis=tracker.cut.analysis_version,
                                                    range=list(tracker.cut.work_range), pids=pids)), encoding="utf-8")


def ready_for(tracker, pid: int) -> bool:
    """Has this person's clothing been parsed for the current analysis and work range?"""
    if not tracker.cloth.exists():
        return False
    try:
        s = json.loads(_state_path(tracker).read_text(encoding="utf-8"))
    except Exception:
        return False
    return s.get("analysis") == tracker.cut.analysis_version and list(s.get("range", [])) == list(tracker.cut.work_range) \
        and int(pid) in s.get("pids", [])


def keep_mask(tracker, t: int, p: int, mode: str) -> Optional[np.ndarray]:
    """Bool mask (matte resolution) of this person's clothing that stays as it is, or None."""
    keep = keep_set(mode)
    if not keep or not tracker.cloth.exists():
        return None
    cm = tracker.cloth.get(t)
    lab = tracker.labels(t)
    if cm is None or lab is None:
        return None
    cls = cm.astype(np.int16) - 1
    own = lab == p
    ids = set(ACCESSORIES)
    for k, _, cl in GARMENTS:
        if k in keep:
            ids |= cl
    m = own & np.isin(cls, list(ids))
    if "dress_low" in keep:
        dress = own & (cls == DRESS)
        if dress.any():
            # a dress is kept from the waist down (the top of it follows the body like any top)
            st = tracker.store
            kp = st.final[p, t]
            sh = [kp[j, :2] for j in (S.L_SH, S.R_SH) if np.isfinite(kp[j, :2]).all()]
            hp = [kp[j, :2] for j in (S.L_HIP, S.R_HIP) if np.isfinite(kp[j, :2]).all()]
            if sh and hp:
                y_waist = (0.38 * np.mean(sh, 0)[1] + 0.62 * np.mean(hp, 0)[1]) * tracker.cloth.h / tracker.cut.info.height
                rows = np.arange(m.shape[0])[:, None]
                m |= dress & (rows >= y_waist)
            else:
                m |= dress
    if not m.any():
        return None
    # pin holes and ragged edges of the parsing
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2).astype(bool)
    return m
