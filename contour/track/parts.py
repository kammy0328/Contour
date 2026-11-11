"""Head parts per person (hair, face skin), for head edits that must stay on their own part.

SAM 2 (already used for the person mattes, Apache-2.0) is prompted from the face landmarks: points in the band
between the face outline and the head outline (over the top) say "hair", points on the eyes, nose, mouth and
cheeks say "face", each the other's negatives. Two masks per person and frame, stored at matte resolution
(1 = hair, 2 = face) next to the matte; the person owning a pixel comes from the person labels.

What the warp does with them:
- the head outline (head size, hair volume) is measured on hair + face instead of the whole person matte, so
  a raised hand or an arm beside the head is no longer taken for hair;
- hair volume moves hair only: face skin and the ears stay where they are."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .exposure import analysis_view

HAIR, FACE = 1, 2
EAR_L, EAR_R = 3, 4          # COCO ear joints


def _head_prompts(fg, hg, alpha_at=None):
    """(hair points, face points, hair box, face box) in source px, or None when there is no usable band."""
    from ..warp import face as FW
    poly = hg.polygon(fg)
    face = fg.contour()
    c = hg.center
    ro = FW._polar_radius(poly, c)
    ri = FW._polar_radius(face, c)
    up, side = -fg.ey, fg.ex
    hair = []
    for a in np.linspace(-np.pi * 0.55, np.pi * 0.55, 9):
        d = up * np.cos(a) + side * np.sin(a)
        b = int((np.arctan2(d[1], d[0]) + np.pi) / (2 * np.pi) * 360) % 360
        if ro[b] - ri[b] < 0.07 * fg.Wf:          # no hair to speak of in that direction (short hair, bald)
            continue
        p = c + d * (0.45 * ri[b] + 0.55 * ro[b])
        if alpha_at is not None and not alpha_at(p):
            continue
        hair.append(p)
    L = fg.lm
    face_pts = np.array([L[30], L[36:42].mean(0), L[42:48].mean(0), L[48:60].mean(0),
                         0.5 * (L[2] + L[30]), 0.5 * (L[14] + L[30])])
    if len(hair) < 3:
        return None
    hb = np.r_[poly.min(0), poly.max(0)]
    m = 0.08 * (hb[2:] - hb[:2])
    hb = np.r_[hb[:2] - m, hb[2:] + m]
    fb = np.r_[face.min(0), face.max(0)]
    return np.array(hair), face_pts, hb, fb


def segment_head(seg, rgb: np.ndarray, fg, hg, sx: float, sy: float, alpha_at=None):
    """Hair and face masks for one head. rgb is the proxy frame (sx, sy: source px -> proxy px).
    Returns (x0, y0, label crop uint8 at proxy px) or None."""
    pr = _head_prompts(fg, hg, alpha_at)
    if pr is None:
        return None
    hair, face_pts, hb, fb = pr
    S = np.array([sx, sy])
    hair, face_pts = hair * S, face_pts * S
    hb, fb = hb * np.r_[S, S], fb * np.r_[S, S]
    H, W = rgb.shape[:2]
    w, h = hb[2] - hb[0], hb[3] - hb[1]
    rect = (max(0.0, hb[0] - 0.25 * w), max(0.0, hb[1] - 0.25 * h), min(float(W), hb[2] + 0.25 * w),
            min(float(H), hb[3] + 0.25 * h))
    if rect[2] - rect[0] < 12 or rect[3] - rect[1] < 12:
        return None
    enc = seg.encode(rgb, rect)
    x0, y0, x1, y1 = enc["rect"]
    size = (int(round(x1 - x0)), int(round(y1 - y0)))

    def run(pos, neg, box):
        lg, q = seg.segment(enc, box, pos, neg)
        if lg is None:
            return None
        return cv2.resize(lg, size, interpolation=cv2.INTER_LINEAR) > 0

    hm = run(hair, face_pts, hb)
    fm = run(face_pts, hair, np.r_[fb[:2] - 0.1 * (fb[2:] - fb[:2]), fb[2:] + 0.1 * (fb[2:] - fb[:2])])
    if fm is None:
        return None
    # plausibility: SAM sometimes takes the whole person or the background for "hair"
    face_area = cv2.contourArea((fg.contour() * S).astype(np.float32))
    head_area = cv2.contourArea((hg.polygon(fg) * S).astype(np.float32))
    if fm.sum() > 2.2 * max(face_area, 1.0) or fm.sum() < 0.25 * face_area:
        return None
    if hm is not None:
        fx = np.clip((face_pts[:, 0] - x0).astype(int), 0, size[0] - 1)
        fy = np.clip((face_pts[:, 1] - y0).astype(int), 0, size[1] - 1)
        if hm.sum() > 2.5 * max(head_area, 1.0) or hm[fy, fx].mean() > 0.3:
            hm = None
    lab = np.zeros((size[1], size[0]), np.uint8)
    if hm is not None:
        lab[hm] = HAIR
    lab[fm] = FACE
    return int(round(x0)), int(round(y0)), lab


# --------------------------------------------------------------------------- store

def _state_path(tracker) -> Path:
    return Path(tracker.parts.path).with_suffix(".json")


def ready_for(tracker, pid: int) -> bool:
    if not hasattr(tracker, "parts") or not tracker.parts.exists():
        return False
    try:
        s = json.loads(_state_path(tracker).read_text(encoding="utf-8"))
    except Exception:
        return False
    # frames outside the range found then simply have no parts (the edit falls back to the head outline there)
    return s.get("analysis") == tracker.cut.analysis_version and int(pid) in s.get("pids", [])


def covers(tracker, pid: int) -> bool:
    """Found for this person over the whole current work range?"""
    if not ready_for(tracker, pid):
        return False
    s = json.loads(_state_path(tracker).read_text(encoding="utf-8"))
    r = s.get("range", [0, 0])
    a, b = tracker.cut.work_range
    return r[0] <= a and r[1] >= b


def _write_state(tracker, persons: List[int]) -> None:
    st = tracker.store
    try:
        old = json.loads(_state_path(tracker).read_text(encoding="utf-8"))
    except Exception:
        old = {}
    same = old.get("analysis") == tracker.cut.analysis_version
    pids = sorted(set(old.get("pids", [])) | {int(st.pids[p]) for p in persons}) \
        if same else sorted(int(st.pids[p]) for p in persons)
    a, b = tracker.cut.work_range
    r = old.get("range", [a, b]) if same else [a, b]
    rng = [min(a, r[0]), max(b, r[1])]
    _state_path(tracker).write_text(json.dumps(dict(analysis=tracker.cut.analysis_version, range=rng, pids=pids)),
                                    encoding="utf-8")


def run(tracker, persons: List[int], progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> bool:
    """Find hair and face of the given people (store indices) on every frame of the work range."""
    from .infer import Segmenter
    st = tracker.store
    proxy = tracker.proxy
    a, b = tracker.cut.work_range
    seg = Segmenter()
    mw, mh = tracker.parts.w, tracker.parts.h
    W, H = proxy.w, proxy.h
    kx, ky = mw / W, mh / H
    sx, sy = proxy.sx, proxy.sy
    mm = tracker.parts.open_rw()
    try:
        for t in range(a, b):
            if cancel and cancel():
                return False
            pres = [p for p in persons if st.present[p, t]]
            if not pres:
                continue
            rgb = analysis_view(np.asarray(proxy.frame(t)))
            lab = tracker.labels(t)
            al = tracker.matte.get(t)
            frame = np.array(mm[t])
            for p in pres:
                if lab is not None:
                    frame[lab == p] = 0
                fg = tracker.face_geom(p, t)
                if fg is None and hasattr(tracker, "head_face"):
                    fg = tracker.head_face(p, t)
                if fg is None:
                    continue
                hg = tracker.head_geom(p, t, fg, use_parts=False)
                if hg is None:
                    continue

                def alpha_at(q, al=al, lab=lab, p=p):
                    if al is None:
                        return True
                    x = int(np.clip(q[0] * sx * kx, 0, mw - 1)); y = int(np.clip(q[1] * sy * ky, 0, mh - 1))
                    return al[y, x] >= 110 and (lab is None or lab[y, x] in (p, -1))
                r = segment_head(seg, rgb, fg, hg, sx, sy, alpha_at)
                if r is None:
                    continue
                x0, y0, cm = r
                mx0, my0 = int(x0 * kx), int(y0 * ky)
                mx1 = min(mw, max(mx0 + 1, int(round((x0 + cm.shape[1]) * kx))))
                my1 = min(mh, max(my0 + 1, int(round((y0 + cm.shape[0]) * ky))))
                if mx1 <= mx0 or my1 <= my0:
                    continue
                small = cv2.resize(cm, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_NEAREST)
                own = (lab[my0:my1, mx0:mx1] == p) | (lab[my0:my1, mx0:mx1] == -1) if lab is not None else (small > 0)
                reg = frame[my0:my1, mx0:mx1]
                put = own & (small > 0)
                reg[put] = small[put]
            mm[t] = frame
            if progress:
                progress((t - a + 1) / max(1, b - a), f"머리 부위 인식 {t - a + 1}/{b - a}")
    finally:
        mm.flush()
        del mm
        tracker.parts.close()
    _write_state(tracker, persons)
    if hasattr(tracker, "_head_raw"):
        tracker._head_raw = {}
    return True


def masks(tracker, t: int, p: int) -> Optional[np.ndarray]:
    """Part labels of this person at matte resolution (0 / HAIR / FACE), or None."""
    if not hasattr(tracker, "parts") or not tracker.parts.exists():
        return None
    if not ready_for(tracker, tracker.store.pids[p]):
        return None
    m = tracker.parts.get(t)
    lab = tracker.labels(t)
    if m is None:
        return None
    if lab is not None:
        m = np.where((lab == p) | (lab == -1), m, 0).astype(np.uint8)
    if not (m > 0).any():
        return None
    return m


class PartMaps:
    """Soft part weights around one head, sampled in source px (for the warp)."""

    def __init__(self, hair: np.ndarray, keep: np.ndarray, x0: int, y0: int, kx: float, ky: float):
        self.hair, self.keep = hair, keep           # float32 0..1, matte px, crop at (x0, y0)
        self.x0, self.y0, self.kx, self.ky = x0, y0, kx, ky

    def sample(self, which: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        m = self.hair if which == "hair" else self.keep
        xs = (X * self.kx - self.x0).astype(np.float32)
        ys = (Y * self.ky - self.y0).astype(np.float32)
        return cv2.remap(m, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def part_maps(tracker, p: int, t: int, fg, kp: Optional[np.ndarray]) -> Optional[PartMaps]:
    """Hair weight and keep weight (face skin + ears) for frame t, averaged with the two frames on each side
    (moved along with the face) so the edit does not flicker with the masks."""
    if fg is None or not hasattr(tracker, "parts") or not ready_for(tracker, tracker.store.pids[p]):
        return None
    cut = tracker.cut
    mw, mh = tracker.parts.w, tracker.parts.h
    kx, ky = mw / cut.info.width, mh / cut.info.height
    # window around the head in matte px
    c = fg.O
    R = 1.6 * fg.Wf
    x0, y0 = int(max(0, (c[0] - R) * kx)), int(max(0, (c[1] - 1.4 * R) * ky))
    x1, y1 = int(min(mw, (c[0] + R) * kx + 1)), int(min(mh, (c[1] + 1.1 * R) * ky + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hair = np.zeros((y1 - y0, x1 - x0), np.float32)
    face = np.zeros_like(hair)
    wsum = 0.0
    for f, w in ((t, 2.0), (t - 1, 1.0), (t + 1, 1.0), (t - 2, 0.5), (t + 2, 0.5)):
        if not 0 <= f < tracker.store.n or not tracker.store.present[p, f]:
            continue
        m = masks(tracker, f, p)
        if m is None:
            continue
        ff = fg if f == t else tracker.face_geom(p, f)
        d = (fg.O - ff.O) if ff is not None else np.zeros(2)
        M = np.float32([[1, 0, d[0] * kx - x0], [0, 1, d[1] * ky - y0]])
        hm = cv2.warpAffine((m == HAIR).astype(np.float32), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        fm = cv2.warpAffine((m == FACE).astype(np.float32), M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        hair += w * hm
        face += w * fm
        wsum += w
    if wsum <= 0:
        return None
    hair /= wsum
    face /= wsum
    if hair.max() < 0.05 and face.max() < 0.05:
        return None
    # ears: beside the face, often under the hair; a disc at each ear joint
    keep = face.copy()
    if kp is not None:
        r = 0.16 * fg.Wf
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        for j in (EAR_L, EAR_R):
            e = kp[j]
            if np.isfinite(e[:2]).all() and e[2] > 0.3:
                d2 = ((xx / kx - e[0]) ** 2 + (yy / ky - e[1]) ** 2) / (r * r)
                keep = np.maximum(keep, np.clip(1.5 - d2, 0, 1).astype(np.float32))
    sg = max(0.6, 0.05 * fg.Wf * kx)
    hair = cv2.GaussianBlur(hair, (0, 0), sg)
    keep = np.clip(cv2.GaussianBlur(keep, (0, 0), sg) * 1.3, 0, 1)
    return PartMaps(hair, keep, x0, y0, kx, ky)


def head_alpha(tracker, p: int, t: int) -> Optional[np.ndarray]:
    """A matte of just this person's head (hair + face) at matte resolution, for the head outline."""
    m = masks(tracker, t, p)
    if m is None:
        return None
    return np.where(m > 0, 255, 0).astype(np.uint8)
