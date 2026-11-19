"""Automatic quality control before rendering.

Checks (per frame, grouped into ranges):
  warp      — displacement too large relative to the person
  fold      — the map folds over itself (Jacobian determinant too small)
  jitter    — tracking jitter that would make the warp shimmer
  line      — straight background lines that come out bent
  leak      — displacement reaching a protected person
  lowconf   — reshaping applied on frames where tracking is uncertain
  fill      — separation mode will need a lot of spatial inpainting
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.project import Cut, QCIssue
from ..core.settings import qc_threshold
from ..track import skeleton as S
from ..track.solver import relative_homographies
from ..warp import field as F

KIND_TEXT = {
    "warp": "변형량 과다",
    "fold": "접힘(왜곡 겹침)",
    "jitter": "프레임 간 떨림",
    "line": "배경 직선 휨",
    "leak": "보정하지 않는 인물에 변형 번짐",
    "lowconf": "추적 불확실 구간에 보정 적용",
    "fill": "배경 복원 인페인팅 많음",     # from the separate mode (removed in 1.22.2): only older saved issues
    "softened": "겹침 방지로 보정 약화",
}


def _lines_in_frame(gray: np.ndarray, people: Optional[np.ndarray], min_len: float) -> np.ndarray:
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
    except Exception:
        lines = None
    if lines is None:
        e = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(e, 1, np.pi / 360, 60, minLineLength=int(min_len), maxLineGap=4)
    if lines is None:
        return np.zeros((0, 4))
    L = lines.reshape(-1, 4).astype(np.float64)
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    L = L[ln >= min_len]
    if people is not None and len(L):
        keep = []
        for x1, y1, x2, y2 in L:
            ts = np.linspace(0, 1, 9)
            xs = np.clip((x1 + (x2 - x1) * ts).astype(int), 0, people.shape[1] - 1)
            ys = np.clip((y1 + (y2 - y1) * ts).astype(int), 0, people.shape[0] - 1)
            keep.append(people[ys, xs].max() < 60)
        L = L[np.array(keep, bool)]
    return L


def _bend(fr: F.FieldResult, seg: np.ndarray) -> float:
    """Max deviation (output px) of a straight source line after warping."""
    x1, y1, x2, y2 = seg
    ts = np.linspace(0, 1, 17)
    q = np.stack([x1 + (x2 - x1) * ts, y1 + (y2 - y1) * ts], 1)
    p = q.copy()
    for _ in range(4):         # solve p + D(p) = q
        p = q - F.sample_points(fr, p)
    if np.hypot(*(p - q).T).max() < 0.05:
        return 0.0
    a, b = p[0], p[-1]
    v = b - a
    L = np.hypot(*v)
    if L < 1:
        return 0.0
    n = np.array([-v[1], v[0]]) / L
    return float(np.abs((p - a) @ n).max())


def run_qc(cut: Cut, tracker, progress: Optional[Callable[[float, str], None]] = None,
           cancel: Optional[Callable[[], bool]] = None) -> List[QCIssue]:
    st = tracker.store
    if st is None:
        return []
    a, b = cut.work_range
    W, H = cut.info.width, cut.info.height
    proxy = tracker.proxy
    grid_step = max(2, int(round(W / 480)))
    thr = {k: qc_threshold(k) for k in ("max_warp_pct", "fold_det", "jitter_px", "line_bend_px", "leak_px", "low_conf")}
    per_frame: Dict[str, Dict[int, Tuple[float, int]]] = {k: {} for k in KIND_TEXT}
    targets = [(p.id, st.index(p.id)) for p in cut.targets() if st.has(p.id)]
    rel = relative_homographies(st.homog if st.homog_ok.any() else None, np.arange(a, b), 1)
    for i, t in enumerate(range(a, b)):
        if cancel and cancel():
            break
        fi = F.build_input(cut, tracker, t)
        if not fi.targets:
            continue
        fr = F.compute_field(fi, W, H, grid_step)
        if fr.empty():
            continue
        mag = np.hypot(fr.D[..., 0], fr.D[..., 1])
        # warp amount vs person height
        for pi in fi.targets:
            sc = st.scale[pi.p, t]
            if np.isfinite(sc) and sc > 0:
                pct = 100.0 * (fr.local_max if fr.local_max >= 0 else fr.max_disp) / sc
                if pct > thr["max_warp_pct"]:
                    per_frame["warp"][t] = (pct, st.pids[pi.p])
        det = F.jacobian_det(fr)
        dmin = float(det[mag > 0.05].min()) if (mag > 0.05).any() else 1.0
        if dmin < thr["fold_det"]:
            per_frame["fold"][t] = (dmin, -1)
        elif fr.repaired > 0.002:
            per_frame["softened"][t] = (fr.repaired * 100, -1)
        # leak onto protected people
        if fi.alpha is not None and fi.labels is not None:
            gh, gw = mag.shape
            al = cv2.resize(fi.alpha, (gw, gh), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(fi.labels.astype(np.int16), (gw, gh), interpolation=cv2.INTER_NEAREST)
            tgt = {pi.p for pi in fi.targets}
            other = (al > 160) & ~np.isin(lab, list(tgt))
            if other.any():
                leak = float(mag[other].max())
                if leak > thr["leak_px"]:
                    per_frame["leak"][t] = (leak, -1)
        # tracking confidence where the warp acts
        for pid, p in targets:
            if st.present[p, t] and st.conf[p, t] < thr["low_conf"]:
                per_frame["lowconf"][t] = (float(st.conf[p, t]), pid)
        # jitter: high-frequency joint motion in camera-stabilised coordinates, scaled by warp strength
        if a < t < b - 1:
            for pi in fi.targets:
                p = pi.p
                k0, k1, k2 = st.final[p, t - 1, :, :2], st.final[p, t, :, :2], st.final[p, t + 1, :, :2]
                if not (np.isfinite(k0).all() and np.isfinite(k1).all() and np.isfinite(k2).all()):
                    continue
                def mapk(kp, M):
                    h = np.c_[kp, np.ones(len(kp))] @ M.T
                    return h[:, :2] / h[:, 2:3]
                kk0 = mapk(k0, rel[i, 0])
                kk2 = mapk(k2, rel[i, 2])
                acc = np.linalg.norm(kk0 - 2 * k1 + kk2, axis=1)
                vel = np.linalg.norm(kk2 - kk0, axis=1) / 2
                # fast real motion produces acceleration too; only count it when the joint is nearly still
                noisy = acc[(vel < 0.01 * max(st.scale[p, t], 1))]
                amt = max(abs(v) for v in pi.amounts.values()) / 100.0 * pi.strength
                if len(noisy) and amt > 0:
                    j = float(np.percentile(noisy, 75)) * amt * 0.6
                    if j > thr["jitter_px"]:
                        per_frame["jitter"][t] = (j, st.pids[p])
        # background lines (every 2nd frame is plenty)
        if (t - a) % 2 == 0:
            g = proxy.gray(t)
            alpha = tracker.matte.get(t)
            people = cv2.resize(np.asarray(alpha), (proxy.w, proxy.h)) if alpha is not None else None
            lines = _lines_in_frame(g, people, 0.08 * proxy.h)
            worst = 0.0
            for seg in lines:
                src = seg / [proxy.sx, proxy.sy, proxy.sx, proxy.sy]
                mx, my = (src[0] + src[2]) / 2, (src[1] + src[3]) / 2
                gx = int(np.clip(mx / W * mag.shape[1], 0, mag.shape[1] - 1))
                gy = int(np.clip(my / H * mag.shape[0], 0, mag.shape[0] - 1))
                if mag[max(0, gy - 6):gy + 7, max(0, gx - 6):gx + 7].max() < 0.3:
                    continue
                worst = max(worst, _bend(fr, src))
            if worst > thr["line_bend_px"]:
                per_frame["line"][t] = (worst, -1)
                per_frame["line"][t + 1] = (worst, -1)
        if progress and i % 5 == 0:
            progress((i + 1) / max(1, b - a), "QC 검사 중")
    issues = _group(per_frame)
    # keep user decisions for issues that still exist
    old = {(q.kind, q.start, q.end): q.status for q in cut.qc}
    for q in issues:
        for (k, s0, e0), stt in old.items():
            if k == q.kind and s0 <= q.end and q.start <= e0 and stt == "ignored":
                q.status = "ignored"
    return issues


def _group(per_frame: Dict[str, Dict[int, Tuple[float, int]]]) -> List[QCIssue]:
    out = []
    sev = {"softened": "warn", "fold": "error", "leak": "error", "warp": "warn", "jitter": "warn", "line": "warn", "lowconf": "warn",
           "fill": "warn"}
    for kind, frames in per_frame.items():
        if not frames:
            continue
        ks = sorted(frames)
        start = prev = ks[0]
        worst = frames[start][0]
        pid = frames[start][1]
        for f in ks[1:] + [None]:
            if f is not None and f - prev <= 2:
                prev = f
                v = frames[f][0]
                worst = min(worst, v) if kind in ("fold", "lowconf") else max(worst, v)
                continue
            out.append(QCIssue(kind=kind, start=start, end=prev, severity=sev[kind],
                               message=_message(kind, worst), person=pid, value=float(worst)))
            if f is not None:
                start = prev = f
                worst = frames[f][0]
                pid = frames[f][1]
    out.sort(key=lambda q: (q.start, q.kind))
    return out


def _message(kind: str, v: float) -> str:
    if kind == "warp":
        return f"변형량이 인물 키의 {v:.1f}%까지 커집니다. 보정량이나 범위를 줄여 보세요."
    if kind == "fold":
        return f"왜곡이 겹치는 지점이 있습니다 (최소 {v:.2f}). 범위 배율을 키우거나 보정량을 줄여 보세요."
    if kind == "jitter":
        return f"추적 떨림이 결과에 약 {v:.1f}px 흔들림으로 보일 수 있습니다. 관절을 확인하거나 스무딩을 올리세요."
    if kind == "line":
        return f"배경 직선이 최대 {v:.1f}px 휘어집니다. 그 부위의 보정량이나 범위 배율을 줄여 보세요."
    if kind == "leak":
        return f"보정하지 않는 인물에 {v:.1f}px 변형이 닿습니다. 인물 매트와 관절을 확인하거나 보정량을 줄여 보세요."
    if kind == "lowconf":
        return f"추적 신뢰도가 {v:.2f}까지 떨어지는 구간에 보정이 적용됩니다. 관절을 확인하세요."
    if kind == "softened":
        return f"부위가 겹쳐 접힘이 생기지 않도록 일부 영역({v:.1f}%)의 보정을 자동으로 약하게 했습니다."
    return KIND_TEXT.get(kind, kind)
