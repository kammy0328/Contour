"""Clip adaptation: learn from the user's corrections inside one cut.

Two learners, both fitted in seconds:
* Systematic bias per joint, in the person's own frame (rotation by torso angle, scale by height). Captures
  things like "the hip sits too high under this skirt" and applies the fix to every frame of that person.
* Appearance templates per joint from corrected frames; a normalised cross-correlation search around the
  current estimate adds a measurement wherever the corrected joint is clearly recognisable.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .store import TrackStore


def _torso_frame(kp: np.ndarray):
    """(e_x, e_y) unit vectors of the person's frame: e_y from neck to pelvis."""
    ls, rs, lh, rh = kp[S.L_SH, :2], kp[S.R_SH, :2], kp[S.L_HIP, :2], kp[S.R_HIP, :2]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all() and np.isfinite(lh).all() and np.isfinite(rh).all()):
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    v = (lh + rh) / 2 - (ls + rs) / 2
    n = np.linalg.norm(v)
    if n < 1e-3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ey = v / n
    ex = np.array([ey[1], -ey[0]])
    return ex, ey


def learn_bias(store: TrackStore, p: int, keys: Dict[int, Dict[int, List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    bias = np.zeros((S.K, 2), np.float32)
    weight = np.zeros(S.K, np.float32)
    for j, fr in keys.items():
        res = []
        for f, xy in fr.items():
            if not (0 <= f < store.n):
                continue
            m = store.meas[p, f]
            sc = store.scale[p, f]
            if not np.isfinite(m[j, :2]).all() or not np.isfinite(sc):
                continue
            ex, ey = _torso_frame(store.meas[p, f])
            r = np.asarray(xy) - m[j, :2]
            res.append([float(r @ ex) / sc, float(r @ ey) / sc])
        if len(res) < 2:
            continue
        res = np.array(res)
        med = np.median(res, 0)
        mad = np.median(np.abs(res - med), 0)
        mag = np.linalg.norm(med)
        if mag < 0.004:
            continue
        consistency = float(np.clip(1 - np.linalg.norm(mad) / max(mag, 1e-6), 0, 1))
        bias[j] = med
        weight[j] = consistency * min(1.0, len(res) / 4.0)
    return bias, weight


def bias_offsets(store: TrackStore, p: int, bias: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """(N,K,2) source-px offsets to add to the measurements of person p."""
    off = np.zeros((store.n, S.K, 2), np.float32)
    if not weight.any():
        return off
    for t in np.nonzero(store.present[p])[0]:
        sc = store.scale[p, t]
        if not np.isfinite(sc):
            continue
        ex, ey = _torso_frame(store.meas[p, t])
        for j in np.nonzero(weight)[0]:
            off[t, j] = weight[j] * sc * (bias[j, 0] * ex + bias[j, 1] * ey)
    return off


def template_measurements(store: TrackStore, p: int, keys: Dict[int, Dict[int, List[float]]],
                          gray: Callable[[int], np.ndarray], sx: float, sy: float, centers: np.ndarray,
                          extra_refs: Optional[Dict[int, Dict[int, List[float]]]] = None,
                          progress=None, frame_range: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Template-matching measurements for person p. centers (N,K,2) source px = where to search.

    Returns xy (N,K,2) and variance (N,K) (NaN/inf when no confident match)."""
    N = store.n
    xy = np.full((N, S.K, 2), np.nan, np.float32)
    var = np.full((N, S.K), np.inf, np.float32)
    refs: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for src in (keys, extra_refs or {}):
        for j, fr in src.items():
            for f, v in fr.items():
                refs.setdefault(j, []).append((int(f), np.asarray(v, float)))
    if not refs:
        return xy, var
    # build templates
    tpls: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    for j, lst in refs.items():
        for f, pos in lst:
            sc = store.scale[p, f] if np.isfinite(store.scale[p, f]) else np.nanmedian(store.scale[p])
            if not np.isfinite(sc):
                continue
            ps = int(np.clip(0.07 * sc * sx, 12, 56)) | 1
            g = gray(f)
            cx, cy = pos[0] * sx, pos[1] * sy
            x0, y0 = int(round(cx)) - ps // 2, int(round(cy)) - ps // 2
            if x0 < 0 or y0 < 0 or x0 + ps > g.shape[1] or y0 + ps > g.shape[0]:
                continue
            patch = g[y0:y0 + ps, x0:x0 + ps]
            if patch.std() < 4:          # flat area: template would match anything
                continue
            tpls.setdefault(j, []).append((f, patch.copy(), ps))
    if not tpls:
        return xy, var
    frames = np.nonzero(store.present[p])[0]
    if frame_range is not None:
        frames = frames[(frames >= frame_range[0]) & (frames < frame_range[1])]
    keyed ={j: {f for f, _ in refs.get(j, [])} for j in tpls}
    for n_i, t in enumerate(frames):
        g = None
        for j, lst in tpls.items():
            if t in keyed[j]:
                continue
            c = centers[t, j]
            if not np.isfinite(c).all():
                continue
            sc = store.scale[p, t]
            if not np.isfinite(sc):
                continue
            if g is None:
                g = gray(t)
            near = sorted(lst, key=lambda x: abs(x[0] - t))[:3]
            best = (-1.0, None, None)
            for f, patch, ps in near:
                r = int(max(6, 0.06 * sc * sx))
                cx, cy = c[0] * sx, c[1] * sy
                x0, y0 = int(round(cx)) - r - ps // 2, int(round(cy)) - r - ps // 2
                x1, y1 = x0 + 2 * r + ps, y0 + 2 * r + ps
                if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0]:
                    continue
                region = g[y0:y1, x0:x1]
                res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv <= best[0]:
                    continue
                # distinctiveness: suppress the peak neighbourhood and look at the runner-up
                r2 = res.copy()
                k = max(2, ps // 4)
                r2[max(0, ml[1] - k):ml[1] + k + 1, max(0, ml[0] - k):ml[0] + k + 1] = -1
                second = float(r2.max()) if r2.size else -1
                if mv - second < 0.06:
                    continue
                # sub-pixel peak
                dx = dy = 0.0
                if 0 < ml[0] < res.shape[1] - 1:
                    a_, b_, c_ = res[ml[1], ml[0] - 1], res[ml[1], ml[0]], res[ml[1], ml[0] + 1]
                    den = a_ - 2 * b_ + c_
                    dx = 0.5 * (a_ - c_) / den if abs(den) > 1e-9 else 0
                if 0 < ml[1] < res.shape[0] - 1:
                    a_, b_, c_ = res[ml[1] - 1, ml[0]], res[ml[1], ml[0]], res[ml[1] + 1, ml[0]]
                    den = a_ - 2 * b_ + c_
                    dy = 0.5 * (a_ - c_) / den if abs(den) > 1e-9 else 0
                px = x0 + ml[0] + ps // 2 + float(np.clip(dx, -0.5, 0.5))
                py = y0 + ml[1] + ps // 2 + float(np.clip(dy, -0.5, 0.5))
                best = (mv, np.array([px / sx, py / sy]), f)
            mv, pos, f = best
            if pos is None or mv < 0.72:
                continue
            q = (mv - 0.68) / 0.32
            age = abs(t - f)
            age_w = 1.0 / (1.0 + age / 90.0)
            xy[t, j] = pos
            var[t, j] = (0.012 * sc) ** 2 / max(1e-3, (q * age_w) ** 2)
        if progress and n_i % 20 == 0:
            progress(n_i / max(1, len(frames)), "클립 학습 적용")
    return xy, var
