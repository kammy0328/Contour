"""Body part geometry: segments derived from the skeleton and silhouette width measurement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..track import skeleton as S


@dataclass(frozen=True)
class SegDef:
    name: str
    part: str
    a: str            # joint index as str, or virtual: 'neck', 'pelvis', 'head'
    b: str
    t0: float         # window start (ramp centred here)
    t1: float         # window end
    ramp: float
    fallback: float   # half-width / segment length when the matte gives nothing
    gauss: Tuple[float, float] = (0.0, 0.0)   # (centre, sigma) for gaussian windows (waist)
    probe: Tuple[float, ...] = (0.35, 0.5, 0.65)


SEGMENTS: List[SegDef] = [
    # upper arm starts a third of the way down so the shoulder line and the deltoid stay put
    SegDef("upper_arm_l", "upper_arm", str(S.L_SH), str(S.L_EL), 0.30, 1.0, 0.14, 0.17, probe=(0.45, 0.6, 0.75)),
    SegDef("upper_arm_r", "upper_arm", str(S.R_SH), str(S.R_EL), 0.30, 1.0, 0.14, 0.17, probe=(0.45, 0.6, 0.75)),
    SegDef("forearm_l", "forearm", str(S.L_EL), str(S.L_WR), 0.0, 0.82, 0.15, 0.14),
    SegDef("forearm_r", "forearm", str(S.R_EL), str(S.R_WR), 0.0, 0.82, 0.15, 0.14),
    SegDef("thigh_l", "thighs", str(S.L_HIP), str(S.L_KN), 0.22, 1.0, 0.16, 0.26),
    SegDef("thigh_r", "thighs", str(S.R_HIP), str(S.R_KN), 0.22, 1.0, 0.16, 0.26),
    SegDef("calf_l", "calves", str(S.L_KN), str(S.L_AN), 0.0, 0.86, 0.16, 0.17),
    SegDef("calf_r", "calves", str(S.R_KN), str(S.R_AN), 0.0, 0.86, 0.16, 0.17),
    SegDef("neck", "neck", "neck", "head", 0.0, 0.5, 0.12, 0.30, probe=(0.25, 0.35, 0.45)),
    SegDef("waist", "waist", "neck", "pelvis", 0.0, 1.0, 0.0, 0.27, gauss=(0.62, 0.12), probe=(0.55, 0.62, 0.69)),
    # knee: the end of the thigh and the start of the calf, narrow windows
    SegDef("knee_tl", "knee", str(S.L_HIP), str(S.L_KN), 0.0, 1.0, 0.0, 0.10, gauss=(0.95, 0.07), probe=(0.88, 0.92, 0.96)),
    SegDef("knee_tr", "knee", str(S.R_HIP), str(S.R_KN), 0.0, 1.0, 0.0, 0.10, gauss=(0.95, 0.07), probe=(0.88, 0.92, 0.96)),
    SegDef("knee_cl", "knee", str(S.L_KN), str(S.L_AN), 0.0, 1.0, 0.0, 0.11, gauss=(0.05, 0.07), probe=(0.04, 0.08, 0.12)),
    SegDef("knee_cr", "knee", str(S.R_KN), str(S.R_AN), 0.0, 1.0, 0.0, 0.11, gauss=(0.05, 0.07), probe=(0.04, 0.08, 0.12)),
    # 1.3.1: pelvis (hip bones, a little below the hip joints) and the shoulder lines (outline points only)
    SegDef("hips", "pelvis", "neck", "pelvis", 0.0, 1.0, 0.0, 0.30, gauss=(1.02, 0.12), probe=(0.96, 1.02, 1.08)),
    SegDef("shoulder_l", "_outline", "neck", str(S.L_SH), 0.35, 1.15, 0.12, 0.30, probe=(0.6, 0.75, 0.9)),
    SegDef("shoulder_r", "_outline", "neck", str(S.R_SH), 0.35, 1.15, 0.12, 0.30, probe=(0.6, 0.75, 0.9)),
]
# which slider(s) drive a segment besides its own part
EXTRA_PARTS = {"thighs": ("legs",), "calves": ("legs",)}
# sliders whose percent is stronger than a plain width change (the pelvis is measured at its widest)
PART_GAIN = {"pelvis": 0.6}
SEG_INDEX = {s.name: i for i, s in enumerate(SEGMENTS)}
N_SEG = len(SEGMENTS)


def _pt(kp: np.ndarray, key: str, conf_min: float = 0.05) -> Optional[np.ndarray]:
    def j(i):
        if np.isfinite(kp[i, :2]).all() and (kp.shape[1] < 3 or kp[i, 2] >= conf_min or not np.isfinite(kp[i, 2])):
            return kp[i, :2].astype(np.float64)
        return None
    if key.isdigit():
        return j(int(key))
    if key == "neck":
        a, b = j(S.L_SH), j(S.R_SH)
        return (a + b) / 2 if a is not None and b is not None else None
    if key == "pelvis":
        a, b = j(S.L_HIP), j(S.R_HIP)
        return (a + b) / 2 if a is not None and b is not None else None
    if key == "head":
        a, b = j(S.L_EAR), j(S.R_EAR)
        if a is not None and b is not None:
            return (a + b) / 2
        n = j(S.NOSE)
        if n is not None:
            ea = a if a is not None else b
            return (n + ea) / 2 if ea is not None else n
        return None
    return None


def segment_endpoints(kp: np.ndarray) -> np.ndarray:
    """(N_SEG, 4) [ax, ay, bx, by] or NaN."""
    out = np.full((N_SEG, 4), np.nan, np.float64)
    for i, s in enumerate(SEGMENTS):
        a, b = _pt(kp, s.a), _pt(kp, s.b)
        if a is not None and b is not None and np.hypot(*(b - a)) > 2:
            out[i] = [a[0], a[1], b[0], b[1]]
    return out


def fallback_widths(kp: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """(N_SEG,) fallback half-width in px."""
    L = np.hypot(ends[:, 2] - ends[:, 0], ends[:, 3] - ends[:, 1])
    fw = np.array([s.fallback for s in SEGMENTS]) * L
    neck = SEG_INDEX["neck"]
    sh = _pt(kp, str(S.L_SH)), _pt(kp, str(S.R_SH))
    if sh[0] is not None and sh[1] is not None:
        fw[neck] = 0.17 * np.hypot(*(sh[0] - sh[1]))
    return fw


# --------------------------------------------------------------------------- person labels from mattes

def label_map(kps: List[Optional[np.ndarray]], shape: Tuple[int, int], scale_xy: Tuple[float, float],
              max_dist_ratio: float = 0.45) -> np.ndarray:
    """Assign each pixel to the nearest skeleton. kps are in source px; output label map at `shape` (h, w).

    -1 means "not near any tracked person" (props, extras, background)."""
    h, w = shape
    best = np.full((h, w), np.inf, np.float32)
    lab = np.full((h, w), -1, np.int16)
    sx, sy = scale_xy
    for p, kp in enumerate(kps):
        if kp is None:
            continue
        xy = kp[:, :2].copy()
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        ok = np.isfinite(xy).all(1) & ((kp[:, 2] > 0.05) if kp.shape[1] > 2 else True)
        if ok.sum() < 3:
            continue
        scale = S.person_scale(np.column_stack([xy, np.ones(len(xy))]))
        if not np.isfinite(scale):
            continue
        m = np.full((h, w), 255, np.uint8)
        th = max(1, int(scale * 0.02))
        for a, b in S.BONES:
            if ok[a] and ok[b]:
                cv2.line(m, tuple(np.round(xy[a]).astype(int)), tuple(np.round(xy[b]).astype(int)), 0, th)
        # torso fill helps waists and wide clothes
        tor = [S.L_SH, S.R_SH, S.R_HIP, S.L_HIP]
        if all(ok[i] for i in tor):
            cv2.fillPoly(m, [np.round(xy[tor]).astype(np.int32)], 0)
        if ok[S.NOSE]:
            cv2.circle(m, tuple(np.round(xy[S.NOSE]).astype(int)), max(2, int(scale * 0.05)), 0, -1)
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        lim = scale * max_dist_ratio
        upd = (dt < best) & (dt < lim)
        # a standing person ends at the feet: what lies further down is floor (a mirror image on a glossy
        # floor, a shadow), never this person
        head = [xy[j, 1] for j in (S.NOSE, S.L_EYE, S.R_EYE) if ok[j]]
        hips = [xy[j, 1] for j in (S.L_HIP, S.R_HIP) if ok[j]]
        if head and hips and np.mean(head) < np.mean(hips):
            top, bot = float(xy[ok, 1].min()), float(xy[ok, 1].max())
            y_lim = int(np.ceil(bot + 0.12 * (bot - top)))
            if y_lim < h:
                upd[y_lim:] = False
        best[upd] = dt[upd]
        lab[upd] = p
    return lab


def measure_widths(kp: np.ndarray, alpha: Optional[np.ndarray], labels: Optional[np.ndarray], pidx: int,
                   scale_xy: Tuple[float, float]) -> np.ndarray:
    """Half-widths (N_SEG, 2) = [side+, side-] in source px from the matte; NaN where unknown."""
    out = np.full((N_SEG, 2), np.nan, np.float32)
    ends = segment_endpoints(kp)
    if alpha is None:
        return out
    h, w = alpha.shape
    sx, sy = scale_xy
    fw = fallback_widths(kp, ends)
    for i, s in enumerate(SEGMENTS):
        e = ends[i]
        if not np.isfinite(e).all():
            continue
        a = np.array([e[0] * sx, e[1] * sy])
        b = np.array([e[2] * sx, e[3] * sy])
        v = b - a
        L = np.hypot(*v)
        if L < 3:
            continue
        u = v / L
        n = np.array([-u[1], u[0]])
        cap = max(4.0, 2.2 * fw[i] * sx)
        steps = np.arange(1, int(cap) + 1, dtype=np.float64)
        res = {1: [], -1: []}
        for t in s.probe:
            c = a + v * t
            for sign in (1, -1):
                pts = c[None, :] + sign * n[None, :] * steps[:, None]
                xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
                yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
                inside = alpha[yi, xi] >= 128
                if labels is not None:
                    inside &= (labels[yi, xi] == pidx) | (labels[yi, xi] == -1)
                outside = ~inside
                oob = (pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)
                if not inside[0]:
                    continue    # axis point itself is outside the silhouette — measurement unusable
                if outside.any():
                    k = int(np.argmax(outside))
                    if oob[k]:
                        continue
                    res[sign].append(steps[k])
        for col, sign in ((0, 1), (1, -1)):
            if len(res[sign]) >= 2:
                out[i, col] = float(np.median(res[sign])) / sx
    return out


def smooth_widths(raw: np.ndarray, fallback: np.ndarray, present: np.ndarray) -> np.ndarray:
    """raw (N, N_SEG, 2) half-widths with NaN; fallback (N, N_SEG). Returns robust, temporally smooth widths.

    Widths are expressed relative to the fallback (which follows the skeleton scale), so the result keeps
    up with zooms and moves while ignoring frame-to-frame silhouette noise."""
    N = raw.shape[0]
    out = np.empty_like(raw)
    for i in range(N_SEG):
        fb = fallback[:, i]
        for c in range(2):
            r = raw[:, i, c] / np.where(fb > 1e-3, fb, np.nan)
            r = np.where((r > 0.45) & (r < 2.4), r, np.nan)
            r = np.minimum(r, 1.6)          # arms or props next to the body are not body width
            valid = np.isfinite(r)
            if valid.sum() >= 3:
                med = float(np.nanmedian(r))
                # rolling median (window 15) then gaussian
                idx = np.arange(N)
                filled = np.interp(idx, idx[valid], r[valid])
                k = 7
                pad = np.pad(filled, k, mode="edge")
                win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
                rm = np.median(win, axis=1)
                g = cv2.GaussianBlur(rm.reshape(-1, 1).astype(np.float32), (1, 0), sigmaX=0, sigmaY=4).ravel()
                # pull towards the clip median where data is thin
                kn = max(1, min(31, N) // 2 * 2 + 1)
                cover = np.convolve(valid.astype(float), np.ones(kn) / kn, mode="same")[:N]
                ratio = cover * g + (1 - cover) * med
            elif valid.any():
                ratio = np.full(N, float(np.nanmedian(r)))
            else:
                ratio = np.ones(N)
            out[:, i, c] = ratio * fb
    out[~present] = np.nan
    return out.astype(np.float32)


# --------------------------------------------------------------------------- body contour points
#
# The silhouette edge measured at many positions along each limb / the torso, on both sides. Positions are
# tied to the skeleton (segment, fraction along it, side), so the same point follows the body through the
# shot. They are the body's equivalent of face landmarks: the warp follows the real outline, and the user
# can move single points to reshape exactly that part.

CONTOUR_K = 9
CONTOUR_SEGS: List[Tuple[str, Tuple[float, float]]] = [
    ("upper_arm_l", (0.28, 0.95)), ("upper_arm_r", (0.28, 0.95)),
    ("forearm_l", (0.08, 0.88)), ("forearm_r", (0.08, 0.88)),
    ("waist", (0.30, 1.02)),
    ("thigh_l", (0.18, 0.94)), ("thigh_r", (0.18, 0.94)),
    ("calf_l", (0.06, 0.90)), ("calf_r", (0.06, 0.90)),
    ("hips", (0.86, 1.22)),
    ("shoulder_l", (0.42, 1.45)), ("shoulder_r", (0.42, 1.45)),
]
NC = len(CONTOUR_SEGS)
CONTOUR_INDEX = {nm: i for i, (nm, _) in enumerate(CONTOUR_SEGS)}
CONTOUR_LABEL = {"upper_arm_l": "왼 위팔", "upper_arm_r": "오른 위팔", "forearm_l": "왼 아래팔", "forearm_r": "오른 아래팔",
                 "waist": "몸통", "thigh_l": "왼 허벅지", "thigh_r": "오른 허벅지", "calf_l": "왼 종아리", "calf_r": "오른 종아리",
                 "hips": "골반", "shoulder_l": "왼 어깨", "shoulder_r": "오른 어깨"}


_RIVALS = {
    "thigh_l": ("thigh_r", "calf_r"), "thigh_r": ("thigh_l", "calf_l"),
    "calf_l": ("calf_r", "thigh_r"), "calf_r": ("calf_l", "thigh_l"),
    "upper_arm_l": ("waist",), "upper_arm_r": ("waist",),
    "forearm_l": ("waist", "thigh_l", "thigh_r"), "forearm_r": ("waist", "thigh_l", "thigh_r"),
    "shoulder_l": ("neck",), "shoulder_r": ("neck",),
}


def contour_ts(ci: int) -> np.ndarray:
    a, b = CONTOUR_SEGS[ci][1]
    return np.linspace(a, b, CONTOUR_K)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def measure_contour(kp: np.ndarray, alpha: Optional[np.ndarray], labels: Optional[np.ndarray], pidx: int,
                    scale_xy: Tuple[float, float], gray: Optional[np.ndarray] = None,
                    gray_scale: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Edge distance from the segment axis (source px) at CONTOUR_K positions per contour segment and side:
    (NC, K, 2) [side +, side -], NaN where the silhouette edge is not seen (other person, frame edge, ...).
    The matte crossing is found with sub-pixel precision and, when a picture is given, snapped to the
    strongest brightness edge right next to it."""
    out = np.full((NC, CONTOUR_K, 2), np.nan, np.float32)
    if alpha is None:
        return out
    ends = segment_endpoints(kp)
    fb = fallback_widths(kp, ends)
    h, w = alpha.shape
    sx, sy = scale_xy
    af = alpha.astype(np.float32) / 255.0 if alpha.dtype == np.uint8 else alpha.astype(np.float32)
    # arms hanging next to the torso belong to the same silhouette: a torso ray that runs into an arm
    # does not see the torso edge (unknown there, the warp falls back to the measured width)
    arms = []
    for nm in ("upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r"):
        j = SEG_INDEX[nm]
        if np.isfinite(ends[j]).all():
            arms.append((ends[j], 1.15 * fb[j]))

    def _in_arm(pts):
        hit = np.zeros(len(pts), bool)
        for e, r in arms:
            a_ = np.array([e[0], e[1]])
            v_ = np.array([e[2], e[3]]) - a_
            L2 = float(v_ @ v_)
            if L2 < 1:
                continue
            tt = np.clip(((pts - a_) @ v_) / L2, 0, 1)
            q = a_[None] + tt[:, None] * v_[None]
            hit |= np.hypot(*(pts - q).T) < r
        return hit

    def _seg_dist(pts, e):
        a_ = np.array([e[0], e[1]])
        v_ = np.array([e[2], e[3]]) - a_
        L2 = max(float(v_ @ v_), 1e-6)
        tt = np.clip(((pts - a_) @ v_) / L2, 0, 1)
        return np.hypot(*(pts - (a_[None] + tt[:, None] * v_[None])).T)

    for ci, (name, _) in enumerate(CONTOUR_SEGS):
        i = SEG_INDEX[name]
        e = ends[i]
        if not np.isfinite(e).all():
            continue
        # a ray that gets closer to another limb's bone than to its own has left this limb (legs inside one
        # skirt, an arm resting on the torso): the edge there belongs to something else
        rivals = [ends[SEG_INDEX[o]] for o in _RIVALS.get(name, ()) if np.isfinite(ends[SEG_INDEX[o]]).all()]
        a = np.array([e[0], e[1]])
        b = np.array([e[2], e[3]])
        v = b - a
        L = float(np.hypot(*v))
        if L < 4:
            continue
        u = v / L
        n = np.array([-u[1], u[0]])
        ts = contour_ts(ci)
        cap = max(6.0, 2.4 * fb[i])                       # source px
        step = 0.5 / max(sx, 1e-6)                        # half a matte pixel
        steps = np.arange(step, cap, step)
        for k, t in enumerate(ts):
            c = a + v * t
            for col, sgn in ((0, 1.0), (1, -1.0)):
                pts = c[None] + sgn * n[None] * steps[:, None]
                mx = pts[:, 0] * sx
                my = pts[:, 1] * sy
                oob = (mx < 0) | (mx > w - 1) | (my < 0) | (my > h - 1)
                al = _bilinear(af, mx, my)
                inside = al >= 0.5
                if labels is not None:
                    li = labels[np.clip(np.round(my).astype(int), 0, h - 1), np.clip(np.round(mx).astype(int), 0, w - 1)]
                    other = (li != pidx) & (li >= 0)
                else:
                    other = np.zeros_like(inside)
                if not inside[0]:
                    continue
                outs = ~inside
                if not outs.any():
                    continue
                j = int(np.argmax(outs))
                if oob[j] or other[min(j, len(other) - 1)] or other[max(0, j - 1)]:
                    continue
                if name in ("waist", "hips") and arms and _in_arm(pts[:j + 1]).any():
                    continue
                if rivals:
                    own = _seg_dist(pts[:j + 1], e)
                    if any((_seg_dist(pts[:j + 1], r) < own * 0.95).any() for r in rivals):
                        continue
                # sub-pixel: linear crossing between j-1 and j
                a0, a1 = al[j - 1] if j > 0 else 1.0, al[j]
                f = (a0 - 0.5) / max(a0 - a1, 1e-6)
                d = steps[j - 1] + f * step if j > 0 else steps[0]
                if gray is not None:
                    # snap to the strongest edge within +-1.5 matte px
                    gx_, gy_ = gray_scale
                    win = np.linspace(d - 3 * step, d + 3 * step, 13)
                    q = c[None] + sgn * n[None] * win[:, None]
                    gv = _bilinear(gray, q[:, 0] * gx_, q[:, 1] * gy_)
                    grad = np.abs(np.gradient(gv))
                    gi = int(np.argmax(grad))
                    if grad[gi] > 6.0 and 0 < gi < len(win) - 1:
                        y0_, y1_, y2_ = grad[gi - 1], grad[gi], grad[gi + 1]
                        den = y0_ - 2 * y1_ + y2_
                        off = 0.5 * (y0_ - y2_) / den if abs(den) > 1e-6 else 0.0
                        de = win[gi] + np.clip(off, -0.5, 0.5) * (win[1] - win[0])
                        d = 0.5 * d + 0.5 * de
                out[ci, k, col] = d
    return out


def smooth_contour(raw: np.ndarray, fallback: np.ndarray, present: np.ndarray) -> np.ndarray:
    """raw (N, NC, K, 2) edge distances with NaN, fallback (N, NC) scale per segment. Robust in time
    (relative to the skeleton's scale, so zooms are followed) and a little along the outline."""
    N = raw.shape[0]
    out = np.full_like(raw, np.nan)
    for ci in range(NC):
        fb = fallback[:, ci]
        for k in range(CONTOUR_K):
            for c in range(2):
                r = raw[:, ci, k, c] / np.where(fb > 1e-3, fb, np.nan)
                r = np.where((r > 0.25) & (r < 3.0), r, np.nan)
                ok = np.isfinite(r)
                if ok.sum() < 2:
                    continue
                idx = np.arange(N)
                filled = np.interp(idx, idx[ok], r[ok])
                kk = 4
                pad = np.pad(filled, kk, mode="edge")
                med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * kk + 1), axis=1)
                bad = ok & (np.abs(r - med) > 0.18 * med)
                r2 = np.where(bad, np.nan, r)
                ok2 = np.isfinite(r2)
                if ok2.sum() < 2:
                    continue
                filled = np.interp(idx, idx[ok2], r2[ok2])
                g = cv2.GaussianBlur(filled.reshape(-1, 1).astype(np.float32), (1, 0), sigmaX=0, sigmaY=1.6).ravel()
                # long gaps (edge not seen) stay unknown, and nothing is invented past the measured frames
                d_prev = np.full(N, 10 ** 9)
                last = -10 ** 9
                for t_ in range(N):
                    if ok2[t_]:
                        last = t_
                    d_prev[t_] = t_ - last
                d_next = np.full(N, 10 ** 9)
                last = 10 ** 9
                for t_ in range(N - 1, -1, -1):
                    if ok2[t_]:
                        last = t_
                    d_next[t_] = last - t_
                keep = ((d_prev <= 10) & (d_next <= 10)) | (np.minimum(d_prev, d_next) <= 2)
                g = np.where(keep, g, np.nan)
                out[:, ci, k, c] = g * fb
        # a little smoothing along the outline (k) for each side
        for c in range(2):
            blk = out[:, ci, :, c]
            ok = np.isfinite(blk)
            sm = blk.copy()
            for k in range(CONTOUR_K):
                lo, hi = max(0, k - 1), min(CONTOUR_K, k + 2)
                wts = np.array([0.25 if j != k else 0.5 for j in range(lo, hi)])
                vals = blk[:, lo:hi]
                okv = np.isfinite(vals)
                num = np.nansum(np.where(okv, vals, 0) * wts, axis=1)
                den = (okv * wts).sum(1)
                sm[:, k] = np.where(ok[:, k] & (den > 0), num / np.maximum(den, 1e-9), np.nan)
            out[:, ci, :, c] = sm
    out[~present] = np.nan
    return out.astype(np.float32)


def contour_points(kp: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(NC, K, 2, 2) edge points in source px from the distances (NaN where unknown)."""
    pts = np.full((NC, CONTOUR_K, 2, 2), np.nan, np.float64)
    ends = segment_endpoints(kp)
    for ci, (name, _) in enumerate(CONTOUR_SEGS):
        e = ends[SEG_INDEX[name]]
        if not np.isfinite(e).all():
            continue
        a = np.array([e[0], e[1]])
        v = np.array([e[2], e[3]]) - a
        L = float(np.hypot(*v))
        if L < 2:
            continue
        n = np.array([-v[1], v[0]]) / L
        for k, t in enumerate(contour_ts(ci)):
            c = a + v * t
            for col, sgn in ((0, 1.0), (1, -1.0)):
                d = dist[ci, k, col]
                if np.isfinite(d):
                    pts[ci, k, col] = c + sgn * n * d
    return pts


def sculpt_key(ci: int, k: int, side: int) -> str:
    return f"{CONTOUR_SEGS[ci][0]}:{k}:{side}"


def sculpt_array(d: Dict[str, float]) -> np.ndarray:
    """User body point edits {"segment:k:side": fraction} -> (NC, K, 2)."""
    out = np.zeros((NC, CONTOUR_K, 2), np.float64)
    for key, v in (d or {}).items():
        try:
            nm, k, side = key.split(":")
            ci = CONTOUR_INDEX[nm]
            out[ci, int(k), int(side)] = float(v)
        except (ValueError, KeyError, IndexError):
            continue
    return out


_LIMB_CONTOURS = {"upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "thigh_l", "thigh_r", "calf_l", "calf_r"}


def side_flips(kp: np.ndarray) -> np.ndarray:
    """(NC,) True where the measured side 0 (+normal) is NOT the person's own reference side, so point edits
    stay on the same part of the body when she turns around:
      limbs     side 0 = the outer side (away from the body, upwards for a raised arm)
      torso     side 0 = the person's left
      shoulders side 0 = the top of the shoulder"""
    out = np.zeros(NC, bool)
    ends = segment_endpoints(kp)
    ls, rs = _pt(kp, str(S.L_SH)), _pt(kp, str(S.R_SH))
    lh, rh = _pt(kp, str(S.L_HIP)), _pt(kp, str(S.R_HIP))
    pts = [q for q in (ls, rs, lh, rh) if q is not None]
    if len(pts) < 2:
        return out
    center = np.mean(pts, 0)
    if ls is not None and rs is not None and lh is not None and rh is not None:
        up = (ls + rs) / 2 - (lh + rh) / 2
    else:
        up = np.array([0.0, -1.0])
    up = up / max(np.hypot(*up), 1e-6)
    left = None
    if ls is not None and rs is not None:
        left = ls - rs
    elif lh is not None and rh is not None:
        left = lh - rh
    for ci, (nm, _) in enumerate(CONTOUR_SEGS):
        e = ends[SEG_INDEX[nm]]
        if not np.isfinite(e).all():
            continue
        a = e[:2]
        v = e[2:] - a
        L = float(np.hypot(*v))
        if L < 2:
            continue
        n = np.array([-v[1], v[0]]) / L
        if nm in _LIMB_CONTOURS:
            m = a + 0.5 * v
            o = m - center
            o = o / max(np.hypot(*o), 1e-6)
            ref = o + (0.6 if nm.startswith(("upper", "fore")) else 0.3) * up
        elif nm.startswith("shoulder"):
            ref = up
        else:
            if left is None:
                continue
            ref = left
        out[ci] = float(n @ ref) < 0
    return out


def sculpt_to_measured(arr: np.ndarray, flips: np.ndarray) -> np.ndarray:
    """Point edits by the person's sides -> by the measured sides of this frame."""
    out = arr.copy()
    out[flips] = out[flips][..., ::-1]
    return out


def part_sculpt(kp: np.ndarray, arm_under: float, belly: float) -> Optional[np.ndarray]:
    """Edge changes (NC, K, 2) by measured sides for the one-sided body sliders:
      arm_under  the hanging (lower / outer-back) side of the upper arms
      belly      the front of the torso, by how much the person is seen from the side"""
    if abs(arm_under) < 1e-4 and abs(belly) < 1e-4:
        return None
    out = np.zeros((NC, CONTOUR_K, 2), np.float64)
    ends = segment_endpoints(kp)
    ls, rs = _pt(kp, str(S.L_SH)), _pt(kp, str(S.R_SH))
    lh, rh = _pt(kp, str(S.L_HIP)), _pt(kp, str(S.R_HIP))
    if ls is None or rs is None:
        return None
    neck = (ls + rs) / 2
    if lh is not None and rh is not None:
        pel = (lh + rh) / 2
        up = neck - pel
    else:
        pel = None
        up = np.array([0.0, -1.0])
    up = up / max(np.hypot(*up), 1e-6)
    right = np.array([-up[1], up[0]])
    center = neck if pel is None else (neck + pel) / 2
    kk = np.arange(CONTOUR_K)
    if abs(arm_under) > 1e-4:
        prof = np.exp(-0.5 * ((kk - 4.5) / 2.2) ** 2)
        for nm in ("upper_arm_l", "upper_arm_r"):
            ci = CONTOUR_INDEX[nm]
            e = ends[SEG_INDEX[nm]]
            if not np.isfinite(e).all():
                continue
            v = e[2:] - e[:2]
            L = float(np.hypot(*v))
            if L < 2:
                continue
            n = np.array([-v[1], v[0]]) / L
            m = e[:2] + 0.5 * v
            o = m - center
            o = o / max(np.hypot(*o), 1e-6)
            ref = -up + 0.6 * o                    # the side gravity pulls on, or the outer back of a hanging arm
            side = 0 if float(n @ ref) > 0 else 1
            out[ci, :, side] += arm_under * prof
    if abs(belly) > 1e-4 and pel is not None:
        sw = float(np.hypot(*(ls - rs)))
        tl = float(np.hypot(*(neck - pel)))
        profile = float(np.clip(1.0 - (sw / max(tl, 1e-6) - 0.25) / 0.40, 0.0, 1.0))
        nose = _pt(kp, str(S.NOSE))
        head = _pt(kp, "head")
        if profile > 0.02 and nose is not None:
            ref_pt = head if head is not None else neck
            fs = float((nose - ref_pt) @ right)
            if abs(fs) > 1e-6:
                front = right * np.sign(fs)
                for nm in ("waist", "hips"):
                    ci = CONTOUR_INDEX[nm]
                    e = ends[SEG_INDEX[nm]]
                    if not np.isfinite(e).all():
                        continue
                    v = e[2:] - e[:2]
                    L = float(np.hypot(*v))
                    n = np.array([-v[1], v[0]]) / max(L, 1e-6)
                    side = 0 if float(n @ front) > 0 else 1
                    ts = contour_ts(ci)
                    w = np.exp(-0.5 * ((ts - 0.84) / 0.13) ** 2)
                    out[ci, :, side] += belly * w * profile
    return out
