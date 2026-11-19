"""Deformation field: skeleton-driven, monotonic (fold-free by construction for the allowed range),
attenuated around protected people, background lines, shapes and the frame border.

The field D is a backward map: an output pixel p samples the source at p + D(p).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import body as B
from . import face as FW
from ..core.params import FACE_KEYS, GEOMETRY_KEYS, HEAD_KEYS
from ..core.project import Cut
from ..track import skeleton as S

MAX_COEF = 0.45     # |a| limit per segment (a = 1/(1+v) - 1), keeps the map monotonic


@dataclass
class PersonInput:
    p: int
    kp: np.ndarray                 # (17,3) source px
    widths: np.ndarray             # (N_SEG, 2) half-widths source px (NaN -> fallback)
    amounts: Dict[str, float]      # % width change per part (negative = slimmer)
    width_mult: Dict[str, float]
    strength: float                # 0..1
    seg_w: Optional[np.ndarray] = None   # (N_SEG,) 0..1 fade for segments whose joints come and go
    sh_w: float = 1.0                    # shoulder weight
    face: Optional[np.ndarray] = None    # (68,3) smoothed face landmarks, source px
    face_w: float = 0.0                  # 0..1 how usable the face is (frontal, visible)
    head_w: float = 0.0                  # 0..1 head edits (also with the back to the camera)
    face_geom: Optional[FW.FaceGeom] = None
    head_geom: Optional[FW.HeadGeom] = None
    sculpt: Dict[int, List[float]] = field(default_factory=dict)
    hands: List[np.ndarray] = field(default_factory=list)     # hand outlines (source px) kept out of the arm warp
    body_half: float = float("nan")      # half body width for the body width ratio
    feather: float = 1.5                 # how softly changes fade (0.5 .. 2.5)
    contour: Optional[np.ndarray] = None # (NC, K, 2) measured edge distances along limbs / torso (source px)
    body_sculpt: Optional[np.ndarray] = None  # (NC, K, 2) user edge changes (fraction, 0 = untouched)
    cloth_keep: Optional[np.ndarray] = None   # bool, matte resolution: this person's clothes that stay as they are
    parts: Optional[object] = None            # track.parts.PartMaps: hair / face+ears weights around the head


# the warp around the retouched people, the same for every cut (the 보호 tab's values, fixed at its defaults)
WARP_SETTINGS = dict(others=True, falloff_pct=3.0, border=True, reach=1.9)


def legacy_protection(cut) -> List[str]:
    """What an older cut had that is no longer applied (shown once when such a project is opened): the separate
    mode, drawn protection lines or shapes, protection settings other than the fixed ones. Nothing is changed."""
    out = []
    if getattr(cut, "mode", "warp") == "separate":
        out.append("분리 + 배경 복원 모드")
    pr = getattr(cut, "protect", None)
    if pr is not None:
        if pr.lines or pr.shapes:
            out.append("직선·영역 보호")
        if (not pr.others or abs(pr.falloff_pct - WARP_SETTINGS["falloff_pct"]) > 1e-6 or
                pr.border != WARP_SETTINGS["border"] or abs(pr.reach - WARP_SETTINGS["reach"]) > 1e-6):
            out.append("보호 설정")
    return out


@dataclass
class FieldInput:
    W: int
    H: int
    targets: List[PersonInput]
    alpha: Optional[np.ndarray] = None      # matte (h,w) uint8
    labels: Optional[np.ndarray] = None     # (h,w) int16 person index
    # people who are not retouched are kept out of the warp (always on since 1.22.2, no setting); the transition
    # width, the reach beyond the body and the frame edge hold are fixed too (the 보호 tab, its drawn lines and
    # shapes and the separate mode were removed: see warp_settings)
    protect_others: bool = True
    falloff_pct: float = 3.0
    border: bool = True
    reach: float = 1.9
    active: Optional[Tuple[float, float, float, float]] = None   # picture area inside letterbox bars (source px)


@dataclass
class FieldResult:
    D: np.ndarray                  # (gh, gw, 2) in output px
    out_w: int
    out_h: int
    max_disp: float
    raw_max: float                 # before protection
    repaired: float = 0.0          # share of the active field that was softened to prevent folds
    local_max: float = -1.0        # largest shape-only displacement (without the even length stretch)

    def empty(self) -> bool:
        return self.max_disp < 1e-3


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _window(t: np.ndarray, s: B.SegDef, soft: float = 1.0) -> np.ndarray:
    if s.gauss[1] > 0:
        c, sg = s.gauss
        sg = sg * (0.85 + 0.15 * soft)
        return np.exp(-0.5 * ((t - c) / sg) ** 2)
    e = s.ramp * soft
    return _smoothstep(s.t0 - e, s.t0 + e, t) * (1 - _smoothstep(s.t1 - e, s.t1 + e, t))


def _profile(u: np.ndarray, R: float) -> np.ndarray:
    """B(u) for u >= 0: identity in the core, C1 smooth decay to 0 at u = R."""
    out = np.where(u <= 1, u, u * (1 - _smoothstep(0, 1, (u - 1) / (R - 1))))
    return np.where(u >= R, 0.0, out)


def _cr(t: np.ndarray, t0: float, h: float, vals: np.ndarray) -> np.ndarray:
    """Catmull-Rom curve through vals sampled every h from t0 (held constant beyond the ends)."""
    n = len(vals)
    u = np.clip((t - t0) / h, 0, n - 1 - 1e-9)
    i = np.floor(u).astype(int)
    f = u - i
    p0 = vals[np.maximum(i - 1, 0)]
    p1 = vals[i]
    p2 = vals[np.minimum(i + 1, n - 1)]
    p3 = vals[np.minimum(i + 2, n - 1)]
    f2, f3 = f * f, f * f * f
    return 0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f3)


def _ss(t: np.ndarray, t0: float, h: float, vals: np.ndarray) -> np.ndarray:
    """Smoothstep blend between samples (no overshoot: a single moved point gives a clean bump)."""
    n = len(vals)
    u = np.clip((t - t0) / h, 0, n - 1 - 1e-9)
    i = np.floor(u).astype(int)
    f = u - i
    f = f * f * (3 - 2 * f)
    return vals[i] * (1 - f) + vals[np.minimum(i + 1, n - 1)] * f


def _fill_k(v: np.ndarray, fallback: float) -> np.ndarray:
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(len(v), fallback)
    k = np.arange(len(v))
    return np.interp(k, k[ok], v[ok])


CONTOUR_SLOPE = 0.45   # steepest edge change along the segment the warp follows (px per px)


def contour_profile(ci: int, dist: Optional[np.ndarray], wp: float, wm: float, fb: float, L: float):
    """Measured edge distances of a contour segment as two smooth curves along the segment (side +, side -),
    kept within sane limits of the bone-length estimate. None when nothing was measured.

    Steps in the outline (a skirt hem, a sleeve end) are bridged from above: the curve may not change faster
    than CONTOUR_SLOPE, taking the wider side, so the hem moves as one piece instead of being sheared."""
    if dist is None or not np.isfinite(dist).any():
        return None
    ts = B.contour_ts(ci)
    step = (ts[1] - ts[0]) * max(L, 1.0) * CONTOUR_SLOPE
    kk = np.arange(len(ts))
    cone = np.abs(kk[:, None] - kk[None, :]) * step
    out = []
    for col, w0 in ((0, wp), (1, wm)):
        v = _fill_k(dist[:, col].astype(np.float64), w0)
        v = np.clip(v, 0.5 * fb, 1.9 * fb)
        v = (v[None, :] - cone).max(1)          # upper envelope with limited slope
        out.append(v)
    return ts, np.stack(out, 1)


def segment_disp(X, Y, ends, a_coef, w_plus, w_minus, R, seg: B.SegDef, soft: float = 1.0,
                 prof=None, delta=None):
    """Push the silhouette of one segment in or out along its normal (backward map). The core (inside the
    edge) scales by 1 + a, beyond the edge the change fades out by R edge distances.

    prof:  (ts, (K,2) edge distances) measured along the segment -> the edge is followed point by point
    delta: (ts, (K,2) fractions) the user's per point edge changes, on top of the slider amount"""
    ax, ay, bx, by = ends
    vx, vy = bx - ax, by - ay
    L = np.hypot(vx, vy)
    ux, uy = vx / L, vy / L
    nx, ny = -uy, ux
    px, py = X - ax, Y - ay
    t = (px * ux + py * uy) / L
    d = px * nx + py * ny
    plus = d >= 0
    if prof is not None:
        ts, wv = prof
        h = ts[1] - ts[0]
        w = np.where(plus, _cr(t, ts[0], h, wv[:, 0]), _cr(t, ts[0], h, wv[:, 1]))
        # far from the measured stretch, back to the single width
        blend = _smoothstep(ts[0] - 2 * h, ts[0], t) * (1 - _smoothstep(ts[-1], ts[-1] + 2 * h, t))
        w = blend * w + (1 - blend) * np.where(plus, w_plus, w_minus)
    else:
        w = np.where(plus, w_plus, w_minus)
    w = np.maximum(w, 1.0)
    u = np.abs(d) / w
    a = a_coef * _window(t, seg, soft)
    if delta is not None:
        ts, dv = delta
        h = ts[1] - ts[0]
        z = np.zeros((1, 2))
        dz = np.concatenate([z, dv, z], 0)
        t0 = ts[0] - h
        dl = np.where(plus, _ss(t, t0, h, dz[:, 0]), _ss(t, t0, h, dz[:, 1]))
        dl = np.where((t < t0) | (t > ts[-1] + h), 0.0, dl)
        dl = np.clip(dl, -0.6, 1.0)
        a = (1 + a) / (1 + dl) - 1
    a = np.clip(a, -MAX_COEF, MAX_COEF)
    mag = a * w * _profile(u, R)
    mag = np.sign(d) * mag
    return mag * nx, mag * ny


def _box_slices(X, Y, x0, y0, x1, y1):
    """Grid index slices covering the source-px box (X/Y are regular grids)."""
    xs = X[0]
    ys = Y[:, 0]
    i0 = int(np.searchsorted(xs, x0, "left"))
    i1 = int(np.searchsorted(xs, x1, "right"))
    j0 = int(np.searchsorted(ys, y0, "left"))
    j1 = int(np.searchsorted(ys, y1, "right"))
    return slice(max(0, j0 - 1), min(len(ys), j1 + 1)), slice(max(0, i0 - 1), min(len(xs), i1 + 1))


def shoulder_disp(X, Y, kp, frac: float):
    """Pull each shoulder towards the neck by `frac` of the half shoulder width (frac > 0 narrows)."""
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    ls, rs = kp[S.L_SH, :2], kp[S.R_SH, :2]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return dx, dy
    m = (ls + rs) / 2
    for s in (ls, rs):
        v = s - m
        L = np.linalg.norm(v)
        if L < 2:
            continue
        R = max(0.95 * L, 3.2 * abs(frac) * L)
        c = s - frac * v                       # new shoulder position (output space)
        sl = _box_slices(X, Y, c[0] - R, c[1] - R, c[0] + R, c[1] + R)
        Xs, Ys = X[sl], Y[sl]
        r2 = ((Xs - c[0]) ** 2 + (Ys - c[1]) ** 2) / (R * R)
        k = np.clip(1 - r2, 0, None) ** 3
        dx[sl] += frac * v[0] * k
        dy[sl] += frac * v[1] * k
    return dx, dy


def _torso_poly(kp: np.ndarray, widths, fb, contour=None) -> Optional[Tuple[np.ndarray, float]]:
    """Torso outline (source px) below the armpits, from the joints and the measured waist / hip widths,
    plus a feather width. None without shoulders and hips."""
    ls, rs, lh, rh = kp[S.L_SH, :2], kp[S.R_SH, :2], kp[S.L_HIP, :2], kp[S.R_HIP, :2]
    if not all(np.isfinite(q).all() for q in (ls, rs, lh, rh)):
        return None
    neck, pel = (ls + rs) / 2, (lh + rh) / 2
    ax = pel - neck
    L = float(np.hypot(*ax))
    if L < 4:
        return None
    u = ax / L
    n = np.array([-u[1], u[0]])
    wi = B.SEG_INDEX["waist"]
    ww = widths[wi] if widths is not None else np.array([np.nan, np.nan])
    w0 = [float(x) if np.isfinite(x) and x > 1 else float(fb[wi]) for x in ww]
    if contour is not None and np.isfinite(contour[B.CONTOUR_INDEX["waist"]]).any():
        # the measured torso outline (fallback width where an arm hid it), started up at the armpits
        ci = B.CONTOUR_INDEX["waist"]
        ts = B.contour_ts(ci)
        d = contour[ci].astype(np.float64)
        sides = []
        for col in (0, 1):
            v = d[:, col]
            v = np.where(np.isfinite(v), v, w0[col])
            sides.append(np.maximum(v, 0.6 * w0[col]))
        tt = np.r_[0.18, ts]
        p0 = [neck + u * t_ * L + n * sides[0][max(0, i - 1)] for i, t_ in enumerate(tt)]
        p1 = [neck + u * t_ * L - n * sides[1][max(0, i - 1)] for i, t_ in enumerate(tt)]
        poly = np.array(p0 + p1[::-1])
        return poly, 0.10 * max(w0)
    top = neck + u * 0.22 * L
    mid = neck + u * 0.62 * L
    bot = pel + u * 0.05 * L
    poly = np.array([top + n * w0[0] * 0.95, mid + n * w0[0] * 0.98, bot + n * w0[0] * 1.05,
                     bot - n * w0[1] * 1.05, mid - n * w0[1] * 0.98, top - n * w0[1] * 0.95])
    return poly, 0.12 * max(w0)


def _invert_local(X, Y, fwd, iters: int = 4):
    """Backward map of a forward displacement given as a function: D(p) = -V(p + D(p))."""
    P = np.stack([X, Y], -1)
    D = -fwd(P)
    for _ in range(iters - 1):
        D = -fwd(P + D)
    return D[..., 0], D[..., 1]


def _shoulder_line_fwd(Q: np.ndarray, kp: np.ndarray, k: float, feather: float = 1.5) -> np.ndarray:
    """Forward displacement for 'straight shoulders': the outer part of each shoulder slope goes up
    (k > 0), most at the shoulder tip, nothing at the neck."""
    V = np.zeros(Q.shape)
    ls, rs = kp[S.L_SH, :2], kp[S.R_SH, :2]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return V
    ax = body_axis(kp)
    up = -ax[1] if ax is not None else np.array([0.0, -1.0])
    neck = (ls + rs) / 2
    f = 0.85 + 0.15 * feather
    for sh in (ls, rs):
        a = neck + (sh - neck) * 0.25
        b = sh + (sh - neck) * 0.12
        ab = b - a
        L = float(np.hypot(*ab))
        if L < 3:
            continue
        t = ((Q[..., 0] - a[0]) * ab[0] + (Q[..., 1] - a[1]) * ab[1]) / (L * L)
        nu = up - (up @ ab) / (L * L) * ab
        nu = nu / max(np.hypot(*nu), 1e-6)
        dn = (Q[..., 0] - a[0]) * nu[0] + (Q[..., 1] - a[1]) * nu[1]
        # the lift grows from the neck to the shoulder tip (which lies well beyond the joint, over the
        # deltoid) and fades out along the arm
        ws = _smoothstep(0.15, 1.3, t) * (1 - _smoothstep(1.55, 1.55 + 0.5 * f, t))
        wn = np.exp(-0.5 * ((dn - 0.15 * L) / (0.5 * L * f)) ** 2)
        m = k * 0.32 * L * ws * wn
        V[..., 0] += up[0] * m
        V[..., 1] += up[1] * m
    return V


def _hip_up_fwd(Q: np.ndarray, kp: np.ndarray, widths, k: float, feather: float = 1.5) -> np.ndarray:
    """Forward displacement for 'hip up': the area around the lower buttock line (upper thighs) goes up."""
    V = np.zeros(Q.shape)
    ax = body_axis(kp)
    if ax is None:
        return V
    up = -ax[1]
    ends = B.segment_endpoints(kp)
    fb = B.fallback_widths(kp, ends)
    f = 0.85 + 0.15 * feather
    for seg in ("thigh_l", "thigh_r"):
        i = B.SEG_INDEX[seg]
        e = ends[i]
        if not np.isfinite(e).all():
            continue
        a = e[:2]
        v = e[2:] - a
        L = float(np.hypot(*v))
        if L < 4:
            continue
        n = np.array([-v[1], v[0]]) / L
        t = ((Q[..., 0] - a[0]) * v[0] + (Q[..., 1] - a[1]) * v[1]) / (L * L)
        d = (Q[..., 0] - a[0]) * n[0] + (Q[..., 1] - a[1]) * n[1]
        wd = fb[i]
        if widths is not None and np.isfinite(widths[i]).any():
            wd = float(np.nanmax(widths[i]))
        wd *= 1.35
        wt = np.exp(-0.5 * ((t - 0.22) / (0.13 * f)) ** 2)
        wl = 1 - _smoothstep(0.8, 0.8 + 0.7 * f, np.abs(d) / max(wd, 1.0))
        m = k * 0.10 * L * wt * wl
        V[..., 0] += up[0] * m
        V[..., 1] += up[1] * m
    return V


def _soft_inside(X, Y, poly: np.ndarray, feather_px: float) -> np.ndarray:
    g = FW._Grid(X, Y)
    m = g.inside(poly)
    sig = feather_px / max(g.hx, 1e-6)
    if sig > 0.3:
        m = cv2.GaussianBlur(m, (0, 0), sig)
        m = np.clip(m * 1.6, 0, 1)
    return m


def person_field(X, Y, pi: PersonInput, reach: float, part: str = "all") -> Tuple[np.ndarray, np.ndarray]:
    """Backward displacement of one person's edits. part: 'all', 'body' (everything but face/head) or 'face'."""
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    kp = pi.kp
    if not np.isfinite(kp[:, :2]).any():
        return dx, dy
    if part == "face":
        return _face_part(X, Y, pi, dx, dy)
    ends = B.segment_endpoints(kp)
    fb = B.fallback_widths(kp, ends)
    for i, s in enumerate(B.SEGMENTS):
        amt = pi.amounts.get(s.part, 0.0) + sum(pi.amounts.get(x, 0.0) for x in B.EXTRA_PARTS.get(s.part, ()))
        amt = amt / 100.0 * pi.strength * B.PART_GAIN.get(s.part, 1.0)
        sw_i = float(pi.seg_w[i]) if pi.seg_w is not None else 1.0
        amt *= sw_i
        ci = B.CONTOUR_INDEX.get(s.name)
        dl = None
        if ci is not None and pi.body_sculpt is not None:
            dv = pi.body_sculpt[ci] * pi.strength * sw_i
            if np.abs(dv).max() > 1e-4:
                dl = (B.contour_ts(ci), dv)
        if (abs(amt) < 1e-4 and dl is None) or not np.isfinite(ends[i]).all():
            continue
        v = max(-0.6, min(1.0, amt))
        a_coef = float(np.clip(1.0 / (1.0 + v) - 1.0, -MAX_COEF, MAX_COEF))
        wp, wm = pi.widths[i] if pi.widths is not None else (np.nan, np.nan)
        wp = wp if np.isfinite(wp) and wp > 1 else fb[i]
        wm = wm if np.isfinite(wm) and wm > 1 else fb[i]
        mult = pi.width_mult.get(s.part, 1.0)
        prof = None
        if ci is not None and pi.contour is not None:
            e_ = ends[i]
            prof = contour_profile(ci, pi.contour[ci], wp, wm, fb[i], float(np.hypot(e_[2] - e_[0], e_[3] - e_[1])))
            if prof is not None:
                prof = (prof[0], prof[1] * mult)
        a_max = abs(a_coef)
        if dl is not None:
            a_max = max(a_max, float(np.abs(1.0 / (1.0 + np.clip(dl[1], -0.6, 1.0)) - 1.0).max()))
        # how far beyond the body edge the change fades: short, but long enough for strong amounts
        R = max(reach, 1.0 + 3.2 * min(a_max, MAX_COEF))
        if s.part == "neck":
            R = min(R, 2.0)
        wmax = max(wp, wm) if prof is None else max(wp * mult, wm * mult, float(prof[1].max())) / mult
        ext = R * wmax * mult
        e = ends[i]
        Lseg = np.hypot(e[2] - e[0], e[3] - e[1])
        pad = ext + 0.35 * Lseg
        sl = _box_slices(X, Y, min(e[0], e[2]) - pad, min(e[1], e[3]) - pad, max(e[0], e[2]) + pad, max(e[1], e[3]) + pad)
        if X[sl].size == 0:
            continue
        soft = 0.7 + 0.3 * pi.feather
        ddx, ddy = segment_disp(X[sl], Y[sl], e, a_coef, wp * mult, wm * mult, R, s, soft, prof, dl)
        if s.part == "neck" and pi.face_geom is not None:
            # the neck must not drag the jaw and chin along
            keep = 1.0 - _soft_inside(X[sl], Y[sl], pi.face_geom.contour(), 0.05 * pi.face_geom.Wf)
            ddx *= keep
            ddy *= keep
        if s.part == "upper_arm":
            # the upper arm's underside borders the torso at the armpit: the torso must not follow it
            tp = _torso_poly(kp, pi.widths, fb, pi.contour)
            if tp is not None:
                keep = 1.0 - _soft_inside(X[sl], Y[sl], tp[0], tp[1])
                ddx *= keep
                ddy *= keep
        if s.part == "forearm" and pi.hands:
            keep = np.ones_like(ddx)
            for hp in pi.hands:
                keep *= 1.0 - _soft_inside(X[sl], Y[sl], hp, 6.0)
            ddx *= keep
            ddy *= keep
        dx[sl] += ddx
        dy[sl] += ddy
    sh = pi.amounts.get("shoulders", 0.0) / 100.0 * pi.strength * pi.sh_w
    if abs(sh) > 1e-4:
        frac = float(np.clip(-sh, -0.3, 0.3))   # -10% width => pull inward by 10%
        ddx, ddy = shoulder_disp(X, Y, kp, frac)
        dx += ddx
        dy += ddy
    sl_ = pi.amounts.get("shoulder_line", 0.0) / 100.0 * pi.strength * pi.sh_w
    if abs(sl_) > 1e-4:
        ddx, ddy = _invert_local(X, Y, lambda Q: _shoulder_line_fwd(Q, kp, sl_, pi.feather))
        dx += ddx
        dy += ddy
    hu = pi.amounts.get("hip_up", 0.0) / 100.0 * pi.strength
    if abs(hu) > 1e-4:
        ddx, ddy = _invert_local(X, Y, lambda Q: _hip_up_fwd(Q, kp, pi.widths, hu, pi.feather))
        dx += ddx
        dy += ddy
    bw = pi.amounts.get("body_width", 0.0) / 100.0 * pi.strength
    if abs(bw) > 1e-4:
        ddx, ddy = body_width_disp(X, Y, kp, bw, pi.body_half, reach)
        dx += ddx
        dy += ddy
    if part == "body":
        return dx, dy
    return _face_part(X, Y, pi, dx, dy)


def _face_part(X, Y, pi: PersonInput, dx, dy):
    kp = pi.kp
    if pi.face_geom is not None and (pi.face_w > 0.02 or pi.head_w > 0.02):
        # face edits fade with how well the face is seen; head edits (hair included) go on when she turns away
        A = {k: pi.amounts.get(k, 0.0) / 100.0 * pi.face_w for k in FACE_KEYS}
        A.update({k: pi.amounts.get(k, 0.0) / 100.0 * max(pi.head_w, pi.face_w) for k in HEAD_KEYS})
        A["_feather"] = pi.feather
        if any(abs(v) > 1e-5 for k, v in A.items() if k not in ("sculpt", "_feather")) or \
                (pi.sculpt and abs(A.get("sculpt", 0)) > 1e-4):
            box = FW.face_extent(pi.face_geom, pi.head_geom, A)
            sl = _box_slices(X, Y, box[0, 0], box[0, 1], box[1, 0], box[1, 1])
            if X[sl].shape[0] > 2 and X[sl].shape[1] > 2:
                ddx, ddy = FW.face_backward(X[sl], Y[sl], pi.face_geom, pi.head_geom, A, pi.sculpt,
                                            pi.strength, pi.feather, pi.parts)
                # arms and hands near the face are not part of the face: keep them still
                # the fade must be soft next to a big change (a much larger head next to a raised arm), or the
                # picture folds between the still arm and the moving head
                dmax = float(np.hypot(ddx, ddy).max()) if ddx.size else 0.0
                feath = max(0.05 * pi.face_geom.Wf * pi.feather, 1.2 * dmax)
                keep = 1.0 - limbs_mask(X[sl], Y[sl], kp, pi.widths, pi.hands, feath)
                dx[sl] += ddx * keep
                dy[sl] += ddy * keep
    return dx, dy


def limbs_mask(X, Y, kp: np.ndarray, widths, hands, feather_px: float) -> np.ndarray:
    """Soft mask (0..1) of the person's arms and hands on the grid window."""
    g = FW._Grid(X, Y)
    h, w = g.shape
    m = np.zeros((h, w), np.uint8)
    segs = [(S.L_SH, S.L_EL, "upper_arm_l"), (S.L_EL, S.L_WR, "forearm_l"),
            (S.R_SH, S.R_EL, "upper_arm_r"), (S.R_EL, S.R_WR, "forearm_r")]
    for a, b, name in segs:
        pa, pb = kp[a, :2], kp[b, :2]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        L = float(np.hypot(*(pb - pa)))
        hw = np.nan
        if widths is not None:
            ww = widths[B.SEG_INDEX[name]]
            if np.isfinite(ww).any():
                hw = float(np.nanmax(ww))
        if not np.isfinite(hw):
            hw = 0.16 * L
        if a in (S.L_SH, S.R_SH):
            pa = pa + (pb - pa) * 0.35        # the shoulder itself belongs to the body
        q = np.round(g.to_g(np.array([pa, pb])) * 4).astype(np.int32)
        th = max(1, int(round(2.3 * hw / max(g.hx, 1e-6))))
        cv2.polylines(m, [q], False, 255, th, cv2.LINE_AA, 2)
    for hp in hands or []:
        q = np.round(g.to_g(np.asarray(hp)) * 4).astype(np.int32)
        cv2.fillPoly(m, [q], 255, cv2.LINE_AA, 2)
    out = m.astype(np.float32) / 255.0
    sg = feather_px / max(g.hx, 1e-6)
    if sg > 0.3 and out.any():
        out = np.clip(cv2.GaussianBlur(out, (0, 0), sg) * 1.4, 0, 1)
    return out


def body_axis(kp: np.ndarray):
    """Neck point, unit axis towards the feet and the body length (source px), or None."""
    ls, rs, lh, rh = kp[S.L_SH, :2], kp[S.R_SH, :2], kp[S.L_HIP, :2], kp[S.R_HIP, :2]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return None
    neck = (ls + rs) / 2
    pel = (lh + rh) / 2 if np.isfinite(lh).all() and np.isfinite(rh).all() else None
    an = [kp[j, :2] for j in (S.L_AN, S.R_AN) if np.isfinite(kp[j, :2]).all()]
    if an:
        foot = np.mean(an, axis=0)
    elif pel is not None:
        foot = pel + 1.25 * (pel - neck)
    else:
        return None
    v = foot - neck
    L = float(np.hypot(*v))
    if L < 10:
        return None
    return neck, v / L, L, pel


def body_width_disp(X, Y, kp: np.ndarray, amt: float, half: float, reach: float):
    """Horizontal (across the body axis) scaling of everything below the neck."""
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    ax = body_axis(kp)
    if ax is None:
        return dx, dy
    neck, u, L, _ = ax
    sw = np.hypot(*(kp[S.L_SH, :2] - kp[S.R_SH, :2]))
    W = half if np.isfinite(half) and half > 2 else max(0.62 * sw, 0.16 * L)
    v = max(-0.5, min(0.6, amt))
    a_coef = float(np.clip(1.0 / (1.0 + v) - 1.0, -MAX_COEF, MAX_COEF))
    R = max(reach, 1.0 + 3.2 * abs(a_coef))
    n = np.array([-u[1], u[0]])
    t = ((X - neck[0]) * u[0] + (Y - neck[1]) * u[1]) / L
    d = (X - neck[0]) * n[0] + (Y - neck[1]) * n[1]
    win = _smoothstep(0.0, 0.08, t) * (1 - _smoothstep(1.0, 1.12, t))
    mag = np.sign(d) * a_coef * W * _profile(np.abs(d) / W, R) * win
    return mag * n[0], mag * n[1]


def shape_disp(X, Y, kp: np.ndarray, height: float, leg: float, lateral: Optional[Tuple[float, float, float]] = None):
    """Backward map (source px) of the length changes. The feet stay planted: a band along the body axis is
    stretched evenly (from the neck for height, from the hips for leg length, down to the lowest ankle) and
    everything above it rises with the body. The lift fades out above the highest point of the person (head,
    raised hands) and, sideways, beyond the person (lateral = (centre, half width, fall-off) across the axis),
    so the background only moves right around the person: the floor, the feet, their shadow or reflection and
    everything beside and below stay where they are."""
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    ax = body_axis(kp)
    if ax is None:
        return dx, dy
    neck, u, L, pel = ax
    an = [float((kp[j, :2] - neck) @ u) for j in (S.L_AN, S.R_AN) if np.isfinite(kp[j, :2]).all()]
    s_foot = max(an) if an else L
    bands = []
    if abs(height) > 1e-4:
        bands.append((0.0, s_foot, height))
    if abs(leg) > 1e-4 and pel is not None:
        s0 = float((pel - neck) @ u)
        if s_foot > s0 + 4:
            bands.append((s0, s_foot, leg))
    if not bands:
        return dx, dy
    s = (X - neck[0]) * u[0] + (Y - neck[1]) * u[1]

    def W(sv):
        out = np.zeros_like(sv)
        for s0, s1, v in bands:
            r = max(4.0, 0.07 * (s1 - s0))
            # integral of a smooth (linear-ramp, C0) indicator: quadratic ramps at both ends
            a0, a1 = s0 - r / 2, s0 + r / 2
            b0, b1 = s1 - r / 2, s1 + r / 2
            x = np.clip(sv, a0, a1)
            ramp_in = (x - a0) ** 2 / (2 * r)
            mid = np.clip(sv, a1, b0) - a1
            x2 = np.clip(sv, b0, b1)
            ramp_out = (x2 - b0) - (x2 - b0) ** 2 / (2 * r)
            out += v * (ramp_in + mid + ramp_out)
        return out

    big = 1e9
    w_tot = float(W(np.array([big]))[0])
    # how far above the person the lift fades out: wide enough that the background above is only gently
    # squeezed (the fade must also stay far gentler than the lift to keep the map one-to-one)
    okk = np.isfinite(kp[:, :2]).all(1)
    s_top = float(((kp[okk, :2] - neck) @ u).min()) - 0.1 * L if okk.any() else -0.4 * L
    fade = max(0.6 * L, 4.0 * abs(w_tot))
    lat = None
    if lateral is not None:
        c, half, fall = lateral
        n = np.array([-u[1], u[0]])
        dn = np.abs((X - neck[0]) * n[0] + (Y - neck[1]) * n[1] - c)
        fall = max(fall, 3.0 * abs(w_tot))
        lat = 1 - _smoothstep(half, half + fall, dn)

    def lift(sv):
        # forward move along u of a source point at sv (negative: upwards when taller)
        e = W(sv) - w_tot
        return e * _smoothstep(s_top - fade, s_top, sv) * (lat if lat is not None else 1.0)

    src = s.copy()
    for _ in range(6):
        src = s - lift(src)
    disp = src - s
    return disp * u[0], disp * u[1]


def lateral_extent(kp: np.ndarray, labels=None, alpha=None, pidx: int = -1, gx=None, gy=None):
    """(centre, half width, fall-off) across the body axis for the length changes: the person's width plus a
    margin, falling off over the free space next to them (less when someone stands close)."""
    ax = body_axis(kp)
    if ax is None:
        return None
    neck, u, L, _ = ax
    n = np.array([-u[1], u[0]])
    okk = np.isfinite(kp[:, :2]).all(1)
    proj = (kp[okk, :2] - neck) @ n
    c = float((proj.min() + proj.max()) / 2)
    half = float((proj.max() - proj.min()) / 2 + 0.12 * L)
    fall = 0.6 * L
    if labels is not None and alpha is not None and gx is not None:
        other = (alpha > 110) & (labels != pidx) & (labels >= 0)
        if other.any():
            ox, oy = gx[other], gy[other]
            sa = (ox - neck[0]) * u[0] + (oy - neck[1]) * u[1]
            sel = (sa > -0.6 * L) & (sa < 1.1 * L)
            if sel.any():
                dn = np.abs((ox[sel] - neck[0]) * n[0] + (oy[sel] - neck[1]) * n[1] - c) - half
                gap = float(np.percentile(dn, 2))
                fall = float(np.clip(0.9 * gap, 0.15 * L, 0.6 * L))
    return c, half, fall


def compute_field(fi: FieldInput, out_w: int, out_h: int, grid_step: int = 4) -> FieldResult:
    gw = max(8, int(np.ceil(out_w / grid_step)))
    gh = max(8, int(np.ceil(out_h / grid_step)))
    s = out_w / fi.W                              # source px -> output px
    sy_ = out_h / fi.H
    xs = (np.arange(gw) + 0.5) * out_w / gw - 0.5
    ys = (np.arange(gh) + 0.5) * out_h / gh - 0.5
    Xo, Yo = np.meshgrid(xs, ys)
    X, Y = Xo / s, Yo / sy_                       # grid in source px
    total = np.zeros((gh, gw, 2), np.float64)
    raw_max = 0.0
    if not fi.targets:
        return FieldResult(total.astype(np.float32), out_w, out_h, 0.0, 0.0)
    # the frame edge (and the people not retouched, per target below)
    base_dist = np.full((gh, gw), np.inf)
    border_w = None
    edge = None
    if fi.border:
        # at the frame edge only what would be fetched from outside the picture is held back; a head growing
        # out of the top of the frame fetches from inside it and may go on (holding everything there squeezed
        # the field against the edge and folded it: the crown looked cut off)
        bw_ = 0.006 * out_w
        edge = (_smoothstep(0, bw_, Xo + 0.5), _smoothstep(0, bw_, out_w - 0.5 - Xo),
                _smoothstep(0, bw_, Yo + 0.5), _smoothstep(0, bw_, out_h - 0.5 - Yo))
    act_w = None
    if fi.active is not None:
        # letterbox / pillarbox bars never move, and the picture edge next to them holds like a frame edge
        ax0, ay0, ax1, ay1 = fi.active
        ax0, ax1 = ax0 * s, ax1 * s
        ay0, ay1 = ay0 * sy_, ay1 * sy_
        act_w = _smoothstep(0, 0.006 * out_w, np.minimum.reduce([Xo - ax0, ax1 - Xo, Yo - ay0, ay1 - Yo]))
        border_w = act_w
    lab_g = alpha_g = None
    if fi.protect_others and fi.alpha is not None and fi.labels is not None:
        alpha_g = cv2.resize(fi.alpha, (gw, gh), interpolation=cv2.INTER_AREA)
        lab_g = cv2.resize(fi.labels.astype(np.int16), (gw, gh), interpolation=cv2.INTER_NEAREST)
    falloff = fi.falloff_pct / 100.0 * out_h
    for pi in fi.targets:
        # only evaluate the grid window the person's field can reach
        ok = np.isfinite(pi.kp[:, :2]).all(1)
        dx = np.zeros_like(X)
        dy = np.zeros_like(Y)
        pts = pi.kp[ok, :2]
        if pi.face_geom is not None and (pi.face_w > 0.02 or pi.head_w > 0.02):
            A = {k: pi.amounts.get(k, 0.0) / 100.0 for k in FACE_KEYS + HEAD_KEYS}
            A["_feather"] = pi.feather
            pts = np.vstack([pts, FW.face_extent(pi.face_geom, pi.head_geom, A)])
        if len(pts) >= 2:
            wmax = np.nanmax(np.r_[pi.widths.ravel() if pi.widths is not None else [], [1.0]])
            sc_ = S.person_scale(pi.kp)
            wmax = np.nanmax([wmax if np.isfinite(wmax) else 0, 0.12 * (sc_ if np.isfinite(sc_) else 300)])
            mult = max(pi.width_mult.values()) if pi.width_mult else 1.0
            margin = wmax * mult * fi.reach * 1.2 + 0.15 * (sc_ if np.isfinite(sc_) else 300)
            x0s, y0s = pts.min(0) - margin
            x1s, y1s = pts.max(0) + margin
            gx0 = int(np.clip(np.searchsorted(xs / s, x0s) - 1, 0, gw))
            gx1 = int(np.clip(np.searchsorted(xs / s, x1s) + 1, 0, gw))
            gy0 = int(np.clip(np.searchsorted(ys / sy_, y0s) - 1, 0, gh))
            gy1 = int(np.clip(np.searchsorted(ys / sy_, y1s) + 1, 0, gh))
            if gx1 > gx0 and gy1 > gy0:
                if pi.cloth_keep is not None:
                    # kept clothes: only the body edits are held back there, face and head edits are not
                    sdx, sdy = person_field(X[gy0:gy1, gx0:gx1], Y[gy0:gy1, gx0:gx1], pi, fi.reach, "body")
                    fdx, fdy = person_field(X[gy0:gy1, gx0:gx1], Y[gy0:gy1, gx0:gx1], pi, fi.reach, "face")
                    bmag = float(np.hypot(sdx, sdy).max()) * s if sdx.size else 0.0
                    cg = cv2.resize(pi.cloth_keep.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_AREA)[gy0:gy1, gx0:gx1]
                    if cg.any() and bmag > 1e-4:
                        dtc = cv2.distanceTransform((1 - (cg > 0)).astype(np.uint8), cv2.DIST_L2, 3) * (out_w / gw)
                        kw = _smoothstep(0, max(falloff * 0.6, 2.5 * bmag), dtc)
                        sdx = sdx * kw
                        sdy = sdy * kw
                    sdx = sdx + fdx
                    sdy = sdy + fdy
                else:
                    sdx, sdy = person_field(X[gy0:gy1, gx0:gx1], Y[gy0:gy1, gx0:gx1], pi, fi.reach)
                dx[gy0:gy1, gx0:gx1] = sdx
                dy[gy0:gy1, gx0:gx1] = sdy
        dx *= s
        dy *= sy_
        mag = np.hypot(dx, dy)
        mmax = float(mag.max()) if mag.size else 0.0
        raw_max = max(raw_max, mmax)
        if mmax < 1e-4:
            continue
        sigma = max(falloff, 3.0 * mmax)
        dist = base_dist.copy()
        if alpha_g is not None:
            other = ((alpha_g > 110) & (lab_g != pi.p)).astype(np.uint8)
            if other.any():
                dt = cv2.distanceTransform((1 - other).astype(np.uint8), cv2.DIST_L2, 3) * (out_w / gw)
                dist = np.minimum(dist, dt)
        prot = _smoothstep(0, sigma, dist) if np.isfinite(dist).any() else 1.0
        wgt = prot if border_w is None else prot * border_w
        dx, dy = dx * wgt, dy * wgt
        if edge is not None:
            dx = np.where(dx < 0, dx * edge[0], dx * edge[1])
            dy = np.where(dy < 0, dy * edge[2], dy * edge[3])
        total[..., 0] += dx
        total[..., 1] += dy
    local_max = float(np.hypot(total[..., 0], total[..., 1]).max())
    # ---- length changes (height, leg length): an even stretch along the body, composed on top
    shape = np.zeros_like(total)
    for pi in fi.targets:
        hgt = pi.amounts.get("height", 0.0) / 100.0 * pi.strength
        leg = pi.amounts.get("leg_length", 0.0) / 100.0 * pi.strength
        if abs(hgt) < 1e-4 and abs(leg) < 1e-4:
            continue
        # kept to the person: the stretch fades out sideways over the free space next to them
        lateral = lateral_extent(pi.kp, lab_g, alpha_g, pi.p, X, Y)
        sdx, sdy = shape_disp(X, Y, pi.kp, hgt, leg, lateral)
        sdx, sdy = sdx * s, sdy * s
        # other people next to this one stay as they are (in a tight group the lift would carry them along)
        smax = float(np.hypot(sdx, sdy).max())
        if alpha_g is not None and smax > 1e-3:
            other = ((alpha_g > 110) & (lab_g != pi.p) & (lab_g >= 0)).astype(np.uint8)
            if other.any():
                dt = cv2.distanceTransform((1 - other).astype(np.uint8), cv2.DIST_L2, 3) * (out_w / gw)
                k = _smoothstep(0, max(falloff, 3.0 * smax), dt)
                sdx, sdy = sdx * k, sdy * k
        if act_w is not None:
            sdx, sdy = sdx * act_w, sdy * act_w
        shape[..., 0] += sdx
        shape[..., 1] += sdy
    if np.abs(shape).max() > 1e-3:
        # output p -> after the stretch q = p + shape(p) -> source q + local(q)
        hx, hy = out_w / gw, out_h / gh
        gxi = (Xo + shape[..., 0] + 0.5) / hx - 0.5
        gyi = (Yo + shape[..., 1] + 0.5) / hy - 0.5
        loc = cv2.remap(total.astype(np.float32), gxi.astype(np.float32), gyi.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        total = shape + loc
        raw_max = max(raw_max, float(np.hypot(shape[..., 0], shape[..., 1]).max()))
    repaired = repair_folds(total, out_w / gw, out_h / gh)
    mx = float(np.hypot(total[..., 0], total[..., 1]).max())
    return FieldResult(total.astype(np.float32), out_w, out_h, mx, raw_max, repaired, min(local_max, mx))


def _det(D: np.ndarray, hx: float, hy: float) -> np.ndarray:
    a = np.gradient(D[..., 0], hx, axis=1)
    b = np.gradient(D[..., 0], hy, axis=0)
    c = np.gradient(D[..., 1], hx, axis=1)
    d = np.gradient(D[..., 1], hy, axis=0)
    return (1 + a) * (1 + d) - b * c, a + d, a * d - b * c


def repair_folds(D: np.ndarray, hx: float, hy: float, min_det: float = 0.3, iters: int = 6) -> float:
    """Soften the field in place wherever overlapping parts would fold the image. Returns repaired share."""
    active = np.hypot(D[..., 0], D[..., 1]) > 0.05
    if not active.any():
        return 0.0
    touched = np.zeros(D.shape[:2], bool)
    for _ in range(iters):
        det, tr, q = _det(D, hx, hy)
        bad = det < min_det
        if not bad.any():
            break
        # largest k in [0,1] with 1 + k*tr + k^2*q >= min_det (bisection, vectorised)
        lo = np.zeros(bad.sum())
        hi = np.ones(bad.sum())
        trb, qb = tr[bad], q[bad]
        for _b in range(12):
            mid = (lo + hi) / 2
            ok = 1 + mid * trb + mid * mid * qb >= min_det
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        k = np.ones(D.shape[:2], np.float32)
        k[bad] = 0.85 * lo
        # the attenuation itself must vary slowly compared to the displacement, or it creates new folds
        dmax = float(np.hypot(D[..., 0], D[..., 1]).max())
        sig = max(2.5, 2.0 * dmax / min(hx, hy))
        r = int(min(41, max(5, sig)) // 2 * 2 + 1)
        k = cv2.erode(k, np.ones((r, r), np.uint8))
        k = cv2.GaussianBlur(k, (0, 0), sig)
        D *= k[..., None]
        touched |= k < 0.98
    return float((touched & active).sum()) / max(1, int(active.sum()))


def sample_field(fr: FieldResult, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bicubic field values at output-pixel coordinates xs (W',) x ys (H',) -> (H', W', 2).

    Same interpolation as resizing the grid to full resolution, but only for the pixels needed."""
    gh, gw, _ = fr.D.shape
    gx = ((xs.astype(np.float32) + 0.5) * gw / fr.out_w - 0.5)
    gy = ((ys.astype(np.float32) + 0.5) * gh / fr.out_h - 0.5)
    mx, my = np.meshgrid(gx, gy)
    return cv2.remap(fr.D, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def sample_points(fr: FieldResult, pts: np.ndarray) -> np.ndarray:
    """Field values at arbitrary output-pixel points (N,2) -> (N,2)."""
    gh, gw, _ = fr.D.shape
    gx = ((pts[:, 0] + 0.5) * gw / fr.out_w - 0.5).astype(np.float32)[None]
    gy = ((pts[:, 1] + 0.5) * gh / fr.out_h - 0.5).astype(np.float32)[None]
    return cv2.remap(fr.D, gx, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)[0]


def upsample(fr: FieldResult, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Full-resolution field (h, w, 2) float32 in output px (or only the ROI x0,y0,x1,y1)."""
    if roi is None:
        return cv2.resize(fr.D, (fr.out_w, fr.out_h), interpolation=cv2.INTER_CUBIC)
    x0, y0, x1, y1 = roi
    return sample_field(fr, np.arange(x0, x1), np.arange(y0, y1))


def active_roi(fr: FieldResult, thresh: float = 0.02, pad: int = 6) -> Optional[Tuple[int, int, int, int]]:
    mag = np.hypot(fr.D[..., 0], fr.D[..., 1])
    ys, xs = np.nonzero(mag > thresh)
    if not len(xs):
        return None
    gh, gw = mag.shape
    sx, sy = fr.out_w / gw, fr.out_h / gh
    x0 = max(0, int((xs.min() - 2) * sx) - pad)
    y0 = max(0, int((ys.min() - 2) * sy) - pad)
    x1 = min(fr.out_w, int((xs.max() + 3) * sx) + pad)
    y1 = min(fr.out_h, int((ys.max() + 3) * sy) + pad)
    return x0, y0, x1, y1


def jacobian_det(fr: FieldResult) -> np.ndarray:
    """det(I + dD/dp) on the grid (fold check)."""
    gh, gw, _ = fr.D.shape
    hx = fr.out_w / gw
    hy = fr.out_h / gh
    dDx_dx = np.gradient(fr.D[..., 0], hx, axis=1)
    dDx_dy = np.gradient(fr.D[..., 0], hy, axis=0)
    dDy_dx = np.gradient(fr.D[..., 1], hx, axis=1)
    dDy_dy = np.gradient(fr.D[..., 1], hy, axis=0)
    return (1 + dDx_dx) * (1 + dDy_dy) - dDx_dy * dDy_dx


# --------------------------------------------------------------------------- building inputs from a cut

def _body_edits(kp, params, bs_on: bool, bs_amt: float, au_amt: float, be_amt: float):
    """Point edits (by the person's sides) plus the one-sided body sliders, by measured sides of this frame."""
    out = None
    if bs_on:
        flips = B.side_flips(kp) if getattr(params, "body_sides", "person") == "person" else np.zeros(B.NC, bool)
        out = B.sculpt_to_measured(B.sculpt_array(params.body_sculpt), flips) * (bs_amt / 100.0)
    ps = B.part_sculpt(kp, au_amt, be_amt)
    if ps is not None:
        out = ps if out is None else out + ps
    return out


def _cloth_keep(tracker, t: int, p: int, params) -> Optional[np.ndarray]:
    mode = getattr(params, "cloth", "include")
    if mode == "include" or not hasattr(tracker, "cloth"):
        return None
    from ..track import clothing as CL
    if not CL.ready_for(tracker, tracker.store.pids[p]):
        return None
    return CL.keep_mask(tracker, t, p, mode)


def build_input(cut: Cut, tracker, t: int, which: str = "final") -> FieldInput:
    st = tracker.store
    fi = FieldInput(W=cut.info.width, H=cut.info.height, targets=[])
    if hasattr(tracker, "active_area"):
        fi.active = tracker.active_area()
    fi.protect_others, fi.falloff_pct, fi.border, fi.reach = (WARP_SETTINGS["others"], WARP_SETTINGS["falloff_pct"],
                                                              WARP_SETTINGS["border"], WARP_SETTINGS["reach"])
    if st is None:
        return fi
    for person in cut.targets():
        if not st.has(person.id):
            continue
        p = st.index(person.id)
        if not st.present[p, t]:
            continue
        params = cut.params_for(person.id)
        amounts = {k: params.amount(k, t) for k in GEOMETRY_KEYS}
        sculpt_on = bool(params.sculpt) and abs(amounts.get("sculpt", 0.0)) > 1e-6
        bs_on = bool(params.body_sculpt) and abs(amounts.get("body_sculpt", 0.0)) > 1e-6
        bs_amt = amounts.pop("body_sculpt", 0.0)
        au_amt = amounts.pop("arm_under", 0.0) / 100.0
        be_amt = amounts.pop("belly", 0.0) / 100.0
        part_on = abs(au_amt) > 1e-6 or abs(be_amt) > 1e-6
        if not any(abs(v) > 1e-6 for k, v in amounts.items() if k != "sculpt") and not sculpt_on and not bs_on \
                and not part_on:
            continue
        if not sculpt_on:
            amounts["sculpt"] = 0.0
        if which == "final" and hasattr(tracker, "warp_geometry"):
            geo, segw, shw = tracker.warp_geometry(p)
            kp, sw, shv = geo[t], segw[t], float(shw[t])
        else:
            kp, sw, shv = getattr(st, which)[p, t], None, 1.0
        if not np.isfinite(kp[:, :2]).any():
            continue
        face = None
        fw = 0.0
        hw = 0.0
        fg = hg = None
        head_on = any(abs(amounts.get(k, 0.0)) > 1e-6 for k in HEAD_KEYS)
        if st.has_face and np.isfinite(st.face[p, t, :, :2]).any():
            face = st.face[p, t]
            fw = float(st.face_vis[p, t])
            if any(abs(amounts.get(k, 0.0)) > 1e-6 for k in FACE_KEYS + HEAD_KEYS):
                fg = tracker.face_geom(p, t) if hasattr(tracker, "face_geom") else FW.FaceGeom.from_landmarks(face)
        if head_on and st.has_face and hasattr(tracker, "head_face") and fw < 0.5:
            # turned away (or turning): head edits go on with the face carried over from where it was seen
            fgh = tracker.head_face(p, t)
            if fgh is not None:
                fg = fgh
        parts = None
        if fg is not None and head_on and hasattr(tracker, "head_geom"):
            hg = tracker.head_geom(p, t, fg)
            if hg is not None and hasattr(tracker, "parts"):
                from ..track import parts as PT
                parts = PT.part_maps(tracker, p, t, fg, kp)
        if hg is not None:
            hw = 1.0
        elif amounts.get("neck", 0.0):
            pass
        if fg is None and abs(amounts.get("neck", 0.0)) > 1e-6 and face is not None:
            fg = FW.FaceGeom.from_landmarks(face)
        wd = st.widths[p, t]
        waist = wd[B.SEG_INDEX["waist"]] if wd is not None else np.array([np.nan, np.nan])
        body_half = float(np.nanmax(waist) * 1.15) if np.isfinite(waist).any() else float("nan")
        hands = tracker.hand_polys(p, t) if hasattr(tracker, "hand_polys") and abs(amounts.get("forearm", 0.0)) > 1e-6 else []
        if (fg is not None or hg is not None) and not hands and hasattr(tracker, "hand_polys"):
            hands = tracker.hand_polys(p, t)
        fi.targets.append(PersonInput(p=p, kp=kp, widths=wd, amounts=amounts,
                                      width_mult=dict(params.widths), strength=params.strength.value(t) / 100.0,
                                      seg_w=sw, sh_w=shv, face=face, face_w=fw, face_geom=fg, head_geom=hg, head_w=hw,
                                      sculpt=params.sculpt if sculpt_on else {}, hands=hands, body_half=body_half,
                                      feather=params.feather_factor(),
                                      contour=st.contour[p, t] if getattr(st, "has_contour", False) else None,
                                      body_sculpt=_body_edits(kp, params, bs_on, bs_amt, au_amt, be_amt),
                                      cloth_keep=_cloth_keep(tracker, t, p, params), parts=parts))
    if fi.targets and fi.protect_others:
        a = tracker.matte.get(t)
        if a is not None:
            fi.alpha = np.asarray(a)
            fi.labels = tracker.labels(t)
    # a cut's own protection settings, lines and shapes (from before 1.22.2) are kept in the project but not used
    return fi


# --------------------------------------------------------------------------- applying the field

def forward_points(fr: Optional[FieldResult], pts: np.ndarray, iters: int = 8) -> np.ndarray:
    """Where source points (output px, (N,2)) end up after the warp (inverts the backward map)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if fr is None or fr.empty() or len(pts) == 0:
        return pts.copy()
    ok = np.isfinite(pts).all(1)
    out = pts.copy()
    q = pts[ok]
    p = q.copy()
    for _ in range(iters):
        p = q - sample_points(fr, p)
    out[ok] = p
    return out


def warp_image(img: np.ndarray, fr: FieldResult, interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Warp an image whose size equals the field's output size (RGB uint8 preview or a plane)."""
    if fr.empty():
        return img
    roi = active_roi(fr)
    if roi is None:
        return img
    x0, y0, x1, y1 = roi
    D = upsample(fr, roi)
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    mx = gx + D[..., 0]
    my = gy + D[..., 1]
    out = img.copy()
    out[y0:y1, x0:x1] = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
    return out


def warp_planes(planes: List[np.ndarray], subsample: List[Tuple[int, int]], fr: FieldResult,
                max_value: float, full_field: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Warp native planes. Chroma planes sample the field at their own (co-sited) positions."""
    if fr.empty():
        return planes
    roi = active_roi(fr)
    if roi is None:
        return planes
    x0, y0, x1, y1 = roi
    out = []
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    for pl, (ssx, ssy) in zip(planes, subsample):
        h, w = pl.shape[:2]
        cx0, cy0 = x0 // ssx, y0 // ssy
        cx1, cy1 = min(w, -(-x1 // ssx)), min(h, -(-y1 // ssy))
        key = (ssx, ssy)
        if key not in cache:
            # field sampled at this plane's sample sites (co-sited with luma)
            cache[key] = sample_field(fr, np.arange(cx0, cx1) * ssx, np.arange(cy0, cy1) * ssy)
        Dp = cache[key]
        gx, gy = np.meshgrid(np.arange(cx0, cx1, dtype=np.float32), np.arange(cy0, cy1, dtype=np.float32))
        mx = gx + Dp[..., 0] / ssx
        my = gy + Dp[..., 1] / ssy
        if pl.dtype == np.uint16 or pl.dtype == np.float32 or pl.dtype == np.uint8:
            res = cv2.remap(pl, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            res = cv2.remap(pl.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if np.issubdtype(pl.dtype, np.integer):
            res = np.minimum(res, np.array(max_value).astype(res.dtype))
        o = np.array(pl, copy=True)
        o[cy0:cy1, cx0:cx1] = res.astype(pl.dtype)
        out.append(o)
    return out
