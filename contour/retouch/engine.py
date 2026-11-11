"""Retouch engine: skin, blemishes, wrinkles, dark circles, shine, tone, grain, lips, teeth, eye whites, knees.

Everything happens before the warp, in Y/Cb/Cr, with masks built from the tracked face landmarks and
skeleton on the proxy frame. The same code serves the preview (8-bit RGB proxy) and the render (native
planes: 4:2:2 / 4:4:4 YUV, or float RGB), so what you see is what renders.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import skin as SK
from ..core.params import RETOUCH_KEYS
from ..track import skeleton as S


@dataclass
class Job:
    p: int
    pid: int
    v: Dict[str, float]                    # values as fractions (0.4 = 40 %)
    keep: float
    scope: str
    grain_size: float
    face: Optional[np.ndarray]             # (68,3) source px
    face_w: float
    scale: float
    skin: Optional[SK.SkinJob] = None
    masks: Dict[str, np.ndarray] = field(default_factory=dict)   # proxy resolution, float32 0..1
    wf_src: float = 0.0                    # face width, source px
    blemish_size: float = 0.5              # 0..1 largest spot taken by the blemish removal


@dataclass
class RetouchFrame:
    t: int
    jobs: List[Job]
    proxy_wh: Tuple[int, int]
    src_wh: Tuple[int, int]

    def empty(self) -> bool:
        return not self.jobs

    def union(self, names=None) -> Optional[np.ndarray]:
        out = None
        for j in self.jobs:
            for k, m in j.masks.items():
                if k == "excl" or (names is not None and k not in names):
                    continue
                out = m.copy() if out is None else np.maximum(out, m)
        return out


# --------------------------------------------------------------------------- masks

def _poly(m: np.ndarray, pts: np.ndarray) -> None:
    cv2.fillPoly(m, [np.round(pts * 4).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=2)


def _curve(m: np.ndarray, pts: np.ndarray, thick: float) -> None:
    from ..warp.face import catmull
    c = catmull(np.asarray(pts, np.float64), 6)
    cv2.polylines(m, [np.round(c * 4).astype(np.int32)], False, 255, max(1, int(round(thick))), cv2.LINE_AA, 2)


def _sat(rgb_f: np.ndarray) -> np.ndarray:
    mx = rgb_f.max(-1)
    mn = rgb_f.min(-1)
    return (mx - mn) / np.maximum(mx, 1e-3)


def _luma(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f[..., 0] * 0.2126 + rgb_f[..., 1] * 0.7152 + rgb_f[..., 2] * 0.0722


def _feather(m: np.ndarray, sig: float) -> np.ndarray:
    f = m.astype(np.float32) / 255.0 if m.dtype == np.uint8 else m.astype(np.float32)
    if sig > 0.3:
        f = cv2.GaussianBlur(f, (0, 0), sig)
    return np.clip(f, 0, 1)


def wrinkle_masks(lm: np.ndarray, shape, prob: np.ndarray, which: str, fth: float = 1.5,
                  wide: float = 1.0) -> np.ndarray:
    """Nasolabial folds ('naso') or the lines down from the mouth corners ('mouth'). lm in mask px.
    wide > 1: a wider band where the fold is looked for (where it really is differs from face to face)."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    mc = L[48:60].mean(0)
    down = L[8] - L[27]
    down = down / max(np.hypot(*down), 1e-6)
    for ala, corner, sign in ((L[31], L[48], -1), (L[35], L[54], 1)):
        out = ala - L[33]
        out = out / max(np.hypot(*out), 1e-6)
        if which == "naso":
            a0 = ala + out * 0.06 * Wf - down * 0.03 * Wf
            c0 = corner + (corner - mc) * 0.22
            d = float(np.hypot(*(c0 - a0)))
            mid = (a0 + c0) / 2 + out * 0.10 * d
            end = c0 + down * 0.20 * d + out * 0.04 * d
            _curve(m, np.array([a0, mid, c0, end]), 0.085 * Wf * wide)
        else:
            c0 = corner + (corner - mc) * 0.12
            ch = L[8] + (corner - L[8]) * 0.35
            end = c0 + (ch - c0) * 0.75 + out * 0.03 * Wf
            mid = (c0 + end) / 2 + out * 0.02 * Wf
            _curve(m, np.array([c0, mid, end]), 0.06 * Wf * wide)
    f = _feather(m, 0.028 * Wf * fth)
    # keep it on skin (never on the lips, in the nostrils or in the groove of the nose wings)
    lip = np.zeros((h, w), np.uint8)
    _poly(lip, L[48:60])
    # a soft ramp off the lips: a fill that stops hard at the lip outline leaves the shadow in the mouth corner
    # standing as a dark notch next to the lifted skin
    d = max(1, int(0.012 * Wf))
    lipf = np.clip(_feather(cv2.dilate(lip, np.ones((2 * d + 1, 2 * d + 1), np.uint8)), 0.03 * Wf) * 1.6, 0, 1)
    ala = _feather(_ala_mask(lm, (h, w)), 0.02 * Wf)
    # the mouth corners: the crease of the corner itself belongs to the mouth's shape. Lifting the skin around it
    # while the lips are left alone leaves the corner's shadow as a dark notch
    cz = np.zeros((h, w), np.uint8)
    for c in (L[48], L[54]):
        cv2.circle(cz, (int(round(c[0] * 4)), int(round(c[1] * 4))), max(4, int(0.045 * Wf * 4)), 255, -1,
                   cv2.LINE_AA, 2)
    cz = np.clip(_feather(cz, 0.035 * Wf) * 1.5, 0, 1)
    soft = np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.02 * Wf)) * 1.5, 0, 1)
    return np.clip(f * 1.3, 0, 1) * soft * (1 - lipf) * (1 - cz) * (1 - np.clip(ala * 1.2, 0, 1))


def _ala_mask(lm: np.ndarray, shape) -> np.ndarray:
    """The wings of the nose with the groove where they meet the cheek (255). Not a crease to fill: it is
    the shape of the nose, and lifting it leaves a pale smudge beside the nostril."""
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    m = np.zeros(shape, np.uint8)
    up = L[27] - L[33]
    up = up / max(np.hypot(*up), 1e-6)
    for ala, sign in ((L[31], -1.0), (L[35], 1.0)):
        side = L[31] - L[35] if sign < 0 else L[35] - L[31]
        side = side / max(np.hypot(*side), 1e-6)
        c = ala + side * 0.035 * Wf + up * 0.035 * Wf
        ang = float(np.degrees(np.arctan2(up[1], up[0])))
        cv2.ellipse(m, (int(round(c[0] * 4)), int(round(c[1] * 4))),
                    (max(4, int(0.085 * Wf * 4)), max(4, int(0.06 * Wf * 4))), ang, 0, 360, 255, -1,
                    cv2.LINE_AA, 2)
    return m


def feature_exclusion(lm: np.ndarray, shape) -> np.ndarray:
    """Eyes, brows, lips, mouth, nostrils and nose wings (soft), for things that must stay out of skin
    averages and fills."""
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    m = SK._features_mask(lm, shape)
    inner = np.zeros(shape, np.uint8)
    _poly(inner, L[60:68])
    m = np.maximum(m, inner)
    m = np.maximum(m, _ala_mask(lm, shape))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return _feather(m, 0.012 * Wf)


def tone_mask(lm: Optional[np.ndarray], shape, prob: np.ndarray, alpha: Optional[np.ndarray], scope: str,
              face_w: float, fth: float = 1.5, body: Optional[np.ndarray] = None) -> np.ndarray:
    """Where skin brightness, tone, saturation and grain apply: the whole face (around the eyes too),
    softly; only the eyeballs and lips are left out. body: the chosen body regions (body_regions)."""
    from ..core.project import scope_set
    if "face" not in scope_set(scope):
        lm = None
    h, w = shape
    region = np.zeros(shape, np.float32)
    psoft = np.clip(cv2.GaussianBlur(prob, (0, 0), 3.0) * 1.6, 0, 1)
    Wf = 0.0
    if lm is not None:
        L = lm[:, :2]
        Wf = float(np.hypot(*(L[16] - L[0])))
        poly = SK.face_polygon(lm)
        if poly is not None:
            fm = np.zeros(shape, np.uint8)
            _poly(fm, poly)
            region = _feather(fm, 0.04 * Wf * fth) * float(np.clip(face_w * 1.5, 0, 1))
    if body is not None:
        region = np.maximum(region, (psoft * body * (alpha if alpha is not None else 1.0)).astype(np.float32))
    if alpha is not None:
        region = region * np.clip(alpha * 1.3, 0, 1)
    if lm is not None and Wf > 0:
        L = lm[:, :2]
        ex = np.zeros(shape, np.uint8)
        for idx in (slice(36, 42), slice(42, 48)):
            e = L[idx]
            c = e.mean(0)
            _poly(ex, c + (e - c) * 1.12)
        c = L[48:60].mean(0)
        _poly(ex, c + (L[48:60] - c) * 1.05)
        region = region * (1 - _feather(ex, 0.018 * Wf * fth))
        # hair falling over the face gets only a little of it
        region = region * (0.35 + 0.65 * psoft)
    return np.clip(region, 0, 1)


def under_eye_mask(lm: np.ndarray, shape, prob: np.ndarray, fth: float = 1.5) -> np.ndarray:
    """Crescents under the eyes, from just below the lashes down towards the cheek."""
    h, w = shape
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    ex = L[45] - L[36]
    ex = ex / max(np.hypot(*ex), 1e-6)
    down = np.array([-ex[1], ex[0]])
    if (L[8] - L[27]) @ down < 0:
        down = -down
    m = np.zeros((h, w), np.uint8)
    for lower, outer_first in ((L[[36, 41, 40, 39]], True), (L[[42, 47, 46, 45]], False)):
        top = lower + down * 0.018 * Wf
        prof = np.array([0.045, 0.11, 0.115, 0.06]) * Wf
        bot = lower + down[None] * prof[:, None]
        # the outer corner reaches a little further out, the inner one stays clear of the nose side
        outer = 0 if outer_first else 3
        sgn = -1.0 if outer_first else 1.0
        bot[outer] = bot[outer] + ex * sgn * 0.03 * Wf
        poly = np.vstack([top, bot[::-1]])
        from ..warp.face import catmull
        _poly(m, catmull(np.vstack([poly, poly[:1]]), 4))
    f = _feather(m, 0.022 * Wf * fth)
    # never on the eye itself
    eye = np.zeros((h, w), np.uint8)
    for idx in (slice(36, 42), slice(42, 48)):
        e = L[idx]
        c = e.mean(0)
        _poly(eye, c + (e - c) * 1.15)
    f = f * (1 - _feather(cv2.dilate(eye, np.ones((3, 3), np.uint8)), 0.008 * Wf))
    soft = np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.02 * Wf)) * 1.5, 0, 1)
    return np.clip(f * 1.2, 0, 1) * soft


def hot_zone_mask(lm: np.ndarray, shape, prob: np.ndarray, fth: float = 1.5) -> np.ndarray:
    """Where a key light makes hotspots a retoucher presses down: forehead centre, the cheek tops under the
    eyes, the nose bridge, the chin. Soft zones (a brush, not a cut-out)."""
    h, w = shape
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    up = L[27] - L[8]
    up = up / max(1e-6, np.linalg.norm(up))
    m = np.zeros((h, w), np.float32)

    def ell(c, ax, ay, val=1.0):
        ang = float(np.degrees(np.arctan2(up[0], -up[1])))
        cv2.ellipse(m, (int(c[0]), int(c[1])), (max(1, int(ax)), max(1, int(ay))), ang, 0, 360, val, -1)

    brow_c = (L[21] + L[22]) / 2
    span = float(abs((L[8] - L[17:27].mean(0)) @ up)) / max(Wf, 1e-6)
    fh = float(np.clip((span - 0.45) / 0.35, 0.0, 1.0))           # how much forehead a tilted head still shows
    if fh > 0.2:
        ell(brow_c + up * 0.17 * Wf * fh, 0.24 * Wf, 0.15 * Wf * fh)         # forehead centre
    for e, o in ((L[36:42], L[31]), (L[42:48], L[35])):
        lo = e[[4, 5]].mean(0)
        c = lo - up * 0.14 * Wf
        ell(c, 0.13 * Wf, 0.11 * Wf)                                         # cheek top under the eye
    ell((L[27] + L[30]) / 2, 0.05 * Wf, 0.14 * Wf)                           # nose bridge
    ell(L[8] + up * 0.06 * Wf, 0.09 * Wf, 0.05 * Wf)                         # chin
    f = cv2.GaussianBlur(m, (0, 0), 0.05 * Wf * fth / 1.5)
    soft = np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.02 * Wf)) * 1.5, 0, 1)
    return np.clip(f * 1.3, 0, 1) * soft


def _steady_body(st, p: int, t: int, r: int = 4):
    """Shoulders and segment widths for the neck, median over a few frames of the same shot: taken frame by
    frame, a shoulder joint that wobbles by a few pixels moved the neck's lower edge with it."""
    kp = st.final[p, t].copy()
    wd = st.widths[p, t].copy() if st.widths is not None else None
    N = st.final.shape[1]
    seg = [s for s in range(max(0, t - r), min(N, t + r + 1)) if st.present[p, s] and st.connected(s, t)]
    if len(seg) >= 3:
        for j in (S.L_SH, S.R_SH):
            pts = st.final[p, seg, j, :2]
            ok = np.isfinite(pts).all(1)
            if ok.sum() >= 3 and np.isfinite(kp[j, :2]).all():
                kp[j, :2] = np.median(pts[ok], 0)
        if wd is not None:
            W = st.widths[p, seg]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(W, 0)
            wd = np.where(np.isfinite(m), m, wd)
    return kp, wd


def _capsule(m: np.ndarray, a, b, r: float):
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or r < 1:
        return
    cv2.line(m, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), 255,
             max(1, int(round(2 * r))), cv2.LINE_AA)
    for q in (a, b):
        cv2.circle(m, (int(round(q[0])), int(round(q[1]))), int(round(r)), 255, -1, cv2.LINE_AA)


def body_regions(regions: set, kp: np.ndarray, hand: Optional[np.ndarray], shape, sx: float, sy: float,
                 neck: Optional[np.ndarray] = None, fth: float = 1.5) -> Optional[np.ndarray]:
    """Where on the body the skin retouch may go (0..1), from the joints: neck, shoulders and collarbone,
    arms, hands, legs. Sizes follow the shoulder width, so a zoom changes nothing. The skin colour model and
    the person matte decide inside these regions what really is skin."""
    want = regions - {"face"}
    if not want or kp is None:
        return None
    h, w = shape
    P = np.full((kp.shape[0], 2), np.nan)
    ok = np.isfinite(kp[:, :2]).all(1)
    if kp.shape[1] > 2:
        ok &= np.nan_to_num(kp[:, 2]) > 0.2
    P[ok] = kp[ok, :2] * [sx, sy]
    ls, rs = P[S.L_SH], P[S.R_SH]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return neck if "neck" in want else None
    sw = float(np.hypot(*(ls - rs)))
    if sw < 8:
        return None
    m = np.zeros(shape, np.uint8)
    mid = (ls + rs) / 2
    hips = [P[S.L_HIP], P[S.R_HIP]]
    hp = [q for q in hips if np.isfinite(q).all()]
    down = (np.mean(hp, 0) - mid) if hp else np.array([0.0, sw])
    down = down / max(1e-6, float(np.hypot(*down)))
    side = (ls - rs) / sw
    if "shoulder" in want:
        # shoulders and collarbone: from just above the shoulder line down about a third of the way to the hips
        top = 0.12 * sw
        bot = 0.45 * sw
        pts = [ls + side * 0.18 * sw - down * top, rs - side * 0.18 * sw - down * top,
               rs - side * 0.10 * sw + down * bot, ls + side * 0.10 * sw + down * bot]
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255, cv2.LINE_AA)
    if "arm" in want:
        for a_, b_, c_ in ((S.L_SH, S.L_EL, S.L_WR), (S.R_SH, S.R_EL, S.R_WR)):
            _capsule(m, P[a_], P[b_], 0.14 * sw)
            _capsule(m, P[b_], P[c_], 0.11 * sw)
    if "hand" in want:
        for el, wr in ((S.L_EL, S.L_WR), (S.R_EL, S.R_WR)):
            if np.isfinite(P[wr]).all():
                d = (P[wr] - P[el]) if np.isfinite(P[el]).all() else np.array([0.0, 1.0])
                d = d / max(1e-6, float(np.hypot(*d)))
                _capsule(m, P[wr], P[wr] + d * 0.30 * sw, 0.16 * sw)
        if hand is not None:
            H = hand[:, :2] * [sx, sy]
            okh = np.isfinite(H).all(1)
            for i0 in range(0, len(H) - 20, 21):
                pts = H[i0:i0 + 21][okh[i0:i0 + 21]]
                if len(pts) >= 5:
                    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 255, cv2.LINE_AA)
    if "leg" in want:
        for a_, b_, c_ in ((S.L_HIP, S.L_KN, S.L_AN), (S.R_HIP, S.R_KN, S.R_AN)):
            _capsule(m, P[a_], P[b_], 0.17 * sw)
            _capsule(m, P[b_], P[c_], 0.12 * sw)
    out = _feather(m, 0.05 * sw * fth) if m.any() else np.zeros(shape, np.float32)
    if "neck" in want and neck is not None:
        out = np.maximum(out, neck)
    return out if out.max() > 0 else None


def neck_mask(lm: np.ndarray, kp: np.ndarray, widths, shape, sx: float, sy: float, prob: np.ndarray,
              alpha, fth: float = 1.5, rgb: Optional[np.ndarray] = None, hands: Optional[np.ndarray] = None,
              parse: Optional[dict] = None) -> np.ndarray:
    """The neck: from the underside of the jaw down to the neck base, as wide as the neck.

    Built from the face first (the jaw line and which way is down), because the shoulders are often the least
    reliable joints exactly when the neck matters: a close-up crops them, a raised hand or hair covers them,
    and a dark costume under a coloured light swallows them. The shoulders only decide where the neck ends
    when they are where a neck would end. The skin colour model (re-centred on this frame's face) then keeps
    only skin: collar, necklace, hair falling beside the neck and the shirt stay out."""
    from ..warp import body as B
    h, w = shape
    L = lm[:, :2]
    if not np.isfinite(L[[0, 8, 16, 27]]).all():
        return np.zeros(shape, np.float32)
    Wf = float(np.hypot(*(L[16] - L[0])))
    ex = L[16] - L[0]
    ex = ex / max(np.hypot(*ex), 1e-6)
    down = np.array([-ex[1], ex[0]])
    if (L[8] - L[27]) @ down < 0:
        down = -down
    # half width of the neck: measured when the body model has it, else from the face
    hw = 0.30 * Wf
    if widths is not None:
        nw = widths[B.SEG_INDEX["neck"]]
        if np.isfinite(nw).any():
            m_ = float(np.nanmax(nw)) * sx
            if 0.18 * Wf < m_ < 0.6 * Wf:
                hw = m_
    # how far down: about three quarters of a face width, less where the shoulders really are higher up
    length = 0.75 * Wf
    if kp is not None:
        ls, rs = kp[S.L_SH, :2], kp[S.R_SH, :2]
        if np.isfinite(ls).all() and np.isfinite(rs).all():
            N = np.array([(ls[0] + rs[0]) / 2 * sx, (ls[1] + rs[1]) / 2 * sy])
            sw = float(np.hypot((ls[0] - rs[0]) * sx, (ls[1] - rs[1]) * sy))
            d = float((N - L[8]) @ down)
            if 1.2 * Wf < sw < 4.0 * Wf and 0.25 * Wf < d < 1.4 * Wf:
                length = min(length, d + 0.05 * Wf)
    # the top follows the jaw (the skin right under it is neck), kept to the neck's width
    jaw = L[3:14]
    top = jaw + down * 0.02 * Wf
    c = L[8] + down * 0.3 * Wf
    u = (top - c) @ ex
    keep = np.abs(u) <= hw * 1.25
    top = top[keep] if keep.sum() >= 3 else top
    base = L[8] + down * length
    bot = [base + ex * hw * 1.1, base - ex * hw * 1.1]
    m = np.zeros(shape, np.uint8)
    _poly(m, np.vstack([top, bot]))
    f = _feather(m, 0.035 * Wf * fth)
    f = f * np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.012 * Wf)) * 1.6, 0, 1)
    if alpha is not None:
        f = f * alpha
    if rgb is not None and f.max() > 0.01:
        # clearly brighter than the skin of the lower face: a white fur collar or a white shirt under the same
        # light, which the skin colour alone lets through (necks measured 0.6 to 1.0 times the face, fur and
        # a white shirt 1.6 to 1.8 times)
        Yl = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        lf = np.zeros(shape, np.uint8)
        _poly(lf, np.vstack([L[2:15], L[[35, 33, 31]]]))
        sel = (lf > 0) & (prob > 0.5)
        if sel.sum() > 30:
            yref = float(np.median(Yl[sel]))
            over = (Yl - (1.3 * yref + 4.0)) / max(0.3 * yref, 4.0)
            g = 1.0 - np.clip(cv2.GaussianBlur(over, (0, 0), max(1.0, 0.01 * Wf)), 0, 1)
            f = f * g
    if parse is not None and parse.get("trust", 0) > 0 and f.max() > 0.01:
        # the face parser tells the neck from the collar, the shirt, a necklace, hair and a hand in front of it,
        # which the skin colour cannot when they share the skin's colour. Below its crop it sees nothing: there
        # the neck only goes on as far as the parser was unsure.
        tr = float(parse["trust"])
        gn = np.clip(cv2.GaussianBlur((parse["neck"] + 0.5 * parse["skin"]).astype(np.float32), (0, 0),
                                      max(1.0, 0.015 * Wf)) * 1.8, 0, 1)
        cov = cv2.GaussianBlur(parse["cover"], (0, 0), max(1.0, 0.04 * Wf))
        f = f * (1.0 - tr * (1.0 - cov * gn) * 0.95)
    if hands is not None:
        # a hand or forearm in front of the neck is skin too, and the same person: carved out by its joints
        hm = np.zeros(shape, np.uint8)
        for i0 in range(0, len(hands) - 20, 21):
            hpts = hands[i0:i0 + 21]
            q = hpts[np.isfinite(hpts[:, :2]).all(1), :2] * [sx, sy]
            if len(q) >= 5:
                cv2.fillConvexPoly(hm, cv2.convexHull(np.round(q).astype(np.int32)), 255)
        if kp is not None:
            for wr, el in ((S.L_WR, S.L_EL), (S.R_WR, S.R_EL)):
                a_, b_ = kp[wr, :2], kp[el, :2]
                if np.isfinite(a_).all() and np.isfinite(b_).all():
                    cv2.line(hm, (int(a_[0] * sx), int(a_[1] * sy)), (int(b_[0] * sx), int(b_[1] * sy)), 255,
                             max(2, int(0.16 * Wf)))
        if hm.any():
            d_ = max(3, int(0.05 * Wf) | 1)
            hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_, d_)))
            f = f * (1.0 - _feather(hm, 0.02 * Wf))
    # a head tilted down hides the throat behind the chin: what lies under the jaw is then the chest, a
    # collar or the hands, never neck
    up_ = L[27] - L[8]
    up_ = up_ / max(1e-6, float(np.linalg.norm(up_)))
    span = float(abs((L[8] - L[17:27].mean(0)) @ up_)) / max(Wf, 1e-6)
    f = f * float(np.clip((span - 0.55) / 0.2, 0, 1))
    return f


def makeup_masks(lm: np.ndarray, shape, prob: np.ndarray, excl: np.ndarray, which, fth: float = 1.5):
    """Blush (cheek apples), shading (nose sides, under the cheekbones, inside the jaw line) and highlight
    (nose bridge, top of the cheekbones, forehead centre, chin) from the face landmarks."""
    h, w = shape
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    e1, e2 = L[36:42].mean(0), L[42:48].mean(0)
    iod = float(np.hypot(*(e2 - e1)))
    ex = (e2 - e1) / max(iod, 1e-6)
    down = np.array([-ex[1], ex[0]])
    if (L[8] - L[27]) @ down < 0:
        down = -down
    soft = np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.02 * Wf)) * 1.5, 0, 1) * (1 - excl)
    out = {}

    def ell(m, c, ax, ay, ang):
        cv2.ellipse(m, (int(round(c[0] * 4)), int(round(c[1] * 4))), (max(1, int(ax * 4)), max(1, int(ay * 4))),
                    ang, 0, 360, 255, -1, cv2.LINE_AA, 2)

    base_ang = float(np.degrees(np.arctan2(ex[1], ex[0])))
    if "blush" in which:
        m = np.zeros(shape, np.uint8)
        for eye, nose_side, jaw, sgn in ((e1, L[31], L[2], -1.0), (e2, L[35], L[14], 1.0)):
            c = 0.42 * nose_side + 0.33 * jaw + 0.25 * eye + ex * sgn * 0.06 * iod - down * 0.04 * iod
            ell(m, c, 0.30 * iod, 0.19 * iod, base_ang - sgn * 18)
        out["blush"] = _feather(m, 0.12 * iod * (0.7 + 0.3 * fth)) * soft
    if "shade" in which:
        m = np.zeros(shape, np.uint8)
        for inner, ala in ((L[39], L[31]), (L[42], L[35])):
            top = 0.55 * L[27] + 0.45 * inner
            _curve(m, np.array([top, 0.5 * top + 0.5 * ala - ex * 0.0, ala - down * 0.08 * iod]), 0.07 * iod)
        for side, corner, sgn in ((L[1], L[48], -1.0), (L[15], L[54], 1.0)):
            a_ = side + ex * (-sgn) * 0.04 * Wf
            b_ = a_ + (corner - a_) * 0.55 + down * 0.02 * Wf
            _curve(m, np.array([a_, 0.5 * (a_ + b_) + down * 0.03 * Wf, b_]), 0.11 * iod)
        jaw = L[3:14].astype(np.float64)
        cj = jaw.mean(0)
        inner_jaw = jaw + (L[30] - jaw) * 0.06
        _curve(m, inner_jaw, 0.05 * Wf)
        out["shade"] = _feather(m, 0.06 * iod * (0.7 + 0.3 * fth)) * soft
    if "hilite" in which:
        m = np.zeros(shape, np.uint8)
        _curve(m, np.array([L[27] + down * 0.15 * iod, L[28], L[29], L[30] - down * 0.05 * iod]), 0.06 * iod)
        for eye_o, sgn in ((L[36], -1.0), (L[45], 1.0)):
            c = eye_o + down * 0.30 * iod + ex * sgn * 0.02 * iod
            ell(m, c, 0.20 * iod, 0.07 * iod, base_ang - sgn * 20)
        brow_c = 0.5 * (L[21] + L[22])
        ell(m, brow_c - down * 0.40 * iod, 0.16 * iod, 0.22 * iod, base_ang)
        ell(m, L[8] - down * 0.13 * Wf, 0.11 * iod, 0.07 * iod, base_ang)
        ell(m, L[51] - down * 0.07 * iod, 0.08 * iod, 0.03 * iod, base_ang)
        out["hilite"] = _feather(m, 0.05 * iod * (0.7 + 0.3 * fth)) * soft
    return out


def lip_mask(lm: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    m = np.zeros((h, w), np.uint8)
    _poly(m, L[48:60])
    inner = np.zeros((h, w), np.uint8)
    _poly(inner, L[60:68])
    m = cv2.subtract(m, inner)
    return _feather(m, 0.008 * Wf)


def teeth_mask(lm: np.ndarray, rgb_f: np.ndarray) -> np.ndarray:
    h, w = rgb_f.shape[:2]
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    m = np.zeros((h, w), np.uint8)
    _poly(m, L[60:68])
    reg = _feather(m, 0.004 * Wf)
    if reg.max() < 0.05:
        return reg * 0
    Y = _luma(rgb_f)
    sat = _sat(rgb_f)
    ys = Y[reg > 0.5]
    lo = float(np.percentile(ys, 35)) if ys.size > 10 else 0.4
    wy = np.clip((Y - lo) / 0.15, 0, 1)
    ws = 1 - np.clip((sat - 0.22) / 0.18, 0, 1)
    return _feather(reg * wy * ws, 0.004 * Wf)


def eye_white_mask(lm: np.ndarray, rgb_f: np.ndarray) -> np.ndarray:
    h, w = rgb_f.shape[:2]
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    Y = _luma(rgb_f)
    sat = _sat(rgb_f)
    out = np.zeros((h, w), np.float32)
    for idx in (slice(36, 42), slice(42, 48)):
        e = L[idx]
        c = e.mean(0)
        m = np.zeros((h, w), np.uint8)
        _poly(m, c + (e - c) * 0.92)
        reg = _feather(m, 0.004 * Wf)
        if reg.max() < 0.05:
            continue
        ys = Y[reg > 0.5]
        if ys.size < 6:
            continue
        lo, hi = float(np.percentile(ys, 45)), float(np.percentile(ys, 85))
        wy = np.clip((Y - lo) / max(hi - lo, 0.03), 0, 1)
        ws = 1 - np.clip((sat - 0.25) / 0.2, 0, 1)
        out = np.maximum(out, reg * wy * ws)
    return _feather(out, 0.003 * Wf)


def knee_mask(kp: np.ndarray, widths, shape, sx: float, sy: float, prob: np.ndarray, alpha) -> np.ndarray:
    from ..warp import body as B
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    for kn, hip, an, seg in ((S.L_KN, S.L_HIP, S.L_AN, "knee_tl"), (S.R_KN, S.R_HIP, S.R_AN, "knee_tr")):
        k = kp[kn, :2]
        if not np.isfinite(k).all():
            continue
        hp = kp[hip, :2]
        ax = (kp[an, :2] - hp) if np.isfinite(kp[an, :2]).all() and np.isfinite(hp).all() else np.array([0.0, 1.0])
        L = float(np.hypot(*ax))
        if L < 4:
            continue
        r = np.nan
        if widths is not None:
            ww = widths[B.SEG_INDEX[seg]]
            if np.isfinite(ww).any():
                r = float(np.nanmean(ww))
        if not np.isfinite(r):
            r = 0.055 * L
        ang = float(np.degrees(np.arctan2(ax[1], ax[0])))
        c = (int(round(k[0] * sx)), int(round(k[1] * sy)))
        cv2.ellipse(out, c, (max(2, int(r * 1.45 * sx)), max(2, int(r * 1.05 * sx))), ang, 0, 360, 255, -1, cv2.LINE_AA)
    f = _feather(out, 3.0)
    f = f * prob
    if alpha is not None:
        f = f * alpha
    return f


# --------------------------------------------------------------------------- building a frame's jobs

VIS_LO = 0.12         # face visibility where the face retouch starts
VIS_FULL = 0.5        # and where it is at full strength
STEADY_R = 2          # frames each side
STEADY_SIGMA = 1.0
STEADY_LRU = 7


def prepare(cut, tracker, t: int, cache: Optional[dict] = None, steady: bool = True) -> Optional[RetouchFrame]:
    """Values and masks for frame t, held steady through time.

    Every mask is built from one frame's pixels (colour, brightness, the parser's view), and small changes in
    those (grain, a light flicker, the neck sliding into shadow) made its edges shake or a mask blink on and off
    between neighbouring frames. Here each mask is averaged with the same person's masks of the frames around it
    (STEADY_R each side, same shot), each one first moved onto this frame along the face's own motion (a
    similarity from the stabilised landmarks), so real movement is followed exactly and only the shake goes."""
    if not steady or tracker.store is None:
        return _prepare_raw(cut, tracker, t, cache)
    st = tracker.store
    try:
        raw = _raw_cached(cut, tracker, t, cache)
        if raw is None or raw.empty():
            return raw
        import copy as _copy
        rf = _copy.copy(raw)
        rf.jobs = []
        for j in raw.jobs:
            jj = _copy.copy(j)
            jj.masks = dict(j.masks)
            # the slider values are read now, not when the masks were cached
            pp = cut.params_for(j.pid)
            jj.v = {k_: pp.amount(k_, t) / 100.0 for k_ in RETOUCH_KEYS}
            jj.keep = float(np.clip(pp.skin_detail / 100.0, 0, 1))
            jj.grain_size = float(pp.grain_size)
            jj.blemish_size = float(np.clip(getattr(pp, "blemish_size", 50.0) / 100.0, 0, 1))
            if j.skin is not None:
                jj.skin = _copy.copy(j.skin)
                jj.skin.strength = float(np.clip(jj.v["skin"], 0, 1))
                jj.skin.keep = jj.keep
            rf.jobs.append(jj)
        return _steady(cut, tracker, st, rf, cache)
    except Exception:
        import traceback
        traceback.print_exc()
        return _prepare_raw(cut, tracker, t, cache)


def _sig(cut, tracker) -> tuple:
    """What the raw masks depend on besides the frame: which ones exist (values on or off), scope, feather."""
    vals = []
    for person in cut.targets():
        pp = cut.params_for(person.id)
        fth = pp.feather_factor() if hasattr(pp, "feather_factor") else 1.5
        on = []
        for k_ in RETOUCH_KEYS:
            c_ = pp.amounts.get(k_)
            if c_ is not None and (abs(c_.base) > 1e-4 or any(abs(v_) > 1e-4 for v_ in c_.keys.values())):
                on.append(k_)
        vals.append((person.id, pp.skin_scope, round(float(fth), 3), tuple(on)))
    return (id(tracker.store), getattr(cut, "analysis_version", 0), tuple(vals))


def _raw_cached(cut, tracker, s: int, cache):
    from collections import OrderedDict
    lru = tracker.__dict__.get("_steady_lru")
    sig = _sig(cut, tracker)
    # the session clears its mask cache whenever an edit changes masks: start over then too
    if lru is None or tracker.__dict__.get("_steady_sig") != sig or (cache is not None and len(cache) == 0):
        lru = tracker.__dict__["_steady_lru"] = OrderedDict()
        tracker.__dict__["_steady_sig"] = sig
    if s in lru:
        lru.move_to_end(s)
        return lru[s]
    rf = _prepare_raw(cut, tracker, s, cache)
    if rf is not None:
        for j in rf.jobs:
            j.masks = {k: v.astype(np.float16) for k, v in j.masks.items()}
    lru[s] = rf
    while len(lru) > STEADY_LRU:
        lru.popitem(last=False)
    return rf


def _steady(cut, tracker, st, rf: RetouchFrame, cache) -> RetouchFrame:
    t = rf.t
    px = tracker.proxy
    N = st.final.shape[1]
    k = np.array([px.sx, px.sy])
    for j in rf.jobs:
        if j.face is None or not j.masks:
            continue
        Lt = j.face[:, :2] * k
        acc = {n: m.astype(np.float32) for n, m in j.masks.items()}
        wsum = 1.0
        for d in range(1, STEADY_R + 1):
            w = float(np.exp(-0.5 * (d / STEADY_SIGMA) ** 2))
            for s in (t - d, t + d):
                if not (0 <= s < N) or not st.present[j.p, s] or not st.connected(s, t):
                    continue
                if not np.isfinite(st.face[j.p, s, :, :2]).all():
                    continue
                rs = _raw_cached(cut, tracker, s, cache)
                js = next((x for x in rs.jobs if x.pid == j.pid), None) if rs is not None else None
                if js is None or js.face is None:
                    continue
                Ls = js.face[:, :2] * k
                A, _ = cv2.estimateAffinePartial2D(Ls.astype(np.float32), Lt.astype(np.float32), method=cv2.LMEDS)
                if A is None:
                    continue
                for n in acc:
                    ms = js.masks.get(n)
                    if ms is None:
                        continue              # that frame had none (below threshold): counts as empty
                    acc[n] += w * cv2.warpAffine(ms.astype(np.float32), A, (acc[n].shape[1], acc[n].shape[0]),
                                                 flags=cv2.INTER_LINEAR, borderValue=0)
                wsum += w
        veto = getattr(j, "veto", None)
        if veto is not None and wsum > 1.0:
            h_, w_ = next(iter(acc.values())).shape
            v_ = cv2.resize(veto.astype(np.float32), (w_, h_), interpolation=cv2.INTER_LINEAR)
            wf_ = float(np.hypot(*(Lt[16] - Lt[0])))
            v_ = np.clip(cv2.GaussianBlur(v_, (0, 0), max(1.0, 0.02 * wf_)) * 2.5, 0, 1)
            # where this frame sees no skin, a neighbour's mask may not hold more than this frame's own
            for n in acc:
                own = j.masks[n].astype(np.float32)
                avg = acc[n] / wsum
                acc[n] = (own + (avg - own).clip(None, 0) + (avg - own).clip(0, None) * v_) * wsum
        j.masks = {n: (m / wsum).astype(np.float32) for n, m in acc.items()}
    return rf


def person_masks(cut, tracker, t: int, pid: int) -> Optional[RetouchFrame]:
    """The selected person's skin area for the K overlay, also before any skin slider was moved (with the
    person's own region and scope settings), so the area can be checked before retouching."""
    return _prepare_raw(cut, tracker, t, None, only=pid)


def _prepare_raw(cut, tracker, t: int, cache: Optional[dict] = None, only: Optional[int] = None) -> Optional[RetouchFrame]:
    """One frame's masks on their own (see prepare). only: just this person, skin shown even at 0."""
    st = tracker.store
    if st is None:
        return None
    px = tracker.proxy
    jobs = []
    rgb_u8 = None
    rgb_f = None
    people = cut.targets() if only is None else [q for q in cut.persons if q.id == only]
    for person in people:
        if not st.has(person.id):
            continue
        p = st.index(person.id)
        if not st.present[p, t]:
            continue
        pp = cut.params_for(person.id)
        v = {k: pp.amount(k, t) / 100.0 for k in RETOUCH_KEYS}
        if only is not None:
            v["skin"] = max(v["skin"], 0.5)
        if not any(abs(x) > 1e-4 for x in v.values()):
            continue
        face = None
        fw = 0.0
        if st.has_face and np.isfinite(st.face[p, t, :, :2]).all():
            face = st.face[p, t]
            # face visibility as a strength: full strength once the face is reasonably sure, so the retouch does
            # not breathe with small confidence changes (a zoom, a turn); it still fades out where the face is
            # really doubtful (heavily covered, carried by motion alone for many frames)
            fw = float(np.clip((float(st.face_vis[p, t]) - VIS_LO) / (VIS_FULL - VIS_LO), 0.0, 1.0))
        sc = float(st.scale[p, t]) if np.isfinite(st.scale[p, t]) else 0.0
        job = Job(p, person.id, v, float(np.clip(pp.skin_detail / 100.0, 0, 1)), pp.skin_scope,
                  float(pp.grain_size), face, fw, sc,
                  blemish_size=float(np.clip(getattr(pp, "blemish_size", 50.0) / 100.0, 0, 1)))
        fth = pp.feather_factor() if hasattr(pp, "feather_factor") else 1.5
        if face is not None:
            job.wf_src = float(np.hypot(*(face[16, :2] - face[0, :2])))
        shape = (px.h, px.w)
        lm = None
        if face is not None:
            lm = face.copy()
            lm[:, 0] *= px.sx
            lm[:, 1] *= px.sy
        state = {}

        def base():
            if "rgb" not in state:
                state["rgb"] = np.asarray(px.frame(t))
                state["rgbf"] = state["rgb"].astype(np.float32) / 255.0
                from ..track import facelayer as FL
                layer = FL.load(tracker)
                fmaps = layer.maps(person.id, t, shape) if (layer is not None and lm is not None) else None
                centre = layer.colour(person.id, t) if (layer is not None and lm is not None) else None
                state["parse"] = fmaps
                state["model"] = SK.frame_model_for(tracker, p, t, state["rgb"], lm, centre)
                state["prob"] = SK.face_skin_probability(state["rgb"], state["model"], lm if fw > 0.3 else None,
                                                         fmaps if fw > 0.3 else None)
                state["alpha"] = SK.person_alpha(tracker, t, p, (px.w, px.h))
            return state

        def get(name, fn, extra=""):
            key = (t, person.id, name, extra)
            if cache is not None and key in cache:
                return cache[key]
            m = fn()
            if cache is not None:
                cache[key] = m
                while len(cache) > 400:
                    cache.pop(next(iter(cache)))
            return m

        from ..core.project import scope_set
        regions = scope_set(job.scope)

        def body():
            """The chosen body regions of this person (neck, shoulders, arms, hands, legs), or None."""
            if not (regions - {"face"}):
                return None
            if "body" not in state:
                kpb, wdb = _steady_body(st, p, t)
                hd = None
                if getattr(st, "has_hands", False) and np.isfinite(st.hand[p, t][:, :2]).any():
                    hd = st.hand[p, t]
                nk = None
                if "neck" in regions and lm is not None:
                    b_ = base()
                    nk = neck_mask(lm, kpb, wdb, shape, px.sx, px.sy, b_["prob"], b_["alpha"], fth, b_["rgb"], None,
                                   b_.get("parse"))
                state["body"] = body_regions(regions, kpb, hd, shape, px.sx, px.sy, nk, fth)
            return state["body"]

        tone_on = any(abs(v.get(k, 0)) > 1e-4 for k in ("skin_bright", "skin_warm", "skin_sat", "grain", "redness"))
        if tone_on:
            def mk_tone():
                b = base()
                return tone_mask(lm, shape, b["prob"], b["alpha"], job.scope, fw, fth, body())
            tm = get("tone", mk_tone, f"{job.scope}:{fth:.2f}")
            if tm.max() > 0.01:
                job.masks["tone"] = tm
        skin_on = v["skin"] > 1e-4
        if skin_on:
            sj = SK.SkinJob(p, float(np.clip(v["skin"], 0, 1)), job.keep, job.scope, face, fw, sc,
                            st.final[p, t, 0:5, :2].copy())

            def mk_skin():
                b = base()
                return SK.skin_mask(sj, b["rgb"], px.sx, px.sy, b["alpha"], b["model"], b.get("parse"), body())
            m = get("skin", mk_skin, job.scope)
            if m.max() > 0.01:
                job.skin = sj
                job.masks["skin"] = m
        if v.get("blemish", 0) > 1e-4 or v.get("shine", 0) > 1e-4:
            def mk_skinx():
                b = base()
                reg = tone_mask(lm, shape, b["prob"], b["alpha"], job.scope, fw, fth, body())
                if lm is not None:
                    reg = reg * (1 - feature_exclusion(lm, shape))
                return np.clip(reg * np.clip(b["prob"] * 1.4, 0, 1), 0, 1)
            sx_ = get("skinx", mk_skinx, f"{job.scope}:{fth:.2f}")
            if sx_.max() > 0.01:
                job.masks["skinx"] = sx_
        if lm is not None and fw > 0.05:
            def pa():
                b = base()
                return b["prob"] * (b["alpha"] if b["alpha"] is not None else 1.0)
            if v.get("dark_circle", 0) > 1e-4:
                job.masks["dark"] = get("dark", lambda: under_eye_mask(lm, shape, pa(), fth), f"{fth:.2f}") * fw
            if v.get("shine", 0) > 1e-4:
                job.masks["hot"] = get("hot", lambda: hot_zone_mask(lm, shape, pa(), fth), f"{fth:.2f}") * fw
            if v["nasolabial"] > 1e-4:
                job.masks["naso"] = get("naso", lambda: wrinkle_masks(lm, shape, pa(), "naso", fth, 1.7),
                                        f"{fth:.2f}:w") * fw
            if v["mouth_lines"] > 1e-4:
                job.masks["mouthl"] = get("mouthl", lambda: wrinkle_masks(lm, shape, pa(), "mouth", fth, 1.5),
                                          f"{fth:.2f}:w") * fw
            if v["nasolabial"] > 1e-4 or v["mouth_lines"] > 1e-4 or v.get("dark_circle", 0) > 1e-4:
                def mk_ex():
                    b = base()
                    ex = feature_exclusion(lm, shape)
                    return np.clip(ex + (1 - np.clip(b["prob"] * 1.5, 0, 1)), 0, 1)
                job.masks["excl"] = get("excl", mk_ex)
            if v.get("neck_lines", 0) > 1e-4 or v.get("neck_bumps", 0) > 1e-4:
                kpn, wdn = _steady_body(st, p, t)

                def mk_neck():
                    b_ = base()
                    # every hand in the picture (anyone's hand can be in front of this neck): tracked hands, else
                    # the whole-body model's hand points where it is sure of them
                    hl = []
                    for q_ in np.nonzero(st.present[:, t])[0]:
                        h_ = st.hand[q_, t] if getattr(st, "has_hands", False) else None
                        if h_ is None or not np.isfinite(h_[:, :2]).any():
                            h_ = st.hand_raw[q_, t].copy()
                            h_[~(np.nan_to_num(h_[:, 2]) >= 1.5)] = np.nan
                        hl.append(h_)
                    hnd = np.concatenate(hl, 0) if hl else None
                    return neck_mask(lm, kpn, wdn, shape, px.sx, px.sy, b_["prob"], b_["alpha"], fth,
                                     b_["rgb"], hnd, b_.get("parse"))
                nm_ = get("neck", mk_neck, f"{fth:.2f}")
                if nm_.max() > 0.01:
                    job.masks["neck"] = nm_ * fw
            mk_which = [n for n, key in (("blush", "blush"), ("shade", "shading"), ("hilite", "highlight"))
                        if v.get(key, 0) > 1e-4]
            if mk_which:
                def mk_makeup(n):
                    b_ = base()
                    ex_ = feature_exclusion(lm, shape)
                    return makeup_masks(lm, shape, b_["prob"], ex_, [n], fth)[n]
                for n in mk_which:
                    job.masks[n] = get(n, lambda n=n: mk_makeup(n), f"{fth:.2f}") * fw
            if any(abs(v[k]) > 1e-4 for k in ("lip_hue", "lip_sat", "lip_bright")):
                job.masks["lips"] = get("lips", lambda: lip_mask(lm, shape)) * fw
            if v["teeth"] > 1e-4:
                job.masks["teeth"] = get("teeth", lambda: teeth_mask(lm, base()["rgbf"])) * fw
            if v["eye_white"] > 1e-4:
                job.masks["eyew"] = get("eyew", lambda: eye_white_mask(lm, base()["rgbf"])) * fw
        if v["knee_dark"] > 1e-4:
            kp = st.final[p, t]

            def mk_knee():
                b = base()
                return knee_mask(kp, st.widths[p, t], shape, px.sx, px.sy, b["prob"], b["alpha"])
            km = get("knee", mk_knee)
            if km.max() > 0.01:
                job.masks["knee"] = km
        if job.masks:
            if "prob" in state:
                # this frame's own view of what is skin (colour, parser, texture): the steadied masks may not
                # carry a neighbour's mask onto something that covers the face here (a hand, a sleeve)
                pr_ = state["prob"]
                if state.get("alpha") is not None:
                    pr_ = pr_ * state["alpha"]
                job.veto = cv2.resize(pr_, (max(1, pr_.shape[1] // 4), max(1, pr_.shape[0] // 4)),
                                      interpolation=cv2.INTER_AREA).astype(np.float16)
            jobs.append(job)
    if not jobs:
        return None
    return RetouchFrame(t, jobs, (px.w, px.h), (cut.info.width, cut.info.height))


# --------------------------------------------------------------------------- operations

def _roi(m: np.ndarray, pad: int):
    ys, xs = np.nonzero(m > 0.004)
    if not len(xs):
        return None
    h, w = m.shape
    return (max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1))


def _lift(I: np.ndarray, m: np.ndarray, k: float, sig: float, keep: float = 0.35) -> np.ndarray:
    """Even out a darker/redder fold with its surroundings, keeping a share of the fine texture."""
    if k < 1e-4:
        return I
    r = _roi(m, int(3 * sig) + 4)
    if r is None:
        return I
    y0, y1, x0, x1 = r
    roi = I[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1]
    wsur = (1 - np.clip(mm * 1.5, 0, 1)).astype(np.float32)
    num = cv2.GaussianBlur(roi * wsur, (0, 0), sig)
    den = cv2.GaussianBlur(wsur, (0, 0), sig)
    sur = np.where(den > 1e-3, num / np.maximum(den, 1e-3), roi)
    small = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.18 * sig))
    target = sur + (roi - small) * keep
    out = I.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + mm * k * (target - roi)
    return out


def _fold_fill(I: np.ndarray, m: np.ndarray, excl: Optional[np.ndarray], k: float, sig_f: float, sig_s: float,
               luma: bool = True) -> np.ndarray:
    """Fill in the shading of a crease from the skin around it.

    The skin next to the crease (eyes, lips, nostrils and non-skin left out) gives the level the crease
    should have; the crease's own shading at its width is compared with it and only the missing part is
    added back. Fine texture is untouched, and luma is only ever lightened, so nothing turns grey or dark."""
    if k < 1e-4:
        return I
    r = _roi(m, int(3 * sig_s) + 4)
    if r is None:
        return I
    y0, y1, x0, x1 = r
    roi = I[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1]
    wsur = (1 - np.clip(mm * 1.8, 0, 1))
    if excl is not None:
        wsur = wsur * (1 - excl[y0:y1, x0:x1])
    wsur = wsur.astype(np.float32)
    num = cv2.GaussianBlur(roi * wsur, (0, 0), sig_s)
    den = cv2.GaussianBlur(wsur, (0, 0), sig_s)
    loc = cv2.GaussianBlur(roi, (0, 0), sig_f)
    ok = den > 0.05
    sur = np.where(ok, num / np.maximum(den, 1e-3), loc)
    d = sur - loc
    if luma:
        d = np.clip(d, 0.0, 0.25)
    else:
        d = np.clip(d, -0.035, 0.035)
    d = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.6 * sig_f))
    out = I.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + mm * k * d * ok
    return out


def _fold_find(Y: np.ndarray, band: np.ndarray, wf: float, width: float) -> np.ndarray:
    """Where in the search band the fold really is: a dark valley at the fold's width (curved strongly across,
    little along), darker than the skin beside it. Returns 0..1 at Y resolution (only inside the band's ROI).
    Continuous, so it does not blink from frame to frame."""
    out = np.zeros(Y.shape, np.float32)
    r = _roi(band, int(4 * width) + 6)
    if r is None:
        return out
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    bb = band[y0:y1, x0:x1].astype(np.float32)
    sv = max(1.0, 0.45 * width)
    Yh = cv2.GaussianBlur(roi, (0, 0), sv)
    Ixx = cv2.Sobel(Yh, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(Yh, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(Yh, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    line = np.where(l1 > 1e-7, 1 - _ss(0.35, 0.8, np.abs(l2) / np.maximum(l1, 1e-7)), 0.0).astype(np.float32)
    # depth against the skin a fold-width or two to either side
    Ym = cv2.GaussianBlur(roi, (0, 0), max(0.8, 0.25 * width))
    wsk = (bb > 0.02).astype(np.float32)
    sl = max(2.0, 1.6 * width)
    b0 = _gm(Ym, wsk, sl)
    dev = Ym - b0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sl), 0)) + 0.003
    wv = wsk * np.exp(-(np.clip(-dev, 0, None) / tex) ** 2)
    base = _gm(Ym, wv, sl)
    D = np.clip(base - Ym, 0, 0.3)
    rel = D / np.maximum(base, 0.02)                          # depth relative to the skin's level (any grade)
    ev = _ss(0.015, 0.08, rel) * line * np.clip(bb * 1.5, 0, 1)
    rd = max(1, int(round(0.6 * width)))
    ev = cv2.dilate(ev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
    ev = np.clip(cv2.GaussianBlur(ev, (0, 0), max(0.8, 0.5 * width)) * 1.4, 0, 1)
    out[y0:y1, x0:x1] = ev
    return out


def _gm(x: np.ndarray, w: np.ndarray, sig: float) -> np.ndarray:
    """Gaussian mean of x weighted by w (falls back to the plain blur where w has no support)."""
    num = cv2.GaussianBlur(x * w, (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), cv2.GaussianBlur(x, (0, 0), sig))


SKIN_CHROMA = 0.3    # colour smoothing relative to brightness smoothing (fit to a hand retouch: 0.55 overdid colour)


def _ss(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _croi(y0, y1, x0, x1, ys, cs_):
    """Luma ROI -> the same area in the chroma planes (any subsampling)."""
    fy, fx = cs_[0] / ys[0], cs_[1] / ys[1]
    c0, d0 = int(y0 * fy), int(x0 * fx)
    c1 = min(cs_[0], max(c0 + 1, int(np.ceil(y1 * fy))))
    d1 = min(cs_[1], max(d0 + 1, int(np.ceil(x1 * fx))))
    return c0, c1, d0, d1


def _blemish(Y, Cb, Cr, mY, mC, k: float, size: float, cs: float):
    """Remove small dark or red spots (blemishes, acne, small moles) inside the skin mask.

    A spot is a blob (curved in both directions, unlike a hair strand or a crease) that is darker or redder
    than the skin around it by clearly more than the skin's own texture, and no larger than `size` (Y px).
    Its broad shading and colour are replaced by the surrounding skin's; the finest texture stays. The test
    is continuous (no on/off threshold), so spots fade in and out smoothly from frame to frame."""
    r = _roi(mY, int(2.5 * size) + 6)
    if r is None or k < 1e-4:
        return Y, Cb, Cr
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = mY[y0:y1, x0:x1].astype(np.float32)
    sig_s = max(0.6, 0.10 * size)            # finer than this is texture and stays
    sig_l = max(2.0, 0.85 * size)            # the skin around a spot
    Ys = cv2.GaussianBlur(roi, (0, 0), sig_s)
    wsk = (mm > 0.05).astype(np.float32)
    base0 = _gm(Ys, wsk, sig_l)
    dev = Ys - base0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sig_l), 0)) + 0.004
    # the surround without the spot itself (dark outliers do not count)
    w = wsk * np.exp(-(np.clip(-dev, 0, None) / tex) ** 2)
    base = _gm(Ys, w, sig_l)
    D = np.clip(base - Ys, 0, 0.2)
    # blob test on the Hessian at the spot scale: both curvatures positive (a dark dot), not a line
    Yh = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.28 * size))
    Ixx = cv2.Sobel(Yh, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(Yh, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(Yh, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    blob = np.where(l1 > 1e-6, _ss(0.08, 0.40, l2 / np.maximum(l1, 1e-6)), 0.0).astype(np.float32)
    blob = cv2.GaussianBlur(blob, (0, 0), max(0.8, 0.35 * size))
    # depth judged against the skin's own level: the same spot is a smaller code-value step on a face in a
    # dark or coloured grade than in daylight
    rel = D / np.maximum(base, 0.05)
    # a short mark (a small scratch, a dry flake, a thin spot) is not round but is no line either: a crease or a
    # hair runs on. Dark marks whose whole extent is within the spot size count like a blob.
    # (from a hand retouched shot, TEST003: the one mark the retoucher took was such a short streak)
    cand = ((D / tex > 1.6) & (rel > 0.015) & (mm > 0.5)).astype(np.uint8)
    short = np.zeros_like(blob)
    if cand.any():
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        ok_ = (np.maximum(st_[:, 2], st_[:, 3]) <= 1.0 * size) & (st_[:, 4] >= 3)
        ok_[0] = False
        if ok_.any():
            short = ok_[lab_].astype(np.float32)
            short = cv2.GaussianBlur(short, (0, 0), max(0.6, 0.15 * size))
    score = _ss(1.2, 2.8, D / tex) * _ss(0.012, 0.05, rel) * np.clip(np.maximum(blob * 1.6, short * 1.5), 0, 1)
    # only with skin all around: where the skin meets hair, a hat brim or the jaw shadow, the dark side bleeds
    # into the skin's edge and looks like a spot (the hand retouch never touched those)
    rin = max(1, int(round(0.6 * size)))
    inner = cv2.erode((mm > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rin + 1, 2 * rin + 1)))
    inner = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), max(0.6, 0.3 * size))
    score = score * inner
    # redness (acne) in the colour planes, measured at luma resolution
    c0, c1, d0, d1 = _croi(y0, y1, x0, x1, Y.shape, Cr.shape)
    Crr = cv2.resize(Cr[c0:c1, d0:d1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) \
        if Cr.shape != Y.shape else Cr[y0:y1, x0:x1]
    Crs = cv2.GaussianBlur(Crr.astype(np.float32), (0, 0), sig_s)
    cbase0 = _gm(Crs, wsk, sig_l)
    cdev = Crs - cbase0
    ctex = np.sqrt(np.maximum(_gm(cdev * cdev, wsk, sig_l), 0)) + 0.002
    cw = wsk * np.exp(-(np.clip(cdev, 0, None) / ctex) ** 2)
    cbase = _gm(Crs, cw, sig_l)
    R = np.clip(Crs - cbase, 0, 0.1)
    rscore = _ss(1.5, 3.0, R / ctex) * _ss(0.004, 0.015, R) * np.clip(blob * 1.6 + 0.3, 0, 1)
    spot = np.maximum(score, rscore * inner)
    # the test is strongest in the middle of a spot: spread it over the whole spot so no ring is left
    rd = max(1, int(round(0.5 * size)))
    spot = cv2.dilate(spot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
    spot = np.clip(cv2.GaussianBlur(spot, (0, 0), max(0.6, 0.25 * size)) * 1.3, 0, 1) * mm
    Yo = Y.astype(np.float32, copy=True)
    Yo[y0:y1, x0:x1] = roi + k * spot * D
    # colour: back to the surrounding skin's colour where a spot was taken
    Cbo, Cro = Cb.astype(np.float32, copy=True), Cr.astype(np.float32, copy=True)
    if Cb.shape == Y.shape:
        c0, c1, d0, d1 = y0, y1, x0, x1
        spc = spot
    else:
        spc = cv2.resize(spot, (d1 - d0, c1 - c0), interpolation=cv2.INTER_AREA)
    sgc, slc = max(0.5, sig_s * cs), max(1.0, sig_l * cs)
    wkc = (cv2.resize(wsk, (d1 - d0, c1 - c0), interpolation=cv2.INTER_NEAREST) if spc.shape != wsk.shape else wsk)
    for P, O in ((Cb, Cbo), (Cr, Cro)):
        pr = P[c0:c1, d0:d1].astype(np.float32)
        ps = cv2.GaussianBlur(pr, (0, 0), sgc)
        wexcl = wkc * (1 - np.clip(spc * 2, 0, 1))
        pb = _gm(ps, wexcl, slc)
        O[c0:c1, d0:d1] = pr + k * spc * np.clip(pb - ps, -0.06, 0.06)
    return Yo, Cbo, Cro


def _shine(Y, Cb, Cr, mY, mC, k: float, wf: float, cs: float, zone: Optional[np.ndarray] = None):
    """Press down oily specular shine: patches brighter than the skin base and whiter than the skin colour.
    Pores and fine texture stay; the shine's colour goes back to the skin's."""
    r = _roi(mY, int(0.12 * wf) + 6)
    if r is None or k < 1e-4:
        return Y, Cb, Cr
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = mY[y0:y1, x0:x1].astype(np.float32)
    sig_s = max(0.7, 0.006 * wf)
    sig_b = max(4.0, 0.11 * wf)
    Ys = cv2.GaussianBlur(roi, (0, 0), sig_s)
    wsk = (mm > 0.05).astype(np.float32)
    b0 = _gm(Ys, wsk, sig_b)
    dev = Ys - b0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sig_b), 0)) + 0.006
    w = wsk * np.exp(-(np.clip(dev, 0, None) / tex) ** 2)
    base = _gm(Ys, w, sig_b)
    H = np.clip(Ys - base, 0, 0.35)
    # whiteness: chroma lower than the skin around it
    if Cb.shape == Y.shape:
        cb, cr = Cb[y0:y1, x0:x1], Cr[y0:y1, x0:x1]
    else:
        sz = (x1 - x0, y1 - y0)
        c0, c1, d0, d1 = _croi(y0, y1, x0, x1, Y.shape, Cb.shape)
        cb = cv2.resize(Cb[c0:c1, d0:d1].astype(np.float32), sz, interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(Cr[c0:c1, d0:d1].astype(np.float32), sz, interpolation=cv2.INTER_LINEAR)
    chroma = cv2.GaussianBlur(np.sqrt(cb.astype(np.float32) ** 2 + cr.astype(np.float32) ** 2), (0, 0), sig_s)
    cbase = _gm(chroma, w, sig_b)
    white = np.clip(1 - chroma / np.maximum(cbase, 1e-3), 0, 1)
    # only the front of the face: rim and back light along the hairline, jaw and cheek edge is lighting
    # design (it shapes the face), not oil. Hand retouches leave it and press the broad frontal hotspots
    # (forehead, under-eye cheek, nose bridge) a little.
    solid = (mm > 0.3).astype(np.uint8)
    cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(solid, cnts, -1, 1, -1)             # eyes and mouth are holes, not edges of the face
    inside = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    interior = _ss(0.035 * wf, 0.10 * wf, inside)
    # broad, low hotspots count too (a whole forehead or cheek top catching the key light)
    Hm = np.clip(cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.02 * wf)) - base, 0, 0.35)
    H = np.maximum(H, Hm)
    sw = _ss(0.003, 0.035, H) * (0.5 + 0.5 * _ss(0.02, 0.3, white)) * mm * interior
    zr = None
    if zone is not None:
        # in the hotspot zones a lower lift already counts, and the bright speckle inside is calmed too
        zr = np.clip(zone[y0:y1, x0:x1], 0, 1) * mm
        sw = np.maximum(sw, _ss(0.0, 0.02, H) * zr)
    sw = cv2.GaussianBlur(sw, (0, 0), sig_s)
    Yo = Y.astype(np.float32, copy=True)
    out = roi - k * 0.5 * sw * H
    if zr is not None:
        fine = roi - Ys
        out = out - k * 0.3 * zr * np.clip(fine, 0, None)
    Yo[y0:y1, x0:x1] = out
    Cbo, Cro = Cb.astype(np.float32, copy=True), Cr.astype(np.float32, copy=True)
    if Cb.shape == Y.shape:
        c0, c1, d0, d1 = y0, y1, x0, x1
        swc = sw
        wc = w
    else:
        swc = cv2.resize(sw, (d1 - d0, c1 - c0), interpolation=cv2.INTER_AREA)
        wc = cv2.resize(w, (d1 - d0, c1 - c0), interpolation=cv2.INTER_AREA)
    for P, O in ((Cb, Cbo), (Cr, Cro)):
        pr = P[c0:c1, d0:d1].astype(np.float32)
        ps = cv2.GaussianBlur(pr, (0, 0), max(0.5, sig_s * cs))
        pb = _gm(ps, wc, max(2.0, sig_b * cs))
        O[c0:c1, d0:d1] = pr + k * 0.7 * swc * np.clip(pb - ps, -0.05, 0.05)
    return Yo, Cbo, Cro


def _bump_soften(Y, Cb, Cr, mY, mC, k: float, wf: float, cs: float):
    """Goose bumps (and similar fine bumpy texture): the band of detail between the grain and the size of a
    bump is turned down, by at most 85 %, so the skin still has its texture and nothing looks wiped.
    Detail much stronger than the skin's own bumps (a necklace, the edge of a collar, a crease, a tendon's
    shading) is left alone: the amount falls off with how far a pixel's detail stands above the typical
    bump in the mask, measured on this frame, so it holds in dark and flash-lit frames alike."""
    if mY.max() < 0.02:
        return Y, Cb, Cr
    s0 = max(0.6, 0.0015 * wf)            # finer than this is grain and pores: kept
    s1 = max(2.0, 0.016 * wf)             # about the size of one bump with its blotch of colour
    fine = cv2.GaussianBlur(Y, (0, 0), s0)
    coarse = cv2.GaussianBlur(Y, (0, 0), s1)
    d = fine - coarse
    sel = mY > 0.5
    if sel.sum() < 50:
        sel = mY > 0.1
    if sel.sum() < 20:
        return Y, Cb, Cr
    ref = float(np.median(np.abs(d[sel]))) * 1.4826 + 1e-5
    # the bumps themselves (even their tops) are turned down evenly; only detail far beyond them is spared
    # (anything the skin mask let through that is not skin: a glint of the necklace, a hair across the neck)
    keep_strong = np.exp(-np.square(np.abs(d) / (6.0 * ref)) * 0.5)
    a = np.clip(mY, 0, 1) * (0.85 * k) * keep_strong
    Y = Y - a * d
    if Cb is not None and Cr is not None:
        s0c, s1c = max(0.6, s0 * cs), max(1.5, s1 * cs)
        ac = np.clip(mC, 0, 1) * (0.8 * k)
        Cb = Cb - ac * (cv2.GaussianBlur(Cb, (0, 0), s0c) - cv2.GaussianBlur(Cb, (0, 0), s1c))
        Cr = Cr - ac * (cv2.GaussianBlur(Cr, (0, 0), s0c) - cv2.GaussianBlur(Cr, (0, 0), s1c))
    return Y, Cb, Cr


def _crease_lift(Y, Cb, Cr, mY, k: float, width: float):
    """Fill thin dark creases (neck lines): a dark valley, curved strongly across and hardly along, darker than
    the skin around it by more than its texture. Only the crease's shading is lifted; broad shading stays."""
    r = _roi(mY, int(3 * width) + 6)
    if r is None or k < 1e-4:
        return Y, Cb, Cr
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = mY[y0:y1, x0:x1].astype(np.float32)
    sig_s = max(0.5, 0.25 * width)
    sig_l = max(2.0, 2.4 * width)
    Ys = cv2.GaussianBlur(roi, (0, 0), sig_s)
    wsk = (mm > 0.05).astype(np.float32)
    b0 = _gm(Ys, wsk, sig_l)
    dev = Ys - b0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sig_l), 0)) + 0.004
    w = wsk * np.exp(-(np.clip(-dev, 0, None) / tex) ** 2)
    # texture of the skin itself, without the creases
    tex = np.sqrt(np.maximum(_gm(dev * dev, w, sig_l), 0)) + 0.004
    base = _gm(Ys, w, sig_l)
    D = np.clip(base - Ys, 0, 0.25)
    Yh = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.6 * width))
    Ixx = cv2.Sobel(Yh, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(Yh, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(Yh, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    line = np.where(l1 > 1e-6, 1 - _ss(0.25, 0.6, np.abs(l2) / np.maximum(l1, 1e-6)), 0.0).astype(np.float32)
    score = _ss(1.0, 2.6, D / tex) * _ss(0.004, 0.02, D) * line
    rd = max(1, int(round(0.4 * width)))
    score = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
    sp = np.clip(cv2.GaussianBlur(score, (0, 0), max(0.6, 0.3 * width)) * 1.3, 0, 1) * mm
    Yo = Y.astype(np.float32, copy=True)
    Yo[y0:y1, x0:x1] = roi + k * sp * D
    return Yo, Cb, Cr


def _redness(Cb, Cr, Y, mC, mY, k: float, wf: float, cs: float):
    """Bring red patches (cheeks, around the nose) back to the face's own skin colour."""
    r = _roi(mC, 4)
    if r is None or k < 1e-4:
        return Y, Cb, Cr
    c0, c1, d0, d1 = r
    mm = mC[c0:c1, d0:d1].astype(np.float32)
    sel = mm > 0.5
    if sel.sum() < 30:
        return Y, Cb, Cr
    cr = Cr[c0:c1, d0:d1].astype(np.float32)
    cb = Cb[c0:c1, d0:d1].astype(np.float32)
    sg = max(0.8, 0.008 * wf * cs)
    crs = cv2.GaussianBlur(cr, (0, 0), sg)
    cbs = cv2.GaussianBlur(cb, (0, 0), sg)
    cr_ref = float(np.percentile(crs[sel], 35))
    low = sel & (crs <= cr_ref)
    cb_ref = float(np.median(cbs[low])) if low.sum() > 10 else float(np.median(cbs[sel]))
    ex = np.clip(crs - cr_ref, 0, 0.12)
    on = _ss(0.0, 0.025, ex)
    Cro, Cbo = Cr.astype(np.float32, copy=True), Cb.astype(np.float32, copy=True)
    Cro[c0:c1, d0:d1] = cr - k * 0.85 * mm * ex
    Cbo[c0:c1, d0:d1] = cb + k * 0.5 * mm * on * np.clip(cb_ref - cbs, 0, 0.06)
    # red patches also read a little dark
    exY = cv2.resize(ex * mm, (0, 0), fx=Y.shape[1] / Cr.shape[1], fy=Y.shape[0] / Cr.shape[0],
                     interpolation=cv2.INTER_LINEAR) if Cr.shape != Y.shape else ex * mm
    yy0, yx0 = int(c0 * Y.shape[0] / Cr.shape[0]), int(d0 * Y.shape[1] / Cr.shape[1])
    Yo = Y.astype(np.float32, copy=True)
    hh, ww = min(exY.shape[0], Y.shape[0] - yy0), min(exY.shape[1], Y.shape[1] - yx0)
    Yo[yy0:yy0 + hh, yx0:yx0 + ww] += k * 0.25 * exY[:hh, :ww]
    return Yo, Cbo, Cro


def _target_chroma(rgb, hue_deg: float = 0.0):
    r, g, b = rgb
    y = _K[0] * r + _K[1] * g + _K[2] * b
    cb, cr = (b - y) / 1.8556, (r - y) / 1.5748
    a = np.radians(hue_deg)
    return cb * np.cos(a) - cr * np.sin(a), cb * np.sin(a) + cr * np.cos(a)


def _grain(shape, size_px: float, seed: int) -> np.ndarray:
    h, w = shape
    s = max(0.5, size_px)
    gh, gw = max(2, int(np.ceil(h / s)) + 2), max(2, int(np.ceil(w / s)) + 2)
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((gh, gw)).astype(np.float32)
    if s > 1.01:
        n = cv2.resize(n, (int(round(gw * s)), int(round(gh * s))), interpolation=cv2.INTER_CUBIC)[:h, :w]
        n /= max(float(n.std()), 1e-6)
    else:
        n = n[:h, :w]
    if n.shape != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    return n


def process_ycc(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, rf: RetouchFrame, y_per_src: float,
                linear_light: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y 0..1, Cb/Cr centred on 0 (any resolution each). y_per_src: Y-plane px per source px."""
    hY, wY = Y.shape
    hC, wC = Cb.shape
    cs = wC / wY                                   # chroma px per luma px
    Y = Y.astype(np.float32, copy=True)
    Cb = Cb.astype(np.float32, copy=True)
    Cr = Cr.astype(np.float32, copy=True)

    def M(m, shape):
        return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) if m.shape != shape else m

    for job in rf.jobs:
        v = job.v
        wf = job.wf_src * y_per_src if job.wf_src > 0 else 0.11 * max(job.scale, 200.0) * y_per_src
        # 0. blemishes (first, so smoothing never smears a spot around)
        if "skinx" in job.masks and v.get("blemish", 0) > 1e-4:
            k = float(np.clip(v["blemish"], 0, 1))
            size = (0.012 + 0.05 * job.blemish_size) * wf
            Y, Cb, Cr = _blemish(Y, Cb, Cr, M(job.masks["skinx"], (hY, wY)), M(job.masks["skinx"], (hC, wC)),
                                 k, max(2.0, size), cs)
        # 1. skin smoothing
        if "skin" in job.masks and v["skin"] > 1e-4 and job.skin is not None:
            mY = M(job.masks["skin"], (hY, wY))
            mC = M(job.masks["skin"], (hC, wC))
            r = SK.radius_px(job.skin, y_per_src)
            Y = SK.smooth_channel(Y, mY, job.skin.strength, job.keep, r, relative=True)
            rc = max(1, int(round(r * cs)))
            Cb = SK.smooth_channel(Cb, mC, job.skin.strength * SKIN_CHROMA, 0.0, rc, eps=0.0004)
            Cr = SK.smooth_channel(Cr, mC, job.skin.strength * SKIN_CHROMA, 0.0, rc, eps=0.0004)
        # 2. wrinkles
        ex = job.masks.get("excl")
        exY = M(ex, (hY, wY)) if ex is not None else None
        exC = M(ex, (hC, wC)) if ex is not None else None
        for key, name, sf, ss, fw_ in (("nasolabial", "naso", 0.005, 0.05, 0.022),
                                       ("mouth_lines", "mouthl", 0.0045, 0.042, 0.016)):
            if name in job.masks and v[key] > 1e-4:
                k = float(np.clip(v[key], 0, 1))
                band = M(job.masks[name], (hY, wY))
                # the fold where it really is in the (wide) search band, plus a little of the band itself
                found = _fold_find(Y, band, wf, max(1.5, fw_ * wf))
                mY = np.clip(band * (0.25 + 0.95 * found), 0, 1)
                Y = _fold_fill(Y, mY, exY, 0.95 * k, max(0.8, sf * wf), max(2.0, ss * wf), True)
                mC = M(mY, (hC, wC))
                Cb = _fold_fill(Cb, mC, exC, 0.7 * k, max(0.8, sf * wf * cs), max(1.5, ss * wf * cs), False)
                Cr = _fold_fill(Cr, mC, exC, 0.7 * k, max(0.8, sf * wf * cs), max(1.5, ss * wf * cs), False)
        # 2b. dark circles: lift the under-eye shading to the cheek's level, cool/purple cast to skin colour
        if "dark" in job.masks and v.get("dark_circle", 0) > 1e-4:
            k = float(np.clip(v["dark_circle"], 0, 1))
            mY = M(job.masks["dark"], (hY, wY))
            Y = _fold_fill(Y, mY, exY, 0.95 * k, max(0.8, 0.008 * wf), max(3.0, 0.07 * wf), True)
            mC = M(job.masks["dark"], (hC, wC))
            Cb = _fold_fill(Cb, mC, exC, 0.8 * k, max(0.8, 0.008 * wf * cs), max(2.0, 0.07 * wf * cs), False)
            Cr = _fold_fill(Cr, mC, exC, 0.8 * k, max(0.8, 0.008 * wf * cs), max(2.0, 0.07 * wf * cs), False)
        # 2c. shine
        if "skinx" in job.masks and v.get("shine", 0) > 1e-4:
            k = float(np.clip(v["shine"], 0, 1))
            hz = job.masks.get("hot")
            Y, Cb, Cr = _shine(Y, Cb, Cr, M(job.masks["skinx"], (hY, wY)), M(job.masks["skinx"], (hC, wC)), k, wf, cs,
                               M(hz, (hY, wY)) if hz is not None else None)
        # 3. knee darkness
        if "knee" in job.masks and v["knee_dark"] > 1e-4:
            k = float(np.clip(v["knee_dark"], 0, 1))
            mY = M(job.masks["knee"], (hY, wY))
            sig = max(2.0, 0.035 * max(job.scale, 200.0) * y_per_src)
            Y = _lift(Y, mY, k, sig, keep=0.6)
            mC = M(job.masks["knee"], (hC, wC))
            Cb = _lift(Cb, mC, k, max(1.0, sig * cs), keep=0.0)
            Cr = _lift(Cr, mC, k, max(1.0, sig * cs), keep=0.0)
        # 3b. neck lines
        if "neck" in job.masks and v.get("neck_lines", 0) > 1e-4:
            k = float(np.clip(v["neck_lines"], 0, 1))
            Y, Cb, Cr = _crease_lift(Y, Cb, Cr, M(job.masks["neck"], (hY, wY)), k, max(1.5, 0.018 * wf))
        # 3b'. goose bumps on the neck: softened, never wiped
        if "neck" in job.masks and v.get("neck_bumps", 0) > 1e-4:
            k = float(np.clip(v["neck_bumps"], 0, 1))
            Y, Cb, Cr = _bump_soften(Y, Cb, Cr, M(job.masks["neck"], (hY, wY)), M(job.masks["neck"], (hC, wC)),
                                     k, wf, cs)
        # 3c. redness
        if "tone" in job.masks and v.get("redness", 0) > 1e-4:
            k = float(np.clip(v["redness"], 0, 1))
            Y, Cb, Cr = _redness(Cb, Cr, Y, M(job.masks["tone"], (hC, wC)), None, k, wf, cs)
        # 4. skin tone
        if "tone" in job.masks and any(abs(v[k]) > 1e-4 for k in ("skin_bright", "skin_warm", "skin_sat")):
            mY = M(job.masks["tone"], (hY, wY))
            mC = M(job.masks["tone"], (hC, wC))
            b = v["skin_bright"]
            if abs(b) > 1e-4:
                if b > 0:
                    Y = Y + mY * b * 0.6 * Y * (1 - Y) * 2.2 + mY * b * 0.08 * (1 - Y)
                else:
                    Y = Y * (1 + mY * b * 0.55)
            sat = v["skin_sat"]
            if abs(sat) > 1e-4:
                g = 1 + mC * sat * 1.2
                Cb = Cb * g
                Cr = Cr * g
            warm = v["skin_warm"]
            if abs(warm) > 1e-4:
                Cr = Cr + mC * warm * 0.035
                Cb = Cb - mC * warm * 0.045
        # 5. lips
        if "lips" in job.masks:
            mY = M(job.masks["lips"], (hY, wY))
            mC = M(job.masks["lips"], (hC, wC))
            hue = v["lip_hue"] * 100.0                       # degrees
            if abs(hue) > 1e-3:
                a = np.radians(hue) * mC
                ca, sa = np.cos(a), np.sin(a)
                Cb, Cr = Cb * ca - Cr * sa, Cb * sa + Cr * ca
            if abs(v["lip_sat"]) > 1e-4:
                g = 1 + mC * v["lip_sat"] * 1.0
                Cb = Cb * g
                Cr = Cr * g
            if abs(v["lip_bright"]) > 1e-4:
                bb = v["lip_bright"]
                Y = Y + mY * bb * (0.5 * Y * (1 - Y) * 2.0 if bb > 0 else 0.6 * Y)
        # 6. teeth
        if "teeth" in job.masks and v["teeth"] > 1e-4:
            k = float(np.clip(v["teeth"], 0, 1))
            mY = M(job.masks["teeth"], (hY, wY))
            mC = M(job.masks["teeth"], (hC, wC))
            Cb = Cb * (1 - 0.85 * k * mC)
            Cr = Cr * (1 - 0.7 * k * mC)
            Y = Y + mY * k * 0.14 * (1 - Y)
        # 7. eye whites
        if "eyew" in job.masks and v["eye_white"] > 1e-4:
            k = float(np.clip(v["eye_white"], 0, 1))
            mY = M(job.masks["eyew"], (hY, wY))
            mC = M(job.masks["eyew"], (hC, wC))
            # thin red veins: even out locally first
            rr = max(1, int(round(0.012 * wf)))
            Y = SK.smooth_channel(Y, mY, 0.8 * k, 0.25, rr, eps=0.002)
            Cr = SK.smooth_channel(Cr, mC, k, 0.0, max(1, int(round(rr * cs))), eps=0.002)
            Cb = Cb * (1 - 0.8 * k * mC)
            Cr = Cr * (1 - 0.85 * k * mC)
            Y = Y + mY * k * 0.10 * (1 - Y)
        # 7b. makeup: blush, shading, highlight
        if "blush" in job.masks and v.get("blush", 0) > 1e-4:
            k = float(np.clip(v["blush"], 0, 1))
            mY = M(job.masks["blush"], (hY, wY))
            mC = M(job.masks["blush"], (hC, wC))
            # rose pink at 0; − towards rose / mauve, + towards coral / peach
            tcb, tcr = _target_chroma((0.95, 0.42, 0.50), v.get("blush_hue", 0.0) * 100.0)
            Cb = Cb + k * mC * 0.55 * (tcb * 0.6 - Cb)
            Cr = Cr + k * mC * 0.55 * (tcr * 0.6 - Cr)
            Y = Y * (1 - 0.06 * k * mY)
        if "shade" in job.masks and v.get("shading", 0) > 1e-4:
            k = float(np.clip(v["shading"], 0, 1))
            mY = M(job.masks["shade"], (hY, wY))
            mC = M(job.masks["shade"], (hC, wC))
            Y = Y * (1 - 0.16 * k * mY)
            Cr = Cr + 0.010 * k * mC
            Cb = Cb - 0.014 * k * mC
        if "hilite" in job.masks and v.get("highlight", 0) > 1e-4:
            k = float(np.clip(v["highlight"], 0, 1))
            mY = M(job.masks["hilite"], (hY, wY))
            mC = M(job.masks["hilite"], (hC, wC))
            Y = Y + mY * k * (0.16 * (1 - Y) + 0.03)
            Cb = Cb * (1 - 0.15 * k * mC)
            Cr = Cr * (1 - 0.15 * k * mC)
        # 8. grain (last, so smoothing never eats it)
        if "tone" in job.masks and v["grain"] > 1e-4:
            g = float(np.clip(v["grain"], 0, 1))
            mY = M(job.masks["tone"], (hY, wY))
            r = _roi(mY, 2)
            if r is not None:
                y0, y1, x0, x1 = r
                size = job.grain_size * hY / 1080.0
                seed = zlib.crc32(f"{rf.t}:{job.pid}:{hY}".encode())
                n = _grain((y1 - y0, x1 - x0), size, seed)
                yy = Y[y0:y1, x0:x1]
                lw = 0.35 + 2.6 * np.clip(yy, 0, 1) * (1 - np.clip(yy, 0, 1))
                Y[y0:y1, x0:x1] = yy + mY[y0:y1, x0:x1] * n * (0.032 * g) * lw
                cg = float(np.clip(v["grain_color"], 0, 1)) * g
                if cg > 1e-4:
                    mC = M(job.masks["tone"], (hC, wC))
                    rc = _roi(mC, 2)
                    if rc is not None:
                        c0, c1, d0, d1 = rc
                        sizec = size * cs
                        for ci, Cx in enumerate((Cb, Cr)):
                            nc = _grain((c1 - c0, d1 - d0), sizec * 1.3, seed + 7 + ci)
                            Cx[c0:c1, d0:d1] += mC[c0:c1, d0:d1] * nc * (0.016 * cg)
    return Y, Cb, Cr


# --------------------------------------------------------------------------- entry points

_K = (0.2126, 0.7152, 0.0722)


def rgb_to_ycc(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = _K[0] * r + _K[1] * g + _K[2] * b
    return Y, (b - Y) / 1.8556, (r - Y) / 1.5748


def ycc_to_rgb(Y, Cb, Cr):
    r = Y + 1.5748 * Cr
    b = Y + 1.8556 * Cb
    g = (Y - _K[0] * r - _K[2] * b) / _K[1]
    return np.stack([r, g, b], -1)


def apply_rgb(img: np.ndarray, rf: Optional[RetouchFrame], y_per_src: float) -> np.ndarray:
    """Preview: uint8 RGB."""
    if rf is None or rf.empty():
        return img
    f = img.astype(np.float32) / 255.0
    Y, Cb, Cr = rgb_to_ycc(f)
    Y, Cb, Cr = process_ycc(Y, Cb, Cr, rf, y_per_src)
    out = ycc_to_rgb(Y, Cb, Cr)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_planes(planes: List[np.ndarray], subsample, max_value: float, pix_fmt: str,
                 rf: Optional[RetouchFrame], y_per_src: float = 1.0) -> List[np.ndarray]:
    """Render: native planes. YUV planes stay YUV (chroma at its own resolution); float RGB goes through
    Y/Cb/Cr and back."""
    if rf is None or rf.empty():
        return planes
    if pix_fmt.startswith("yuv"):
        Y = planes[0].astype(np.float32) / max_value
        Cb = planes[1].astype(np.float32) / max_value - 0.5
        Cr = planes[2].astype(np.float32) / max_value - 0.5
        Y, Cb, Cr = process_ycc(Y, Cb, Cr, rf, y_per_src)
        out = []
        for pl, val in zip(planes, (Y, Cb + 0.5, Cr + 0.5)):
            if np.issubdtype(pl.dtype, np.integer):
                out.append(np.clip(np.rint(val * max_value), 0, max_value).astype(pl.dtype))
            else:
                out.append(val.astype(pl.dtype))
        return out + list(planes[3:])
    # gbrpf32le: planes are G, B, R
    g, b, r = planes[0], planes[1], planes[2]
    rgb = np.stack([r, g, b], -1).astype(np.float32)
    Y, Cb, Cr = rgb_to_ycc(rgb)
    Y, Cb, Cr = process_ycc(Y, Cb, Cr, rf, y_per_src)
    o = ycc_to_rgb(Y, Cb, Cr)
    return [o[..., 1].astype(planes[0].dtype), o[..., 2].astype(planes[1].dtype), o[..., 0].astype(planes[2].dtype)] \
        + list(planes[3:])
