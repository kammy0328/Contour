"""Retouch engine: skin, blemishes, wrinkles, dark circles, shine, tone, grain, lips, teeth, eye whites, knees.

Everything happens before the warp, in Y/Cb/Cr, with masks built from the tracked face landmarks and
skeleton on the proxy frame. The same code serves the preview (8-bit RGB proxy) and the render (native
planes: 4:2:2 / 4:4:4 YUV, or float RGB), so what you see is what renders.
"""
from __future__ import annotations

import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import bodyskin as BS
from . import bodyspots as BP
from . import bodytone as BTN
from . import detect as DT
from . import skin as SK
from ..core.params import RETOUCH_KEYS
from ..track import skeleton as S


# Where the time of preparing and retouching goes, for a measurement (tools, the PC test jobs): TIMING = {} switches
# it on, and every stage adds its seconds and its count under its name. None (always, in the app): nothing is kept.
TIMING: Optional[dict] = None


_TM_OPEN = threading.local()


class _tm:
    """with _tm("stage"): ...  adds the block's seconds to TIMING["stage"] = [seconds, calls, its own seconds] when
    TIMING is on. Its own seconds: without the stages measured inside it on the same thread (a frame's masks made
    while a neighbour's are steadied count for the frame's masks, not for the steadying)."""
    __slots__ = ("name", "t0", "inner")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if TIMING is not None:
            import time
            self.inner = 0.0
            st_ = _TM_OPEN.__dict__.setdefault("stack", [])
            st_.append(self)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        d = TIMING
        if d is not None and getattr(self, "t0", None) is not None:
            import time
            dt = time.perf_counter() - self.t0
            st_ = _TM_OPEN.__dict__.get("stack") or []
            if st_ and st_[-1] is self:
                st_.pop()
                if st_:
                    st_[-1].inner += dt
            e = d.setdefault(self.name, [0.0, 0, 0.0])
            e[0] += dt
            e[1] += 1
            if len(e) > 2:
                e[2] += dt - self.inner
        return False


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
    wf_arm: float = 0.0                    # the armpit stage's own size reference (a face width, source px: ARMPIT_FACE_PER_RADIUS times its zone's radius) for a person whose face is not in the picture
    blemish_size: float = 0.5              # 0..1 largest spot taken by the blemish removal
    sharp_q: float = 1.0                   # this frame's skin sharpness over that of the frames round it
    # where the skin's flaws are (detect.py), found once per frame and held steady through time (1.19):
    # {"box": (x0, y0, x1, y1) in proxy px, "maps": {kind: uint8 (h, w)}, "thumb": uint8 luma (for the alignment check)}
    det: Optional[dict] = None
    # body skin (1.23, retouch.bodyskin): {"ref": the face width a body of this size has (source px), "model": the
    # body skin's colour model of this frame, "parsed": the body parser was used, "box": the person's piece,
    # "mask": where the body skin effects go (bodyskin.pack_mask: the piece of the analysis picture that is not empty)}
    body: Optional[dict] = None
    body_keep: float = 0.6                 # 결 보존: share of the fine texture the body smoothing gives back
    body_size: float = 0.4                 # 0..1 largest spot taken by the body blemish removal
    bdet: Optional[dict] = None            # the body's spot candidates of this frame (retouch.bodyspots)
    bmarks: Optional[list] = None          # the operator's body spot marks as they lie on this frame
    tone: Optional[dict] = None            # where the knees and the belly are (retouch.bodytone.person_geometry), source px
    arm_axes: Optional[list] = None        # the armpit stage's hanging arms (ARMPIT_LINE_HOLD): [(the upper arm's angle from the vertical in degrees, weight 0..1, zone centre x, y, zone radius)] in proxy px


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
            if names is None or "body" in names:
                b = self.body_area(j)
                if b is not None:
                    out = b if out is None else np.maximum(out, b)
        return out

    def body_area(self, j) -> Optional[np.ndarray]:
        """A job's body skin area at the analysis picture's size (float32), or None."""
        pm = (j.body or {}).get("mask") if getattr(j, "body", None) else None
        return BS.unpack_mask(pm) if pm is not None else None


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
                  wide: float = 1.0, paint: Optional[np.ndarray] = None) -> np.ndarray:
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
    out = np.clip(f * 1.3, 0, 1) * soft * (1 - lipf) * (1 - cz) * (1 - ALA_KEEP * np.clip(ala * 1.2, 0, 1))
    if paint is not None:
        # face paint, drawn lines and hair strands across the cheek are dark lines too: never "folds" to fill.
        # But a deep smile line is a thin dark line too, and lies along the band: those are the fold itself
        pk = _paint_off_fold((paint > 0.2).astype(np.uint8), f, Wf)
        if pk is not None:
            k = max(3, int(0.012 * Wf) | 1)
            pk = cv2.dilate(pk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            out = out * (1 - np.clip(cv2.GaussianBlur(pk.astype(np.float32), (0, 0), max(0.8, 0.006 * Wf)) * 1.3, 0, 1))
    return out


def _paint_off_fold(pm: np.ndarray, band: np.ndarray, Wf: float) -> Optional[np.ndarray]:
    """The painted lines that are not the fold: inside the band, a line running along it (within 25 degrees of
    the band's own direction there) and long enough is the crease itself, not paint or a hair strand."""
    sel = pm & (band > 0.05).astype(np.uint8)
    if not sel.any():
        return None
    n, lab, stt, cen = cv2.connectedComponentsWithStats(sel, 8)
    if n <= 1:
        return None
    # the band's direction: across it the band's edge gradient, along it the curve
    bb = cv2.GaussianBlur(band.astype(np.float32), (0, 0), max(1.0, 0.03 * Wf))
    gx = cv2.Sobel(bb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bb, cv2.CV_32F, 0, 1, ksize=3)
    sw = max(1.0, 0.05 * Wf)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sw)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sw)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sw)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 3:
            continue
        ln = float(np.hypot(stt[i, 2], stt[i, 3]))
        cx, cy = int(round(cen[i, 0])), int(round(cen[i, 1]))
        # the component's own direction
        P = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
        ev, evec = np.linalg.eigh(P @ P.T)
        d_line = evec[:, 1]
        # the band's: perpendicular to its gradient
        a, b_, c_ = float(jxx[cy, cx]), float(jxy[cy, cx]), float(jyy[cy, cx])
        th = 0.5 * np.arctan2(2 * b_, a - c_)              # gradient direction
        d_band = np.array([-np.sin(th), np.cos(th)])
        along = abs(float(d_line @ d_band)) > np.cos(np.radians(25))
        if not (along and ln > 0.05 * Wf):
            keep[i] = True                                  # paint (or a strand): not to be filled
    out = keep[lab].astype(np.uint8) | (pm & (band <= 0.05).astype(np.uint8))
    return out if out.any() else None


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


DARK_INNER_TAPER = 0.35   # the dark circle fades in over this share of the eye's width from the inner corner
DARK_VALLEY = 0.06        # the under-eye is lifted by at most its own depth against the skin within this share of the face
DARK_CAP = 0.07           # and never by more than this (luma 0..1)


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
    if DARK_INNER_TAPER > 0:
        # cut 009 (user 16:46): at the inner corner the crescent ran into the deep shadow beside the nose bridge and lifted
        # it to the cheek's level, a pale patch by the eye. The crescent now fades out towards the inner corner.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mid = (L[39] + L[42]) / 2
        side = (xx - mid[0]) * ex[0] + (yy - mid[1]) * ex[1]          # < 0: the first eye (36..41), > 0: the second
        tap = np.ones((h, w), np.float32)
        for inner, outer, sel in ((L[39], L[36], side < 0), (L[42], L[45], side >= 0)):
            o = outer - inner
            ew = max(1e-3, float(np.hypot(*o)))
            o = o / ew
            along = (xx - inner[0]) * o[0] + (yy - inner[1]) * o[1]
            ramp = np.clip(along / (DARK_INNER_TAPER * ew), 0, 1)
            ramp = ramp * ramp * (3 - 2 * ramp)
            tap = np.where(sel, ramp, tap)
        f = f * tap
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


def face_owned(st, t: int, p: int, lm: np.ndarray, alpha: np.ndarray, shape, sx: float, sy: float) -> np.ndarray:
    """The person matte, with this person's clearly visible face always theirs.

    In a tight group the segmentation sometimes hands a whole face to the person behind (a track that slid onto
    the head in front) or splits it into stripes between two people; the retouch of that face then dropped out
    for that frame. A face whose own landmarks are sure is in front of whatever the matte says, except where
    another person's visible face lies over it."""
    poly = SK.face_polygon(lm)
    if poly is None or float(st.face_vis[p, t]) < 0.8:
        return alpha
    # the face detector (independent of the landmark models) must see this face here too: landmarks that slid
    # onto a neighbour's head are not a face to hand pixels to
    yn = getattr(st, "raw", {}).get("face_yunet")
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    if yn is None or not np.nan_to_num(yn[p, t, 0, 2]) > 0:
        return alpha
    if float(np.hypot(*(yn[p, t, 0, :2] * [sx, sy] - lm[30, :2]))) > 0.3 * Wf:
        return alpha
    h, w = shape
    face = np.zeros(shape, np.uint8)
    cv2.fillPoly(face, [np.round(poly).astype(np.int32)], 255)
    inside = face > 0
    if not inside.any() or float(alpha[inside].mean()) > 0.9:
        return alpha                                  # the matte already has it
    for q in np.nonzero(st.present[:, t])[0]:
        if q == p or st.face_vis[q, t] < 0.5 or not np.isfinite(st.face[q, t, :, :2]).all():
            continue
        oq = st.face[q, t].copy()
        oq[:, 0] *= sx
        oq[:, 1] *= sy
        op = SK.face_polygon(oq)
        if op is not None:
            cv2.fillPoly(face, [np.round(op).astype(np.int32)], 0)
    soft = cv2.GaussianBlur(face.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.02 * Wf))
    return np.maximum(alpha, soft).astype(np.float32)


COVER_SEGS = ("upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "thigh_l", "thigh_r", "calf_l", "calf_r")


LIMB_FALSE = 0.65     # share of the parser's face skin a limb may hide before it is taken for a wrong limb
HAND_RAW_MIN = 2.2    # confidence of the whole-body model's hand points to count as a hand in front of a face (1.5 took
                      # hands that are not in the picture for real: in the close-up of cut 069 both "hands" sat on the
                      # lower face (1.4 and 1.7), and mouth, chin and jaw lost their retouch; real ones are 2.6 and up)


def limb_cover(st, t: int, lm: np.ndarray, shape, sx: float, sy: float) -> Optional[np.ndarray]:
    """0..1 over the face and neck where an arm, a leg or a hand (this person's or anyone's) passes in front.

    A leg in a high kick, a forearm across the face, a hand at the cheek: skin coloured, so the colour model
    takes it for skin, and a half-covered face still has its landmarks, so the face outline holds the retouch
    there too. The limb's own joints say where it is; a limb that crosses the middle of the face is in front of
    it (one behind the head is hidden by it and has no visible joints over the face)."""
    from ..warp.body import SEGMENTS, SEG_INDEX
    poly = SK.face_polygon(lm)
    if poly is None:
        return None
    h, w = shape
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    if Wf < 12:
        return None
    # the face and the neck under it
    face = np.zeros(shape, np.uint8)
    down = np.array([0.0, 0.7 * Wf])
    if np.isfinite(lm[[27, 8], :2]).all():
        d = lm[8, :2] - lm[27, :2]
        n_ = float(np.hypot(*d))
        if n_ > 1e-3:
            down = d / n_ * 0.7 * Wf
    zone = cv2.convexHull(np.round(np.vstack([poly, poly + down])).astype(np.int32))
    cv2.fillConvexPoly(face, zone, 255)
    x0, y0 = zone[:, 0].min(0) - 0.2 * Wf
    x1, y1 = zone[:, 0].max(0) + 0.2 * Wf
    m = np.zeros(shape, np.uint8)
    hit = False
    for q in np.nonzero(st.present[:, t])[0]:
        kp = st.final[q, t]
        for name in COVER_SEGS:
            i = SEG_INDEX[name]
            s = SEGMENTS[i]
            ja, jb = int(s.a), int(s.b)
            if not (np.isfinite(kp[[ja, jb], :2]).all()):
                continue
            conf = float(min(np.nan_to_num(kp[ja, 2]), np.nan_to_num(kp[jb, 2])))
            if conf < 0.15 or not (st.vis[q, t, ja] == 1 and st.vis[q, t, jb] == 1):
                continue
            a = kp[ja, :2] * [sx, sy]
            b = kp[jb, :2] * [sx, sy]
            a = a + (b - a) * float(s.t0)          # upper arm and thigh: not the shoulder or the hip itself
            # only a limb that reaches over the face box matters
            if max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1 or max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1:
                continue
            wd = st.widths[q, t, i]
            r = float(np.nanmax(wd)) * sx if np.isfinite(wd).any() else 0.0
            if not r > 0:
                sc = float(st.scale[q, t]) if np.isfinite(st.scale[q, t]) else 0.0
                r = 0.06 * sc * sx
            r = min(r, 0.6 * Wf)
            if r < 2:
                continue
            _capsule(m, a, b, 0.9 * r)
            hit = True
        # hands: tracked hand points, else the whole-body model's hand points where it is sure of them
        tracked = getattr(st, "has_hands", False) and np.isfinite(st.hand[q, t, :, :2]).any()
        H = st.hand[q, t] if tracked else st.hand_raw[q, t]
        thr = 0.4 if tracked else HAND_RAW_MIN
        for i0, (el, wr) in ((0, (S.L_EL, S.L_WR)), (21, (S.R_EL, S.R_WR))):
            hh = H[i0:i0 + 21]
            okh = np.isfinite(hh[:, :2]).all(1) & (np.nan_to_num(hh[:, 2]) >= thr)
            pts = None
            fore = None
            if np.isfinite(kp[[el, wr], :2]).all():
                fore = float(np.hypot(*(kp[wr, :2] - kp[el, :2]))) * sx
            if okh.sum() >= 10:
                pts = hh[okh, :2] * [sx, sy]
                # a hand is about as long as the forearm's half and starts at the wrist: points scattered over
                # half the picture (the model lost the hand) are no hand
                span = float(np.ptp(pts, 0).max())
                ref_ = fore if fore and fore > 4 else 0.25 * float(np.nan_to_num(st.scale[q, t])) * sx
                if ref_ <= 0 or span > 1.1 * ref_:
                    pts = None
                elif np.isfinite(kp[wr, :2]).all():
                    w_ = kp[wr, :2] * [sx, sy]
                    if float(np.min(np.hypot(*(pts - w_).T))) > 0.5 * ref_:
                        pts = None
            if pts is None and np.isfinite(kp[[el, wr], :2]).all() and min(np.nan_to_num(kp[el, 2]), np.nan_to_num(kp[wr, 2])) >= 0.3 \
                    and st.vis[q, t, wr] == 1:
                # no hand points: the hand as the forearm's continuation
                e_, w_ = kp[el, :2] * [sx, sy], kp[wr, :2] * [sx, sy]
                L = float(np.hypot(*(w_ - e_)))
                if L > 2:
                    pts = np.array([w_, w_ + (w_ - e_) / L * 0.45 * L])
            if pts is None:
                continue
            if pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or pts[:, 1].max() < y0 or pts[:, 1].min() > y1:
                continue
            if len(pts) == 2:
                _capsule(m, pts[0], pts[1], 0.5 * float(np.hypot(*(pts[1] - pts[0]))))
            else:
                hm = np.zeros(shape, np.uint8)
                cv2.fillConvexPoly(hm, cv2.convexHull(np.round(pts).astype(np.int32)), 255, cv2.LINE_AA)
                span = float(np.ptp(pts, 0).max())
                k = max(3, int(0.15 * span) | 1)
                m = np.maximum(m, cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
            hit = True
    if not hit:
        return None
    m = cv2.bitwise_and(m, cv2.dilate(face, np.ones((max(3, int(0.1 * Wf) | 1),) * 2, np.uint8)))
    if not m.any():
        return None
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.03 * Wf))


ARMPIT = 0.0          # 0 = off. Skin retouch on the armpits too (cut 055: raised arms, user 2026-11-14); the render script
ARMPIT_PIDS = None    # switches it on per cut, for every person or only for these ids
ARMPIT_R = 0.30       # (the first version's zone) reach round the shoulder joint, in shoulder widths
ARMPIT_ALONG = 0.06   # (the first version's zone) centre moved this far from the joint along the upper arm
ARMPIT_RAISE = None   # None = the zone of both sides always. A number: only the side whose arm is raised (the elbow's height below the
                      # shoulder over the upper arm's length is under it: 1 = hanging down, 0 = out to the side, < 0 = up)
ARMPIT_FOLD = 0.0     # 0 = off. The armpit's own creases (2611 IWALY c010, 2026-11-22, user: 겨드랑이 주름 제거): the thin fold where the arm
                      # meets the chest is taken out like a neck line, inside the armpit's skin, at this share (0..1)
ARMPIT_W = 0.018      # of the face width: the width of the crease looked for
ARMPIT_SMALL_WF = (90.0, 150.0)   # 2611 IWALY c003 (2026-11-22, own review of a one-frame test): a wide shot's people (faces 67..110 px wide) have no creases to find: the line finder takes the grain, the strands of long
                                  # hair lying over the arm and the garment's seams for creases, and the armpit stage lifted hair strands there by up to 11 % of the brightness. Face width (px) below the first value: this
                                  # stage is off; above the second: in full (None = in full everywhere)
NECK_SMALL_WF = (75.0, 130.0)     # the same for the neck lines and the goose bumps (c003: a strand of hair lying at the neck's edge on a face 67 px wide was lifted; the user's green mark at the jaw, 00:00:39:17)
ARMPIT_HAIR_REACH = 0.012         # 2611 IWALY c029 f96 / c010 f0 (2026-11-22, own review of the armpit stage on large faces): long hair lying over the shoulder is skin to the armpit's colour test (blond hair has
                                  # the skin's hue and its brightness), so the thin-line filler lifted dark strand gaps in it by up to 6 % of the brightness and the fold flattener evened out its light and
                                  # dark clumps by up to 4 %. The armpit's creases are looked for on skin only: what the body parser calls hair (and the garment masks call clothes or hair) is left out of the
                                  # stage's mask, this share of the person's size round it as well (0 = off)
ARMPIT_LINES_MAX = 0.005          # a safety net for what the parser misses (a seam, strands over bare skin, a parser that cannot run): the armpit's own creases are a handful of short lines (c010: 130 px of
                                  # lift at a face 260 px wide), so when more than this share of the face width squared is lifted the finder is looking at texture, not at creases, and the stage
                                  # (the lifts and the fold flattening) is taken back, fully at 2.5 times that. 0 = off
ARMPIT_LINES_MAX_BIG = None        # None = ARMPIT_LINES_MAX at every face size. A number: the share a close-up's armpit gets (the share grows from ARMPIT_LINES_MAX to this one as the face goes from the first to the second width in ARMPIT_LINES_BIG_WF)
ARMPIT_LINES_BIG_WF = (150.0, 250.0)
ARMPIT_THIN = None                # 2611 IWALY c010 f0 (2026-11-22, user: 겨드랑이 주름 제거): the line filler takes a valley for a crease when it is thin against its own surroundings (THIN_LO, THIN_HI = 0.4, 0.7). The armpit's creases run
                                  # on into the shade under the arm, where the valley lies in a broad dark band: it is thin to a lesser degree there and the filler left the lower half of the crease as it was.
                                  # (lo, hi) for the armpit in place of that pair (None = THIN_LO, THIN_HI)
ARMPIT_CLOTH_OUT = True           # 2611 IWALY c010 f0 (2026-11-22, own check of the front-left person: the armpit zone lay on a black dress, and the stage's line finder lifted and flattened the dress's folds by up to 2 %
                                  # of the brightness): the body parser's clothes / accessories class is left out of the armpit stage's mask like its hair class (False: hair only)
ARMPIT_FACE_REF = 0.85            # ... and the cloth gate (NECK_CLOTH_GATE) of the stage goes by at least this share of the person's own face skin's brightness, not by the mask's own pixels alone: a
                                  # zone that is all dark cloth was its own reference and let itself through (0 = the mask's own pixels only)
ARMPIT_BAND = 0.0                 # 2611 IWALY c010 f0 (user: 겨드랑이 주름 제거): after the valley is filled, the crease's flanks (the light lip beside the dark valley) are brought down to the skin's own level
                                  # too: inside the lines the finder found (CREASE_BAND_THR) and CREASE_BAND_HW of the line width on either side, the picture's soft level is replaced by what the skin
                                  # round the band holds (CREASE_BAND_SIG of the line width wide), at this share (the fine grain stays). 0 = off
CREASE_BAND_THR, CREASE_BAND_HW, CREASE_BAND_SIG = 0.3, 1.6, 2.2
ARMPIT_FOLD_INSET = 0.0           # 2611 IWALY c029 f86 (2026-11-22, own review of the second render): the fold flattener of the armpit stage (_fold_flat: an edge-keeping filter 0.08 of the face width wide, whose edge threshold is the neck's)
                                  # took the arm's outline for a soft fold where the skin (0.40 of the brightness) meets the wall behind it (0.52..0.57): the skin along the outline came out up to 5 % lighter and the wall just outside it up to
                                  # 5 % darker, and a dark stripe of the wall ended 26 px short of the arm (the armpit's mask reaches beyond the skin by the width of the person's matte). The flattener now works only inside the body's
                                  # skin area (the one the body smoothing uses), kept this share of the face width away from its edge, fading in over the same distance. 0 = as before
ARMPIT_FLAT = 1.0                 # the share of the flattener's own change that is kept in the armpit stage (1 = in full, 0 = the thin-line lifts and the crease's flanks only)
ARMPIT_FLAT_EPS = 1.0             # 2611 IWALY c029 f117 / f8 (2026-11-22, user's checked stills): the armpit's hollow is a soft fold (a lens-shaped shade, an oval's outline), fainter and wider than a neck's ring: the flattener's edge threshold (FOLD_EPS)
ARMPIT_FLAT_R = 1.0               # times this, and the width of its edge-keeping filter (FOLD_R) times this, in the armpit stage only (1 = as the neck's rings)
ARMPIT_LIFT_INSET = 0.0           # of the face width: the thin-line lifts and the crease's flanks of the armpit stage work only inside the body's skin area, this far from its outline (c029 f86: the end of a dark stripe of the wall
                                  # next to the arm was lifted over 26 px and the arm's own dark rim by up to 4 %). 0 = as before (the stage's mask only)
ARMPIT_EDGE_CLOSE = 0.02          # of the face width: a crack or a hole in the body's skin area narrower than twice this (a crease, a strand over the skin) is closed before the outline is looked for
ARMPIT_CORE = None                # (lo, hi) or None: the armpit's mask is cut at its soft margin, mask * smoothstep(lo, hi, mask): the margin of the person's matte is where the wall's colour passes the stage's colour test at a third of
                                  # the weight (c029 f86..f90: the wall beside the arm lifted by 2..4 %, a stripe of it bitten), and a weight below the lower edge is no armpit. Unlike the body-skin inset it does not
                                  # keep away from shadow (a crease between the arm and the chest lies at the edge of the body skin's colour model). None = the mask as it is
ARMPIT_LIFT_GAIN = 1.0            # 2611 IWALY c010 f0 (2026-11-22, user: 겨드랑이 주름 제거; own check of the one-frame test: about a quarter of the crease's contrast was still there). The line filler
                                  # fills a thin valley from a copy of the picture blurred over a pixel or two, which is shallower than the valley itself, and the score at the valley's flanks is under 1. This
                                  # many times the fill, held to the valley's own depth (it never fills more than the skin beside it: no light line is made). 1 = as the neck lines
# 2611 IWALY c029 / c010 (2026-11-22, user, on the stills he checked): the armpit stage is for the OPEN armpits only (the arm held away from the body, the fold where it meets the chest), nowhere else:
# at c029 f86 / f96 (the arm hanging along the body, its edge against the wall) there is nothing to retouch, and at c010 f23 the arm's own line above the fold (green) has to stay while the fold (red) goes.
ARMPIT_OPEN = None                # (lo, hi) degrees or None (= every arm): the upper arm's angle from the torso's axis (shoulder to hip), median over ARMPIT_OPEN_SPAN frames each side. Weight 0 at lo and under, 1 at hi and over.
                                  # A hanging arm is 0..20 deg (c029 f86 / f96: 16..7), the open ones the user checked 33..58 (c010 f23 33, c029 f79 58, f117 56)
ARMPIT_OPEN_SPAN = 2
ARMPIT_RADIAL = None              # (lo, hi) or None: the zone's own weight falls from 1 at lo to 0 at hi times its radius (the fold lies near the zone's centre, 0.3..0.6 of the radius in the stills the user checked; c010 f23's arm line
                                  # he wants kept lies at 0.76). None = the whole disc as before
ARMPIT_NEAR_CLOTH = None          # (near, far) of the face width or None: the fold stage works near the garment's edge (the armhole: the folds the user checked lie 0..0.30 face widths from it, the arm line he wants kept 0.6 and more):
                                  # weight 1 within near of the nearest hair / clothes / garment pixel, 0 beyond far. None = anywhere in the zone
ARMPIT_CLOSE = 0.0                # of the face width: the fold is a dark valley, and the armpit's mask (the skin's brightness split, the clothes gate) leaves it out as if it were cloth, so the lifts never saw it. A hole or a
                                  # bay in the stage's mask narrower than twice this is closed before the lifts and the flattener (0 = as before)
ARMPIT_PIT_W = None               # of the face width or None: a second pass of the thin-line filler at this width (the fold is wider than a crease line: 0.04 of the face = 13 px at 330)
ARMPIT_FINE = True                # False: the armpit's thin-line filler does without the long-line (strand) finder (the one that also finds hair strands). 2611 IWALY c010 f23 (2026-11-22, user: the arm's own line, green, stays; the fold, red, goes): a
                                  # long soft line along the arm is a long faint dark line to that finder, and the filler took it out with the folds. The valley finder alone looks only at what is short, deep and curved
                                  # across (a crease). True = as the neck lines
ARMPIT_LINE_HOLD = 0.0            # 0 = off. 2611 IWALY c010 f23 (2026-11-22, user: the arm's own line, green, stays; the fold, red, goes): where the upper arm hangs (ARMPIT_LINE_HANG) the armpit stage leaves the arm's
                                  # own line, a long thin dark line lying along the upper arm, as it was. The stage's fillers and its flattener cannot tell it from a crease (both are a thin dark valley); what
                                  # tells it is its direction and its length: a valley within ARMPIT_LINE_TOL degrees of the upper arm's direction that goes on for ARMPIT_LINE_LEN of the face width. Found on the picture
                                  # before the stage, and given back after it: inside the line (and where it goes on faintly, ARMPIT_LINE_EXT) the picture's detail finer than ARMPIT_LINE_SIG line widths is the source's again,
                                  # the soft light and shade under it is the retouched one. This share (0..1) of it is given back
ARMPIT_LINE_HANG = (24.0, 30.0)   # degrees: the upper arm's angle from the picture's vertical (median over ARMPIT_OPEN_SPAN frames each side); the hold is in full at the first value and under, 0 at the second and over.
                                  # An arm held away from the body (c010 f0 33, c029 f79 62, f117 52, f8 60) has creases in its armpit, a hanging one (c010 f23 21) has the arm's line
ARMPIT_LINE_TOL = 20.0            # degrees: a valley whose direction differs from the upper arm's by this much or less is the arm's line, 0 at 15 degrees more
ARMPIT_LINE_LEN = 0.20            # of the face width: the length along the arm over which a line is judged (the mean of the finder's score along it)
ARMPIT_LINE_THR = (0.03, 0.08)    # that mean's (lo, hi): 0 under lo, 1 over hi
ARMPIT_LINE_EXT = 0.25            # of the face width: a held line is followed this far along the arm where it gets too faint for the finder (valley, direction as above)
ARMPIT_LINE_SIG = 3.0             # of the crease width (ARMPIT_W of the face width): the detail finer than this is given back, the broader light and shade stays the stage's
ARMPIT_LINE_GSPAN = 2             # frames each side: the hold's weight (how far the arm hangs, ARMPIT_LINE_HANG) of a frame is the largest of those of the frames t - GSPAN .. t + GSPAN of the same shot, so that a short lift
                                  # of a hanging arm (c010 f26-28: the pose reads 28 degrees for three frames between 21 and 24) does not take the arm's line out for those frames and give it back after them. 0 = the frame's own
ARMPIT_ONE_SHOULDER = False      # a person cut by the frame's edge (one shoulder in the picture): the zone from that shoulder and its elbow, the shoulder width taken as ARMPIT_ONE_SHOULDER_SW times the upper arm, the axis straight down
ARMPIT_ONE_SHOULDER_SW = 1.4
ARMPIT_FACE_PER_RADIUS = 2.07     # the face width a zone of this radius goes with (measured on c029 / c010: 2.0..2.2), for a person whose face is not in the picture (Job.wf_arm)


def _small_face_gate(wf: float, rng) -> float:
    """0..1: how much of a stage that needs a face of some size is worked: 0 below rng[0], 1 above rng[1]."""
    if rng is None:
        return 1.0
    return float(_ss(float(rng[0]), float(rng[1]), np.array(float(wf))))


def _skin_inner(m: np.ndarray, close_px: float, inset_px: float) -> np.ndarray:
    """The body's skin area m (0..1) as a weight that is 0 outside it and keeps inset_px away from its outline (the arm's edge against the wall, cloth or hair), fading in over about the same distance.
    A crack or a hole in the area narrower than 2 * close_px (a crease, a strand lying over the skin) is no outline: it is closed first. A frame's own border is no outline either."""
    r_ = _roi(m, 2)
    if r_ is None:
        return np.zeros(m.shape, np.float32)
    c_ = max(0, int(round(close_px)))
    e_ = max(2, int(round(inset_px)))
    pad = c_ + e_ + 3
    y0, y1, x0, x1 = r_
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(m.shape[0], y1 + pad), min(m.shape[1], x1 + pad)
    core = (m[y0:y1, x0:x1] > 0.5).astype(np.uint8)
    if c_ >= 1:
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c_ + 1, 2 * c_ + 1)))
    core = cv2.erode(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_ + 1, 2 * e_ + 1)))
    soft = np.clip(cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(1.0, 0.5 * e_)) * 1.25, 0, 1)
    out = np.zeros(m.shape, np.float32)
    out[y0:y1, x0:x1] = soft
    return out


def _close_soft(m: np.ndarray, r: int) -> np.ndarray:
    """The mask m (0..1) with its holes and bays narrower than 2 r + 1 px closed (a grey-scale closing: the valleys of a soft mask are filled, nothing grows outward). m where r < 1."""
    if r < 1:
        return m
    rr = _roi(m, r + 3)
    if rr is None:
        return m
    y0, y1, x0, x1 = rr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sub = np.ascontiguousarray(m[y0:y1, x0:x1], dtype=np.float32)
    c = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k)
    out = np.array(m, dtype=np.float32, copy=True)
    out[y0:y1, x0:x1] = np.maximum(sub, c)
    return out


def _armpit_without_hair(ap: np.ndarray, tracker, t: int, p: int, rgb_u8: np.ndarray, shape, wf_p: float = 0.0) -> np.ndarray:
    """The armpit skin ap (0..1) with this person's hair and garments taken out of it (ARMPIT_HAIR_REACH): hair is what the body parser calls hair, and clothes what it calls clothes or accessories
    (ARMPIT_CLOTH_OUT; the same NOT_SKIN thresholds the body skin mask goes by), a garment is what the clothing / hair masks of the analysis call so. A share of the person's size round them goes too (the parser's edge is a few pixels
    off at the picture's own size). ap itself where neither is known.
    wf_p (the face width, proxy px) with ARMPIT_NEAR_CLOTH: the weight also falls with the distance from that hair / clothes / garment (1 within the first share of the face width, 0 beyond the second)."""
    if ap is None:
        return ap
    reach_on = bool(ARMPIT_HAIR_REACH and ARMPIT_HAIR_REACH > 0)
    prox_on = ARMPIT_NEAR_CLOTH is not None and float(wf_p) > 1.0
    if not reach_on and not prox_on:
        return ap
    h, w = int(shape[0]), int(shape[1])
    ns = None
    size = None
    try:
        own, _o = BS.ownership(tracker, t, p, (w, h), others=False)
        if own is not None and float(own.max()) > 0.3:
            size = float(np.sqrt(max(1.0, float((own > 0.3).sum()))))
            pr = BS._parse_cached(tracker, p, t, rgb_u8, own)
            if pr is not None and "hair" in pr:
                no_ = BS._place(pr, "hair", (h, w))
                if ARMPIT_CLOTH_OUT and "cloth" in pr:
                    no_ = np.maximum(no_, BS._place(pr, "cloth", (h, w)))
                ns = BS._ss(BS.NOT_SKIN[0], BS.NOT_SKIN[1], no_)
    except Exception:
        ns = None
    try:
        g = BS.garments(tracker, t, p, (w, h))
        if g is not None:
            g = np.asarray(g, np.float32)
            ns = g if ns is None else np.maximum(ns, g)
    except Exception:
        pass
    if ns is None:
        return ap
    if size is None:
        size = float(np.sqrt(max(1.0, float((ap > 0.3).sum())))) * 3.0
    reach = max(2, int(round(ARMPIT_HAIR_REACH * size))) if reach_on else 0
    pad = 3 * reach + 4
    if prox_on:
        pad = max(pad, int(float(ARMPIT_NEAR_CLOTH[1]) * float(wf_p)) + 8)
    r = _roi(ap, pad)
    if r is None:
        return ap
    y0, y1, x0, x1 = r
    nz = np.ascontiguousarray(ns[y0:y1, x0:x1], dtype=np.float32)
    if reach_on:
        nz = cv2.dilate(nz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
        nz = np.clip(cv2.GaussianBlur(nz, (0, 0), max(1.0, 0.5 * reach)) * 1.25, 0, 1)
    else:
        nz = np.zeros_like(nz)
    keep = 1.0 - nz
    if prox_on:
        nsr = ns[y0:y1, x0:x1] > 0.5
        if bool(nsr.any()):                              # (nothing to measure from where no hair or clothes were found: no distance weight)
            dist = cv2.distanceTransform((~nsr).astype(np.uint8), cv2.DIST_L2, 3)
            keep = keep * (1.0 - _ss(float(ARMPIT_NEAR_CLOTH[0]) * float(wf_p), float(ARMPIT_NEAR_CLOTH[1]) * float(wf_p), dist)).astype(np.float32)
    out = np.zeros_like(ap)
    out[y0:y1, x0:x1] = ap[y0:y1, x0:x1] * keep
    return out


def _lines_budget(dY: np.ndarray, m: np.ndarray, wf: float):
    """(map, log): how much of the armpit stage is kept (1 .. 0, at m's size) after its thin-line filler lifted dY, piece by piece of the mask m (ARMPIT_LINES_MAX): the creases of an armpit
    are a handful of short lines; more lifted pixels in one armpit than that share of the face width squared is the finder reading texture (hair, seams, grain) and not creases.
    Each armpit (each separate piece of the mask) is judged on its own: hair over one shoulder takes nothing from the crease of the other. map is None where every piece is kept
    in full; log: [(lifted px, budget px, kept)] of the pieces that are not."""
    if not ARMPIT_LINES_MAX or ARMPIT_LINES_MAX <= 0 or not wf > 1:
        return None, []
    r = _roi(m, 4)
    if r is None:
        return None, []
    y0, y1, x0, x1 = r
    n, lab = cv2.connectedComponents((m[y0:y1, x0:x1] > 0.05).astype(np.uint8), connectivity=8)
    if n < 2:
        return None, []
    cnt = np.bincount(lab.ravel(), weights=(dY[y0:y1, x0:x1] > 0.01).ravel().astype(np.float64), minlength=n)
    lm_ = float(ARMPIT_LINES_MAX)
    if ARMPIT_LINES_MAX_BIG is not None and float(ARMPIT_LINES_MAX_BIG) > lm_:
        # a close-up's armpit has real, long creases (c029, a face 390 px wide: 1700..2100 px lifted at f85..f90, 0.011..0.014 of the face width squared, the budget of a wide shot's 0.005 took the whole stage back
        # for those frames and gave it again after them): the share grows with the face from the wide shot's to this one
        lm_ = lm_ + (float(ARMPIT_LINES_MAX_BIG) - lm_) * float(_ss(float(ARMPIT_LINES_BIG_WF[0]), float(ARMPIT_LINES_BIG_WF[1]), np.array(float(wf))))
    B = lm_ * float(wf) * float(wf)
    keep = (1.0 - _ss(B, 2.5 * B, cnt.astype(np.float32))).astype(np.float32)
    keep[0] = 1.0
    if float(keep.min()) > 0.999:
        return None, []
    out = np.ones(m.shape, np.float32)
    out[y0:y1, x0:x1] = keep[lab]
    return out, [(int(cnt[i]), B, float(keep[i])) for i in range(1, n) if keep[i] <= 0.999]
# 2611 IWALY c010 f0 (2026-11-22, user): the first version put the zone round the shoulder JOINT, and let it in only for a raised arm; the crease the
# user marked lies 200 px under the joint, where the arm meets the chest (the elbow of that arm was 0.83 of the upper arm's length below the
# shoulder: "hanging"). The armpit is under the joint, on the way down to the hip: ARMPIT_DOWN of the way from the shoulder joint to the hip of its
# side, the zone's radius that share of that length. Whether there is an armpit to see is for the picture to say (armpit_skin: no skin in the zone, no
# armpit), not the pose.
ARMPIT_DOWN = 0.20    # the zone's centre: this share of the way from the shoulder joint to the hip (1.7 shoulder widths where the hip is not known)
ARMPIT_REACH = 0.14   # its radius, of the shoulder-to-hip length


def _armpit_points(kp: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """The joints that are believed (confidence > 0.2) in the proxy's px, the others NaN."""
    P = np.full((kp.shape[0], 2), np.nan)
    ok = np.isfinite(kp[:, :2]).all(1)
    if kp.shape[1] > 2:
        ok &= np.nan_to_num(kp[:, 2]) > 0.2
    P[ok] = kp[ok, :2] * [sx, sy]
    return P


def _armpit_axis(P: np.ndarray, side: int):
    """(shoulder, axis, shoulder width, tip) of one side (0 = left, 1 = right): the shoulder joint; the axis from it to the hip (where the hip is not under it, 1.7 shoulder widths straight down the body); the shoulder width
    the zone goes by (the distance of the two shoulders; with only one of them in the picture ARMPIT_ONE_SHOULDER_SW times its upper arm); the elbow (else the wrist). None where the side has no zone."""
    sh, el, wr, hp_i = ((S.L_SH, S.L_EL, S.L_WR, S.L_HIP), (S.R_SH, S.R_EL, S.R_WR, S.R_HIP))[side]
    ls, rs = P[S.L_SH], P[S.R_SH]
    s_ = P[sh]
    if not np.isfinite(s_).all():
        return None
    tip = P[el] if np.isfinite(P[el]).all() else (P[wr] if np.isfinite(P[wr]).all() else None)
    both = bool(np.isfinite(ls).all() and np.isfinite(rs).all())
    if both:
        sw = float(np.hypot(*(ls - rs)))
    else:
        if not ARMPIT_ONE_SHOULDER or tip is None:
            return None
        sw = float(ARMPIT_ONE_SHOULDER_SW) * float(np.hypot(*(tip - s_)))
    if sw < 8:
        return None
    d = None
    if np.isfinite(P[hp_i]).all():
        d = P[hp_i] - s_
        if float(np.hypot(*d)) < 0.8 * sw:
            d = None                                    # a hip that is not under its shoulder: not believed
    if d is None:
        if both:
            nv = np.array([-(ls - rs)[1], (ls - rs)[0]], np.float64)
            nv = nv / max(float(np.hypot(*nv)), 1e-6)
            if nv[1] < 0:
                nv = -nv
        else:
            nv = np.array([0.0, 1.0])
        d = nv * 1.7 * sw
    return s_, d, sw, tip


def armpit_open_weights(st, p: int, t: int, sx: float, sy: float, rng, span: int = 2):
    """[left, right]: how open each armpit is (0..1, ARMPIT_OPEN): the angle between the upper arm (shoulder to elbow, else to wrist) and the torso's axis (shoulder to hip), the median over the frames
    t - span .. t + span of the same shot, 0 at rng[0] and under, 1 at rng[1] and over. An arm whose angle is not known (no elbow, no shoulder) is 0."""
    N = st.final.shape[1]
    seg = [s for s in range(max(0, t - span), min(N, t + span + 1)) if bool(st.present[p, s]) and (s == t or st.connected(s, t))]
    if t not in seg:
        seg.append(t)
    angs = ([], [])
    for s in seg:
        P = _armpit_points(st.final[p, s], sx, sy)
        for side in (0, 1):
            a_ = _armpit_axis(P, side)
            if a_ is None or a_[3] is None:
                continue
            s_, d, _sw, tip = a_
            dd = tip - s_
            nn, nd = float(np.hypot(*dd)), float(np.hypot(*d))
            if nn < 1e-3 or nd < 1e-3:
                continue
            angs[side].append(float(np.degrees(np.arccos(np.clip(float(dd @ d) / (nn * nd), -1.0, 1.0)))))
    out = []
    for side in (0, 1):
        if not angs[side]:
            out.append(0.0)
        else:
            out.append(float(_ss(float(rng[0]), float(rng[1]), np.array(float(np.median(angs[side]))))))
    return out


def armpit_hold_axes(st, p: int, t: int, sx: float, sy: float, info, wts, span: int = 2):
    """The armpit zones whose arm hangs (ARMPIT_LINE_HANG): [(the upper arm's angle from the picture's vertical in degrees (+ = the elbow on the picture's right), weight 0..1, zone centre x, y, zone radius)] in proxy px,
    one per zone of info (side, centre, radius, shoulder width) that is open (wts, ARMPIT_OPEN) and whose arm hangs. The arm's direction is the median over the frames t - span .. t + span of the same shot.
    The weight is the largest of the frames t - ARMPIT_LINE_GSPAN .. t + ARMPIT_LINE_GSPAN (each with the median of its own span), see ARMPIT_LINE_GSPAN."""
    N = st.final.shape[1]
    gsp = int(max(0, int(ARMPIT_LINE_GSPAN)))
    frames = [s for s in range(max(0, t - span - gsp), min(N, t + span + gsp + 1)) if bool(st.present[p, s]) and (s == t or st.connected(s, t))]
    if t not in frames:
        frames.append(t)
        frames.sort()
    dirs_at = {}
    for s in frames:
        P = _armpit_points(st.final[p, s], sx, sy)
        one = [None, None]
        for side in (0, 1):
            a_ = _armpit_axis(P, side)
            if a_ is None or a_[3] is None:
                continue
            dd = a_[3] - a_[0]
            nn = float(np.hypot(*dd))
            if not np.isfinite(nn) or nn < 1e-3:
                continue
            one[side] = dd / nn
        dirs_at[s] = one

    def med_dir(c: int, side: int):
        ds = [dirs_at[s][side] for s in frames if abs(s - c) <= span and dirs_at[s][side] is not None]
        if not ds:
            return None
        v_ = np.median(np.array(ds), axis=0)
        nv_ = float(np.hypot(*v_))
        return None if (not np.isfinite(nv_) or nv_ < 1e-3) else v_ / nv_

    def hang(v_) -> float:
        return 1.0 - float(_ss(float(ARMPIT_LINE_HANG[0]), float(ARMPIT_LINE_HANG[1]), np.array(float(np.degrees(np.arccos(np.clip(float(v_[1]), -1.0, 1.0)))))))

    out = []
    for (side, c, rad, _sw) in info:
        if wts is not None and (side >= len(wts) or wts[side] <= 1e-3):
            continue
        v = med_dir(t, side)
        if v is None:
            continue
        g = hang(v)
        for c2 in frames:
            if c2 == t or abs(c2 - t) > gsp:
                continue
            v2 = med_dir(c2, side)
            if v2 is not None:
                g = max(g, hang(v2))
        if g <= 1e-3:
            continue
        out.append((float(np.degrees(np.arctan2(float(v[0]), float(v[1])))), g, float(c[0]), float(c[1]), float(rad)))
    return out


ARMPIT_WEDGE = False        # 2611 IWALY c029 f4/f5/f6/f69/f136/f159 (2026-11-22, user's deck: the arm, a palm, a shoulder, the back and an arm's outline retouched "for no reason", the
                            # arm blurred, its line gone): the zone was a disc round a point under the shoulder joint, and took the shoulder's top, the outer arm, the back and a hand
                            # held in front of the body with it. True = only the wedge between the upper arm and the torso: on the torso's side of the upper arm's axis (shoulder to
                            # elbow), on the arm's side of the torso's axis (shoulder to hip) and below the shoulder joint, each with a soft margin of ARMPIT_WEDGE_M shoulder widths
ARMPIT_WEDGE_M = (0.04, 0.10)   # (outside .. inside) of the shoulder width: the margin over which each side of the wedge fades from 0 to 1
ARMPIT_WEDGE_T = (0.45, 0.25)   # (0 at .. 1 at) shoulder widths INTO the torso past its axis (the shoulder joint lies inside the body: the fold shows on the torso's side of that axis, c010 f23 0.15 sw)


def _armpit_wedge(shape, s_, d, tip, sw):
    """0..1 at shape: the wedge between the upper arm (s_ -> tip) and the torso (s_ -> s_ + d), soft by ARMPIT_WEDGE_M * sw."""
    h, w = shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    ux, uy = gx - float(s_[0]), gy - float(s_[1])
    a = np.asarray(tip, np.float64) - np.asarray(s_, np.float64)
    na = float(np.hypot(*a))
    nd = float(np.hypot(*d))
    if na < 1e-3 or nd < 1e-3:
        return np.ones(shape[:2], np.float32)
    a = a / na
    dn = np.asarray(d, np.float64) / nd
    lo, hi = float(ARMPIT_WEDGE_M[0]) * sw, float(ARMPIT_WEDGE_M[1]) * sw
    # signed distance from the arm's axis, positive on the torso's side
    sa = float(np.sign(a[0] * dn[1] - a[1] * dn[0]) or 1.0)
    da = sa * (a[0] * uy - a[1] * ux)
    # signed distance from the torso's axis, positive on the arm's side
    st_ = float(np.sign(dn[0] * a[1] - dn[1] * a[0]) or 1.0)
    dt = st_ * (dn[0] * uy - dn[1] * ux)
    # below the shoulder joint along the torso
    dd = ux * dn[0] + uy * dn[1]
    g = _ss(-lo, hi, da) * _ss(-float(ARMPIT_WEDGE_T[0]) * sw, -float(ARMPIT_WEDGE_T[1]) * sw, dt) * _ss(-lo, lo, dd)
    return g.astype(np.float32)


def armpit_region(kp: np.ndarray, shape, sx: float, sy: float, parts: bool = False, info: Optional[list] = None):
    """The skin under the arms (0..1): a zone under each shoulder joint, on the way to the hip. parts: the two sides as a list (a side without a zone is zeros).
    info: a list that gets (side, centre, radius, shoulder width) of every zone made (proxy px)."""
    if kp is None:
        return None
    P = _armpit_points(kp, sx, sy)
    out = []
    n_ok = 0
    for side in (0, 1):
        a_ = _armpit_axis(P, side)
        if a_ is None:
            if parts:
                out.append(np.zeros(shape, np.float32))
            continue
        s_, d, sw, tip = a_
        if ARMPIT_RAISE is not None:
            el = S.L_EL if side == 0 else S.R_EL
            if tip is None:
                if parts:
                    out.append(np.zeros(shape, np.float32))
                continue
            dd = tip - s_
            nn = float(np.hypot(*dd))
            if nn < 1e-3 or float(dd[1]) / nn >= (ARMPIT_RAISE if np.isfinite(P[el]).all() else ARMPIT_RAISE - 0.25):
                if parts:
                    out.append(np.zeros(shape, np.float32))
                continue
        n = float(np.hypot(*d))
        c = s_ + ARMPIT_DOWN * d
        m = np.zeros(shape, np.uint8)
        rad = max(2, int(round(ARMPIT_REACH * n)))
        if ARMPIT_RADIAL is not None:
            # the weight falls from 1 at lo to 0 at hi times the radius: a disc of the middle radius, feathered over the width between them
            lo_, hi_ = float(ARMPIT_RADIAL[0]), float(ARMPIT_RADIAL[1])
            cv2.circle(m, (int(round(c[0])), int(round(c[1]))), max(2, int(round(0.5 * (lo_ + hi_) * rad))), 255, -1, cv2.LINE_AA)
            sig = max(1.0, (hi_ - lo_) * rad / 3.0)
        else:
            cv2.circle(m, (int(round(c[0])), int(round(c[1]))), rad, 255, -1, cv2.LINE_AA)
            sig = max(0.06 * sw, 0.10 * rad)
        zf = _feather(m, sig)
        if ARMPIT_WEDGE and tip is not None:
            try:
                zf = zf * _armpit_wedge(shape, s_, d, tip, sw)
            except Exception:
                pass
        out.append(zf)
        n_ok += 1
        if info is not None:
            info.append((side, (float(c[0]), float(c[1])), float(rad), float(sw)))
    if not n_ok or not out:
        return [] if parts else None
    return out if parts else np.maximum.reduce(out)


def armpit_skin(rgb_u8: np.ndarray, zone: np.ndarray, alpha: Optional[np.ndarray], prob: np.ndarray) -> Optional[np.ndarray]:
    """The skin inside one armpit zone (0..1). The face's colour model loses it (it lies in shadow, cut 055), so it
    is found from the zone itself: the person's own pixels there fall into a dark group (clothes, hair) and a
    lighter one (skin); the lighter one, with one colour, is the skin. Meant for dark clothes (it is switched on
    per cut and person): light clothes in the zone would be taken along."""
    a = alpha if alpha is not None else np.ones(zone.shape, np.float32)
    f = rgb_u8.astype(np.float32)
    Y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    Cb = 0.564 * (f[..., 2] - Y)
    Cr = 0.713 * (f[..., 0] - Y)
    sel = (zone > 0.3) & (a > 0.6)
    if int(sel.sum()) < 120:
        return None
    lg = np.log(np.maximum(Y, 1.0))
    v = lg[sel]
    # two groups by brightness (Otsu on the log): black cloth and hair against skin, lit or in shadow
    hist, edges = np.histogram(v, bins=64, range=(0.0, 5.55))
    pr = hist.astype(np.float64) / max(1, hist.sum())
    mid = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(pr); w1 = 1.0 - w0
    m0 = np.cumsum(pr * mid)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m0) ** 2 / (w0 * w1)
    sb[~np.isfinite(sb)] = 0
    i = int(np.argmax(sb))
    thr = float(edges[i + 1])
    lo, hi = v[v <= thr], v[v > thr]
    if len(hi) < 60:
        return None
    if len(lo) >= 0.03 * len(v) and float(hi.mean() - lo.mean()) > 0.6:
        hsel = sel & (lg > thr)
        sy_ = np.clip((lg - (thr - 0.05)) / 0.35, 0.0, 1.0)
    else:
        # one group only: skin if the face's colour model takes it for skin, else nothing can be said
        if float(prob[sel].mean()) < 0.3:
            return None
        hsel = sel
        sy_ = np.ones(zone.shape, np.float32)
    # one colour, measured against the brightness (skin in shadow keeps its hue, not its chroma)
    Yn = np.maximum(Y, 8.0)
    cbn, crn = Cb / Yn, Cr / Yn
    b0, r0 = float(np.median(cbn[hsel])), float(np.median(crn[hsel]))
    dc = np.sqrt((cbn - b0) ** 2 + (crn - r0) ** 2)
    sc = np.clip(1.0 - (dc - 0.07) / 0.09, 0.0, 1.0)
    m = sc * sy_ * zone * a
    sg = max(0.8, 0.002 * zone.shape[0])
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sg), 0, 1)


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


NECK_SHADE = True          # the parser's neck counts where the colour model misses a broad stretch of it (shade)
NECK_DARK = (0.40, 0.58)    # of the face's own skin brightness: the neck mask fades out between these (hair, a black top)
NECK_SHADE_MIN = 0.045     # of the face width: narrower stretches the colour model leaves out stay out (hair, a chain)


NECK_PROTECT = 1.0         # how far the hair strands and paint the skin mask leaves out are kept out of the neck's mask (0 = as before:
                           # the mask blurs the skin probability, a strand a few pixels wide is blurred away, and the neck-line filler
                           # lifted hairs lying over the neck as if they were creases: cut IWALY_c050)


def neck_mask(lm: np.ndarray, kp: np.ndarray, widths, shape, sx: float, sy: float, prob: np.ndarray,
              alpha, fth: float = 1.5, rgb: Optional[np.ndarray] = None, hands: Optional[np.ndarray] = None,
              parse: Optional[dict] = None, protect: Optional[np.ndarray] = None) -> np.ndarray:
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
    pc = np.clip(cv2.GaussianBlur(prob, (0, 0), max(1.0, 0.012 * Wf)) * 1.6, 0, 1)
    if NECK_SHADE and parse is not None and parse.get("trust", 0) > 0:
        # 1.19 (2611 cuts 009, 024): the side of the neck in shade, and the skin under the chin, are not the
        # skin colour model's skin (darker, less colour), so half the neck was no neck and its lines were never
        # looked at. Where the face parser is sure of neck, a broad stretch the colour model leaves out counts;
        # thin things it leaves out (a strand of hair, a chain, a painted line) stay out
        try:
            sure = _ss(0.55, 0.85, cv2.GaussianBlur(parse["neck"].astype(np.float32), (0, 0), max(1.0, 0.015 * Wf))
                       * cv2.GaussianBlur(parse["cover"].astype(np.float32), (0, 0), max(1.0, 0.04 * Wf)))
            hole = (sure * (1.0 - pc)).astype(np.float32)
            kb = max(3, int(NECK_SHADE_MIN * Wf) | 1)
            seb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
            broad = cv2.dilate(cv2.erode(hole, seb), seb)
            pc = np.maximum(pc, float(parse["trust"]) * cv2.GaussianBlur(broad, (0, 0), max(1.0, 0.008 * Wf)))
        except Exception:
            pass
    f = f * pc
    if protect is not None and NECK_PROTECT > 0 and protect.shape == f.shape:
        # what the skin mask cut out of the face (a hair strand lying over the neck, paint) stays out of the neck too: the blur above
        # closes a hole a few pixels wide
        f = f * (1.0 - NECK_PROTECT * np.clip(protect, 0.0, 1.0))
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
            if NECK_DARK is not None:
                # far darker than the face's own skin: hair falling beside or over the neck, a black top under it.
                # A neck in shade is darker than the face too, but not by this much (measured on a log clip seen
                # in Rec.709: shaded necks 0.6 to 0.9 times the face, hair and black cloth under 0.45)
                Ys = cv2.GaussianBlur(Yl, (0, 0), max(1.0, 0.012 * Wf))
                f = f * _ss(NECK_DARK[0] * yref, NECK_DARK[1] * yref, Ys)
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

VIS_PARSE = True      # the face parser's view of the frame can lift the face weight (see _prepare_raw)
VIS_PARSE_SHARE = (0.40, 0.65)   # share of the face outline the parser calls skin: from .. to full weight
FOCUS_ZONE = True     # frames with a doubtful face: skin far from any sharp structure of the face is a defocused cover, not the face
FOCUS_ZONE_THR, FOCUS_ZONE_DIL = 0.40, 0.12
MATTE_PARSE = True    # the parser's sure skin inside the face outline belongs to the person even where the matte misses it


def _focus_zone(rgb_u8: np.ndarray, lm: np.ndarray) -> Optional[np.ndarray]:
    """A head or a hand that passes in front of the face out of focus (cut 086, TC 01:02:20:18) is skin coloured and
    smooth, and the parser and the matte hand it to the face. The face behind it is sharp: eyes, lashes, nose edge,
    lips, hairline. Returns a 0..1 map that is 1 within FOCUS_ZONE_DIL of the face's steep edges and falls to 0
    beyond; None when the face has too little structure to judge."""
    try:
        pts = np.asarray(lm[:, :2], np.float32)
        if not np.isfinite(pts).all():
            return None
        h, w = rgb_u8.shape[:2]
        wf = float(pts[:, 0].max() - pts[:, 0].min())
        if wf < 24:
            return None
        hull = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
        k = int(0.1 * wf) | 1
        hh = cv2.dilate(hull, np.ones((k, k), np.uint8)) > 0
        g = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        gs = cv2.GaussianBlur(g, (0, 0), max(0.7, 0.006 * wf))
        gm = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        if hh.sum() < 50:
            return None
        ref = float(np.percentile(gm[hh], 99.5))
        if ref < 1.0:
            return None
        sharp = (gm > FOCUS_ZONE_THR * ref).astype(np.uint8)
        sharp = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        r = max(2, int(FOCUS_ZONE_DIL * wf))
        z = cv2.dilate(sharp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return np.clip(cv2.GaussianBlur(z.astype(np.float32), (0, 0), max(1.0, 0.04 * wf)), 0.0, 1.0)
    except Exception:
        return None


SHARP_ADAPT = True     # the retouch follows the footage's own sharpness (see skin.FS_SQ)
SHARP_WIN = 12         # frames each side that make "the sharpness round it"
SHARP_BAND = (0.0016, 0.0064)   # the measured band, of the face width (Gaussian sigma)
SHARP_MAD_MAX = 0.30   # spread (median absolute deviation over the median) of the frames round it above which it is not followed
SHARP_MIN_WF = 60.0    # face width in proxy px below which the measure is not trusted


def _sharp_e(px, lm: np.ndarray, t: int) -> Optional[float]:
    """The structure of the skin at pore size on frame t: the 75th percentile of |fine band| over the cheeks and the
    forehead (eyes, brows, nose and mouth left out). A softer frame gives a smaller value, the same face sharp a larger."""
    try:
        pts = np.asarray(lm[:, :2], np.float32)
        if not np.isfinite(pts).all():
            return None
        wf = float(np.hypot(*(pts[16] - pts[0])))
        if wf < SHARP_MIN_WF:
            return None
        rgb = np.asarray(px.frame(t))
        h, w = rgb.shape[:2]
        x0 = int(max(0, pts[:, 0].min() - 4)); x1 = int(min(w, pts[:, 0].max() + 5))
        y0 = int(max(0, pts[:, 1].min() - 0.3 * wf)); y1 = int(min(h, pts[:, 1].max() + 5))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        Y = (0.299 * rgb[y0:y1, x0:x1, 0] + 0.587 * rgb[y0:y1, x0:x1, 1] + 0.114 * rgb[y0:y1, x0:x1, 2]).astype(np.float32) / 255.0
        o = np.array([x0, y0], np.float32)
        m = np.zeros(Y.shape, np.uint8)
        outline = np.concatenate([pts[0:17], pts[26:16:-1]]) - o
        cv2.fillPoly(m, [np.round(outline).astype(np.int32)], 1)
        k = max(3, int(0.06 * wf)) | 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
        kd = max(3, int(0.09 * wf)) | 1
        for a, b in ((17, 27), (36, 48), (48, 68), (27, 36)):
            hull = cv2.convexHull(np.round(pts[a:b] - o).astype(np.int32))
            z = np.zeros(Y.shape, np.uint8)
            cv2.fillConvexPoly(z, hull, 1)
            m[cv2.dilate(z, np.ones((kd, kd), np.uint8)) > 0] = 0
        if int(m.sum()) < 400:
            return None
        s1, s2 = SHARP_BAND[0] * wf, SHARP_BAND[1] * wf
        b = cv2.GaussianBlur(Y, (0, 0), max(0.3, s1)) - cv2.GaussianBlur(Y, (0, 0), max(0.6, s2))
        v = np.abs(b[m > 0])
        # skin brightness differs between frames: the band is taken relative to the local level
        lvl = float(Y[m > 0].mean())
        return float(np.percentile(v, 75)) / max(lvl, 0.05)
    except Exception:
        return None


def _sharp_q(tracker, st, p: int, t: int, px) -> float:
    """This frame's skin sharpness over the median of the frames round it (same shot, same face). 1 = as round it."""
    try:
        cache = tracker.__dict__.setdefault("_sharp_cache", {})
        key0 = id(st)

        def e(s):
            k_ = (key0, p, s)
            if k_ not in cache:
                v = None
                if 0 <= s < st.final.shape[1] and st.present[p, s] and np.isfinite(st.face[p, s, :, :2]).all():
                    lm_ = st.face[p, s].copy()
                    lm_[:, 0] *= px.sx
                    lm_[:, 1] *= px.sy
                    v = _sharp_e(px, lm_, s)
                cache[k_] = v
            return cache[k_]
        if len(cache) > 6000:
            cache.clear()
        own = [e(s) for s in (t - 1, t, t + 1) if st.connected(s, t)] if hasattr(st, "connected") else [e(t)]
        own = [x for x in own if x is not None]
        if not own:
            return 1.0
        cur = float(np.mean(own))
        ref = []
        for s in range(t - SHARP_WIN, t + SHARP_WIN + 1):
            if s == t or not (0 <= s < st.final.shape[1]):
                continue
            if hasattr(st, "connected") and not st.connected(s, t):
                continue
            v = e(s)
            if v is not None:
                ref.append(v)
        if len(ref) < 5:
            return 1.0
        r = float(np.median(ref))
        if r <= 1e-6:
            return 1.0
        # the frames round it must agree on what "sharp" is (a cut that goes from a dark blur to a lit face, a zoom, a
        # flash: the measure then says more about the light than the focus): otherwise no following (cut 018: 3.25)
        mad = float(np.median(np.abs(np.asarray(ref) - r))) / r
        q = float(cur / r)
        if mad > SHARP_MAD_MAX or not (0.45 <= q <= 1.8):
            return 1.0
        return q
    except Exception:
        return 1.0


def _parse_visible(tracker, pid: int, t: int, face: np.ndarray, px) -> float:
    """0..1: how plainly this face shows on frame t going by the face parser (its skin inside the face outline)."""
    try:
        from ..track import facelayer as FL
        layer = FL.load(tracker)
        if layer is None:
            return 0.0
        fm = layer.maps(pid, t, (px.h, px.w))
        if fm is None or float(fm.get("trust", 0)) <= 0:
            return 0.0
        lm = np.asarray(face, np.float32)[:, :2] * np.array([px.sx, px.sy], np.float32)
        hull = cv2.convexHull(np.round(lm).astype(np.int32))
        x, y, w_, h_ = cv2.boundingRect(hull)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(px.w, x + w_), min(px.h, y + h_)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0
        mk = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mk, hull - [x0, y0], 1)
        if int(mk.sum()) < 30:
            return 0.0
        share = float(fm["skin"][y0:y1, x0:x1][mk > 0].mean())
        return float(np.clip((share - VIS_PARSE_SHARE[0]) / (VIS_PARSE_SHARE[1] - VIS_PARSE_SHARE[0]), 0.0, 1.0)) * float(fm["trust"])
    except Exception:
        return 0.0


VIS_LO = 0.12         # face visibility where the face retouch starts
VIS_FULL = 0.5        # and where it is at full strength
CACHE_BYTES = 900 * 2 ** 20   # the preview's mask cache
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
            jj.body_keep = float(np.clip(getattr(pp, "body_detail", 60.0) / 100.0, 0, 1))
            jj.body_size = float(np.clip(getattr(pp, "body_blemish_size", 40.0) / 100.0, 0, 1))
            jj.bmarks = _body_marks(st, j.p, t, pp, jj.v)
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
        vals.append((person.id, pp.skin_scope, round(float(fth), 3), tuple(on),
                     round(float(getattr(pp, "blemish_size", 50.0)), 1),      # the spot sizes the flaws are looked for at
                     getattr(pp, "body_scope", "all"), round(float(getattr(pp, "body_blemish_size", 40.0)), 1)))
    return (id(tracker.store), getattr(cut, "analysis_version", 0), tuple(vals), BS.garment_token(tracker))


def _raw_cached(cut, tracker, s: int, cache):
    from collections import OrderedDict
    lru = tracker.__dict__.get("_steady_lru")
    sig = _sig(cut, tracker)
    # the session clears its mask cache whenever an edit changes masks: start over then too (a cache that kept the
    # bodies' entries over a tracking change says so: kept_over_tracks). The bodies' motions between frames stay:
    # each is kept with what the tracks said of its two frames (_body_motion)
    if lru is None or tracker.__dict__.get("_steady_sig") != sig \
            or (cache is not None and (len(cache) == 0 or cache.pop("__new__", False))):
        lru = tracker.__dict__["_steady_lru"] = OrderedDict()
        tracker.__dict__["_steady_sig"] = sig
    if s in lru:
        lru.move_to_end(s)
        return lru[s]
    with _tm("raw"):
        rf = _prepare_raw(cut, tracker, s, cache)
    if rf is not None:
        for j in rf.jobs:
            j.masks = {k: v.astype(np.float16) for k, v in j.masks.items()}
    lru[s] = rf
    while len(lru) > STEADY_LRU:
        lru.popitem(last=False)
    return rf


STEADY_ROBUST = True   # a mask that drops out on one or two frames (the parser or the colour model missing the forehead for a
STEADY_WIN = 0.15      # frame: cut 086 frames 23/27/33, cut 100 frame 4) is pulled to the median of the frames round it


def _steady(cut, tracker, st, rf: RetouchFrame, cache) -> RetouchFrame:
    t = rf.t
    px = tracker.proxy
    N = st.final.shape[1]
    k = np.array([px.sx, px.sy])
    try:
        _body_motions_ahead(cut, tracker, st, rf, cache)
    except Exception:
        import traceback
        traceback.print_exc()
    for j in rf.jobs:
        # the body's mask has no face to follow: it is held steady along the picture's own motion (_steady_bskin)
        if j.body and j.body.get("mask") is not None:
            try:
                with _tm("body.steady_area"):
                    j.body = dict(j.body, mask=_steady_bskin(cut, tracker, st, rf, j, cache))
            except Exception:
                import traceback
                traceback.print_exc()
        if getattr(j, "bdet", None) is not None:
            try:
                with _tm("body.steady_spots"):
                    j.bdet = _steady_bdet(cut, tracker, st, rf, j, cache)
            except Exception:
                import traceback
                traceback.print_exc()
        if j.face is None or not j.masks:
            continue
        Lt = j.face[:, :2] * k
        own = {n: m.astype(np.float32) for n, m in j.masks.items()}
        shp = next(iter(own.values())).shape
        frames = [(1.0, own)]                       # (weight, masks on this frame's geometry)
        veto = getattr(j, "veto", None)
        vetos = [veto.astype(np.float32)] if veto is not None else []
        dets = []                                   # (weight, a neighbour's score maps, its motion onto this frame)
        def usable(s):
            return 0 <= s < N and bool(st.present[j.p, s]) and st.connected(s, t) \
                and bool(np.isfinite(st.face[j.p, s, :, :2]).all())

        for d in range(1, STEADY_R + 1):
            w = float(np.exp(-0.5 * (d / STEADY_SIGMA) ** 2))
            # both sides or neither: on the first and last frames of a shot (or next to a frame the face is missing on)
            # the frames of one side alone pulled the mask towards where the face was, and its cover dropped by a
            # quarter on that frame only (lngshot cut 011, last frame: 0.97 raw, 0.72 steadied)
            if not (usable(t - d) and usable(t + d)):
                break
            for s in (t - d, t + d):
                rs = _raw_cached(cut, tracker, s, cache)
                js = next((x for x in rs.jobs if x.pid == j.pid), None) if rs is not None else None
                if js is None or js.face is None:
                    continue
                Ls = js.face[:, :2] * k
                A, _ = cv2.estimateAffinePartial2D(Ls.astype(np.float32), Lt.astype(np.float32), method=cv2.LMEDS)
                if A is None:
                    continue
                ws = {}
                for n in own:
                    ms = js.masks.get(n)
                    # that frame had none (below threshold): counts as empty
                    ws[n] = np.zeros(shp, np.float32) if ms is None else \
                        cv2.warpAffine(ms.astype(np.float32), A, (shp[1], shp[0]), flags=cv2.INTER_LINEAR, borderValue=0)
                frames.append((w, ws))
                if j.det is not None:
                    dets.append((w, js.det, A))
                vs = getattr(js, "veto", None)
                if veto is not None and vs is not None and vs.shape == veto.shape:
                    q_ = veto.shape[1] / float(shp[1])
                    Aq = A.copy(); Aq[:, 2] *= q_
                    vetos.append(cv2.warpAffine(vs.astype(np.float32), Aq, (veto.shape[1], veto.shape[0]),
                                                flags=cv2.INTER_LINEAR, borderValue=0))
        wsum = float(sum(w for w, _ in frames))
        acc = {}
        robust = STEADY_ROBUST and len(frames) >= 3
        for n in own:
            if robust:
                # only where any of the frames has something: the median of five full masks is slow
                any_ = np.zeros(shp, bool)
                for _, ms in frames:
                    any_ |= ms[n] > 0.004
                ys, xs = np.nonzero(any_.any(1))[0], np.nonzero(any_.any(0))[0]
                out = np.zeros(shp, np.float32)
                if len(ys) and len(xs):
                    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
                    stk = np.stack([ms[n][y0:y1, x0:x1] for _, ms in frames])
                    med = np.median(stk, axis=0)
                    stk = np.clip(stk, med - STEADY_WIN, med + STEADY_WIN)
                    wv = np.array([w for w, _ in frames], np.float32)[:, None, None]
                    out[y0:y1, x0:x1] = (stk * wv).sum(0)
                acc[n] = out
            else:
                a_ = np.zeros(shp, np.float32)
                for w, ms in frames:
                    a_ += w * ms[n]
                acc[n] = a_
        if veto is not None and wsum > 1.0:
            h_, w_ = shp
            vv = vetos[0]
            if robust and len(vetos) >= 3:
                # what covers the face stays for more than a frame or two; a frame on which the skin is lost for
                # one or two frames is this frame's failure, not a cover
                vv = np.maximum(vv, np.median(np.stack(vetos), axis=0))
            v_ = cv2.resize(vv, (w_, h_), interpolation=cv2.INTER_LINEAR)
            wf_ = float(np.hypot(*(Lt[16] - Lt[0])))
            v_ = np.clip(cv2.GaussianBlur(v_, (0, 0), max(1.0, 0.02 * wf_)) * 2.5, 0, 1)
            # where this frame sees no skin, a neighbour's mask may not hold more than this frame's own
            for n in acc:
                o_ = own[n]
                avg = acc[n] / wsum
                acc[n] = (o_ + (avg - o_).clip(None, 0) + (avg - o_).clip(0, None) * v_) * wsum
        j.masks = {n: (m / wsum).astype(np.float32) for n, m in acc.items()}
        if j.det is not None and dets:
            try:
                j.det = _det_fuse(j.det, dets)
            except Exception:
                import traceback
                traceback.print_exc()
    return rf


BODY_STEADY_HOLD = 2.5      # a neighbour's skin counts where this frame's own mask (softened) times this reaches 1


def _gray(px, t: int) -> np.ndarray:
    return cv2.cvtColor(np.asarray(px.frame(t)), cv2.COLOR_RGB2GRAY)


def _body_marks(st, p: int, t: int, pp, v) -> Optional[list]:
    """The person's body spot marks as they lie on frame t (None: no marks, or 몸 잡티 제거 is off)."""
    marks = getattr(pp, "body_marks", None)
    if not marks or v.get("body_blemish", 0) <= 1e-4:
        return None
    try:
        return BS.marks_at(st, p, t, marks) or None
    except Exception:
        return None


BODY_EFFECTS = ("body_skin", "body_blemish", "knee_tone", "belly_fold")   # the values that bring the body skin's area with them
BDET_NOW = True            # a job that came without its spots (a single frame, a test) has them found on the plane


def _bdet_raw(cut, px, t: int, job, rgb_u8: np.ndarray) -> Optional[dict]:
    """One frame's body spot candidates for a job, on the best picture there is of the body: the original where it
    has clearly more pixels than the analysis picture (never more than the body's size asks for: bodyspots.DET_REF),
    else the analysis picture. One piece of the original is read for the whole body and looked at tile by tile."""
    info = job.body
    bm = BS.unpack_mask(info.get("mask")) if info else None
    if not info or bm is None or not info.get("box"):
        return None
    ref_src = float(info["ref"])
    x0, y0, x1, y1 = info["box"]
    W, H = int(px.info.width), int(px.info.height)
    box_src = (max(0, int(x0 / px.sx) // 2 * 2), max(0, int(y0 / px.sy) // 2 * 2),
               min(W, -(-int(np.ceil(x1 / px.sx)) // 2) * 2), min(H, -(-int(np.ceil(y1 / px.sy)) // 2) * 2))
    want = min(1.0, BP.DET_REF / max(ref_src, 1.0))                # picture px per source px
    img, scale, from_src = None, min(px.sx, want), False
    if DET_SRC and want > px.sx * BP.DET_GAIN:
        try:
            from ..core import frames as FRM
            from ..warp.render import _vs_for
            sc = FRM.source_crops(px, vs=_vs_for(cut))
            if sc is not None:
                sc_scale = float(np.clip(round(want * 8.0) / 8.0, 0.125, 1.0))
                with _tm("body.spots.read"):
                    img = sc.piece(t, box_src, sc_scale)
                scale, from_src = sc_scale, True
        except Exception:
            import traceback
            traceback.print_exc()
            img = None
    if img is None:
        ax0, ay0 = int(box_src[0] * px.sx), int(box_src[1] * px.sy)
        ax1, ay1 = min(rgb_u8.shape[1], int(np.ceil(box_src[2] * px.sx))), min(rgb_u8.shape[0], int(np.ceil(box_src[3] * px.sy)))
        img = rgb_u8[ay0:ay1, ax0:ax1]
        box_src = (ax0 / px.sx, ay0 / px.sy, ax1 / px.sx, ay1 / px.sy)
        scale, from_src = px.sx, False
        if want < px.sx * 0.9:
            f_ = want / px.sx
            img = cv2.resize(img, (max(8, int(round(img.shape[1] * f_))), max(8, int(round(img.shape[0] * f_)))),
                             interpolation=cv2.INTER_AREA)
            scale = want
    if img is None or img.shape[0] < 8 or img.shape[1] < 8:
        return None
    ih, iw = img.shape[:2]
    fx, fy = iw / max(box_src[2] - box_src[0], 1e-6), ih / max(box_src[3] - box_src[1], 1e-6)
    bmf = bm

    def pieces(sx0, sy0, sx1, sy1):
        u0, v0 = int(round((sx0 - box_src[0]) * fx)), int(round((sy0 - box_src[1]) * fy))
        u1, v1 = int(round((sx1 - box_src[0]) * fx)), int(round((sy1 - box_src[1]) * fy))
        u0, v0, u1, v1 = max(0, u0), max(0, v0), min(iw, u1), min(ih, v1)
        if u1 - u0 < 8 or v1 - v0 < 8:
            return None
        return img[v0:v1, u0:u1]

    def region_at(sx0, sy0, sx1, sy1, shp):
        # analysis px -> this piece's px
        u0, v0 = max(0, int(round((sx0 - box_src[0]) * fx))), max(0, int(round((sy0 - box_src[1]) * fy)))
        A = np.array([[fx / px.sx, 0, -box_src[0] * fx - u0 + 0.5 * fx / px.sx - 0.5],
                      [0, fy / px.sy, -box_src[1] * fy - v0 + 0.5 * fy / px.sy - 0.5]], np.float32)
        return cv2.warpAffine(bmf, A, (shp[1], shp[0]), flags=cv2.INTER_LINEAR, borderValue=0)

    net_for = None
    if BP.SPOTNET:
        from . import spotnet_run as SR
        if SR.available():
            net_for = lambda im, ref_: (lambda region, size: SR.run(im.astype(np.float32) / 255.0, region, ref_, max_size=size))
    with _tm("body.spots.find"):
        det = BP.detect(pieces, region_at, info.get("model"), ref_src, job.body_size, 0.5 * (fx + fy), box_src, net_for)
    det["src"] = from_src
    if BP.HANDS_HELD and info.get("hands"):
        c = det["cands"]
        for i in range(len(c)):
            if c["hold"][i] in (BP.HOLD_NONE, BP.HOLD_WEAK) and BS.on_hand(info["hands"], float(c["x"][i]), float(c["y"][i])):
                c["hold"][i] = BP.HOLD_HAND
    return det


def _bdet_now(job, Y, Cb, Cr, bmY, y_per_src: float) -> Optional[dict]:
    """The body's spots of a job found on this very picture (no candidates came with the job)."""
    info = job.body
    if not info or not info.get("box"):
        return None
    hY, wY = Y.shape
    cbY = Cb if Cb.shape == Y.shape else cv2.resize(Cb, (wY, hY), interpolation=cv2.INTER_LINEAR)
    crY = Cr if Cr.shape == Y.shape else cv2.resize(Cr, (wY, hY), interpolation=cv2.INTER_LINEAR)
    ys = np.nonzero(bmY.max(1) > 0.3)[0]
    xs = np.nonzero(bmY.max(0) > 0.3)[0]
    if not len(ys) or not len(xs):
        return None
    k = y_per_src
    box_src = (int(xs[0] / k) // 2 * 2, int(ys[0] / k) // 2 * 2, -(-int(np.ceil((xs[-1] + 1) / k)) // 2) * 2,
               -(-int(np.ceil((ys[-1] + 1) / k)) // 2) * 2)

    def cut_(a, sx0, sy0, sx1, sy1):
        u0, v0 = max(0, int(round(sx0 * k))), max(0, int(round(sy0 * k)))
        u1, v1 = min(wY, int(round(sx1 * k))), min(hY, int(round(sy1 * k)))
        return a[v0:v1, u0:u1] if (u1 - u0 >= 8 and v1 - v0 >= 8) else None

    def pieces(sx0, sy0, sx1, sy1):
        y_ = cut_(Y, sx0, sy0, sx1, sy1)
        return None if y_ is None else (y_, cut_(cbY, sx0, sy0, sx1, sy1), cut_(crY, sx0, sy0, sx1, sy1))

    def region_at(sx0, sy0, sx1, sy1, shp):
        r_ = cut_(bmY, sx0, sy0, sx1, sy1)
        return r_ if (r_ is not None and r_.shape == tuple(shp)) else None
    return BP.detect(pieces, region_at, None, float(info["ref"]), job.body_size, k, box_src)


def _steady_bskin(cut, tracker, st, rf: RetouchFrame, j, cache) -> Optional[dict]:
    """The body skin mask of job j held steady through time (float32, the analysis picture's size).

    As with the face's masks: the same person's masks of the frames round it (STEADY_R each side, same shot, both
    sides or neither), each carried onto this frame, pulled to their median and averaged. A body has no landmarks,
    so each neighbour is carried along the picture's own motion (bodyskin.motion); where the carried picture is not
    this frame's picture (something came in front, another person, a jump) that neighbour has no say. And whatever
    the neighbours say, skin is only where this frame's own mask has it close by: nothing is carried onto a cover,
    across a cut, or onto a person who took the place."""
    t = rf.t
    raw = j.body["mask"]
    box = j.body.get("box")
    if box is None or STEADY_R < 1:
        return raw
    # it does not depend on the sliders: kept with the frame's other masks (a slider being dragged asks for it again
    # and again, and the motion of four neighbours is not measured each time)
    key = (t, j.pid, "bskin+", "")
    tok = _body_tok(tracker, st, j.p, t, STEADY_R, j.body.get("key"))
    got = _cache_look(cache, key, tok)
    if got is not _MISS:
        return got["pm"]
    out = BS.pack_mask(_steady_bskin_make(cut, tracker, st, rf, j, cache, BS.unpack_mask(raw), box))
    if out is None:
        out = raw
    if cache is not None and len(cache) > 0:
        _cache_put(cache, key, dict(maps={"m": out["m"]}, thumb=None, pm=out), tok)
    return out


def _body_neighbours(cut, tracker, st, rf: RetouchFrame, j, cache, both: bool = True) -> list:
    """The frames round rf.t that can say something about job j's body on this frame: [(weight, frame, that frame's
    job of the same person or None, the motion of this frame onto it)], nearest first. Same shot, the person there.
    both: a distance counts only with both of its frames (the masks: one side alone pulls an edge its way); else
    whichever are there (the spots: at a shot's first and last frames one side is better than none). Measured once
    for a frame and job (the mask and the spots ask for the same)."""
    memo = rf.__dict__.setdefault("_bnb", {})
    key = (j.pid, both)
    if key in memo:
        return memo[key]
    t = rf.t
    px = tracker.proxy
    N = st.final.shape[1]
    box = (j.body or {}).get("box")
    out = []
    memo[key] = out
    if box is None:
        return out
    full = memo.get((j.pid, False)) if both else None

    def usable(s):
        return 0 <= s < N and bool(st.present[j.p, s]) and st.connected(s, t)
    for d in range(1, STEADY_R + 1):
        ok = [s_ for s_ in (t - d, t + d) if usable(s_)]
        if both and len(ok) < 2:
            break
        w = float(np.exp(-0.5 * (d / STEADY_SIGMA) ** 2))
        got = []
        for s_ in ok:
            hit = next((e for e in (full or []) if e[1] == s_), None)
            if hit is not None:
                got.append(hit)
                continue
            rs = _raw_cached(cut, tracker, s_, cache)
            js = next((x for x in rs.jobs if x.pid == j.pid), None) if rs is not None else None
            mo = _body_motion(tracker, j, t, s_, box, (js.body or {}).get("box") if js is not None else None)
            if mo is None:
                if both:
                    got = None
                    break
                continue
            got.append((w, s_, js, mo))
        if got is None:
            break
        out += got
    return out


BFLOW_LRU = 10          # pairs of frames a person's motion is kept for (a frame asks for four: two each side)
import threading as _threading_mod
_BFLOW_LOCK = _threading_mod.Lock()


def _body_motion(tracker, j, t: int, s: int, box_t, box_s) -> Optional[dict]:
    """The motion of person j's piece of frame t onto frame s (bodyskin.motion_pair). A pair of frames is measured
    once, both ways (each way checks the other): frame s asks for the other way when its turn comes. Kept on the
    tracker beside the frames of _raw_cached and dropped with them."""
    from collections import OrderedDict
    lru = tracker.__dict__.get("_bflow")
    sig = tracker.__dict__.get("_steady_sig")
    if lru is None or tracker.__dict__.get("_bflow_sig") != sig:
        lru = tracker.__dict__["_bflow"] = OrderedDict()
        tracker.__dict__["_bflow_sig"] = sig
    lo, hi = (t, s) if t < s else (s, t)
    key = (j.pid, lo, hi)
    tok = _motion_tok(tracker, lo, hi)
    with _BFLOW_LOCK:
        got = lru.get(key)
        if got is not None and got[0] != tok:
            got = None                              # (measured with other owners: a tracking update since)
        if got is not None:
            lru.move_to_end(key)
    if got is None:
        px = tracker.proxy
        size = (px.w, px.h)
        try:
            w_t = BS.ownership(tracker, t, j.p, size, others=False)[0]
            w_s = BS.ownership(tracker, s, j.p, size, others=False)[0]
        except Exception:
            w_t = w_s = None
        with _tm("body.motion"):
            a, b = BS.motion_pair(_gray(px, t), _gray(px, s), box_t, box_s, w_t, w_s)
        got = (tok,) + ((a, b) if t < s else (b, a))
        with _BFLOW_LOCK:
            lru[key] = got
            lru.move_to_end(key)
            n = BFLOW_LRU * max(1, len({k[0] for k in lru}))
            while len(lru) > n:
                lru.popitem(last=False)
    return got[1] if t < s else got[2]


def _motion_tok(tracker, lo: int, hi: int) -> tuple:
    """What a body's motion between two frames was measured with besides the pictures: the owners of both frames'
    pixels, which the tracks of those frames give (bodyskin.track_token)."""
    st = tracker.store
    return (BS.track_token(st, lo, lo), BS.track_token(st, hi, hi))


MOTION_PAIRS = 4        # motions of the bodies of a frame measured side by side (1: each when it is asked for, one after
                        # the other: 0.07 to 0.12 s a pair, two new pairs a person and frame: j2799)


def _body_motions_ahead(cut, tracker, st, rf: RetouchFrame, cache) -> None:
    """The motions the steadying of this frame's bodies will ask for (_body_neighbours), measured side by side and
    kept where _body_motion finds them. What each needs of the tracker (the frames, the owners) is fetched here,
    one after the other; the measuring itself touches nothing shared."""
    if MOTION_PAIRS <= 1 or STEADY_R < 1:
        return
    from collections import OrderedDict
    t = rf.t
    px = tracker.proxy
    N = st.final.shape[1]
    todo = []
    for j in rf.jobs:
        box = (j.body or {}).get("box") if j.body else None
        if box is None or j.body.get("mask") is None:
            continue
        # (a frame whose steadied area and spots are kept asks for no motion at all)
        tok = _body_tok(tracker, st, j.p, t, STEADY_R, j.body.get("key"))
        if _cache_look(cache, (t, j.pid, "bskin+", ""), tok) is not _MISS and \
                (getattr(j, "bdet", None) is None or _cache_look(cache, (t, j.pid, "bdet+", f"{j.body_size:.3f}"), tok) is not _MISS):
            continue
        for d in range(1, STEADY_R + 1):
            for s_ in (t - d, t + d):
                if not (0 <= s_ < N and bool(st.present[j.p, s_]) and st.connected(s_, t)):
                    continue
                rs = _raw_cached(cut, tracker, s_, cache)       # (this may start the motions' table over)
                lru = tracker.__dict__.get("_bflow")
                key = (j.pid, min(t, s_), max(t, s_))
                with _BFLOW_LOCK:
                    known = lru is not None and tracker.__dict__.get("_bflow_sig") == tracker.__dict__.get("_steady_sig") \
                        and key in lru and lru[key][0] == _motion_tok(tracker, key[1], key[2])
                if known or any(e[0] == key for e in todo):
                    continue
                js = next((x for x in rs.jobs if x.pid == j.pid), None) if rs is not None else None
                todo.append((key, j, s_, box, (js.body or {}).get("box") if (js is not None and js.body) else None))
    if len(todo) < 2:
        return
    size = (px.w, px.h)
    grays, owns = {}, {}
    for _key, j, s_, _b, _bs in todo:
        for f in (t, s_):
            if f not in grays:
                grays[f] = _gray(px, f)
            if (j.p, f) not in owns:
                try:
                    owns[(j.p, f)] = BS.ownership(tracker, f, j.p, size, others=False)[0]
                except Exception:
                    owns[(j.p, f)] = None

    def one(e):
        _key, j, s_, box, box_s = e
        with _tm("body.motion"):
            return BS.motion_pair(grays[t], grays[s_], box, box_s, owns[(j.p, t)], owns[(j.p, s_)])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(MOTION_PAIRS, len(todo))) as ex:
        got = list(ex.map(one, todo))
    sig = tracker.__dict__.get("_steady_sig")
    lru = tracker.__dict__.get("_bflow")
    if lru is None or tracker.__dict__.get("_bflow_sig") != sig:
        lru = tracker.__dict__["_bflow"] = OrderedDict()
        tracker.__dict__["_bflow_sig"] = sig
    with _BFLOW_LOCK:
        for (key, _j, s_, _b, _bs), (a, b) in zip(todo, got):
            lru[key] = (_motion_tok(tracker, key[1], key[2]),) + ((a, b) if t < s_ else (b, a))
            lru.move_to_end(key)
        n = BFLOW_LRU * max(1, len({k[0] for k in lru}))
        while len(lru) > n:
            lru.popitem(last=False)


def _steady_bdet(cut, tracker, st, rf: RetouchFrame, j, cache) -> dict:
    """The body's spots of job j held through time (bodyspots.fuse), kept with the frame's masks: they do not
    depend on the sliders' strengths."""
    t = rf.t
    px = tracker.proxy
    own = j.bdet
    if STEADY_R < 1 or not j.body:
        return own
    key = (t, j.pid, "bdet+", f"{j.body_size:.3f}")
    tok = _body_tok(tracker, st, j.p, t, STEADY_R, j.body.get("key"))
    got = _cache_look(cache, key, tok)
    if got is not _MISS:
        return got
    nb = _body_neighbours(cut, tracker, st, rf, j, cache, both=False)

    def where_on(s_, js):
        """A place of this frame on frame s_ by the body part it lies on, with that frame's own skin area there."""
        if js is None or not js.body:
            return None
        box = {}

        def where(x, y):
            q = BS.carry_point(st, j.p, t, s_, (x, y))
            if q is None:
                return None
            if "m" not in box:
                box["m"] = BS.unpack_mask(js.body.get("mask"))
            ms = box["m"]
            if ms is None:
                return None
            u, v = int(round(q[0] * px.sx)), int(round(q[1] * px.sy))
            vis = float(ms[v, u]) if (0 <= v < ms.shape[0] and 0 <= u < ms.shape[1]) else 0.0
            return q[0], q[1], vis
        return where
    others = [(w, getattr(js, "bdet", None) if js is not None else BP.empty(own["scale"]), mo, where_on(s_, js))
              for w, s_, js, mo in nb]
    out = BP.fuse(own, others, float(j.body["ref"]), px.sx, px.sy)
    if cache is not None and len(cache) > 0:
        _cache_put(cache, key, out, tok)
    return out


STEADY_PIECE = True     # the body's area is steadied on the piece the person and their neighbours' pieces take (False:
                        # every map of it the whole picture's, 1.23's first build)


def _steady_bskin_make(cut, tracker, st, rf: RetouchFrame, j, cache, own: np.ndarray, box) -> np.ndarray:
    shp = own.shape
    nb = _body_neighbours(cut, tracker, st, rf, j, cache, both=True)
    if len(nb) < 2:
        return own
    # the piece of the picture anything can happen on: this frame's own area and the pieces the neighbours' motion
    # was measured on (outside them a neighbour says what this frame says, and this frame says nothing)
    wx0, wy0, wx1, wy1 = 0, 0, shp[1], shp[0]
    if STEADY_PIECE:
        r_ = _roi(own, 0)
        bx = [mo["box"] for _w, _s, _js, mo in nb] + ([(r_[2], r_[0], r_[3], r_[1])] if r_ is not None else [])
        wx0, wy0 = max(0, min(int(b_[0]) for b_ in bx)), max(0, min(int(b_[1]) for b_ in bx))
        wx1, wy1 = min(shp[1], max(int(b_[2]) for b_ in bx)), min(shp[0], max(int(b_[3]) for b_ in bx))
        if wx1 <= wx0 or wy1 <= wy0:
            return own
    win = (wx0, wy0, wx1, wy1)
    own_w = own[wy0:wy1, wx0:wx1]
    frames = [(1.0, own_w)]
    for w, _s, js, mo in nb:
        ms = BS.unpack_mask(js.body.get("mask")) if (js is not None and js.body) else None
        carried, conf = BS.carry(mo, ms if (ms is not None and ms.shape == shp) else np.zeros(shp, np.float32), shp, win)
        # where the neighbour's picture is not this one, it says what this frame says
        frames.append((w, conf * carried + (1.0 - conf) * own_w))
    any_ = np.zeros(own_w.shape, bool)
    for _, m in frames:
        any_ |= m > 0.004
    ys, xs = np.nonzero(any_.any(1))[0], np.nonzero(any_.any(0))[0]
    if not len(ys) or not len(xs):
        return own
    y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
    stk = np.stack([m[y0:y1, x0:x1] for _, m in frames])
    med = np.median(stk, axis=0)
    stk = np.clip(stk, med - STEADY_WIN, med + STEADY_WIN)
    wv = np.array([w for w, _ in frames], np.float32)[:, None, None]
    avg = (stk * wv).sum(0) / float(wv.sum())
    o_ = own_w[y0:y1, x0:x1]
    ref_px = float((j.body or {}).get("ref", 0.0)) * tracker.proxy.sx
    near = np.clip(cv2.GaussianBlur(o_, (0, 0), max(1.0, 0.02 * ref_px)) * BODY_STEADY_HOLD, 0, 1)
    # what the frames round it add comes only beside this frame's own area, and never onto what this frame itself
    # rules out: something that lies over the skin here (a sleeve, hair, a prop, another person) is not retouched
    # because the skin under it was seen a frame ago
    allow = BS.unpack_allow((j.body or {}).get("allow"))
    if allow is not None and allow.shape == shp:
        near = near * allow[wy0 + y0:wy0 + y1, wx0 + x0:wx0 + x1]
    else:
        near = near * 0.0
    out = own.copy()
    out[wy0 + y0:wy0 + y1, wx0 + x0:wx0 + x1] = o_ + (avg - o_).clip(None, 0) + (avg - o_).clip(0, None) * near
    return out


def person_masks(cut, tracker, t: int, pid: int) -> Optional[RetouchFrame]:
    """The selected person's skin area for the K overlay, also before any skin slider was moved (with the
    person's own region and scope settings), so the area can be checked before retouching."""
    return _prepare_raw(cut, tracker, t, None, only=pid)


PARSE_NOW = True          # a face the analysis kept no parser maps for is parsed when its frame is retouched
PARSE_NOW_MIN = 60.0      # proxy px of face width from which that is done
_FPARSER = [None]
_FP_COOL = [0.0]          # out of GPU memory: no parse-now until then (time.time())
import threading as _threading
import time as _time
_FPLOCK = _threading.Lock()


def _drop_parser() -> None:
    """GPU memory is short: the parser is made again the next time it is needed."""
    if _FPARSER[0]:
        _FPARSER[0] = None


try:
    from ..track import models as _M
    _M.on_low_gpu_memory(_drop_parser)
except Exception:
    pass


def _parse_now(rgb_u8: np.ndarray, lm: np.ndarray) -> Optional[dict]:
    """The face layer's maps for one face, made on the spot (v3). Without them only the colour model says what is
    skin, and a fringe, a pale collar or bleached hair pass for it (cut 059, the person on the right)."""
    try:
        if not np.isfinite(lm[:, :2]).all() or float(np.hypot(*(lm[16, :2] - lm[0, :2]))) < PARSE_NOW_MIN:
            return None
        from ..track import faceparse as FP
        from ..track import facelayer as FL
        if _FP_COOL[0] > _time.time():
            return None
        with _FPLOCK:                           # frames are prepared on several threads: one parser, one call at a time
            if _FPARSER[0] is None:
                _FPARSER[0] = FP.make_parser() if FP.ready() else False
            if not _FPARSER[0]:
                return None
            r = _FPARSER[0].parse(rgb_u8, np.asarray(lm, np.float32))
        if r is None:
            return None
        shape = rgb_u8.shape[:2]
        cover = FP.to_image(np.ones((FP.KEEP, FP.KEEP), np.float32), r["M"], shape)
        rel = FL._reliability(r["skin"], FL._crop_lm(np.asarray(lm, np.float32), r["M"]))
        return dict(tex=None, skin=FP.to_image(r["skin"], r["M"], shape), neck=FP.to_image(r["neck"], r["M"], shape),
                    cover=cover, trust=float(np.clip((rel - 0.35) / 0.25, 0, 1)))
    except Exception as e_:
        from ..track import models as M_
        if M_.is_oom(e_):
            # short of GPU memory (an analysis or another program): not for good, again in a while
            print("parse-now paused (GPU memory):", repr(e_)[:200], flush=True)
            _FPARSER[0] = None
            _FP_COOL[0] = _time.time() + 30.0
            return None
        if _FPARSER[0] is not False:
            print("parse-now off:", repr(e_)[:200], flush=True)
        _FPARSER[0] = False
        return None


DOUBLE_DIST = 0.3     # face centres closer than this share of the face width: one face held by two tracks


def _double_faces(st, tracker, t: int) -> set:
    """Tracks whose face on frame t is another track's face: one person taken for two (lngshot cut 005: a girl
    holding a magazine up to her eyes became one track for her head and one for the magazine and her hands, both
    with her face). Retouched twice, and the second one's mask spread over the magazine (the first track's pixels).
    The face stays with the track the person matte gives it to (then the surer face); the other loses it."""
    if st is None or not getattr(st, "has_face", False):
        return set()
    ids = [p for p in np.nonzero(st.present[:, t])[0] if np.isfinite(st.face[p, t, :, :2]).all()]
    if len(ids) < 2:
        return set()
    geo = {}
    for p in ids:
        F = st.face[p, t, :, :2]
        geo[p] = (F.mean(0), float(np.hypot(*(F[16] - F[0]))))
    lab = None
    try:
        lab = tracker.labels(t)
    except Exception:
        lab = None

    def owns(p):
        if lab is None:
            return 0
        c_ = geo[p][0]
        x = int(c_[0] * lab.shape[1] / max(1.0, float(tracker.cut.info.width)))
        y = int(c_[1] * lab.shape[0] / max(1.0, float(tracker.cut.info.height)))
        return int(0 <= x < lab.shape[1] and 0 <= y < lab.shape[0] and int(lab[y, x]) == p)

    out = set()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if a in out or b in out:
                continue
            (ca, wa), (cb, wb) = geo[a], geo[b]
            if min(wa, wb) <= 0 or max(wa, wb) / min(wa, wb) > 1.6:
                continue
            if float(np.hypot(*(ca - cb))) > DOUBLE_DIST * max(wa, wb):
                continue
            ka = (owns(a), float(st.face_vis[a, t]))
            kb = (owns(b), float(st.face_vis[b, t]))
            out.add(b if ka >= kb else a)
    return out


def _nbytes(m) -> int:
    """Memory of a cached mask, or of a frame's score maps (a dict of them)."""
    if isinstance(m, dict):
        return int(sum(int(getattr(x, "nbytes", 0)) for x in m.get("maps", {}).values()) + int(getattr(m.get("thumb"), "nbytes", 0)))
    return int(getattr(m, "nbytes", 0))


_CACHE_OWN = ("__bytes__", "__tok__", "__new__")     # a frame cache's own entries: its byte count, what its kept
                                                    # entries were made from, and that it was just started over
_MISS = object()
_CACHE_LOCK = threading.RLock()                     # the people of a frame are prepared side by side (PREP_THREADS): the frame
                                                    # cache's book-keeping (its byte count, its tokens, what it lets go of) is
                                                    # changed by one of them at a time


def _cache_look(cache: Optional[dict], key, tok=None):
    """What a frame cache keeps under key, or _MISS. tok: what the entry must have been made from (_body_tok): one
    made from something else is not there."""
    if cache is None:
        return _MISS
    with _CACHE_LOCK:
        if key not in cache:
            return _MISS
        if tok is not None and (cache.get("__tok__") or {}).get(key) != tok:
            return _MISS
        return cache[key]


def _cache_put(cache: Optional[dict], key, m, tok=None) -> None:
    """m kept under key in a frame cache (prepare's), within its memory. tok: what it was made from, kept with it
    (an entry with one outlives a tracking update: kept_over_tracks)."""
    if cache is None:
        return
    with _CACHE_LOCK:
        nb = cache.get("__bytes__", 0)
        toks = cache.get("__tok__")
        if key in cache:                                # made again (the tracks changed where it reaches): in its place
            nb -= _nbytes(cache.pop(key))
            if toks:
                toks.pop(key, None)
        cache[key] = m
        if tok is not None:
            if toks is None:
                toks = cache["__tok__"] = {}
            toks[key] = tok
        # bounded by memory, not by count: at a 1920 proxy every mask is 8 MB, and 400 of them took
        # over 3 GB in the preview
        nb += _nbytes(m)
        n = len(cache) - sum(1 for k_ in _CACHE_OWN if k_ in cache)
        while (n > 400 or nb > CACHE_BYTES) and n > 1:
            k0 = next(k_ for k_ in cache if k_ not in _CACHE_OWN)
            nb -= _nbytes(cache.pop(k0))
            if toks:
                toks.pop(k0, None)
            n -= 1
        cache["__bytes__"] = max(0, nb)


def kept_over_tracks(cache: Optional[dict]) -> dict:
    """The frame cache to go on with after the tracks changed (a joint moved, a person swapped, a range solved
    again): a new one holding only the entries that were kept with what they were made from, the bodies' areas and
    spots. Whether each is still good its next use says (_body_tok: the tracks of the frames it reaches, as they
    are then), so a tracking update makes the body's maps again on the frames it reaches and nowhere else
    (instruction 2026-11-20 §2). Everything else is made again, as it was with an empty cache."""
    out: dict = {}
    try:
        toks = list(((cache or {}).get("__tok__") or {}).items())
        kept = {}
        nb = 0
        for k_, tk in toks:
            m = cache.get(k_, _MISS)
            if m is _MISS:
                continue
            out[k_] = m
            kept[k_] = tk
            nb += _nbytes(m)
        if out:
            out["__tok__"] = kept
            out["__bytes__"] = nb
            out["__new__"] = True
    except Exception:
        return {}
    return out


def _body_tok(tracker, st, p: int, t: int, reach: int = 0, extra=None) -> tuple:
    """What a body's area and spots of person p (store index) on frame t were made from besides the picture, short
    enough to compare at every use: what the tracks say of the frames they read (bodyskin.track_token: the frames
    round t that the body's size is taken over; reach: and the frames either side a steadied one is made from), and
    the colour of the person's face in this shot that the area starts from."""
    r = BS.REF_FRAMES + int(reach)
    mt = 0
    try:
        m = SK.model_for(tracker, p)
        if m is not None:
            import zlib
            mt = zlib.crc32(np.concatenate([np.asarray(x, np.float64).ravel() for x in m]).tobytes())
    except Exception:
        mt = -1
    return (BS.track_token(st, t - r, t + r), mt, extra)


PREP_THREADS = 1           # people of one frame whose masks are made side by side (_prepare_raw: the face's maps and masks, the body's area, the
                           # spots: each person's own, from the picture and the tracks). 1: one after the other, as ever. A batch sets it where the
                           # computer has cores to spare: the masks of one picture were the one thing a render did on a single core (j2815)
BDET_PEOPLE = 4            # people of one frame whose body spots are looked for side by side, each on their own piece
                           # of the original (looking is most of what a frame of an export costs once the areas are
                           # made on the people's own pieces: 0.3 to 1 s a person, one after the other: j2796).
                           # 1: one after the other, as 1.23's first build


def _bdet_people(cut, px, t: int, later: list, cache: Optional[dict]) -> None:
    """The body spots of the jobs in later [(job, cache key, the frame's picture, what they are made from)]: found,
    set on the jobs and kept.
    Several people: side by side (the same lists, in the people's order: each looks at their own pixels)."""
    failed = object()

    def one(e):
        try:
            with _tm("body.spots"):
                return _bdet_raw(cut, px, t, e[0], e[2])
        except Exception:
            import traceback
            traceback.print_exc()
            return failed
    if len(later) > 1:
        if DET_SRC:
            try:                                    # (the piece reader is made once, before the threads ask for it)
                from ..core import frames as FRM
                from ..warp.render import _vs_for
                FRM.source_crops(px, vs=_vs_for(cut))
            except Exception:
                pass
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(BDET_PEOPLE, len(later))) as ex:
            got = list(ex.map(one, later))
    else:
        got = [one(later[0])]
    for (job, key, _rgb, tok), d in zip(later, got):
        if d is failed:
            continue
        job.bdet = d
        _cache_put(cache, key, d, tok)


def _prime(tracker, st, px, t: int, people) -> None:
    """What the people of a frame all ask the tracker for, asked once, here, before they are made side by side (PREP_THREADS): the frame's owners, its
    picture and matte and clothing map, the face layer, the colour model of each person present. What the tracker makes on its first ask is then made
    once, by this thread, and the people only find it. Nothing here changes what anybody gets: the same asks, made a moment earlier."""
    def ask(fn):
        try:
            fn()
        except Exception:
            pass
    ask(lambda: tracker.labels(t))
    ask(lambda: tracker.matte.get(t))
    ask(lambda: px.frame(t))
    ask(lambda: _prime_face_layer(tracker))
    ask(lambda: _prime_cloth(tracker, t))
    for q in people:
        try:
            if st.has(q.id) and st.present[st.index(q.id), t]:
                SK.model_for(tracker, st.index(q.id))
        except Exception:
            pass


def _prime_face_layer(tracker) -> None:
    from ..track import facelayer as FL
    FL.load(tracker)


def _prime_cloth(tracker, t: int) -> None:
    from ..track import skincloth as SC
    SC.get(tracker, t)


def _prepare_raw(cut, tracker, t: int, cache: Optional[dict] = None, only: Optional[int] = None) -> Optional[RetouchFrame]:
    """One frame's masks on their own (see prepare). only: just this person, skin shown even at 0."""
    st = tracker.store
    if st is None:
        return None
    px = tracker.proxy
    jobs = []
    bdet_later = []                 # (job, cache key, the frame's picture): body spots still to be looked for
    people = cut.targets() if only is None else [q for q in cut.persons if q.id == only]
    doubles = _double_faces(st, tracker, t)

    def one_person(person):
        """One person's job of this frame: (the job or None, [(job, cache key, the frame's picture, what it is made from): body spots still to be
        looked for]). A person's masks are made from the picture, the tracks and the caches (kept under the person's own keys) and from nothing
        another person makes, so the people of a frame can be made side by side (PREP_THREADS)."""
        bdet_here = []
        if not st.has(person.id):
            return None, bdet_here
        p = st.index(person.id)
        if not st.present[p, t]:
            return None, bdet_here
        pp = cut.params_for(person.id)
        v = {k: pp.amount(k, t) / 100.0 for k in RETOUCH_KEYS}
        if only is not None:
            v["skin"] = max(v["skin"], 0.5)
        if not any(abs(x) > 1e-4 for x in v.values()):
            return None, bdet_here
        face = None
        fw = 0.0
        fw_doubt = False
        if st.has_face and np.isfinite(st.face[p, t, :, :2]).all() and p not in doubles:
            face = st.face[p, t]
            # face visibility as a strength: full strength once the face is reasonably sure, so the retouch does
            # not breathe with small confidence changes (a zoom, a turn); it still fades out where the face is
            # really doubtful (heavily covered, carried by motion alone for many frames)
            fw = float(np.clip((float(st.face_vis[p, t]) - VIS_LO) / (VIS_FULL - VIS_LO), 0.0, 1.0))
            fw_doubt = fw < 1.0
            if fw < 1.0 and VIS_PARSE:
                # the tracker's visibility drops on the frames next to a cover (cut 086: a head passes in front, and
                # the two frames before it and the ones after it came out raw or half strength although the face is
                # in plain view). The face parser looks at the frame itself: where it is sure of this face's skin
                # over a good part of the outline, the face is visible
                fw = max(fw, _parse_visible(tracker, person.id, t, face, px))
        sc = float(st.scale[p, t]) if np.isfinite(st.scale[p, t]) else 0.0
        job = Job(p, person.id, v, float(np.clip(pp.skin_detail / 100.0, 0, 1)), pp.skin_scope,
                  float(pp.grain_size), face, fw, sc,
                  blemish_size=float(np.clip(getattr(pp, "blemish_size", 50.0) / 100.0, 0, 1)),
                  body_keep=float(np.clip(getattr(pp, "body_detail", 60.0) / 100.0, 0, 1)),
                  body_size=float(np.clip(getattr(pp, "body_blemish_size", 40.0) / 100.0, 0, 1)))
        fth = pp.feather_factor() if hasattr(pp, "feather_factor") else 1.5
        if face is not None:
            job.wf_src = float(np.hypot(*(face[16, :2] - face[0, :2])))
            if SHARP_ADAPT and getattr(SK, "TEX_FS", False) and v.get("skin", 0.0) > 1e-4:
                job.sharp_q = _sharp_q(tracker, st, p, t, px)
        shape = (px.h, px.w)
        lm = None
        if face is not None:
            lm = face.copy()
            lm[:, 0] *= px.sx
            lm[:, 1] *= px.sy
        state = {}

        def body_base():
            """What the body's area needs of the frame: the picture and the face's colour model of it (none without
            a face). Not the face's own maps: a job with nothing but body values on never asks for them (they were
            0.6 to 0.9 s a person of a 2560 px picture, a quarter of an export's time: j2796). The same picture and
            the same model base() has when the face's values are on as well."""
            if "rgb" in state:
                return state
            if "bbase" not in state:
                rgb_ = np.asarray(px.frame(t))
                from ..track import facelayer as FL
                layer_ = FL.load(tracker)
                centre_ = layer_.colour(person.id, t) if (layer_ is not None and lm is not None) else None
                state["bbase"] = dict(rgb=rgb_, model=SK.frame_model_for(tracker, p, t, rgb_, lm, centre_))
            return state["bbase"]

        def base():
            if "rgb" not in state:
                state["rgb"] = state["bbase"]["rgb"] if "bbase" in state else np.asarray(px.frame(t))
                state["rgbf"] = state["rgb"].astype(np.float32) / 255.0
                from ..track import facelayer as FL
                layer = FL.load(tracker)
                fmaps = layer.maps(person.id, t, shape) if (layer is not None and lm is not None) else None
                centre = layer.colour(person.id, t) if (layer is not None and lm is not None) else None
                if fmaps is None and lm is not None and PARSE_NOW:
                    fmaps = _parse_now(state["rgb"], lm)
                state["parse"] = fmaps
                state["model"] = SK.frame_model_for(tracker, p, t, state["rgb"], lm, centre)
                state["prob"], state["protect"] = SK.face_skin_and_protect(state["rgb"], state["model"], lm if fw > 0.3 else None,
                                                                          fmaps if fw > 0.3 else None)
                state["prob0"] = state["prob"]      # before the cover and the focus zone: what the skin mask starts from too
                if lm is not None:
                    # an arm, a leg or a hand in front of the face: skin coloured, but not this face's skin
                    try:
                        cv_ = limb_cover(st, t, lm, shape, px.sx, px.sy)
                    except Exception:
                        cv_ = None
                    if cv_ is not None and fmaps is not None:
                        # the face parser leaves anything in front of the face out of its skin (CelebAMask-HQ
                        # marks occluders as background): where it is sure of face skin, no limb is in front.
                        # A cheek resting on a hand, a head lying on an arm put the limb's capsule over the
                        # whole face and the face lost its retouch (cut 095: 88 % of the skin gone)
                        # ... but the parser also calls a forearm across the face or fingers at the mouth skin
                        # (cuts 032, 033, 035). A limb in front hides part of the face; one that would hide most
                        # of a face whose landmarks are all found is a wrong limb, and only then the parser wins
                        sure = np.clip((fmaps["skin"] - 0.5) / 0.35, 0.0, 1.0) * float(fmaps.get("trust", 1.0))
                        sk_ = float(sure.sum())
                        hid = float((sure * cv_).sum()) / max(sk_, 1.0)
                        if sk_ > 50 and hid > LIMB_FALSE:
                            cv_ = (cv_ * (1.0 - sure)).astype(np.float32)
                    if cv_ is not None:
                        state["prob"] = state["prob"] * (1.0 - cv_)
                    state["cover"] = cv_
                state["alpha"] = SK.person_alpha(tracker, t, p, (px.w, px.h))
                if state["alpha"] is not None and lm is not None and fw > 0.5:
                    try:
                        state["alpha"] = face_owned(st, t, p, lm, state["alpha"], shape, px.sx, px.sy)
                    except Exception:
                        pass
                if MATTE_PARSE and state["alpha"] is not None and lm is not None and fw > 0.3 and fmaps is not None \
                        and float(fmaps.get("trust", 0)) > 0:
                    # the person's matte can miss its own face (cut 086, first frame: the matte gave the face to
                    # the head passing in front, a fifth of it was left). Inside the face's own outline, what the
                    # parser is sure is this face's skin belongs to the person
                    try:
                        own = SK.face_polygon(lm, SK.FOREHEAD_PARSED)
                        if own is not None:
                            oz = np.zeros(shape, np.uint8)
                            cv2.fillPoly(oz, [np.round(own).astype(np.int32)], 255)
                            Wo = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
                            oz = cv2.GaussianBlur(oz.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.02 * Wo))
                            sure = np.clip((fmaps["skin"].astype(np.float32) - 0.6) / 0.25, 0, 1) * float(fmaps["trust"])
                            state["alpha"] = np.maximum(state["alpha"], oz * sure).astype(np.float32)
                    except Exception:
                        pass
                if FOCUS_ZONE and fw_doubt and lm is not None and fw > 0.3:
                    fz = _focus_zone(state["rgb"], lm)
                    if fz is not None:
                        state["prob"] = (state["prob"] * fz).astype(np.float32)
                        if state["alpha"] is not None:
                            state["alpha"] = (state["alpha"] * fz).astype(np.float32)
                        state["focus_zone"] = fz
            return state

        def get(name, fn, extra="", tok=None):
            key = (t, person.id, name, extra)
            got = _cache_look(cache, key, tok)
            if got is not _MISS:
                return got
            m = fn()
            _cache_put(cache, key, m, tok)
            return m

        from ..core.project import scope_set
        regions = scope_set(job.scope)

        arm_on = ARMPIT > 0 and (ARMPIT_PIDS is None or person.id in ARMPIT_PIDS)

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
                                   b_.get("parse"), protect=b_.get("protect"))
                br = body_regions(regions, kpb, hd, shape, px.sx, px.sy, nk, fth)
                state["body"] = br
            return state["body"]

        def armpit():
            """The armpit skin of this person on this frame (0..1), or None."""
            if not arm_on:
                return None
            if "armpit" not in state:
                state["armpit"] = None
                try:
                    kpa, _wa = _steady_body(st, p, t)
                    zinfo_ = []
                    zones = armpit_region(kpa, shape, px.sx, px.sy, parts=True, info=zinfo_)
                    # only the armpits that are open (ARMPIT_OPEN): the arm held away from the body. Closed, there is no fold to take out
                    wts_ = None
                    if zones and ARMPIT_OPEN is not None:
                        wts_ = armpit_open_weights(st, p, t, px.sx, px.sy, ARMPIT_OPEN, int(ARMPIT_OPEN_SPAN))
                    try:
                        import os as _os_dbg
                        if _os_dbg.environ.get("IWALY_ARMPIT_DEBUG"):
                            print("ARMPIT zones t=%d pid=%d: open weights %s, zones (side, centre, radius, shoulder width) %s" % (
                                t, int(person.id), None if wts_ is None else [round(x, 2) for x in wts_], [(i_[0], (round(i_[1][0]), round(i_[1][1])), round(i_[2]), round(i_[3])) for i_ in zinfo_]), flush=True)
                    except Exception:
                        pass
                    if zones and ARMPIT_LINE_HOLD > 1e-4:
                        # the hanging arms among the open ones: their own line is left as it is (ARMPIT_LINE_HOLD)
                        try:
                            job.arm_axes = armpit_hold_axes(st, p, t, px.sx, px.sy, zinfo_, wts_, int(ARMPIT_OPEN_SPAN)) or None
                        except Exception:
                            import traceback
                            traceback.print_exc()
                    if zones:
                        b_ = base()
                        ap = None
                        al_ = b_["alpha"]
                        try:
                            # people lying against each other: the matte hands this person's shoulder to the one
                            # beside (cut 055). Anybody's matte counts in the zone, other people's faces left out
                            ma_ = tracker.matte.get(t)
                            if ma_ is not None:
                                ma_ = cv2.resize(np.asarray(ma_).astype(np.float32) / 255.0, (px.w, px.h), interpolation=cv2.INTER_LINEAR)
                                oth = np.zeros(shape, np.uint8)
                                for q_ in np.nonzero(st.present[:, t])[0]:
                                    if q_ == p or not (st.has_face and np.isfinite(st.face[q_, t, :, :2]).all()):
                                        continue
                                    lq = st.face[q_, t].copy(); lq[:, 0] *= px.sx; lq[:, 1] *= px.sy
                                    pq = SK.face_polygon(lq, SK.FOREHEAD_PARSED)
                                    if pq is not None:
                                        cv2.fillPoly(oth, [np.round(pq).astype(np.int32)], 255)
                                        wq = float(np.hypot(*(lq[16, :2] - lq[0, :2])))
                                        kq = max(3, int(0.12 * wq) | 1)
                                        oth = cv2.dilate(oth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kq, kq)))
                                ma_ = ma_ * (1.0 - cv2.GaussianBlur(oth.astype(np.float32) / 255.0, (0, 0), 2.0))
                                al_ = ma_ if al_ is None else np.maximum(al_, ma_)
                        except Exception:
                            pass
                        for zi_, zone in enumerate(zones):
                            if wts_ is not None and (zi_ >= len(wts_) or wts_[zi_] <= 1e-3):
                                continue
                            a1 = armpit_skin(b_["rgb"], zone, al_, b_["prob"])
                            if a1 is not None:
                                if wts_ is not None:
                                    a1 = (a1 * np.float32(wts_[zi_])).astype(np.float32)
                                ap = a1 if ap is None else np.maximum(ap, a1)
                        if ap is not None:
                            state["armpit"] = (ap * float(min(1.0, ARMPIT))).astype(np.float32)
                            # a person whose face is not in the picture: the stage's size reference from its zone (source px)
                            try:
                                rs_ = [zi[2] for zi_i, zi in enumerate(zinfo_) if wts_ is None or wts_[zi[0]] > 1e-3]
                                if rs_ and px.sx > 0:
                                    job.wf_arm = float(ARMPIT_FACE_PER_RADIUS) * max(rs_) / float(px.sx)
                            except Exception:
                                pass
                except Exception:
                    import traceback
                    traceback.print_exc()
            return state["armpit"]

        def armpit_fold():
            """The armpit skin the fold stage works on: armpit() without this person's hair and garments (ARMPIT_HAIR_REACH), or None."""
            ap0 = armpit()
            if ap0 is None:
                return None
            if "armpit_fold" not in state:
                state["armpit_fold"] = ap0
                try:
                    wfp_ = (float(job.wf_src) if job.wf_src > 0 else float(job.wf_arm)) * float(px.sx)
                    state["armpit_fold"] = _armpit_without_hair(ap0, tracker, t, p, base()["rgb"], shape, wfp_)
                except Exception:
                    import traceback
                    traceback.print_exc()
            return state["armpit_fold"]

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
                m_ = SK.skin_mask(sj, b["rgb"], px.sx, px.sy, b["alpha"], b["model"], b.get("parse"), body(),
                                  b.get("cover"), prob0=b.get("prob0"))
                ap_ = armpit()
                if ap_ is not None:
                    m_ = np.maximum(m_, ap_)
                    # the steadied masks go by what this frame takes for skin: the armpit counts
                    b["prob"] = np.maximum(b["prob"], ap_)
                return m_
            m = get("skin", mk_skin, job.scope + (":armpit" if arm_on else ""))
            if m.max() > 0.01:
                job.skin = sj
                job.masks["skin"] = m
            if arm_on and ARMPIT_FOLD > 1e-4:
                try:
                    ap2_ = armpit_fold()
                    if ap2_ is not None and float(ap2_.max()) > 0.05:
                        job.masks["armpit"] = ap2_
                except Exception:
                    pass
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
            def paint():
                if "paint" not in state:
                    state["paint"] = SK.paint_mask(base()["rgb"], lm)
                return state["paint"]

            def pa():
                b = base()
                return b["prob"] * (b["alpha"] if b["alpha"] is not None else 1.0)
            if v.get("dark_circle", 0) > 1e-4:
                job.masks["dark"] = get("dark", lambda: under_eye_mask(lm, shape, pa(), fth), f"{fth:.2f}") * fw
            if v.get("shine", 0) > 1e-4:
                job.masks["hot"] = get("hot", lambda: hot_zone_mask(lm, shape, pa(), fth), f"{fth:.2f}") * fw
            if v["nasolabial"] > 1e-4:
                job.masks["naso"] = get("naso", lambda: wrinkle_masks(lm, shape, pa(), "naso", fth, 1.7, paint()),
                                        f"{fth:.2f}:w") * fw
            if v["mouth_lines"] > 1e-4:
                job.masks["mouthl"] = get("mouthl", lambda: wrinkle_masks(lm, shape, pa(), "mouth", fth, 1.5,
                                                                        paint()), f"{fth:.2f}:w") * fw
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
                                     b_["rgb"], hnd, b_.get("parse"), protect=b_.get("protect"))
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
        if any(v.get(k_, 0) > 1e-4 for k_ in BODY_EFFECTS):
            # body skin (1.23): its own area, apart from the face's regions (retouch.bodyskin)
            try:
                from ..core.project import body_scope_set, norm_body_scope
                bscope = norm_body_scope(getattr(pp, "body_scope", "all"))
                if v.get("knee_tone", 0) > 1e-4 or v.get("belly_fold", 0) > 1e-4:
                    # (from the tracks alone: the knee tone reads the leg's skin from the picture itself where the body
                    # skin's area is not there)
                    # (the legs' widths are the median of the neighbouring pictures': the measure of one picture jumps, the leg does not)
                    job.tone = BTN.person_geometry_t(st.final[p], st.widths[p] if getattr(st, "widths", None) is not None else None, int(t))

                def mk_body():
                    with _tm("body.base"):
                        b = body_base()
                    with _tm("body.area"):
                        m_, info_ = BS.body_mask(tracker, p, t, b["rgb"], body_scope_set(bscope), b.get("model"))
                    pm_ = BS.pack_mask(m_) if m_ is not None else None
                    pa_ = BS.pack_allow(info_.allow) if pm_ is not None else None
                    maps_ = {"m": pm_["m"]} if pm_ is not None else {}
                    if pa_ is not None:
                        maps_["a"] = pa_["m"]
                    return dict(maps=maps_, thumb=None, pm=pm_,
                                info=dict(ref=float(info_.ref), model=info_.model, parsed=bool(info_.parsed),
                                          box=info_.box, allow=pa_, hands=BS.hand_shapes(st, p, t, info_)))
                # the clothing the analysis found is part of the area: masks made before it was found (or with
                # another state of it) are made again
                ctok = BS.garment_token(tracker)
                # ... and so are the tracks: an area is kept with what they said of the frames it was made from, and
                # is made again when that changed (a tracking update reaches the frames round it, not the cut)
                btok = _body_tok(tracker, st, p, t)
                bd = get("bskin", mk_body, f"{bscope}:{ctok}", btok)
                if bd["pm"] is not None and tuple(bd["pm"]["shape"]) == tuple(shape) and bd["info"]["ref"] > 0:
                    job.body = dict(bd["info"], mask=bd["pm"], key=f"{bscope}:{ctok}")
                    if v.get("body_blemish", 0) > 1e-4:
                        def mk_bdet():
                            with _tm("body.spots"):
                                return _bdet_raw(cut, px, t, job, body_base()["rgb"])
                        bkey = f"{bscope}:{ctok}:{job.body_size:.3f}:sn{int(BP.SPOTNET)}h{int(BP.HANDS_HELD)}"
                        if BDET_PEOPLE > 1 and _cache_look(cache, (t, person.id, "bdet", bkey), btok) is _MISS:
                            # looked for after the loop, every person of the frame side by side (_bdet_people)
                            bdet_here.append((job, (t, person.id, "bdet", bkey), body_base()["rgb"], btok))
                        else:
                            job.bdet = get("bdet", mk_bdet, bkey, btok)
                        job.bmarks = _body_marks(st, p, t, pp, v)
            except Exception:
                import traceback
                traceback.print_exc()
        if v["knee_dark"] > 1e-4:
            kp = st.final[p, t]

            def mk_knee():
                b = base()
                return knee_mask(kp, st.widths[p, t], shape, px.sx, px.sy, b["prob"], b["alpha"])
            km = get("knee", mk_knee)
            if km.max() > 0.01:
                job.masks["knee"] = km
        if DET_ON and job.masks and lm is not None and _det_wanted(v):
            try:
                job.det = get("det", lambda: _det_raw(cut, px, t, job, lm, base()["rgb"]),
                              f"{job.blemish_size:.3f}:{int(v.get('blemish', 0) > 1e-4)}{int(v.get('neck_lines', 0) > 1e-4)}{int(v.get('neck_bumps', 0) > 1e-4)}:{fth:.2f}"
                              f":sn{int(_spot_net_wanted(job))}")
            except Exception:
                import traceback
                traceback.print_exc()
                job.det = None
        if job.masks or job.body:
            if "prob" in state:
                # this frame's own view of what is skin (colour, parser, texture): the steadied masks may not
                # carry a neighbour's mask onto something that covers the face here (a hand, a sleeve)
                pr_ = state["prob"]
                if state.get("alpha") is not None:
                    pr_ = pr_ * state["alpha"]
                job.veto = cv2.resize(pr_, (max(1, pr_.shape[1] // 4), max(1, pr_.shape[0] // 4)),
                                      interpolation=cv2.INTER_AREA).astype(np.float16)
            return job, bdet_here
        return None, bdet_here

    n_thr = max(1, min(int(PREP_THREADS), len(people)))
    if n_thr > 1:
        _prime(tracker, st, px, t, people)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_thr, thread_name_prefix="prep") as ex:
            made = list(ex.map(one_person, people))             # (in the people's order: what comes out is what one after the other gives)
    else:
        made = [one_person(q) for q in people]
    for job_, later_ in made:
        if job_ is not None:
            jobs.append(job_)
        bdet_later.extend(later_)
    if bdet_later:
        _bdet_people(cut, px, t, bdet_later, cache)
    if not jobs:
        return None
    return RetouchFrame(t, jobs, (px.w, px.h), (cut.info.width, cut.info.height))


# --------------------------------------------------------------------------- the knee tone through time (1.26)

def knee_probe(rf: Optional["RetouchFrame"], px) -> Dict[tuple, float]:
    """The support of each knee of a prepared picture in its person's body skin area (bodytone.knee_support): {(person id, 'l' | 'r'): 0..1}. A person
    without an area (the knee tone then reads the leg's own strip) is not limited: 1."""
    out: Dict[tuple, float] = {}
    if rf is None:
        return out
    sc = 0.5 * (float(px.sx) + float(px.sy))
    for job in rf.jobs:
        tone = getattr(job, "tone", None)
        if not tone or not tone.get("knees"):
            continue
        sides = tone.get("sides") or []
        pm = (job.body or {}).get("mask") if getattr(job, "body", None) else None
        m = None
        if pm is not None:
            try:
                m = BS.unpack_mask(pm)
            except Exception:
                m = None
        sup = BTN.knee_support(m, sc, tone["knees"]) if m is not None else [1.0] * len(tone["knees"])
        for i, v_ in enumerate(sup):
            out[(job.pid, sides[i] if i < len(sides) else str(i))] = float(v_)
    return out


def knee_gains_apply(rf: Optional["RetouchFrame"], gain_of) -> None:
    """Lay the strength each knee of a prepared picture may be corrected with on it: gain_of(person id, side) -> 0..1."""
    if rf is None:
        return
    for job in rf.jobs:
        tone = getattr(job, "tone", None)
        if not tone or not tone.get("knees"):
            continue
        sides = tone.get("sides") or []
        tone["gain"] = [float(gain_of(job.pid, sides[i] if i < len(sides) else str(i))) for i in range(len(tone["knees"]))]


class KneeLook:
    """The render's look ahead for the knee tone. The pictures of a cut are prepared here, in order, in a thread of their own, a few ahead of the pictures
    that are worked on (the work of preparing is the same as ever, it is only made a little earlier, and once); every prepared picture is looked at for the support
    its knees have (knee_probe), and a picture is handed to the work only when the pictures round it have been looked at: its knees are given the strength
    bodytone.knee_envelope allows them. get(t) -> (RetouchFrame or None, the field input) for the picture t, as prepare and build_input would give.

    prepare, build_input: callables (t) -> ..., lock: the one the render holds while it prepares, a, b: the pictures [a, b) that are worked on, px: the
    analysis picture's scale (tracker.proxy), lead: how many pictures it may be ahead of the first not yet handed on (at least reach + 1)."""

    def __init__(self, prepare, build_input, lock, a: int, b: int, px, step: Optional[float] = None, lead: Optional[int] = None):
        import threading
        self.prepare, self.build_input, self.lock, self.a, self.b, self.px = prepare, build_input, lock, int(a), int(b), px
        self.step = step
        self.reach = BTN.env_reach(step)
        self.lead = max(int(lead if lead is not None else self.reach + 12), self.reach + 2)
        self.cv = threading.Condition()
        self.items: Dict[int, tuple] = {}
        self.sup: Dict[int, dict] = {}
        self.done = self.a - 1
        self.low = self.a
        self.taken: set = set()
        self.err: Optional[BaseException] = None
        self.stopped = False
        self.th = threading.Thread(target=self._run, name="knee-look", daemon=True)

    def start(self) -> "KneeLook":
        self.th.start()
        return self

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def _run(self) -> None:
        try:
            for t in range(self.a, self.b):
                with self.cv:
                    while not self.stopped and t - self.low > self.lead:
                        self.cv.wait(0.5)
                    if self.stopped:
                        return
                with self.lock:
                    rf = self.prepare(t)
                    fi = self.build_input(t)
                sup = knee_probe(rf, self.px) if rf is not None else {}
                with self.cv:
                    self.items[t] = (rf, fi)
                    self.sup[t] = sup
                    self.done = t
                    self.cv.notify_all()
        except BaseException as e:        # (handed to whoever is waiting: the render stops with it as it would have)
            with self.cv:
                self.err = e
                self.cv.notify_all()

    def get(self, t: int):
        need = min(t + self.reach, self.b - 1)
        with self.cv:
            while self.done < need and self.err is None and not self.stopped:
                self.cv.wait(0.5)
            if self.err is not None:
                raise self.err
            if self.stopped:
                raise RuntimeError("knee look stopped")
            rf, fi = self.items.pop(t)
            lo, hi = self.a, self.b - 1
            window = {u: self.sup.get(u, {}) for u in range(max(lo, t - self.reach), min(hi, t + self.reach) + 1)}
            self.taken.add(t)
            while self.low in self.taken:
                self.taken.discard(self.low)
                self.sup.pop(self.low - self.reach - 1, None)
                self.low += 1
            self.cv.notify_all()
        if rf is not None:
            def gain_of(pid, side):
                return BTN.knee_envelope({u: w.get((pid, side), 0.0) for u, w in window.items()}, t, lo, hi, self.step)
            knee_gains_apply(rf, gain_of)
        return rf, fi


# --------------------------------------------------------------------------- operations

def _roi(m: np.ndarray, pad: int):
    sel = m > 0.004
    ys = np.flatnonzero(sel.any(1))
    if not len(ys):
        return None
    xs = np.flatnonzero(sel.any(0))
    h, w = m.shape
    return (max(0, int(ys[0]) - pad), min(h, int(ys[-1]) + pad + 1), max(0, int(xs[0]) - pad), min(w, int(xs[-1]) + pad + 1))


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


FOLD_OWN = 1.0        # a fold is filled up to this share of its own depth (1.25 overshot: on a face screaming or laughing
                      # the deep fold beside the mouth came out as a pale stripe, cut 097)
MOUTH_OPEN = (0.30, 0.65)   # mouth opening (gap between the lips over the mouth's width) from which the folds beside the
MOUTH_OPEN_KEEP = 0.4       # mouth are expression: at the widest only this share of the fold filling is left


def mouth_open(lm: Optional[np.ndarray]) -> float:
    """0 closed .. 1 wide open (a scream, a big laugh), from the inner lip landmarks."""
    try:
        L = np.asarray(lm, np.float32)[:, :2]
        if not np.isfinite(L[[48, 54, 62, 66]]).all():
            return 0.0
        w = float(np.hypot(*(L[54] - L[48])))
        g = float(np.hypot(*(L[66] - L[62])))
        return float(_ss(MOUTH_OPEN[0], MOUTH_OPEN[1], np.array(g / max(w, 1e-3))))
    except Exception:
        return 0.0


def _fold_fill(I: np.ndarray, m: np.ndarray, excl: Optional[np.ndarray], k: float, sig_f: float, sig_s: float,
               luma: bool = True, valley: float = 0.0, cap: float = 0.25) -> np.ndarray:
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
    if luma and valley > 0:
        # never more than the crease's own depth against the skin right beside it: the broad shading of a
        # cheek in side light is the face's modelling, not a fold
        own = cv2.GaussianBlur(roi, (0, 0), valley) - loc
        d = np.minimum(d, np.clip(own, 0, None) * FOLD_OWN)
    if luma:
        d = np.clip(d, 0.0, cap)
    else:
        d = np.clip(d, -0.035, 0.035)
    d = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.6 * sig_f))
    out = I.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + mm * k * d * ok
    return out


# 2611 cut 024 (user 2026-11-17: at 강하게 with 팔자주름·입가주름 제거 the folds stayed). On a young full face the fold
# is not a thin valley but a soft shading band from the nose wing to the mouth corner, a tenth of the face wide: the
# valley finder above sees nothing (the band lifted 4 %). This lifts what is darker than the skin around it at that
# width, inside the fold's band only, the lips and nostrils (and a margin round them) left alone.
FOLD_BROAD = 0.8            # share of the fold's value
FOLD_BROAD_SCALE = {"nasolabial": 0.12, "mouth_lines": 0.08}   # widest shading taken, share of the face width
FOLD_BROAD_CAP = 0.06


def _fold_broad(Y: np.ndarray, m: np.ndarray, excl: Optional[np.ndarray], wf: float, k: float, scale: float,
                cap: float = FOLD_BROAD_CAP) -> np.ndarray:
    """Lift the dark shading of a broad fold: a closing (black-hat) at `scale` of the face width finds what is darker
    than the skin either side of it; only that is added back, smoothly, inside m. Luma only, never darker."""
    if k < 1e-4:
        return Y
    r = _roi(m, int(scale * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1].astype(np.float32)
    Ys = cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.006 * wf))
    ks = max(3, int(scale * wf) | 1)
    bh = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bh = cv2.GaussianBlur(np.clip(bh, 0, cap), (0, 0), max(1.0, 0.01 * wf))
    if excl is not None:
        e = np.clip(excl[y0:y1, x0:x1].astype(np.float32), 0, 1)
        d_ = max(3, int(0.04 * wf) | 1)
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_, d_)))
        mm = mm * (1.0 - np.clip(cv2.GaussianBlur(e, (0, 0), max(1.0, 0.02 * wf)), 0, 1))
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * mm * bh
    return out


FOLD_LEVEL = 1.25          # 2611 cut 024 again (2026-11-18): at 강하게 the soft nasolabial shading stayed: on a young face
                           # the fold is a soft step on the cheek's gradient, which neither the valley fill nor the
                           # closing sees. Its middle scale (between FOLD_LEVEL_SIG) is evened out inside the band, the
                           # broad shading of the face and the fine texture kept; times the fold value (강하게 0.8 -> 1.0).
                           # A first version only lifted what was darker than the skin beside the band: around a
                           # dimple at the mouth corner (2611 cut 011) that left a dark notch, so lips and nostrils
                           # are kept out of the reference and the band fades out well before them.
FOLD_LEVEL_SIG = (0.008, 0.12)  # fine .. broad scale kept (share of the face width)
FOLD_LEVEL_CAP = (0.08, 0.008)  # most lift / most darkening (0..1 luma). 2611 IWALY c001 (2026-11-22, user): on a profile the fold's band
                                # runs over the cheek and the jaw, and evening out the middle scale pressed the cheek's soft highlight down by
                                # up to 1.7 % (0.03 here): the cheek looked darker. A fold is only ever lightened; the little darkening left
                                # is what keeps a dimple's edge from standing out
FOLD_LEVEL_EXCL = (0.08, 0.04)  # the band fades out this far from the lips and nostrils (dilation, blur; face width)
FOLD_LEVEL_EDGE = (0.02, 0.03)  # the same round hair strands and sharp edges in the band
# 2611 IWALY c029 f96, c010 f0 (2026-11-22, user, 팔자주름이 보정이 안 됨): on a young face the fold from the nose wing to the mouth corner is a soft step in
# the cheek's shading, 1 to 3 % deep: the fold's steps above (the valley fill, the closing, the evening of the middle scale) took 0.5 to 1 % of it (the
# middle scale's lift capped at 0.8 %, faded out over a tenth of the face round the nose and the lips, and round every edge). The nasolabial fold's own
# values, over the general ones (the mouth lines keep theirs): a stronger share of the broad steps, the evening's cap raised, the fades shorter.
# None = the general values for the nasolabial fold too.
NASO_FOLD = {"broad": 1.1, "broad_cap": 0.08, "level": 1.7, "excl": (0.05, 0.02), "edge": (0.012, 0.02), "cap": (0.10, 0.004)}
NECK_LEVEL = 1.25          # the same on the neck (cut 024: the rings and the broad band across the neck stayed)
NECK_LEVEL_SIG = (0.008, 0.10)
NECK_LEVEL_CAP = (0.08, 0.008)   # the neck's own cap of the evening (the nasolabial fold's is raised above: the neck keeps this)

# 2611 IWALY c029 (2026-11-22, user, 00:01:50:19): a black sleeve cuff under the chin lay inside the neck's mask (the colour model
# and the steadied masks let dark brown through) and the neck stages lifted it by a quarter of its own brightness, in the shape of the
# mask's straight edge. The neck stages now go by the picture too: where it is far darker than the skin the mask covers, they stop.
NECK_CLOTH_GATE = (0.36, 0.55)    # of the neck's own skin brightness (a high percentile of the picture under its mask): the stages fade out
                                  # between these. A neck in shade is 0.6 to 0.9 of its lit side, black cloth and hair under 0.45. None = off
NECK_CLOTH_Q = 75.0


def _skin_level(Y: np.ndarray, m: np.ndarray, q: float = NECK_CLOTH_Q) -> float:
    """The skin's brightness under m (0..1): a high percentile of Y where m is above a half; 0 where there is too little of it to say."""
    r = _roi(m, 6)
    if r is None:
        return 0.0
    y0, y1, x0, x1 = r
    sel = m[y0:y1, x0:x1] > 0.5
    if int(sel.sum()) < 40:
        return 0.0
    return float(np.percentile(cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.2)[sel], q))


def _skin_gate(Y: np.ndarray, m: np.ndarray, lo: float, hi: float, q: float = NECK_CLOTH_Q, ref_min: float = 0.0) -> Optional[np.ndarray]:
    """0..1 at Y's size: 0 where Y is far darker than the skin that m covers (a sleeve, hair, a dark top), 1 on the skin itself and
    wherever m has nothing. The skin's level is a high percentile of Y under the mask, so a mask that is half cloth still finds it.
    ref_min: the level is at least this (the person's own face when m may be all cloth)."""
    r = _roi(m, 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    sel = m[y0:y1, x0:x1] > 0.5
    if int(sel.sum()) < 40:
        return None
    Ys = cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.2)
    ref = max(float(np.percentile(Ys[sel], q)), float(ref_min))
    if ref < 0.02:
        return None
    out = np.ones(Y.shape, np.float32)
    out[y0:y1, x0:x1] = _ss(lo * ref, hi * ref, Ys)
    return out


# 2611 IWALY c029 (2026-11-22, user, 00:01:50:19, "mask error" at the black sleeve cuff under the chin) - the skin smoothing itself. Its bands are taken
# down by a factor wherever the mask reaches, and a black cuff against skin is a step as big as the skin's own brightness, with a soft edge (an out of
# focus cuff: 25 px from black to skin): the bands of the edge are taken down too, which blurs it (a lighter line along the black's edge, a darker band on
# the skin beside it, +2.2 % and -2.6 % of the brightness), and the mask (soft edged, steadied through time) reaches ten pixels into the cloth. Now,
# outside the face's outline (the face is held by its own maps and kept as it was), beside cloth and hair:
#  - the cloth's CORE is found by the picture (darker than the middle of SKIN_CLOTH_GATE of the skin's own brightness, wider than a few pixels: a strand
#    or a chain is not cloth), and the edge's RAMP: what lies within SKIN_CLOTH_REACH of the core and is darker than SKIN_CLOTH_RAMP of the skin's level beside it
#  - the smoothing is faded out there (zone, soft edged), and
#  - its input has the zone replaced by the skin's own level taken from the skin beside it (a blur weighted by what is not in the zone), faded in the same
#    way, so that there is no step or ramp for the bands to take down; what comes out is added to the picture as it was (the cloth keeps its own pixels).
# None = off
SKIN_CLOTH_GATE = (0.20, 0.38)    # of the skin's own brightness (a high percentile of the picture under the mask): the cloth's core is darker than the middle of these
SKIN_CLOTH_REACH = 0.016          # of the picture's height (35 px at 2160): how far from the core the ramp is looked for
SKIN_CLOTH_RAMP = 0.90            # the ramp: darker than this much of the skin's level beside the cloth
SKIN_CLOTH_THIN = 0.004           # of the face's width: cloth that is not wider than this (twice) is a strand or a chain, not cloth
SKIN_CLOTH_SOFT = 0.004           # of the picture's height: the fade of the zone's edge
SKIN_CLOTH_FILL = 0.01            # of the picture's height: the reach of the fill


def _cloth_zone(Y: np.ndarray, m: np.ndarray, wf: float) -> Optional[np.ndarray]:
    """0..1 at Y's size: 1 on the cloth (or hair) beside the skin m covers and on the ramp of its edge, soft edged, 0 on the skin; None where there is none"""
    r = _roi(m, 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    sel = m[y0:y1, x0:x1] > 0.5
    if int(sel.sum()) < 40:
        return None
    ref = float(np.percentile(cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.2)[sel], NECK_CLOTH_Q))
    if ref < 0.02:
        return None
    hh = Y.shape[0]
    reach = max(6, int(round(SKIN_CLOTH_REACH * hh)))
    sg = max(2.0, SKIN_CLOTH_SOFT * hh)
    pad = reach + int(3 * sg) + 4
    w0, w1 = max(0, y0 - pad), min(Y.shape[0], y1 + pad)
    v0, v1 = max(0, x0 - pad), min(Y.shape[1], x1 + pad)
    Ys = cv2.GaussianBlur(Y[w0:w1, v0:v1].astype(np.float32), (0, 0), 1.2)
    core = Ys < 0.5 * (SKIN_CLOTH_GATE[0] + SKIN_CLOTH_GATE[1]) * ref
    kt = max(1, int(round(SKIN_CLOTH_THIN * wf)))
    if kt > 1:
        core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kt + 1, 2 * kt + 1))) > 0
    if not core.any():
        return None
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    lit = _ss(0.80 * ref, 0.97 * ref, Ys)
    sl = max(8.0, 0.02 * hh)
    lvl = (cv2.GaussianBlur(Ys * lit, (0, 0), sl) + 0.02 * ref) / (cv2.GaussianBlur(lit, (0, 0), sl) + 0.02)     # the skin's level beside the cloth
    zone = (core | ((dist <= reach) & (Ys < SKIN_CLOTH_RAMP * lvl))).astype(np.float32)
    out = np.zeros(Y.shape, np.float32)
    out[w0:w1, v0:v1] = np.clip(2.0 * cv2.GaussianBlur(zone, (0, 0), sg), 0.0, 1.0)
    return out


def _cloth_input(I: np.ndarray, dk: np.ndarray) -> np.ndarray:
    """I with the cloth's zone dk replaced by the level of the skin beside it (a blur of I weighted by what is not in the zone), by dk. I itself where there is no zone."""
    bad = dk > 0.01
    ys = np.flatnonzero(bad.any(1))
    if not len(ys):
        return I
    xs = np.flatnonzero(bad.any(0))
    s = max(6.0, SKIN_CLOTH_FILL * I.shape[0])
    pad = int(3 * s) + 3
    y0, y1 = max(0, int(ys[0]) - pad), min(I.shape[0], int(ys[-1]) + pad + 1)
    x0, x1 = max(0, int(xs[0]) - pad), min(I.shape[1], int(xs[-1]) + pad + 1)
    Ic = I[y0:y1, x0:x1].astype(np.float32)
    dc = dk[y0:y1, x0:x1].astype(np.float32)
    w = (1.0 - dc) ** 3
    num = cv2.GaussianBlur(Ic * w, (0, 0), s)
    den = cv2.GaussianBlur(w, (0, 0), s)
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), Ic)
    out = I.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = (1.0 - dc) * Ic + dc * fill
    return out


# 2611 IWALY c001 (2026-11-22, user, 00:00:31:14, the green ellipse on the cheek: "it gets darker"). On a face turned to a profile or three quarters the landmarks' outline of the face
# (jaw line, forehead arc) lies out over the hair beside the cheek, and the cloth zone above was kept out of that outline: the hair's edge (hair 0.11, skin 0.35..0.5 of the picture's
# brightness, a step of the skin's own size) reached the skin smoothing as it is, whose bands are taken down across it: a band of skin along the hair, 10 to 20 px wide, came out 1.5 %
# darker (and the hair's edge softer). Now the zone is kept inside the outline too wherever the exterior (hair, the background, cloth: what the exclusion holds that is not a feature,
# and not wider than a strand or a brow: SKIN_CLOTH_EXT) is near, which is as far as the zone itself reaches (the core's ramp and its fade). Away from the exterior nothing changes: a
# dark thing inside the face (a brow, an eye, a nostril, a mole) is no cloth. False = the zone stops at the outline as before.
SKIN_CLOTH_INFACE = True
SKIN_CLOTH_EXT = 0.10        # of the face width: the exterior is what is wider than this (a brow is not: it is thinner; hair, the background and cloth are wider)


def _exterior_near(exY: Optional[np.ndarray], lm: Optional[np.ndarray], shape, y_per_src: float, wf: float, m: np.ndarray, near: float, soft: float) -> Optional[np.ndarray]:
    """0..1 at `shape`: 1 within `near` px of the face's exterior (hair, the background, cloth: what the exclusion exY holds that is not a feature (feature_exclusion), wider than
    SKIN_CLOTH_EXT of the face width), fading out over `soft` px beyond it; 0 elsewhere. Worked out round the skin mask m only. None where there is no exterior near it (or it cannot be made)."""
    if exY is None or lm is None:
        return None
    try:
        r = _roi(m, int(near + soft) + 6)
        if r is None:
            return None
        y0, y1, x0, x1 = r
        lmc = np.array(lm, np.float32, copy=True)
        if not np.isfinite(lmc[:, :2]).all():
            return None
        lmc[:, :2] = lmc[:, :2] * float(y_per_src) - np.array([x0, y0], np.float32)
        fx = feature_exclusion(lmc, (y1 - y0, x1 - x0))
        nonf = np.clip(np.clip(exY[y0:y1, x0:x1].astype(np.float32), 0, 1) - fx, 0.0, 1.0)
        ext = (nonf > 0.5).astype(np.uint8)
        ko = max(3, int(round(SKIN_CLOTH_EXT * wf)) | 1)
        ext = cv2.morphologyEx(ext, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
        if not bool(ext.any()):
            return None
        d = cv2.distanceTransform((1 - ext).astype(np.uint8), cv2.DIST_L2, 3)
        out = np.zeros(shape[:2], np.float32)
        out[y0:y1, x0:x1] = 1.0 - _ss(float(near), float(near) + float(soft), d)
        return out
    except Exception:
        return None


# 2611 IWALY c003, c010 (2026-11-22, user, 00:00:39:17 and 00:00:58:12): "the retouch of the neck is too strong, the neck's bones cannot be seen".
# The neck lines' broad steps (the flattening of the soft folds, the ring filling, the levelling, the found folds) work on the middle scale, and the
# neck's own build (the tendons, the collarbone, the hollow between them, the shadow under the chin) is middle scale too: on a small neck they took
# 1.5 to 3.5 % of the picture's brightness out of it (the structure's own contrast is 3 to 8 %), under a jaw in shade up to 9 %. Now
#  - these steps change a place only where the structure there lies ACROSS the neck, as the rings do (its brightness changes along the neck's axis):
#    the tendons run along it, a hollow or a bone's relief has no one direction, and both are left as they are (NECK_ACROSS: the cosine between the
#    neck's axis and the direction the brightness changes most in, from where .. in full; NECK_ACROSS_COH: the same for how one-directional the
#    structure is). None = off
#  - the whole of the neck lines' change of a pixel is capped (NECK_CAP: most lighter, most darker; luma 0..1), softly: a crease is 1 to 3 % deep
NECK_ACROSS = (0.45, 0.80)
NECK_ACROSS_COH = (0.20, 0.50)
NECK_CAP = (0.015, 0.006)
# ... and (2026-11-22, the same review, after the first fix: c010's neck still moved by up to 1.8 % and the shadow under the chin was lifted by 3 to 6 %):
#  - the broad steps are made at half of their value (NECK_BROAD): a neck of a young person has no soft rings to take out, and what the broad steps
#    find on it is its own modelling (the hollow above the collarbone, the shadow the chin throws). The thin creases are filled in full.
#  - the broad steps fade in below the jaw (NECK_UNDER_CHIN: from this distance from the jaw line to this one, in face widths; None = off): the
#    underside of a raised chin is a broad shadow, not a fold
NECK_BROAD = 0.5
NECK_UNDER_CHIN = (0.05, 0.30)


def jaw_under(lm: Optional[np.ndarray], shape, scale: float, wf: float, reach) -> Optional[np.ndarray]:
    """0 on the jaw line, 1 at reach[1] face widths from it (smooth in between; reach[0] is where it starts to rise): what is only to be changed
    away from the jaw. lm in source px, scale = px of `shape` per source px, wf in px of `shape`."""
    if lm is None or not np.isfinite(lm[2:15, :2]).all():
        return None

    def make():
        pts = np.round(lm[2:15, :2] * scale).astype(np.int32)
        far = reach[1] * wf
        x0, y0, x1, y1 = _window(shape, pts, far + 6)
        out = np.ones(shape[:2], np.float32)
        if x1 <= x0 or y1 <= y0:
            return out
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(m, [pts - np.array([x0, y0], np.int32)], False, 255, 1, cv2.LINE_AA)
        d = cv2.distanceTransform((m < 128).astype(np.uint8), cv2.DIST_L2, 3)
        out[y0:y1, x0:x1] = _ss(reach[0] * wf, reach[1] * wf, d).astype(np.float32)
        return out
    return _cached("jawunder_%.3f_%.3f" % (reach[0], reach[1]), lm, shape, scale, wf, make)


def _across_weight(Y: np.ndarray, m: np.ndarray, axis, wf: float) -> Optional[np.ndarray]:
    """0..1 at Y's size: where the structure of Y under the mask m (the neck) lies across the neck's axis (its brightness changes most along the axis, and
    in one direction), 1 outside the mask's region. None where there is no axis or no mask."""
    if axis is None:
        return None
    r = _roi(m, int(0.12 * wf) + 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    s = max(1.0, 0.012 * wf)
    Ys = cv2.GaussianBlur(roi, (0, 0), s)
    gx = cv2.Sobel(Ys, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ys, cv2.CV_32F, 0, 1, ksize=3)
    sg = max(2.0, 0.06 * wf)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sg)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sg)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sg)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-9)
    th = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)                      # the direction the brightness changes most in
    c = np.abs(np.cos(th) * float(axis[0]) + np.sin(th) * float(axis[1]))
    out = np.ones(Y.shape, np.float32)
    out[y0:y1, x0:x1] = (_ss(NECK_ACROSS[0], NECK_ACROSS[1], c) * _ss(NECK_ACROSS_COH[0], NECK_ACROSS_COH[1], coh)).astype(np.float32)
    return out


def _cap_change(Y0: np.ndarray, Y1: np.ndarray, m: np.ndarray, cap) -> np.ndarray:
    """Y1 with its change against Y0 capped softly (most lighter cap[0], most darker cap[1]) inside the region of the mask m: small changes pass as they are,
    large ones are pressed towards the cap (c * (1 - exp(-d / c)))."""
    r = _roi(m, 6)
    if r is None:
        return Y1
    y0, y1, x0, x1 = r
    a = Y0[y0:y1, x0:x1].astype(np.float32)
    d = Y1[y0:y1, x0:x1].astype(np.float32) - a
    up, dn = max(float(cap[0]), 1e-4), max(float(cap[1]), 1e-4)
    dc = np.where(d > 0, up * (1.0 - np.exp(-np.maximum(d, 0.0) / up)), -dn * (1.0 - np.exp(-np.maximum(-d, 0.0) / dn))).astype(np.float32)
    out = Y1.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = a + dc
    return out


def _masked_blur(Y: np.ndarray, w: np.ndarray, sx: float, sy: float) -> np.ndarray:
    num = cv2.GaussianBlur(Y * w, (0, 0), sigmaX=max(0.5, sx), sigmaY=max(0.5, sy))
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=max(0.5, sx), sigmaY=max(0.5, sy))
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), Y).astype(np.float32)


def _even_mid(roi: np.ndarray, w: np.ndarray, wf: float, sig, cap=None) -> np.ndarray:
    """What evens out the middle scale of roi: the broad level minus the level at the fine scale (both measured on
    w only), capped (cap = (most lift, most darkening); default FOLD_LEVEL_CAP)."""
    cp = FOLD_LEVEL_CAP if cap is None else cap
    mid = _masked_blur(roi, w, sig[0] * wf, sig[0] * wf)
    low = _masked_blur(roi, w, sig[1] * wf, sig[1] * wf)
    return np.clip(low - mid, -cp[1], cp[0])


def _fold_level(Y: np.ndarray, band: np.ndarray, skin: np.ndarray, excl: Optional[np.ndarray], wf: float,
                k: float, edges: Optional[np.ndarray] = None, fade_excl=None, fade_edge=None, cap=None) -> np.ndarray:
    """Even out a fold's soft shading inside its band (luma): the fold's own scale goes, the face's broad modelling
    and the skin's fine texture stay. Lips, nostrils and eyes (excl) are out of the reference, and the band fades
    out before them."""
    if k < 1e-4:
        return Y
    r = _roi(band, int(3 * FOLD_LEVEL_SIG[1] * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    b = np.clip(band[y0:y1, x0:x1].astype(np.float32), 0, 1)
    w = np.clip(skin[y0:y1, x0:x1].astype(np.float32), 0, 1)
    if excl is not None:
        e = np.clip(excl[y0:y1, x0:x1].astype(np.float32), 0, 1)
        d_ = max(3, int(0.03 * wf) | 1)
        w = w * (1.0 - cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_, d_))))
        fx_ = FOLD_LEVEL_EXCL if fade_excl is None else fade_excl
        d2 = max(3, int(fx_[0] * wf) | 1)
        e2 = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d2, d2)))
        b = b * (1.0 - np.clip(cv2.GaussianBlur(e2, (0, 0), max(1.0, fx_[1] * wf)), 0, 1))
    if edges is not None:
        # a strand of hair or a sharp edge in the band is left as it is, and the skin round it is evened less the
        # nearer it is: lifted right up to it, a dark strand stood out as a notch (2611 cut 011)
        h = np.clip(edges[y0:y1, x0:x1].astype(np.float32), 0, 1)
        w = w * (1.0 - h)
        fe_ = FOLD_LEVEL_EDGE if fade_edge is None else fade_edge
        d3 = max(3, int(fe_[0] * wf) | 1)
        h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d3, d3)))
        b = b * (1.0 - np.clip(cv2.GaussianBlur(h, (0, 0), max(1.0, fe_[1] * wf)), 0, 1))
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * b * _even_mid(roi, w, wf, FOLD_LEVEL_SIG, cap)
    return out


def _neck_level(Y: np.ndarray, m: np.ndarray, hold: Optional[np.ndarray], wf: float, k: float, axis=None) -> np.ndarray:
    """Even out the neck's soft rings and bands (luma), measured on the neck only; hold (the jaw line) keeps the
    jaw's own shadow."""
    if k < 1e-4:
        return Y
    r = _roi(m, int(3 * NECK_LEVEL_SIG[1] * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = np.clip(m[y0:y1, x0:x1].astype(np.float32), 0, 1)
    d = _even_mid(roi, (mm > 0.3).astype(np.float32), wf, NECK_LEVEL_SIG, NECK_LEVEL_CAP)
    if hold is not None:
        mm = mm * hold[y0:y1, x0:x1]
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * mm * d
    return out


NECK_RING = 0.8            # 2611 cut 024: the soft rings across the neck (a tenth of the face high) stayed at 강하게
NECK_RING_HW = (0.35, 0.10)  # the closing's width and height (share of the face width): lies across the neck


def _ring_fill(Y: np.ndarray, m: np.ndarray, hold: Optional[np.ndarray], wf: float, k: float, axis=None,
               cap: float = FOLD_BROAD_CAP) -> np.ndarray:
    """Lift the broad soft rings across the neck: darker than the skin above and below them at a tenth of the face
    width, measured with a wide flat closing turned to lie across the neck. Luma only, never darker; hold (the
    jaw line) keeps the jaw's own shadow."""
    if k < 1e-4:
        return Y
    w_, h_ = NECK_RING_HW
    r = _roi(m, int(w_ * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1].astype(np.float32)
    Ys = cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.006 * wf))
    W, H = max(3, int(w_ * wf) | 1), max(3, int(h_ * wf) | 1)
    se = np.zeros((max(W, H), max(W, H)), np.uint8)
    cv2.ellipse(se, (se.shape[1] // 2, se.shape[0] // 2), (W // 2, H // 2),
                float(np.degrees(np.arctan2(-axis[0], axis[1]))) if axis is not None else 0.0, 0, 360, 1, -1)
    bh = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, se)
    bh = cv2.GaussianBlur(np.clip(bh, 0, cap), (0, 0), max(1.0, 0.012 * wf))
    if hold is not None:
        mm = mm * hold[y0:y1, x0:x1]
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * mm * bh
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


# ----------------------------------------------------------------------------- flaws found by detect.py (1.19)
FOLD_ZONE_OLD = True     # with 팔자주름·입가주름 on, the pre-1.19 spot finder still works along those folds
DET_ON = True              # False: every correction looks for its flaw itself, on the frame, at one size (before 1.19)
DET_SURE = (0.45, 0.85)        # score from which a flaw is sure enough for the grain band inside it to be given up .. in full
DET_FINE = (0.04, 0.5, 1.2)    # what is finer than this stays inside a filled spot (share of the spot size, px floor and cap):
                               # the picture's own grain at pixel size; the grain protection of the skin around is not changed


def spot_size(job, wf: float) -> float:
    """The largest spot's diameter in pixels of a picture where the face is wf pixels wide."""
    return max(2.0, (0.012 + 0.05 * job.blemish_size) * wf)


DET_WF_MAX = 640.0         # the face is at most this many pixels wide on the picture the flaws are looked for on
DET_SRC = True             # read the piece from the original where it has clearly more pixels than the analysis picture
DET_SRC_GAIN = 1.25        # "clearly more": this many times the analysis picture's


def _det_wanted(v) -> bool:
    """A correction that works from the detection maps is on (잡티 제거, 목주름, 목 오돌토돌)."""
    return any(v.get(k, 0) > 1e-4 for k in ("blemish", "neck_lines", "neck_bumps"))


def _det_raw(cut, px, t: int, job, lm: np.ndarray, rgb_u8: np.ndarray) -> Optional[dict]:
    """One frame's score maps for a job, on the best picture there is of its face and neck: a piece of the original
    at (up to) its own resolution where the analysis picture is a reduced copy (a 4K original seen at 1920: fine
    lines and small spots are gone from the copy), else the analysis picture itself. Never a reduced picture made
    larger. The masks, the landmarks and the frame number are the analysis picture's; the piece is in the same
    display view."""
    v = job.v
    sk = job.masks.get("skinx") if v.get("blemish", 0) > 1e-4 else None
    nk = job.masks.get("neck") if _det_wanted(v) else None
    if sk is None and nk is None:
        return None
    reg = sk if nk is None else (nk if sk is None else np.maximum(sk, nk))
    wf_p = float(job.wf_src) * px.sx
    r = _roi(reg, int(0.08 * wf_p) + 4)
    if r is None or wf_p < 8:
        return None
    y0, y1, x0, x1 = r
    want = min(1.0 / px.sx, DET_WF_MAX / max(wf_p, 1.0))        # picture pixels per analysis pixel wanted
    img = None
    from_src = False
    if DET_SRC and want > DET_SRC_GAIN:
        try:
            from ..core import frames as FRM
            from ..warp.render import _vs_for
            sc = FRM.source_crops(px, vs=_vs_for(cut))
            if sc is not None:
                # steps of an eighth: the reader of the piece goes on while the step stays the same
                scale = float(np.clip(round(want * px.sx * 8.0) / 8.0, 0.125, 1.0))
                bx = (int(x0 / px.sx) // 2 * 2, int(y0 / px.sy) // 2 * 2,
                      min(int(px.info.width), -(-int(np.ceil(x1 / px.sx)) // 2) * 2),
                      min(int(px.info.height), -(-int(np.ceil(y1 / px.sy)) // 2) * 2))
                img = sc.piece(t, bx, scale)
                from_src = True
                # the piece's place in analysis pixels (it was widened to whole source pixels)
                x0f, y0f, x1f, y1f = bx[0] * px.sx, bx[1] * px.sy, bx[2] * px.sx, bx[3] * px.sy
        except Exception:
            import traceback
            traceback.print_exc()
            img = None
    if img is None:
        img = rgb_u8[y0:y1, x0:x1]
        x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
        if wf_p > DET_WF_MAX * 1.1:
            f_ = DET_WF_MAX / wf_p
            img = cv2.resize(img, (max(8, int(round(img.shape[1] * f_))), max(8, int(round(img.shape[0] * f_)))),
                             interpolation=cv2.INTER_AREA)
    hd, wd = img.shape[:2]
    fx, fy = wd / max(x1f - x0f, 1e-6), hd / max(y1f - y0f, 1e-6)
    # analysis pixels -> picture pixels
    A = np.array([[fx, 0, -x0f * fx], [0, fy, -y0f * fy]], np.float32)

    def to_pic(m):
        return None if m is None else cv2.warpAffine(m.astype(np.float32), A, (wd, hd), flags=cv2.INTER_LINEAR,
                                                     borderValue=0)
    Y, Cb, Cr = rgb_to_ycc(img.astype(np.float32) / 255.0)
    lmd = np.asarray(lm, np.float32)[:, :2].copy()
    lmd[:, 0] = (lmd[:, 0] - x0f) * fx
    lmd[:, 1] = (lmd[:, 1] - y0f) * fy
    wf_d = wf_p * fx
    _det_cfg()
    maps = DT.detect(Y, Cb, Cr, to_pic(sk), to_pic(nk), wf_d, spot_size(job, wf_d), neck_axis(lmd), lmd,
                     net=_spot_net(job, img.astype(np.float32) / 255.0, wf_d))
    th = cv2.resize(Y, (max(4, wd // 4), max(4, hd // 4)), interpolation=cv2.INTER_AREA)
    return dict(box=(x0f, y0f, x1f, y1f), maps={k: (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8) for k, m in maps.items()},
                thumb=(np.clip(th, 0, 1) * 255 + 0.5).astype(np.uint8), src=from_src, wf=float(wf_d))


DET_ALIGN = 0.006          # of the face width: how far a neighbour's score may sit from this frame's (the alignment's slack)
DET_SAME = 0.07            # brightness difference (0..1) from which a neighbour's picture is not this frame's skin (a hand, a turn)
DET_CONFIRM = (0.10, 0.40)  # the neighbours' score from which a flaw found here counts .. counts in full


def _det_fuse(own: dict, others: list) -> dict:
    """The frame's score maps held steady through time. others: [(weight, det of a neighbouring frame, A)], A the
    similarity that carries that frame's analysis pixels onto this frame's (the face's own motion).

    A flaw is on the skin for as long as the skin is in view; grain is new on every frame. So each neighbour's
    scores are carried onto this frame, and
    - where a neighbour's picture is not this frame's picture (something in front of the skin, a turn, a blur, the
      alignment off: its carried brightness differs) it has no say there;
    - a score found here that the neighbours do not have is taken down (grain taken for a spot);
    - a score the neighbours have and this frame misses (the grain hid it) is raised.
    With no neighbour that has a say, the frame's own scores stand: nothing is carried across a cut, a cover or a
    failed alignment, and nothing lags behind (the window is the same both ways)."""
    if not others:
        return own
    hd, wd = next(iter(own["maps"].values())).shape
    x0, y0, x1, y1 = own["box"]
    fx, fy = wd / max(x1 - x0, 1e-6), hd / max(y1 - y0, 1e-6)
    Tt = np.array([[fx, 0, -x0 * fx], [0, fy, -y0 * fy], [0, 0, 1]], np.float64)
    th_t = own["thumb"].astype(np.float32) / 255.0
    rad = max(1, int(round(DET_ALIGN * own.get("wf", 300.0))))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    acc = {k: np.zeros((hd, wd), np.float32) for k in own["maps"]}
    wn = np.zeros((hd, wd), np.float32)
    wtot = 0.0
    for w, ds, A in others:
        if ds is None or not ds.get("maps"):
            continue
        hs, ws = next(iter(ds["maps"].values())).shape
        sx0, sy0, sx1, sy1 = ds["box"]
        gx, gy = ws / max(sx1 - sx0, 1e-6), hs / max(sy1 - sy0, 1e-6)
        Ts_inv = np.array([[1 / gx, 0, sx0], [0, 1 / gy, sy0], [0, 0, 1]], np.float64)
        A3 = np.vstack([np.asarray(A, np.float64), [0, 0, 1]])
        Mm = (Tt @ A3 @ Ts_inv)[:2].astype(np.float32)
        # the same skin? the neighbour's picture carried here against this frame's (a quarter of the size)
        q = np.diag([0.25, 0.25, 1.0])
        qi = np.diag([4.0, 4.0, 1.0])
        Mt = (q @ np.vstack([Mm, [0, 0, 1]]) @ qi)[:2].astype(np.float32)
        ths = cv2.warpAffine(ds["thumb"].astype(np.float32) / 255.0, Mt, (th_t.shape[1], th_t.shape[0]),
                             flags=cv2.INTER_LINEAR, borderValue=-1.0)
        dif = cv2.GaussianBlur(np.abs(ths - th_t), (0, 0), 1.5)
        g = np.exp(-np.square(dif / DET_SAME)) * (ths >= 0)
        g = cv2.resize(g.astype(np.float32), (wd, hd), interpolation=cv2.INTER_LINEAR) * float(w)
        for k in acc:
            ms = ds["maps"].get(k)
            if ms is None:
                continue
            mw = cv2.warpAffine(ms, Mm, (wd, hd), flags=cv2.INTER_LINEAR, borderValue=0)
            acc[k] += g * cv2.dilate(mw, se).astype(np.float32) / 255.0
        wn += g
        wtot += float(w)
    if wtot <= 0:
        return own
    conf = np.clip(wn / (0.5 * wtot), 0, 1)                     # how much the neighbours have a say at each point
    out = {}
    for k, m in own["maps"].items():
        o = m.astype(np.float32) / 255.0
        p = acc[k] / np.maximum(wn, 1e-3)                       # what the neighbours found here
        if k == "protect":
            f = np.maximum(o, p * conf)                         # kept wherever any frame says so
        else:
            both = (o + p * wn) / (1.0 + wn)
            f = conf * both * _ss(DET_CONFIRM[0], DET_CONFIRM[1], p) + (1.0 - conf) * o
        out[k] = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
    return dict(own, maps=out)


def _det_now(job, Y, Cb, Cr, wf: float, y_per_src: float, M) -> dict:
    """The score maps of this job found on this picture (no detection came with the job: a single frame worked on
    without its neighbours, a test)."""
    hY, wY = Y.shape
    v = job.v
    skin = M(job.masks["skinx"], (hY, wY)) if ("skinx" in job.masks and v.get("blemish", 0) > 1e-4) else None
    neck = None
    if "neck" in job.masks and _det_wanted(v):
        neck = M(job.masks["neck"], (hY, wY))
    if skin is None and neck is None:
        return {}
    cbY = Cb if Cb.shape == Y.shape else cv2.resize(Cb, (wY, hY), interpolation=cv2.INTER_LINEAR)
    crY = Cr if Cr.shape == Y.shape else cv2.resize(Cr, (wY, hY), interpolation=cv2.INTER_LINEAR)
    lm = None if job.face is None else np.asarray(job.face, np.float32)[:, :2] * y_per_src
    rgb = None
    if _spot_net_wanted(job):
        rgb = ycc_to_rgb(Y, cbY, crY)
    _det_cfg()
    return DT.detect(Y, cbY, crY, skin, neck, wf, spot_size(job, wf), neck_axis(lm), lm,
                     net=_spot_net(job, rgb, wf))


def _spot_net_wanted(job) -> bool:
    """The learned spot detector is used: 잡티 제거 on, a large face (SPOTNET_WF source px), the model there."""
    from . import spotnet_run as SR
    return job.v.get("blemish", 0) > 1e-4 and float(job.wf_src) >= SR.SPOTNET_WF and SR.available()


def _spot_net(job, rgb, wf: float):
    """The callable DT.detect runs on its region, or None (the rule detector's spots)."""
    if rgb is None or not _spot_net_wanted(job):
        return None
    from . import spotnet_run as SR
    return lambda region, size: SR.run(rgb, region, wf, max_size=size)


def _det_place(job, rf, shape) -> dict:
    """The job's steadied score maps put on a plane of this shape (float32 0..1)."""
    out = {}
    d = job.det
    if not d or not d.get("maps"):
        return out
    hY, wY = shape
    pw, ph = rf.proxy_wh
    fx, fy = wY / float(pw), hY / float(ph)
    x0, y0, x1, y1 = d["box"]
    hd, wd = next(iter(d["maps"].values())).shape
    # picture pixels -> plane pixels
    ax, ay = (x1 - x0) * fx / wd, (y1 - y0) * fy / hd
    A = np.array([[ax, 0, x0 * fx], [0, ay, y0 * fy]], np.float32)
    inter = cv2.INTER_AREA if ax < 0.75 else cv2.INTER_LINEAR
    for k, m in d["maps"].items():
        if inter == cv2.INTER_AREA:
            a = np.zeros(shape, np.float32)
            X0, Y0 = max(0, int(round(x0 * fx))), max(0, int(round(y0 * fy)))
            X1, Y1 = min(wY, int(round(x1 * fx))), min(hY, int(round(y1 * fy)))
            if X1 - X0 >= 2 and Y1 - Y0 >= 2:
                a[Y0:Y1, X0:X1] = cv2.resize(m, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        else:
            a = cv2.warpAffine(m, A, (wY, hY), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255.0
        out[k] = a
    return out


# the most a found spot's colour is moved. 0.06 left a strong pimple (Cr +0.1) 40 % red (review 6c3dcb3); what is far
# redder than any pimple is protected (detect.RED_THING) and never reaches this
SPOT_CHROMA_CAP = 0.12


def _spot_fill(Y, Cb, Cr, mY, S, Sred, k: float, size: float, cs: float, plane_h: Optional[int] = None):
    """Fill what detect.py found from the skin around it: S (dark spots and raised bumps, with their dark side) in
    brightness and colour, Sred (red spots) in colour and brightness. The spot is replaced by the level of the
    skin beside it (the spots themselves left out of that level), whichever way it differs: a dark spot is lifted,
    a bump's bright top is taken down and its dark side lifted. What is finer than DET_FINE stays as it is (the
    picture's grain goes on through the spot), nothing is added."""
    Sall = np.maximum(S, Sred) if Sred is not None else S
    r = _roi(Sall, int(2.5 * size) + 6)
    if r is None or k < 1e-4:
        return Y, Cb, Cr
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = mY[y0:y1, x0:x1].astype(np.float32)
    sp = np.clip(Sall[y0:y1, x0:x1], 0, 1) * np.clip(mm * 1.5, 0, 1)
    # a red spot is filled in colour; in brightness only half as far (a pimple is hardly darker than the skin, and
    # what is redder for another reason, skin in shade, must keep its shade)
    spl = sp if Sred is None else np.clip(np.maximum(S[y0:y1, x0:x1], 0.5 * Sred[y0:y1, x0:x1]), 0, 1) * np.clip(mm * 1.5, 0, 1)
    sig_f = float(np.clip(DET_FINE[0] * size, DET_FINE[1], DET_FINE[2]))
    # what stays of the picture's fine detail inside a fill: the whole grain band (the one the skin smoothing gives
    # back, _keep_fine) where the flaw is only likely; where it is sure, only what is finer than DET_FINE, because
    # a small spot's sharp core is in that band and stayed as a dot. Nothing is put in its place
    # (plane_h: the plane's height when Y is only a piece of it: the grain band follows the plane, not the piece)
    sig_g = float(np.clip(GRAIN_SIGMA * (plane_h or Y.shape[0]) / 1080.0, 0.8, 2.4))
    Ys = cv2.GaussianBlur(roi, (0, 0), sig_f)
    if sig_g > sig_f * 1.2:
        sure = _ss(DET_SURE[0], DET_SURE[1], sp)
        Yg = cv2.GaussianBlur(roi, (0, 0), sig_g)
        Ys = Yg + sure * (Ys - Yg)
    wex = (mm > 0.05).astype(np.float32) * (1.0 - np.clip(sp * 2.0, 0, 1))

    def around(P, w, sizes):
        # the nearest skin that is not a spot: the narrow surround where it has enough of it, else a wider one
        out, have = None, None
        for sg in sizes:
            num = cv2.GaussianBlur(P * w, (0, 0), sg)
            den = cv2.GaussianBlur(w, (0, 0), sg)
            b = num / np.maximum(den, 1e-3)
            ok = np.clip((den - 0.15) / 0.2, 0, 1)
            if out is None:
                out, have = b * ok, ok
            else:
                out = out + b * ok * (1 - have)
                have = have + ok * (1 - have)
        return np.where(have > 0.05, out / np.maximum(have, 0.05), P), np.clip(have * 2, 0, 1)

    sizes = (max(1.5, 0.45 * size), max(3.0, 0.9 * size), max(6.0, 2.0 * size))
    base, ok = around(Ys, wex, sizes)
    Yo = Y.astype(np.float32, copy=True)
    Yo[y0:y1, x0:x1] = roi + k * spl * ok * np.clip(base - Ys, -0.2, 0.2)
    Cbo, Cro = Cb.astype(np.float32, copy=True), Cr.astype(np.float32, copy=True)
    if Cb.shape == Y.shape:
        c0, c1, d0, d1 = y0, y1, x0, x1
        spc, wc = sp, wex
    else:
        c0, c1, d0, d1 = _croi(y0, y1, x0, x1, Y.shape, Cb.shape)
        spc = cv2.resize(sp, (d1 - d0, c1 - c0), interpolation=cv2.INTER_AREA)
        wc = cv2.resize(wex, (d1 - d0, c1 - c0), interpolation=cv2.INTER_AREA)
    csz = tuple(max(1.0, x * cs) for x in sizes)
    for P, O in ((Cb, Cbo), (Cr, Cro)):
        pr = P[c0:c1, d0:d1].astype(np.float32)
        ps = cv2.GaussianBlur(pr, (0, 0), max(0.5, sig_f * cs))
        pb, okc = around(ps, wc, csz)
        O[c0:c1, d0:d1] = pr + k * spc * okc * np.clip(pb - ps, -SPOT_CHROMA_CAP, SPOT_CHROMA_CAP)
    return Yo, Cbo, Cro


def _fold_any(Y: np.ndarray, m: np.ndarray, S: np.ndarray, hold: Optional[np.ndarray], wf: float, k: float,
              cap: float = 0.06) -> np.ndarray:
    """Broad folds found by detect.py in any direction (a head turned or tilted: the rings do not lie across the
    picture's neck axis): what is darker than the skin either side at a tenth of the face width, inside the found
    fold only. Luma only, never darker."""
    if k < 1e-4:
        return Y
    r = _roi(S * m, int(0.12 * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    ww = (S[y0:y1, x0:x1] * m[y0:y1, x0:x1]).astype(np.float32)
    Ys = cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.006 * wf))
    ks = max(3, int(0.10 * wf) | 1)
    bh = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bh = cv2.GaussianBlur(np.clip(bh, 0, cap), (0, 0), max(1.0, 0.012 * wf))
    if hold is not None:
        ww = ww * hold[y0:y1, x0:x1]
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * ww * bh
    return out


# 2611 cut 024 (user 2026-11-17): the shine's white went the skin's colour. A highlight keeps its own colour; the
# shine slider only takes its brightness down (and the basic adjustment no longer sets it)
SHINE_KEEP_COLOUR = True


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
    if SHINE_KEEP_COLOUR:
        return Yo, Cbo, Cro
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


CHIN_DIMPLE = 0.0      # 0..1: soften a pebbled chin, only on frames where it shows (set by the caller; no slider yet)
FACE_LINES = 0.0       # 0..1: fill fine lines on the face skin (forehead, under the eyes, cheeks)
CHIN_LO, CHIN_HI = 0.045, 0.085


def chin_zone(shape, lm: np.ndarray, wf: float) -> Optional[np.ndarray]:
    """The chin between the lower lip and the jaw (soft), lm in px of `shape`."""
    if lm is None or not np.isfinite(lm[[57, 8], :2]).all():
        return None
    a, b = lm[57, :2], lm[8, :2]
    c = 0.4 * a + 0.6 * b
    v = b - a
    L = float(np.hypot(*v))
    if L < 4:
        return None
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (int(c[0]), int(c[1])), (max(2, int(0.42 * L)), max(2, int(0.24 * wf))),
                float(np.degrees(np.arctan2(v[1], v[0]))), 0, 360, 255, -1)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.03 * wf))


def _dimple_map(Y: np.ndarray, wf: float) -> np.ndarray:
    """Pits and bumps a hundredth of the face wide (a pebbled chin): the determinant of the Hessian, so the
    edge of a shadow or the lip line (curved one way only) does not answer."""
    sg = max(1.0, 0.01 * wf)
    ys = cv2.GaussianBlur(Y, (0, 0), sg)
    base = cv2.GaussianBlur(Y, (0, 0), 4 * sg) + 8.0 / 255.0
    xx = cv2.Sobel(ys, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(ys, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(ys, cv2.CV_32F, 1, 1, ksize=3)
    return np.sqrt(np.maximum(xx * yy - xy * xy, 0)) * sg * sg / base


def _chin_dimple(Y: np.ndarray, lm: Optional[np.ndarray], skin: np.ndarray, wf: float, k: float) -> np.ndarray:
    """Soften a pebbled chin (the chin muscle pulled tight). How pebbled the chin is, is measured on this very
    frame: on a frame where the chin is smooth the amount is 0 and nothing is touched, and it rises with the
    dimples. The broad light and shade of the chin stay as they are."""
    z = chin_zone(Y.shape, lm, wf)
    if z is None:
        return Y
    r_ = _roi(z, int(0.15 * wf) + 8)
    if r_ is None:
        return Y
    y0, y1, x0, x1 = r_
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    zz, ss = z[y0:y1, x0:x1], skin[y0:y1, x0:x1]
    reg = (zz > 0.5) & (ss > 0.5)
    if reg.sum() < 40:
        return Y
    s = float(np.percentile(_dimple_map(roi, wf)[reg], 90))
    amt = float(np.clip((s - CHIN_LO) / (CHIN_HI - CHIN_LO), 0, 1)) * k
    if amt < 0.02:
        return Y
    m = (zz * ss * amt).astype(np.float32)
    r = max(2, int(round(0.03 * wf)))
    out = SK.smooth_channel(roi, m, 1.0, 0.0, r, eps=0.003, relative=True)
    d = out - roi
    sig = max(3.0, 4.0 * r)
    low = cv2.GaussianBlur(d, (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 0.05)
    Yo = Y.astype(np.float32, copy=True)
    Yo[y0:y1, x0:x1] = out - m * np.clip(low, -0.2, 0.2)
    return Yo


THIN_LO, THIN_HI = 0.4, 0.7
LINE_DEEP_LO, LINE_DEEP_HI = 0.20, 0.32   # a skin line is at most this much darker than the skin beside it


def neck_axis(lm: Optional[np.ndarray]):
    """Unit vector down the face (nose bridge to chin): the direction the neck runs in, picture coordinates."""
    if lm is None or not np.isfinite(lm[[27, 8], :2]).all():
        return None
    v = lm[8, :2] - lm[27, :2]
    n = float(np.hypot(*v))
    return None if n < 1e-3 else (float(v[0] / n), float(v[1] / n))


_MCACHE: dict = {}
_MC_LOCK = threading.Lock()         # (people of a frame are prepared side by side: PREP_THREADS)


def _cached(name: str, lm: np.ndarray, shape, scale: float, wf: float, make):
    """The face masks below depend only on the landmarks and the frame size; they are asked for more than once
    per face per frame (by several retouch steps), and are made once."""
    key = (name, np.asarray(lm[:, :2], np.float32).tobytes(), tuple(shape), float(scale), float(wf))
    with _MC_LOCK:
        hit = _MCACHE.get(key)
    if hit is None:
        if len(_MCACHE) > 48:
            with _MC_LOCK:
                _MCACHE.clear()
        hit = make()                # (made outside the lock: the same mask made twice by two threads is the same mask)
        with _MC_LOCK:
            _MCACHE[key] = hit
    return hit


def _window(shape, pts: np.ndarray, margin: float):
    """The part of the frame within `margin` px of `pts` (x0, y0, x1, y1). A mask drawn there and blurred is
    the same as one drawn on the whole frame, as long as the margin covers the drawing's reach and the blur's."""
    h, w = shape[:2]
    m = int(np.ceil(margin)) + 2
    x0, y0 = np.floor(pts.min(0)).astype(int) - m
    x1, y1 = np.ceil(pts.max(0)).astype(int) + m + 1
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def jaw_hold(lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """1 off the jaw line, 0 on it (soft): the shadow line under the jaw and chin is the face's outline, not a
    neck line, and stays. lm in source px, scale = px of `shape` per source px, wf in px of `shape`."""
    if lm is None or not np.isfinite(lm[2:15, :2]).all():
        return None
    def make():
        pts = np.round(lm[2:15, :2] * scale).astype(np.int32)
        th = max(1, int(round(0.07 * wf)))
        sig = max(1.0, 0.025 * wf)
        x0, y0, x1, y1 = _window(shape, pts, th + 8 * sig + 4)
        out = np.ones(shape[:2], np.float32)
        if x1 <= x0 or y1 <= y0:
            return out
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(m, [pts - np.array([x0, y0], np.int32)], False, 255, th, cv2.LINE_AA)
        out[y0:y1, x0:x1] = 1.0 - cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sig)
        return out
    return _cached("jaw", lm, shape, scale, wf, make)


GLINT_LO, GLINT_HI = 0.22, 0.38   # a thin bright thing on the skin (a light hair, a chain's glint)
THING_MIN_LEN = 0.07        # of the face width: shorter than this is no "thing" lying on the skin
PAINT_DC = (0.018, 0.03)     # chroma off the skin's own (0..1 YCbCr units) at which a thin line is paint


def _chroma_off(Cb: np.ndarray, Cr: np.ndarray, shape, box, wf: float) -> Optional[np.ndarray]:
    """How far each pixel's colour is from the skin's colour around it (0..), inside box (y0, y1, x0, x1) of a
    frame of `shape`. Face paint, tattoos and drawn veins are blue, purple, red or black lines on skin; a
    crease or a pore is skin in shade and keeps the skin's colour."""
    if Cb is None or Cr is None:
        return None
    y0, y1, x0, x1 = box
    hh, ww = shape
    def at(c):
        if c.shape != (hh, ww):
            c = cv2.resize(c, (ww, hh), interpolation=cv2.INTER_LINEAR)
        return c[y0:y1, x0:x1].astype(np.float32)
    cb, cr = at(Cb), at(Cr)
    cb = cv2.GaussianBlur(cb, (0, 0), 1.0)
    cr = cv2.GaussianBlur(cr, (0, 0), 1.0)
    s = max(2.0, 0.06 * wf)
    return np.hypot(cb - cv2.GaussianBlur(cb, (0, 0), s), cr - cv2.GaussianBlur(cr, (0, 0), s))


def _things(Y: np.ndarray, m: np.ndarray, wf: float, Cb: Optional[np.ndarray] = None,
            Cr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Thin things lying on the skin that are not skin (a necklace chain, a cord, a strap, a dark hair): 0..1.
    Much darker than the skin beside them, or glinting. The skin smoothing leaves them alone (it used to
    soften a chain on the chest, which no mask left out)."""
    r = _roi(m, int(0.1 * wf) + 4)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.5)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.045 * wf)) + 0.02
    kn = max(3, int(round(0.03 * wf)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn))
    dk = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, se) / base
    gl = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, se) / base
    kd = max(3, int(round(0.012 * wf)) | 1)
    t = np.maximum(_ss(LINE_DEEP_LO, LINE_DEEP_HI, dk), _ss(GLINT_LO, GLINT_HI, gl))
    # thin lines in a colour of their own (face paint, tattoos, drawn veins): left as they are
    try:
        dc = _chroma_off(Cb, Cr, Y.shape, r, wf)
        if dc is not None:
            # also bright ones (light hair over a pale cheek is a thin line whose brightness is the skin's but whose
            # colour is not; light paint): the tophat counts like the blackhat
            t = np.maximum(t, _ss(*PAINT_DC, dc) * np.maximum(_ss(0.02, 0.05, dk), _ss(0.02, 0.05, gl)))
    except Exception:
        pass
    if THING_MIN_LEN > 0:
        # a thing is long (a chain, a cord, a strand, a drawn line). Pores, the grain of the skin in a close-up and
        # the sparkle of a sweaty or powdered cheek are as dark or as bright against the skin but only a few
        # pixels across: taken for things they took 30 to 75 % of the face out of the smoothing (cut 069), in
        # patches that changed from frame to frame
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats((t > 0.3).astype(np.uint8), connectivity=8)
        if n_ > 1:
            ln_ = np.hypot(st_[:, cv2.CC_STAT_WIDTH], st_[:, cv2.CC_STAT_HEIGHT])
            keep_ = ln_ >= THING_MIN_LEN * wf
            keep_[0] = False
            t = t * keep_[lab_]
        else:
            t = t * 0.0
    t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    out = np.zeros(Y.shape, np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(t, (0, 0), max(1.0, 0.006 * wf)) * 1.4, 0, 1)
    return out


NASO_REACH = 0.07           # the same in the nasolabial stage. 2611 IWALY c029, c010 (2026-11-22, user): the groove down from the nose wing is
                            # the top of the nasolabial fold and has to be retouched; held as "nose" out to 0.07 of the face width it got a third
                            # of a per cent. Smaller = the fold is taken closer to the wing (the wing itself, inside the landmarks' hull, stays held)
ALA_KEEP = 1.0              # share of the nose wing's own exclusion (_ala_mask) kept in the fold's band (1 = as before; the band is made at the
                            # preparation of the frame)
NASO_DEEP_NOSE = 0.0        # 2611 IWALY c010 f15, c001 f92 (2026-11-22, user: "콧구멍 사라지고 코라인이 덜 선명해진 오류", the nose wing retouched): the nasolabial stage lifted the nostril's
                            # shade on a turned face (the landmarks' hull missed it). 1 = what the skin smoothing holds as deep (compact, far darker than the skin) is held within NASO_DEEP_REACH of the nose
NASO_DEEP_REACH = 0.15
NOSE_SIDES = False          # 2611 IWALY c029 f4/f124, c010 f4/f15, c001 f92 (2026-11-22, user: "코 라인 밝아진 오류", "콧구멍 사라지고 코라인이 덜 선명해진 오류"): the nose's own
                            # outline (the shaded side wall from the inner eye corner down to the wing, the nostril's rim) was lifted by the nasolabial stage: the held hull
                            # was that of the bridge's middle line and the nose's base only (landmarks 27..35), so on a turned face its side wall lay outside it. True = the
                            # hull also takes the inner eye corners (39, 42) and the wings' outer points pushed out by NOSE_SIDES_WING of the base's width
NOSE_SIDES_WING = 0.12
NOSE_REACH = 0.07           # how far past the nose's outline the nose is held (0.14 held the top of the fold
                            # beside the nostril too, which a frown deepens)
EXT_GUARD = (0.03, 0.10)    # 2611 IWALY c010 (2026-11-22, own review of the user's frame 00:00:58:12): the fold stages worked up to the face's outer edge: on a face turned
                            # three quarters the band of the fold reaches the cheek's contour, where the skin falls off into the shade by the hair and the finder takes that
                            # shading for a fold, and the contour was lifted (+0.04..0.07 in a band along the hair). Within this distance of the exterior (hair, background:
                            # what the exclusion holds that is not a feature) the band fades out: from where (of the face width) .. to where fully in. None = off
EXT_OPEN = 0.03             # of the face width: what is thinner than this (a strand of hair across the cheek) is not the exterior (EXT_GUARD); 0 = every piece counts
EXT_OUTLINE = True          # 2611 IWALY c003 (2026-11-22, user, 00:00:39:17, the green mark at the jaw "mask error"): on a small face (67 px wide) against a beige wall the colour model takes the wall for
                            # skin (the exclusion is 0 on it, so there is no "exterior" to keep the band from), the band of the fold reached the wall beside the cheek and the fold stages lifted the
                            # wall's pixels next to the face's edge by up to 6.6 % of the brightness (a light strip along the whole cheek). The landmarks' own outline of the face (jaw line and
                            # forehead arc) lay within 2 px of the real edge there: the band is held to it as well (the same distances as EXT_GUARD, inside the outline). False = the parser's exterior only
FOLD_BASE = {"nasolabial": 0.06, "mouth_lines": 0.06}   # share of a fold's band filled where no fold is found (2611 IWALY c029/c010/c001, user: "팔자주름 제거 안 되어 있음": soft folds the finder misses)
SMALL_WF = (110.0, 200.0)   # face width (px) below which a fold band is softened as a whole
SMALL_FOLD = 0.45           # how much of the band, at the smallest
NOSE_SKIN = 1.0            # share of the skin smoothing left on the nose (v3: all of it; the narrow smoothing keeps its shape)
HOLD_DEEP = (0.30, 0.45)    # depth (relative to the skin) of a compact dark thing that is shape, not a flaw
HOLD_DEEP_K = 0.035         # of the face width: widest dark part held (0.08 took the eye sockets, the side of the nose
                            # and the upper lip of a face in hard light for "deep": the middle of the face was left out)
HOLD_EDGE = (0.35, 0.60)    # light/shade step across a twentieth of the face that is the light's edge
HOLD_OFF = (0.07, 0.11)
HOLD_MOLE = (0.30, 0.45)    # depth of a small dark spot that is a mole, kept
HOLD_MOLE_ALONE = 0.10      # of the face width: spots with two or more others this near are rough skin, not moles (0 = off)
PAINT_HUE = (40.0, 65.0)    # degrees of hue away from the skin at which colour is paint, not skin     # colour (per unit of brightness) away from the face's own skin: clothes, hair, props


FS_FOREHEAD = True


def _fs_forehead(lm: np.ndarray, shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """The band of forehead just above the brows (from a fifth of the face width above them down to the brows),
    inside the face outline and away from its edge, brows out. 0..1 at `shape`."""
    L = np.asarray(lm, np.float32)[:, :2] * scale
    if not np.isfinite(L[0:27]).all() or wf < 60:
        return None
    h, w = shape
    poly = SK.face_polygon(L)
    if poly is None:
        return None
    reg = np.zeros((h, w), np.uint8)
    cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 1)
    ke = max(3, int(0.10 * wf) | 1)
    reg = cv2.erode(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
    brow = L[17:27]
    xs = np.arange(w, dtype=np.float32)
    by = np.interp(xs, brow[:, 0], brow[:, 1], left=np.nan, right=np.nan)
    yy = np.arange(h, dtype=np.float32)[:, None]
    band = (yy < by[None, :] - 0.03 * wf) & (yy > by[None, :] - 0.22 * wf)
    band &= np.isfinite(by)[None, :]
    feat = SK._features_mask(L, (h, w)) > 0
    kf = max(3, int(0.04 * wf) | 1)
    feat = cv2.dilate(feat.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kf, kf))) > 0
    m = (reg > 0) & band & ~feat
    if not m.any():
        return None
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.02 * wf)), 0, 1)


# 2611 IWALY c001 (2026-11-22, user, 00:00:31:14): on a profile the eye's landmarks sit right but the eye is foreshortened, so the
# skin mask's hole (an ellipse of the eye's width) is smaller than the lashes, the lid lines and the corners are: the smoothing took the
# inner corner's dark crease and the lids' lines for skin flaws (the corner by 6 %). Inside a zone round each eye, every compact dark
# thing (clearly darker than the skin round it) is now held, whatever its contrast against the whole face.
EYE_HOLD = 1.0
EYE_ZONE = (1.2, 0.80, 0.45)      # of the eye's width: along the eye from its middle (past the corners), above it, below it (j2817i: was (0.80, 0.50, 0.24), see the note below)
EYE_DEPTH = (0.05, 0.12)          # depth against the skin level round it: from where it is held .. in full (j2817i: was (0.07, 0.16))
EYE_KERNEL = 0.34                 # of the eye's width: the widest dark part (a lash fan, a lid line, a corner) that is held
# 2611 IWALY c001 (2026-11-22, user, 00:00:31:14, again): on a profile the eye's two corner landmarks come together (11 px apart on a face 253 px wide,
# the eye itself 60 px wide): the zone above, and the kernel that picks the dark parts, were sized by that and covered a fifth of the eye, so the
# lash fan and the outer corner stayed inside the skin smoothing's mask and were lightened by up to 6 % of the picture's brightness. The eye is never
# taken narrower than this share of the face's width (a frontal eye is 0.19 to 0.22 of it).
EYE_MIN_W = 0.21
# the zone is held in full inside this share of its (elliptical) radius and fades out to its edge (0.375 = what it was: 1.6 - 1.6 r). A fade that begins
# at a third of the way out left the lash fan's tips and the lower lid's line (just outside the zone's full part) held by a tenth of what they are worth
# 2611 IWALY c001 (2026-11-22, the same frame, own check of the fix): the lash fan of a profile's eye reaches further than the zone above (two small hot spots at the eye's
# corner, up to 2.3 % of the brightness, were left): the zone is wider and higher (1.2 along the eye, 0.80 above, 0.45 below), held in full to 0.7 of its radius, and the dark
# parts are held from a smaller depth on (EYE_DEPTH). On the one-frame test the spots are gone (1.4 % at most) and the skin round the eye is as it was
EYE_FULL = 0.7


def _eye_structure(rb: np.ndarray, base: np.ndarray, lm: np.ndarray, scale: float, org, wf: float) -> Optional[np.ndarray]:
    """0..1 at rb's size (a piece of the picture starting at org = (x0, y0)): the dark structure of the eyes, found in the picture
    inside a zone round each eye from the landmarks (36-41, 42-47)."""
    h, w = rb.shape
    out = np.zeros((h, w), np.float32)
    P = np.asarray(lm, np.float32)[:, :2] * scale - np.array([org[0], org[1]], np.float32)
    if not np.isfinite(P[36:48]).all():
        return None
    dn0 = None
    if np.isfinite(P[[8, 27]]).all():
        dn0 = P[8] - P[27]
    # the line the eyes lie on (outer corner to outer corner), for an eye whose own corners have come together
    eye_line = None
    if np.isfinite(P[[36, 45]]).all() and float(np.hypot(*(P[45] - P[36]))) > 4.0:
        eye_line = (P[45] - P[36]) / float(np.hypot(*(P[45] - P[36])))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for idx in (range(36, 42), range(42, 48)):
        e = P[list(idx)]
        c = e.mean(0)
        ew = float(np.hypot(*(e[3] - e[0])))
        ew_min = EYE_MIN_W * float(wf)
        if ew >= 0.55 * ew_min:
            ax = (e[3] - e[0]) / max(ew, 1e-6)
        elif eye_line is not None:
            ax = eye_line                                   # corners together: the eyes' own line gives the direction
        else:
            continue
        ew = max(ew, ew_min)
        dn = np.array([-ax[1], ax[0]], np.float32)
        if dn0 is not None and float(dn @ dn0) < 0:
            dn = -dn
        u = (xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]              # along the eye
        v = (xx - c[0]) * dn[0] + (yy - c[1]) * dn[1]              # down (towards the mouth)
        ru = EYE_ZONE[0] * ew
        rv = np.where(v < 0, EYE_ZONE[1], EYE_ZONE[2]) * ew
        zone = np.clip((1.0 - np.sqrt((u / ru) ** 2 + (v / rv) ** 2)) / max(1.0 - EYE_FULL, 1e-3), 0.0, 1.0)
        if not zone.any():
            continue
        k = max(5, int(round(EYE_KERNEL * ew)) | 1)
        bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        st = _ss(EYE_DEPTH[0], EYE_DEPTH[1], bh / base)
        out = np.maximum(out, zone * st)
    return out if out.any() else None


DET_BUMP_MAX = None         # 2611 IWALY c029 f4 (2026-11-22, user: "빨간색 같은 잡티는 아예 지워져야해"): a raised spot whose top catches the light was taken for a glint
DET_GLINT = None            # (detect.BUMP_MAX / detect.GLINT) and left; None = the detector's own values. (lo, hi) of the relative brightness over the skin


DET_RULE_BUMP = None        # True: detect.NET_KEEP_RULE_BUMP (the rule detector's bumps kept beside the learned detector's)


def _det_cfg():
    if DET_RULE_BUMP is not None:
        DT.NET_KEEP_RULE_BUMP = bool(DET_RULE_BUMP)
    if DET_BUMP_MAX is not None:
        DT.BUMP_MAX = tuple(DET_BUMP_MAX)
    if DET_GLINT is not None:
        DT.GLINT = tuple(DET_GLINT)


MOLE_KEEP = 0.0             # 2611 IWALY c029 f4 (2026-11-22, user: "점 연해졌는데 살려야함"): a small dark spot that is not redder than the skin round it (a mole,
                            # a beauty mark) is left to this share by the blemish fill (0 = off: as before; a pimple or its mark is redder and still goes)
MOLE_DEPTH = (0.10, 0.20)   # how much darker than the skin round it (relative) a spot must be to count as a mole
MOLE_RED = (0.005, 0.014)   # how much redder (Cr over the skin round it) a dark spot may be and still count as a mole: from the first to the second it fades out
MOLE_SIZE = 0.03            # of the face width: the largest mole


def _mole_like(Y, Cb, Cr, reg, wf):
    """0..1 at Y resolution: small dark round spots that are not redder than the skin round them (moles), inside reg. None: none."""
    r = _roi(reg, int(0.1 * wf) + 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    hY, wY = Y.shape
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    rb = cv2.GaussianBlur(roi, (0, 0), 1.0)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.045 * wf)) + 0.02
    km = max(3, int(round(MOLE_SIZE * wf)) | 1)
    bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))) / base
    dep = _ss(MOLE_DEPTH[0], MOLE_DEPTH[1], bh)
    dep = dep * (cv2.GaussianBlur(dep, (0, 0), max(1.0, 0.012 * wf)) < 0.5)      # spots, not lines or patches
    if not (dep > 0.05).any():
        return None
    crf = Cr.astype(np.float32)
    if crf.shape != Y.shape:
        crf = cv2.resize(crf, (wY, hY), interpolation=cv2.INTER_LINEAR)
    cr = crf[y0:y1, x0:x1]
    crd = cv2.GaussianBlur(cr, (0, 0), 1.5) - cv2.GaussianBlur(cr, (0, 0), max(2.0, 0.045 * wf))
    notred = 1.0 - _ss(MOLE_RED[0], MOLE_RED[1], crd)
    m = dep * notred
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out = np.zeros(Y.shape, np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(m, (0, 0), 1.0) * 1.3, 0, 1)
    return out


def _hold_maps(Y: np.ndarray, Cb: Optional[np.ndarray], Cr: Optional[np.ndarray], m: np.ndarray, wf: float,
               lm: Optional[np.ndarray], scale: float, parts: bool = False):
    """What no smoothing or line filler may touch, 0..1 each, at Y resolution:
    deep: compact dark parts much darker than the skin (nostrils, the mouth's corners, the hollow beside a
          neck tendon): the face's build. A soft or badly found face had its nostril smeared to grey.
    edge: a hard step from light to shade (a face lit from the side): smoothing or a fold filler softened the
          shadow's edge, and the shade looked painted over.
    off:  lit parts whose colour is not this face's skin (a jacket over the chin, a prop): the skin mask ran
          over them and they were smoothed. Measured per unit of brightness, only where it is bright enough
          to tell (shade loses its colour)."""
    r = _roi(m, int(0.1 * wf) + 4)
    if r is None:
        return (None, None, None) if parts else (None, None)
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    rb = cv2.GaussianBlur(roi, (0, 0), 1.5)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.045 * wf)) + 0.02
    kn = max(5, int(round(HOLD_DEEP_K * wf)) | 1)
    deep = _ss(*HOLD_DEEP, cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn))) / base)
    # moles and beauty marks: small round dark spots, clearly darker than the skin (a pore is smaller and
    # fainter). Strong smoothing wiped them.
    km = max(3, int(round(0.03 * wf)) | 1)
    mole = _ss(*HOLD_MOLE, cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))) / base)
    mole = mole * (cv2.GaussianBlur(mole, (0, 0), max(1.0, 0.012 * wf)) < 0.5)   # spots, not lines or patches
    if HOLD_MOLE_ALONE > 0:
        # a mole stands alone. The pits of rough skin in hard light are as dark and as round and come many together
        # (cut 069, under the eye and beside the nose): held as "moles" they stayed as dots in a smoothed cheek,
        # where the user's own retouch evens them out. Three or more spots within a tenth of the face are skin
        try:
            nb_, lab_, st_, cen_ = cv2.connectedComponentsWithStats((mole > 0.5).astype(np.uint8), 8)
            if nb_ > 1:
                imp = np.zeros(mole.shape, np.float32)
                cy_ = np.clip(np.round(cen_[1:, 1]).astype(int), 0, mole.shape[0] - 1)
                cx_ = np.clip(np.round(cen_[1:, 0]).astype(int), 0, mole.shape[1] - 1)
                np.add.at(imp, (cy_, cx_), 1.0)
                kq = max(5, int(round(HOLD_MOLE_ALONE * wf)) | 1)
                cnt = cv2.boxFilter(imp, -1, (kq, kq), normalize=False)
                crowd = np.zeros(nb_, bool)
                crowd[1:] = cnt[cy_, cx_] >= 2.5
                if crowd.any():
                    kc = max(3, int(round(0.012 * wf)) | 1)
                    cm = cv2.dilate(crowd[lab_].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
                    mole = mole * (1.0 - cm)
        except Exception:
            pass
    deep = np.maximum(deep, mole)
    if EYE_HOLD > 0 and lm is not None and np.isfinite(np.asarray(lm)[36:48, :2]).all():
        try:
            eh_ = _eye_structure(rb, base, lm, scale, (x0, y0), wf)
            if eh_ is not None:
                deep = np.maximum(deep, np.clip(EYE_HOLD, 0, 1) * eh_)
        except Exception:
            pass
    yb = cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.01 * wf))
    g = np.hypot(cv2.Sobel(yb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(yb, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    edge = _ss(*HOLD_EDGE, g * (0.05 * wf) / np.maximum(cv2.GaussianBlur(yb, (0, 0), max(2.0, 0.03 * wf)), 0.03))
    off = np.zeros_like(roi)
    if Cb is not None and Cr is not None and lm is not None and np.isfinite(lm[27:36, :2]).all():
        hh, ww = Y.shape
        def at(c):
            if c.shape != (hh, ww):
                c = cv2.resize(c, (ww, hh), interpolation=cv2.INTER_LINEAR)
            return cv2.GaussianBlur(c[y0:y1, x0:x1].astype(np.float32), (0, 0), max(1.0, 0.006 * wf))
        cb, cr = at(Cb), at(Cr)
        nb, nr = cb / (yb + 0.03), cr / (yb + 0.03)
        # the face's own skin colour: from the inside of the face (well within its outline), and how much it
        # varies there. Under coloured stage light the lit and the shaded side differ in hue by more than paint
        # differs from skin under white light, so the limits follow the face's own spread (cut 069: a fixed
        # limit took two thirds of the face for "not skin").
        core = np.zeros(roi.shape, np.uint8)
        if np.isfinite(lm[0:27, :2]).all():
            cv2.fillConvexPoly(core, cv2.convexHull(np.round(lm[0:27, :2] * scale - np.array([x0, y0])).astype(np.int32)), 1)
            ke = max(3, int(round(0.24 * wf)) | 1)
            core = cv2.erode(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
        core = (core > 0) & (m[y0:y1, x0:x1] > 0.5)
        lit = core & (yb > 0.2)
        if lit.sum() > 30:
            r0 = np.median(nb[lit]), np.median(nr[lit])
            dist = np.hypot(nb - r0[0], nr - r0[1])
            lo = max(HOLD_OFF[0], float(np.percentile(dist[lit], 95)) + 0.03)
            off = _ss(lo, lo + (HOLD_OFF[1] - HOLD_OFF[0]), dist) * _ss(0.18, 0.28, yb)
        cc = np.hypot(cb, cr)
        col = core & (cc > 0.010)
        if col.sum() > 30:
            # a hue far from the skin's own, wherever there is colour enough to tell (face paint and tattoos
            # in purple or blue, also in shade and blurred by motion: wiped by the smoothing)
            ha = np.arctan2(cr, cb)
            h0 = np.angle(np.mean(np.exp(1j * ha[col])))
            dh = np.degrees(np.abs(np.angle(np.exp(1j * (ha - h0)))))
            lo = max(PAINT_HUE[0], float(np.percentile(dh[col], 95)) + 15.0)
            off = np.maximum(off, _ss(lo, lo + (PAINT_HUE[1] - PAINT_HUE[0]), dh) * _ss(0.010, 0.022, cc))
    kd = max(3, int(round(0.012 * wf)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    sg = max(1.0, 0.006 * wf)
    outs = []
    for a in ((deep, np.maximum(edge, off), off) if parts else (deep, np.maximum(edge, off))):
        o = np.zeros(Y.shape, np.float32)
        o[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(cv2.dilate(a.astype(np.float32), se), (0, 0), sg) * 1.3, 0, 1)
        outs.append(o)
    return tuple(outs) if parts else (outs[0], outs[1])


FOLD_EPS, FOLD_R = 0.005, 0.08


def neck_zone(lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """1 on the neck proper (from the jaw down about half a face width), fading to 0 below it. A thin chain on
    the collarbone or the chest looks to every measure like a neck line (as faint, as thin, skin coloured
    through its links); what tells them apart is where they are. Neck lines are on the neck."""
    if lm is None or not np.isfinite(lm[0:17, :2]).all():
        return None
    def make():
        hull = cv2.convexHull(np.round(lm[:, :2] * scale).astype(np.int32))
        x0, y0, x1, y1 = _window(shape, hull.reshape(-1, 2), 1.0 * wf + 8)      # the chamfer paths need room
        out = np.zeros(shape[:2], np.float32)
        if x1 <= x0 or y1 <= y0:
            return out
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, hull - np.array([x0, y0], np.int32), 1)
        dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
        out[y0:y1, x0:x1] = 1.0 - _ss(0.34 * wf, 0.48 * wf, dist)
        return out
    return _cached("neckzone", lm, shape, scale, wf, make)


def _fold_flat(Y: np.ndarray, m: np.ndarray, wf: float, k: float, eps_mul: float = 1.0, r_mul: float = 1.0) -> np.ndarray:
    """The soft folds of a neck (rings across it, wider and fainter than a line): the faint light and dark
    bands between the size of the skin's texture and a twelfth of the face are taken out with an edge-keeping
    filter, so the strong shading of the neck's own build (tendons, the jaw's shadow) is an edge and stays, and
    the broad light and shade is given back. The skin's fine texture is not touched.
    eps_mul / r_mul: the edge threshold and the width of the edge-keeping filter, as multiples of FOLD_EPS / FOLD_R (the armpit's hollow is softer and wider than a neck's ring)."""
    r = _roi(m, int(0.25 * wf * max(1.0, float(r_mul))) + 4)
    if r is None or k < 1e-4:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1].astype(np.float32)
    loc = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.1 * wf))
    eps = FOLD_EPS * float(eps_mul) * np.clip(loc / 0.45, 0.15, 2.0) ** 2
    b1 = SK.guided_self(roi, max(1, int(round(0.008 * wf))), eps)
    b2 = SK.guided_self(b1, max(2, int(round(FOLD_R * float(r_mul) * wf))), eps)
    d = b2 - b1
    sig = max(3.0, 0.16 * wf)
    low = cv2.GaussianBlur(d * mm, (0, 0), sig) / np.maximum(cv2.GaussianBlur(mm, (0, 0), sig), 0.05)
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * mm * (d - np.clip(low, -0.2, 0.2))
    return out


def _line_kernel(L: float, ang_deg: float) -> np.ndarray:
    """A one pixel wide line of length L px at ang_deg from the picture's vertical (x right, y down), as a kernel that sums to 1."""
    th = np.radians(float(ang_deg))
    ux, uy = float(np.sin(th)), float(np.cos(th))
    half = int(L / 2.0) + 2
    k = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
    cv2.line(k, (int(round(half - ux * L / 2.0)), int(round(half - uy * L / 2.0))), (int(round(half + ux * L / 2.0)), int(round(half + uy * L / 2.0))), 1.0, 1)
    return k / max(float(k.sum()), 1.0)


def _armpit_line_hold(Y: np.ndarray, mA: np.ndarray, wA: float, wfA: float, axes, fp: float):
    """Where the arm's own line lies (0..1, ARMPIT_LINE_HOLD): a long thin dark valley that runs along a hanging upper arm. Looked for on the picture before the armpit stage, with the filler's own valley finder
    (what the filler would take out) kept to the valleys whose direction is the upper arm's (ARMPIT_LINE_TOL) and judged by its mean along the arm (ARMPIT_LINE_LEN): a crease is short or lies across the arm.
    axes: [(the upper arm's angle from the vertical in degrees, weight, zone centre x, y, radius)] in proxy px, fp: proxy px to this plane's px. Returns the map of the plane's size, or None."""
    r = _roi(mA, int(3 * wA) + 6)
    if r is None or not axes:
        return None
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = mA[y0:y1, x0:x1].astype(np.float32)
    # the filler's valley score (_crease_lift)
    sig_s = max(0.5, 0.25 * wA)
    sig_l = max(2.0, 2.4 * wA)
    Ys = cv2.GaussianBlur(roi, (0, 0), sig_s)
    wsk = (mm > 0.05).astype(np.float32)
    b0 = _gm(Ys, wsk, sig_l)
    dev = Ys - b0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sig_l), 0)) + 0.004
    w = wsk * np.exp(-(np.clip(-dev, 0, None) / tex) ** 2)
    tex = np.sqrt(np.maximum(_gm(dev * dev, w, sig_l), 0)) + 0.004
    base = _gm(Ys, w, sig_l)
    D = np.clip(base - Ys, 0, 0.25)
    Yh = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.6 * wA))
    Ixx = cv2.Sobel(Yh, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(Yh, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(Yh, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc_ = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy)
    l1 = 0.5 * (tr + disc_)
    l2 = 0.5 * (tr - disc_)
    line = np.where(l1 > 1e-6, 1 - _ss(0.25, 0.6, np.abs(l2) / np.maximum(l1, 1e-6)), 0.0).astype(np.float32)
    score = _ss(1.0, 2.6, D / tex) * _ss(0.004, 0.02, D) * line
    kn = max(3, int(round(1.6 * wA)) | 1)
    kw = max(kn + 4, int(round(6.0 * wA)) | 1)
    bn = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
    bw = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw)))
    tlo_, thi_ = (THIN_LO, THIN_HI) if ARMPIT_THIN is None else (float(ARMPIT_THIN[0]), float(ARMPIT_THIN[1]))
    score = score * _ss(tlo_, thi_, cv2.GaussianBlur(bn, (0, 0), sig_s * 2) / (cv2.GaussianBlur(bw, (0, 0), sig_s * 2) + 1e-4))
    if ARMPIT_FINE:
        try:
            fine = SK._strand_map(np.clip(roi, 0, 1) * 255.0, wA / 0.018, dark_only=True, gain=1.6)
            if fine is not None:
                score = np.maximum(score, fine)
        except Exception:
            pass
    score = score.astype(np.float32)
    # the direction of the valley: across it is the direction of the strongest curvature
    th = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy) + np.pi / 2.0
    ang = (np.degrees(np.arctan2(np.cos(th), np.sin(th))) + 90.0) % 180.0 - 90.0
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    H = np.zeros(roi.shape, np.float32)
    lo_, hi_ = float(ARMPIT_LINE_THR[0]), float(ARMPIT_LINE_THR[1])
    tol = float(ARMPIT_LINE_TOL)
    for ax_deg, gate, cx, cy, rad in axes:
        if gate <= 1e-3:
            continue
        dz = np.hypot(gx - cx * fp, gy - cy * fp) / max(1.0, rad * fp)
        disc = (1.0 - _ss(1.0, 1.35, dz)).astype(np.float32)
        da = np.abs(ang - float(ax_deg)) % 180.0
        da = np.minimum(da, 180.0 - da)
        sa = (score * (1.0 - _ss(tol, tol + 15.0, da)) * disc).astype(np.float32)
        if float(sa.max()) < 0.15:
            continue
        S2 = cv2.filter2D(sa, -1, _line_kernel(max(5.0, ARMPIT_LINE_LEN * wfA), ax_deg))
        core = (sa > 0.15).astype(np.float32)
        rd = max(1, int(round(0.5 * wA)))
        near = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
        Hi = (_ss(lo_, hi_, S2) * near).astype(np.float32)
        if ARMPIT_LINE_EXT and ARMPIT_LINE_EXT > 0:
            # where the line gets too faint for the finder it goes on along the arm: a valley of the same direction next to a held part
            ke = (_line_kernel(max(5.0, ARMPIT_LINE_EXT * wfA), ax_deg) > 0).astype(np.uint8)
            He = cv2.dilate(Hi, ke) * (D > 0.004) * (1.0 - _ss(tol + 10.0, tol + 30.0, da)) * disc
            Hi = np.maximum(Hi, He.astype(np.float32))
        H = np.maximum(H, Hi * float(gate))
    if float(H.max()) < 1e-3:
        return None
    H = np.clip(cv2.GaussianBlur(H, (0, 0), max(0.8, 0.4 * wA)) * 1.4, 0, 1).astype(np.float32)
    out = np.zeros(Y.shape, np.float32)
    out[y0:y1, x0:x1] = H
    return out


ARMPIT_LONG_KEEP = 0.0     # 0 = off. 2611 IWALY c029 f69/f136/f159/f103, c010 f4/f22 (2026-11-22, user's deck: an arm's line "사라진 오류", "연해진", hair strands and the
                           # stray hairs on the back retouched, a garment's shadow line gone): the armpit stage's line lifter takes every thin valley in its zone for a crease. An
                           # armpit's creases are short; an arm's outline, a hair strand or a strap's shadow is a long straight line. Every thin dark line of the picture before the
                           # stage that is longer than ARMPIT_LONG_SPAN face widths and ARMPIT_LONG_ELONG times longer than wide is given back (its detail finer than the line's width)
ARMPIT_LONG_SPAN = 0.40
ARMPIT_LONG_ELONG = 5.0
ARMPIT_LONG_THR = 0.03     # how much darker than what lies round it (relative) a line must be to be looked at


def _armpit_long_lines(Y0: np.ndarray, m: np.ndarray, wA: float, wf: float) -> Optional[np.ndarray]:
    """0..1: the long straight thin dark lines of Y0 (hair strands, an arm's outline, a strap's shadow) inside m (> 0.05). None: none."""
    r = _roi(m, int(ARMPIT_LONG_SPAN * wf) + 8)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = Y0[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1] > 0.05
    if not mm.any():
        return None
    rb = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.3 * wA))
    k = max(3, int(round(3.0 * wA)) | 1)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 2.0 * wA)) + 0.02
    bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) / base
    cand = ((bh > ARMPIT_LONG_THR) & cv2.dilate(mm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    span_min = ARMPIT_LONG_SPAN * wf
    for i in range(1, n):
        if max(st[i, 2], st[i, 3]) < 0.5 * span_min or st[i, 4] < 8:
            continue
        ys, xs = np.nonzero(lab[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] == i)
        pts = np.c_[xs, ys].astype(np.float64)
        cv_ = np.cov((pts - pts.mean(0)).T)
        ev = np.linalg.eigvalsh(cv_)
        length = 4.0 * float(np.sqrt(max(ev[1], 1e-9)))
        width = 4.0 * float(np.sqrt(max(ev[0], 1e-9)))
        if length >= span_min and length >= ARMPIT_LONG_ELONG * max(width, 1.0):
            keep[i] = True
    if not keep.any():
        return None
    H = keep[lab].astype(np.uint8)
    kd = max(3, int(round(1.0 * wA)) | 1)
    H = cv2.dilate(H, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))).astype(np.float32)
    H = np.clip(cv2.GaussianBlur(H, (0, 0), max(1.0, 0.5 * wA)) * 1.3, 0, 1)
    out = np.zeros(Y0.shape, np.float32)
    out[y0:y1, x0:x1] = H
    return out


def _line_restore(Y: np.ndarray, Ysrc: np.ndarray, H: np.ndarray, sig: float) -> np.ndarray:
    """Y with the detail finer than sig of Ysrc given back where H (0..1) is: Y + H * (hp(Ysrc) - hp(Y)), hp = the picture minus its Gaussian blur of sig. The soft light and shade stays Y's."""
    r = _roi(H, int(3 * sig) + 2)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    a = Y[y0:y1, x0:x1].astype(np.float32)
    b = Ysrc[y0:y1, x0:x1].astype(np.float32)
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = a + H[y0:y1, x0:x1] * ((b - cv2.GaussianBlur(b, (0, 0), sig)) - (a - cv2.GaussianBlur(a, (0, 0), sig)))
    return out


FROWN_LINES = 1.0     # 0..1: thin frown lines between the brows and across the top of the nose (bunny lines)


def frown_zone(lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """Between the brows and down the bridge to the middle of the nose, a little to either side: where a frown
    puts its lines (vertical between the brows, slanted across the top of the nose). The nose's own side
    shading is broad and the crease filler leaves broad shading alone, so it is not held here."""
    if lm is None or not np.isfinite(lm[17:36, :2]).all():
        return None
    def make():
        pts = np.vstack([lm[[19, 20, 21, 22, 23, 24], :2], lm[27:31, :2], lm[[39, 42], :2]]) * scale
        hull = cv2.convexHull(np.round(pts).astype(np.int32))
        k = max(3, int(round(0.08 * wf)) | 1)
        sig = max(1.0, 0.03 * wf)
        x0, y0, x1, y1 = _window(shape, hull.reshape(-1, 2), k + 6 * sig + 4)
        out = np.zeros(shape[:2], np.float32)
        if x1 <= x0 or y1 <= y0:
            return out
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, hull - np.array([x0, y0], np.int32), 255)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        out[y0:y1, x0:x1] = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sig)
        return out
    return _cached("frown", lm, shape, scale, wf, make)


# 2611 IWALY c001 (2026-11-22, user, 00:00:31:14: the red circle on the eye, "mask error"): on a three-quarter view the brow and nose landmarks sit on the eye, so the frown zone
# (their hull) lay over the eye's outer half and its lash fan, and the thin-line filler took every lash for a crease and lifted it by up to 7 % of the picture's brightness (the
# skin smoothing itself never more than 2 %). The ground round an eye is no ground of the thin-line fillers: LASH_REACH of the face width from the eye's own hole in the skin
# mask (the hole that holds the eye landmarks' centre), held in full to 0.6 of that and fading out at the reach. 0 = off.
LASH_REACH = 0.12
LASH_HOLE_MAX = 0.45      # a hole wider or higher than this share of the face width is no single eye (it has run into the hair or the background): there is no lash ground then


def _lash_zone(ex: Optional[np.ndarray], lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """0..1 at shape: the ground within LASH_REACH of the face width from each eye's own hole in ex (1 = not skin: the parser's eyes, brows, lips, hair, plus the features' polygons),
    where the thin-line fillers leave every dark strand alone (a lash is a thin dark line on the skin too). The eye's hole is the one that holds the centre of its landmarks 36-41 /
    42-47 (or lies within a twelfth of the face of it); None where there is no such hole or ex is not there."""
    if LASH_REACH <= 0 or ex is None or lm is None or not wf > 8:
        return None
    P_ = np.asarray(lm, np.float32)[:, :2] * scale
    if not np.isfinite(P_[36:48]).all():
        return None
    h, w = int(shape[0]), int(shape[1])
    reach = LASH_REACH * wf
    R = int(round(0.5 * LASH_HOLE_MAX * wf + reach + 0.05 * wf))
    out = None
    for idx in (range(36, 42), range(42, 48)):
        c = P_[list(idx)].mean(0)
        cx, cy = int(round(float(c[0]))), int(round(float(c[1])))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        x0, x1 = max(0, cx - R), min(w, cx + R + 1)
        y0, y1 = max(0, cy - R), min(h, cy + R + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        hole = (np.asarray(ex[y0:y1, x0:x1], np.float32) > 0.5).astype(np.uint8)
        if not hole.any():
            continue
        nl, lab = cv2.connectedComponents(hole, connectivity=8)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        near = (hole > 0) & (((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.08 * wf) ** 2)
        if not near.any():
            continue
        labs, cnt = np.unique(lab[near], return_counts=True)
        comp = lab == int(labs[int(np.argmax(cnt))])
        ys, xs = np.nonzero(comp)
        if (xs.max() - xs.min() + 1) > LASH_HOLE_MAX * wf or (ys.max() - ys.min() + 1) > LASH_HOLE_MAX * wf:
            continue
        if xs.min() <= 0 or ys.min() <= 0 or xs.max() >= comp.shape[1] - 1 or ys.max() >= comp.shape[0] - 1:
            continue
        dist = cv2.distanceTransform((~comp).astype(np.uint8), cv2.DIST_L2, 3)
        z = (1.0 - _ss(0.6 * reach, reach, dist)).astype(np.float32)
        if out is None:
            out = np.zeros((h, w), np.float32)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], z)
    return out


SKIN_THINGS_OFF_FACE = True
SKIN_EDGE_FREE = True


def _face_zone(lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """1 inside the face outline (jaw line and forehead arc), soft edged, at this resolution."""
    if lm is None or not np.isfinite(lm[0:27, :2]).all():
        return None

    def make():
        poly = SK.face_polygon(np.asarray(lm, np.float32), SK.FOREHEAD_PARSED)
        if poly is None:
            return None
        z = np.zeros(shape, np.uint8)
        cv2.fillPoly(z, [np.round(poly * scale).astype(np.int32)], 255)
        return cv2.GaussianBlur(z.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.02 * wf))
    return _cached("facezone", lm, shape, scale, wf, make)


def nose_hold(lm: Optional[np.ndarray], shape, scale: float, wf: float, reach: Optional[float] = None) -> Optional[np.ndarray]:
    """1 off the nose, 0 on it (soft): the lines down the sides of the nose are its shape. No line filler may
    lift them (on a soft or motion-blurred face they are all that is left of the nose). reach: how far past the
    nose's outline it is held, of the face width (default NOSE_REACH)."""
    if lm is None or not np.isfinite(lm[27:36, :2]).all():
        return None
    rch = NOSE_REACH if reach is None else float(reach)

    def make():
        pts = np.array(lm[27:36, :2], np.float64)
        if NOSE_SIDES and lm.shape[0] >= 48 and np.isfinite(lm[[39, 42], :2]).all():
            base_w = float(np.linalg.norm(lm[35, :2] - lm[31, :2]))
            c_ = 0.5 * (lm[31, :2] + lm[35, :2])
            wing = []
            for i_ in (31, 35):
                d_ = lm[i_, :2] - c_
                n_ = float(np.linalg.norm(d_))
                if n_ > 1e-6:
                    wing.append(lm[i_, :2] + d_ / n_ * NOSE_SIDES_WING * base_w)
            pts = np.vstack([pts, lm[[39, 42], :2]] + ([np.array(wing)] if wing else []))
        hull = cv2.convexHull(np.round(pts * scale).astype(np.int32))
        k = max(3, int(round(rch * wf)) | 1)
        sig = max(1.0, 0.02 * wf)
        x0, y0, x1, y1 = _window(shape, hull.reshape(-1, 2), k + 8 * sig + 4)
        out = np.ones(shape[:2], np.float32)
        if x1 <= x0 or y1 <= y0:
            return out
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, hull - np.array([x0, y0], np.int32), 255)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        out[y0:y1, x0:x1] = 1.0 - cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sig)
        return out
    return _cached(("nose" if reach is None else f"nose{rch:.4f}") + ("s%.3f" % NOSE_SIDES_WING if NOSE_SIDES else ""), lm, shape, scale, wf, make)


def _exterior_guard(exY: np.ndarray, lm: Optional[np.ndarray], shape, y_per_src: float, wf: float, band: np.ndarray) -> Optional[np.ndarray]:
    """0 at the face's outer edge, 1 from EXT_GUARD[1] of the face width in (soft), at this resolution, for what a fold's band may work on. The exterior is what the exclusion
    (exY: features and what is not skin) holds that is not a feature (feature_exclusion: eyes, lips, nostrils): hair, the background, a hand. None where there is no exterior
    near the band (or it cannot be made)."""
    if not EXT_GUARD or exY is None or lm is None:
        return None
    try:
        g0, g1 = float(EXT_GUARD[0]), float(EXT_GUARD[1])
        r = _roi(band, int(g1 * wf) + 6)
        if r is None:
            return None
        y0, y1, x0, x1 = r
        lmc = np.array(lm, np.float32, copy=True)
        if not np.isfinite(lmc[:, :2]).all():
            return None
        lmc[:, :2] = lmc[:, :2] * float(y_per_src) - np.array([x0, y0], np.float32)
        fx = feature_exclusion(lmc, (y1 - y0, x1 - x0))
        nonf = np.clip(np.clip(exY[y0:y1, x0:x1].astype(np.float32), 0, 1) - fx, 0.0, 1.0)
        ext = nonf > 0.5
        if EXT_OPEN > 0 and bool(ext.any()):
            ko = max(3, int(round(EXT_OPEN * wf)) | 1)
            ext = cv2.morphologyEx(ext.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))) > 0
        if not bool(ext.any()):
            return None
        d = cv2.distanceTransform((~ext).astype(np.uint8), cv2.DIST_L2, 3)
        out = np.ones(shape[:2], np.float32)
        out[y0:y1, x0:x1] = _ss(g0 * wf, g1 * wf, d)
        return out
    except Exception:
        return None


def _outline_guard(lm: Optional[np.ndarray], shape, y_per_src: float, wf: float, band: np.ndarray) -> Optional[np.ndarray]:
    """0 outside the face's outline from the landmarks (jaw line and forehead arc) and at it, 1 from EXT_GUARD[1] of the face width inside, at this resolution, for what a fold's band may work on:
    the parser's exterior (_exterior_guard) is blind where the background has the skin's colour (a beige wall). None where it cannot be made (EXT_OUTLINE off, no landmarks)."""
    if not EXT_GUARD or not EXT_OUTLINE or lm is None or not wf > 8:
        return None
    try:
        g0, g1 = float(EXT_GUARD[0]), float(EXT_GUARD[1])
        fz = _face_zone(lm, shape, y_per_src, wf)
        if fz is None:
            return None
        r = _roi(band, int(g1 * wf) + 6)
        if r is None:
            return None
        y0, y1, x0, x1 = r
        inside = (np.asarray(fz)[y0:y1, x0:x1] > 0.5).astype(np.uint8)
        if not bool(inside.any()):
            return None
        d = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        out = np.ones(shape[:2], np.float32)
        out[y0:y1, x0:x1] = _ss(g0 * wf, g1 * wf, d)
        return out
    except Exception:
        return None


def neck_dir(mY: np.ndarray, lm: Optional[np.ndarray], scale: float):
    """The direction the neck runs in, from the chin to the middle of the neck mask (so a tilted head, whose
    face points one way while the neck goes another, is followed); the face's own axis when that fails."""
    try:
        if lm is not None and np.isfinite(lm[8, :2]).all():
            ys, xs = np.nonzero(mY > 0.3)
            if len(xs) > 50:
                v = np.array([xs.mean(), ys.mean()]) - lm[8, :2] * scale
                n = float(np.hypot(*v))
                if n > 4:
                    return (float(v[0] / n), float(v[1] / n))
    except Exception:
        pass
    return neck_axis(lm)


def _crease_lift(Y, Cb, Cr, mY, k: float, width: float, axis=None, hold: Optional[np.ndarray] = None,
                 fine_lines: bool = True, extra: Optional[np.ndarray] = None, gain: float = 1.0, thin_rng=None, band: float = 0.0):
    """Fill thin dark creases (neck lines): a dark valley, curved strongly across and hardly along, darker than
    the skin around it by more than its texture. Only the crease's shading is lifted; broad shading stays.
    gain: this many times the fill, held to the valley's depth against the skin beside it (and to what is held as a thing): a valley is filled up to its skin's level at most.
    thin_rng: (lo, hi) in place of THIN_LO, THIN_HI: how thin against its own surroundings a valley must be to count as a crease (a valley that lies in a broad shade is thin to a lesser degree)
    band: (0..1) a crease is a dark valley with a light lip beside it, and filling the valley leaves the lip as a line: round the lines found (CREASE_BAND_THR), CREASE_BAND_HW of the line width on either
    side, the soft level of the picture is replaced by what the skin round that band holds, at this share; the grain finer than a pixel stays

    axis: the direction the neck runs in. A neck line lies across the neck; the shading of the neck's own
    build (the tendons and the hollow between them, the edge of the jaw's shadow) runs along it, and was lifted
    as if it were a crease, which made the neck's shading lighter than the original. With an axis only valleys
    lying across it are taken. hold: 0 where nothing may be lifted (the jaw line)."""
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
    # thin only: a neck line is all there under a narrow black-hat; the shading of the neck's own build (the
    # tendons, the hollow between them, the shadow under the jaw) keeps growing as the kernel widens
    kn = max(3, int(round(1.6 * width)) | 1)
    kw = max(kn + 4, int(round(6.0 * width)) | 1)
    bn = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
    bw = cv2.morphologyEx(Ys, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kw)))
    tlo_, thi_ = (THIN_LO, THIN_HI) if thin_rng is None else (float(thin_rng[0]), float(thin_rng[1]))
    thin = _ss(tlo_, thi_, cv2.GaussianBlur(bn, (0, 0), sig_s * 2) / (cv2.GaussianBlur(bw, (0, 0), sig_s * 2) + 1e-4))
    score = score * thin
    if axis is not None:
        # the direction of strongest curvature is across the valley: for a fold lying across the neck that is
        # along the neck. The neck's own shading (tendons, the hollow between them) lies the other way
        phi = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        along = np.abs(np.cos(phi) * axis[0] + np.sin(phi) * axis[1])
        score = score * _ss(0.6, 0.85, cv2.GaussianBlur(along, (0, 0), max(1.0, 0.5 * width)))
    # faint long lines: in grain they are below the skin's own texture at any one point and only show along
    # their length, so they are found by the line finder that also finds hair strands (dark lines only). A
    # hair over the neck is not in the mask (the skin mask leaves it out), so only neck lines are lifted here
    fine = None
    try:
        if fine_lines:
                fine = SK._strand_map(np.clip(roi, 0, 1) * 255.0, width / 0.018, dark_only=True, gain=1.6)
    except Exception:
        fine = None
    if fine is not None:
        score = np.maximum(score, fine)
    if extra is not None:
        # lines found by detect.py: several widths, any direction, held steady through time (1.19). They are filled
        # like the fine lines: from the picture itself, and kept off chains, cords, hair and paint by the test below
        ex = extra[y0:y1, x0:x1].astype(np.float32)
        score = np.maximum(score, ex)
        fine = ex if fine is None else np.maximum(fine, ex)
    if hold is not None:
        score = score * hold[y0:y1, x0:x1]
    rd = max(1, int(round(0.4 * width)))
    score = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
    sp = np.clip(cv2.GaussianBlur(score, (0, 0), max(0.6, 0.3 * width)) * 1.3, 0, 1) * mm
    Yo = Y.astype(np.float32, copy=True)
    lift = sp * D
    rs = cv2.GaussianBlur(roi, (0, 0), 0.8)
    D2 = np.clip(base - rs, 0, 0.25)
    if fine is not None:
        # a line a couple of pixels wide is filled from the picture itself, not from its blurred copy
        fs = cv2.dilate(fine, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        fs = np.clip(cv2.GaussianBlur(fs, (0, 0), max(0.6, 0.2 * width)) * 2.2, 0, 1) * np.clip(mm * 1.5, 0, 1)
        if hold is not None:
            fs = fs * hold[y0:y1, x0:x1]
        lift = np.maximum(lift, fs * D2)
    # a necklace chain, a cord, a strap, a hair is a thin dark line on the skin too. What tells it from a skin
    # line: it is far darker than the skin beside it, it glints (beads, links), or it has lost the skin's
    # colour (a crease is skin in shade and keeps it; a chain or a black cord is grey). None of these is lifted
    kd = max(3, int(round(1.2 * width)) | 1)
    sed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * kd, 3 * kd))
    rq = cv2.GaussianBlur(roi, (0, 0), 1.5)                 # grain alone reaches a fifth of the brightness at 0.8
    deep = cv2.dilate(np.clip(base - rq, 0, 1) / (base + 0.02), sed)
    glint = cv2.dilate(cv2.morphologyEx(rq, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
                       / (base + 0.02), seg)
    thing = np.maximum(_ss(LINE_DEEP_LO, LINE_DEEP_HI, deep), _ss(0.3, 0.5, glint))
    try:
        if Cb is not None and Cr is not None:
            hh, ww = Y.shape
            cm = np.hypot(cv2.resize(Cb, (ww, hh), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1] if Cb.shape != Y.shape else Cb[y0:y1, x0:x1],
                          cv2.resize(Cr, (ww, hh), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1] if Cr.shape != Y.shape else Cr[y0:y1, x0:x1])
            cm = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 1.5)
            cref = _gm(cm, w, sig_l) + 1e-4
            grey = cv2.dilate(np.clip(1.0 - cm / cref, 0, 1), sed)
            thing = np.maximum(thing, _ss(0.5, 0.75, grey) * _ss(0.015, 0.03, cref))
        dc = _chroma_off(Cb, Cr, Y.shape, (y0, y1, x0, x1), width / 0.018)
        if dc is not None:
            thing = np.maximum(thing, cv2.dilate(_ss(*PAINT_DC, dc), sed))      # paint, tattoo, drawn lines
    except Exception:
        pass
    free = 1.0 - np.clip(cv2.GaussianBlur(thing, (0, 0), max(1.0, 0.5 * width)) * 1.5, 0, 1)
    lift = lift * free
    if gain is not None and abs(float(gain) - 1.0) > 1e-3:
        lift = np.minimum(float(gain) * lift, np.maximum(D, D2) * free)
    Yo[y0:y1, x0:x1] = roi + k * lift
    if band and band > 1e-4:
        crease_ = (sp if fine is None else np.maximum(sp, fs)) * free
        core_ = (crease_ > CREASE_BAND_THR).astype(np.float32)
        if float(core_.max()) > 0:
            Ycur = Yo[y0:y1, x0:x1]
            hw_ = max(2, int(round(CREASE_BAND_HW * width)))
            bm_ = cv2.dilate(core_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hw_ + 1, 2 * hw_ + 1)))
            b_ = np.clip(cv2.GaussianBlur(bm_, (0, 0), max(1.0, 0.5 * width)) * 1.3, 0, 1) * mm
            wgt_ = (1.0 - np.clip(cv2.GaussianBlur(bm_, (0, 0), 1.0), 0, 1)) * wsk
            si_ = max(2.0, CREASE_BAND_SIG * width)
            den_ = cv2.GaussianBlur(wgt_, (0, 0), si_)
            lpi_ = cv2.GaussianBlur(Ycur * wgt_, (0, 0), si_) / np.maximum(den_, 1e-4)
            conf_ = np.clip(den_ / 0.25, 0, 1)
            lpy_ = cv2.GaussianBlur(Ycur, (0, 0), 1.0)
            Yo[y0:y1, x0:x1] = Ycur + float(band) * b_ * conf_ * (lpi_ - lpy_)
    return Yo, Cb, Cr


REDNESS_KEEP_MEAN = True
# 2611 cut 024 (user 2026-11-17 21:57: at 강하게 the blush on the cheeks was gone). The red excess was measured
# against one reference for the whole face, so a broad cheek colour (blush, a flush that is part of the look) was
# taken as a red patch and pulled to the face's colour. Now the reference follows the colour of the skin around each
# point over REDNESS_BROAD of the face width: patches smaller than that (spots, blotches, the redness of a pimple)
# are evened out, a colour spread wider than that stays.
REDNESS_BROAD = 0.07
GRAIN_KEEP = True        # the skin smoothing gives the original's finest band (film grain, sensor noise) back
GRAIN_SIGMA = 1.2        # that band, in pixels of a 1080-line picture (scaled with the plane's height)


def _keep_fine(before: np.ndarray, after: np.ndarray, m: np.ndarray, rows: Optional[int] = None) -> np.ndarray:
    """after with the finest detail band of before inside m: whatever the smoothing took of the grain is put back,
    so every strength keeps the footage's own grain. The band follows the plane's size (about a pixel at 1080);
    rows: the plane's height when before and after are a piece of it."""
    if before.shape != after.shape or m is None or m.shape != after.shape:
        return after
    sg = float(np.clip(GRAIN_SIGMA * (rows or after.shape[0]) / 1080.0, 0.8, 2.4))
    fb = before - cv2.GaussianBlur(before, (0, 0), sg)
    fa = after - cv2.GaussianBlur(after, (0, 0), sg)
    # all of it wherever the smoothing reached (scaled by the mask, the partly smoothed edges kept less grain)
    return after + np.clip(m * 4.0, 0, 1) * (fb - fa)


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
    if REDNESS_BROAD > 0:
        broad = _gm(crs, mm, max(2.0, REDNESS_BROAD * wf * cs))
        ex = np.clip(crs - np.maximum(broad, cr_ref), 0, 0.12)
    else:
        ex = np.clip(crs - cr_ref, 0, 0.12)
    on = _ss(0.0, 0.025, ex)
    Cro, Cbo = Cr.astype(np.float32, copy=True), Cb.astype(np.float32, copy=True)
    dcr = -k * 0.85 * mm * ex
    dcb = k * 0.5 * mm * on * np.clip(cb_ref - cbs, 0, 0.06)
    if REDNESS_KEEP_MEAN:
        # the face keeps its own average colour: only the patches are evened out (cut 009, user 16:46: at 강하게 the
        # skin read paler and cooler, the redness step alone took dE 0.6-0.7 off the face's mean colour)
        wsum = float((mm * mm).sum()) + 1e-6
        dcr = dcr - mm * (float((dcr * mm).sum()) / wsum)
        dcb = dcb - mm * (float((dcb * mm).sum()) / wsum)
    Cro[c0:c1, d0:d1] = cr + dcr
    Cbo[c0:c1, d0:d1] = cb + dcb
    # red patches also read a little dark
    exY = cv2.resize(ex * mm, (0, 0), fx=Y.shape[1] / Cr.shape[1], fy=Y.shape[0] / Cr.shape[0],
                     interpolation=cv2.INTER_LINEAR) if Cr.shape != Y.shape else ex * mm
    yy0, yx0 = int(c0 * Y.shape[0] / Cr.shape[0]), int(d0 * Y.shape[1] / Cr.shape[1])
    Yo = Y.astype(np.float32, copy=True)
    hh, ww = min(exY.shape[0], Y.shape[0] - yy0), min(exY.shape[1], Y.shape[1] - yx0)
    Yo[yy0:yy0 + hh, yx0:yx0 + ww] += k * 0.25 * exY[:hh, :ww]
    return Yo, Cbo, Cro


BODY_FINE = 0.0048         # of the size (BodyInfo.ref): what is finer than this is the fine texture 결 보존 gives back
BODY_HOLD = True           # things lying on the skin, moles and ink are left out of the body smoothing


BODY_INSET = 0.05          # of the size: the smoothing stays this far inside the skin's edge (a limb's outline, a hem)
                           # and away from whatever is held (about 2.5 times its widest band)


def body_inner(m: np.ndarray, ref: float) -> np.ndarray:
    """m kept away from its own edge by BODY_INSET. The smoothing thins bands of detail up to 2 % of the size wide;
    at the outline of an arm or a leg (skin against the background, against cloth) those bands hold the edge itself,
    and thinning them there softened the body's outline. The rim left out is where the skin turns away from the
    camera."""
    r_ = _roi(m, 2)
    if r_ is None:
        return m
    y0, y1, x0, x1 = r_
    re_ = max(2, int(round(BODY_INSET * ref)))
    roi = m[y0:y1, x0:x1]
    core = cv2.erode((roi > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * re_ + 1, 2 * re_ + 1)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=1)
    soft = np.clip(cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(1.0, 0.5 * re_)) * 1.25, 0, 1)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = np.minimum(roi, soft)
    return out


def body_hold(Y, Cb, Cr, mY, ref: float) -> Optional[np.ndarray]:
    """What on the body is not skin texture and stays as it is under the smoothing (0..1 at Y's size): straps,
    chains, cords and hair lying on the skin, lace, glints (thin, far darker or brighter than the skin, or of
    another colour); moles and other compact dark marks; ink and paint (another hue); hard edges."""
    hY, wY = Y.shape
    out = None
    th_ = _things(Y, mY, ref, Cb, Cr)
    if th_ is not None:
        out = th_
    hd_ = _hold_maps(Y, None, None, mY, ref, None, 1.0)[0]
    if hd_ is not None:
        out = hd_ if out is None else np.maximum(out, hd_)
    cbY = Cb if Cb.shape == Y.shape else cv2.resize(Cb, (wY, hY), interpolation=cv2.INTER_LINEAR)
    crY = Cr if Cr.shape == Y.shape else cv2.resize(Cr, (wY, hY), interpolation=cv2.INTER_LINEAR)
    pr_ = DT.protect(Y, cbY, crY, mY, ref)
    out = pr_ if out is None else np.maximum(out, pr_)
    return out


BODY_PIECE = True          # the body smoothing is made on the area's own piece of the planes, with room for the widest
                           # of its filters round it: the same numbers. False: on the whole planes (1.23's first
                           # build: a second of a thread for each person on each frame of a 4K export, j2796)


def _body_window(mY: np.ndarray, shape_c, pad: int):
    """The piece of the planes the body smoothing needs: (y0, y1, x0, x1) on the luma plane, on the chroma planes'
    own lines. None: nearly the whole plane, or planes whose sizes do not divide."""
    hY, wY = mY.shape
    hC, wC = shape_c
    if hC < 1 or wC < 1 or hY % hC or wY % wC:
        return None
    fy, fx = hY // hC, wY // wC
    r_ = _roi(mY, pad)
    if r_ is None:
        return None
    y0, y1, x0, x1 = r_
    y0, x0 = y0 // fy * fy, x0 // fx * fx
    y1, x1 = min(hY, -(-y1 // fy) * fy), min(wY, -(-x1 // fx) * fx)
    if (y1 - y0) * (x1 - x0) > 0.85 * hY * wY:
        return None
    return y0, y1, x0, x1


def _body_smooth(Y, Cb, Cr, mY, job, y_per_src: float, cs: float, src: Optional[dict] = None):
    """The skin smoothing of the face (skin.smooth_channel: the texture's bands thinned, nothing wider than a few
    pores changed, the picture's grain given back) on the body's area mY, its bands following the body's size.
    Light and shade, the body's outline and whatever lies on the skin are wider than the bands or held (body_hold).
    job.body_keep (결 보존) gives the finest band (pores, fine hair) back by that share.

    Every filter of it looks no further than a tenth of the body's size from the area, so it is made on the area's
    own piece of the planes (_body_window) and laid back: an arm in a 4K frame is a twentieth of the plane."""
    hY, wY = Y.shape
    k = float(np.clip(job.v.get("body_skin", 0.0), 0, 1))
    ref = float(job.body["ref"]) * y_per_src
    if k < 1e-4 or ref < 4 or float(mY.max()) < 0.01:
        return Y, Cb, Cr
    win = None
    if BODY_PIECE and mY.shape == Y.shape and Cb.shape == Cr.shape:
        r = int(max(1, round(0.022 * ref)))
        rc = max(1, int(round(r * cs)))
        up = max(hY // max(Cb.shape[0], 1), wY // max(Cb.shape[1], 1), 1)
        # the widest reach of what follows (detect.protect, _things, _hold_maps: a tenth of the size; the bands; the
        # fine band given back; the colour's radius on its own plane; the grain's band), and the one line of a piece
        # a chroma plane enlarged to the luma's size does not have right at the piece's edge
        pad = max(int(0.1 * ref) + 6, 3 * r + 4, int(3 * max(1.0, BODY_FINE * ref)) + 2, (3 * rc + 4) * up, 12) + 2
        win = _body_window(mY, Cb.shape, pad)
    if win is None:
        got = _body_smooth_on(Y, Cb, Cr, mY, job, k, ref, cs, src, hY)
        return (Y, Cb, Cr) if got is None else got
    y0, y1, x0, x1 = win
    fy, fx = hY // Cb.shape[0], wY // Cb.shape[1]
    yw = (slice(y0, y1), slice(x0, x1))
    cw = (slice(y0 // fy, y1 // fy), slice(x0 // fx, x1 // fx))
    piece = was = None
    if src is not None and src["Y"].shape == Y.shape:
        # the piece's own views: what is added to dY is added to the plane itself
        piece = {n: src[n][yw] for n in ("Y", "dY", "Jy", "Jcb", "Jcr")}
        was = piece["Y"]
    got = _body_smooth_on(np.ascontiguousarray(Y[yw]), np.ascontiguousarray(Cb[cw]), np.ascontiguousarray(Cr[cw]),
                          np.ascontiguousarray(mY[yw]), job, k, ref, cs, piece, hY)
    if got is None:
        return Y, Cb, Cr
    Y, Cb, Cr = Y.copy(), Cb.copy(), Cr.copy()
    Y[yw], Cb[cw], Cr[cw] = got
    if piece is not None and piece["Y"] is not was:
        sy_ = src["Y"].copy()               # (a new plane, as before: the one handed in stays as it was)
        sy_[yw] = piece["Y"]
        src["Y"] = sy_
    return Y, Cb, Cr


def _body_smooth_on(Y, Cb, Cr, mY, job, k: float, ref: float, cs: float, src: Optional[dict], rows: int):
    """_body_smooth on the planes given (whole, or the area's piece of them; rows: the whole luma plane's height).
    None: nothing to do."""
    hY, wY = Y.shape
    hC, wC = Cb.shape
    if BODY_HOLD:
        hold = body_hold(Y, Cb, Cr, mY, ref)
        if hold is not None:
            mY = mY * (1.0 - np.clip(hold, 0, 1))
    # away from the skin's own edge and from what is held: the bands (and the colour smoothing's radius) reach that far
    mY = body_inner(mY, ref)
    if float(mY.max()) < 0.01:
        return None
    mC = mY if (hC, wC) == (hY, wY) else cv2.resize(mY, (wC, hC), interpolation=cv2.INTER_AREA)
    r = int(max(1, round(0.022 * ref)))
    if hasattr(SK, "_TLS"):
        SK._TLS.wf = float(ref)
        SK._TLS.sq = 1.0
        SK._TLS.shape = None
        SK._TLS.pitm = None
    sig_f = max(1.0, BODY_FINE * ref)
    keep = float(np.clip(getattr(job, "body_keep", 0.6), 0, 1))

    def smooth(P):
        P1 = SK.smooth_channel(P, mY, k, keep, r, relative=True)
        if GRAIN_KEEP:
            P1 = _keep_fine(P, P1, mY, rows)
        if keep > 1e-3:
            d_ = P - P1                                   # what was taken: its finest part comes back
            r_ = _roi(mY, int(3 * sig_f) + 2)
            if r_ is not None:
                y0, y1, x0, x1 = r_
                dd = d_[y0:y1, x0:x1]
                P1 = P1.copy()
                P1[y0:y1, x0:x1] += keep * (dd - cv2.GaussianBlur(dd, (0, 0), sig_f))
        return P1
    if src is not None and src["Y"].shape == Y.shape:
        Ys0 = src["Y"]
        Ys1 = smooth(Ys0)
        dS = Ys1 - Ys0
        src["Y"] = Ys1
        src["dY"] += dS
        Y = Y + src["Jy"] * dS
        if Cb.shape == dS.shape:
            Cb = Cb + src["Jcb"] * dS
            Cr = Cr + src["Jcr"] * dS
    else:
        Y = smooth(Y)
    rc = max(1, int(round(r * cs)))
    Cb = SK.smooth_channel(Cb, mC, k * SKIN_CHROMA, 0.0, rc, eps=0.0004)
    Cr = SK.smooth_channel(Cr, mC, k * SKIN_CHROMA, 0.0, rc, eps=0.0004)
    return Y, Cb, Cr


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


# --------------------------------------------------------------------------- the last pass (2611 IWALY, user 2026-11-23: "콧구멍, 코라인, 볼록잡티, 점, 팔라인, 자갈턱 무조건 해결")
# After every stage: what the user does not want changed is given back from the picture as it came in, whatever stage changed it.
FINAL_NOSE = 0.0           # 0..1: the nose (its bridge, side walls, wings and nostrils) keeps its shape: the change of every stage there wider than the skin's
                           # texture (FINAL_NOSE_SIG of the face width) is taken back; the texture's thinning stays
FINAL_NOSE_WING = 0.38     # the wings' outer points pushed out by this share of the nose base's width
FINAL_NOSE_PAD = 0.025     # the nose's area grown by this share of the face width (soft over the same)
FINAL_NOSE_SIG = 0.008
FINAL_NOSE_HF = 0.006      # of the change finer than FINAL_NOSE_SIG, the part beyond this (display luma) is taken back too: an edge of the
                           # nose (a wing's crease) the retouch smoothed comes back whole, so no light seam is left beside it (c029 f4)
FINAL_MOLE = 0.0           # 0..1: small dark round spots that are not redder than the skin round them (moles), found on the picture as it came in, are given back
FINAL_MOLE_DEPTH = (0.09, 0.14)   # a candidate from the first, kept when its darkest point is past the second (relative to the skin round it)
FINAL_MOLE_RED = 0.012     # its median Cr over the skin round it at most this (redder: a pimple's mark, which goes)
FINAL_MOLE_ALONE = 0.06    # of the face width: at most one other candidate this near (pits of rough skin come many together)
FINAL_MOLE_SIZE = 0.03
FINAL_BUMP = 0.0           # 0..1: a raised spot (a small bright top with a dark side) on the face's skin is filled from the skin round it
FINAL_BUMP_SIZE = 0.035    # of the face width: the largest bump
FINAL_BUMP_LIGHT = (0.06, 0.08)    # relative brightness of its top over the skin round it (candidate, kept)
FINAL_BUMP_TOP = 0.025     # of the face width: the largest bright top
FINAL_BUMP_SHADE = 0.03    # its darker side: at least this much darker than the skin round it (relative), within the top's size
FINAL_BUMP_ALONE = 0.05
FINAL_BUMP_BUSY = 0.12      # share of a ring round it that may hold other light or dark marks
FINAL_BUMP_MAXLIGHT = 0.30         # brighter than this: a glint, left
FINAL_LONG = 0.0           # 0..1: long straight thin dark lines (an arm's outline, hair strands, a strap's shadow) inside an armpit zone or on the body's skin are given back in full
FINAL_LONG_SPAN = 0.30
FINAL_LONG_ELONG = 4.0
FINAL_LONG_THR = 0.025
FINAL_NASO = 0.0           # 0..1: the nasolabial folds (팔자주름) and the shade at the mouth's corners: in a zone made from the landmarks alone (the nose wing to the mouth's
                           # corner and on below it, FINAL_NASO_W of the face width wide), what is darker than its surroundings at the fold's scale (FINAL_NASO_BAND) is lifted;
                           # hair strands (far darker, thin) and the nose / the lips are left. Works where the skin mask or the fold finder missed the fold (a turned, tilted, blurred face)
FINAL_NASO_W = 0.10
FINAL_NASO_NOSE_GAP = 0.04  # of the face width: the zone stays this far from the nose (soft)
FINAL_NASO_BAND = (0.004, 0.06)
FINAL_NASO_EDGE = 0.10     # of the face width: the zone stays this far inside the landmarks' outline of the face
FINAL_NASO_MAXDARK = 0.18  # darker than this (relative to the local level) at the fine scale: a strand or a feature, left
FINAL_CHIN = 0.0           # 0..1: a pebbled chin: the pits and bumps between FINAL_CHIN_BAND of the face width taken out on the chin (the broad shape and the finest texture stay)
FINAL_CHIN_BAND = (0.006, 0.035)
FINAL_CHIN_LIGHT = 0.015     # see _final_chin (None: both sides alike)
FINAL_CHIN_LIGHT_MAX = 0.03


def _nose_points_ok(L: np.ndarray, wf: float) -> np.ndarray:
    """bool per landmark index: the nose's points (27..35) and the inner eye corners (39, 42) that lie where a nose's would (near the nose tip, 30): on a turned
    face the far side's points are put anywhere (c001 f92: the nose base points under the far eye)"""
    ok = np.zeros(len(L), bool)
    t = L[30]
    for i in range(27, 36):
        ok[i] = float(np.linalg.norm(L[i] - t)) <= 0.30 * wf
    for i in (39, 42):
        ok[i] = float(np.linalg.norm(L[i] - t)) <= 0.45 * wf
    return ok


def _nose_give_back(Y: np.ndarray, Y0: np.ndarray, w: np.ndarray, s: float) -> np.ndarray:
    """Y less w times the change wider than the texture (blur of sigma s) and the part of the finer change beyond FINAL_NOSE_HF."""
    d = Y - Y0
    lp = cv2.GaussianBlur(d, (0, 0), s)
    hf = d - lp
    t = float(FINAL_NOSE_HF)
    back = lp + (hf - np.clip(hf, -t, t)) if t >= 0 else lp
    return Y - w * back


def _nose_area(lm: np.ndarray, shape, wf: float) -> Optional[np.ndarray]:
    """0..1: the nose with its side walls (inner eye corners to the wings), its wings and nostrils, grown by FINAL_NOSE_PAD (lm in px of shape)."""
    if lm is None or lm.shape[0] < 48 or not np.isfinite(lm[[27, 28, 29, 30, 31, 32, 33, 34, 35, 39, 42], :2]).all():
        return None
    L = np.asarray(lm[:, :2], np.float64)
    ok = _nose_points_ok(L, wf)
    pts = [L[[i for i in range(27, 36) if ok[i]]]] + [L[[i for i in (39, 42) if ok[i]]]]
    base = [i for i in (31, 32, 33, 34, 35) if ok[i]]
    if len(base) >= 2:
        c = L[base].mean(0)
        bw = max(float(np.linalg.norm(L[base[-1]] - L[base[0]])), 0.12 * wf)
        for i in (31, 35):
            if ok[i]:
                d = L[i] - c
                n = float(np.linalg.norm(d))
                if n > 1e-6:
                    pts.append((L[i] + d / n * FINAL_NOSE_WING * bw)[None])
        # the nostrils' lower rim: a little below the base
        dn = c - L[27] if ok[27] else c - L[30]
        nd = float(np.linalg.norm(dn))
        if nd > 1e-6:
            pts.append((L[base] + dn / nd * 0.12 * bw))
    P = np.vstack([p_ for p_ in pts if len(p_)])
    if len(P) < 3:
        return None
    pad = max(2, int(round(FINAL_NOSE_PAD * wf)))
    x0, y0 = int(max(0, P[:, 0].min() - 3 * pad)), int(max(0, P[:, 1].min() - 3 * pad))
    x1, y1 = int(min(shape[1], P[:, 0].max() + 3 * pad)), int(min(shape[0], P[:, 1].max() + 3 * pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(P - [x0, y0]).astype(np.int32)), 255)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    out = np.zeros(shape, np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.6 * pad)), 0, 1)
    return out


def _feature_holes(lm: np.ndarray, shape, wf: float) -> np.ndarray:
    """bool: the eyes, brows, lips and nostrils (their landmarks' hulls) grown by 0.03 face widths: not skin spots."""
    m = np.zeros(shape, np.uint8)
    e = np.zeros(shape, np.uint8)
    L = np.round(np.asarray(lm)[:, :2]).astype(np.int32)
    for idx in (range(48, 60), range(17, 22), range(22, 27), range(31, 36)):
        cv2.fillConvexPoly(m, cv2.convexHull(L[list(idx)]), 255)
    for idx in (range(36, 42), range(42, 48)):
        cv2.fillConvexPoly(e, cv2.convexHull(L[list(idx)]), 255)
    k = max(3, int(0.06 * wf) | 1)
    ke = max(3, int(0.16 * wf) | 1)          # the eyes with their lashes, lids and catchlights
    return (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) | (cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))) > 0)


def _alone(cen: np.ndarray, ids: list, wf: float, iso: float) -> list:
    """the ids whose centre has at most one other spot candidate (cen[1:]) within iso face widths: rough skin's pits come many together"""
    allc = cen[1:]
    out = []
    for i in ids:
        d = np.hypot(*(allc - cen[i]).T)
        if int((d < iso * wf).sum()) - 1 <= 1:
            out.append(i)
    return out


def _final_moles(Y0: np.ndarray, Cr0: np.ndarray, region: np.ndarray, wf: float, lm: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """0..1: moles on the picture as it came in (Y0, Cr0 at Y's size), inside region (> 0.5): small, round, alone, FINAL_MOLE_DEPTH darker than the skin round them,
    not redder than it (a pimple's mark is)."""
    r = _roi(region, int(0.05 * wf) + 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = Y0[y0:y1, x0:x1].astype(np.float32)
    rb = cv2.GaussianBlur(roi, (0, 0), 0.8)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.03 * wf)) + 0.02
    km = max(3, int(round(FINAL_MOLE_SIZE * wf)) | 1)
    bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))) / base
    cand = (bh > FINAL_MOLE_DEPTH[0]) & (region[y0:y1, x0:x1] > 0.5)
    if lm is not None:
        lmr = np.asarray(lm, np.float32)[:, :2] - [x0, y0]
        cand &= ~_feature_holes(lmr, roi.shape, wf)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    cr = Cr0[y0:y1, x0:x1].astype(np.float32)
    crd = cv2.GaussianBlur(cr, (0, 0), 1.2) - cv2.GaussianBlur(cr, (0, 0), max(2.0, 0.03 * wf))
    ok = []
    for i in range(1, n):
        w_, h_, a_ = st[i, 2], st[i, 3], st[i, 4]
        if max(w_, h_) > FINAL_MOLE_SIZE * wf or a_ < 3 or max(w_, h_) > 2.5 * max(1, min(w_, h_)):
            continue
        sel = lab == i
        if float(bh[sel].max()) < FINAL_MOLE_DEPTH[1] or float(np.median(crd[sel])) > FINAL_MOLE_RED:
            continue
        ok.append(i)
    keep = _alone(cen, ok, wf, FINAL_MOLE_ALONE)
    if not keep:
        return None
    m = np.isin(lab, keep).astype(np.uint8)
    k = max(3, int(round(0.008 * wf)) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    out = np.zeros(Y0.shape, np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(m, (0, 0), max(0.8, 0.003 * wf)) * 1.3, 0, 1)
    return out


def _final_bumps(Y0: np.ndarray, region: np.ndarray, wf: float, lm: Optional[np.ndarray] = None, Ycur: Optional[np.ndarray] = None):
    """(mask 0..1, fill) of raised spots on Y0 inside region (> 0.5): a small top FINAL_BUMP_LIGHT brighter than the skin round it (not a glint), alone, with a darker side
    within its own size. The fill is the skin round it (inpainted). None: none."""
    r = _roi(region, int(FINAL_BUMP_SIZE * wf) + 6)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = Y0[y0:y1, x0:x1].astype(np.float32)
    rb = cv2.GaussianBlur(roi, (0, 0), 0.8)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.03 * wf)) + 0.02
    kb = max(3, int(round(FINAL_BUMP_SIZE * wf)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    th = cv2.morphologyEx(rb, cv2.MORPH_TOPHAT, se) / base
    bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, se) / base
    cand = (th > FINAL_BUMP_LIGHT[0]) & (region[y0:y1, x0:x1] > 0.5)
    if lm is not None:
        lmr = np.asarray(lm, np.float32)[:, :2] - [x0, y0]
        cand &= ~_feature_holes(lmr, roi.shape, wf)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    ok = []
    for i in range(1, n):
        w_, h_, a_ = st[i, 2], st[i, 3], st[i, 4]
        if max(w_, h_) > FINAL_BUMP_TOP * wf or a_ < 3:
            continue
        sel = lab == i
        tmax = float(th[sel].max())
        if tmax < FINAL_BUMP_LIGHT[1] or tmax > FINAL_BUMP_MAXLIGHT:
            continue
        cx, cy = cen[i]
        rr = int(max(w_, h_)) + 3
        win = bh[max(0, int(cy) - rr):int(cy) + rr + 1, max(0, int(cx) - rr):int(cx) + rr + 1]
        if win.size == 0 or float(win.max()) < FINAL_BUMP_SHADE:
            continue
        # calm skin round it (hair, lashes, a hand's edge are busy): few other light or dark marks in a ring of three times its size
        R = 3 * rr
        ring_t = th[max(0, int(cy) - R):int(cy) + R + 1, max(0, int(cx) - R):int(cx) + R + 1]
        ring_b = bh[max(0, int(cy) - R):int(cy) + R + 1, max(0, int(cx) - R):int(cx) + R + 1]
        busy = float(((ring_t > 0.5 * FINAL_BUMP_LIGHT[0]) | (ring_b > 0.5 * FINAL_BUMP_LIGHT[0])).mean())
        if busy > FINAL_BUMP_BUSY:
            continue
        ok.append(i)
    keep = _alone(cen, ok, wf, FINAL_BUMP_ALONE)
    if not keep:
        return None
    m = np.isin(lab, keep).astype(np.uint8)
    k = max(3, int(round(0.4 * FINAL_BUMP_SIZE * wf)) | 1)
    # the bump's own shade (its darker side) next to the top goes with it
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    m = np.maximum(m, (near > 0) & (bh > 0.4 * FINAL_BUMP_SHADE)).astype(np.uint8)
    md = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 2 | 1), max(3, k // 2 | 1))))
    src_ = roi if Ycur is None else Ycur[y0:y1, x0:x1].astype(np.float32)      # filled from the picture as it is now (the skin round it is retouched)
    fill = cv2.inpaint(np.clip(src_ * 65535.0, 0, 65535).astype(np.uint16), md, max(3, k), cv2.INPAINT_TELEA).astype(np.float32) / 65535.0
    ms = np.clip(cv2.GaussianBlur(md.astype(np.float32), (0, 0), max(0.8, 0.2 * k)) * 1.4, 0, 1)
    M_ = np.zeros(Y0.shape, np.float32)
    F_ = (Y0 if Ycur is None else Ycur).astype(np.float32, copy=True)
    M_[y0:y1, x0:x1] = ms
    F_[y0:y1, x0:x1] = fill
    return M_, F_


def _final_long_lines(Y0: np.ndarray, region: np.ndarray, wf: float) -> Optional[np.ndarray]:
    """0..1: long straight thin dark lines of Y0 inside region (> 0.05)."""
    r = _roi(region, int(FINAL_LONG_SPAN * wf) + 8)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    roi = Y0[y0:y1, x0:x1].astype(np.float32)
    w = max(1.5, ARMPIT_W * wf)
    rb = cv2.GaussianBlur(roi, (0, 0), max(0.7, 0.3 * w))
    k = max(3, int(round(3.0 * w)) | 1)
    base = cv2.GaussianBlur(roi, (0, 0), max(2.0, 2.0 * w)) + 0.02
    bh = cv2.morphologyEx(rb, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) / base
    cand = ((bh > FINAL_LONG_THR) & (region[y0:y1, x0:x1] > 0.05)).astype(np.uint8)
    n, lab, st, _c = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros(n, bool)
    span_min = FINAL_LONG_SPAN * wf
    for i in range(1, n):
        if max(st[i, 2], st[i, 3]) < 0.5 * span_min or st[i, 4] < 8:
            continue
        ys, xs = np.nonzero(lab[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] == i)
        pts = np.c_[xs, ys].astype(np.float64)
        ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T))
        length, width = 4.0 * float(np.sqrt(max(ev[1], 1e-9))), 4.0 * float(np.sqrt(max(ev[0], 1e-9)))
        if length >= span_min and length >= FINAL_LONG_ELONG * max(width, 1.0):
            keep[i] = True
    if not keep.any():
        return None
    H = keep[lab].astype(np.uint8)
    kd = max(3, int(round(2.0 * w)) | 1)
    H = cv2.dilate(H, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))).astype(np.float32)
    out = np.zeros(Y0.shape, np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(H, (0, 0), max(1.0, 0.7 * w)) * 1.3, 0, 1)
    return out


def _final_chin(Y: np.ndarray, lm: np.ndarray, skin: np.ndarray, wf: float, k: float) -> np.ndarray:
    """The pits and bumps of a pebbled chin taken out: the band FINAL_CHIN_BAND of the luma, on the chin's zone of the skin."""
    z = chin_zone(Y.shape, lm, wf)
    if z is None:
        return Y
    r = _roi(z, int(0.1 * wf) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    m = z[y0:y1, x0:x1] * np.clip(skin[y0:y1, x0:x1], 0, 1) * float(k)
    s0, s1 = max(0.8, FINAL_CHIN_BAND[0] * wf), max(2.0, FINAL_CHIN_BAND[1] * wf)
    band = cv2.GaussianBlur(roi, (0, 0), s0) - cv2.GaussianBlur(roi, (0, 0), s1)
    if FINAL_CHIN_LIGHT is not None:
        # the light side (bumps, a sheen) only up to a scale of FINAL_CHIN_LIGHT of the face width and FINAL_CHIN_LIGHT_MAX deep: the chin's broad
        # highlight is its shape (c010 f0); the pits (the dark side) go whole
        s2 = max(s0 + 0.5, FINAL_CHIN_LIGHT * wf)
        b2 = cv2.GaussianBlur(roi, (0, 0), s0) - cv2.GaussianBlur(roi, (0, 0), s2)
        band = np.where(band > 0, np.clip(np.minimum(band, np.maximum(b2, 0.0)), 0.0, FINAL_CHIN_LIGHT_MAX), band)
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi - m * band
    return out


def _naso_zone(lm: np.ndarray, shape, wf: float) -> Optional[np.ndarray]:
    """0..1: both nasolabial folds and the mouth's corners, from the landmarks (px of shape): a thick line from beside each nose wing to the mouth's corner
    and on a little below it, without the nose (with its wings) and the lips."""
    if lm is None or lm.shape[0] < 68 or not np.isfinite(lm[[27, 31, 33, 35, 48, 54, 51, 57, 8], :2]).all():
        return None
    L = np.asarray(lm[:, :2], np.float64)
    okn = _nose_points_ok(L, wf)
    bw = float(np.linalg.norm(L[35] - L[31])) if (okn[31] and okn[35]) else 0.15 * wf
    c = L[33] if okn[33] else L[30]
    down = L[8] - L[27]
    down = down / max(float(np.linalg.norm(down)), 1e-6)
    w = max(3, int(round(FINAL_NASO_W * wf)))
    m = np.zeros(shape, np.uint8)
    ok = okn
    for wing, corner in ((31, 48), (35, 54)):
        if not ok[wing] or float(np.linalg.norm(L[corner] - L[wing])) > 0.6 * wf:
            continue
        d = L[wing] - c
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        a = L[wing] + d * 0.55 * bw - down * 0.10 * bw      # beside the wing, a little above its base
        b = L[corner] + d * 0.15 * bw                         # the mouth's corner, a little outside
        e = b + down * 0.35 * bw                              # and on below it
        for p0, p1 in ((a, b), (b, e)):
            cv2.line(m, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), 255, w, cv2.LINE_AA)
    # without the nose and the lips
    nose = _nose_area(lm, shape, wf)
    lips = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(lips, cv2.convexHull(np.round(L[48:60]).astype(np.int32)), 255)
    kl = max(3, int(round(0.04 * wf)) | 1)
    lips = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kl, kl)))
    out = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.25 * w))
    out = out * (1.0 - cv2.GaussianBlur(lips.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.01 * wf)))
    # inside the face's outline only, away from it (the cheek's contour against hair or the background is shade, not a fold)
    if np.isfinite(L[0:17]).all():
        hull = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.round(np.vstack([L[0:17], L[17:27]])).astype(np.int32)), 255)
        ke = max(3, int(round(FINAL_NASO_EDGE * wf)) | 1)
        hull = cv2.erode(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
        out = out * cv2.GaussianBlur(hull.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.3 * ke))
    if nose is not None:
        # kept clear of the nose by a soft gap: the wing's crease right beside it is the nose's shape, and lifting it up to the nose's edge leaves a
        # light seam along the wing (c029 f4)
        g = int(round(FINAL_NASO_NOSE_GAP * wf))
        if g > 0:
            nb = cv2.dilate((nose > 0.02).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))).astype(np.float32)
            nose = np.maximum(nose, cv2.GaussianBlur(nb, (0, 0), max(1.0, 0.5 * g)))
        out = out * (1.0 - nose)
    return np.clip(out, 0, 1).astype(np.float32)


def _final_naso(Y: np.ndarray, Y0: np.ndarray, lm: np.ndarray, wf: float, k: float) -> np.ndarray:
    """The folds' dark band lifted inside _naso_zone (see FINAL_NASO)."""
    z = _naso_zone(lm, Y.shape, wf)
    if z is None:
        return Y
    r = _roi(z, int(FINAL_NASO_BAND[1] * wf * 3) + 8)
    if r is None:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    s0, s1 = max(0.8, FINAL_NASO_BAND[0] * wf), max(3.0, FINAL_NASO_BAND[1] * wf)
    lo = cv2.GaussianBlur(roi, (0, 0), s1)
    band = cv2.GaussianBlur(roi, (0, 0), s0) - lo
    # strands and features: far darker than the level at the fine scale, on the picture as it came in
    r0 = Y0[y0:y1, x0:x1].astype(np.float32)
    fine = (cv2.GaussianBlur(r0, (0, 0), max(2.0, 0.02 * wf)) - cv2.GaussianBlur(r0, (0, 0), 0.8)) / (cv2.GaussianBlur(r0, (0, 0), max(2.0, 0.02 * wf)) + 0.02)
    keep = _ss(FINAL_NASO_MAXDARK, 1.6 * FINAL_NASO_MAXDARK, fine)
    # hard edges (the lips' outline, a nostril's rim, the jaw against the neck): a step of a fifth of the level within a fold's width is shape, not a fold
    rq = cv2.GaussianBlur(r0, (0, 0), 1.0)
    gx_, gy_ = cv2.Sobel(rq, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(rq, cv2.CV_32F, 0, 1, ksize=3)
    step = np.hypot(gx_, gy_) / 8.0 * max(2.0, 0.02 * wf) / (cv2.GaussianBlur(r0, (0, 0), max(2.0, 0.02 * wf)) + 0.02)
    keep = np.maximum(keep, _ss(0.20, 0.35, step))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = z[y0:y1, x0:x1] * (1.0 - keep) * float(k)
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi - m * np.minimum(band, 0.0)
    return out


def _final_pass(Y, Cb, Cr, Y0, Cb0, Cr0, rf, y_per_src, M):
    """The last pass (FINAL_*): see the constants. Y0/Cb0/Cr0: the planes as they came in."""
    if not (FINAL_NOSE > 1e-4 or FINAL_MOLE > 1e-4 or FINAL_BUMP > 1e-4 or FINAL_LONG > 1e-4 or FINAL_CHIN > 1e-4 or FINAL_NASO > 1e-4):
        return Y, Cb, Cr
    hY, wY = Y.shape
    hC, wC = Cb.shape
    Cr0Y = Cr0 if Cr0.shape == Y.shape else cv2.resize(Cr0, (wY, hY), interpolation=cv2.INTER_LINEAR)
    give = np.zeros(Y.shape, np.float32)          # full give-back (luma and colour)
    for job in rf.jobs:
        try:
            wf = job.wf_src * y_per_src if job.wf_src > 0 else 0.11 * max(job.scale, 200.0) * y_per_src
            lmY = None if job.face is None else np.asarray(job.face, np.float32)[:, :2] * y_per_src
            skin = M(job.masks["skin"], (hY, wY)) if "skin" in job.masks else None
            if FINAL_NOSE > 1e-4 and lmY is not None and job.wf_src > 0:
                nm = _nose_area(lmY, (hY, wY), wf)
                if nm is not None:
                    nm = nm * float(np.clip(FINAL_NOSE, 0, 1))
                    s_ = max(1.0, FINAL_NOSE_SIG * wf)
                    r_ = _roi(nm, int(3 * s_) + 2)
                    if r_ is not None:
                        a0, a1, b0, b1 = r_
                        Y[a0:a1, b0:b1] = _nose_give_back(Y[a0:a1, b0:b1], Y0[a0:a1, b0:b1], nm[a0:a1, b0:b1], s_)
                    nC = M(nm, (hC, wC))
                    Cb = Cb + nC * (Cb0 - Cb)
                    Cr = Cr + nC * (Cr0 - Cr)
            # where moles and bumps are looked for: the skin, and the landmarks' outline of the face (the skin mask leaves out spots of a strong
            # colour or a hard edge as "things": c029 f4's bump lay outside it)
            spot_reg = skin
            if lmY is not None and job.wf_src > 0 and np.isfinite(lmY[0:27]).all():
                hull_ = np.zeros((hY, wY), np.uint8)
                cv2.fillConvexPoly(hull_, cv2.convexHull(np.round(lmY[0:27]).astype(np.int32)), 255)
                hull_ = hull_.astype(np.float32) / 255.0
                spot_reg = hull_ if skin is None else np.maximum(skin, hull_)
            if FINAL_MOLE > 1e-4 and spot_reg is not None and job.wf_src > 0:
                mm = _final_moles(Y0, Cr0Y, spot_reg, wf, lmY)
                if mm is not None:
                    give = np.maximum(give, mm * float(np.clip(FINAL_MOLE, 0, 1)))
            if FINAL_LONG > 1e-4:
                reg = None
                if "armpit" in job.masks:
                    reg = M(job.masks["armpit"], (hY, wY))
                if reg is not None:
                    lm_ = _final_long_lines(Y0, reg, wf if job.wf_src > 0 else (float(job.wf_arm) * y_per_src if job.wf_arm > 0 else wf))
                    if lm_ is not None:
                        give = np.maximum(give, lm_ * float(np.clip(FINAL_LONG, 0, 1)))
            if FINAL_BUMP > 1e-4 and spot_reg is not None and job.wf_src > 0:
                bb = _final_bumps(Y0, spot_reg, wf, lmY, Y)
                if bb is not None:
                    bm, bf = bb
                    bm = bm * float(np.clip(FINAL_BUMP, 0, 1))
                    Y = Y + bm * (bf - Y)
            if FINAL_NASO > 1e-4 and lmY is not None and job.wf_src > 0:
                Y = _final_naso(Y, Y0, lmY, wf, float(np.clip(FINAL_NASO, 0, 1)))
            if FINAL_CHIN > 1e-4 and lmY is not None and skin is not None:
                Y = _final_chin(Y, lmY, skin, wf, float(np.clip(FINAL_CHIN, 0, 1)))
        except Exception:
            import traceback
            traceback.print_exc()
    if give.max() > 1e-3:
        Y = Y + give * (Y0 - Y)
        gC = M(give, (hC, wC))
        Cb = Cb + gC * (Cb0 - Cb)
        Cr = Cr + gC * (Cr0 - Cr)
    return Y, Cb, Cr


def process_ycc(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, rf: RetouchFrame, y_per_src: float,
                linear_light: bool = False, src: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y 0..1, Cb/Cr centred on 0 (any resolution each). y_per_src: Y-plane px per source px.

    src (log footage, retouch.logwrap): Y/Cb/Cr are then the display picture (the Rec.709 view), used to find
    what to hold and where. The skin texture itself is thinned on the source luma src["Y"] (the original code values),
    its change goes to src["dY"], and the view follows through src["Jy"/"Jcb"/"Jcr"] (what a unit change of the source
    luma does to the view) so the steps after it see the same picture they would on a display-referred file."""
    hY, wY = Y.shape
    hC, wC = Cb.shape
    cs = wC / wY                                   # chroma px per luma px
    Y = Y.astype(np.float32, copy=True)
    Cb = Cb.astype(np.float32, copy=True)
    Cr = Cr.astype(np.float32, copy=True)
    Y_in, Cb_in, Cr_in = Y.copy(), Cb.copy(), Cr.copy()

    def M(m, shape):
        if m.shape == shape:
            return m
        if m.dtype not in (np.float32, np.uint8):
            m = m.astype(np.float32)        # OpenCV 5 cannot resize float16 / float64 / bool masks (cut 055 armpit, 2026-11-14)
        return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    for job in rf.jobs:
        v = job.v
        wf = job.wf_src * y_per_src if job.wf_src > 0 else 0.11 * max(job.scale, 200.0) * y_per_src
        hd_ = he_ = None
        det = {}
        if DET_ON and _det_wanted(v):
            # where the flaws are: found before anything is changed. With the job when it was prepared with its
            # neighbours (held steady through time), else found here on this picture
            try:
                det = _det_place(job, rf, (hY, wY)) if job.det else _det_now(job, Y, Cb, Cr, wf, y_per_src, M)
            except Exception:
                import traceback
                traceback.print_exc()
                det = {}
        prot_ = det.get("protect")
        # body skin (1.23): its area on this picture, laid at this picture's own pixels, found before anything changes
        bmY = None
        if job.body and job.body.get("mask") is not None and any(v.get(k_, 0) > 1e-4 for k_ in BODY_EFFECTS):
            try:
                with _tm("apply.body_area"):
                    bmY = BS.refine(job.body["mask"], job.body.get("model"), Y, Cb, Cr)
            except Exception:
                import traceback
                traceback.print_exc()
                bmY = None
        # 0b. body blemishes: before the face's own (what the face's area holds is the face's: the body's spots are
        # filled only inside the rest of the body's area) and before any smoothing
        if bmY is not None and v.get("body_blemish", 0) > 1e-4:
            try:
                regB = bmY
                if v.get("blemish", 0) > 1e-4:
                    for nm_ in ("skinx", "neck"):
                        if nm_ in job.masks:
                            regB = regB * (1.0 - np.clip(M(job.masks[nm_], (hY, wY)), 0, 1))
                bd_ = getattr(job, "bdet", None)
                if bd_ is None and BDET_NOW:
                    bd_ = _bdet_now(job, Y, Cb, Cr, regB, y_per_src)
                if bd_ is None and getattr(job, "bmarks", None):
                    bd_ = BP.empty(y_per_src)
                if bd_ is not None:
                    if getattr(job, "bmarks", None):
                        bd_ = BP.apply_marks(bd_, job.bmarks, bd_["scale"])
                    with _tm("apply.body_spots"):
                        Y, Cb, Cr = BP.fill(Y, Cb, Cr, regB, bd_, float(np.clip(v["body_blemish"], 0, 1)), job.body_size,
                                            float(job.body["ref"]) * y_per_src, y_per_src, cs,
                                            lambda *a_: _spot_fill(*a_, plane_h=hY))
            except Exception:
                import traceback
                traceback.print_exc()
        # 0c. knee tone (1.25): the knee's blotches, redness and bruise-like patches to the leg's own colour; the grain stays
        if job.tone and job.tone.get("knees") and v.get("knee_tone", 0) > 1e-4:
            try:
                regK = bmY
                for nm_ in ("skinx", "neck"):
                    if regK is not None and nm_ in job.masks:      # (what the face's area holds is the face's)
                        regK = regK * (1.0 - np.clip(M(job.masks[nm_], (hY, wY)), 0, 1))
                kn_ = [((cx * y_per_src, cy * y_per_src), (ax_, ay_), hf_ * y_per_src)
                       for (cx, cy, ax_, ay_, hf_) in job.tone["knees"]]
                with _tm("apply.knee_tone"):
                    Y, Cb, Cr = BTN.knee_tone(Y, Cb, Cr, regK, kn_, float(np.clip(v["knee_tone"], 0, 1)), cs, gains=job.tone.get("gain"))
            except Exception:
                import traceback
                traceback.print_exc()
        # 0d. belly creases (1.25): the folds a bent trunk makes, filled from the skin round them; none found, nothing changed
        if job.tone and job.tone.get("belly") and v.get("belly_fold", 0) > 1e-4:
            try:
                with _tm("apply.belly_fold"):
                    Y, Cb, Cr = BTN.belly_fold(Y, Cb, Cr, bmY, job.tone["belly"], float(np.clip(v["belly_fold"], 0, 1)), y_per_src)
            except Exception:
                import traceback
                traceback.print_exc()
        # 0. blemishes (first, so smoothing never smears a spot around)
        if v.get("blemish", 0) > 1e-4 and det and ("skinx" in job.masks or "neck" in job.masks):
            k = float(np.clip(v["blemish"], 0, 1))
            size = spot_size(job, wf)
            reg = None
            for nm_ in ("skinx", "neck"):
                if nm_ in job.masks:
                    reg = M(job.masks[nm_], (hY, wY)) if reg is None else np.maximum(reg, M(job.masks[nm_], (hY, wY)))
            dk_ = det["dark"]
            if MOLE_KEEP > 0 and reg is not None:
                try:
                    mk_ = _mole_like(Y, Cb, Cr, reg, wf)
                    if mk_ is not None:
                        dk_ = dk_ * (1.0 - float(np.clip(MOLE_KEEP, 0, 1)) * mk_)
                        det = dict(det)
                        det["red"] = det["red"] * (1.0 - float(np.clip(MOLE_KEEP, 0, 1)) * mk_)
                except Exception:
                    import traceback
                    traceback.print_exc()
            S_ = np.maximum(dk_, det["bump"])
            R_ = det["red"]
            if prot_ is not None:
                S_ = S_ * (1.0 - prot_)
                R_ = R_ * (1.0 - prot_)
            Yin_, Cbin_, Crin_ = Y, Cb, Cr
            Y, Cb, Cr = _spot_fill(Y, Cb, Cr, reg, S_, R_, k, size, cs)
            # the dips along the folds beside the mouth are the face's own shade, so the detector leaves them. But
            # when the user asked for those folds to go (팔자주름, 입가주름) they are to go too: there the spot finder
            # of before 1.19 still runs, as it did (2611 cut 011, j2721: the hollow at the mouth's corner was left
            # darker than 1.18 left it). Lightening only, inside the fold zone only
            wfold_ = float(np.clip(max(v.get("nasolabial", 0), v.get("mouth_lines", 0)) / 0.3, 0, 1))
            if FOLD_ZONE_OLD and wfold_ > 1e-3 and "skinx" in job.masks and job.face is not None:
                try:
                    zone_ = DT.fold_zone(np.asarray(job.face, np.float32)[:, :2] * y_per_src, (hY, wY), wf, "folds")
                    if zone_ is not None and zone_.max() > 0.02:
                        Yb_, Cbb_, Crb_ = _blemish(Yin_, Cbin_, Crin_, M(job.masks["skinx"], (hY, wY)),
                                                   M(job.masks["skinx"], (hC, wC)), k, max(2.0, size), cs)
                        zw_ = zone_ * wfold_
                        Y = Y + np.clip(Yb_ - Y, 0, None) * zw_
                        zc_ = zw_ if (hC, wC) == (hY, wY) else cv2.resize(zw_, (wC, hC), interpolation=cv2.INTER_AREA)
                        take_ = (cv2.resize(np.clip(Yb_ - Yin_, 0, None), (wC, hC), interpolation=cv2.INTER_AREA) > 1e-4) * zc_
                        Cb = Cb + (Cbb_ - Cb) * take_
                        Cr = Cr + (Crb_ - Cr) * take_
                except Exception:
                    pass
        elif "skinx" in job.masks and v.get("blemish", 0) > 1e-4:
            k = float(np.clip(v["blemish"], 0, 1))
            size = (0.012 + 0.05 * job.blemish_size) * wf
            Y, Cb, Cr = _blemish(Y, Cb, Cr, M(job.masks["skinx"], (hY, wY)), M(job.masks["skinx"], (hC, wC)),
                                 k, max(2.0, size), cs)
        # 1. skin smoothing
        if "skin" in job.masks and v["skin"] > 1e-4 and job.skin is not None:
            mY = M(job.masks["skin"], (hY, wY))
            mC = M(job.masks["skin"], (hC, wC))
            # the cloth's zone beside the skin (SKIN_CLOTH_GATE): 1 on cloth, hair and the ramp of their edge outside the face's outline, 0 elsewhere; None when there is none
            dkz_ = None
            if SKIN_CLOTH_GATE is not None:
                try:
                    dkz_ = _cloth_zone(Y, mY, wf)
                    if dkz_ is not None:
                        fzg_ = _face_zone(job.face, (hY, wY), y_per_src, wf) if job.face is not None else None
                        if fzg_ is not None:
                            fzc_ = np.clip(fzg_, 0.0, 1.0)
                            nxg_ = None
                            if SKIN_CLOTH_INFACE:
                                # inside the face's outline the zone stays where the exterior (hair, background, cloth) is near: the outline of a turned face lies over the hair
                                rch_ = max(6, int(round(SKIN_CLOTH_REACH * hY)))
                                sgm_ = max(2.0, SKIN_CLOTH_SOFT * hY)
                                nxg_ = _exterior_near(M(job.masks["excl"], (hY, wY)) if job.masks.get("excl") is not None else None, job.face, (hY, wY), y_per_src, wf,
                                                      mY, rch_ + 2.0 * sgm_ + 4.0, sgm_)
                            dkz_ = dkz_ * (1.0 - fzc_ if nxg_ is None else 1.0 - fzc_ * (1.0 - nxg_))
                        if float(dkz_.max()) < 0.02:
                            dkz_ = None
                except Exception:
                    dkz_ = None
            try:
                th_ = _things(Y, mY, wf, Cb, Cr)
                if th_ is not None and SKIN_THINGS_OFF_FACE and job.face is not None:
                    # v3: on the face itself strands and paint are already out of the mask (found on the proxy and
                    # held steady through time); the finder here took the nose's shadow side, the lash lines and
                    # the lips' edge for "things" and changed from frame to frame. It stays for the neck and chest
                    # (chains, cords, straps)
                    fz_ = _face_zone(job.face, (hY, wY), y_per_src, wf)
                    if fz_ is not None:
                        th_ = th_ * (1.0 - fz_)
                thp_ = th_
                hd_, he_, ho_ = _hold_maps(Y, Cb, Cr, mY, wf, job.face, y_per_src, parts=True)
                if hd_ is not None:
                    # the narrow smoothing keeps the edges of light and shade by itself: only what is compact and
                    # dark (moles, creases of the lids, lash lines) and what is not this face's colour is held
                    hs_ = np.maximum(hd_, ho_) if SKIN_EDGE_FREE else np.maximum(hd_, he_)
                    th_ = hs_ if th_ is None else np.maximum(th_, hs_)
                if FS_FOREHEAD and getattr(SK, "TEX_FS", False) and job.face is not None:
                    # cut 069 (user 18:19, 56:11/56:17/57:00): above the brows the parser's skin left a soft hole beside
                    # the painted crack whose size changed from frame to frame, so that skin was retouched on some frames
                    # and not on others. The brow band of the face is taken as skin; the paint itself stays held below
                    ff_ = _fs_forehead(job.face, (hY, wY), y_per_src, wf)
                    if ff_ is not None:
                        mY = np.maximum(mY, ff_)
                if th_ is not None:
                    mY = mY * (1.0 - th_)
                    # the colour smoothing reaches as far as its radius: paint must be kept out by that much
                    # more, or its colour bleeds into the skin around it as a purple haze
                    rc_ = max(1, int(round(SK.radius_px(job.skin, y_per_src) * cs)))
                    thC = M(th_, (hC, wC))
                    if thp_ is not None:
                        thC = np.maximum(thC, cv2.dilate(M(thp_, (hC, wC)), cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * rc_ + 1, 2 * rc_ + 1))))
                    mC = mC * (1.0 - np.clip(thC, 0, 1))
            except Exception:
                pass
            # the nose's sides and the bridge: smoothed much less (strong smoothing flattened the nose's own
            # shading sideways, a band of blur across the face at the nose)
            try:
                nhs_ = nose_hold(job.face, (hY, wY), y_per_src, wf)
                if nhs_ is not None:
                    if NOSE_SKIN < 1.0:
                        f_ = NOSE_SKIN + (1.0 - NOSE_SKIN) * nhs_
                        mY = mY * f_
                        mC = mC * M(f_, (hC, wC))
                    if hasattr(SK, "_TLS"):
                        SK._TLS.shape = nhs_          # the wide passes of the texture smoothing spare the nose
            except Exception:
                pass
            if dkz_ is not None:
                mY = mY * (1.0 - dkz_)
                mC = mC * M(1.0 - dkz_, (hC, wC))
            r = SK.radius_px(job.skin, y_per_src)
            try:
                if getattr(SK, "PIT", 0) > 0 and hasattr(SK, "_TLS"):
                    pm_ = M(job.masks["skin"], (hY, wY))
                    if ho_ is not None:
                        pm_ = pm_ * (1.0 - np.clip(ho_, 0, 1))
                    if thp_ is not None:
                        pm_ = pm_ * (1.0 - np.clip(thp_, 0, 1))
                    if dkz_ is not None:
                        pm_ = pm_ * (1.0 - dkz_)
                    SK._TLS.pitm = pm_.astype(np.float32)
            except Exception:
                pass
            if hasattr(SK, "_TLS"):
                SK._TLS.wf = float(wf)                 # the face width without rounding (band widths follow it smoothly)
                SK._TLS.sq = float(getattr(job, "sharp_q", 1.0))   # the footage's sharpness against its surroundings
            if src is not None and src["Y"].shape == Y.shape:
                Ys0 = src["Y"]
                # the smoothing's input: the cloth replaced by the skin's own level beside it (SKIN_CLOTH_GATE); its change is added to the picture itself
                Ysi = Ys0 if dkz_ is None else _cloth_input(Ys0, dkz_)
                Ys1 = SK.smooth_channel(Ysi, mY, job.skin.strength, job.keep, r, relative=True)
                if GRAIN_KEEP:
                    Ys1 = _keep_fine(Ysi, Ys1, mY)
                dS = Ys1 - Ysi
                if Ysi is not Ys0:
                    Ys1 = Ys0 + dS
                src["Y"] = Ys1
                src["dY"] += dS
                Y = Y + src["Jy"] * dS
                if Cb.shape == dS.shape:
                    Cb = Cb + src["Jcb"] * dS
                    Cr = Cr + src["Jcr"] * dS
            else:
                Y0_ = Y
                Yi_ = Y if dkz_ is None else _cloth_input(Y, dkz_)
                Y = SK.smooth_channel(Yi_, mY, job.skin.strength, job.keep, r, relative=True)
                if GRAIN_KEEP:
                    Y = _keep_fine(Yi_, Y, mY)
                if Yi_ is not Y0_:
                    Y = Y0_ + (Y - Yi_)
            rc = max(1, int(round(r * cs)))
            Cb = SK.smooth_channel(Cb, mC, job.skin.strength * SKIN_CHROMA, 0.0, rc, eps=0.0004)
            Cr = SK.smooth_channel(Cr, mC, job.skin.strength * SKIN_CHROMA, 0.0, rc, eps=0.0004)
        # 1b. body skin: the same smoothing on the body's own area, at the body's own size. Where the face's area
        # (its neck, an older project's body regions) already took a pixel, the body takes only what is left of it:
        # the two never add up to more than one retouch
        if bmY is not None and v.get("body_skin", 0) > 1e-4:
            try:
                mB = bmY
                if "skin" in job.masks and v["skin"] > 1e-4 and job.skin is not None:
                    mB = mB * (1.0 - np.clip(M(job.masks["skin"], (hY, wY)), 0, 1))
                with _tm("apply.body_smooth"):
                    Y, Cb, Cr = _body_smooth(Y, Cb, Cr, mB, job, y_per_src, cs, src)
            except Exception:
                import traceback
                traceback.print_exc()
        # 1c. the armpit's creases (ARMPIT_FOLD): the thin fold where a raised arm meets the chest, inside the armpit's own skin
        if ARMPIT_FOLD > 1e-4 and "armpit" in job.masks and v.get("skin", 0) > 1e-4:
            try:
                mA = np.clip(M(job.masks["armpit"], (hY, wY)), 0, 1)
                if ARMPIT_CORE is not None:
                    mA = mA * _ss(float(ARMPIT_CORE[0]), float(ARMPIT_CORE[1]), mA)
                if NECK_CLOTH_GATE is not None:
                    fref_ = 0.0
                    if ARMPIT_FACE_REF and "skin" in job.masks:
                        fref_ = ARMPIT_FACE_REF * _skin_level(Y, np.clip(M(job.masks["skin"], (hY, wY)), 0, 1))
                    cg_ = _skin_gate(Y, mA, *NECK_CLOTH_GATE, ref_min=fref_)
                    if cg_ is not None:
                        mA = mA * cg_
                # the size the stage goes by: the face's width, else (the face is not in the picture) the one that goes with its zone (Job.wf_arm)
                wfA = wf if job.wf_src > 0 else (float(job.wf_arm) * y_per_src if job.wf_arm > 0 else wf)
                if ARMPIT_CLOSE and ARMPIT_CLOSE > 0:
                    # the fold is a dark valley the mask's brightness / clothes tests leave out: holes and bays narrower than twice this are closed
                    mA = _close_soft(mA, max(2, int(round(ARMPIT_CLOSE * wfA))))
                kA = float(np.clip(ARMPIT_FOLD, 0, 1)) * _small_face_gate(wfA, ARMPIT_SMALL_WF)
                try:
                    import os as _os_dbg
                    if _os_dbg.environ.get("IWALY_ARMPIT_DEBUG"):
                        print("ARMPIT apply pid=%d: face width %.0f px (stage size %.0f px, Job.wf_arm %.0f), small-face gate %.2f, mask > 0.5: %d px, > 0.05: %d px" % (
                            int(job.pid), float(wf), float(wfA), float(job.wf_arm), kA, int((mA > 0.5).sum()), int((mA > 0.05).sum())), flush=True)
                except Exception:
                    pass
                if kA > 1e-4:
                    with _tm("apply.armpit"):
                        Y_arm0 = Y                      # the picture as the stage found it (ARMPIT_LINE_HOLD gives the arm's own line back from it)
                        wA = max(1.5, ARMPIT_W * wfA)
                        gA = float(ARMPIT_LIFT_GAIN) if ARMPIT_LIFT_GAIN else 1.0
                        # the stage works inside the body's skin only (ARMPIT_LIFT_INSET for the lifts, ARMPIT_FOLD_INSET for the flattener): the armpit's mask reaches beyond the skin by the width of the person's
                        # matte, and the finder took the end of a dark stripe of the wall beside the arm for a crease and the outline's own dark rim for a thin line
                        skin_ok = (bmY is not None and bmY.shape == mA.shape)
                        if skin_ok and ARMPIT_LIFT_INSET > 1e-6:
                            mA = mA * _skin_inner(bmY, ARMPIT_EDGE_CLOSE * wfA, ARMPIT_LIFT_INSET * wfA)
                        mA_hold = mA
                        Y1, Cb, Cr = _crease_lift(Y, Cb, Cr, mA, kA, wA, None, None, fine_lines=bool(ARMPIT_FINE), gain=gA, thin_rng=ARMPIT_THIN, band=ARMPIT_BAND)
                        if ARMPIT_THIN is not None or abs(gA - 1.0) > 1e-3 or ARMPIT_BAND:
                            # the lines budget is judged on what the neck lines' own tests would lift (what it was set against), not on the fuller fill of the armpit's creases
                            lifted = _crease_lift(Y, Cb, Cr, mA, kA, wA, None, None)[0] - Y
                        else:
                            lifted = Y1 - Y
                        if ARMPIT_PIT_W:
                            # the fold itself is wider than a crease line: a second pass of the filler at its width
                            Y1, Cb, Cr = _crease_lift(Y1, Cb, Cr, mA, kA, max(1.5, float(ARMPIT_PIT_W) * wfA), None, None, fine_lines=bool(ARMPIT_FINE), gain=gA, thin_rng=ARMPIT_THIN, band=ARMPIT_BAND)
                        sA, log_ = _lines_budget(lifted, mA, wfA)
                        if sA is not None:
                            for n_, b_, k_ in log_:
                                print("ARMPIT lines: %d px lifted in one armpit at a face %.0f px wide (budget %.0f): that armpit's stage is kept to %.2f" % (n_, wfA, b_, k_), flush=True)
                            Y1 = Y + sA * (Y1 - Y)
                            mA = mA * sA
                        mF = mA
                        if skin_ok and ARMPIT_FOLD_INSET > 1e-6:
                            # the flattener only inside the body's skin, away from its outline (the arm's edge against the wall)
                            mF = mA * _skin_inner(bmY, ARMPIT_EDGE_CLOSE * wfA, ARMPIT_FOLD_INSET * wfA)
                        kF = kA * float(np.clip(ARMPIT_FLAT, 0.0, 1.0))
                        Y = _fold_flat(Y1, mF, wfA, kF, ARMPIT_FLAT_EPS, ARMPIT_FLAT_R) if kF > 1e-4 else Y1
                        if ARMPIT_LINE_HOLD > 1e-4 and job.arm_axes:
                            # a hanging arm's own line stays (the lifts and the flattener took it for a crease): given back from the picture before the stage
                            try:
                                fp_ = float(wY) / float(job.masks["armpit"].shape[1])
                                Hl_ = _armpit_line_hold(Y_arm0, mA_hold, wA, wfA, job.arm_axes, fp_)
                                if Hl_ is not None:
                                    Y = _line_restore(Y, Y_arm0, Hl_ * float(np.clip(ARMPIT_LINE_HOLD, 0.0, 1.0)), max(2.0, ARMPIT_LINE_SIG * wA))
                                try:
                                    import os as _os_dbg
                                    if _os_dbg.environ.get("IWALY_ARMPIT_DEBUG"):
                                        print("ARMPIT line hold pid=%d: hanging arms (angle, weight, cx, cy, r) %s, held > 0.5: %d px, > 0.05: %d px" % (
                                            int(job.pid), [(round(a_[0]), round(a_[1], 2), round(a_[2]), round(a_[3]), round(a_[4])) for a_ in job.arm_axes],
                                            0 if Hl_ is None else int((Hl_ > 0.5).sum()), 0 if Hl_ is None else int((Hl_ > 0.05).sum())), flush=True)
                                except Exception:
                                    pass
                            except Exception:
                                import traceback
                                traceback.print_exc()
                        if ARMPIT_LONG_KEEP > 1e-4:
                            # long straight lines (an arm's outline, hair, a strap's shadow) are not creases: given back from the picture before the stage
                            try:
                                Hn_ = _armpit_long_lines(Y_arm0, mA_hold, wA, wfA)
                                if Hn_ is not None:
                                    Y = _line_restore(Y, Y_arm0, Hn_ * float(np.clip(ARMPIT_LONG_KEEP, 0.0, 1.0)), max(2.0, 3.0 * wA))
                            except Exception:
                                import traceback
                                traceback.print_exc()
            except Exception:
                import traceback
                traceback.print_exc()
        # 2. wrinkles
        ex = job.masks.get("excl")
        exY = M(ex, (hY, wY)) if ex is not None else None
        exC = M(ex, (hC, wC)) if ex is not None else None
        for key, name, sf, ss, fw_ in (("nasolabial", "naso", 0.005, 0.05, 0.022),
                                       ("mouth_lines", "mouthl", 0.0045, 0.042, 0.016)):
            if name in job.masks and v[key] > 1e-4:
                k = float(np.clip(v[key], 0, 1))
                if job.face is not None:
                    k *= 1.0 - (1.0 - MOUTH_OPEN_KEEP) * mouth_open(job.face)
                band = M(job.masks[name], (hY, wY))
                # the fold where it really is in the (wide) search band, plus a little of the band itself
                found = _fold_find(Y, band, wf, max(1.5, fw_ * wf))
                # only where a fold is really found (a quarter of the whole band used to be lifted too: on a
                # face lit from the side that flattened the cheek's shading, which looked painted over)
                # a small face (a wide shot): the fold is only a few pixels wide and is not found, so its band
                # is softened as a whole (lighter, not wiped)
                b0_ = float(FOLD_BASE.get(key, 0.06))
                base_ = b0_ + max(0.0, SMALL_FOLD - b0_) * float(1.0 - _ss(SMALL_WF[0], SMALL_WF[1], np.array(wf)))
                # the face's outer edge (hair, background) is not worked on: the cheek's contour is shading, not a fold (EXT_GUARD)
                gx_ = _exterior_guard(exY, job.face, (hY, wY), y_per_src, wf, band) if (ex is not None and job.face is not None) else None
                # and the landmarks' outline of the face (the parser's exterior does not see a wall of the skin's colour)
                go_ = _outline_guard(job.face, (hY, wY), y_per_src, wf, band) if job.face is not None else None
                if go_ is not None:
                    gx_ = go_ if gx_ is None else np.minimum(gx_, go_)
                bandg = band if gx_ is None else band * gx_
                mY = np.clip(bandg * (base_ + 0.95 * found), 0, 1)
                if key == "nasolabial":
                    nh_ = nose_hold(job.face, (hY, wY), y_per_src, wf, None if NASO_REACH == NOSE_REACH else NASO_REACH)
                    if NASO_DEEP_NOSE > 1e-4 and hd_ is not None and hd_.shape == (hY, wY):
                        # the nostrils and the shade under the nose's wings (compact, far darker than the skin) near the nose are its shape
                        try:
                            nz_ = nose_hold(job.face, (hY, wY), y_per_src, wf, NASO_DEEP_REACH)
                            if nz_ is not None:
                                dh_ = np.clip(hd_ * (1.0 - nz_) * float(NASO_DEEP_NOSE), 0, 1)
                                nh_ = (1.0 - dh_) if nh_ is None else nh_ * (1.0 - dh_)
                        except Exception:
                            pass
                    if nh_ is not None:
                        mY = mY * nh_
                if he_ is not None:
                    mY = mY * (1.0 - he_)
                Y = _fold_fill(Y, mY, exY, 0.95 * k, max(0.8, sf * wf), max(2.0, ss * wf), True,
                               valley=max(1.5, 1.6 * fw_ * wf))
                fcf_ = NASO_FOLD if (key == "nasolabial" and NASO_FOLD) else {}
                fb_ = fcf_.get("broad", FOLD_BROAD)
                fl_ = fcf_.get("level", FOLD_LEVEL)
                if fb_ > 1e-4:
                    mb_ = bandg
                    if key == "nasolabial" and nh_ is not None:
                        mb_ = mb_ * nh_
                    if he_ is not None:
                        mb_ = mb_ * (1.0 - he_)
                    Y = _fold_broad(Y, mb_, exY, wf, fb_ * k, FOLD_BROAD_SCALE[key], fcf_.get("broad_cap", FOLD_BROAD_CAP))
                if fl_ > 1e-4 and "skin" in job.masks:
                    mb_ = bandg
                    if key == "nasolabial" and nh_ is not None:
                        mb_ = mb_ * nh_
                    Y = _fold_level(Y, mb_, M(job.masks["skin"], (hY, wY)), exY, wf, fl_ * k, he_,
                                    fcf_.get("excl"), fcf_.get("edge"), fcf_.get("cap"))
                mC = M(mY, (hC, wC))
                Cb = _fold_fill(Cb, mC, exC, 0.7 * k, max(0.8, sf * wf * cs), max(1.5, ss * wf * cs), False)
                Cr = _fold_fill(Cr, mC, exC, 0.7 * k, max(0.8, sf * wf * cs), max(1.5, ss * wf * cs), False)
        # 2a. fine lines on the face skin, and a pebbled chin (both only where they are found)
        if (FACE_LINES > 1e-4 or CHIN_DIMPLE > 1e-4) and "skin" in job.masks and job.face is not None and v["skin"] > 1e-4:
            lmY = np.asarray(job.face, np.float32)[:, :2] * y_per_src
            skY = M(job.masks["skin"], (hY, wY))
            if FACE_LINES > 1e-4:
                mfl = skY if exY is None else skY * (1.0 - exY)
                nh_ = nose_hold(job.face, (hY, wY), y_per_src, wf)
                if nh_ is not None:
                    mfl = mfl * nh_
                if hd_ is not None:
                    mfl = mfl * (1.0 - np.maximum(hd_, he_))
                try:
                    lz_ = _lash_zone(exY, job.face, (hY, wY), y_per_src, wf)
                    if lz_ is not None:
                        mfl = mfl * (1.0 - lz_)
                except Exception:
                    pass
                Y, Cb, Cr = _crease_lift(Y, Cb, Cr, mfl, float(np.clip(FACE_LINES, 0, 1)), max(1.5, 0.012 * wf),
                                         None, jaw_hold(job.face, (hY, wY), y_per_src, wf))
            if CHIN_DIMPLE > 1e-4:
                Y = _chin_dimple(Y, lmY, skY, wf, float(np.clip(CHIN_DIMPLE, 0, 1)))
        # 2a'. frown lines between the brows and across the top of the nose (thin creases only)
        if FROWN_LINES > 1e-4 and "skin" in job.masks and job.face is not None and v["skin"] > 1e-4:
            try:
                fz_ = frown_zone(job.face, (hY, wY), y_per_src, wf)
                if fz_ is not None:
                    mfz = fz_ * M(job.masks["skin"], (hY, wY))
                    if exY is not None:
                        mfz = mfz * (1.0 - exY)
                    if hd_ is not None:
                        mfz = mfz * (1.0 - np.maximum(hd_, he_))
                    try:
                        lz_ = _lash_zone(exY, job.face, (hY, wY), y_per_src, wf)    # the eye's lash fan is no frown line (LASH_REACH)
                        if lz_ is not None:
                            mfz = mfz * (1.0 - lz_)
                    except Exception:
                        pass
                    Y, Cb, Cr = _crease_lift(Y, Cb, Cr, mfz, float(np.clip(FROWN_LINES, 0, 1)), max(1.5, 0.014 * wf),
                                             None, None)
            except Exception:
                pass
        # 2b. dark circles: lift the under-eye shading to the cheek's level, cool/purple cast to skin colour
        if "dark" in job.masks and v.get("dark_circle", 0) > 1e-4:
            k = float(np.clip(v["dark_circle"], 0, 1))
            mY = M(job.masks["dark"], (hY, wY))
            Y = _fold_fill(Y, mY, exY, 0.95 * k, max(0.8, 0.008 * wf), max(3.0, 0.07 * wf), True,
                           valley=max(1.5, DARK_VALLEY * wf) if DARK_VALLEY > 0 else 0.0, cap=DARK_CAP)
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
        if "neck" in job.masks and v.get("neck_lines", 0) > 1e-4 and _small_face_gate(wf, NECK_SMALL_WF) > 1e-4:
            k = float(np.clip(v["neck_lines"], 0, 1)) * _small_face_gate(wf, NECK_SMALL_WF)
            lmj = job.face
            mN = M(job.masks["neck"], (hY, wY))
            nz_ = neck_zone(lmj, (hY, wY), y_per_src, wf)
            if nz_ is not None:
                mN = mN * nz_
            if NECK_CLOTH_GATE is not None:
                try:
                    cg_ = _skin_gate(Y, mN, *NECK_CLOTH_GATE)
                    if cg_ is not None:
                        mN = mN * cg_
                except Exception:
                    pass
            try:
                nd_, ne_ = _hold_maps(Y, Cb, Cr, mN, wf, lmj, y_per_src)
                if nd_ is not None:
                    mN = mN * (1.0 - np.maximum(nd_, ne_))
                # hair lying on the neck (a dark strand is a "line" to the filler) and chains: left alone
                nt_ = _things(Y, mN, wf, Cb, Cr)
                if nt_ is not None:
                    mN = mN * (1.0 - nt_)
            except Exception:
                pass
            jh_ = jaw_hold(lmj, (hY, wY), y_per_src, wf)
            ex_ = det.get("line") if det else None
            if ex_ is not None and prot_ is not None:
                ex_ = ex_ * (1.0 - prot_)
            Y_pre_ = Y
            acw_ = None
            if NECK_ACROSS is not None:
                try:
                    acw_ = _across_weight(Y, mN, neck_axis(lmj), wf)
                except Exception:
                    acw_ = None
            Y, Cb, Cr = _crease_lift(Y, Cb, Cr, mN, k, max(1.5, 0.018 * wf), None, jh_, extra=ex_)
            Y_cr_ = Y
            if det and det.get("fold") is not None:
                fo_ = det["fold"] if prot_ is None else det["fold"] * (1.0 - prot_)
                Y = _fold_any(Y, mN, fo_, jh_, wf, NECK_RING * k)
            # the soft folds (rings) of the neck, wider and fainter than a line
            Y = _fold_flat(Y, mN if jh_ is None else mN * jh_, wf, k)
            if NECK_RING > 1e-4:
                try:
                    ax_ = neck_axis(lmj)
                except Exception:
                    ax_ = None
                Y = _ring_fill(Y, mN, jh_, wf, NECK_RING * k, ax_)
            if NECK_LEVEL > 1e-4:
                try:
                    ax2_ = neck_axis(lmj)
                except Exception:
                    ax2_ = None
                Y = _neck_level(Y, mN, jh_, wf, NECK_LEVEL * k, ax2_)
            bw_ = acw_                                                       # the broad steps only where the structure lies across the neck
            if NECK_BROAD < 0.999:
                bw_ = NECK_BROAD if bw_ is None else bw_ * NECK_BROAD
            if NECK_UNDER_CHIN is not None:
                try:
                    ju_ = jaw_under(lmj, (hY, wY), y_per_src, wf, NECK_UNDER_CHIN)
                except Exception:
                    ju_ = None
                if ju_ is not None:
                    bw_ = ju_ if bw_ is None else bw_ * ju_
            if bw_ is not None:
                Y = (Y_cr_ + (Y - Y_cr_) * bw_).astype(np.float32)
            if NECK_CAP is not None:
                Y = _cap_change(Y_pre_, Y, mN, NECK_CAP)
        # 3b'. goose bumps on the neck: softened, never wiped
        if "neck" in job.masks and v.get("neck_bumps", 0) > 1e-4 and _small_face_gate(wf, NECK_SMALL_WF) > 1e-4:
            k = float(np.clip(v["neck_bumps"], 0, 1)) * _small_face_gate(wf, NECK_SMALL_WF)
            mNb_ = M(job.masks["neck"], (hY, wY))
            if NECK_CLOTH_GATE is not None:
                try:
                    cg_ = _skin_gate(Y, mNb_, *NECK_CLOTH_GATE)
                    if cg_ is not None:
                        mNb_ = mNb_ * cg_
                except Exception:
                    pass
            if det and "bump" in det:
                # the raised bumps the detector is sure of go (as with 잡티 제거, but on the neck and at this
                # slider's strength); the softening below then evens the rest of the texture (review 6c3dcb3: the
                # slider alone never used the detection)
                mN_ = mNb_
                Sb_ = det["bump"] * mN_ * ((1.0 - prot_) if prot_ is not None else 1.0)
                if float(Sb_.max()) > 0.05:
                    Y, Cb, Cr = _spot_fill(Y, Cb, Cr, mN_, Sb_, None, k, spot_size(job, wf), cs)
            Y, Cb, Cr = _bump_soften(Y, Cb, Cr, mNb_, M(mNb_, (hC, wC)), k, wf, cs)
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
    Y, Cb, Cr = _final_pass(Y, Cb, Cr, Y_in, Cb_in, Cr_in, rf, y_per_src, M)
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


def yuv_scale(max_value: float, rng: Optional[str]) -> Tuple[float, float, float, float]:
    """(Y offset, Y span, chroma centre, chroma span) of a YUV plane's code values: the same convention as
    media.yuv_to_rgb, so Y 0..1 and Cb/Cr here are exactly what the preview gets from the decoded RGB (BT.709)."""
    s = (float(max_value) + 1.0) / 256.0
    if (rng or "tv") == "tv":
        return 16.0 * s, 219.0 * s, 128.0 * s, 224.0 * s
    return 0.0, float(max_value), 128.0 * s, float(max_value)


def apply_planes(planes: List[np.ndarray], subsample, max_value: float, pix_fmt: str,
                 rf: Optional[RetouchFrame], y_per_src: float = 1.0, disp=None, rng: Optional[str] = "tv") -> List[np.ndarray]:
    """Render: native planes. YUV planes stay YUV (chroma at its own resolution); float RGB goes through
    Y/Cb/Cr and back. disp = (ColourManager, ViewSettings, range) for footage that is not display-referred
    (log): the finding of what to hold is done on the Rec.709 view, the skin texture is thinned on the original
    code values (retouch.logwrap)."""
    if rf is None or rf.empty():
        return planes
    if disp is not None:
        from . import logwrap as LW
        if LW.wanted(disp[0], disp[1]):
            return LW.apply_planes_display(planes, subsample, max_value, pix_fmt, rf, y_per_src,
                                           disp[0], disp[1], disp[2] if len(disp) > 2 else "tv")
    if pix_fmt.startswith("yuv"):
        # the file's own range (limited: Y 64..940 in 10 bits): read as if full range, every retouch that depends on
        # the brightness worked on other values than in the preview (grey 51 + skin brightness 50 %: 86 / 95)
        y0, ys, c0, cs_ = yuv_scale(max_value, rng)
        Y = (planes[0].astype(np.float32) - y0) / ys
        Cb = (planes[1].astype(np.float32) - c0) / cs_
        Cr = (planes[2].astype(np.float32) - c0) / cs_
        Y, Cb, Cr = process_ycc(Y, Cb, Cr, rf, y_per_src)
        out = []
        for pl, val in zip(planes, (Y * ys + y0, Cb * cs_ + c0, Cr * cs_ + c0)):
            if np.issubdtype(pl.dtype, np.integer):
                out.append(np.clip(np.rint(val), 0, max_value).astype(pl.dtype))
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
