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

from . import detect as DT
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
    sharp_q: float = 1.0                   # this frame's skin sharpness over that of the frames round it
    # where the skin's flaws are (detect.py), found once per frame and held steady through time (1.19):
    # {"box": (x0, y0, x1, y1) in proxy px, "maps": {kind: uint8 (h, w)}, "thumb": uint8 luma (for the alignment check)}
    det: Optional[dict] = None


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
    out = np.clip(f * 1.3, 0, 1) * soft * (1 - lipf) * (1 - cz) * (1 - np.clip(ala * 1.2, 0, 1))
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
ARMPIT_R = 0.30       # reach round the shoulder joint, in shoulder widths: the fold of a raised arm lies inside it
ARMPIT_ALONG = 0.06   # centre moved this far from the joint along the upper arm


def armpit_region(kp: np.ndarray, shape, sx: float, sy: float, parts: bool = False):
    """Round both shoulder joints (0..1): the skin under a raised arm. parts: the two sides as a list."""
    if kp is None:
        return None
    P = np.full((kp.shape[0], 2), np.nan)
    ok = np.isfinite(kp[:, :2]).all(1)
    if kp.shape[1] > 2:
        ok &= np.nan_to_num(kp[:, 2]) > 0.2
    P[ok] = kp[ok, :2] * [sx, sy]
    ls, rs = P[S.L_SH], P[S.R_SH]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return None
    sw = float(np.hypot(*(ls - rs)))
    if sw < 8:
        return None
    out = []
    for sh, el in ((S.L_SH, S.L_EL), (S.R_SH, S.R_EL)):
        m = np.zeros(shape, np.uint8)
        c = P[sh].copy()
        if np.isfinite(P[el]).all():
            d = P[el] - P[sh]
            n = float(np.hypot(*d))
            if n > 1e-3:
                c = c + d / n * min(n, ARMPIT_ALONG * sw)
        cv2.circle(m, (int(round(c[0])), int(round(c[1]))), max(1, int(round(ARMPIT_R * sw))), 255, -1, cv2.LINE_AA)
        out.append(_feather(m, 0.06 * sw))
    return out if parts else np.maximum(out[0], out[1])


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
NECK_SHADE_MIN = 0.045     # of the face width: narrower stretches the colour model leaves out stay out (hair, a chain)


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
                     round(float(getattr(pp, "blemish_size", 50.0)), 1)))     # the spot sizes the flaws are looked for at
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


STEADY_ROBUST = True   # a mask that drops out on one or two frames (the parser or the colour model missing the forehead for a
STEADY_WIN = 0.15      # frame: cut 086 frames 23/27/33, cut 100 frame 4) is pulled to the median of the frames round it


def _steady(cut, tracker, st, rf: RetouchFrame, cache) -> RetouchFrame:
    t = rf.t
    px = tracker.proxy
    N = st.final.shape[1]
    k = np.array([px.sx, px.sy])
    for j in rf.jobs:
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
                _FPARSER[0] = FP.FaceParser() if FP.ready() else False
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
    doubles = _double_faces(st, tracker, t)
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
                  blemish_size=float(np.clip(getattr(pp, "blemish_size", 50.0) / 100.0, 0, 1)))
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

        def base():
            if "rgb" not in state:
                state["rgb"] = np.asarray(px.frame(t))
                state["rgbf"] = state["rgb"].astype(np.float32) / 255.0
                from ..track import facelayer as FL
                layer = FL.load(tracker)
                fmaps = layer.maps(person.id, t, shape) if (layer is not None and lm is not None) else None
                centre = layer.colour(person.id, t) if (layer is not None and lm is not None) else None
                if fmaps is None and lm is not None and PARSE_NOW:
                    fmaps = _parse_now(state["rgb"], lm)
                state["parse"] = fmaps
                state["model"] = SK.frame_model_for(tracker, p, t, state["rgb"], lm, centre)
                state["prob"] = SK.face_skin_probability(state["rgb"], state["model"], lm if fw > 0.3 else None,
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

        def get(name, fn, extra=""):
            key = (t, person.id, name, extra)
            if cache is not None and key in cache:
                return cache[key]
            m = fn()
            if cache is not None:
                cache[key] = m
                # bounded by memory, not by count: at a 1920 proxy every mask is 8 MB, and 400 of them took
                # over 3 GB in the preview
                nb = cache.get("__bytes__", 0) + _nbytes(m)
                while (len(cache) > 401 or nb > CACHE_BYTES) and len(cache) > 2:
                    k0 = next(k_ for k_ in cache if k_ != "__bytes__")
                    nb -= _nbytes(cache.pop(k0))
                cache["__bytes__"] = max(0, nb)
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
                                   b_.get("parse"))
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
                    zones = armpit_region(kpa, shape, px.sx, px.sy, parts=True)
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
                        for zone in zones:
                            a1 = armpit_skin(b_["rgb"], zone, al_, b_["prob"])
                            if a1 is not None:
                                ap = a1 if ap is None else np.maximum(ap, a1)
                        if ap is not None:
                            state["armpit"] = (ap * float(min(1.0, ARMPIT))).astype(np.float32)
                except Exception:
                    import traceback
                    traceback.print_exc()
            return state["armpit"]

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
        if DET_ON and job.masks and lm is not None and _det_wanted(v):
            try:
                job.det = get("det", lambda: _det_raw(cut, px, t, job, lm, base()["rgb"]),
                              f"{job.blemish_size:.3f}:{int(v.get('blemish', 0) > 1e-4)}{int(v.get('neck_lines', 0) > 1e-4)}{int(v.get('neck_bumps', 0) > 1e-4)}:{fth:.2f}"
                              f":sn{int(_spot_net_wanted(job))}")
            except Exception:
                import traceback
                traceback.print_exc()
                job.det = None
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
FOLD_LEVEL_CAP = (0.08, 0.03)   # most lift / most darkening (0..1 luma)
FOLD_LEVEL_EXCL = (0.08, 0.04)  # the band fades out this far from the lips and nostrils (dilation, blur; face width)
FOLD_LEVEL_EDGE = (0.02, 0.03)  # the same round hair strands and sharp edges in the band
NECK_LEVEL = 1.25          # the same on the neck (cut 024: the rings and the broad band across the neck stayed)
NECK_LEVEL_SIG = (0.008, 0.10)


def _masked_blur(Y: np.ndarray, w: np.ndarray, sx: float, sy: float) -> np.ndarray:
    num = cv2.GaussianBlur(Y * w, (0, 0), sigmaX=max(0.5, sx), sigmaY=max(0.5, sy))
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=max(0.5, sx), sigmaY=max(0.5, sy))
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), Y).astype(np.float32)


def _even_mid(roi: np.ndarray, w: np.ndarray, wf: float, sig) -> np.ndarray:
    """What evens out the middle scale of roi: the broad level minus the level at the fine scale (both measured on
    w only), capped."""
    mid = _masked_blur(roi, w, sig[0] * wf, sig[0] * wf)
    low = _masked_blur(roi, w, sig[1] * wf, sig[1] * wf)
    return np.clip(low - mid, -FOLD_LEVEL_CAP[1], FOLD_LEVEL_CAP[0])


def _fold_level(Y: np.ndarray, band: np.ndarray, skin: np.ndarray, excl: Optional[np.ndarray], wf: float,
                k: float, edges: Optional[np.ndarray] = None) -> np.ndarray:
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
        d2 = max(3, int(FOLD_LEVEL_EXCL[0] * wf) | 1)
        e2 = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d2, d2)))
        b = b * (1.0 - np.clip(cv2.GaussianBlur(e2, (0, 0), max(1.0, FOLD_LEVEL_EXCL[1] * wf)), 0, 1))
    if edges is not None:
        # a strand of hair or a sharp edge in the band is left as it is, and the skin round it is evened less the
        # nearer it is: lifted right up to it, a dark strand stood out as a notch (2611 cut 011)
        h = np.clip(edges[y0:y1, x0:x1].astype(np.float32), 0, 1)
        w = w * (1.0 - h)
        d3 = max(3, int(FOLD_LEVEL_EDGE[0] * wf) | 1)
        h = cv2.dilate(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d3, d3)))
        b = b * (1.0 - np.clip(cv2.GaussianBlur(h, (0, 0), max(1.0, FOLD_LEVEL_EDGE[1] * wf)), 0, 1))
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * b * _even_mid(roi, w, wf, FOLD_LEVEL_SIG)
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
    d = _even_mid(roi, (mm > 0.3).astype(np.float32), wf, NECK_LEVEL_SIG)
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


def _spot_fill(Y, Cb, Cr, mY, S, Sred, k: float, size: float, cs: float):
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
    sig_g = float(np.clip(GRAIN_SIGMA * Y.shape[0] / 1080.0, 0.8, 2.4))
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


def _cached(name: str, lm: np.ndarray, shape, scale: float, wf: float, make):
    """The face masks below depend only on the landmarks and the frame size; they are asked for more than once
    per face per frame (by several retouch steps), and are made once."""
    key = (name, np.asarray(lm[:, :2], np.float32).tobytes(), tuple(shape), float(scale), float(wf))
    hit = _MCACHE.get(key)
    if hit is None:
        if len(_MCACHE) > 48:
            _MCACHE.clear()
        hit = _MCACHE[key] = make()
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


NOSE_REACH = 0.07           # how far past the nose's outline the nose is held (0.14 held the top of the fold
                            # beside the nostril too, which a frown deepens)
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


def _fold_flat(Y: np.ndarray, m: np.ndarray, wf: float, k: float) -> np.ndarray:
    """The soft folds of a neck (rings across it, wider and fainter than a line): the faint light and dark
    bands between the size of the skin's texture and a twelfth of the face are taken out with an edge-keeping
    filter, so the strong shading of the neck's own build (tendons, the jaw's shadow) is an edge and stays, and
    the broad light and shade is given back. The skin's fine texture is not touched."""
    r = _roi(m, int(0.25 * wf) + 4)
    if r is None or k < 1e-4:
        return Y
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1].astype(np.float32)
    loc = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.1 * wf))
    eps = FOLD_EPS * np.clip(loc / 0.45, 0.15, 2.0) ** 2
    b1 = SK.guided_self(roi, max(1, int(round(0.008 * wf))), eps)
    b2 = SK.guided_self(b1, max(2, int(round(FOLD_R * wf))), eps)
    d = b2 - b1
    sig = max(3.0, 0.16 * wf)
    low = cv2.GaussianBlur(d * mm, (0, 0), sig) / np.maximum(cv2.GaussianBlur(mm, (0, 0), sig), 0.05)
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = roi + k * mm * (d - np.clip(low, -0.2, 0.2))
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


def nose_hold(lm: Optional[np.ndarray], shape, scale: float, wf: float) -> Optional[np.ndarray]:
    """1 off the nose, 0 on it (soft): the lines down the sides of the nose are its shape. No line filler may
    lift them (on a soft or motion-blurred face they are all that is left of the nose)."""
    if lm is None or not np.isfinite(lm[27:36, :2]).all():
        return None
    def make():
        hull = cv2.convexHull(np.round(lm[27:36, :2] * scale).astype(np.int32))
        k = max(3, int(round(NOSE_REACH * wf)) | 1)
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
    return _cached("nose", lm, shape, scale, wf, make)


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
                 fine_lines: bool = True, extra: Optional[np.ndarray] = None):
    """Fill thin dark creases (neck lines): a dark valley, curved strongly across and hardly along, darker than
    the skin around it by more than its texture. Only the crease's shading is lifted; broad shading stays.

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
    thin = _ss(THIN_LO, THIN_HI, cv2.GaussianBlur(bn, (0, 0), sig_s * 2) / (cv2.GaussianBlur(bw, (0, 0), sig_s * 2) + 1e-4))
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
    lift = lift * (1.0 - np.clip(cv2.GaussianBlur(thing, (0, 0), max(1.0, 0.5 * width)) * 1.5, 0, 1))
    Yo[y0:y1, x0:x1] = roi + k * lift
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


def _keep_fine(before: np.ndarray, after: np.ndarray, m: np.ndarray) -> np.ndarray:
    """after with the finest detail band of before inside m: whatever the smoothing took of the grain is put back,
    so every strength keeps the footage's own grain. The band follows the plane's size (about a pixel at 1080)."""
    if before.shape != after.shape or m is None or m.shape != after.shape:
        return after
    sg = float(np.clip(GRAIN_SIGMA * after.shape[0] / 1080.0, 0.8, 2.4))
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
        # 0. blemishes (first, so smoothing never smears a spot around)
        if v.get("blemish", 0) > 1e-4 and det and ("skinx" in job.masks or "neck" in job.masks):
            k = float(np.clip(v["blemish"], 0, 1))
            size = spot_size(job, wf)
            reg = None
            for nm_ in ("skinx", "neck"):
                if nm_ in job.masks:
                    reg = M(job.masks[nm_], (hY, wY)) if reg is None else np.maximum(reg, M(job.masks[nm_], (hY, wY)))
            dk_ = det["dark"]
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
            r = SK.radius_px(job.skin, y_per_src)
            try:
                if getattr(SK, "PIT", 0) > 0 and hasattr(SK, "_TLS"):
                    pm_ = M(job.masks["skin"], (hY, wY))
                    if ho_ is not None:
                        pm_ = pm_ * (1.0 - np.clip(ho_, 0, 1))
                    if thp_ is not None:
                        pm_ = pm_ * (1.0 - np.clip(thp_, 0, 1))
                    SK._TLS.pitm = pm_.astype(np.float32)
            except Exception:
                pass
            if hasattr(SK, "_TLS"):
                SK._TLS.wf = float(wf)                 # the face width without rounding (band widths follow it smoothly)
                SK._TLS.sq = float(getattr(job, "sharp_q", 1.0))   # the footage's sharpness against its surroundings
            if src is not None and src["Y"].shape == Y.shape:
                Ys0 = src["Y"]
                Ys1 = SK.smooth_channel(Ys0, mY, job.skin.strength, job.keep, r, relative=True)
                if GRAIN_KEEP:
                    Ys1 = _keep_fine(Ys0, Ys1, mY)
                dS = Ys1 - Ys0
                src["Y"] = Ys1
                src["dY"] += dS
                Y = Y + src["Jy"] * dS
                if Cb.shape == dS.shape:
                    Cb = Cb + src["Jcb"] * dS
                    Cr = Cr + src["Jcr"] * dS
            else:
                Y0_ = Y
                Y = SK.smooth_channel(Y, mY, job.skin.strength, job.keep, r, relative=True)
                if GRAIN_KEEP:
                    Y = _keep_fine(Y0_, Y, mY)
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
                if job.face is not None:
                    k *= 1.0 - (1.0 - MOUTH_OPEN_KEEP) * mouth_open(job.face)
                band = M(job.masks[name], (hY, wY))
                # the fold where it really is in the (wide) search band, plus a little of the band itself
                found = _fold_find(Y, band, wf, max(1.5, fw_ * wf))
                # only where a fold is really found (a quarter of the whole band used to be lifted too: on a
                # face lit from the side that flattened the cheek's shading, which looked painted over)
                # a small face (a wide shot): the fold is only a few pixels wide and is not found, so its band
                # is softened as a whole (lighter, not wiped)
                base_ = 0.06 + (SMALL_FOLD - 0.06) * float(1.0 - _ss(SMALL_WF[0], SMALL_WF[1], np.array(wf)))
                mY = np.clip(band * (base_ + 0.95 * found), 0, 1)
                if key == "nasolabial":
                    nh_ = nose_hold(job.face, (hY, wY), y_per_src, wf)
                    if nh_ is not None:
                        mY = mY * nh_
                if he_ is not None:
                    mY = mY * (1.0 - he_)
                Y = _fold_fill(Y, mY, exY, 0.95 * k, max(0.8, sf * wf), max(2.0, ss * wf), True,
                               valley=max(1.5, 1.6 * fw_ * wf))
                if FOLD_BROAD > 1e-4:
                    mb_ = band
                    if key == "nasolabial" and nh_ is not None:
                        mb_ = mb_ * nh_
                    if he_ is not None:
                        mb_ = mb_ * (1.0 - he_)
                    Y = _fold_broad(Y, mb_, exY, wf, FOLD_BROAD * k, FOLD_BROAD_SCALE[key])
                if FOLD_LEVEL > 1e-4 and "skin" in job.masks:
                    mb_ = band
                    if key == "nasolabial" and nh_ is not None:
                        mb_ = mb_ * nh_
                    Y = _fold_level(Y, mb_, M(job.masks["skin"], (hY, wY)), exY, wf, FOLD_LEVEL * k, he_)
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
        if "neck" in job.masks and v.get("neck_lines", 0) > 1e-4:
            k = float(np.clip(v["neck_lines"], 0, 1))
            lmj = job.face
            mN = M(job.masks["neck"], (hY, wY))
            nz_ = neck_zone(lmj, (hY, wY), y_per_src, wf)
            if nz_ is not None:
                mN = mN * nz_
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
            Y, Cb, Cr = _crease_lift(Y, Cb, Cr, mN, k, max(1.5, 0.018 * wf), None, jh_, extra=ex_)
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
        # 3b'. goose bumps on the neck: softened, never wiped
        if "neck" in job.masks and v.get("neck_bumps", 0) > 1e-4:
            k = float(np.clip(v["neck_bumps"], 0, 1))
            if det and "bump" in det:
                # the raised bumps the detector is sure of go (as with 잡티 제거, but on the neck and at this
                # slider's strength); the softening below then evens the rest of the texture (review 6c3dcb3: the
                # slider alone never used the detection)
                mN_ = M(job.masks["neck"], (hY, wY))
                Sb_ = det["bump"] * mN_ * ((1.0 - prot_) if prot_ is not None else 1.0)
                if float(Sb_.max()) > 0.05:
                    Y, Cb, Cr = _spot_fill(Y, Cb, Cr, mN_, Sb_, None, k, spot_size(job, wf), cs)
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
