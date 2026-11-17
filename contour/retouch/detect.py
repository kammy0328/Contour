"""Where the skin's flaws are: one score map per kind of flaw (1.19).

Before 1.19 each correction looked for its flaw itself, on the frame it was correcting, at one size: the blemish
step saw dark and red spots of one size only (never a raised bump of the skin's own colour: bright on top), the
neck step thin lines of one width, and every frame decided for itself, so a faint spot or line came and went with
the grain. This module only finds; the engine fills what is found from the skin around it.

Score maps (float32 0..1, at the resolution of the picture given):
  dark     dark spots: darker than the skin around them, round or short (moles are spots too: the blemish size
           slider says how large a spot may be)
  red      red spots: redder than the skin around them (a pimple, a blotch), whatever their brightness
  bump     raised relief of the skin's own colour: a small bright top (often with a dark side), single bumps and
           the dense bumpy texture alike
  line     thin dark lines (neck lines, any direction, several widths)
  fold     broad soft folds (the rings across a neck), found at larger widths, lying across the neck
  protect  what must stay whatever the scores say: strands of hair, jewellery and glints, glitter, the collar's edge,
           strong edges and the deep shading of the neck's own build

Every score is measured at several sizes that follow the face width and joined, against two references for what
is "more than the skin's own texture": the texture right around the point and the texture of the calm part of
the face. The second is what finds a bumpy cheek: where every point is a bump the local texture is the bumps
themselves and none of them stood out.

Nothing here decides how much is corrected (the sliders do) and nothing is thresholded hard: scores rise smoothly
with the evidence, so they can be averaged through time (engine: detection held steady) and fade in and out."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

KINDS = ("dark", "red", "bump", "line", "fold", "protect")

# spot sizes (diameter), share of the largest size taken (the blemish size slider)
SPOT_SIZES = (0.3, 0.55, 1.0)
# line half widths, share of the face width: a hair-thin neck line .. a clear crease
LINE_W = (0.004, 0.008, 0.016)
# broad folds: Gaussian width of the valley, share of the face width
FOLD_W = (0.028, 0.05)
WORK_SPOT = 8.0             # pixels a spot is across on the picture it is looked for on (larger ones: a reduced picture)
WORK_LINE = 2.5             # the same for a line's Gaussian width
GLOBAL_TEX_Q = 30.0         # percentile of the local texture over the skin taken as "calm skin"
GLOBAL_TEX_MAX = 1.6        # the local texture counts at most this many times the calm skin's
SMALL_GAIN = 1.35           # how many times further above the texture a 3 px spot must stand than a large one
BUMP_SIZE_MAX = 1.0         # share of the largest spot size up to which bumps are looked for
LINE_LEN = 0.05             # of the face width: a neck line is at least this long (shorter marks are texture)
FOLD_LEN = 0.12
LINE_DOWN = (0.2, 0.45)      # |cos| between a line's normal and the neck's axis: below, the line runs down the neck
LINE_DEEP = 0.2             # a valley this much darker than the skin anywhere along its run is a thing, not a skin line
RUN_GAP = 0.01              # of the face width: gaps this long in a valley's run are bridged
TWO_SIDED = (0.3, 0.65)      # share of a valley's depth it must also have against its darker side (a shadow's foot has none)
BUMP_MAX = (0.16, 0.26)     # a bright dot this much over the skin (relative) is glitter, a glint, a drop: not a bump
DARK_MAX = (0.45, 0.65)     # a dot this much darker than the skin is a thing (a stud, a painted dot), not a blemish


def _ss(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


GM_WORK = 2.5               # a wide weighted mean is taken on a reduced picture: its Gaussian is this many pixels there


def _gm(x: np.ndarray, w: np.ndarray, sig: float) -> np.ndarray:
    """Mean of x weighted by w under a Gaussian. A wide one is smooth, so it is worked out on a reduced picture
    (the same answer to within a rounding, at a fraction of the time: detection cost was mostly this)."""
    f = sig / GM_WORK
    if f >= 2.0 and min(x.shape[:2]) >= 4 * f:
        h, ww = x.shape[:2]
        sz = (max(4, int(round(ww / f))), max(4, int(round(h / f))))
        num = cv2.GaussianBlur(cv2.resize(x * w, sz, interpolation=cv2.INTER_AREA), (0, 0), sig * sz[0] / ww)
        den = cv2.GaussianBlur(cv2.resize(w, sz, interpolation=cv2.INTER_AREA), (0, 0), sig * sz[0] / ww)
        plain = cv2.GaussianBlur(cv2.resize(x, sz, interpolation=cv2.INTER_AREA), (0, 0), sig * sz[0] / ww)
        out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), plain).astype(np.float32)
        return cv2.resize(out, (ww, h), interpolation=cv2.INTER_LINEAR)
    num = cv2.GaussianBlur(x * w, (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), cv2.GaussianBlur(x, (0, 0), sig)).astype(np.float32)


def _hessian(I: np.ndarray, sig: float):
    """Eigenvalues (l1 >= l2) of the Hessian at scale sig, scale normalised, and the direction across (angle of the
    strongest curvature)."""
    Ih = cv2.GaussianBlur(I, (0, 0), max(0.5, sig))
    Ixx = cv2.Sobel(Ih, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(Ih, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(Ih, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    phi = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    return l1, l2, phi


def _levels(Ys: np.ndarray, wsk: np.ndarray, sig: float, sign: float, floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """The skin's level around each point without the outliers on the side looked for (sign -1: dark ones, +1:
    bright ones), and the local texture (standard deviation of what is left)."""
    b0 = _gm(Ys, wsk, sig)
    dev = Ys - b0
    tex = np.sqrt(np.maximum(_gm(dev * dev, wsk, sig), 0)) + floor
    w = wsk * np.exp(-(np.clip(sign * dev, 0, None) / tex) ** 2)
    base = _gm(Ys, w, sig)
    dev = Ys - base
    tex = np.sqrt(np.maximum(_gm(dev * dev, w, sig), 0)) + floor
    return base, tex


def _calm(tex: np.ndarray, sel: np.ndarray) -> float:
    v = tex[sel]
    return float(np.percentile(v, GLOBAL_TEX_Q)) if v.size > 50 else float(tex.mean())


def _roi(m: np.ndarray, pad: int) -> Optional[Tuple[int, int, int, int]]:
    ys = np.nonzero(m.max(1) > 0.02)[0]
    xs = np.nonzero(m.max(0) > 0.02)[0]
    if not len(ys) or not len(xs):
        return None
    h, w = m.shape
    return max(0, int(ys[0]) - pad), min(h, int(ys[-1]) + 1 + pad), max(0, int(xs[0]) - pad), min(w, int(xs[-1]) + 1 + pad)


# --------------------------------------------------------------------------------------------- spots
def spots(Y: np.ndarray, Cr: np.ndarray, skin: np.ndarray, size: float,
          sizes: Sequence[float] = SPOT_SIZES, shade: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """dark, red and bump scores inside skin (same shape as Y; Cr at Y's resolution). size: the largest spot's
    diameter in pixels. shade: where the face has its own round shading (fold_zone): large dark spots do not
    count there."""
    out = {k: np.zeros(Y.shape, np.float32) for k in ("dark", "red", "bump")}
    r = _roi(skin, int(2.5 * size) + 6)
    if r is None:
        return out
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    crr = Cr[y0:y1, x0:x1].astype(np.float32)
    mm = skin[y0:y1, x0:x1].astype(np.float32)
    wsk = (mm > 0.05).astype(np.float32)
    sel = mm > 0.5
    if sel.sum() < 30:
        return out
    acc = {k: np.zeros(roi.shape, np.float32) for k in out}
    shd = None if shade is None else np.clip(shade[y0:y1, x0:x1].astype(np.float32), 0, 1)
    full = (roi, crr, mm)
    for f in sizes:
        s = max(3.0, f * size)
        # a large spot is looked for on a smaller picture (the same spot, WORK_SPOT pixels across): the wide blurs
        # of the full picture took seconds per face and found nothing more
        red_ = max(1.0, s / WORK_SPOT)
        if red_ > 1.25:
            hw_ = (max(8, int(round(full[0].shape[1] / red_))), max(8, int(round(full[0].shape[0] / red_))))
            roi, crr, mm = (cv2.resize(x, hw_, interpolation=cv2.INTER_AREA) for x in full)
            s = s / red_
        else:
            roi, crr, mm = full
            red_ = 1.0
        wsk = (mm > 0.05).astype(np.float32)
        sel = mm > 0.5
        if sel.sum() < 30:
            continue
        acc_s = {}
        # the picture smoothed to the spot's own size (a matched filter): the grain averages out, a spot of this
        # size stays. A small spot is a few pixels of the grain's own size, so it must stand further above the
        # texture than a large one to count (g: 1 at the largest size .. SMALL_GAIN at 3 px)
        sig_s = max(0.6, 0.22 * s)
        sig_l = max(2.0, 0.85 * s)
        g = 1.0 + (SMALL_GAIN - 1.0) * float(np.clip((9.0 - s * red_) / 6.0, 0.0, 1.0))
        Ys = cv2.GaussianBlur(roi, (0, 0), sig_s)
        l1, l2, _ = _hessian(roi, max(0.7, 0.28 * s))
        # with skin all around: at the skin's edge the dark (or light) side beyond it looks like a spot
        rin = max(1, int(round(1.0 * s)))
        inner = cv2.erode((mm > 0.6).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rin + 1, 2 * rin + 1)))
        inner = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), max(0.6, 0.3 * s))
        spread = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(round(0.5 * s))) + 1,) * 2)

        def whole(sc):
            # the score is strongest in the middle of a spot: spread over the whole spot so no ring is left
            sc = cv2.dilate(sc, spread)
            return np.clip(cv2.GaussianBlur(sc, (0, 0), max(0.6, 0.25 * s)) * 1.3, 0, 1)

        # ---- dark
        base, tex = _levels(Ys, wsk, sig_l, -1.0, 0.004)
        texr = np.minimum(tex, GLOBAL_TEX_MAX * _calm(tex, sel))
        D = np.clip(base - Ys, 0, 0.3)
        rel = D / np.maximum(base, 0.05)
        blob = np.where(l1 > 1e-6, _ss(0.08, 0.40, l2 / np.maximum(l1, 1e-6)), 0.0).astype(np.float32)
        blob = cv2.GaussianBlur(blob, (0, 0), max(0.8, 0.35 * s))
        cand = ((D / texr > 1.6 * g) & (rel > 0.015) & (mm > 0.5)).astype(np.uint8)
        short = np.zeros_like(blob)
        if cand.any():
            # a short mark (a scratch, a flake) is no blob but no line either: its whole extent within the size
            n_, lab_, st_, _c = cv2.connectedComponentsWithStats(cand, connectivity=8)
            ok_ = (np.maximum(st_[:, 2], st_[:, 3]) <= 1.0 * s) & (st_[:, 4] >= 3)
            ok_[0] = False
            if ok_.any():
                short = cv2.GaussianBlur(ok_[lab_].astype(np.float32), (0, 0), max(0.6, 0.15 * s))
        sc = _ss(1.2 * g, 2.8 * g, D / texr) * _ss(0.012, 0.05, rel) * np.clip(np.maximum(blob * 1.6, short * 1.5), 0, 1)
        sc = sc * (1.0 - _ss(DARK_MAX[0], DARK_MAX[1], rel)) * inner
        # not a piece of a fold: along a crease (beside the mouth, under the eye, on the neck) the valley has dips
        # that are round at the spot's own size; at twice that size the crease is a valley running on
        b1, b2, _ = _hessian(roi, max(1.0, 0.7 * s))
        run = np.where(b1 > 1e-7, 1 - _ss(0.25, 0.6, np.abs(b2) / np.maximum(b1, 1e-7)), 0.0).astype(np.float32)
        strong = _ss(1.5, 3.0, b1 / (float(np.median(np.abs(b1[sel]))) + 1e-7))
        sc = sc * (1.0 - cv2.GaussianBlur(run * strong, (0, 0), max(0.8, 0.3 * s)))
        if shd is not None and f > 0.75:
            # the dip at the end of the fold beside the mouth, a dimple, the hollow under the lip's corner: round
            # and dark at the largest spot size, and part of the face
            sc = sc * (1.0 - (shd if red_ == 1.0 else cv2.resize(shd, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)))
        acc_s["dark"] = whole(sc)
        # ---- bump: a bright top, both curvatures negative (a dome), of the skin's own colour
        base, tex = _levels(Ys, wsk, sig_l, +1.0, 0.004)
        texr = np.minimum(tex, GLOBAL_TEX_MAX * _calm(tex, sel))
        E = np.clip(Ys - base, 0, 0.3)
        relb = E / np.maximum(base, 0.05)
        dome = np.where(l2 < -1e-6, _ss(0.08, 0.40, l1 / np.minimum(l2, -1e-6)), 0.0).astype(np.float32)
        dome = cv2.GaussianBlur(dome, (0, 0), max(0.8, 0.35 * s))
        sb = _ss(1.3 * g, 2.8 * g, E / texr) * _ss(0.010, 0.04, relb) * np.clip(dome * 1.6, 0, 1)
        # glitter, a glint, a drop of sweat: far brighter at its sharpest point than a bump's top
        top = cv2.dilate(cv2.GaussianBlur(roi, (0, 0), 0.5), np.ones((3, 3), np.uint8))
        sb = sb * (1.0 - _ss(BUMP_MAX[0], BUMP_MAX[1], (top - base) / np.maximum(base, 0.05))) * inner
        if f <= BUMP_SIZE_MAX:
            # a bump is small and stands alone: the bright patch it is the top of is no wider than the bump. The
            # light on the nose's ridge, under the eyes, on the chin (the face's own highlights, which stay) is
            # bright over a wider patch
            cb_ = ((E / texr > 1.6 * g) & (relb > 0.010) & (mm > 0.3)).astype(np.uint8)
            okb = np.zeros_like(sb)
            if cb_.any():
                n_, lab_, st_, _c = cv2.connectedComponentsWithStats(cb_, connectivity=8)
                ok_ = (np.maximum(st_[:, 2], st_[:, 3]) <= 1.6 * s) & (st_[:, 4] >= 2)
                ok_[0] = False
                if ok_.any():
                    okb = cv2.GaussianBlur(ok_[lab_].astype(np.float32), (0, 0), max(0.6, 0.15 * s))
            acc_s["bump"] = whole(sb * np.clip(okb * 1.5, 0, 1))
        # ---- red
        # red in the plane itself and in the colour without the light (Cr over the brightness), both: across the
        # edge of a shadow or beside a highlight the lit skin is "redder" in Cr (it grows with the light) and the
        # skin in shade is redder in colour; a red spot is redder both ways
        Cra = cv2.GaussianBlur(crr, (0, 0), sig_s)
        abase, _t = _levels(Cra, wsk, sig_l, +1.0, 0.002)
        awide, _t = _levels(Cra, wsk, 3.0 * sig_l, +1.0, 0.002)
        Ra = np.minimum(Cra - abase, Cra - awide + 0.003)
        lum = np.maximum(cv2.GaussianBlur(roi, (0, 0), max(1.0, 2.0 * sig_s)), 0.06)
        ref_l = float(np.median(lum[sel]))
        Crs = Cra / lum * ref_l
        cbase, ctex = _levels(Crs, wsk, sig_l, +1.0, 0.002)
        ctexr = np.minimum(ctex, GLOBAL_TEX_MAX * _calm(ctex, sel))
        # redder than the skin right around it and than the skin further out: beside a highlight (which has
        # little colour) plain skin is "redder than what is around it" too
        cwide, _t = _levels(Crs, wsk, 3.0 * sig_l, +1.0, 0.002)
        R = np.clip(np.minimum(np.minimum(Crs - cbase, Crs - cwide + 0.003), Ra), 0, 0.1)
        m1, m2, _ = _hessian(crr, max(0.7, 0.28 * s))
        cdome = np.where(m2 < -1e-7, _ss(0.08, 0.40, m1 / np.minimum(m2, -1e-7)), 0.0).astype(np.float32)
        cdome = cv2.GaussianBlur(cdome, (0, 0), max(0.8, 0.35 * s))
        sr = _ss(1.5 * g, 3.0 * g, R / ctexr) * _ss(0.004, 0.015, R) * np.clip(np.maximum(cdome, blob) * 1.6 + 0.3, 0, 1) * inner
        acc_s["red"] = whole(sr)
        for k, v in acc_s.items():
            if red_ > 1.0:
                v = cv2.resize(v, (full[0].shape[1], full[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            acc[k] = np.maximum(acc[k], v)
    for k in out:
        out[k][y0:y1, x0:x1] = acc[k] * full[2]
    return out


def fold_zone(lm: Optional[np.ndarray], shape, wf: float) -> Optional[np.ndarray]:
    """Where the face has its own round shading: along the folds from the nose's wings past the mouth's corners, and
    round the corners (dimples). lm: 68 landmarks in this picture's pixels."""
    if lm is None or len(lm) < 68 or not np.isfinite(lm[[31, 35, 48, 54, 8], :2]).all():
        return None
    L = np.asarray(lm, np.float32)[:, :2]
    m = np.zeros(shape, np.uint8)
    down = L[8] - L[27]
    down = down / max(float(np.hypot(*down)), 1e-6)
    for wing, corner, side in ((31, 48, -1.0), (35, 54, 1.0)):
        out_ = np.array([-down[1], down[0]]) * side
        if (L[corner] - L[30]) @ out_ < 0:
            out_ = -out_
        a = L[wing] + out_ * 0.05 * wf
        b = L[corner] + out_ * 0.08 * wf
        c = b + down * 0.16 * wf
        pts = np.round(np.stack([a, b, c])).astype(np.int32)
        cv2.polylines(m, [pts], False, 1, max(2, int(0.10 * wf)))
        cv2.circle(m, (int(b[0]), int(b[1])), max(2, int(0.10 * wf)), 1, -1)
    # the hollow between the lower lip and the chin, and the hollows under the eyes (from the inner corner along
    # the lower lid): round shade of the face's build, as large as a spot (lngshot 058: both were marked dark)
    if np.isfinite(L[57]).all():
        c = L[57] + down * 0.07 * wf
        ang = float(np.degrees(np.arctan2(down[1], down[0]))) - 90.0
        cv2.ellipse(m, (int(c[0]), int(c[1])), (max(2, int(0.15 * wf)), max(2, int(0.07 * wf))), ang, 0, 360, 1, -1)
    for ids in ((39, 40, 41, 36), (42, 47, 46, 45)):
        if np.isfinite(L[list(ids)]).all():
            pts = np.round(L[list(ids)] + down * 0.06 * wf).astype(np.int32)
            cv2.polylines(m, [pts], False, 1, max(2, int(0.07 * wf)))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.02 * wf))


# --------------------------------------------------------------------------------------------- lines and folds
def lines(Y: np.ndarray, neck: np.ndarray, wf: float, axis: Optional[Tuple[float, float]] = None,
          widths: Sequence[float] = LINE_W, folds: Sequence[float] = FOLD_W) -> Dict[str, np.ndarray]:
    """line (thin, any direction) and fold (broad, across the neck) scores inside the neck mask.
    axis: unit vector down the neck (picture coordinates); None: folds in any direction count."""
    out = {k: np.zeros(Y.shape, np.float32) for k in ("line", "fold")}
    r = _roi(neck, int(0.2 * wf) + 6)
    if r is None:
        return out
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = neck[y0:y1, x0:x1].astype(np.float32)
    wsk = (mm > 0.05).astype(np.float32)
    sel = mm > 0.5
    if sel.sum() < 30:
        return out
    full = (roi, mm)
    # how much darker than the skin beside it, at the picture's own sharpness (a chain a pixel wide is deep here
    # and shallow on the pictures the wider lines are looked for on)
    rq_ = cv2.GaussianBlur(roi, (0, 0), max(0.6, 0.003 * wf))
    b0_ = _gm(rq_, wsk, max(3.0, 0.05 * wf))
    deep_full = np.clip(b0_ - rq_, 0, 1) / np.maximum(b0_, 0.05)
    for kind, ws in (("line", widths), ("fold", folds)):
        acc = np.zeros(full[0].shape, np.float32)
        for f in ws:
            w = max(1.0, f * wf)                    # the valley's Gaussian width
            red_ = max(1.0, w / WORK_LINE)
            if red_ > 1.25:
                hw_ = (max(8, int(round(full[0].shape[1] / red_))), max(8, int(round(full[0].shape[0] / red_))))
                roi, mm = (cv2.resize(x, hw_, interpolation=cv2.INTER_AREA) for x in full)
                w = w / red_
            else:
                roi, mm = full
                red_ = 1.0
            wsk = (mm > 0.05).astype(np.float32)
            sel = mm > 0.5
            if sel.sum() < 30:
                continue
            Ys = cv2.GaussianBlur(roi, (0, 0), max(0.5, 0.5 * w))
            sig_l = max(2.0, 3.0 * w)
            base, tex = _levels(Ys, wsk, sig_l, -1.0, 0.003)
            texr = np.minimum(tex, GLOBAL_TEX_MAX * _calm(tex, sel))
            D = np.clip(base - Ys, 0, 0.25)
            rel = D / np.maximum(base, 0.05)
            l1, l2, phi = _hessian(roi, max(0.7, 1.0 * w))
            # a valley: curved strongly across, hardly along
            line = np.where(l1 > 1e-7, 1 - _ss(0.25, 0.6, np.abs(l2) / np.maximum(l1, 1e-7)), 0.0).astype(np.float32)
            # with skin on both sides, both lighter: the foot of a shadow's edge (the collar's shadow, the dark
            # side of the neck, the jaw's shadow) is dark against one side only
            gy_, gx_ = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float32)
            dx_, dy_ = np.cos(phi).astype(np.float32) * (2.5 * w), np.sin(phi).astype(np.float32) * (2.5 * w)
            Yw = cv2.GaussianBlur(Ys, (0, 0), max(0.8, 1.0 * w))    # the sides: the skin's level there, not its grain
            side = []
            for sg_ in (1.0, -1.0):
                mx_, my_ = gx_ + sg_ * dx_, gy_ + sg_ * dy_
                side.append((cv2.remap(Yw, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                             cv2.remap(mm, mx_, my_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)))
            two = np.clip(np.minimum(side[0][0], side[1][0]) - Ys, 0, 0.25)
            # a gate, not the depth itself: the darker side's own grain must not thin a real line's score
            line = line * _ss(TWO_SIDED[0], TWO_SIDED[1], two / np.maximum(D, 1e-4))
            line = line * _ss(0.4, 0.8, np.minimum(side[0][1], side[1][1]))
            if kind == "line":
                sc = _ss(1.0, 2.4, D / texr) * _ss(0.006, 0.025, rel) * line
                if axis is not None:
                    # any direction but the neck's own: what runs down the neck is a hanging strand of hair, a cord,
                    # a tendon's shade. A line tilted up to some 70 degrees from "across" still counts in full
                    along = np.abs(np.cos(phi) * axis[0] + np.sin(phi) * axis[1])
                    sc = sc * _ss(LINE_DOWN[0], LINE_DOWN[1], cv2.GaussianBlur(along.astype(np.float32), (0, 0), max(1.0, 0.5 * w)))
            else:
                sc = _ss(0.8, 2.0, D / texr) * _ss(0.006, 0.03, rel) * line
                if axis is not None:
                    # the strongest curvature is across the valley: for a ring across the neck that is along the
                    # neck. The neck's own shading (tendons, the hollow between them) lies the other way
                    along = np.abs(np.cos(phi) * axis[0] + np.sin(phi) * axis[1])
                    sc = sc * _ss(0.55, 0.85, cv2.GaussianBlur(along.astype(np.float32), (0, 0), max(1.0, 0.5 * w)))
            # inside the neck: at its edge the collar's shadow, the hair beside it, the jaw's outline are valleys too
            rin = max(1, int(round(2.0 * w)) + 1)
            inner = cv2.erode((mm > 0.5).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rin + 1, 2 * rin + 1)))
            sc = sc * cv2.GaussianBlur(inner.astype(np.float32), (0, 0), max(0.6, 0.5 * w))
            # long enough to be a line: what is as short as it is wide is the skin's texture
            need = (LINE_LEN if kind == "line" else FOLD_LEN) * wf / red_
            cand = (sc > 0.3).astype(np.uint8)
            keep = np.zeros_like(sc)
            if cand.any():
                n_, lab_, st_, _c = cv2.connectedComponentsWithStats(cand, connectivity=8)
                ok_ = np.hypot(st_[:, 2], st_[:, 3]) >= need
                ok_[0] = False
                if kind == "line" and ok_.any():
                    # one thing along its whole length: a chain, a cord, a strand of hair is deep where it is in
                    # focus and in the light and faint elsewhere. Its faint stretch alone reads as a neck line, so
                    # a valley that is deep anywhere along its run is left alone everywhere
                    run = ((line * _ss(0.003, 0.01, rel)) > 0.4).astype(np.uint8)
                    rb_ = max(1, int(round(RUN_GAP * wf / red_)))        # grain breaks a faint run: short gaps are bridged
                    run = cv2.dilate(run, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb_ + 1, 2 * rb_ + 1)))
                    nr_, labr_ = cv2.connectedComponents(run, connectivity=8)
                    if nr_ > 1:
                        dp_ = deep_full
                        if red_ > 1.0:
                            kd_ = int(np.ceil(red_)) | 1
                            dp_ = cv2.resize(cv2.dilate(deep_full, np.ones((kd_, kd_), np.uint8)),
                                             (sc.shape[1], sc.shape[0]), interpolation=cv2.INTER_AREA)
                        dmax = np.zeros(nr_, np.float32)
                        np.maximum.at(dmax, labr_.ravel(), np.maximum(rel, dp_).ravel())
                        thing_ = dmax >= LINE_DEEP
                        thing_[0] = False
                        if thing_.any():
                            bad_ = np.zeros(n_, bool)
                            bad_[np.unique(lab_[thing_[labr_]])] = True
                            ok_ &= ~bad_
                if ok_.any():
                    keep = cv2.GaussianBlur(ok_[lab_].astype(np.float32), (0, 0), max(0.6, 0.5 * w))
            sc = sc * np.clip(keep * 1.5, 0, 1)
            rd = max(1, int(round(0.8 * w)))
            sc = cv2.dilate(sc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1, 2 * rd + 1)))
            sc = np.clip(cv2.GaussianBlur(sc, (0, 0), max(0.6, 0.5 * w)) * 1.3, 0, 1)
            if red_ > 1.0:
                sc = cv2.resize(sc, (full[0].shape[1], full[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            acc = np.maximum(acc, sc)
        out[kind][y0:y1, x0:x1] = acc * full[1]
    return out


# --------------------------------------------------------------------------------------------- protection
def protect(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, region: np.ndarray, wf: float) -> np.ndarray:
    """1 where nothing may be filled: thin things far darker than the skin (hair, a cord, a chain), glints and
    glitter, grey things (metal, a black cord) and things of another colour than the skin (paint, a tattoo,
    cloth), and strong edges (the collar, the jaw's outline against the neck)."""
    out = np.zeros(Y.shape, np.float32)
    r = _roi(region, int(0.1 * wf) + 6)
    if r is None:
        return out
    y0, y1, x0, x1 = r
    roi = Y[y0:y1, x0:x1].astype(np.float32)
    mm = region[y0:y1, x0:x1].astype(np.float32)
    wsk = (mm > 0.05).astype(np.float32)
    w = max(1.5, 0.018 * wf)
    rq = cv2.GaussianBlur(roi, (0, 0), max(0.8, 0.004 * wf))
    base = _gm(rq, wsk, max(3.0, 2.4 * w))
    kd = max(3, int(round(1.2 * w)) | 1)
    sed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    kn = max(3, int(round(1.6 * w)) | 1)
    # far darker than the skin beside it, and thin
    bh = cv2.morphologyEx(rq, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
    deep = cv2.dilate(bh / (base + 0.02), sed)
    # a glint: far brighter than the skin, and small
    th = cv2.morphologyEx(rq, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
    glint = cv2.dilate(th / (base + 0.02), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1)))
    thing = np.maximum(_ss(0.22, 0.40, deep), _ss(0.22, 0.40, glint))
    # colour: grey against the skin, or another hue
    cb = Cb[y0:y1, x0:x1].astype(np.float32)
    cr = Cr[y0:y1, x0:x1].astype(np.float32)
    cbs, crs = cv2.GaussianBlur(cb, (0, 0), 1.5), cv2.GaussianBlur(cr, (0, 0), 1.5)
    cm = np.hypot(cbs, crs)
    cref = _gm(cm, wsk, max(3.0, 2.4 * w)) + 1e-4
    grey = cv2.dilate(np.clip(1.0 - cm / cref, 0, 1), sed)
    thing = np.maximum(thing, _ss(0.5, 0.75, grey) * _ss(0.015, 0.03, cref))
    rb, rr = _gm(cbs, wsk, max(3.0, 2.4 * w)), _gm(crs, wsk, max(3.0, 2.4 * w))
    dc = np.hypot(cbs - rb, crs - rr)
    thing = np.maximum(thing, cv2.dilate(_ss(0.03, 0.06, dc), sed))
    # strong edges: a step of a quarter of the skin's level within a line width
    gx = cv2.Sobel(rq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(rq, cv2.CV_32F, 0, 1, ksize=3)
    step = np.hypot(gx, gy) / 8.0 * w / (base + 0.02)        # Sobel's 3x3 kernel sums to 8 per unit of slope
    thing = np.maximum(thing, cv2.dilate(_ss(0.35, 0.7, step), sed))
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(thing, (0, 0), max(1.0, 0.5 * w)) * 1.5, 0, 1)
    return out


# --------------------------------------------------------------------------------------------- all of it
def detect(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, skin: Optional[np.ndarray], neck: Optional[np.ndarray],
           wf: float, spot_size: float, axis: Optional[Tuple[float, float]] = None,
           lm: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Every score map for one person on one picture. Y 0..1; Cb, Cr centred on 0, at Y's resolution; skin: where
    spots are looked for (the blemish mask: face skin without eyes, brows, lips), neck: the neck mask; either may
    be None. wf: face width in pixels of this picture; spot_size: the largest spot's diameter in pixels; lm: the
    face landmarks in this picture's pixels (None: no face)."""
    out = {k: np.zeros(Y.shape, np.float32) for k in KINDS}
    region = None
    if skin is not None and skin.max() > 0.02:
        out.update(spots(Y, Cr, skin, spot_size, shade=fold_zone(lm, Y.shape, wf)))
        region = skin
    if neck is not None and neck.max() > 0.02:
        out.update(lines(Y, neck, wf, axis))
        # spots on the neck too (the same sizes)
        ns = spots(Y, Cr, neck, spot_size)
        for k in ("dark", "red", "bump"):
            out[k] = np.maximum(out[k], ns[k])
        region = neck if region is None else np.maximum(region, neck)
    if region is not None:
        out["protect"] = protect(Y, Cb, Cr, region, wf)
    return out
