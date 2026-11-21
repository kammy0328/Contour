"""Knee tone and belly creases (1.25): two body effects that work on what the skin looks like at the size of a blotch or a
fold, apart from the body skin's texture work (retouch.bodyskin) and from the face's.

Knee tone. A knee is often blotched: bruise-like grey and purple patches, red areas, a patch that is darker than the leg
around it. The colour of the skin the knee should have is read from the leg's own skin round it (a plane fitted to it
and the local rest, so that the leg's light across the knee is kept), and what lies in the knee's own area is brought to
it: the colour (Cb, Cr) fully, the darker blotches of the brightness by most, the lighter parts of the knee's relief (the
cap, the highlight) mostly kept. Only the band of the picture a blotch lives in is changed (more than the grain, less
than the leg's light): the grain, the pores and the fine hair of the original stay as they are, bit for bit.

Belly creases. See crease_zone / belly_geometry: the folds of a bent trunk are found as dark thin valleys across the
trunk's axis, filled like the neck's lines, and the bulge above them is held in by a small push of the warp (the
'belly_fold' key of retouch.engine drives both).

Nothing here knows the engine: the functions take planes, masks and sizes and give planes back.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# ---- knee tone
KNEE_ALONG = 1.55         # the knee's own area reaches this many half widths of the leg along the leg ...
KNEE_ACROSS = 1.20        # ... and this many across it (1 = the leg's own width at the knee)
KNEE_EDGE = (0.70, 1.15)  # elliptical radius over which the area falls from full to nothing
RING = (1.05, 3.4)        # the leg's skin read from: from this elliptical radius (where the area is nearly gone) out to this
RES_FROM = 1.5            # elliptical radius from which the leg's own shading is read for the knee (what is nearer may be the blotch's)
RES_DEN = 0.02            # the weight of ring pixels near enough (within the BROAD blur) from which the leg's shading is spread over the knee
FINE = 0.12               # of a half width (at least 2 px): what is finer than this (grain, pores, hair) is never changed
BROAD = 0.9               # of a half width: the leg's own shading (light across the knee) is broader than this
DARK_KEEP = 0.12          # of a blotch darker than the leg round it: this share stays
LIGHT_KEEP = 0.65         # of a part lighter than the leg round it (the cap's relief, a highlight): this share stays
CHROMA_KEEP = 0.06        # of a colour blotch (red, grey, purple): this share stays
NOT_SKIN = (0.45, 0.65)   # brightness over the leg's: from where a pixel is skin of the leg .. fully (a sock, a hem is darker)
FIRM = (0.55, 1.6)        # a blotch's size over the leg's own colour variation: from where it is fully taken .. (a smaller
                          # difference is the skin's own and is taken less: never less than FIRM_MIN of it)
FIRM_MIN = 0.55
MAX_CHANGE = {"luma": 0.05, "chroma": 0.04}   # the most one pixel's plane may move (soft limit): the knee is evened, not repainted


def _ss(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _roi(m: np.ndarray, pad: int, thr: float = 0.002) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(m > thr)
    if not len(xs):
        return None
    h, w = m.shape
    return (max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad), max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad))


def _nconv(I: np.ndarray, w: np.ndarray, sig: float) -> Tuple[np.ndarray, np.ndarray]:
    num = cv2.GaussianBlur(I * w, (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), 0.0).astype(np.float32), den


def _plane(I: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
    """The plane a + b x + c y that fits I over the pixels w is on (least squares, weights w). None when too few."""
    ys, xs = np.nonzero(w > 0.2)
    if len(xs) < 40:
        return None
    if len(xs) > 6000:
        s = np.random.default_rng(7).choice(len(xs), 6000, replace=False)
        ys, xs = ys[s], xs[s]
    ww = w[ys, xs].astype(np.float64)
    x0, y0 = xs.mean(), ys.mean()
    A = np.stack([np.ones(len(xs)), xs - x0, ys - y0], 1) * np.sqrt(ww)[:, None]
    b = I[ys, xs].astype(np.float64) * np.sqrt(ww)
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    h, wd = I.shape
    gy, gx = np.mgrid[0:h, 0:wd].astype(np.float32)
    return (c[0] + c[1] * (gx - x0) + c[2] * (gy - y0)).astype(np.float32)


def knee_geometry(shape: Tuple[int, int], centre, axis, half: float):
    """The elliptical radius of every pixel from a knee (1 = the edge of the ellipse with the semi axes of KNEE_ALONG and
    KNEE_ACROSS half widths), for a knee at centre (x, y), a leg running along axis (unit, x y) and half its width."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = gx - float(centre[0]), gy - float(centre[1])
    u = dx * float(axis[0]) + dy * float(axis[1])
    v = -dx * float(axis[1]) + dy * float(axis[0])
    return np.sqrt((u / (KNEE_ALONG * half)) ** 2 + (v / (KNEE_ACROSS * half)) ** 2).astype(np.float32)


def leg_coords(shape: Tuple[int, int], centre, axis):
    """(u, v) of every pixel from a knee: u along the leg, v across it (pixels)."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = gx - float(centre[0]), gy - float(centre[1])
    return dx * float(axis[0]) + dy * float(axis[1]), -dx * float(axis[1]) + dy * float(axis[0])


SHIN_SHARE = 0.4          # the share of the shin's weight against the thigh's, in the leg's colour read from the picture
DARK_RIM = 0.80           # of a half width: kept off the leg's skin round what is dark in the strip (sock, hem, shoe)
LEG_EDGE = (0.62, 0.92)   # across the leg, in half widths: from where a pixel is the leg's .. to where it is not
LOCAL_MIN_Y = 0.25         # a leg's skin in a picture is brighter than this (the 75th percentile of the strip) ...
LOCAL_MIN_CHROMA = 0.012   # ... and its (Cb, Cr) is this far from grey
CHROMA_SIG = 0.004        # at least: how far (Cb, Cr) may be from the leg's own colour and still be the leg's skin (a dim, flat picture has small colour differences)


def _skin_like(yref: float, mb: float, mr: float) -> bool:
    """Whether a piece of the picture of this brightness and (Cb, Cr) can be skin: bright enough, not grey, on the warm
    side (red over blue; any light's cast keeps that for skin)."""
    return yref >= LOCAL_MIN_Y and float(np.hypot(mb, mr)) >= LOCAL_MIN_CHROMA and mr > 0.006 and mb < 0.012 and mr > -mb * 0.3


def local_zone_skin(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, zone: np.ndarray) -> Optional[np.ndarray]:
    """The skin inside a zone (0..1) when the body skin's area is not there: what has the zone's own colour, if that
    colour can be skin (_skin_like) and its brightness is not far under the zone's. None when it cannot."""
    hY, wY = Y.shape
    core = zone > 0.9
    if int(core.sum()) < 80:
        return None
    if Cb.shape != (hY, wY):
        cb = cv2.resize(Cb.astype(np.float32), (wY, hY), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(Cr.astype(np.float32), (wY, hY), interpolation=cv2.INTER_LINEAR)
    else:
        cb, cr = Cb.astype(np.float32), Cr.astype(np.float32)
    yref = float(np.percentile(Y[core], 75))
    light = _ss(NOT_SKIN[0], NOT_SKIN[1], Y / max(yref, 1e-4)).astype(np.float32)
    c2 = core & (light > 0.9)
    if int(c2.sum()) < 60:
        return None
    mb, mr = float(np.median(cb[c2])), float(np.median(cr[c2]))
    if not _skin_like(yref, mb, mr):
        return None
    d = np.hypot(cb - mb, cr - mr)
    sig = max(CHROMA_SIG, 2.5 * float(np.median(d[c2])))
    return (zone * light * np.exp(-0.5 * (d / sig) ** 2)).astype(np.float32)


def local_skin(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, centre, axis, half: float, e: np.ndarray):
    """The skin of one leg round a knee when the body skin's area is not there (the parser found nothing to hold on to:
    a dim picture, a colour cast, dark tights): only the leg's own strip, from the joints. -> (area, ring): the strip
    without the dark (sock, hem, shoe), and the part of it round the knee whose colour is the leg's (what is read from).
    None where the strip does not show enough of a leg."""
    hY, wY = Y.shape
    u, v = leg_coords((hY, wY), centre, axis)
    leg = (1.0 - _ss(LEG_EDGE[0], LEG_EDGE[1], np.abs(v) / half)).astype(np.float32)
    band = (leg > 0.9) & (e > RING[0]) & (e < RING[1])
    if int(band.sum()) < 80:
        return None
    yb = Y[band]
    yref = float(np.percentile(yb, 75))                    # (a sock, a hem or a shoe in the strip is darker than the leg)
    light = _ss(NOT_SKIN[0], NOT_SKIN[1], Y / max(yref, 1e-4)).astype(np.float32)
    core = band & (light > 0.9)
    if int(core.sum()) < 60:
        return None
    if Cb.shape != (hY, wY):
        cb = cv2.resize(Cb.astype(np.float32), (wY, hY), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(Cr.astype(np.float32), (wY, hY), interpolation=cv2.INTER_LINEAR)
    else:
        cb, cr = Cb.astype(np.float32), Cr.astype(np.float32)
    mb, mr = float(np.median(cb[core])), float(np.median(cr[core]))
    if not _skin_like(yref, mb, mr):
        return None                                        # black tights, a grey trouser, a blue top: cloth, not skin
    d = np.hypot(cb - mb, cr - mr)
    mad = float(np.median(d[core]))
    sig = max(CHROMA_SIG, 2.5 * mad)
    near = np.exp(-0.5 * (d / sig) ** 2).astype(np.float32)
    # the rim of a sock, a hem, a shoe (its blur, the shade it throws on the leg) is not the leg's skin either
    r = int(max(2, DARK_RIM * half))
    dark = cv2.dilate((light < 0.6).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    rim = 1.0 - cv2.GaussianBlur(dark.astype(np.float32), (0, 0), max(1.0, 0.4 * r))
    lt = (light * rim).astype(np.float32)
    thigh = 1.0 - (1.0 - SHIN_SHARE) * _ss(0.0, 0.5, u / half)     # (the thigh runs up from the knee, the shin down: the shin is read less, it is the one that sits in a sock's shade)
    ring = (leg * lt * near * thigh * _ss(RING[0], RING[0] + 0.35, e) * (1.0 - _ss(RING[1] - 0.6, RING[1], e))).astype(np.float32)
    return (leg * lt).astype(np.float32), ring


def tone_plane(P: np.ndarray, area: np.ndarray, ring: np.ndarray, half: float, k: float, kind: str,
               firm: Optional[np.ndarray] = None, ring_res: Optional[np.ndarray] = None) -> np.ndarray:
    """One plane (a luma or a chroma plane) of the knee's tone work. area: 0..1 where the knee's own area is (inside the
    leg's skin), ring: 0..1 the skin the colour is read from, half: half the leg's width in this plane's pixels,
    k: 0..1, kind: 'luma' or 'chroma'. -> the plane (a copy)."""
    out = P.astype(np.float32, copy=True)
    r = _roi(area, int(3.2 * half) + 6)
    if r is None or k < 1e-4:
        return out
    y0, y1, x0, x1 = r
    I = P[y0:y1, x0:x1].astype(np.float32)
    A = area[y0:y1, x0:x1].astype(np.float32)
    W = ring[y0:y1, x0:x1].astype(np.float32)
    if float(W.sum()) < 40.0:
        return out                                        # no skin of the leg round the knee to read the colour from
    sig_f = max(2.0, FINE * half)
    sig_b = max(2.0, BROAD * half)
    base = cv2.GaussianBlur(I, (0, 0), sig_f)             # everything but the grain
    pl = _plane(base, W) if kind == "luma" else None     # (colour: the ring's own, one value; the leg's light is what runs along it)
    S = pl if pl is not None else np.full_like(base, float((base * W).sum() / max(float(W.sum()), 1e-6)))
    flat = base - S                                       # (S is the plane here: what the leg's skin does by itself, as seen from the ring)
    ringpx = W > 0.5
    nat = 1.4826 * float(np.median(np.abs(flat[ringpx] - np.median(flat[ringpx])))) if int(ringpx.sum()) > 30 else 0.0
    Wr = ring_res[y0:y1, x0:x1].astype(np.float32) if ring_res is not None else W
    res_, den = _nconv(base - S, Wr, sig_b)               # what the plane misses of the leg round the knee, spread over the knee ...
    S = S + res_ * (den / (den + RES_DEN)).astype(np.float32)      # ... as far as there is leg round it near enough to say
    dev = base - S                                        # the knee's own difference from the leg round it
    if kind == "luma":
        dev2 = np.where(dev < 0, DARK_KEEP * dev, LIGHT_KEEP * dev)
    else:
        dev2 = CHROMA_KEEP * dev
    # how far a pixel is from what the leg's skin does by itself: a difference of the size of the leg's own variation is
    # taken less than a blotch
    g = FIRM_MIN + (1.0 - FIRM_MIN) * _ss(FIRM[0], FIRM[1], np.abs(dev) / max(nat, 1e-4))
    if firm is not None:
        g = g * firm[y0:y1, x0:x1]
    new_base = S + dev2
    d = A * float(k) * g * (new_base - base)
    cap = MAX_CHANGE.get(kind) if MAX_CHANGE else None
    if cap:
        c0 = 0.6 * cap                                     # soft limit: what is under 0.6 of the cap passes as it is, more is
        a = np.abs(d)                                      # squeezed towards the cap
        d = np.where(a <= c0, d, np.sign(d) * (c0 + (cap - c0) * np.tanh((a - c0) / (cap - c0)))).astype(np.float32)
    out[y0:y1, x0:x1] = I + d
    return out


def _masked_skin(Yo, skin, e, area, half):
    """(area, ring) weights of one knee from the body skin's area, or None where it has not enough of the leg."""
    ringw = _ss(RING[0], RING[0] + 0.35, e) * (1.0 - _ss(RING[1] - 0.6, RING[1], e))
    sk = np.clip(skin, 0, 1).astype(np.float32)
    rr = _roi(ringw * sk, 4, 0.2)
    if rr is None:
        return None
    a0, a1, b0, b1 = rr
    sel = (ringw[a0:a1, b0:b1] * sk[a0:a1, b0:b1]) > 0.5
    if int(sel.sum()) < 60:
        return None
    # a sock, a hem, a shoe is not the leg's skin even where the area takes it for skin: darker than the leg round it
    yref = float(np.median(Yo[a0:a1, b0:b1][sel]))
    light = _ss(NOT_SKIN[0], NOT_SKIN[1], Yo / max(yref, 1e-4)).astype(np.float32)
    # the rim of a sock, a hem, a shoe (its blur, the shade it throws on the leg) is not the leg's skin either: the area
    # takes the blur's pixels for skin, and the knee would be pulled up to the leg's light there (a bright crescent along the
    # sock's top edge, strongest where the sock is nearest the knee)
    r = int(max(2, DARK_RIM * half))
    dark = cv2.dilate((light < 0.6).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    light = (light * (1.0 - cv2.GaussianBlur(dark.astype(np.float32), (0, 0), max(1.0, 0.4 * r)))).astype(np.float32)
    # holes in the area (a blotch the parser takes for something else): closed up as far as the leg's colour is there
    half_k = 2 * int(max(1, 0.30 * half)) + 1
    sk_c = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (half_k, half_k)))
    sk_w = np.maximum(sk, np.minimum(sk_c, 1.0) * 0.9) * light
    return (area * sk_w).astype(np.float32), (ringw * _ss(0.6, 0.9, sk) * light).astype(np.float32)


def knee_tone(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, skin: Optional[np.ndarray], knees, k: float, cs: float = 1.0):
    """The knee tone of one person on one picture.

    Y, Cb, Cr: the picture's planes (float; Cb and Cr cs times the size of Y, in pixels, 1 = the same size),
    skin: 0..1 the person's skin area at Y's size (retouch.bodyskin; None: there is none, read from the leg's own strip, local_skin), knees: [(centre (x, y), leg axis (unit), half the
    leg's width)] in Y's pixels, k: 0..1. -> (Y, Cb, Cr), copies (the planes themselves where nothing changed)."""
    if k < 1e-4 or not knees:
        return Y, Cb, Cr
    hY, wY = Y.shape
    hC, wC = Cb.shape
    Yo, Cbo, Cro = Y, Cb, Cr
    for centre, axis, half in knees:
        half = float(half)
        if not (half > 1.5):
            continue
        e = knee_geometry((hY, wY), centre, axis, half)
        area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
        if float(area.max()) < 0.05:
            continue
        got = None
        if skin is not None:                              # the body area says where the skin is (and where it is not)
            got = _masked_skin(Yo, skin, e, area, half)
        else:                                             # no body area (the parser found nothing to hold on to): the leg's own strip
            ls = local_skin(Yo, Cbo, Cro, centre, axis, half, e)
            if ls is not None and int((ls[1] > 0.5).sum()) >= 60:
                got = ((area * ls[0]).astype(np.float32), ls[1])
        if got is None:
            continue
        w_area, w_ring = got
        w_far = (w_ring * _ss(RES_FROM, RES_FROM + 0.4, e)).astype(np.float32)
        Yo = tone_plane(Yo, w_area, w_ring, half, k, "luma", ring_res=w_far)
        for nm in ("cb", "cr"):
            Pc = Cbo if nm == "cb" else Cro
            if Pc.shape == (hY, wY):
                wa, wr, wf, hf = w_area, w_ring, w_far, half
            else:
                wa = cv2.resize(w_area, (wC, hC), interpolation=cv2.INTER_AREA)
                wr = cv2.resize(w_ring, (wC, hC), interpolation=cv2.INTER_AREA)
                wf = cv2.resize(w_far, (wC, hC), interpolation=cv2.INTER_AREA)
                hf = max(1.5, half * cs)
            Pn = tone_plane(Pc, wa, wr, hf, k, "chroma", ring_res=wf)
            if nm == "cb":
                Cbo = Pn
            else:
                Cro = Pn
    return Yo, Cbo, Cro


# ---- where the knees and the belly are, from the tracks

def person_geometry(kp: np.ndarray, widths: Optional[np.ndarray]) -> dict:
    """The places the two effects work on, in source pixels, from one person's keypoints (17, >= 2) and the widths the
    tracking measured (one row of retouch's store: (segments, 2), may be None).

    -> dict(knees=[(cx, cy, ax, ay, half)], belly=None | dict(c=(x, y), u=(ux, uy), along=, across=, trunk=)):
    a knee's centre, the direction the leg runs in and half its width; the belly's centre (a little over half way down
    the trunk, from the neck to the pelvis), the trunk's direction, half the zone's length along it and across it, and
    the trunk's length."""
    from ..warp import body as B
    from ..track import skeleton as S
    out = dict(knees=[], belly=None)
    try:
        ends = B.segment_endpoints(np.asarray(kp, np.float32))
    except Exception:
        return out

    def seg(name):
        return B.SEG_INDEX[name]

    def finite(a):
        return a is not None and np.isfinite(np.asarray(a, np.float64)).all()
    for kn_i, hip_i, an_i, tag in ((S.L_KN, S.L_HIP, S.L_AN, "l"), (S.R_KN, S.R_HIP, S.R_AN, "r")):
        k = np.asarray(kp[kn_i, :2], np.float64)
        if not finite(k):
            continue
        hp = np.asarray(kp[hip_i, :2], np.float64)
        an = np.asarray(kp[an_i, :2], np.float64)
        v1 = (k - hp) if finite(hp) else None
        v2 = (an - k) if finite(an) else None
        parts = []
        for v in (v1, v2):
            if v is not None and float(np.hypot(*v)) > 4:
                parts.append(v / float(np.hypot(*v)))
        if not parts:
            continue
        ax = np.sum(parts, 0)
        n = float(np.hypot(*ax))
        ax = ax / n if n > 1e-3 else parts[0]
        L = float(np.hypot(*(an - hp))) if (finite(hp) and finite(an)) else float(np.hypot(*(v1 if v1 is not None else v2)) * 2.0)
        half = np.nan
        if widths is not None:
            ws = []
            for nm in ("knee_t" + tag, "knee_c" + tag):
                w = np.asarray(widths[seg(nm)], np.float64)
                if np.isfinite(w).any():
                    ws.append(float(np.nanmean(w)))
            if ws:
                half = float(np.mean(ws))
        if not np.isfinite(half) or half <= 0:
            half = 0.055 * L
        out["knees"].append((float(k[0]), float(k[1]), float(ax[0]), float(ax[1]), float(half)))
    try:
        e = np.asarray(ends[seg("waist")], np.float64)
        if finite(e):
            a, b = e[:2], e[2:4]
            T = b - a
            Lt = float(np.hypot(*T))
            if Lt > 12:
                u = T / Lt
                r = np.nan
                if widths is not None:
                    w = np.asarray(widths[seg("waist")], np.float64)
                    if np.isfinite(w).any():
                        r = float(np.nanmean(w))
                if not np.isfinite(r) or r <= 0:
                    r = 0.20 * Lt
                c = a + BELLY_AT * T
                out["belly"] = dict(c=(float(c[0]), float(c[1])), u=(float(u[0]), float(u[1])), along=BELLY_ALONG * Lt,
                                    across=float(r), trunk=Lt)
    except Exception:
        pass
    return out


BELLY_AT = 0.66           # of the trunk (neck to pelvis): where the belly's zone is centred
BELLY_ALONG = 0.20        # of the trunk: half the zone's length along it


# ---- belly creases

BELLY_ACROSS = 1.10       # of the waist's half width: the zone reaches this far across the trunk
BELLY_EDGE = (0.65, 1.05)
CREASE_WIDTH = 0.011      # of the trunk's length (at least 1.5 px): the width of a fold's valley
SOFT_FOLD = 0.5           # the share of the strength the broad, faint folds are flattened with


def belly_zone(shape: Tuple[int, int], geom: dict, scale: float) -> np.ndarray:
    """0..1: the belly's zone, an ellipse (along the trunk BELLY_ALONG of its length, across it the waist's half width) on
    a picture `scale` times the source."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = geom["c"][0] * scale, geom["c"][1] * scale
    ux, uy = geom["u"]
    dx, dy = gx - cx, gy - cy
    a = dx * ux + dy * uy
    b = -dx * uy + dy * ux
    e = np.sqrt((a / max(geom["along"] * scale, 1.0)) ** 2 + (b / max(BELLY_ACROSS * geom["across"] * scale, 1.0)) ** 2)
    return (1.0 - _ss(BELLY_EDGE[0], BELLY_EDGE[1], e)).astype(np.float32)


def belly_fold(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, skin: Optional[np.ndarray], geom: Optional[dict], k: float, scale: float):
    """The belly's creases filled from the skin round them (the same finder and fill the neck's lines use, laid across the
    trunk's axis), and the broad, faint folds beside them flattened a little. Where there is no crease nothing changes.
    -> (Y, Cb, Cr) (the planes themselves where nothing changed)."""
    if k < 1e-4 or not geom:
        return Y, Cb, Cr
    from . import engine as E
    z = belly_zone(Y.shape, geom, scale)
    if float(z.max()) < 0.05:
        return Y, Cb, Cr
    m = (z * np.clip(skin, 0, 1)).astype(np.float32) if skin is not None else local_zone_skin(Y, Cb, Cr, z)
    if m is None or float(m.max()) < 0.05:
        return Y, Cb, Cr
    width = max(1.5, CREASE_WIDTH * geom["trunk"] * scale)
    Yo, Cbo, Cro = E._crease_lift(Y, Cb, Cr, m, float(k), width, axis=np.asarray(geom["u"], np.float32), hold=None)
    if SOFT_FOLD > 1e-4:
        Yo = E._fold_flat(Yo, m, geom["trunk"] * scale / 3.55, float(k) * SOFT_FOLD)       # (3.55 face widths from the shoulders to the hips)
    return Yo, Cbo, Cro
