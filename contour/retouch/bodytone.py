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
DARK_KEEP = 0.25          # of a blotch darker than the leg round it: this share stays
LIGHT_KEEP = 0.65         # of a part lighter than the leg round it (the cap's relief, a highlight): this share stays
CHROMA_KEEP = 0.06        # of a colour blotch (red, grey, purple): this share stays
NOT_SKIN = (0.45, 0.65)   # brightness over the leg's: from where a pixel is skin of the leg .. fully (a sock, a hem is darker)
FIRM = (0.55, 1.6)        # a blotch's size over the leg's own colour variation: from where it is fully taken .. (a smaller
                          # difference is the skin's own and is taken less: never less than FIRM_MIN of it)
FIRM_MIN = 0.55
MAX_CHANGE = {"luma": 0.018, "chroma": 0.02}
# ---- (11/24, IWALY Mac) a bruise's colour without a grey knee
# The knee's colour was taken to the ring's one colour (CHROMA_KEEP 0.06) and every darker part of it lifted: a knee that is a little
# redder and darker than the leg by nature came out grey and flat (11/23 Windows feedback: +1.9 %, colour gone). Now:
#   - the knee keeps its own even tint (the mean of its skin that is not blotched) by KNEE_TINT_KEEP; only the blotches go,
#   - no pixel comes out less saturated than the leg's skin round it (KNEE_SAT_FLOOR of the ring's chroma, hue kept),
#   - the brightness is evened where the knee is blotched in colour or far darker than the leg's own variation, not elsewhere.
KNEE_V3 = False
KNEE_TINT_KEEP = 1.0      # of the knee's own even tint (non-blotched skin of its area against the ring): this share stays (the fallback target)
KNEE_SAT_FLOOR = 0.97     # the corrected chroma is never shorter than this share of the ring's chroma (no grey patch)
KNEE_BLOTCH = (1.6, 3.2)  # a pixel's hue offset across the skin's tone axis over the leg's spread across it: from where it is a blotch .. fully
KNEE_RED_FAR = (3.5, 6.0)  # ... and along the axis (more red than the leg, beyond its spread): from where .. fully
KNEE_DARK_FAR = (3.0, 5.0)  # a darker patch with no colour blotch is evened only from this far under its surround (in the knee's own luma variation)
KNEE_NAT_C_MIN = 0.0006   # the knee's chroma variation is never read as smaller than this (log material: a bruise is 0.002..0.004 off)
KNEE_EVEN = 0.0           # of the strength: how far the knee's skin that is not blotched is taken towards its surround (0: only blotches change)
KNEE_INNER = (0.45, 0.85)  # the knee area's weight from which a pixel can be a blotch .. fully (not at the skin's own edges)
KNEE_FILL = 0.6           # of a half width: the surround a blotch's colour and brightness are filled from (the knee's own skin round it, not the ring)
KNEE_DARK_KEEP = 0.15     # of a blotch's darkness under its surround: this share stays
KNEE_BLOTCH_CAP = {"luma": 0.035, "chroma": 0.012}   # the most a blotch pixel moves (soft limit)
# ---- (11/24) the edge of what another person's leg (or anything owned by someone else) covers: the correction fades in from it
OWNER_FEATHER = 0.30      # of the leg's half width (at least 2 px): the ramp inward from an occluder's edge   # the most one pixel's plane may move (soft limit): the knee is evened, not repainted (the reference's knee moves by about 0.01 in log)

# ---- the knee tone's steadiness from one picture to the next (1.26)
# A knee is read from the picture it is in: how much of the leg's skin is to be seen round it, how much of the knee is seen at all (the
# rest under cloth, behind another leg), how wide the leg is. When these jump from one picture to the next (a width measured
# 75 px, then 143, a ring of 20000 skin pixels, then 180) the correction jumps with them: on, off, twice as strong. So:
#   - the leg's width is the median of the neighbouring pictures' (person_geometry_t),
#   - a knee is changed by as much as its read of the leg can be trusted (reliability: soft, never on or off),
#   - the plane of the leg's light is not run along the leg out of a ring that lies on one side of the knee only,
#   - two knees that lie on each other are not changed twice.
KNEE_V2 = True            # False: the 1.25 knee tone as it was
REL_RING = (0.02, 0.07)   # of the leg's skin the ring could have (the leg's strip over the ring's geometry): the share that is read: from where a knee is changed at all .. fully
REL_AREA = (0.05, 0.20)   # the same for the knee's own area (the leg's strip over it): a sliver of a knee is left alone
TWO_SIDED = (0.05, 0.25)  # how evenly the ring lies on both sides of the knee along the leg (0 = one side only, 1 = even): from where the plane may run along the leg .. fully
NAT_MIN = 0.008           # the leg's own variation (luma) is never read as smaller than this: a ring of a few alike pixels says nothing of what is natural
GEO_WIN = 4               # pictures each side whose width measures are taken with the picture's own (median)
DEBUG_INFO = None         # a list: every knee's reliability numbers are appended (for the desk, retouch tests)
# ... and from one picture to the next: the skin area of a person is made for every picture on its own, and a knee that is plainly there in one
# picture has none in the next (a parser that lost the leg for a picture, a hem over it). The correction then is on, off, on, at the full size of the
# blotch it takes away. The render looks a few pictures ahead (knee_support of each picture is made known beforehand) and the strength of a knee
# is held to what the pictures round it allow: it is never more than the support of the picture itself, and never more than ENV_STEP over the
# support of a picture one step further away, per step (knee_envelope). A knee with support for a long stretch is corrected fully in its
# middle, and fades in and out at its ends; a stretch of a picture or two is hardly corrected at all.
KNEE_ENV = False          # the render's look ahead (warp/render.py) is on for the knee tone; False: every picture on its own
ENV_STEP = 0.2            # the most the strength of a knee may change from one picture to the next (= 1 / the pictures a fade takes)


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


def _plane_keep(I: np.ndarray, w: np.ndarray, axis, keep: float) -> Optional[np.ndarray]:
    """_plane, with the plane's slope along the leg (axis, unit x y) multiplied by keep (0..1) and the plane laid again over
    the pixels (its level the weighted mean of what is left of them): a ring on one side of the knee only says what the leg's
    light does over its own length there, not what it does at the knee."""
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
    ax, ay = float(axis[0]), float(axis[1])
    gu = (c[1] * ax + c[2] * ay) * float(np.clip(keep, 0.0, 1.0))        # along the leg
    gv = -c[1] * ay + c[2] * ax                                           # across it
    bx, by = gu * ax - gv * ay, gu * ay + gv * ax
    lvl = float((ww * (I[ys, xs].astype(np.float64) - bx * (xs - x0) - by * (ys - y0))).sum() / max(float(ww.sum()), 1e-9))
    h, wd = I.shape
    gy, gx = np.mgrid[0:h, 0:wd].astype(np.float32)
    return (lvl + bx * (gx - x0) + by * (gy - y0)).astype(np.float32)


def knee_geometry(shape: Tuple[int, int], centre, axis, half: float, along: float = 1.0):
    """The elliptical radius of every pixel from a knee (1 = the edge of the ellipse with the semi axes of KNEE_ALONG and
    KNEE_ACROSS half widths), for a knee at centre (x, y), a leg running along axis (unit, x y) and half its width.
    along: the share of KNEE_ALONG the ellipse is long (1; less for a leg whose bare stretch is short, see knee_rescue_geometry)."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = gx - float(centre[0]), gy - float(centre[1])
    u = dx * float(axis[0]) + dy * float(axis[1])
    v = -dx * float(axis[1]) + dy * float(axis[0])
    return np.sqrt((u / (KNEE_ALONG * float(along) * half)) ** 2 + (v / (KNEE_ACROSS * half)) ** 2).astype(np.float32)


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


# 2611 IWALY c003 (2026-11-22, user, 00:00:39:17, "the knee's retouch must be applied"): the tracks put a knee at the top edge of a knee-high sock (or under a hem),
# 58 px (1.7 half widths) below the knee the leg shows: the skin lies above it. The knee's own area then lay on the sock, hardly any of it was skin
# (a thirtieth), the read of the leg was not trusted and nothing was changed. The centre of a knee whose own area is mostly not skin is moved along its
# leg to where the skin is (the area as the skin area has it, kept DARK_RIM away from what is not skin along the leg, as the skin read for the knee is).
KNEE_SLIDE = 2.2          # the most a knee's centre is moved along the leg, in half widths (0 = off)
KNEE_SLIDE_FROM = 0.60    # only a knee whose own area is less than this much on skin is moved
KNEE_SLIDE_GAIN = 0.15    # ... and only to a place whose area is this much more on skin
KNEE_SLIDE_COST = 0.04    # (per half width moved: the nearer of two alike places)


def knee_slide(sk: Optional[np.ndarray], centre, axis, half: float, slide: Optional[float] = None):
    """The centre of a knee moved along its leg (axis, unit x y) to where the skin is: sk: 0..1 the skin area in a plane, centre (x, y) and half (half the leg's
    width) in that plane's pixels. -> the centre (the one given where the knee's own area is on skin already, or nowhere does better)."""
    smax = KNEE_SLIDE if slide is None else float(slide)
    if smax <= 0 or sk is None or not half > 1.5:
        return centre
    h, w = sk.shape
    cx, cy = float(centre[0]), float(centre[1])
    R = int((smax + KNEE_ALONG * KNEE_EDGE[1] + DARK_RIM + 0.3) * half) + 4
    x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return centre
    sub = np.clip(np.asarray(sk[y0:y1, x0:x1], np.float32), 0.0, 1.0)
    f = max(1, int(half // 8))                                   # (looked at on a smaller picture: the area is smooth)
    if f > 1:
        sub = cv2.resize(sub, (max(8, (x1 - x0) // f), max(8, (y1 - y0) // f)), interpolation=cv2.INTER_AREA)
    hs = half / f
    c0 = ((cx - x0) / f, (cy - y0) / f)
    ax, ay = float(axis[0]), float(axis[1])
    r = max(1, int(round(DARK_RIM * hs)))
    ln = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.line(ln, (r - int(round(ax * r)), r - int(round(ay * r))), (r + int(round(ax * r)), r + int(round(ay * r))), 1, 1)
    inner = cv2.erode((sub > 0.5).astype(np.uint8), ln, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(np.float32)
    inner = np.minimum(sub, cv2.GaussianBlur(inner, (0, 0), max(0.8, 0.12 * hs)))
    ds = np.arange(-smax, smax + 1e-6, 0.2)
    covs = np.zeros(len(ds), np.float64)
    for i, d in enumerate(ds):
        c_ = (c0[0] + d * hs * ax, c0[1] + d * hs * ay)
        e = knee_geometry(inner.shape, c_, (ax, ay), hs)
        area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
        _, leg = _leg_strip(inner.shape, c_, (ax, ay), hs)
        al = area * leg
        covs[i] = float((al * inner).sum()) / max(float(al.sum()), 1e-6)
    i0 = int(np.argmin(np.abs(ds)))
    if covs[i0] >= KNEE_SLIDE_FROM:
        return centre
    ib = int(np.argmax(covs - KNEE_SLIDE_COST * np.abs(ds)))
    if covs[ib] - covs[i0] < KNEE_SLIDE_GAIN:
        return centre
    d = float(ds[ib])
    return (cx + d * half * ax, cy + d * half * ay)


# 2611 IWALY c003 (2026-11-22, user, 00:00:39:17, "the knee's retouch must be applied"), the second half of the same knee: the leg's bare stretch there is SHORT (a sock under the
# knee, a hem over it: 133 px bare at a half width of 34, 2.3 half widths of it with DARK_RIM kept off the cloth at both ends), the knee's own area (3.1 half widths long) and
# its ring (from 1.6 half widths out) do not fit in it, wherever the knee is moved to (KNEE_SLIDE): the area lies on cloth for the most part and nothing is changed.
# A knee that is not trusted (reliability under RESCUE_FROM) is looked at again with a SHORTER model: the same ellipse, its length along the leg RESCUE_ALONG times (the
# longest that has the skin it needs), put where the bare stretch has the most room on both sides, the leg's skin being the area's skin and, in the shade round it, the leg's
# own colour. The look ahead (knee_support) does the same on the area alone, so that it does not hold back what the tone is able to change.
KNEE_RESCUE = True
RESCUE_FROM = 0.5         # a knee is looked at again when its read is trusted less than this
RESCUE_ALONG = (1.0, 0.85, 0.70, 0.58, 0.45)   # the shares of KNEE_ALONG tried, the longest first (a knee on a stretch too short for the last is left alone)
RESCUE_SLIDE = 2.6        # the most the knee is moved along the leg, in half widths
RESCUE_AREA = 0.60        # the shortened knee's own area must be this much on skin (the skin with DARK_RIM kept off the cloth)
RESCUE_COST_D = 0.04      # of the score per half width the knee is moved
RESCUE_TWO = 0.5          # the share of the ring's evenness on both sides of the knee in the score (the knee in the middle of the stretch)
RESCUE_LONG = 0.5         # the share of the model's length (a longer one takes in more of the knee) in the score
RESCUE_NEAR = 4.0         # the width of colour that is the leg's, in the shade of the knee: this times the median spread of the ring's colour
RESCUE_SEED = 0.8         # of a half width: the leg's own colour is added to the skin area only this near to the area
RESCUE_TRUST = 0.9        # a rescued knee is trusted this much of what its own numbers say ...
RESCUE_GAIN = 0.10        # ... and is taken over the knee as it was read only when it is this much more trusted
RESCUE_READ = 0.7         # the leg's colour is read from the part of the strip the body skin's area holds to this degree (when it holds enough of it)
RESCUE_MAD_MIN = 0.004    # the least spread of the leg's colour the lenient reading assumes (a very clean picture has none: a half of the leg in shade is as far from the colour as a few spreads)
RESCUE_TABLE = None       # a list: every place tried is appended (along, shift, area, ring, two, fits, score), for the desk's debugging


def _ringw(e: np.ndarray) -> np.ndarray:
    """The weight of the ring (0..1) from a knee's elliptical radius."""
    return (_ss(RING[0], RING[0] + 0.35, e) * (1.0 - _ss(RING[1] - 0.6, RING[1], e))).astype(np.float32)


def _knee_window(sk: np.ndarray, centre, half: float, reach: float):
    """The window of a skin plane round a knee (reach half widths each side), on a smaller picture (the area is smooth): (sub, hs, c0, f): the window, half the leg's
    width and the knee's centre in its pixels, the reduction. None where the window is too small."""
    h, w = sk.shape
    cx, cy = float(centre[0]), float(centre[1])
    R = int(reach * half) + 4
    x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sub = np.clip(np.asarray(sk[y0:y1, x0:x1], np.float32), 0.0, 1.0)
    f = max(1, int(half // 8))
    if f > 1:
        sub = cv2.resize(sub, (max(8, (x1 - x0) // f), max(8, (y1 - y0) // f)), interpolation=cv2.INTER_AREA)
    return sub, half / f, ((cx - x0) / f, (cy - y0) / f), f


def _inner_skin(sub: np.ndarray, hs: float, axis) -> np.ndarray:
    """A skin plane with DARK_RIM (of a half width, hs pixels) kept off what is not skin along the leg (axis): the rim of a sock, a hem, a shoe is not the leg's."""
    ax, ay = float(axis[0]), float(axis[1])
    r = max(1, int(round(DARK_RIM * hs)))
    ln = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.line(ln, (r - int(round(ax * r)), r - int(round(ay * r))), (r + int(round(ax * r)), r + int(round(ay * r))), 1, 1)
    inner = cv2.erode((sub > 0.5).astype(np.uint8), ln, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(np.float32)
    return np.minimum(sub, cv2.GaussianBlur(inner, (0, 0), max(0.8, 0.12 * hs)))


def knee_rescue(sk: Optional[np.ndarray], centre, axis, half: float, slide: Optional[float] = None):
    """A knee on a short bare stretch of its leg: the model of the knee (the share of KNEE_ALONG its own area is long) that has the skin it needs (the longest that is
    read from both sides of the stretch), and the place along the leg where the stretch has the most room for it. sk: 0..1 the skin area in a plane, centre (x, y) and half (half the leg's width) in that plane's
    pixels. -> (x, y, along, info) of the knee as it is put, or None (no skin to be read from, or the stretch is too short even for the shortest model)."""
    if not KNEE_RESCUE or sk is None or not half > 1.5:
        return None
    smax = RESCUE_SLIDE if slide is None else float(slide)
    win = _knee_window(sk, centre, half, smax + RING[1] * KNEE_ALONG)
    if win is None:
        return None
    sub, hs, c0, f = win
    ax, ay = float(axis[0]), float(axis[1])
    inner = _inner_skin(sub, hs, (ax, ay))
    ds = np.arange(-smax, smax + 1e-6, 0.2)
    best = None
    for a in RESCUE_ALONG:
        for d in ds:
            c_ = (c0[0] + d * hs * ax, c0[1] + d * hs * ay)
            e = knee_geometry(inner.shape, c_, (ax, ay), hs, a)
            area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
            u_, leg = _leg_strip(inner.shape, c_, (ax, ay), hs)
            al = area * leg
            rw = _ringw(e) * leg
            sa, sr = float(al.sum()), float(rw.sum())
            if sa < 1e-3 or sr < 1e-3:
                continue
            cov_a = float((al * inner).sum()) / sa
            rin = rw * inner
            cov_r = float(rin.sum()) / sr
            up, dn = float(rin[u_ < 0].sum()), float(rin[u_ >= 0].sum())
            two = 2.0 * min(up, dn) / max(up + dn, 1e-6)
            ok = cov_a >= RESCUE_AREA and cov_r >= REL_RING[1]
            sc = cov_a + RESCUE_TWO * two + RESCUE_LONG * float(a) - RESCUE_COST_D * abs(float(d))
            if RESCUE_TABLE is not None:
                RESCUE_TABLE.append((float(a), round(float(d), 2), round(cov_a, 3), round(cov_r, 4), round(two, 3), bool(ok), round(sc, 3)))
            if ok and (best is None or sc > best[0]):
                best = (sc, float(d), cov_a, cov_r, two, float(a))
    if best is None:
        return None
    d, a = best[1], best[5]
    return (float(centre[0]) + d * half * ax, float(centre[1]) + d * half * ay, a,
            dict(along=a, shift=round(d, 2), fit_area=round(best[2], 3), fit_ring=round(best[3], 4), fit_two=round(best[4], 3)))


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


def _local_skin_parts(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, centre, axis, half: float, e: np.ndarray, lenient: float = 0.0, seed: Optional[np.ndarray] = None):
    """What local_skin works out, in pieces: a dict of the leg's strip (leg), its light with the rim of what is dark off it (lt), the leg's colour near each pixel
    (near; near2 with lenient times the spread's width more room, for a half of the leg in shade) and the ring (ring); None where the strip does not show enough
    of a leg. seed: the body skin's area (0..1) when it is known: the leg's colour is read from the part of the strip it holds (RESCUE_READ of it or more), not from the whole
    strip, which reaches the wall beside a short leg and takes its flat colour for the leg's."""
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
    if seed is not None:
        known = core & (seed > RESCUE_READ)
        if int(known.sum()) >= 60:
            core = known
            yref = float(np.percentile(Y[core], 75))
            light = _ss(NOT_SKIN[0], NOT_SKIN[1], Y / max(yref, 1e-4)).astype(np.float32)
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
    near2 = None
    if lenient > 0:
        sig2 = max(2.0 * CHROMA_SIG, float(lenient) * max(mad, RESCUE_MAD_MIN))
        near2 = np.exp(-0.5 * (d / sig2) ** 2).astype(np.float32)
    # the rim of a sock, a hem, a shoe (its blur, the shade it throws on the leg) is not the leg's skin either
    r = int(max(2, DARK_RIM * half))
    dark = cv2.dilate((light < 0.6).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    rim = 1.0 - cv2.GaussianBlur(dark.astype(np.float32), (0, 0), max(1.0, 0.4 * r))
    lt = (light * rim).astype(np.float32)
    thigh = 1.0 - (1.0 - SHIN_SHARE) * _ss(0.0, 0.5, u / half)     # (the thigh runs up from the knee, the shin down: the shin is read less, it is the one that sits in a sock's shade)
    ring = (leg * lt * near * thigh * _ss(RING[0], RING[0] + 0.35, e) * (1.0 - _ss(RING[1] - 0.6, RING[1], e))).astype(np.float32)
    return dict(leg=leg, light=light, lt=lt, near=near, near2=near2, ring=ring)


def local_skin(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, centre, axis, half: float, e: np.ndarray):
    """The skin of one leg round a knee when the body skin's area is not there (the parser found nothing to hold on to:
    a dim picture, a colour cast, dark tights): only the leg's own strip, from the joints. -> (area, ring): the strip
    without the dark (sock, hem, shoe), and the part of it round the knee whose colour is the leg's (what is read from).
    None where the strip does not show enough of a leg."""
    p = _local_skin_parts(Y, Cb, Cr, centre, axis, half, e)
    if p is None:
        return None
    return (p["leg"] * p["lt"]).astype(np.float32), p["ring"]


def tone_plane(P: np.ndarray, area: np.ndarray, ring: np.ndarray, half: float, k: float, kind: str,
               firm: Optional[np.ndarray] = None, ring_res: Optional[np.ndarray] = None,
               rel: float = 1.0, axis=None, keep: float = 1.0) -> np.ndarray:
    """One plane (a luma or a chroma plane) of the knee's tone work. area: 0..1 where the knee's own area is (inside the
    leg's skin), ring: 0..1 the skin the colour is read from, half: half the leg's width in this plane's pixels,
    k: 0..1, kind: 'luma' or 'chroma'. rel: 0..1 how far the read of the leg can be trusted (what is changed is multiplied
    by it, after the limit), axis, keep: the leg's direction and the share of the leg's light along it that is run over the
    knee (a luma plane). -> the plane (a copy)."""
    out = P.astype(np.float32, copy=True)
    r = _roi(area, int(3.2 * half) + 6)
    if r is None or k < 1e-4 or rel < 1e-4:
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
    if kind == "luma" and axis is not None and keep < 0.999:
        pl = _plane_keep(base, W, axis, keep)
    else:
        pl = _plane(base, W) if kind == "luma" else None  # (colour: the ring's own, one value; the leg's light is what runs along it)
    S = pl if pl is not None else np.full_like(base, float((base * W).sum() / max(float(W.sum()), 1e-6)))
    flat = base - S                                       # (S is the plane here: what the leg's skin does by itself, as seen from the ring)
    ringpx = W > 0.5
    nat = 1.4826 * float(np.median(np.abs(flat[ringpx] - np.median(flat[ringpx])))) if int(ringpx.sum()) > 30 else 0.0
    if KNEE_V2 and kind == "luma":
        nat = max(nat, NAT_MIN)
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
    if rel < 0.9999:
        d = d * float(rel)
    out[y0:y1, x0:x1] = I + d
    return out


def _knee_luma_v3(Y, w_area, w_ring, half, k, w_far, rel, axis, keep, blotch):
    """The knee's brightness (v3): evened only where the colour says blotch (blotch, 0..1 on Y's grid) or where it is far darker than
    the leg's own variation (KNEE_DARK_FAR); a knee darker than the leg by nature keeps its shading."""
    r = _roi(w_area, int(3.2 * half) + 6)
    if r is None:
        return Y.astype(np.float32, copy=True)
    y0, y1, x0, x1 = r
    sig_f = max(2.0, FINE * half)
    base = cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), sig_f)
    W = w_ring[y0:y1, x0:x1].astype(np.float32)
    sw = float(W.sum())
    if sw < 40.0:
        return Y.astype(np.float32, copy=True)
    A = w_area[y0:y1, x0:x1].astype(np.float32)
    ws = np.clip(A + W, 0, 1)
    sg = max(2.0, 0.6 * half)
    loc = cv2.GaussianBlur(base * ws, (0, 0), sg) / np.maximum(cv2.GaussianBlur(ws, (0, 0), sg), 1e-4)
    dv = loc - base
    inA = A > 0.5
    nat = max(NAT_MIN * 0.25, 1.4826 * float(np.median(np.abs(dv[inA] - np.median(dv[inA]))))) if int(inA.sum()) > 30 else NAT_MIN
    far = _ss(KNEE_DARK_FAR[0], KNEE_DARK_FAR[1], np.maximum(dv, 0.0) / nat) * _ss(KNEE_INNER[0], KNEE_INNER[1], A)
    gate = far if blotch is None else np.maximum(far, blotch[y0:y1, x0:x1])
    gate = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), max(1.5, 0.15 * half))
    firm = np.zeros(Y.shape, np.float32)
    firm[y0:y1, x0:x1] = gate
    return tone_plane(Y, w_area, w_ring, half, k, "luma", firm=firm, ring_res=w_far, rel=rel, axis=axis, keep=keep)


def owner_feather(gate: np.ndarray, owner: Optional[np.ndarray], half: float) -> np.ndarray:
    """gate (0..1) faded in from the edge of `owner` (> 0.5): 0 at that edge, full OWNER_FEATHER half widths inside. owner: what this
    person owns (another person's leg in front ends it), or the skin gate itself (then the sock's edge, a hem's, an occluder's too)."""
    if owner is None:
        return gate
    rad = max(2.0, OWNER_FEATHER * float(half))
    inside = cv2.distanceTransform((owner > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    ramp = _ss(0.0, rad, inside).astype(np.float32)
    ramp = cv2.GaussianBlur(ramp, (0, 0), max(0.8, 0.25 * rad))
    return (gate * ramp).astype(np.float32)


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


def _leg_strip(shape, centre, axis, half: float):
    """(u, leg): u the distance along the leg from the knee (px), leg 0..1 where a pixel is across the leg's own width."""
    u, v = leg_coords(shape, centre, axis)
    leg = (1.0 - _ss(LEG_EDGE[0], LEG_EDGE[1], np.abs(v) / half)).astype(np.float32)
    return u, leg


def knee_reliability(e: np.ndarray, area: np.ndarray, w_area: np.ndarray, w_ring: np.ndarray, centre, axis, half: float):
    """How far one knee's read of the leg can be trusted, from how much of the skin it could have read is there.

    e, area: the knee's elliptical radius and own area, w_area, w_ring: the knee's skin weights (_masked_skin / local_skin).
    -> (rel, keep, info): rel 0..1, the share of the knee's correction that is made (soft: nothing jumps from all to none),
    keep 0..1, the share of the leg's light along the leg that is run over the knee (0: the ring is on one side of the knee
    only), info: the numbers behind them."""
    u, leg = _leg_strip(e.shape, centre, axis, half)
    ringw = _ss(RING[0], RING[0] + 0.35, e) * (1.0 - _ss(RING[1] - 0.6, RING[1], e))
    nom_r = float((ringw * leg).sum())
    nom_a = float((area * leg).sum())
    sr, sa = float(w_ring.sum()), float(w_area.sum())
    cov_r = min(1.0, sr / max(nom_r, 1e-6))
    cov_a = min(1.0, sa / max(nom_a, 1e-6))
    sup = w_ring > 0.2                                    # (counted, not weighed: the leg's own strip reads the shin less on purpose)
    up = float(sup[u < 0].sum())
    dn = float(sup[u >= 0].sum())
    two = 2.0 * min(up, dn) / max(up + dn, 1e-6)
    r_ring = float(_ss(REL_RING[0], REL_RING[1], cov_r))
    r_area = float(_ss(REL_AREA[0], REL_AREA[1], cov_a))
    keep = float(_ss(TWO_SIDED[0], TWO_SIDED[1], two))
    rel = r_ring * r_area
    return rel, keep, dict(cov_ring=round(cov_r, 4), cov_area=round(cov_a, 4), two=round(two, 3), r_ring=round(r_ring, 3), r_area=round(r_area, 3),
                           keep=round(keep, 3), rel=round(rel, 3), half=round(float(half), 1), cx=round(float(centre[0]), 1), cy=round(float(centre[1]), 1))


def _cover_at(m: np.ndarray, c0, axis, hh: float, along: float = 1.0) -> float:
    """How much of the leg's skin a skin area has round a knee put at c0 (the area's pixels, hh = half the leg's width in them): the weighing of knee_reliability's, on the
    area as it is (0..1)."""
    h, w = m.shape
    ax, ay = float(axis[0]), float(axis[1])
    R = int(RING[1] * KNEE_ALONG * float(along) * hh) + 2
    x0, x1 = max(0, int(c0[0]) - R), min(w, int(c0[0]) + R + 1)
    y0, y1 = max(0, int(c0[1]) - R), min(h, int(c0[1]) + R + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    sub = np.asarray(m[y0:y1, x0:x1], np.float32)
    c_ = (c0[0] - x0, c0[1] - y0)
    e = knee_geometry(sub.shape, c_, (ax, ay), hh, along)
    area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
    _, leg = _leg_strip(sub.shape, c_, (ax, ay), hh)
    ringw = _ringw(e)
    nom_a = float((area * leg).sum())
    nom_r = float((ringw * leg).sum())
    if nom_a < 1e-3 or nom_r < 1e-3:
        return 0.0
    cov_a = min(1.0, float((area * leg * sub).sum()) / nom_a)
    cov_r = min(1.0, float((ringw * leg * sub).sum()) / nom_r)
    return float(_ss(REL_AREA[0], REL_AREA[1], cov_a) * _ss(REL_RING[0], REL_RING[1], cov_r))


def knee_support(m: np.ndarray, scale: float, knees) -> list:
    """How much of the leg's skin one picture's skin area has round each of a person's knees: the same weighing as knee_reliability's, on the area as
    the analysis made it (before the picture's own colour and brightness are asked: that is made when the picture is worked on). A knee on a short bare
    stretch of its leg (knee_rescue) is looked at with the shorter model, as the tone does.

    m: the person's body skin area on the analysis picture (float32 0..1), scale: that picture's pixels per source pixel,
    knees: person_geometry's [(cx, cy, ax, ay, half)] in source px. -> [0..1 for each knee]"""
    out = []
    for (cx, cy, ax, ay, half) in knees:
        hh = float(half) * float(scale)
        if not hh > 1.5:
            out.append(0.0)
            continue
        c0 = (float(cx) * scale, float(cy) * scale)
        if KNEE_SLIDE > 0:
            try:
                c0 = knee_slide(m, c0, (ax, ay), hh)
            except Exception:
                pass
        s0 = _cover_at(m, c0, (ax, ay), hh)
        if KNEE_RESCUE and s0 < RESCUE_FROM:
            try:
                fit = knee_rescue(m, c0, (ax, ay), hh)
            except Exception:
                fit = None
            if fit is not None:
                s0 = max(s0, RESCUE_TRUST * _cover_at(m, (fit[0], fit[1]), (ax, ay), hh, fit[2]))
        out.append(float(s0))
    return out


def env_reach(step: Optional[float] = None) -> int:
    """How many pictures each side of a picture its envelope looks at (a support of 1 is no more than step per picture away from 0 after this many)."""
    st = ENV_STEP if step is None else float(step)
    return int(np.ceil(1.0 / max(st, 1e-3)))


def knee_owned_share(owner: np.ndarray, scale: float, knee) -> float:
    """0..1: how much of a knee's own area (in source px: (cx, cy, ax, ay, half)) this person owns on a picture `scale` times the source
    (another person's leg in front takes it). Read into the look ahead's support so that a knee being covered fades (knee_envelope)."""
    try:
        cx, cy, ax, ay, half = [float(v) for v in knee]
        if not np.isfinite([cx, cy, ax, ay, half]).all() or half <= 0:
            return 1.0
        h, w = owner.shape
        c = (cx * scale, cy * scale)
        hh = half * scale
        R = int(KNEE_ALONG * KNEE_EDGE[1] * hh) + 2
        x0, x1 = max(0, int(c[0]) - R), min(w, int(c[0]) + R + 1)
        y0, y1 = max(0, int(c[1]) - R), min(h, int(c[1]) + R + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return 1.0
        e = knee_geometry((y1 - y0, x1 - x0), (c[0] - x0, c[1] - y0), (ax, ay), max(hh, 1.0))
        area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
        sa = float(area.sum())
        if sa < 4:
            return 1.0
        share = float((area * np.clip(owner[y0:y1, x0:x1], 0, 1)).sum()) / sa
        return float(_ss(0.15, 0.5, share))
    except Exception:
        return 1.0


KNEE_OVERLAP = (1.0, 2.4)   # distance of a person's two knees in half widths: from where they overlap (crossing legs) .. apart
KNEE_OVERLAP_FLOOR = 0.35   # the share of the knee work left while the knees overlap (the front leg's shade on the other one was
                            # evened for a frame or two as a blotch: a flash, c017 f28/29); the look ahead ramps it in and out


KNEE_CROSS_MOVE = (0.25, 0.7)   # how fast the knees' distance changes (half widths per picture): from where it counts as crossing .. fully


def _knee_dist(knees):
    if len(knees) != 2:
        return None
    a, b = [np.asarray(k_, np.float64) for k_ in knees]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    h = 0.5 * (a[4] + b[4])
    return None if h <= 0 else float(np.hypot(a[0] - b[0], a[1] - b[1]) / h)


def knee_overlap_support(knees, knees_prev=None) -> list:
    """Per knee: KNEE_OVERLAP_FLOOR while the person's two knees cross each other (close together and their distance changing fast
    from the picture before), 1 otherwise: knees held together stay corrected in full, only the crossing is ramped (knee_envelope)."""
    out = [1.0] * len(knees)
    try:
        d = _knee_dist(knees)
        dp = _knee_dist(knees_prev) if knees_prev is not None else None
        if d is not None and dp is not None:
            close = 1.0 - float(_ss(KNEE_OVERLAP[0], KNEE_OVERLAP[1], min(d, dp)))
            moving = float(_ss(KNEE_CROSS_MOVE[0], KNEE_CROSS_MOVE[1], abs(d - dp)))
            f = 1.0 - (1.0 - KNEE_OVERLAP_FLOOR) * close * moving
            out = [f] * len(knees)
    except Exception:
        pass
    return out


def knee_envelope(support, t: int, lo: int, hi: int, step: Optional[float] = None) -> float:
    """The most a knee may be corrected on picture t: min over the pictures u in [lo, hi] of support[u] + step * |t - u|, 0 for a picture that has no
    support entry. support: dict picture -> 0..1 (or any object with .get). Never more than the picture's own support; a picture the supported stretch
    is short of (step per picture) reaches full strength only in the middle of a stretch of 2 / step pictures."""
    st = ENV_STEP if step is None else float(step)
    r = env_reach(st)
    best = 1.0
    for u in range(max(lo, t - r), min(hi, t + r) + 1):
        v = float(support.get(u, 0.0)) + st * abs(t - u)
        if v < best:
            best = v
    return float(max(0.0, min(1.0, best)))


def _knee_tone_v1(Y, Cb, Cr, skin, knees, k, cs):
    """The knee tone as 1.25 had it: one knee after the other, each on what the one before left."""
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


def _merge(d_all: Optional[np.ndarray], d: np.ndarray) -> np.ndarray:
    """Two knees' changes of a plane laid together: where both change a pixel, the larger of the two stays (never their sum)."""
    if d_all is None:
        return d
    return np.where(np.abs(d) > np.abs(d_all), d, d_all).astype(np.float32)


def _union_skin(Y, Cb, Cr, skin, centre, axis, half: float, e: np.ndarray) -> np.ndarray:
    """The body skin's area with the leg's own colour added to it where the leg is in the shade or the area has a hole: in the leg's strip, what has the colour the leg's skin has
    round the knee (a wider spread of it than the ring's, RESCUE_NEAR) and is not dark, within RESCUE_SEED half widths of the area. The area itself where that cannot be read."""
    sk = np.clip(skin, 0.0, 1.0).astype(np.float32)
    p = _local_skin_parts(Y, Cb, Cr, centre, axis, half, e, lenient=RESCUE_NEAR, seed=sk)
    if p is None or p.get("near2") is None:
        return sk
    hY, wY = sk.shape
    R = int((RESCUE_SLIDE + RING[1] * KNEE_ALONG) * half) + 4
    cx, cy = float(centre[0]), float(centre[1])
    x0, x1 = max(0, int(cx) - R), min(wY, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(hY, int(cy) + R + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return sk
    r = max(2, int(RESCUE_SEED * half))
    core = (sk[y0:y1, x0:x1] > 0.5).astype(np.uint8)
    seed = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    add = (p["leg"] * p["near2"] * p["light"])[y0:y1, x0:x1] * cv2.GaussianBlur(seed, (0, 0), max(1.0, 0.25 * r))
    out = sk.copy()
    out[y0:y1, x0:x1] = np.maximum(sk[y0:y1, x0:x1], add)
    return out


def _knee_read(Y, Cb, Cr, skin, centre, axis, half: float, along: float = 1.0):
    """One knee read from the picture: (e, area, w_area, w_ring, rel, keep, info) for a knee at centre whose own area is along (the share of KNEE_ALONG) long, or None where
    there is no area or no skin of the leg to read."""
    hY, wY = Y.shape
    e = knee_geometry((hY, wY), centre, axis, half, along)
    area = 1.0 - _ss(KNEE_EDGE[0], KNEE_EDGE[1], e)
    if float(area.max()) < 0.05:
        return None
    got = None
    if skin is not None:
        got = _masked_skin(Y, skin, e, area, half)
    else:
        ls = local_skin(Y, Cb, Cr, centre, axis, half, e)
        if ls is not None and int((ls[1] > 0.5).sum()) >= 60:
            got = ((area * ls[0]).astype(np.float32), ls[1])
    if got is None:
        return None
    w_area, w_ring = got
    rel, keep, info = knee_reliability(e, area, w_area, w_ring, centre, axis, half)
    info["along"] = round(float(along), 3)
    return e, area, w_area, w_ring, rel, keep, info


def _knee_rescue_read(Y, Cb, Cr, skin, centre, axis, half: float):
    """The knee read with the shorter model on a short bare stretch (knee_rescue, on the area with the leg's own colour added: _union_skin): as _knee_read, or None."""
    if skin is None:
        return None
    hY, wY = Y.shape
    e0 = knee_geometry((hY, wY), centre, axis, half)
    sku = _union_skin(Y, Cb, Cr, skin, centre, axis, half, e0)
    fit = knee_rescue(sku, centre, axis, half)
    if fit is None:
        return None
    rd = _knee_read(Y, Cb, Cr, sku, (fit[0], fit[1]), axis, half, fit[2])
    if rd is None:
        return None
    rd[6].update(rescue=True, shift=fit[3]["shift"], fit_area=fit[3]["fit_area"], fit_ring=fit[3]["fit_ring"], fit_two=fit[3]["fit_two"],
                 cx=round(float(fit[0]), 1), cy=round(float(fit[1]), 1))
    return rd


KNEE_ROI = 10.0           # > 0: every knee is worked on a piece of the picture this many half widths of the leg round it (its area, its ring out to RING[1] and
                          # the furthest slide lie within 8), not on the whole frame: the same numbers, a fraction of the time and memory on a 4K frame with
                          # several people's knees. 0: the whole frame for every knee, as it was


def _knee_tone_v2(Y, Cb, Cr, skin, knees, k, cs, gains=None):
    if not (KNEE_ROI > 0):
        return _knee_tone_v2_full(Y, Cb, Cr, skin, knees, k, cs, gains)
    hY, wY = Y.shape
    hC, wC = Cb.shape
    same = (hC, wC) == (hY, wY)
    fy, fx = hC / float(hY), wC / float(wY)
    sy = int(round(1.0 / fy)) if fy < 1 else 1          # luma rows per chroma row
    sx = int(round(1.0 / fx)) if fx < 1 else 1
    dY = dCb = dCr = None
    for ki_, (centre, axis, half) in enumerate(knees):
        half = float(half)
        if not (half > 1.5):
            continue
        R = int(KNEE_ROI * half) + 16
        x0, x1 = max(0, int(centre[0]) - R), min(wY, int(centre[0]) + R + 1)
        y0, y1 = max(0, int(centre[1]) - R), min(hY, int(centre[1]) + R + 1)
        if not same:
            x0, y0 = x0 - x0 % sx, y0 - y0 % sy
            x1, y1 = min(wY, x1 + (-x1) % sx), min(hY, y1 + (-y1) % sy)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        if same:
            cb_, cr_ = Cb[y0:y1, x0:x1], Cr[y0:y1, x0:x1]
            c0y, c1y, c0x, c1x = y0, y1, x0, x1
        else:
            c0y, c1y, c0x, c1x = y0 // sy, min(hC, -(-y1 // sy)), x0 // sx, min(wC, -(-x1 // sx))
            cb_, cr_ = Cb[c0y:c1y, c0x:c1x], Cr[c0y:c1y, c0x:c1x]
        ys_ = Y[y0:y1, x0:x1]
        sk_ = None if skin is None else skin[y0:y1, x0:x1]
        g_ = None if gains is None else [gains[ki_] if ki_ < len(gains) else None]
        yo, cbo, cro = _knee_tone_v2_full(ys_, cb_, cr_, sk_, [((float(centre[0]) - x0, float(centre[1]) - y0), axis, half)], k, cs, g_)
        if yo is not ys_:
            if dY is None:
                dY = np.zeros((hY, wY), np.float32)
            dY[y0:y1, x0:x1] = _merge(dY[y0:y1, x0:x1], (yo - ys_).astype(np.float32))
        if cbo is not cb_:
            if dCb is None:
                dCb = np.zeros((hC, wC), np.float32)
            dCb[c0y:c1y, c0x:c1x] = _merge(dCb[c0y:c1y, c0x:c1x], (cbo - cb_).astype(np.float32))
        if cro is not cr_:
            if dCr is None:
                dCr = np.zeros((hC, wC), np.float32)
            dCr[c0y:c1y, c0x:c1x] = _merge(dCr[c0y:c1y, c0x:c1x], (cro - cr_).astype(np.float32))
    Yo = Y if dY is None else (Y + dY).astype(np.float32)
    Cbo = Cb if dCb is None else (Cb + dCb).astype(np.float32)
    Cro = Cr if dCr is None else (Cr + dCr).astype(np.float32)
    return Yo, Cbo, Cro


def _knee_chroma_v3(Cb, Cr, wa, wr, half, k, rel):
    """The knee's colour (both chroma planes at once): blotches (purple, red, grey) to the knee's own even tint, which is the ring's
    colour plus KNEE_TINT_KEEP of the knee's non-blotched offset from it; never less saturated than the ring (KNEE_SAT_FLOOR).
    Only the band above the grain is changed (FINE). -> (Cb', Cr', blotch 0..1) on the chroma planes' grid, or None."""
    r = _roi(wa, int(3.2 * half) + 6)
    if r is None or k < 1e-4 or rel < 1e-4:
        return None
    y0, y1, x0, x1 = r
    A = wa[y0:y1, x0:x1].astype(np.float32)
    W = wr[y0:y1, x0:x1].astype(np.float32)
    sw = float(W.sum())
    if sw < 40.0:
        return None
    sig_f = max(2.0, FINE * half)
    B = [cv2.GaussianBlur(P[y0:y1, x0:x1].astype(np.float32), (0, 0), sig_f) for P in (Cb, Cr)]
    ring = [float((b * W).sum() / sw) for b in B]
    # the knee's even tint and its own chroma variation (robust: the blotches are the outliers of the knee's own colour, not of the
    # ring's, which holds the leg's shading and the other leg and read every bruise as natural)
    sa = float(A.sum())
    if sa < 20.0:
        return None
    inA = A > 0.5
    ringpx = W > 0.5
    if int(inA.sum()) < 30:
        return None
    tint = [float(np.median(b[inA])) for b in B]
    nat_c = KNEE_NAT_C_MIN
    for _ in range(3):
        dist_ = np.hypot(B[0] - tint[0], B[1] - tint[1])
        nat_c = max(KNEE_NAT_C_MIN, 1.4826 * float(np.median(dist_[inA])))
        wt = A * (1.0 - _ss(KNEE_BLOTCH[0], KNEE_BLOTCH[1], dist_ / nat_c))
        s_ = float(wt.sum())
        if s_ < 0.15 * sa:
            break
        tint = [float((b * wt).sum() / s_) for b in B]
    tint = [ring[i] + KNEE_TINT_KEEP * (tint[i] - ring[i]) for i in range(2)]
    # A blotch is told by its hue, not by its distance from the knee's mean: the skin's own variation (shade, a little more or less
    # colour) runs along the skin's tone axis (the ring's chroma direction), a bruise (purple, blue, red-violet) turns away from it.
    # The spread across the axis is read from the ring and the knee together (a bruise over a quarter of the knee does not hide
    # itself in the knee's own spread). A red patch along the axis counts only far out (KNEE_RED_FAR).
    rn = float(np.hypot(*ring))
    if rn < 1e-5:
        return None
    u = (ring[0] / rn, ring[1] / rn)
    along = B[0] * u[0] + B[1] * u[1]
    perp = -B[0] * u[1] + B[1] * u[0]
    both = ringpx | inA
    pm = float(np.median(perp[both]))
    am = float(np.median(along[both]))
    nat_p = max(KNEE_NAT_C_MIN, 1.4826 * float(np.median(np.abs(perp[ringpx] - np.median(perp[ringpx]))))) if int(ringpx.sum()) > 30 else KNEE_NAT_C_MIN
    nat_a = max(KNEE_NAT_C_MIN, 1.4826 * float(np.median(np.abs(along[ringpx] - np.median(along[ringpx]))))) if int(ringpx.sum()) > 30 else KNEE_NAT_C_MIN
    hue_off = _ss(KNEE_BLOTCH[0], KNEE_BLOTCH[1], np.abs(perp - pm) / nat_p)
    red_off = _ss(KNEE_RED_FAR[0], KNEE_RED_FAR[1], (along - am) / nat_a)
    # (only well inside the knee's skin: the skin next to a sock, a hem or another leg is shaded and coloured by it, and was read as a
    # blotch and repainted along that edge)
    blotch = (np.maximum(hue_off, red_off) * _ss(KNEE_INNER[0], KNEE_INNER[1], A)).astype(np.float32)
    # (a blotch's soft rim is part of it: no ring of the old colour round a corrected spot)
    blotch = np.maximum(blotch, np.clip(1.6 * cv2.GaussianBlur(blotch, (0, 0), max(1.0, 0.20 * half)), 0, 1) * _ss(KNEE_INNER[0], KNEE_INNER[1], A)).astype(np.float32)
    # the knee's even tint again, now from its skin that is not blotched by the hue test (the first estimate held the bruise)
    wt_ = A * (1.0 - blotch)
    s_ = float(wt_.sum())
    if s_ >= 0.15 * sa:
        tint = [float((b * wt_).sum() / s_) for b in B]
    else:
        tint = list(ring)
    g = KNEE_EVEN + (1.0 - KNEE_EVEN) * blotch
    w = A * float(k) * float(rel) * g
    # the colour a blotch is filled with: the knee's own skin round it that is not blotched (its hue across the knee kept), the even
    # tint where there is too little of that
    non = (A * (1.0 - blotch) + 0.25 * W).astype(np.float32)
    sg = max(3.0, KNEE_FILL * half)
    den = cv2.GaussianBlur(non, (0, 0), sg)
    have = _ss(0.05, 0.25, den)
    ref = [have * (cv2.GaussianBlur(B[i] * non, (0, 0), sg) / np.maximum(den, 1e-4)) + (1.0 - have) * tint[i] for i in range(2)]
    nb = [B[i] + w * ((ref[i] + CHROMA_KEEP * (B[i] - ref[i])) - B[i]) for i in range(2)]
    # no grey: a changed pixel never comes out less saturated than the skin it is filled from (hue kept)
    ln = np.hypot(nb[0], nb[1])
    floor = KNEE_SAT_FLOOR * np.hypot(ref[0], ref[1]) * (w > 0.01)
    sc = np.where((ln < floor) & (ln > 1e-6), floor / np.maximum(ln, 1e-6), 1.0).astype(np.float32)
    nb = [B[i] + (nb[i] * sc - B[i]) for i in range(2)]
    out = []
    cap = KNEE_BLOTCH_CAP.get("chroma")
    for i, P in enumerate((Cb, Cr)):
        d = nb[i] - B[i]
        if cap:
            c0 = 0.6 * cap
            a = np.abs(d)
            d = np.where(a <= c0, d, np.sign(d) * (c0 + (cap - c0) * np.tanh((a - c0) / (cap - c0)))).astype(np.float32)
        o = P.astype(np.float32, copy=True)
        o[y0:y1, x0:x1] += d
        out.append(o)
    bl = np.zeros(Cb.shape, np.float32)
    bl[y0:y1, x0:x1] = blotch * (A > 0.02)
    return out[0], out[1], bl


def _knee_blotch_luma(Y, w_area, w_ring, blotch, half, k, rel):
    """A blotch's darkness filled from the knee's own skin round it (KNEE_FILL), above the grain only; lighter parts are left."""
    r = _roi(w_area, int(3.2 * half) + 6)
    if r is None or blotch is None:
        return Y.astype(np.float32, copy=True)
    y0, y1, x0, x1 = r
    A = w_area[y0:y1, x0:x1].astype(np.float32)
    W = w_ring[y0:y1, x0:x1].astype(np.float32)
    bl = blotch[y0:y1, x0:x1].astype(np.float32)
    sig_f = max(2.0, FINE * half)
    base = cv2.GaussianBlur(Y[y0:y1, x0:x1].astype(np.float32), (0, 0), sig_f)
    non = (A * (1.0 - bl) + 0.25 * W).astype(np.float32)
    sg = max(3.0, KNEE_FILL * half)
    den = cv2.GaussianBlur(non, (0, 0), sg)
    ref = cv2.GaussianBlur(base * non, (0, 0), sg) / np.maximum(den, 1e-4)
    d = np.maximum(ref - base, 0.0) * (1.0 - KNEE_DARK_KEEP) * _ss(0.05, 0.25, den)
    d = _soft_cap(A * bl * float(k) * float(rel) * d, KNEE_BLOTCH_CAP["luma"])
    out = Y.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] += d
    return out


def _knee_tone_v2_full(Y, Cb, Cr, skin, knees, k, cs, gains=None):
    """The knee tone with its read of the leg weighed (knee_reliability): every knee reads the picture as it came (not what
    another knee changed), changes it by as much as the read can be trusted, and where two knees change the same pixel the
    larger change stays. A knee that is not trusted is read again with a shorter model (knee_rescue) and that read is taken when it is trusted more."""
    hY, wY = Y.shape
    hC, wC = Cb.shape
    dY = dCb = dCr = None
    for ki_, (centre, axis, half) in enumerate(knees):
        half = float(half)
        if not (half > 1.5):
            continue
        gain = 1.0
        if gains is not None and ki_ < len(gains) and gains[ki_] is not None:
            gain = float(np.clip(gains[ki_], 0.0, 1.0))
            if gain < 1e-3:
                continue
        if skin is not None and KNEE_SLIDE > 0:
            try:
                centre = knee_slide(skin, centre, axis, half)
            except Exception:
                pass
        rd = _knee_read(Y, Cb, Cr, skin, centre, axis, half)
        if KNEE_RESCUE and skin is not None and (rd is None or rd[4] < RESCUE_FROM):
            try:
                rs = _knee_rescue_read(Y, Cb, Cr, skin, centre, axis, half)
            except Exception:
                rs = None
            if rs is not None and (rd is None or RESCUE_TRUST * rs[4] > rd[4] + RESCUE_GAIN):
                rd = (rs[0], rs[1], rs[2], rs[3], RESCUE_TRUST * rs[4], rs[5], rs[6])
        if rd is None:
            continue
        e, area, w_area, w_ring, rel, keep, info = rd
        info["gain"] = round(gain, 3)
        if DEBUG_INFO is not None:
            DEBUG_INFO.append(info)
        rel *= gain
        if rel < 1e-3:
            continue
        w_far = (w_ring * _ss(RES_FROM, RES_FROM + 0.4, e)).astype(np.float32)
        if KNEE_V3:
            if Cb.shape == (hY, wY):
                wa, wr, hf = w_area, w_ring, half
            else:
                wa = cv2.resize(w_area, (wC, hC), interpolation=cv2.INTER_AREA)
                wr = cv2.resize(w_ring, (wC, hC), interpolation=cv2.INTER_AREA)
                hf = max(1.5, half * cs)
            cr3 = _knee_chroma_v3(Cb, Cr, wa, wr, hf, k, rel)
            if cr3 is not None:
                dCb = _merge(dCb, cr3[0] - Cb)
                dCr = _merge(dCr, cr3[1] - Cr)
                bl = cr3[2] if cr3[2].shape == (hY, wY) else cv2.resize(cr3[2], (wY, hY), interpolation=cv2.INTER_LINEAR)
                Yb = _knee_blotch_luma(Y, w_area, w_ring, bl, half, k, rel)
                dY = _merge(dY, Yb - Y)
            # (a patch far darker than the knee's own variation with no colour to it: evened by the tone as before, gated)
            Yn = _knee_luma_v3(Y, w_area, w_ring, half, k, w_far, rel, axis, keep, None)
            dY = _merge(dY, Yn - Y)
            continue
        Yn = tone_plane(Y, w_area, w_ring, half, k, "luma", ring_res=w_far, rel=rel, axis=axis, keep=keep)
        dY = _merge(dY, Yn - Y)
        for nm in ("cb", "cr"):
            Pc = Cb if nm == "cb" else Cr
            if Pc.shape == (hY, wY):
                wa, wr, wf, hf = w_area, w_ring, w_far, half
            else:
                wa = cv2.resize(w_area, (wC, hC), interpolation=cv2.INTER_AREA)
                wr = cv2.resize(w_ring, (wC, hC), interpolation=cv2.INTER_AREA)
                wf = cv2.resize(w_far, (wC, hC), interpolation=cv2.INTER_AREA)
                hf = max(1.5, half * cs)
            Pn = tone_plane(Pc, wa, wr, hf, k, "chroma", ring_res=wf, rel=rel)
            if nm == "cb":
                dCb = _merge(dCb, Pn - Pc)
            else:
                dCr = _merge(dCr, Pn - Pc)
    Yo = Y if dY is None else (Y + dY).astype(np.float32)
    Cbo = Cb if dCb is None else (Cb + dCb).astype(np.float32)
    Cro = Cr if dCr is None else (Cr + dCr).astype(np.float32)
    return Yo, Cbo, Cro


def knee_tone(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, skin: Optional[np.ndarray], knees, k: float, cs: float = 1.0, gains=None):
    """The knee tone of one person on one picture.

    Y, Cb, Cr: the picture's planes (float; Cb and Cr cs times the size of Y, in pixels, 1 = the same size),
    skin: 0..1 the person's skin area at Y's size (retouch.bodyskin; None: there is none, read from the leg's own strip, local_skin), knees: [(centre (x, y), leg axis (unit), half the
    leg's width)] in Y's pixels, k: 0..1, gains: [0..1 for each knee]: the most each may be corrected on this picture (knee_envelope; None: no limit).
    -> (Y, Cb, Cr), copies (the planes themselves where nothing changed)."""
    if k < 1e-4 or not knees:
        return Y, Cb, Cr
    if KNEE_V2:
        return _knee_tone_v2(Y, Cb, Cr, skin, knees, k, cs, gains)
    return _knee_tone_v1(Y, Cb, Cr, skin, knees, k, cs)


# ---- where the knees and the belly are, from the tracks

def _finite(a) -> bool:
    return a is not None and np.isfinite(np.asarray(a, np.float64)).all()


def _knee_raw(kp: np.ndarray, widths: Optional[np.ndarray], tag: str):
    """One knee ('l' or 'r') of one picture as the tracks have it: (cx, cy, ax, ay, half) in source pixels, or None where the
    knee, or the line along its leg, is not there."""
    from ..warp import body as B
    from ..track import skeleton as S
    kn_i, hip_i, an_i = (S.L_KN, S.L_HIP, S.L_AN) if tag == "l" else (S.R_KN, S.R_HIP, S.R_AN)
    k = np.asarray(kp[kn_i, :2], np.float64)
    if not _finite(k):
        return None
    hp = np.asarray(kp[hip_i, :2], np.float64)
    an = np.asarray(kp[an_i, :2], np.float64)
    v1 = (k - hp) if _finite(hp) else None
    v2 = (an - k) if _finite(an) else None
    parts = []
    for v in (v1, v2):
        if v is not None and float(np.hypot(*v)) > 4:
            parts.append(v / float(np.hypot(*v)))
    if not parts:
        return None
    ax = np.sum(parts, 0)
    n = float(np.hypot(*ax))
    ax = ax / n if n > 1e-3 else parts[0]
    L = float(np.hypot(*(an - hp))) if (_finite(hp) and _finite(an)) else float(np.hypot(*(v1 if v1 is not None else v2)) * 2.0)
    half = np.nan
    if widths is not None:
        ws = []
        for nm in ("knee_t" + tag, "knee_c" + tag):
            w = np.asarray(widths[B.SEG_INDEX[nm]], np.float64)
            if np.isfinite(w).any():
                ws.append(float(np.nanmean(w)))
        if ws:
            half = float(np.mean(ws))
    if not np.isfinite(half) or half <= 0:
        half = 0.055 * L
    return (float(k[0]), float(k[1]), float(ax[0]), float(ax[1]), float(half))


def person_geometry_t(final_p: np.ndarray, widths_p: Optional[np.ndarray], t: int, win: Optional[int] = None) -> dict:
    """person_geometry of picture t of one person (final_p: the person's keypoints over the cut (T, 17, >= 2), widths_p: the
    widths over the cut (T, segments, 2) or None), with each knee's half width the median of the widths measured in the pictures
    t - win .. t + win: a leg's width does not change from one picture to the next, its measure does (75 px, then 143, then 61),
    and everything the knee tone reads is laid out in that width. From the tracks alone: the same for a picture whatever
    order the pictures are made in."""
    win = GEO_WIN if win is None else int(win)
    kp = final_p[t]
    wd = widths_p[t] if widths_p is not None else None
    if not KNEE_V2 or win <= 0:
        return person_geometry(kp, wd)
    halves = {}
    n = len(final_p)
    for tag in ("l", "r"):
        vals = []
        for s_ in range(max(0, t - win), min(n, t + win + 1)):
            try:
                g = _knee_raw(final_p[s_], widths_p[s_] if widths_p is not None else None, tag)
            except Exception:
                g = None
            if g is not None and np.isfinite(g[4]) and g[4] > 0:
                vals.append(g[4])
        if len(vals) >= 3:
            halves[tag] = float(np.median(vals))
    geometry = person_geometry(kp, wd, halves)
    # Reject isolated position spikes only when both neighbouring leg segments
    # also agree on the direction. Constant velocity dance motion passes through.
    if 0 < t < n - 1:
        for i, side in enumerate(geometry["sides"]):
            before, after = (_knee_raw(final_p[u], widths_p[u] if widths_p is not None else None, side)
                             for u in (t-1, t+1))
            if before is None or after is None:
                continue
            x, y, ax, ay, half = geometry["knees"][i]
            if np.dot(before[2:4], after[2:4]) < 0.92 or np.hypot(after[0]-before[0], after[1]-before[1]) > half:
                continue
            mid = (np.asarray(before[:2]) + np.asarray(after[:2])) * 0.5
            error = np.hypot(x-mid[0], y-mid[1]) / max(half, 1)
            blend = float(np.clip((error - 1.25) / 0.75, 0, 1))
            geometry["knees"][i] = (x + blend*(mid[0]-x), y + blend*(mid[1]-y), ax, ay, half)
    return geometry


def person_geometry(kp: np.ndarray, widths: Optional[np.ndarray], halves: Optional[dict] = None) -> dict:
    """The places the two effects work on, in source pixels, from one person's keypoints (17, >= 2) and the widths the
    tracking measured (one row of retouch's store: (segments, 2), may be None).

    -> dict(knees=[(cx, cy, ax, ay, half)], sides=['l' | 'r' of each knee], belly=None | dict(c=(x, y), u=(ux, uy), along=, across=, trunk=)):
    a knee's centre, the direction the leg runs in and half its width (halves: {'l': .., 'r': ..} replaces the measured one);
    the belly's centre (a little over half way down the trunk, from the neck to the pelvis), the trunk's direction, half the
    zone's length along it and across it, and the trunk's length."""
    from ..warp import body as B
    out = dict(knees=[], sides=[], belly=None)
    try:
        ends = B.segment_endpoints(np.asarray(kp, np.float32))
    except Exception:
        return out

    def seg(name):
        return B.SEG_INDEX[name]

    def finite(a):
        return _finite(a)
    for tag in ("l", "r"):
        g = _knee_raw(kp, widths, tag)
        if g is None:
            continue
        half = g[4]
        if halves and tag in halves and np.isfinite(halves[tag]) and halves[tag] > 0:
            half = float(halves[tag])
        out["knees"].append((g[0], g[1], g[2], g[3], float(half)))
        out["sides"].append(tag)
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


KNEE_FOLD_CAP = 0.028      # the most knee_fold moves one pixel's luma (soft limit, as MAX_CHANGE): creases are filled, the knee is not repainted
KNEE_FOLD_BOOST_FROM = 0.85   # the strength from which the wider shadow of a deep crease is rebuilt too (was 0.7)
KNEE_FOLD_BOOST_CAP = 0.04    # ... by at most this much luma


def _soft_cap(d, cap):
    c0 = 0.6 * cap
    a = np.abs(d)
    return np.where(a <= c0, d, np.sign(d) * (c0 + (cap - c0) * np.tanh((a - c0) / (cap - c0)))).astype(np.float32)


def knee_fold(Y, Cb, Cr, skin, visible, knees, k, gains=None):
    """Fill current knee creases, clipping the final change to visible owned skin.

    No reconstructed image is carried across frames. Geometry may be smoothed by person_geometry_t; the strength of each knee
    follows the render's look ahead (gains, knee_envelope) so that a knee lost for a picture fades instead of switching off.
    The change fades in from the edge of anything another person owns (owner_feather) and is soft limited (KNEE_FOLD_CAP).
    skin None: the leg's skin is read from the picture round each knee (local_zone_skin) instead of switching the knee off.
    """
    if k < 1e-4 or visible is None:
        return Y, Cb, Cr
    from . import engine as E
    owner = np.clip(visible, 0, 1).astype(np.float32)
    result = Y
    dall = None          # (every knee reads the picture as it came; where two knees overlap the larger change stays, not both)
    chroma = None
    h, w = Y.shape
    for ki, (centre, axis, half) in enumerate(knees):
        if not np.isfinite((*centre, *axis, half)).all() or half < 3:
            continue
        gain = 1.0
        if gains is not None and ki < len(gains) and gains[ki] is not None:
            gain = float(np.clip(gains[ki], 0.0, 1.0))
        if gain < 1e-3:
            continue
        radius = int(2.5 * half) + 8
        x0, x1 = max(0, int(centre[0]) - radius), min(w, int(centre[0]) + radius + 1)
        y0, y1 = max(0, int(centre[1]) - radius), min(h, int(centre[1]) + radius + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - centre[0], yy - centre[1]
        norm = max(float(np.hypot(*axis)), 1e-6)
        ax, ay = axis[0] / norm, axis[1] / norm
        along, across = dx * ax + dy * ay, -dx * ay + dy * ax
        zone = np.clip(1 - (along / (1.4 * half)) ** 2 - (across / half) ** 2, 0, 1)
        if chroma is None:
            chroma = tuple(c if c.shape == Y.shape else cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
                           for c in (Cb, Cr))
        roi = Y[y0:y1, x0:x1].astype(np.float32)
        cb, cr = (c[y0:y1, x0:x1] for c in chroma)
        own = owner[y0:y1, x0:x1]
        if skin is not None:
            sk = np.clip(skin[y0:y1, x0:x1], 0, 1).astype(np.float32)
        else:
            sk = local_zone_skin(roi, cb, cr, zone.astype(np.float32))
            if sk is None:
                continue
            sk = np.clip(sk, 0, 1).astype(np.float32)
        sk = sk * skin_likeness(roi, cb, cr)          # (never cloth: c017 v3 black trousers)
        allowed = cv2.erode((sk * own).astype(np.float32), np.ones((3, 3), np.uint8))
        # (faded in from every edge of the skin it may touch: another person's leg, the sock, a hem: a crease filled up to a sock's
        # edge left a bright line along it)
        gate = owner_feather(allowed, allowed, half)
        mask = (zone * gate).astype(np.float32)
        if mask.max() < 0.05:
            continue
        strength = float(np.clip(k, 0, 1))
        filled, _, _ = E._crease_lift(roi, cb, cr, mask, strength,
                                      max(1.5, 0.08 * half), axis=(ax, ay), band=0.6 * strength)
        # Reduce residual mid-scale indentations. Normalized skin-only samples
        # keep the sock and the background out of the local brightness estimate.
        weight = (sk * own).astype(np.float32)
        sigma = max(2.0, 0.28 * half)
        support = cv2.GaussianBlur(weight, (0, 0), sigma)
        local = cv2.GaussianBlur(filled * weight, (0, 0), sigma) / np.maximum(support, 1e-4)
        fine = cv2.GaussianBlur(filled, (0, 0), max(0.65, 0.035 * half))
        depression = np.clip(local - fine, 0.0, 0.075)
        filled = filled + strength * depression * mask * _ss(0.55, 0.85, support)
        # At high strength, reconstruct the wider shadow of a deep indentation (limited: the knee's relief stays).
        boost = float(np.clip((strength - KNEE_FOLD_BOOST_FROM) / max(1e-3, 1.0 - KNEE_FOLD_BOOST_FROM), 0, 1))
        if boost > 0:
            rad = max(2.0, 0.50 * half)
            den = cv2.GaussianBlur(weight, (0, 0), rad)
            surround = cv2.GaussianBlur(filled * weight, (0, 0), rad) / np.maximum(den, 1e-4)
            residual = surround - cv2.GaussianBlur(filled, (0, 0), max(0.7, 0.035 * half))
            coverage = _ss(0.0, 0.65, zone) * gate * _ss(0.55, 0.85, den)
            filled = filled + boost * coverage * np.clip(residual, 0, KNEE_FOLD_BOOST_CAP)
        d = _soft_cap((filled - roi) * gate, KNEE_FOLD_CAP) * gain
        full = np.zeros(Y.shape, np.float32)
        full[y0:y1, x0:x1] = d
        dall = _merge(dall, full)
    if dall is not None:
        result = (Y + dall).astype(np.float32)
    return result, Cb, Cr

# ---- knee clean (v4, 11/24 IWALY c016 feedback): the knee evened for sure, not by what a ring of leg round it says
# The tone (v2/v3) read the leg's colour from a ring round the knee and changed the knee only as far as that ring could be trusted:
# a raised leg (the knee near the shorts, the ring on cloth) switched it off for twelve frames (c016 P2 f77..f88), and a bruise
# that was not an outlier of the knee's own colour was left (purple patches all over the knee). v4 works like a frequency
# separation on the knee's own skin: what is finer than the grain (K4_FINE) and broader than the leg's shading (K4_MID) stays, the
# band between (blotches, bruises, mottling) is evened; the broad purple of a bruise is turned back to the leg's skin hue (across
# the skin-tone axis only: the warmth is kept, nothing turns grey). Every weight is smooth (no threshold, no edge), the strength
# follows only how much of the knee this person owns (another leg in front), and slider 70 is the full correction (KNEE_GAIN).
KNEE_V4 = False
KNEE_GAIN = 1.0 / 0.7       # the slider (0..1) times this, at most 1: 70 = full, a user's lower keyframes keep their share
K4_ALONG, K4_ACROSS = 1.6, 1.3   # the knee's zone: semi axes in half widths of the leg (along it, across it)
K4_EDGE = (0.55, 1.0)       # elliptical radius over which the zone falls from full to nothing
K4_FINE = 0.06              # of a half width (at least 1.2 px): finer is grain, pores, fine hair: never changed
K4_MID = 0.55               # of a half width: broader is the leg's shading and the knee's form: kept
K4_DARK_KEEP = 0.06         # of the band's darker part (a bruise, a blotch): this share stays
K4_LIGHT_KEEP = 0.55        # of the band's lighter part (the cap's highlight): this share stays
K4_CHROMA_KEEP = 0.05       # of the band's colour (red, purple patches): this share stays
K4_HUE_PULL = 0.85          # of the broad hue offset across the skin axis (the purple of a large bruise): taken back
K4_CAP = {"luma": 0.06, "chroma": 0.016}
K4_FEATHER = 0.55           # of a half width: the ramp inward from every edge of the skin it may touch (sock, shorts, another leg)
K4_SAT_FLOOR = 0.97


def _nblur(X, S, sig):
    den = cv2.GaussianBlur(S, (0, 0), sig)
    return cv2.GaussianBlur(X * S, (0, 0), sig) / np.maximum(den, 1e-4), den


# v5 (11/24 c016 v3 feedback: the knee looked smoothed out and its shape showed; the thigh above kept its texture): three bands
#   grain (finer than K5_GRAIN): never changed
#   texture (K5_GRAIN .. K5_TEX): softened by K5_TEX_SOFT, on the knee and, weaker, along the leg (thigh and shin): skin texture work
#   blotch (K5_TEX .. K4_MID): evened as v4 (bruises, mottling), full on the knee, K5_LEG_BLOTCH along the leg
# The leg zone runs K5_UP half widths up the thigh and K5_DOWN down the shin, so the knee is not a smooth island on a textured leg.
K5_GRAIN = 0.035            # of a half width (at least 1 px)
K5_TEX = 0.12               # of a half width (at least 1.6 px)
K5_TEX_SOFT = 0.40          # share of the texture band taken away (knee); along the leg it is scaled by the leg weight
K5_LEG = 0.55               # the leg's weight against the knee's (texture and blotch work along the thigh and shin)
K5_LEG_BLOTCH = 0.6         # share of the blotch evening along the leg (times K5_LEG)
K5_UP, K5_DOWN, K5_ACROSS = 3.6, 2.6, 1.9   # the leg zone in half widths (up the thigh, down the shin, across)
K5_SKIN_LUMA = (0.45, 0.70)  # the leg's own skin when the body area is missing: brightness over the leg's (cloth is far darker)


# (11/24 c017 v3: black trousers were taken for the leg's skin and the correction came and went on them from frame to frame)
# A pixel can be skin only by its own colour: warm and coloured enough and not dark (the trousers: Y 0.16, chroma 0.006, Cr 0.002;
# the legs' skin: Y 0.27..0.30, chroma 0.026, Cr 0.015, LogC4 code values). Soft, so nothing switches.
# (11/24 c017 f28/29: two legs of one dancer crossing; the shade the front leg casts along its edge on the other was evened as a
# blotch for two frames: a bright flash at the edge). A bruise's edge is soft; an edge where one leg passes another is sharp: the
# blotch and texture work are held back within K5_EDGE_REACH half widths of an edge steeper than K5_EDGE (luma per pixel over the level).
K5_EDGE = (0.010, 0.022)
K5_EDGE_REACH = 0.30
K5_SKIN_CHROMA = (0.009, 0.016)
K5_SKIN_CR = (0.005, 0.010)
K5_SKIN_Y = (0.17, 0.22)


def skin_likeness(Y, Cb, Cr):
    """0..1 per pixel: how far this colour can be skin (warm, coloured, not dark). Y, Cb, Cr on one grid."""
    c = np.hypot(Cb, Cr)
    w = _ss(K5_SKIN_CHROMA[0], K5_SKIN_CHROMA[1], c) * _ss(K5_SKIN_CR[0], K5_SKIN_CR[1], Cr) * _ss(K5_SKIN_Y[0], K5_SKIN_Y[1], Y)
    w = w * _ss(-0.3, 0.0, Cr + 0.3 * Cb)          # (red over blue, as _skin_like)
    return cv2.GaussianBlur(w.astype(np.float32), (0, 0), 1.0)


def _leg_skin(Yr, owner_r, zone, cbr=None, crr=None):
    """The leg's skin read from the picture: the person's own area where the colour is skin's (skin_likeness)."""
    if cbr is None or crr is None:
        return None
    S = skin_likeness(Yr, cbr, crr)
    return S if owner_r is None else S * owner_r


def knee_clean(Y, Cb, Cr, skin, owner, knees, k, cs: float = 1.0, gains=None):
    """The knee's blotches and bruises evened and the leg's skin texture softened on this person's own skin (KNEE_V4 / v5).
    skin, owner: 0..1 on Y's grid (None: read from the picture / all). knees: [((cx, cy), axis, half)] in Y's pixels.
    gains: [0..1] per knee (look ahead on ownership) or None. -> (Y, Cb, Cr)."""
    k = float(np.clip(float(k) * KNEE_GAIN, 0.0, 1.0))
    if k < 1e-4:
        return Y, Cb, Cr
    hY, wY = Y.shape
    hC, wC = Cb.shape
    sub = (hC, wC) != (hY, wY)
    dY = dCb = dCr = None
    for ki, (centre, axis, half) in enumerate(knees):
        half = float(half)
        if not (half > 2.0) or not np.isfinite([centre[0], centre[1], axis[0], axis[1], half]).all():
            continue
        gain = 1.0
        if gains is not None and ki < len(gains) and gains[ki] is not None:
            gain = float(np.clip(gains[ki], 0.0, 1.0))
        if gain < 1e-3:
            continue
        nrm = max(float(np.hypot(axis[0], axis[1])), 1e-6)
        ax, ay = axis[0] / nrm, axis[1] / nrm
        R = int(max(K5_UP, K5_DOWN, K5_ACROSS) * half) + 12
        x0, x1 = max(0, int(centre[0]) - R), min(wY, int(centre[0]) + R + 1)
        y0, y1 = max(0, int(centre[1]) - R), min(hY, int(centre[1]) + R + 1)
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy_ = xx - float(centre[0]), yy - float(centre[1])
        al = (dx * ax + dy_ * ay) / half          # + down the leg (to the ankle), - up the thigh
        ac = (-dx * ay + dy_ * ax) / half
        e = np.sqrt((al / K4_ALONG) ** 2 + (ac / K4_ACROSS) ** 2)
        wk = (1.0 - _ss(K4_EDGE[0], K4_EDGE[1], e)).astype(np.float32)
        wl = ((1.0 - _ss(K5_UP - 1.0, K5_UP, -al)) * (1.0 - _ss(K5_DOWN - 0.8, K5_DOWN, al))
              * (1.0 - _ss(K5_ACROSS - 0.6, K5_ACROSS, np.abs(ac)))).astype(np.float32)
        zone = np.maximum(wk, wl)
        Yr = Y[y0:y1, x0:x1].astype(np.float32)
        own = None if owner is None else np.clip(owner[y0:y1, x0:x1], 0, 1).astype(np.float32)
        cbr_ = (Cb if not sub else cv2.resize(Cb, (wY, hY), interpolation=cv2.INTER_LINEAR))[y0:y1, x0:x1].astype(np.float32)
        crr_ = (Cr if not sub else cv2.resize(Cr, (wY, hY), interpolation=cv2.INTER_LINEAR))[y0:y1, x0:x1].astype(np.float32)
        Sl = _leg_skin(Yr, own, zone, cbr_, crr_)
        if skin is not None:
            S = np.clip(skin[y0:y1, x0:x1], 0, 1).astype(np.float32)
            if own is not None:
                S = S * own
            # the body area and the leg's own skin together (the body area misses a knee for a stretch of frames, a blurred or moving
            # leg); never where the colour cannot be skin (trousers, socks), the same rule on every frame: nothing switches
            S = np.maximum(S, Sl) * skin_likeness(Yr, cbr_, crr_)
        else:
            S = Sl
        if S is None or float((S * zone).sum()) < 50.0:
            continue
        core = (S * wk) > 0.5
        if int(core.sum()) >= 30:
            yk = float(np.median(Yr[core]))
            S = S * _ss(0.35, 0.6, Yr / max(yk, 1e-4)).astype(np.float32)
        if float((S * zone).sum()) < 50.0:
            continue
        rad = max(2.0, K4_FEATHER * half)
        inside = cv2.distanceTransform((S > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        ramp = cv2.GaussianBlur(_ss(0.0, rad, inside).astype(np.float32), (0, 0), max(0.8, 0.3 * rad))
        # the sharp edges (a leg in front of another, a hem): the blotch work stays off them
        ys_ = cv2.GaussianBlur(Yr, (0, 0), max(1.0, 0.03 * half))
        gx_, gy_ = cv2.Sobel(ys_, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(ys_, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        lvl_ = max(float(np.median(Yr[S > 0.5])) if int((S > 0.5).sum()) > 30 else 0.3, 0.05)
        ed_ = _ss(K5_EDGE[0], K5_EDGE[1], np.hypot(gx_, gy_) * (0.3 / lvl_) * max(1.0, half / 45.0))
        rr_ = max(2, int(K5_EDGE_REACH * half))
        ed_ = cv2.dilate(ed_.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr_ + 1, 2 * rr_ + 1)))
        guard = 1.0 - np.clip(cv2.GaussianBlur(ed_, (0, 0), max(1.0, 0.5 * rr_)), 0, 1)
        base_w = ramp * S * (k * gain) * guard
        Wb = (np.maximum(wk, K5_LEG * K5_LEG_BLOTCH * wl) * base_w).astype(np.float32)
        Wt = (np.maximum(wk, K5_LEG * wl) * base_w).astype(np.float32)
        if float(max(Wb.max(), Wt.max())) < 0.01:
            continue
        s0, s1, s2 = max(1.0, K5_GRAIN * half), max(1.6, K5_TEX * half), max(4.0, K4_MID * half)
        y0_ = _nblur(Yr, S, s0)[0]
        y1_ = _nblur(Yr, S, s1)[0]
        y2_ = _nblur(Yr, S, s2)[0]
        tex = y0_ - y1_
        mid = y1_ - y2_
        d = -np.where(mid < 0, (1.0 - K4_DARK_KEEP) * mid, (1.0 - K4_LIGHT_KEEP) * mid).astype(np.float32)
        d = _soft_cap(d, K4_CAP["luma"]) * Wb - (K5_TEX_SOFT * tex) * Wt
        full = np.zeros((hY, wY), np.float32)
        full[y0:y1, x0:x1] = d
        dY = _merge(dY, full)
        if sub:
            sx, sy = wC / float(wY), hC / float(hY)
            c0, r0 = int(x0 * sx), int(y0 * sy)
            c1, r1 = min(wC, max(c0 + 2, int(np.ceil(x1 * sx)))), min(hC, max(r0 + 2, int(np.ceil(y1 * sy))))
            Sc = cv2.resize(S, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
            Wc = cv2.resize(Wb, (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA)
            hc = half * 0.5 * (sx + sy)
        else:
            c0, c1, r0, r1, Sc, Wc, hc = x0, x1, y0, y1, S, Wb, half
        Bc = [P[r0:r1, c0:c1].astype(np.float32) for P in (Cb, Cr)]
        t1, t2 = max(1.6, K5_TEX * hc), max(4.0, K4_MID * hc)
        lo1 = [_nblur(b_, Sc, t1)[0] for b_ in Bc]
        lo2 = [_nblur(b_, Sc, t2)[0] for b_ in Bc]
        wide = [_nblur(b_, Sc, max(6.0, 1.6 * hc))[0] for b_ in Bc]
        ln = np.maximum(np.hypot(wide[0], wide[1]), 1e-6)
        u0, u1 = wide[0] / ln, wide[1] / ln
        perp = -lo2[0] * u1 + lo2[1] * u0
        tgt = [lo1[i] - (1.0 - K4_CHROMA_KEEP) * (lo1[i] - lo2[i]) for i in range(2)]
        tgt = [tgt[0] + K4_HUE_PULL * perp * u1, tgt[1] - K4_HUE_PULL * perp * u0]
        lt = np.hypot(tgt[0], tgt[1])
        fl = K4_SAT_FLOOR * np.hypot(lo2[0], lo2[1])
        scl = np.where((lt < fl) & (lt > 1e-6), fl / np.maximum(lt, 1e-6), 1.0).astype(np.float32)
        tgt = [tgt[i] * scl for i in range(2)]
        for i in range(2):
            dc = _soft_cap(tgt[i] - lo1[i], K4_CAP["chroma"]) * Wc
            full = np.zeros((hC, wC), np.float32)
            full[r0:r1, c0:c1] = dc
            if i == 0:
                dCb = _merge(dCb, full)
            else:
                dCr = _merge(dCr, full)
    Yo = Y if dY is None else (Y + dY).astype(np.float32)
    Cbo = Cb if dCb is None else (Cb + dCb).astype(np.float32)
    Cro = Cr if dCr is None else (Cr + dCr).astype(np.float32)
    return Yo, Cbo, Cro


def visible_knee_tone(Y, Cb, Cr, skin, ownership, knees, strength, cs, gains):
    """Keep the visible portion of each knee and feather inwards from an occluder."""
    if KNEE_V4:
        return knee_clean(Y, Cb, Cr, skin, None if ownership is None else np.clip(ownership, 0, 1).astype(np.float32),
                          knees, strength, cs, gains)
    if skin is None or ownership is None:
        return Y, Cb, Cr
    owner = np.clip(ownership, 0, 1).astype(np.float32)
    gate = np.clip(skin * owner, 0, 1).astype(np.float32)
    halves = [float(h) for (_c, _a, h) in knees if np.isfinite(h) and h > 1.5]
    gate = owner_feather(gate, gate, min(halves) if halves else Y.shape[0] / 360.0 / OWNER_FEATHER)
    corrected = knee_tone(Y, Cb, Cr, gate, knees, strength, cs, gains=gains)
    output = []
    for original, filled in zip((Y, Cb, Cr), corrected):
        weight = gate if original.shape == gate.shape else cv2.resize(
            gate, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
        output.append(original + weight * (filled - original))
    return tuple(output)
