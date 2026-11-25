"""흰자 정리 (eye white cleanup, 2026-11-25; IU NES BASE_CL_03_RE / BASE_CL_29, the user: "점, 얼룩, 실핏줄을 완전 제거해서 깨끗한 흰자"):
spots, stains, veins and the bruise-like shade beside the iris are taken off the sclera; the white's own shading, its tone and the picture's
grain stay, the lashes and their shade under the upper lid stay, and nothing outside the white changes.

clean_eye(Y, Cb, Cr, eye6, k): one eye on planes of the same size (Y 0..1, Cb / Cr centred on 0), eye6 its six 68-point landmarks (corner,
two upper, corner, two lower) in px of the planes, k the strength 0..1.
clean_planes(Y, Cb, Cr, lm68, k): both eyes of a face on the retouch's planes (the chroma planes may be smaller: 4:2:2 / 4:2:0).

How (each step was the user's correction of the one before, on the BASE_CL_29 still):
1. The eye: the landmark polygon with the lids' curves, shrunk a little; the iris: the darkest place at its scale, its edge found on rays from
   its middle (where the light rises most steeply: an iris seen from the side is no circle); the white: light and of the white's colour (the
   lid's skin in a corner and the caruncle's pink are not), with the marks inside it filled in.
2. The white's clean level: a soft average of the white alone over a mark's scale, the dark marks weighed out step by step; bounded by a grey
   closing at a mark's size (a dip is a mark; the white's own shading, the lid's rim, is no dip and stays).
3. Lashes are held (dark, reaching the upper lid's edge, or long and thin); a round dot is cleaned even beside the lashes. Under the upper lid
   (FADE lash_band) only the dots are touched: the lashes' shade is the eye's own.
4. The correction is ADDED as a smooth field: the dips are filled up to the level (the grain stays: v1, a white painted anew, was "one colour,
   the mask's edge seen"); their red and yellow-red taken to the clean white's colour round them (v8: "red left where the blotches were").
5. Beside the iris (near_iris of the eye's width) the shade is taken up to the clean white right outside it, in luma and colour (v3..v8: "a
   bruise", "a grey crescent", "the mask seen in grey"); the thin dark ring at the iris' edge stays (the eye's outline).
"""
from __future__ import annotations

import numpy as np
import cv2


def _poly_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64)).astype(np.int32)], 1)
    return m > 0


def clean_eye(Y, Cb, Cr, eye6, k=1.0, shrink=0.96, iris_grow=1.03, rim=0.015, near_iris=(0.06, 0.16), lash_band=(0.05, 0.11), dbg=None):
    """One eye (see the module's notes). Returns (Y, Cb, Cr, info); the planes as they came where nothing is done."""
    h, w = Y.shape
    E6 = np.asarray(eye6, np.float64)
    c = E6.mean(0)
    ew = float(np.hypot(*(E6[3] - E6[0])))                    # the eye's width (corner to corner)
    if ew < 12:
        return Y, Cb, Cr, {"skip": "small"}
    # a polygon from the six points with the lids' curves (each lid through its two points and the corners: a smooth arc)
    def arc(p0, p1, p2, p3, n=24):
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    # Bezier control points that make the curve pass near the landmarks: the landmarks pushed out by a third
    up = arc(E6[0], E6[1] + (E6[1] - (E6[0] + E6[3]) / 2) * 0.33, E6[2] + (E6[2] - (E6[0] + E6[3]) / 2) * 0.33, E6[3])
    lo = arc(E6[3], E6[4] + (E6[4] - (E6[0] + E6[3]) / 2) * 0.33, E6[5] + (E6[5] - (E6[0] + E6[3]) / 2) * 0.33, E6[0])
    poly = np.vstack([up, lo])
    polys = c + (poly - c) * shrink
    x0, y0 = np.floor(poly.min(0) - 0.15 * ew).astype(int)
    x1, y1 = np.ceil(poly.max(0) + 0.15 * ew).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    sl = (slice(y0, y1), slice(x0, x1))
    y_, cb_, cr_ = Y[sl].astype(np.float32), Cb[sl].astype(np.float32), Cr[sl].astype(np.float32)
    shp = y_.shape
    E = _poly_mask(shp, polys - [x0, y0])
    if E.sum() < 30:
        return Y, Cb, Cr, {"skip": "eye"}
    # ---- the iris: the darkest place at the iris' scale, its radius from the dark area round it
    big = cv2.GaussianBlur(y_, (0, 0), 0.10 * ew)
    bb = np.where(E, big, 9.0)
    iy, ix = np.unravel_index(int(np.argmin(bb)), shp)
    ys_e = y_[E]
    lo_, hi_ = float(np.percentile(ys_e, 10)), float(np.percentile(ys_e, 90))
    dark = (y_ < lo_ + 0.45 * (hi_ - lo_)) & E
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    r_ir = 0.20 * ew
    lid_ = lab[iy, ix]
    if lid_ > 0:
        r_ir = float(np.clip(np.sqrt(st[lid_, 4] / np.pi), 0.14 * ew, 0.30 * ew))
        # the iris is cut by the lids: its width (not its area) tells its radius better
        r_ir = float(np.clip(max(r_ir, 0.5 * st[lid_, 2]), 0.14 * ew, 0.30 * ew))
        icx, icy = cen[lid_]
        ix, iy = int(round(0.5 * (ix + icx))), int(round(0.5 * (iy + icy)))
    yy, xx = np.mgrid[0:shp[0], 0:shp[1]].astype(np.float32)
    # the iris' edge itself: along rays from its middle, where the light rises most steeply (the white begins); the median over the rays that
    # reach the white (the lids cut the others)
    try:
        gy_ = cv2.GaussianBlur(y_, (0, 0), 1.0)
        rads = []
        angs = []
        for ang in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            rr = np.arange(0.6 * r_ir, 1.6 * r_ir, 0.5)
            px_ = ix + rr * np.cos(ang)
            py_ = iy + rr * np.sin(ang)
            ok_ = (px_ >= 0) & (px_ < shp[1] - 1) & (py_ >= 0) & (py_ < shp[0] - 1)
            if ok_.sum() < 6:
                continue
            px_, py_, rr = px_[ok_], py_[ok_], rr[ok_]
            inE = E[py_.astype(int), px_.astype(int)]
            if inE.mean() < 0.9:
                continue
            prof = gy_[py_.astype(int), px_.astype(int)]
            d_ = np.diff(prof)
            j_ = int(np.argmax(d_))
            if d_[j_] > 0.01:
                rads.append(rr[j_])
                angs.append(ang)
        if len(rads) >= 4:
            r_ir = float(np.clip(np.median(rads), 0.12 * ew, 0.32 * ew))
    except Exception:
        pass
    ird = np.hypot(xx - ix, yy - iy)
    # the iris' edge as found on each ray (an iris seen from the side is no circle): a radius for every direction, the rays that did not
    # reach the white taken between their neighbours, a little smoothed
    rmap = np.full(shp, r_ir, np.float32)
    try:
        if len(rads) >= 6:
            a_ = np.asarray(angs)
            r_ = np.asarray(rads, np.float64)
            grid = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            ext_a = np.concatenate([a_ - 2 * np.pi, a_, a_ + 2 * np.pi])
            ext_r = np.concatenate([r_, r_, r_])
            o_ = np.argsort(ext_a)
            rg = np.interp(grid, ext_a[o_], ext_r[o_])
            k_ = np.ones(5) / 5.0
            rg = np.convolve(np.concatenate([rg[-2:], rg, rg[:2]]), k_, mode="valid")
            th = np.mod(np.arctan2(yy - iy, xx - ix), 2 * np.pi)
            rmap = np.interp(th.ravel(), np.concatenate([grid, [2 * np.pi]]), np.concatenate([rg, rg[:1]])).reshape(shp).astype(np.float32)
            rmap = np.clip(rmap, 0.10 * ew, 0.36 * ew)
    except Exception:
        rmap = np.full(shp, r_ir, np.float32)
    iris = ird < rmap * iris_grow
    S = E & ~iris
    if S.sum() < 20:
        return Y, Cb, Cr, {"skip": "no white"}
    # what of it is the white itself: light and of the white's colour (the lid's skin in a corner, the pink of the caruncle are not), with the
    # marks inside it (holes: a stain, a vein, a mole are surrounded by white) filled in; a landmark polygon that reaches past the white does not
    # take the skin with it
    lt_ = S & (y_ > float(np.percentile(y_[S], 55)))
    cbw, crw = float(np.median(cb_[lt_])), float(np.median(cr_[lt_]))
    dch = np.hypot(cb_ - cbw, cr_ - crw)
    core = S & (dch < 0.035) & (y_ > lo_ + 0.40 * (hi_ - lo_))
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n4, lab4, st4, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    keep_ = np.zeros(shp, np.uint8)
    for i in range(1, n4):
        if st4[i, 4] >= 0.05 * float(S.sum()):
            keep_[lab4 == i] = 1
    if keep_.sum() < 20:
        return Y, Cb, Cr, {"skip": "no white seen"}
    ff_ = keep_.copy() * 255
    pad_ = np.zeros((shp[0] + 2, shp[1] + 2), np.uint8)
    cv2.floodFill(ff_, pad_, (0, 0), 128)
    filled = (ff_ != 128)                                       # the white with its holes filled
    kd_ = max(1, int(round(0.015 * ew)))
    filled = cv2.dilate(filled.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd_ + 1, 2 * kd_ + 1))) > 0
    S = S & filled
    # ---- the clean level of the white: a soft average of the white alone at a scale over a mark's, the dark marks weighed out step by step
    #      (each pass: what lies well under the level counts less), so the lid's shade and the white's gradient stay and a spot, a stain, a vein go
    sig = max(2.0, 0.11 * ew)
    Sf = S.astype(np.float32)
    v = y_
    q30 = float(np.percentile(v[S], 30))
    wgt = Sf * (0.25 + 0.75 * (v >= q30))
    T = v
    sd = 0.01
    for _ in range(5):
        den = cv2.GaussianBlur(wgt, (0, 0), sig)
        T = cv2.GaussianBlur(v * wgt, (0, 0), sig) / np.maximum(den, 1e-4)
        res = (v - T)[S]
        sd = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-4
        r_ = v - T
        wgt = Sf * np.where(r_ < -sd, np.exp(-((r_ + sd) / (1.2 * sd)) ** 2), 1.0).astype(np.float32)
    T = T.astype(np.float32)
    # a mark is a dip, lighter white on its sides: what the grey closing (a mark's size) does not fill is the white's own shading (the lid's
    # rim, the shade under the lashes) and is left; the clean level is the lower of the two
    rc = max(2, int(round(0.08 * ew)))
    Tc = cv2.morphologyEx(cv2.GaussianBlur(v, (0, 0), 0.8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1)))
    Tl = T.copy()                                               # the white's own level
    T = np.minimum(T, Tc)
    resid = (T - v)                                              # how far under the clean level
    # ---- the lashes over the white: dark and thin lines (a ridge: dark across, not along), much darker than the white (a vein is faint and red),
    #      in the upper part of the eye; a round mark (a mole, a stain) has no direction and is not held
    g_ = cv2.GaussianBlur(v, (0, 0), max(0.8, 0.008 * ew))
    dxx = cv2.Sobel(g_, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g_, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g_, cv2.CV_32F, 1, 1, ksize=3)
    tr_ = dxx + dyy
    dt_ = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4 * dxy * dxy, 0))
    l1, l2 = 0.5 * (tr_ + dt_), 0.5 * (tr_ - dt_)                # l1 >= l2; a dark line: l1 large, l2 about 0
    line = np.clip((l1 - 2.0 * np.abs(l2)) / (np.abs(l1) + 1e-6), 0, 1) * (l1 > 0)
    deep = np.clip((resid / np.maximum(T, 1e-3) - 0.12) / 0.10, 0, 1)    # darker than the white by 12 .. 22 % of its level
    redish = np.clip((cr_ - float(np.median(cr_[S]))) / 0.02, 0, 1)
    zone = np.zeros(shp, np.uint8)
    cv2.fillPoly(zone, [np.round(np.vstack([up, (up + [0, 0.35 * float(np.ptp(lo[:, 1]) + np.ptp(up[:, 1]))])[::-1]]) - [x0, y0]).astype(np.int32)], 1)
    # the dark marks of the upper part: each one a lash (long or thin, or touching others in a fringe) unless it is round and solid (a mole, a dot)
    dk = ((deep > 0.5) | ((line > 0.35) & (deep > 0.2))) & (redish < 0.6) & (zone > 0) & S
    # the iris' own soft edge (its shade on the white right beside it) is no lash
    dk &= (ird - rmap) > 0.025 * ew
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(dk.astype(np.uint8), connectivity=8)
    upm = np.zeros(shp, np.uint8)
    cv2.polylines(upm, [np.round(up - [x0, y0]).astype(np.int32)], False, 1, max(2, int(round(0.03 * ew))))
    lash = np.zeros(shp, bool)
    dots = np.zeros(shp, bool)
    rimband = cv2.dilate(upm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    for i in range(1, n3):
        comp = lab3 == i
        ys_, xs_ = np.nonzero(comp)
        touches = bool((rimband & comp).any())
        if len(xs_) >= 3:
            ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs_, ys_]).astype(np.float64)) + 0.25 * np.eye(2)))
            elong = float(np.sqrt(ev[1] / ev[0]))
            span = float(np.hypot(np.ptp(xs_), np.ptp(ys_)))
            solid = len(xs_) >= 0.6 * float(st3[i, 2] * st3[i, 3])
        else:
            elong, span, solid = 1.0, 1.0, True
        # a lash reaches the lid's edge, or is a long thin line of its own; a dot or a round mole (even beside the lashes) is cleaned
        if (touches and not (solid and elong < 1.8 and len(xs_) >= (0.025 * ew) ** 2)) or (elong > 2.5 and span > 0.05 * ew):
            lash |= comp
        else:
            dots |= comp
    hold = cv2.dilate(lash.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    mark = (resid > 2.5 * sd) & S
    # ---- the mask: the white, soft at the iris and at the lids
    m = (S | (iris & E)).astype(np.float32)          # (the iris side is faded on its own edge below, not by this blur)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.03 * ew))
    m = m * np.clip((ird - rmap * iris_grow) / max(1.0, 0.02 * ew) + 0.5, 0, 1)
    m = m * (1.0 - cv2.GaussianBlur(hold, (0, 0), 1.0))
    # the lids' rims (the lash line above, the waterline below) are the eye's own shape: the mask fades in from them
    dE = cv2.distanceTransform(E.astype(np.uint8), cv2.DIST_L2, 5)
    e0, e1 = rim * ew, 2.5 * rim * ew
    tt = np.clip((dE - e0) / max(e1 - e0, 1.0), 0, 1)
    m = m * (tt * tt * (3 - 2 * tt))
    # under the upper lid the lashes and their shade are the eye's own: only the dots there are taken (a mole beside the lashes)
    upl = np.zeros(shp, np.uint8)
    cv2.polylines(upl, [np.round(up - [x0, y0]).astype(np.int32)], False, 1, 1)
    dup = cv2.distanceTransform((1 - upl).astype(np.uint8), cv2.DIST_L2, 5)
    tu = np.clip((dup - lash_band[0] * ew) / max((lash_band[1] - lash_band[0]) * ew, 1.0), 0, 1)
    fade_up = tu * tu * (3 - 2 * tu)
    dotm = cv2.GaussianBlur(cv2.dilate(dots.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32), (0, 0), 1.0)
    m = m * np.maximum(fade_up, np.clip(dotm, 0, 1))
    m = np.clip(m, 0, 1) * float(k)
    # ---- only the marks are taken away: a correction made of smooth fields (the picture's grain, the white's own shading and tone stay as
    #      they are; the correction is nought where nothing is under the clean level, so the mask's edge does not show)
    ys_ = cv2.GaussianBlur(y_, (0, 0), 0.9)                       # without the grain
    grain = y_ - cv2.GaussianBlur(y_, (0, 0), 0.8)
    gsd = 1.4826 * float(np.median(np.abs(grain[S]))) + 1e-4
    # a dip (lighter white round it: the closing fills it), filled up to the white round it, never over the white's own level by much
    tgt = np.minimum(Tc, Tl + 1.5 * sd)
    # beside the iris the white is often shaded (the iris' rim and the lid round it): that shade, which reads as a bruise on a clean white, is
    # taken up to the white's own level too; the thin dark ring right at the iris' edge stays (the mask's fade there)
    de_ = np.clip(ird - rmap, 0, None)
    tn_ = np.clip((de_ - near_iris[0] * ew) / max((near_iris[1] - near_iris[0]) * ew, 1.0), 0, 1)
    wn_ = 1.0 - tn_ * tn_ * (3 - 2 * tn_)
    wfar = (Sf * (de_ > near_iris[1] * ew)).astype(np.float32)
    if wfar.sum() > 20:
        sgf = max(3.0, 0.18 * ew)
        Tfar = cv2.GaussianBlur(ys_ * wfar * (ys_ > np.percentile(ys_[wfar > 0], 30)), (0, 0), sgf) / np.maximum(
            cv2.GaussianBlur(wfar * (ys_ > np.percentile(ys_[wfar > 0], 30)), (0, 0), sgf), 1e-4)
    else:
        Tfar = Tl
    # the band's level: the clean white right outside it (within 0.30 of the eye's width of the iris, marks left out), carried in over the band
    wnb = (Sf * (de_ > near_iris[1] * ew) * (de_ < 0.30 * ew) * (resid < 1.0 * sd)).astype(np.float32)
    if wnb.sum() > 40:
        wnb = wnb * (ys_ >= np.percentile(ys_[wnb > 0], 40))      # its clean, light part (mottles left out)
    if wnb.sum() < 20:
        wnb = wfar
    sgf2 = max(2.0, 0.07 * ew)
    dnb = cv2.GaussianBlur(wnb, (0, 0), sgf2)
    reach_ = np.clip(dnb / 0.04, 0, 1)                          # where that white is near enough to carry its level and colour
    Tnb = cv2.GaussianBlur(ys_ * wnb, (0, 0), sgf2) / np.maximum(dnb, 1e-4)
    Tband = Tnb * reach_ + np.maximum(Tl, Tfar) * (1 - reach_)
    tgt = tgt * (1 - wn_) + np.maximum(tgt, Tband) * wn_
    dip = np.clip(tgt - ys_ - 0.5 * gsd * (1.0 - wn_), 0.0, None)
    corrY = dip
    yo = y_ + m * corrY
    # colour: the white's own local colour (a soft average of the white, the reddest parts weighed out step by step); only what is redder
    # (Cr over it) or more yellow-red (Cb under it) than that is taken back, at a mark's scale; the white's own tint and its colour noise stay
    def level(c_, sign):
        cs_ = cv2.GaussianBlur(c_, (0, 0), 1.0)
        wq = Sf.copy()
        lv = cs_
        sgc = max(2.0, 0.04 * ew)                               # a vein's and a dot's scale only: the white's own tint (even a broad warm patch) stays
        for _ in range(4):
            den = cv2.GaussianBlur(wq, (0, 0), sgc)
            lv = cv2.GaussianBlur(cs_ * wq, (0, 0), sgc) / np.maximum(den, 1e-4)
            d_ = sign * (cs_ - lv)
            sdc = 1.4826 * float(np.median(np.abs(d_[S]))) + 1e-4
            wq = Sf * np.where(d_ > sdc, np.exp(-((d_ - sdc) / (1.2 * sdc)) ** 2), 1.0).astype(np.float32)
        return cs_, lv, sdc
    crs, crl, sdr = level(cr_, +1.0)
    cbs, cbl, sdb = level(cb_, -1.0)
    corrR = np.clip(crs - crl - 0.1 * sdr, 0.0, None)
    corrB = np.clip(cbl - cbs - 0.3 * sdb, 0.0, None)
    cro = cr_ - m * corrR
    cbo = cb_ + m * corrB
    # beside the iris the lifted shade keeps the iris' cool grey: its colour is taken to the white's own colour farther out (in both
    # directions), so the lifted band is not a grey crescent on a warmer white
    # (the colour of the white right next to the band: within 0.30 of the eye's width of the iris, outside the band; the white's colour
    # changes over the eye, a far average brought a pink of the corner to the iris' side)
    if wnb.sum() > 20:
        crf = cv2.GaussianBlur(cv2.GaussianBlur(cr_, (0, 0), 1.5) * wnb, (0, 0), sgf2) / np.maximum(dnb, 1e-4)
        cbf = cv2.GaussianBlur(cv2.GaussianBlur(cb_, (0, 0), 1.5) * wnb, (0, 0), sgf2) / np.maximum(dnb, 1e-4)
        crs2 = cv2.GaussianBlur(cr_, (0, 0), 1.5)
        cbs2 = cv2.GaussianBlur(cb_, (0, 0), 1.5)
        wl_ = m * wn_ * np.clip(dip / max(2.0 * gsd, 1e-4), 0, 1) * reach_       # as far as the band was lifted
        cro = cro + wl_ * (crf - crs2)
        cbo = cbo + wl_ * (cbf - cbs2)
    # where a mark was filled (its luma lifted), its colour too is taken to the clean white's round it: the red a removed blotch leaves
    wcl = (Sf * (resid < 1.0 * sd)).astype(np.float32)
    if wcl.sum() > 20:
        sgm = max(2.0, 0.05 * ew)
        dcl = np.maximum(cv2.GaussianBlur(wcl, (0, 0), sgm), 1e-4)
        crc = cv2.GaussianBlur(cv2.GaussianBlur(cr_, (0, 0), 1.0) * wcl, (0, 0), sgm) / dcl
        cbc = cv2.GaussianBlur(cv2.GaussianBlur(cb_, (0, 0), 1.0) * wcl, (0, 0), sgm) / dcl
        wm_ = m * np.clip(dip / max(3.0 * gsd, 1e-4), 0, 1) * np.clip(dcl / 0.05, 0, 1)
        crn = cv2.GaussianBlur(cro, (0, 0), 1.0)
        cbn = cv2.GaussianBlur(cbo, (0, 0), 1.0)
        cro = cro - wm_ * np.clip(crn - crc, 0, None)              # redder than the clean white: down to it
        cbo = cbo + wm_ * np.clip(cbc - cbn, 0, None)              # more yellow-red (lower Cb) than it: up to it
    cbt, crt = float(np.median(cbl[S])), float(np.median(crl[S]))
    Yo, Cbo, Cro = Y.copy(), Cb.copy(), Cr.copy()
    Yo[sl], Cbo[sl], Cro[sl] = yo, cbo, cro
    info = dict(box=(x0, y0, x1, y1), iris=(ix + x0, iy + y0, r_ir), sd=sd, white=(cbt, crt), lashes=int(lash.sum()))
    if dbg is not None:
        dbg.update(E=E, iris=iris, S=S, T=T, mark=mark, lash=lash, m=m, sl=sl)
    return Yo, Cbo, Cro, info


MIN_EYE_PX = 24.0     # an eye narrower than this (px of the planes, corner to corner) is not touched: nothing to see, and too few pixels to tell


def clean_planes(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, lm: np.ndarray, k: float, min_px: float = MIN_EYE_PX):
    """Both eyes of one face on the retouch's planes. lm: (68, >=2) landmarks in px of Y. The chroma planes may be subsampled: the eye's piece
    of them is brought to the luma's size for the work and the change taken back to their own size."""
    if k <= 1e-4 or lm is None or len(lm) < 48:
        return Y, Cb, Cr
    L = np.asarray(lm, np.float64)[:, :2]
    hY, wY = Y.shape[:2]
    hC, wC = Cb.shape[:2]
    sx, sy = wC / float(wY), hC / float(hY)
    for idx in (range(36, 42), range(42, 48)):
        e = L[list(idx)]
        if not np.isfinite(e).all():
            continue
        ew = float(np.hypot(*(e[3] - e[0])))
        if ew < min_px:
            continue
        pad = 0.30 * ew + 4
        x0, y0 = int(max(0, np.floor(e[:, 0].min() - pad))), int(max(0, np.floor(e[:, 1].min() - pad)))
        x1, y1 = int(min(wY, np.ceil(e[:, 0].max() + pad))), int(min(hY, np.ceil(e[:, 1].max() + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        yc = Y[y0:y1, x0:x1].astype(np.float32)
        if (hC, wC) == (hY, wY):
            cbc, crc = Cb[y0:y1, x0:x1].astype(np.float32), Cr[y0:y1, x0:x1].astype(np.float32)
            cx0, cy0, cx1, cy1 = x0, y0, x1, y1
        else:
            cx0, cy0 = int(np.floor(x0 * sx)), int(np.floor(y0 * sy))
            cx1, cy1 = max(cx0 + 1, int(np.ceil(x1 * sx))), max(cy0 + 1, int(np.ceil(y1 * sy)))
            # the luma piece made to match the chroma piece exactly (its edges on the chroma's grid)
            x0, y0 = int(round(cx0 / sx)), int(round(cy0 / sy))
            x1, y1 = min(wY, int(round(cx1 / sx))), min(hY, int(round(cy1 / sy)))
            yc = Y[y0:y1, x0:x1].astype(np.float32)
            size = (x1 - x0, y1 - y0)
            cbc = cv2.resize(Cb[cy0:cy1, cx0:cx1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
            crc = cv2.resize(Cr[cy0:cy1, cx0:cx1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        try:
            yo, cbo, cro, info = clean_eye(yc, cbc, crc, e - [x0, y0], float(k))
        except Exception:
            import traceback
            traceback.print_exc()
            continue
        if "skip" in info:
            continue
        Y[y0:y1, x0:x1] = yo
        if (hC, wC) == (hY, wY):
            Cb[y0:y1, x0:x1] = cbo
            Cr[y0:y1, x0:x1] = cro
        else:
            csz = (cx1 - cx0, cy1 - cy0)
            Cb[cy0:cy1, cx0:cx1] += cv2.resize(cbo - cbc, csz, interpolation=cv2.INTER_AREA)
            Cr[cy0:cy1, cx0:cx1] += cv2.resize(cro - crc, csz, interpolation=cv2.INTER_AREA)
    return Y, Cb, Cr
