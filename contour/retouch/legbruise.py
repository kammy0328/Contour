"""다리 멍 정리 (2026-11-25, the user: "IWALY_c011에서 허벅지에 멍자국 제거 되어야 하는데"): a bruise on a thigh or a calf, filled with the leg's own
light and colour round it; the grain stays.

A bruise is a dark patch closed all round by lighter skin. The light of the leg's skin is filled the way water fills a landscape's basins, with
the skin's edge as the drain: a basin open to the edge (the gap between the thighs, the shade at the side of a leg, a hem's shadow) runs out
and is left, a closed one is a candidate. A candidate is a bruise when its colour is off the skin's round it towards blue or grey (not the red
of a hand or a scratch) or it is a wide dark patch, and it is not small and sharp (a mole, a thread of the clothing: kept) nor at the skin's
edge. The correction is added as smooth fields (the picture's own grain and pores stay): the light to the filled level or the skin round it,
the colour to the skin's round it.

Works on the retouch's planes as they are (the source's encoding); the measures are relative to the skin's own grain."""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

WORK_LIMB = 240.0           # the leg's width is brought to at most this many pixels for the work
REACH = 1.7                 # of a leg's half width: how far across the leg's line its area reaches
END = 0.15                  # of a segment's length: how far past its ends
BASIN = 1.5                 # of the grain: the depth from which a closed basin is a candidate
HUE_Z = 2.5                 # the hue's shift to blue or grey (in its own spread) from which a dark patch off the skin's hue is a candidate
MIN_SIZE = 0.008            # of the leg's width: the smallest bruise (side of its area)
MAX_SIZE = 0.35             # ... the largest
MAX_LEN = 0.6               # of the leg's width: nothing longer across
WIDE = 0.06                 # of the leg's width: a patch this wide (side of its area) and dark is a bruise whatever its colour
EDGE = 0.03                 # of the leg's width: a candidate this near the skin's edge (its median) is left
GROW = 0.035                # of the leg's width: the soft rim round a bruise taken with it
THING = 0.12              # a small patch this much darker than the skin round it (relative) is a thing (a thread, a hole), not a bruise
HALO = 0.3                # of the grain ...
HUE_REACH = 0.05          # of the leg's width: how far round a bruise its faint off-hue ends are looked for
HALO_DEPTH = 0.15         # ... and of the bruise's own depth: the basin round a bruise down to this depth is its soft rim, taken with it
DARK = 8.0                # of the grain: a patch this much darker than the skin round it ...
DARK_IN = 0.05            # ... this far inside the leg (of its width), compact, is a bruise whatever its hue
MAX_LIFT = 0.2              # the light is raised by at most this share of the level it is raised to


def _nblur(a: np.ndarray, w: np.ndarray, s: float) -> np.ndarray:
    return cv2.GaussianBlur(a * w, (0, 0), s) / np.maximum(cv2.GaussianBlur(w, (0, 0), s), 1e-4)


def _membrane(planes, known: np.ndarray, skin: np.ndarray, step: float, iters: int = 400):
    """Each plane over the skin where it is not known (known: 0..1), as the smoothest surface through the skin round it that is (a membrane:
    the leg's shading carried through the patch, its slope and its level at every side). Solved on a grid of about step px, the change
    brought back to full size."""
    h, w = known.shape[:2]
    q = max(1.0, float(step))
    ws, hs = max(4, int(round(w / q))), max(4, int(round(h / q)))
    K = cv2.resize(known.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA)
    Sk = cv2.resize(skin.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA)
    fix = K > 0.5
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    den = cv2.filter2D(Sk, -1, cross, borderType=cv2.BORDER_REPLICATE)
    out = []
    for p in planes:
        Kp = cv2.resize((p * known).astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) / np.maximum(K, 1e-4)
        num0 = cv2.GaussianBlur(Kp * fix, (0, 0), 3.0) / np.maximum(cv2.GaussianBlur(fix.astype(np.float32), (0, 0), 3.0), 1e-4)
        T = np.where(fix, Kp, num0).astype(np.float32)
        for _ in range(iters):
            Tn = cv2.filter2D(T * Sk, -1, cross, borderType=cv2.BORDER_REPLICATE) / np.maximum(den, 1e-4)
            T = np.where(fix, Kp, Tn)
        out.append(cv2.resize(T, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def fill_basins(f: np.ndarray, drain: np.ndarray, max_iter: Optional[int] = None) -> np.ndarray:
    """The grey level f with its closed basins filled to their rims (morphological reconstruction by erosion): drain (bool) is where the water
    runs out (it keeps f there)."""
    hi = float(f.max())
    mk = np.where(drain, f, hi).astype(np.float32)
    f = f.astype(np.float32)
    k3 = np.ones((3, 3), np.uint8)
    n = max_iter if max_iter is not None else 3 * (f.shape[0] + f.shape[1])
    for it in range(n):
        nm = np.maximum(cv2.erode(mk, k3), f)
        if it % 8 == 7:
            if float(np.abs(nm - mk).max()) < 1e-7:
                return nm
        mk = nm
    return mk


def leg_area(shape: Tuple[int, int], legs: Sequence[tuple]) -> np.ndarray:
    """The legs' area (0/1, float32) from their segments [(ax, ay, bx, by, half)] in px of the plane: a is the segment's outer end (the hip, the
    ankle; the area reaches a little past it, rounded), b its end towards the knee (cut square there: the knee's own shading is the knee tone's)."""
    m = np.zeros(shape, np.uint8)
    for ax, ay, bx, by, half in legs:
        a, b = np.array([ax, ay], np.float64), np.array([bx, by], np.float64)
        d = b - a
        L = float(np.hypot(*d))
        if L < 2 or not np.isfinite(half) or half <= 0:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        a2 = a - u * END * L
        r = REACH * half
        quad = np.array([a2 + n * r, b + n * r, b - n * r, a2 - n * r])
        cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
        cv2.circle(m, (int(round(a2[0])), int(round(a2[1]))), int(round(r)), 1, -1)
    return m.astype(np.float32)


def _maps(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, S: np.ndarray, limb: float) -> dict:
    """What the finding and the filling read of the leg's skin S (0..1) on planes of one size; limb: the leg's width in their pixels."""
    Sb = S > 0.5
    Sf = S.astype(np.float32)
    grain = Y - cv2.GaussianBlur(Y, (0, 0), 0.8)
    gsd = 1.4826 * float(np.median(np.abs(grain[Sb]))) + 1e-4
    ys = cv2.GaussianBlur(Y, (0, 0), max(1.5, 0.006 * limb))
    er = cv2.erode(Sb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dip = np.where(Sb, fill_basins(ys, ~er) - ys, 0).astype(np.float32)
    # the hue apart from the light, and its shift from the skin round (robust: the patches off it are let out of the round)
    cbs, crs = cv2.GaussianBlur(Cb, (0, 0), 1.5), cv2.GaussianBlur(Cr, (0, 0), 1.5)
    yl = np.maximum(np.abs(ys), 0.05)
    u, v = cbs / yl, crs / yl
    sig = max(4.0, 0.10 * limb)
    w = Sf.copy()
    for _ in range(4):
        ul, vl = _nblur(u, w, sig), _nblur(v, w, sig)
        b = (u - ul) - (v - vl)                         # + : bluer, less red
        sb = 1.4826 * float(np.median(np.abs(b[Sb]))) + 1e-6
        w = Sf * np.exp(-np.clip(np.abs(b) / sb - 1.5, 0, None) ** 2).astype(np.float32)
    zb = b / sb
    Tl = _nblur(ys, w, sig)
    dk = np.clip(Tl - ys, 0, None)
    dE = cv2.distanceTransform(Sb.astype(np.uint8), cv2.DIST_L2, 5)
    return dict(Sb=Sb, Sf=Sf, gsd=gsd, ys=ys, dip=dip, cbs=cbs, crs=crs, u=u, v=v, sig=sig, w=w, zb=zb, Tl=Tl, dk=dk, dE=dE, limb=float(limb))


def detect(M: dict, dbg: Optional[dict] = None):
    """The bruises on the maps M (_maps): (the bruises with their soft rims (0/1 float32), those of them off the skin's hue or wide (0/1), the
    candidates looked at)."""
    Sb, gsd, dip, zb, Tl, dk, dE, limb = M["Sb"], M["gsd"], M["dip"], M["zb"], M["Tl"], M["dk"], M["dE"], M["limb"]
    # candidates: a closed basin of the light; a patch off the skin's hue; a clearly dark compact patch inside the leg (open to a hem, so no basin)
    # (each kind its own patches: a dark patch running into a hem's shadow is not let to swallow a small bruise next to it)
    cand = ((dip > BASIN * gsd) | ((zb > HUE_Z) & (dk > gsd) & (dE > 0.02 * limb))).astype(np.uint8)
    candD = (((dk > DARK * gsd) & (dE > DARK_IN * limb)) & (cand == 0)).astype(np.uint8)
    nA, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    nD, labD, stD, _ = cv2.connectedComponentsWithStats(candD)
    lab = np.where(labD > 0, labD + nA - 1, lab).astype(np.int32)
    st = np.vstack([st, stD[1:]])
    n = nA + nD - 1
    keep = np.zeros(n, bool)
    offhue = np.zeros(n, bool)
    info = []
    deep = np.maximum(dip, dk)
    for i in range(1, n):
        a = int(st[i, 4])
        if a < (MIN_SIZE * limb) ** 2 or a > (MAX_SIZE * limb) ** 2 or max(st[i, 2], st[i, 3]) > MAX_LEN * limb:
            continue
        x0, y0, ww, hh = (int(q) for q in st[i, :4])
        r = lab[y0:y0 + hh, x0:x0 + ww] == i
        depth = float(np.percentile(deep[y0:y0 + hh, x0:x0 + ww][r], 90)) / gsd
        hue = float(np.percentile(zb[y0:y0 + hh, x0:x0 + ww][r], 75))
        de = float(np.median(dE[y0:y0 + hh, x0:x0 + ww][r])) / limb
        lvl = max(float(np.median(np.abs(Tl[y0:y0 + hh, x0:x0 + ww][r]))), 0.05)
        basin = float(np.percentile(dip[y0:y0 + hh, x0:x0 + ww][r], 90)) > BASIN * gsd
        ok = i < nA and ((hue > 2.0 and depth > 2.0) or (basin and a > (WIDE * limb) ** 2 and depth > 3.0))
        el = max(ww, hh) / max(1, min(ww, hh))
        if i >= nA and depth > DARK and de > DARK_IN and el < 3.0 and (0.03 * limb) ** 2 <= a <= (0.25 * limb) ** 2:
            # darkest inside, not at its side towards the skin's edge (a hem's or a fold's shadow is darkest at the edge it falls from)
            dkr = np.where(r, dk[y0:y0 + hh, x0:x0 + ww], -1.0)
            pk_ = np.unravel_index(int(np.argmax(dkr)), dkr.shape)
            ok = float(dE[y0 + pk_[0], x0 + pk_[1]]) >= 0.7 * de * limb and a >= 0.4 * ww * hh and el < 2.5
        ok = ok and de > EDGE and not (depth * gsd / lvl > THING and a < (0.1 * limb) ** 2)
        keep[i] = ok
        offhue[i] = ok and (hue > 2.0 or a > (WIDE * limb) ** 2)
        info.append(dict(x=x0 + ww / 2, y=y0 + hh / 2, area=a, depth=round(depth, 1), hue=round(hue, 1), edge=round(de, 3), ok=bool(ok)))
    if dbg is not None:
        dbg.update(info=info, gsd=gsd, zb=zb, dk=dk, dip=dip, dE=dE)
    m0 = keep[lab].astype(np.uint8)
    if not keep.any():
        return m0.astype(np.float32), m0.astype(np.float32), info
    # the whole basin a bruise lies in (its soft rim too), as far as it is closed: down to a share of the bruise's own depth
    m1 = np.zeros_like(m0)
    for i in np.nonzero(keep)[0]:
        x0, y0, ww, hh = (int(q) for q in st[i, :4])
        r = lab[y0:y0 + hh, x0:x0 + ww] == i
        pk = float(np.percentile(dip[y0:y0 + hh, x0:x0 + ww][r], 90))
        thr = max(HALO * gsd, HALO_DEPTH * pk)
        rh = max(2, int(HUE_REACH * limb))
        near = cv2.dilate((lab == i).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rh + 1, 2 * rh + 1)))
        nb, lb_ = cv2.connectedComponents(((dip > thr) | ((((zb > 0.5 * HUE_Z) & (dk > 0.5 * gsd)) | (dk > 0.4 * DARK * gsd)) & (near > 0) & (dE > 0.015 * limb)) | (lab == i)).astype(np.uint8))
        ids = np.unique(lb_[y0:y0 + hh, x0:x0 + ww][r])
        m1 |= np.isin(lb_, ids[ids > 0]).astype(np.uint8)
    return (m1 | m0).astype(np.float32), offhue[lab].astype(np.float32), info


def fill(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, M: dict, m0: np.ndarray, off: np.ndarray, k: float = 1.0, dbg: Optional[dict] = None):
    """The bruises m0 (0..1, with their rims; off: those to be lifted to the skin's level round them over a wider reach) filled with the leg's
    own light and colour round them on planes of one size (maps M of them, _maps). -> Y, Cb, Cr."""
    if k <= 1e-4 or float(m0.max()) < 0.5:
        return Y, Cb, Cr
    Sf, ys, dip, cbs, crs, u, v, w, Tl, dE, sig, limb = (M[q] for q in ("Sf", "ys", "dip", "cbs", "crs", "u", "v", "w", "Tl", "dE", "sig", "limb"))
    m0 = (m0 > 0.5).astype(np.float32)
    rr = max(3, int(GROW * limb))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    grow = cv2.dilate(m0, ker)
    w2 = Sf * w * (1.0 - cv2.dilate(grow, ker))         # the skin round the bruises, not they nor their soft rims
    # the skin round, carried in from the nearest side first (the leg's own shading through the patch, not a flat level)
    Tl2, ul, vl = _membrane((cv2.GaussianBlur(ys, (0, 0), 0.5 * rr), u, v), w2 * Sf, Sf, rr / 3.0)
    T = np.maximum(ys + dip * grow, Tl2)
    if float(off.max()) > 0.5:
        # a bruise off the skin's hue lying in a shade (along a hem, in a fold's valley): its light up to the skin's round it over a wider reach
        gh = cv2.GaussianBlur(cv2.dilate((off > 0.5).astype(np.float32), ker), (0, 0), rr * 0.5)
        T = T + gh * np.clip(Tl - T, 0, None)
    d_ = np.clip(T - ys, 0, MAX_LIFT * np.abs(T))
    inner = np.clip((dE - 0.01 * limb) / max(0.02 * limb, 1.0), 0, 1)      # nothing at the skin's edge (a hem, the leg's outline)
    m = (cv2.GaussianBlur(grow, (0, 0), rr * 0.5) * Sf * inner * float(np.clip(k, 0, 1))).astype(np.float32)
    lv = ys + d_
    Yo = Y + m * d_
    # the colour moved only where the patch is off it: darker than the skin round, or off its hue (clean skin under the patch's rim keeps its
    # own colour: a skin of another hue round, or a shade, is not laid on it as a grey patch, IWALY_c011 spot try f0..f5)
    zb = M.get("zb")
    wc = np.clip(d_ / (1.5 * M["gsd"]), 0, 1)
    if zb is not None:
        wc = np.maximum(wc, np.clip((np.abs(zb) - 1.0) / 2.0, 0, 1))
    wc = cv2.GaussianBlur(wc.astype(np.float32), (0, 0), max(1.0, 0.25 * rr))
    Cbo = Cb + m * wc * (ul * lv - cbs)
    Cro = Cr + m * wc * (vl * lv - crs)
    if dbg is not None:
        dbg.update(m=m, T=T, w2=w2, w=w)
    return Yo.astype(Y.dtype), Cbo.astype(Cb.dtype), Cro.astype(Cr.dtype)


def clean(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, S: np.ndarray, limb: float, k: float = 1.0, dbg: Optional[dict] = None):
    """The bruises on the skin S (0..1, the leg's skin) of planes of one size, found and filled on this picture alone; limb: the leg's width in
    their pixels. -> Y, Cb, Cr."""
    if k <= 1e-4 or limb < 16 or (S > 0.5).sum() < 64:
        return Y, Cb, Cr
    M = _maps(Y, Cb, Cr, S, limb)
    m0, off, info = detect(M, dbg)
    if float(m0.max()) < 0.5:
        return Y, Cb, Cr
    return fill(Y, Cb, Cr, M, m0, off, k, dbg)


def clean_planes(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, skin: np.ndarray, legs: Sequence[tuple], k: float,
                 dbg: Optional[dict] = None):
    """The legs of one person on the retouch's planes. skin: the person's body skin on the luma's grid (0..1); legs: [(ax, ay, bx, by, half)]
    in px of the luma. The chroma planes may be subsampled: the legs' piece is worked at the luma's size (made smaller for a wide leg) and
    the change taken back to each plane's own size."""
    if k <= 1e-4 or skin is None or not legs:
        return Y, Cb, Cr
    hY, wY = Y.shape[:2]
    hC, wC = Cb.shape[:2]
    area = leg_area((hY, wY), legs) * np.clip(skin, 0, 1)
    ys_, xs_ = np.where(area > 0.5)
    if len(ys_) < 64:
        return Y, Cb, Cr
    halves = [float(h) for *_, h in legs if np.isfinite(h) and h > 0]
    limb = 2.0 * float(np.median(halves))
    pad = int(0.1 * limb) + 4
    x0, y0 = max(0, int(xs_.min()) - pad), max(0, int(ys_.min()) - pad)
    x1, y1 = min(wY, int(xs_.max()) + pad + 1), min(hY, int(ys_.max()) + pad + 1)
    sx, sy = wC / float(wY), hC / float(hY)
    if (hC, wC) != (hY, wY):
        cx0, cy0 = int(np.floor(x0 * sx)), int(np.floor(y0 * sy))
        cx1, cy1 = max(cx0 + 1, int(np.ceil(x1 * sx))), max(cy0 + 1, int(np.ceil(y1 * sy)))
        x0, y0 = int(round(cx0 / sx)), int(round(cy0 / sy))
        x1, y1 = min(wY, int(round(cx1 / sx))), min(hY, int(round(cy1 / sy)))
    else:
        cx0, cy0, cx1, cy1 = x0, y0, x1, y1
    yc = Y[y0:y1, x0:x1].astype(np.float32)
    size = (x1 - x0, y1 - y0)
    cbc = Cb[cy0:cy1, cx0:cx1].astype(np.float32)
    crc = Cr[cy0:cy1, cx0:cx1].astype(np.float32)
    if cbc.shape[:2] != yc.shape[:2]:
        cbc = cv2.resize(cbc, size, interpolation=cv2.INTER_LINEAR)
        crc = cv2.resize(crc, size, interpolation=cv2.INTER_LINEAR)
    sc = min(1.0, WORK_LIMB / max(limb, 1.0))
    ar = area[y0:y1, x0:x1]
    if sc < 0.999:
        ws = (max(8, int(round(size[0] * sc))), max(8, int(round(size[1] * sc))))
        y_, cb_, cr_ = (cv2.resize(p, ws, interpolation=cv2.INTER_AREA) for p in (yc, cbc, crc))
        a_ = cv2.resize(ar, ws, interpolation=cv2.INTER_AREA)
        yo, cbo, cro = clean(y_, cb_, cr_, a_, limb * sc, k, dbg)
        dY = cv2.resize(yo - y_, size, interpolation=cv2.INTER_LINEAR)
        dB = cv2.resize(cbo - cb_, size, interpolation=cv2.INTER_LINEAR)
        dR = cv2.resize(cro - cr_, size, interpolation=cv2.INTER_LINEAR)
    else:
        yo, cbo, cro = clean(yc, cbc, crc, ar, limb, k, dbg)
        dY, dB, dR = yo - yc, cbo - cbc, cro - crc
    dY = dY * (ar > 0.01)
    Y[y0:y1, x0:x1] = (yc + dY).astype(Y.dtype)
    if (hC, wC) != (hY, wY):
        cs_ = (cx1 - cx0, cy1 - cy0)
        dB = cv2.resize(dB, cs_, interpolation=cv2.INTER_AREA)
        dR = cv2.resize(dR, cs_, interpolation=cv2.INTER_AREA)
    Cb[cy0:cy1, cx0:cx1] = (Cb[cy0:cy1, cx0:cx1].astype(np.float32) + dB).astype(Cb.dtype)
    Cr[cy0:cy1, cx0:cx1] = (Cr[cy0:cy1, cx0:cx1].astype(np.float32) + dR).astype(Cr.dtype)
    return Y, Cb, Cr


KNEE_FREE = 0.30          # of a thigh's or a calf's length: the part next to the knee left out (its shading is the knee's)


def _outer_first(nm: str, e: np.ndarray, half: float) -> tuple:
    """A thigh (hip -> knee) or a calf (knee -> ankle) as (outer x, y, knee-side x, y, half), the knee side cut back by KNEE_FREE."""
    a, b = (e[:2], e[2:4]) if nm.startswith("thigh") else (e[2:4], e[:2])
    k = b + (a - b) * KNEE_FREE
    return (float(a[0]), float(a[1]), float(k[0]), float(k[1]), float(half))


def legs_of(kp: np.ndarray, widths: Optional[np.ndarray]) -> List[tuple]:
    """The thighs and the calves of one person's keypoints (17, >= 2) and widths (segments, 2) as [(ax, ay, bx, by, half)] in source px."""
    from ..warp import body as B
    out = []
    try:
        ends = B.segment_endpoints(np.asarray(kp, np.float32))
    except Exception:
        return out
    for nm in ("thigh_l", "thigh_r", "calf_l", "calf_r"):
        e = np.asarray(ends[B.SEG_INDEX[nm]], np.float64)
        if not np.isfinite(e).all():
            continue
        L = float(np.hypot(e[2] - e[0], e[3] - e[1]))
        half = np.nan
        if widths is not None:
            w = np.asarray(widths[B.SEG_INDEX[nm]], np.float64)
            if np.isfinite(w).any():
                half = float(np.nanmean(w))
        if not np.isfinite(half) or half <= 0:
            half = (0.16 if nm.startswith("thigh") else 0.11) * L
        out.append(_outer_first(nm, e, half))
    return out


def legs_of_t(final_p: np.ndarray, widths_p: Optional[np.ndarray], t: int, win: int = 6) -> List[tuple]:
    """legs_of picture t of one person (final_p: (T, 17, >= 2), widths_p: (T, segments, 2) or None), each half width the median of the
    pictures t - win .. t + win: a leg's measure jumps from one picture to the next, the leg does not (and the work is laid out in it)."""
    from ..warp import body as B
    n = len(final_p)
    legs = legs_of(final_p[t], widths_p[t] if widths_p is not None else None)
    if widths_p is None or win <= 0 or not legs:
        return legs
    names = [nm for nm in ("thigh_l", "thigh_r", "calf_l", "calf_r")]
    halves = {}
    for nm in names:
        vals = []
        j = B.SEG_INDEX[nm]
        for s_ in range(max(0, t - win), min(n, t + win + 1)):
            w = np.asarray(widths_p[s_][j], np.float64)
            if np.isfinite(w).any():
                vals.append(float(np.nanmean(w)))
        if len(vals) >= 3:
            halves[nm] = float(np.median(vals))
    try:
        ends = B.segment_endpoints(np.asarray(final_p[t], np.float32))
    except Exception:
        return legs
    out = []
    for nm in names:
        e = np.asarray(ends[B.SEG_INDEX[nm]], np.float64)
        if not np.isfinite(e).all():
            continue
        h = halves.get(nm)
        if h is None:
            w = np.asarray(widths_p[t][B.SEG_INDEX[nm]], np.float64)
            h = float(np.nanmean(w)) if np.isfinite(w).any() else (0.16 if nm.startswith("thigh") else 0.11) * float(np.hypot(e[2] - e[0], e[3] - e[1]))
        if np.isfinite(h) and h > 0:
            out.append(_outer_first(nm, e, h))
    return out
