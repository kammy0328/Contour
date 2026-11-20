"""Body blemishes (1.23): small spots on the body's skin, found on the original's own pixels piece by piece, held
through time along the picture's motion, filled from the skin around them.

The face's blemish step looks at one face on one picture. A body is larger than a face and its skin is spread over
the frame, so here:

  where     the body's area (bodyskin) on the best picture there is: the original where it has clearly more pixels
            than the analysis picture, read in overlapping pieces (tiles) and only where there is skin enough; the
            sizes follow the body's size (BodyInfo.ref), never the frame's
  what      the spots of detect.spots (dark, red, raised), then what makes a mark part of the person or of what
            they wear is left alone: things on the skin and other colours (detect.protect: ink, straps, lace, glints),
            deep dark marks (moles), marks that come many together (freckles, a tattoo's dots, a print), long marks
            (scars, hairs, lines), and anything at the skin's edge. In doubt nothing is removed.
  when      a spot is on the skin for as long as the skin is in view and grain is new on every frame: each frame's
            spots are carried onto the frames round it along the picture's own motion and count when they are found
            there too; a spot one frame misses is put back from its neighbours. Nothing is carried over a cut, onto
            a cover or onto another person, and whatever is filled lies inside the present frame's skin.
  by hand   marks of the operator: 'remove' (this spot goes, whatever the rules say) and 'exclude' (nothing is
            removed here), fixed to the body part they were made on.

A frame's spots are a short list (place, size, score, a small patch each), not maps of the frame's size: a 4K frame
with several people adds a few kilobytes to the caches.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import detect as DT

# ------------------------------------------------------------------------------------------------- finding
DET_REF = 240.0             # the body's size (face width) is at most this many pixels on the picture spots are looked for on
SMALL = 0.012               # of the body's size: the smallest spot size looked for, whatever the largest is (3 px at least)
THREADS = 4                 # pieces looked at side by side at most
DET_GAIN = 1.25             # the original is read when it has this many times the analysis picture's pixels
TILE = 768                  # a piece's side (pixels of the picture looked at)
TILE_LAP = 128              # pieces overlap by this much: a spot at one piece's edge is whole in the next
TILE_SKIN = 0.02            # a piece with less skin than this share is not looked at
CAND_MIN = 0.25             # a mark's score from which it is a candidate
TAKE_MIN = 0.35             # and its peak from which it may be removed at all
MOLE = (0.20, 0.30)         # a dark mark this much darker than the skin round it (relative) is a mole: kept
CROWD_N = 5                 # this many dark marks ...
CROWD_R = 5.0               # ... within this many largest-spot sizes of each other are a pattern (freckles, ink dots): kept
LONG = 2.4                  # a mark this many times longer than wide, and longer than ...
LONG_MIN = 0.7              # ... this share of the largest spot size, is a line (a scar, a hair, a drawn line): kept
PROTECT = 0.4               # share of a mark under detect.protect from which it is a thing, not a blemish
# a blemish is a small mark standing alone on calm skin. Round each mark (out to SUR_R of its own size, at least
# SUR_MIN of the largest spot):
SUR_W = 0.8                 # the round is this many largest-spot sizes wide (4 px at least) ...
SUR_GAP = 0.25              # ... and starts this share of the mark's patch outside it (a spot's own soft rim is the spot)
SUR_SKIN = 0.92             # this share is the body's skin (less: the mark is at the skin's edge, on a hem, between fingers)
SUR_CALM = 0.10             # nothing there stands this far off the skin's own level (relative): no edge, no bead, no fold
SUR_TAIL = (5.0, 0.03)      # nor far off what the round's own texture is: its strongest 3 % at most this many times its
                            # middle (an even texture: about 3; a line or an edge through calm skin: 10 and more), counted
                            # from this level up
BIG = 3.0                   # a mark whose score patch is this many times the largest spot size across is not a spot of
                            # that size (the patch is the spot and the finder's own spread, about one size more)
SUR_HUE = 0.022             # and nothing is this far off the skin's colour (Cb, Cr): no gold, no cloth, no ink
SUR_THING = 0.06            # and at most this share is a thing (detect.protect)
PATCH_PAD = 2
SPOTNET = False             # the learned spot detector of large faces on the body too (an experiment: §3)
DEBUG = None                # a list: what was measured round each mark is appended (diagnosis only)

HOLD_NONE, HOLD_THING, HOLD_MOLE, HOLD_CROWD, HOLD_LONG, HOLD_USER, HOLD_WEAK, HOLD_EDGE = 0, 1, 2, 3, 4, 5, 6, 7
HOLD_ALONE = 8              # seen on this frame only: no frame beside it shows the same mark at that place (fuse)
HOLD_TEXT = {HOLD_NONE: "제거", HOLD_THING: "옷·장신구·문신 등", HOLD_MOLE: "점", HOLD_CROWD: "주근깨·무늬",
             HOLD_LONG: "흉터·선", HOLD_USER: "제외 표시", HOLD_WEAK: "불확실", HOLD_EDGE: "피부 가장자리·주름",
             HOLD_ALONE: "한 프레임에만 보임"}
KIND_DARK, KIND_RED, KIND_BUMP, KIND_USER = 0, 1, 2, 3

CAND = np.dtype([("x", "f4"), ("y", "f4"),              # centre, source px
                 ("d", "f4"),                            # the mark's size across, source px
                 ("s", "f4"), ("r", "f4"),               # peak scores: dark / raised, red
                 ("kind", "u1"), ("hold", "u1"),
                 ("lum", "f4"),                          # the skin's level round it (0..1)
                 ("px0", "f4"), ("py0", "f4"),           # its patch: top left in source px,
                 ("off", "i4"), ("w", "i2"), ("h", "i2")])   # place in the packed patches, size in picture px


def spot_size(body_size: float, ref: float) -> float:
    """The largest spot's size across, in pixels of a picture on which the body's size (face width) is ref."""
    return max(2.0, (0.012 + 0.05 * float(body_size)) * float(ref))


def empty(scale: float = 1.0) -> dict:
    z = np.zeros(0, np.uint8)
    return dict(maps={"S": z, "R": z.copy()}, thumb=None, cands=np.zeros(0, CAND), scale=float(scale), tiles=0,
                src=False)


def _tiles(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[tuple, tuple]]:
    """Overlapping pieces covering [x0, x1) x [y0, y1): [(piece, core)], both (xa, ya, xb, yb). A piece's core is the
    part whose spots are taken from it: the cores meet in the middle of each overlap and do not overlap, so every
    spot is taken once, from the piece it lies well inside."""
    def cuts(a, b):
        n = b - a
        if n <= TILE:
            return [(a, b, a, b)]
        k = int(np.ceil((n - TILE_LAP) / float(TILE - TILE_LAP)))
        step = (n - TILE) / float(max(1, k - 1))
        seg = [(int(round(a + i * step)), int(round(a + i * step)) + TILE) for i in range(k)]
        out = []
        for i, (p0, p1) in enumerate(seg):
            c0 = a if i == 0 else (seg[i - 1][1] + p0) // 2
            c1 = b if i == k - 1 else (p1 + seg[i + 1][0]) // 2
            out.append((p0, p1, c0, c1))
        return out
    return [((xa, ya, xb, yb), (cxa, cya, cxb, cyb)) for ya, yb, cya, cyb in cuts(y0, y1) for xa, xb, cxa, cxb in cuts(x0, x1)]


def sizes_for(ref: float, size: float) -> tuple:
    """The spot sizes looked for, as shares of the largest: the smallest spot is as small with the slider at its
    largest as at its smallest (a third size between them when they are far apart)."""
    small = max(3.0, SMALL * ref)
    if small >= 0.8 * size:
        return (1.0,)
    if size <= 2.6 * small:
        return (small / size, 1.0)
    return (small / size, float(np.sqrt(small * size)) / size, 1.0)


def find(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, region: np.ndarray, ref: float, size: float,
         origin=(0.0, 0.0), scale: float = 1.0, core=None, net=None) -> Tuple[np.ndarray, list, list]:
    """The spot candidates of one piece of picture. Y 0..1; Cb, Cr centred on 0 at Y's size; region: the body's
    skin there (0..1); ref: the body's size and size: the largest spot, in this picture's pixels. origin: the
    piece's top left in source px, scale: picture px per source px. core (xa, ya, xb, yb in this picture's px):
    only spots whose middle lies inside are returned. -> (candidates, their S patches, their R patches)."""
    maps = DT.spots(Y, Cr, region, size, sizes=sizes_for(ref, size))
    if net is not None:
        try:
            nm = net(region, size)
            if nm:
                maps.update({k: nm[k] for k in ("dark", "red", "bump") if k in nm})
        except Exception:
            pass
    S = np.maximum(maps["dark"], maps["bump"])
    R = maps["red"]
    A = np.maximum(S, R)
    on = (A > CAND_MIN).astype(np.uint8)
    if not on.any():
        return np.zeros(0, CAND), [], []
    prot = DT.protect(Y, Cb, Cr, region, ref)
    n, lab, st, cen = cv2.connectedComponentsWithStats(on, connectivity=8)
    out = np.zeros(n - 1, CAND)
    ps, pr = [], []
    H, W = Y.shape
    keep = np.zeros(n - 1, bool)
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in st[i]]
        cx, cy = float(cen[i][0]), float(cen[i][1])
        if core is not None and not (core[0] <= cx < core[2] and core[1] <= cy < core[3]):
            continue
        xa, ya = max(0, x - PATCH_PAD), max(0, y - PATCH_PAD)
        xb, yb = min(W, x + w + PATCH_PAD), min(H, y + h + PATCH_PAD)
        blob = lab[ya:yb, xa:xb] == i
        s_ = S[ya:yb, xa:xb]
        r_ = R[ya:yb, xa:xb]
        # the patch: this mark's own scores (a neighbour's mark inside the box is not taken along)
        near = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sp = np.where(near, s_, 0.0)
        rp = np.where(near, r_, 0.0)
        c = out[i - 1]
        c["x"], c["y"] = origin[0] + (cx + 0.5) / scale, origin[1] + (cy + 0.5) / scale
        c["d"] = float(np.sqrt(4.0 * area / np.pi)) / scale
        c["s"], c["r"] = float(sp.max()), float(rp.max())
        dark = float(maps["dark"][ya:yb, xa:xb][blob].max())
        bump = float(maps["bump"][ya:yb, xa:xb][blob].max())
        c["kind"] = KIND_RED if c["r"] >= c["s"] else (KIND_DARK if dark >= bump else KIND_BUMP)
        c["px0"], c["py0"] = origin[0] + xa / scale, origin[1] + ya / scale
        c["w"], c["h"] = xb - xa, yb - ya
        # the skin round it: a ring of the mark's own width, on skin, off the mark
        rr = max(2, int(round(0.6 * max(w, h))))
        X0, Y0, X1, Y1 = max(0, x - rr), max(0, y - rr), min(W, x + w + rr), min(H, y + h + rr)
        big = lab[Y0:Y1, X0:X1] == i
        ring = (cv2.dilate(big.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) > 0) \
            & ~(cv2.dilate(big.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (region[Y0:Y1, X0:X1] > 0.5) \
            & (A[Y0:Y1, X0:X1] < CAND_MIN)
        yy = Y[Y0:Y1, X0:X1]
        base = float(np.median(yy[ring])) if int(ring.sum()) >= 6 else float(np.median(yy[region[Y0:Y1, X0:X1] > 0.5])) \
            if (region[Y0:Y1, X0:X1] > 0.5).any() else float(yy.mean())
        c["lum"] = base
        hold = HOLD_NONE
        if float(prot[ya:yb, xa:xb][blob].mean()) > PROTECT:
            hold = HOLD_THING
        else:
            # what lies round the mark
            gap = 2 + int(round(SUR_GAP * max(w, h)))
            rs = gap + max(4, int(round(SUR_W * size)))
            U0, V0, U1, V1 = max(0, x - rs), max(0, y - rs), min(W, x + w + rs), min(H, y + h + rs)
            mine = lab[V0:V1, U0:U1] == i
            sur = (cv2.dilate(mine.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1))) > 0) \
                & ~(cv2.dilate(mine.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))) > 0)
            n_sur = int(sur.sum())
            # (what of the round falls outside the picture is not skin either)
            full = np.pi * (rs + 0.5 * max(w, h)) ** 2 - np.pi * (0.5 * max(w, h) + gap) ** 2
            reg_ = region[V0:V1, U0:U1]
            skin_share = float((reg_[sur] > 0.6).sum()) / max(float(max(n_sur, full * 0.85)), 1.0)
            thing_share = float((prot[V0:V1, U0:U1][sur] > 0.5).mean()) if n_sur else 1.0
            dev95 = hue95 = 0.0
            if skin_share < SUR_SKIN:
                hold = HOLD_EDGE
            elif thing_share > SUR_THING:
                hold = HOLD_THING
            else:
                ys_ = Y[V0:V1, U0:U1]
                loc = cv2.GaussianBlur(ys_, (0, 0), max(1.5, 0.5 * rs))
                dev = np.abs(ys_ - loc)[sur] / max(base, 0.05)
                cbs, crs = Cb[V0:V1, U0:U1][sur], Cr[V0:V1, U0:U1][sur]
                hue = np.hypot(cbs - np.median(cbs), crs - np.median(crs))
                dev95, hue95 = float(np.percentile(dev, 95)), float(np.percentile(hue, 95))
                d97, d50 = float(np.percentile(dev, 97)), float(np.median(dev))
                if dev95 > SUR_CALM or hue95 > SUR_HUE or (d97 > SUR_TAIL[1] and d97 > SUR_TAIL[0] * max(d50, 1e-4)):
                    hold = HOLD_EDGE
            if DEBUG is not None:
                DEBUG.append(dict(x=float(c["x"]), y=float(c["y"]), d=float(c["d"]), s=float(c["s"]), r=float(c["r"]),
                                  skin=round(skin_share, 3), thing=round(thing_share, 3), dev=round(dev95, 4),
                                  hue=round(hue95, 4), rs=rs, gap=gap, hold=hold))
        if hold == HOLD_NONE and float(np.sqrt(4.0 * area / np.pi)) > BIG * size:
            hold = HOLD_LONG                    # larger than the largest spot asked for
        if hold == HOLD_NONE:
            # (the whole box: a mark far darker than a blemish scores only on its rim, its dark middle is inside)
            deep = (base - float(cv2.GaussianBlur(Y[ya:yb, xa:xb], (0, 0), 0.8).min())) / max(base, 0.05)
            if deep > 0.5 * (MOLE[0] + MOLE[1]):
                hold = HOLD_MOLE
        if hold == HOLD_NONE and area >= 6:
            # long marks: the spread of the mark's own pixels along and across its direction
            py_, px_ = np.nonzero(blob)
            if len(px_) >= 6:
                cov = np.cov(np.stack([px_, py_]).astype(np.float64))
                ev = np.sort(np.linalg.eigvalsh(cov))
                la, lb = 4.0 * np.sqrt(max(ev[1], 1e-6)), 4.0 * np.sqrt(max(ev[0], 1e-6)) + 1.0
                if la > LONG * lb and la > LONG_MIN * size:
                    hold = HOLD_LONG
        if hold == HOLD_NONE and max(c["s"], c["r"]) < TAKE_MIN:
            hold = HOLD_WEAK
        c["hold"] = hold
        keep[i - 1] = True
        ps.append((np.clip(sp, 0, 1) * 255 + 0.5).astype(np.uint8))
        pr.append((np.clip(rp, 0, 1) * 255 + 0.5).astype(np.uint8))
    return out[keep], ps, pr


def crowd(c: np.ndarray, size_src: float) -> np.ndarray:
    """Dark marks that come many together are part of the skin or of a drawing on it (freckles, the dots of a
    tattoo, a print showing through): none of them is removed. c: all of one person's candidates on a frame."""
    c = c.copy()
    dark = np.nonzero((c["kind"] == KIND_DARK) & np.isin(c["hold"], (HOLD_NONE, HOLD_MOLE, HOLD_WEAK)))[0]
    if len(dark) < CROWD_N:
        return c
    P = np.stack([c["x"][dark], c["y"][dark]], 1).astype(np.float64)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    cnt = (d2 <= (CROWD_R * size_src) ** 2).sum(1)              # the mark itself counts
    for i in dark[cnt >= CROWD_N]:
        if c["hold"][i] in (HOLD_NONE, HOLD_WEAK):
            c["hold"][i] = HOLD_CROWD
    return c


def pack(cands: np.ndarray, ps: list, pr: list, scale: float, tiles: int = 0, src: bool = False) -> dict:
    """One person's candidates of a frame as the caches keep them: the list and its patches end to end."""
    off = 0
    for i in range(len(cands)):
        cands["off"][i] = off
        off += int(cands["w"][i]) * int(cands["h"][i])
    S = np.concatenate([p.ravel() for p in ps]) if ps else np.zeros(0, np.uint8)
    R = np.concatenate([p.ravel() for p in pr]) if pr else np.zeros(0, np.uint8)
    return dict(maps={"S": S, "R": R}, thumb=None, cands=cands, scale=float(scale), tiles=int(tiles), src=bool(src))


def patch(det: dict, i: int) -> Tuple[np.ndarray, np.ndarray]:
    c = det["cands"][i]
    n = int(c["w"]) * int(c["h"])
    o = int(c["off"])
    shp = (int(c["h"]), int(c["w"]))
    return det["maps"]["S"][o:o + n].reshape(shp), det["maps"]["R"][o:o + n].reshape(shp)


def detect(pieces, region_at, model, ref_src: float, body_size: float, scale: float, box_src, net_for=None) -> dict:
    """One person's body spots on one frame.

    pieces(x0, y0, x1, y1) -> the picture of that box of the source (source px, even) at `scale` picture px per
    source px: RGB uint8, or (Y, Cb, Cr) float planes of one size (Y 0..1, Cb and Cr centred on 0);
    region_at(x0, y0, x1, y1, shape) -> the body's skin on that picture (float32 0..1);
    model: the body skin's colour model (the region's edge is laid at the picture's own pixels, bodyskin.refine);
    ref_src: the body's size, body_size: the 최대 잡티 크기 slider (0..1), box_src: the body's box in source px."""
    from . import bodyskin as BS
    from .engine import rgb_to_ycc
    ref = ref_src * scale
    size = spot_size(body_size, ref)
    x0, y0, x1, y1 = [int(v) for v in box_src]
    bx = (int(np.floor(x0 * scale)), int(np.floor(y0 * scale)), int(np.ceil(x1 * scale)), int(np.ceil(y1 * scale)))
    if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 8 or ref < 6:
        return empty(scale)
    def one(tc):
        tile, core = tc
        # the piece in source px (whole even pixels), then what was really got
        sa = (int(tile[0] / scale) // 2 * 2, int(tile[1] / scale) // 2 * 2,
              -(-int(np.ceil(tile[2] / scale)) // 2) * 2, -(-int(np.ceil(tile[3] / scale)) // 2) * 2)
        img = pieces(*sa)
        if img is None:
            return None
        planes = isinstance(img, tuple)
        hh, ww = (img[0] if planes else img).shape[:2]
        if hh < 8 or ww < 8:
            return None
        reg = region_at(sa[0], sa[1], sa[2], sa[3], (hh, ww))
        if reg is None or float((reg > 0.5).mean()) < TILE_SKIN:
            return None
        Y, Cb, Cr = img if planes else rgb_to_ycc(img.astype(np.float32) / 255.0)
        reg = BS.refine(reg, model, Y, Cb, Cr)
        if float((reg > 0.5).mean()) < 0.5 * TILE_SKIN:
            return None
        fx, fy = ww / float(sa[2] - sa[0]), hh / float(sa[3] - sa[1])
        core_px = ((core[0] / scale - sa[0]) * fx, (core[1] / scale - sa[1]) * fy,
                   (core[2] / scale - sa[0]) * fx, (core[3] / scale - sa[1]) * fy)
        net = net_for(img, ref) if (net_for is not None and not planes) else None
        return find(Y, Cb, Cr, reg, ref, size, origin=(sa[0], sa[1]), scale=fx, core=core_px, net=net)

    tiles = _tiles(*bx)
    if THREADS > 1 and len(tiles) > 1 and net_for is None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(THREADS, len(tiles))) as ex:
            got = list(ex.map(one, tiles))          # in the pieces' order whichever finishes first
    else:
        got = [one(tile) for tile in tiles]
    all_c, all_s, all_r = [], [], []
    used = 0
    for g in got:
        if g is None:
            continue
        used += 1
        c, ps, pr = g
        if len(c):
            all_c.append(c)
            all_s += ps
            all_r += pr
    if not all_c:
        d = empty(scale)
        d["tiles"] = used
        return d
    cands = crowd(np.concatenate(all_c), size / scale)
    return pack(cands, all_s, all_r, scale, used)


# ------------------------------------------------------------------------------------------------- through time
ALONE_HELD = True           # a spot no neighbouring frame confirms is held (False: it stands on its own frame's word)
ANCHOR_TOL = 0.06           # of the body's size: how far a neighbour's spot may sit from the place the joints carry this
                            # frame's spot to and be the same (the joints place a point coarser than the picture's motion)
ANCHOR_SKIN = 0.25          # the neighbour's own skin area at that place from which the place is to be seen there
TOL = 0.012                 # of the body's size: how far a neighbour's spot may sit from this frame's and be the same
CONFIRM = (0.10, 0.40)      # the neighbours' score from which a spot found here counts .. counts in full
SAME_MIN = 0.5              # how alike the pictures must be at a spot for a neighbour to have a say there


def _mapped(mo: dict, cs: np.ndarray, sx: float, sy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Where the candidates cs of frame s lie on frame t: (positions (n, 2) in source px, NaN where the motion does
    not show them, how alike the two pictures are there 0..1). mo: bodyskin.motion of t onto s (for each point of
    t's piece, its place on s, in analysis px); sx, sy: analysis px per source px."""
    n = len(cs)
    pos = np.full((n, 2), np.nan, np.float64)
    same = np.zeros(n, np.float32)
    if n == 0:
        return pos, same
    mp = mo["map"]
    x0, y0, x1, y1 = mo["box"]
    h, w = mp.shape[:2]
    fx, fy = w / float(x1 - x0), h / float(y1 - y0)
    # each point of t's piece, as a place on s in the piece's own reduced px
    ux = np.ascontiguousarray((mp[..., 0] - x0 + 0.5) * fx - 0.5, np.float32)
    uy = np.ascontiguousarray((mp[..., 1] - y0 + 0.5) * fy - 0.5, np.float32)
    qx = ((cs["x"].astype(np.float64) * sx - x0 + 0.5) * fx - 0.5).astype(np.float32)
    qy = ((cs["y"].astype(np.float64) * sy - y0 + 0.5) * fy - 0.5).astype(np.float32)
    # the point of t that lands on q: start where q itself is and step by what is still missing (the motion is
    # smooth: a few steps)
    px_, py_ = qx.copy(), qy.copy()
    for _ in range(6):
        fxp = cv2.remap(ux, px_[None], py_[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        fyp = cv2.remap(uy, px_[None], py_[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
        px_ = px_ + (qx - fxp)
        py_ = py_ + (qy - fyp)
    fxp = cv2.remap(ux, px_[None], py_[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    fyp = cv2.remap(uy, px_[None], py_[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    ok = (np.hypot(fxp - qx, fyp - qy) <= 0.75) & (px_ >= 0) & (px_ <= w - 1) & (py_ >= 0) & (py_ <= h - 1)
    sm = cv2.remap(mo["same"], px_[None], py_[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    pos[ok, 0] = ((px_[ok] + 0.5) / fx - 0.5 + x0) / sx
    pos[ok, 1] = ((py_[ok] + 0.5) / fy - 0.5 + y0) / sy
    same[ok] = sm[ok]
    return pos, same


def _same_at(mo: dict, x: float, y: float, sx: float, sy: float) -> float:
    """How alike frame s (carried) and frame t are at the point (x, y) of t (source px); 0 outside the piece."""
    x0, y0, x1, y1 = mo["box"]
    h, w = mo["same"].shape
    u = int(round((x * sx - x0 + 0.5) * w / float(x1 - x0) - 0.5))
    v = int(round((y * sy - y0 + 0.5) * h / float(y1 - y0) - 0.5))
    if not (0 <= u < w and 0 <= v < h):
        return 0.0
    return float(mo["same"][v, u])


def fuse(own: dict, others: list, ref_src: float, sx: float, sy: float) -> dict:
    """A frame's spots held steady through time. others: [(weight, det of a neighbouring frame, motion of this frame
    onto that one)]. As the face's detection is fused (engine._det_fuse), on the spots themselves:
    - a neighbour has a say at a place only where its picture, carried here, is this frame's picture;
    - a spot found here that the neighbours with a say do not have is taken down (grain taken for a spot);
    - a spot the neighbours have and this frame misses is put here, with its neighbour's patch;
    - a spot found here and on the frames round it is filled over the same patch on all of them: this frame's own
      and its best neighbour's laid on it (a spot that is fainter on one frame is else filled less there, and
      what is left of it blinks);
    - a mark any frame holds (a thing, a mole, a pattern, a line) is held here too;
    - a spot is removed only when a frame beside this one shows the same spot at that place on the body: by the
      picture's own motion where that has a say, else by the body part the spot lies on (others' fourth entry,
      `where`: the joints carry the place to the neighbour, and the neighbour's own skin area tells whether the
      place is to be seen there). A neighbour that sees the place and has no such spot speaks against it; one on
      which the place is covered or out of the picture cannot say. A spot some neighbour could have confirmed and
      none did is held (HOLD_ALONE). On real footage (j2790, 2026-11-21: fast dance, 4K) the picture's motion had
      nothing to say on most frames, the spots then stood on their own frame's word, and every one of them was
      gone again on the next frame (0 of 9 seen again): grain or a streak taken for a spot, and a retouch that
      blinks. The operator's own mark removes such a spot all the same (apply_marks).
    A spot no neighbour can say anything about (a cut of one frame; the place covered on every frame beside this
    one) stands on its own frame's word."""
    where_of = {id(o[1]): (o[3] if len(o) > 3 else None) for o in others if o[1] is not None and o[2] is not None}
    others = [(o[0], o[1], o[2]) for o in others if o[1] is not None and o[2] is not None]
    if not others:
        return own
    co = own["cands"]
    tol0 = TOL * ref_src
    n0 = len(co)
    wtot = float(sum(w for w, _, _ in others))
    acc_p = np.zeros(n0, np.float64)             # the neighbours' weighted scores at each of this frame's spots
    acc_w = np.zeros(n0, np.float64)             # and the weight of those that have a say there
    hold = co["hold"].copy()
    best = {}                                    # this frame's spot -> (weight * score, det, index) of its best partner
    extra = []                                   # spots of neighbours with no partner here: [x, y, w, score, d, det, index]
    for w, ds, mo in others:
        cs = ds["cands"]
        pos, same = _mapped(mo, cs, sx, sy)
        used = np.zeros(len(cs), bool)
        for i in range(n0):
            sm = _same_at(mo, float(co["x"][i]), float(co["y"][i]), sx, sy)
            if sm < SAME_MIN:
                continue
            acc_w[i] += w * sm
            if len(cs):
                d = np.hypot(pos[:, 0] - co["x"][i], pos[:, 1] - co["y"][i])
                d[~np.isfinite(d)] = np.inf
                j = int(np.argmin(d))
                if d[j] <= max(tol0, 0.6 * float(co["d"][i])):
                    used[j] = True
                    sc_j = float(max(cs["s"][j], cs["r"][j]))
                    acc_p[i] += w * sm * sc_j
                    if cs["hold"][j] in (HOLD_NONE, HOLD_WEAK) and (i not in best or w * sm * sc_j > best[i][0]):
                        best[i] = (w * sm * sc_j, ds, int(j))
                    if cs["hold"][j] in (HOLD_THING, HOLD_MOLE, HOLD_CROWD, HOLD_LONG) and hold[i] in (HOLD_NONE, HOLD_WEAK):
                        hold[i] = cs["hold"][j]
        for j in np.nonzero(~used & np.isfinite(pos[:, 0]) & (same >= SAME_MIN))[0]:
            extra.append((float(pos[j, 0]), float(pos[j, 1]), w * float(same[j]), float(max(cs["s"][j], cs["r"][j])),
                          float(cs["d"][j]), ds, int(j)))
    out_c = co.copy()
    scale_o = np.ones(n0, np.float32)            # what each own patch is multiplied by
    fused = np.full(n0, -1.0, np.float32)        # the score of each spot with its neighbours (-1: they have no say)
    for i in range(n0):
        o = float(max(co["s"][i], co["r"][i]))
        wn = acc_w[i]
        if wn <= 0 or o <= 0:
            continue
        p = acc_p[i] / wn
        conf = float(np.clip(wn / (0.5 * wtot), 0, 1))
        both = (o + p * wn) / (1.0 + wn)
        f = conf * both * float(DT._ss(CONFIRM[0], CONFIRM[1], np.float32(p))) + (1.0 - conf) * o
        scale_o[i] = float(np.clip(f / o, 0, 1.5))
        fused[i] = f
        if hold[i] == HOLD_WEAK and f >= TAKE_MIN:
            hold[i] = HOLD_NONE                  # faint here, sure on the frames round it
    if ALONE_HELD:
        tol_a = ANCHOR_TOL * ref_src
        for i in range(n0):
            if hold[i] != HOLD_NONE or i in best:
                continue
            said = acc_w[i] > 0                  # a neighbour saw the place by the picture's motion: nothing there
            ok = False
            x, y = float(co["x"][i]), float(co["y"][i])
            for _w, ds, _mo in others:
                wh = where_of.get(id(ds))
                q = wh(x, y) if wh is not None else None
                if q is None or q[2] < ANCHOR_SKIN:
                    continue                     # the place is not to be seen on that frame: it cannot say
                said = True
                cs = ds["cands"]
                if not len(cs):
                    continue
                d = np.hypot(cs["x"] - q[0], cs["y"] - q[1])
                jn = int(np.argmin(d))
                if d[jn] > max(tol_a, 0.6 * float(co["d"][i])):
                    continue
                if cs["hold"][jn] in (HOLD_THING, HOLD_MOLE, HOLD_CROWD, HOLD_LONG):
                    hold[i] = cs["hold"][jn]     # what that frame keeps there is kept here
                    ok = True
                    break
                if cs["hold"][jn] in (HOLD_NONE, HOLD_WEAK) and max(cs["s"][jn], cs["r"][jn]) >= CAND_MIN:
                    ok = True                    # the same spot at the same place of the body: it stands
            if said and not ok:
                hold[i] = HOLD_ALONE             # a frame beside this one could have shown it and none did
    out_c["hold"] = hold
    out_c["s"] = np.clip(co["s"] * scale_o, 0, 1)
    out_c["r"] = np.clip(co["r"] * scale_o, 0, 1)
    ps, pr = [], []
    so = float(own["scale"])
    for i in range(n0):
        a, b = patch(own, i)
        k = float(scale_o[i])
        a = np.clip(a.astype(np.float32) * k, 0, 255)
        b = np.clip(b.astype(np.float32) * k, 0, 255)
        bp = best.get(i)
        if bp is not None and fused[i] > 0:
            # the best partner's patch, brought to the fused score, with its middle on this spot's middle
            _, ds, j = bp
            cj = ds["cands"][j]
            kn = float(min(1.0, fused[i] / max(float(max(cj["s"], cj["r"])), 1e-3)))
            a2, b2 = patch(ds, j)
            fw = so / float(ds["scale"])
            if abs(fw - 1.0) > 1e-6:
                nw, nh = max(1, int(round(a2.shape[1] * fw))), max(1, int(round(a2.shape[0] * fw)))
                a2 = cv2.resize(a2, (nw, nh), interpolation=cv2.INTER_LINEAR)
                b2 = cv2.resize(b2, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ox0, oy0 = int(round(float(co["px0"][i]) * so)), int(round(float(co["py0"][i]) * so))
            nx0 = int(round((float(cj["px0"]) + float(co["x"][i]) - float(cj["x"])) * so))
            ny0 = int(round((float(cj["py0"]) + float(co["y"][i]) - float(cj["y"])) * so))
            u0, v0 = min(ox0, nx0), min(oy0, ny0)
            u1, v1 = max(ox0 + a.shape[1], nx0 + a2.shape[1]), max(oy0 + a.shape[0], ny0 + a2.shape[0])
            if u1 - u0 <= 4 * max(a.shape[1], a2.shape[1]) and v1 - v0 <= 4 * max(a.shape[0], a2.shape[0]):
                A = np.zeros((v1 - v0, u1 - u0), np.float32)
                Bm = np.zeros_like(A)
                A[oy0 - v0:oy0 - v0 + a.shape[0], ox0 - u0:ox0 - u0 + a.shape[1]] = a
                Bm[oy0 - v0:oy0 - v0 + b.shape[0], ox0 - u0:ox0 - u0 + b.shape[1]] = b
                sl = (slice(ny0 - v0, ny0 - v0 + a2.shape[0]), slice(nx0 - u0, nx0 - u0 + a2.shape[1]))
                A[sl] = np.maximum(A[sl], a2.astype(np.float32) * kn)
                Bm[sl] = np.maximum(Bm[sl], b2.astype(np.float32) * kn)
                a, b = A, Bm
                out_c["px0"][i], out_c["py0"][i] = u0 / so, v0 / so
                out_c["w"][i], out_c["h"][i] = u1 - u0, v1 - v0
                out_c["s"][i] = max(float(out_c["s"][i]), float(np.clip(float(cj["s"]) * kn, 0, 1)))
                out_c["r"][i] = max(float(out_c["r"][i]), float(np.clip(float(cj["r"]) * kn, 0, 1)))
        ps.append(np.clip(a, 0, 255).astype(np.uint8))
        pr.append(np.clip(b, 0, 255).astype(np.uint8))
    # spots only the neighbours have: the same spot seen from several of them is one
    add = []
    extra.sort(key=lambda e: -e[2] * e[3])
    taken = []
    for x, y, w, sc, d, ds, j in extra:
        if any(np.hypot(x - tx, y - ty) <= max(tol0, 0.6 * d) for tx, ty in taken):
            continue
        grp = [e for e in extra if np.hypot(e[0] - x, e[1] - y) <= max(tol0, 0.6 * d)]
        taken.append((x, y))
        # every neighbour with a say at this place counts, those that have nothing there too
        wn = 0.0
        for w_, _d, mo in others:
            sm = _same_at(mo, x, y, sx, sy)
            if sm >= SAME_MIN:
                wn += w_ * sm
        if wn <= 0:
            continue
        p = sum(e[2] * e[3] for e in grp) / wn
        conf = float(np.clip(wn / (0.5 * wtot), 0, 1))
        f = conf * (p * wn / (1.0 + wn)) * float(DT._ss(CONFIRM[0], CONFIRM[1], np.float32(p)))
        if f < CAND_MIN or len(grp) < 2:
            continue                             # one neighbour alone does not put a spot here
        c = ds["cands"][j].copy()
        a, b = patch(ds, j)
        k = f / max(sc, 1e-3)
        dx, dy = x - float(c["x"]), y - float(c["y"])
        c["x"], c["y"] = x, y
        c["px0"], c["py0"] = float(c["px0"]) + dx, float(c["py0"]) + dy
        c["s"], c["r"] = float(np.clip(c["s"] * k, 0, 1)), float(np.clip(c["r"] * k, 0, 1))
        if c["hold"] == HOLD_WEAK and f >= TAKE_MIN:
            c["hold"] = HOLD_NONE
        # (the neighbour's picture may be of another size than this frame's: its patch is kept at its own)
        if abs(ds["scale"] - own["scale"]) > 1e-6:
            fw = own["scale"] / ds["scale"]
            nw, nh = max(1, int(round(a.shape[1] * fw))), max(1, int(round(a.shape[0] * fw)))
            a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_LINEAR)
            c["w"], c["h"] = nw, nh
        add.append(c)
        ps.append(np.clip(a.astype(np.float32) * k, 0, 255).astype(np.uint8))
        pr.append(np.clip(b.astype(np.float32) * k, 0, 255).astype(np.uint8))
    cands = np.concatenate([out_c, np.array(add, CAND)]) if add else out_c
    out = pack(cands, ps, pr, own["scale"], own.get("tiles", 0), own.get("src", False))
    out["carried"] = len(add)
    return out


# ------------------------------------------------------------------------------------------------- marks by hand
def apply_marks(det: dict, marks: list, scale: float) -> dict:
    """The operator's marks on a frame's spots. marks: [(kind, points (n, 2) source px, radius source px)] as they
    lie on this frame. 'exclude': every spot under the mark stays. 'remove': the spots under it go whatever held
    them, and the marked place itself is filled (a spot the detection did not find)."""
    if not marks:
        return det
    c = det["cands"].copy()
    ps, pr = [], []
    for i in range(len(c)):
        a, b = patch(det, i)
        ps.append(a)
        pr.append(b)

    def under(pts, r):
        if not len(c):
            return np.zeros(0, bool)
        d = np.full(len(c), np.inf)
        P = np.asarray(pts, np.float64).reshape(-1, 2)
        for k in range(len(P)):
            a = P[k]
            b = P[min(k + 1, len(P) - 1)]
            ab = b - a
            L2 = float(ab @ ab)
            q = np.stack([c["x"], c["y"]], 1).astype(np.float64) - a
            tt = np.clip((q @ ab) / L2, 0, 1) if L2 > 1e-9 else np.zeros(len(c))
            d = np.minimum(d, np.hypot(*(q - tt[:, None] * ab).T))
        return d <= r + 0.5 * c["d"]
    add = []
    for kind, pts, r in marks:
        hit = under(pts, r)
        if kind == "exclude":
            c["hold"][hit] = HOLD_USER
            continue
        c["hold"][hit & (c["hold"] != HOLD_USER)] = HOLD_NONE
        for k in np.nonzero(hit)[0]:             # (a held mark's score may be low: it is the operator's now)
            m = max(float(c["s"][k]), float(c["r"][k]), 1e-3)
            g = 1.0 / m
            ps[k] = np.clip(ps[k].astype(np.float32) * g, 0, 255).astype(np.uint8)
            pr[k] = np.clip(pr[k].astype(np.float32) * g, 0, 255).astype(np.uint8)
            c["s"][k], c["r"][k] = min(1.0, c["s"][k] * g), min(1.0, c["r"][k] * g)
        # the marked place itself
        P = np.asarray(pts, np.float64).reshape(-1, 2)
        rp = max(1.0, r * scale)
        x0, y0 = P[:, 0].min() - r - 1, P[:, 1].min() - r - 1
        w = int(np.ceil((P[:, 0].max() - P[:, 0].min() + 2 * r + 2) * scale)) + 1
        h = int(np.ceil((P[:, 1].max() - P[:, 1].min() + 2 * r + 2) * scale)) + 1
        if w > 2048 or h > 2048:
            continue
        m = np.zeros((h, w), np.uint8)
        q = np.round((P - [x0, y0]) * scale).astype(np.int32)
        if len(q) == 1:
            cv2.circle(m, (int(q[0, 0]), int(q[0, 1])), int(round(rp)), 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(m, [q], False, 255, max(1, int(round(2 * rp))), cv2.LINE_AA)
            for pt in (q[0], q[-1]):
                cv2.circle(m, (int(pt[0]), int(pt[1])), int(round(rp)), 255, -1, cv2.LINE_AA)
        n = np.zeros(1, CAND)[0]
        n["x"], n["y"] = float(P[:, 0].mean()), float(P[:, 1].mean())
        n["d"] = 2.0 * r
        n["s"], n["r"], n["kind"], n["hold"] = 1.0, 0.0, KIND_USER, HOLD_NONE
        n["px0"], n["py0"], n["w"], n["h"] = x0, y0, w, h
        add.append(n)
        ps.append(m)
        pr.append(np.zeros_like(m))
    cands = np.concatenate([c, np.array(add, CAND)]) if add else c
    out = pack(cands, ps, pr, det["scale"], det.get("tiles", 0), det.get("src", False))
    for k in ("carried",):
        if k in det:
            out[k] = det[k]
    return out


# ------------------------------------------------------------------------------------------------- filling
FILL_CELL = 512             # the spots are filled in groups of about this many plane pixels across


def taken(det: dict) -> np.ndarray:
    """Which candidates are removed."""
    c = det["cands"]
    return (c["hold"] == HOLD_NONE) & (np.maximum(c["s"], c["r"]) >= CAND_MIN)


def raster(det: dict, idx, box, y_per_src: float) -> Tuple[np.ndarray, np.ndarray]:
    """The S and R scores of candidates idx on a piece (y0, y1, x0, x1) of a plane with y_per_src plane px per
    source px (float32 0..1)."""
    y0, y1, x0, x1 = box
    S = np.zeros((y1 - y0, x1 - x0), np.float32)
    R = np.zeros_like(S)
    k = y_per_src / det["scale"]                  # plane px per patch px
    for i in idx:
        c = det["cands"][i]
        a, b = patch(det, int(i))
        w, h = max(1, int(round(a.shape[1] * k))), max(1, int(round(a.shape[0] * k)))
        px0, py0 = int(round(float(c["px0"]) * y_per_src)) - x0, int(round(float(c["py0"]) * y_per_src)) - y0
        if abs(k - 1.0) > 1e-3:
            inter = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
            a = cv2.resize(a, (w, h), interpolation=inter)
            b = cv2.resize(b, (w, h), interpolation=inter)
        xa, ya, xb, yb = max(0, px0), max(0, py0), min(S.shape[1], px0 + w), min(S.shape[0], py0 + h)
        if xb <= xa or yb <= ya:
            continue
        sa = a[ya - py0:yb - py0, xa - px0:xb - px0].astype(np.float32) / 255.0
        sb = b[ya - py0:yb - py0, xa - px0:xb - px0].astype(np.float32) / 255.0
        S[ya:yb, xa:xb] = np.maximum(S[ya:yb, xa:xb], sa)
        R[ya:yb, xa:xb] = np.maximum(R[ya:yb, xa:xb], sb)
    return S, R


def fill(Y, Cb, Cr, mY, det: dict, k: float, body_size: float, ref: float, y_per_src: float, cs: float, spot_fill):
    """The spots of det filled from the skin round them (engine._spot_fill), group by group: only the pieces of the
    planes that have a spot are worked on. mY: the body's skin on this plane; ref: the body's size in plane px.
    What the detection found goes by the slider's strength k; what the operator marked goes in full."""
    tk = taken(det)
    user = det["cands"]["kind"] == KIND_USER
    Y, Cb, Cr = _fill(Y, Cb, Cr, mY, det, np.nonzero(tk & ~user)[0], k, body_size, ref, y_per_src, cs, spot_fill)
    return _fill(Y, Cb, Cr, mY, det, np.nonzero(tk & user)[0], 1.0, body_size, ref, y_per_src, cs, spot_fill)


def _fill(Y, Cb, Cr, mY, det: dict, idx, k: float, body_size: float, ref: float, y_per_src: float, cs: float, spot_fill):
    if not len(idx) or k < 1e-4:
        return Y, Cb, Cr
    hY, wY = Y.shape
    hC, wC = Cb.shape
    sx_, sy_ = wY // wC if wC else 1, hY // hC if hC else 1          # chroma subsampling (1 or 2 each way)
    size = spot_size(body_size, ref)
    c = det["cands"]
    Yo, Cbo, Cro = Y, Cb, Cr
    copied = False
    cell = {}
    for i in idx:
        cell.setdefault((int(c["y"][i] * y_per_src) // FILL_CELL, int(c["x"][i] * y_per_src) // FILL_CELL), []).append(int(i))
    for (cy, cx), ii in sorted(cell.items()):
        # hand marks may be larger than the largest spot of the slider: each group is filled at its own largest size
        sz = max(size, float(max(c["d"][i] * y_per_src for i in ii if c["kind"][i] == KIND_USER) if any(c["kind"][i] == KIND_USER for i in ii) else 0.0))
        pad = int(2.5 * sz) + 8
        xs = [c["px0"][i] * y_per_src for i in ii] + [(c["px0"][i] + c["w"][i] / det["scale"]) * y_per_src for i in ii]
        ys = [c["py0"][i] * y_per_src for i in ii] + [(c["py0"][i] + c["h"][i] / det["scale"]) * y_per_src for i in ii]
        x0, x1 = max(0, int(min(xs)) - pad), min(wY, int(np.ceil(max(xs))) + pad)
        y0, y1 = max(0, int(min(ys)) - pad), min(hY, int(np.ceil(max(ys))) + pad)
        x0, y0 = x0 // sx_ * sx_, y0 // sy_ * sy_
        x1, y1 = min(wY, -(-x1 // sx_) * sx_), min(hY, -(-y1 // sy_) * sy_)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        S, R = raster(det, ii, (y0, y1, x0, x1), y_per_src)
        if float(max(S.max(), R.max())) < 0.02:
            continue
        if not copied:
            Yo, Cbo, Cro = Y.astype(np.float32, copy=True), Cb.astype(np.float32, copy=True), Cr.astype(np.float32, copy=True)
            copied = True
        c0, c1, d0, d1 = y0 // sy_, y1 // sy_, x0 // sx_, x1 // sx_
        yy, cb, cr = spot_fill(Yo[y0:y1, x0:x1], Cbo[c0:c1, d0:d1], Cro[c0:c1, d0:d1], mY[y0:y1, x0:x1], S, R, k, sz, cs)
        Yo[y0:y1, x0:x1] = yy
        Cbo[c0:c1, d0:d1] = cb
        Cro[c0:c1, d0:d1] = cr
    return Yo, Cbo, Cro
