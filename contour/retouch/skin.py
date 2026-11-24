"""Skin smoothing that never touches eyes, brows, lips, nostrils or hair.

Mask  = skin colour probability (learned per person from their own face)
        x the person's matte (only the selected person)
        x face area (scope 'face') or the whole body (scope 'all')
        x NOT (eyes, lashes, brows, lips, nostrils) built from the tracked 68 face landmarks.
Filter = edge-preserving guided filter: the large-scale tone is evened out, a chosen share of the fine
        texture (pores) is put back so skin does not turn plastic. Works on luma and, more gently, chroma.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from . import gpuops as _GPU   # (11/24) cv2 with its GaussianBlur on the GPU when engine.GPU_MASKS is on (see gpuops)
cv2 = _GPU.proxy()

from ..track.store import VIS_VISIBLE

EYE_R = list(range(36, 42))
EYE_L = list(range(42, 48))
BROW_R = list(range(17, 22))
BROW_L = list(range(22, 27))
LIPS = list(range(48, 60))
NOSTRILS = [31, 32, 33, 34, 35]


@dataclass
class SkinJob:
    p: int
    strength: float          # 0..1
    keep: float              # 0..1 share of fine texture kept
    scope: str               # face | all
    face: Optional[np.ndarray]   # (68,3) source px
    face_w: float
    scale: float             # person height, source px
    head: Optional[np.ndarray] = None   # (5,2) nose, eyes, ears from the body skeleton, source px


def skin_jobs(cut, tracker, t: int) -> List[SkinJob]:
    st = tracker.store
    out = []
    if st is None:
        return out
    for person in cut.targets():
        if not st.has(person.id):
            continue
        p = st.index(person.id)
        if not st.present[p, t]:
            continue
        pp = cut.params_for(person.id)
        s = pp.amount("skin", t) / 100.0
        if s <= 1e-3:
            continue
        face = None
        fw = 0.0
        if st.has_face and np.isfinite(st.face[p, t, :, :2]).any():
            face = st.face[p, t]
            fw = float(st.face_vis[p, t])
        from ..core.project import scope_set
        if scope_set(pp.skin_scope) == {"face"} and (face is None or fw < 0.05):
            continue
        sc = float(st.scale[p, t]) if np.isfinite(st.scale[p, t]) else 0.0
        head = st.final[p, t, 0:5, :2].copy()
        out.append(SkinJob(p, float(np.clip(s, 0, 1)), float(np.clip(pp.skin_detail / 100.0, 0, 1)),
                           pp.skin_scope, face, fw, sc, head))
    return out


# --------------------------------------------------------------------------- colour model

def _ycrcb(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)


def face_polygon(lm: np.ndarray, lift: Optional[float] = None) -> Optional[np.ndarray]:
    """Jaw line plus a forehead arc (brows lifted by a third of the face width). lift: a fixed height of the arc
    (of the face width) instead of the one guessed from the head's tilt."""
    pts = lm[:, :2]
    if not np.isfinite(pts[list(range(0, 27))]).all():
        return None
    jaw = pts[0:17]
    brows = pts[17:27]
    Wf = float(np.hypot(*(pts[16] - pts[0])))
    up = -np.array([-(pts[16] - pts[0])[1], (pts[16] - pts[0])[0]]) / max(Wf, 1e-6)
    if (pts[8] - (pts[0] + pts[16]) / 2) @ up > 0:
        up = -up
    # a head tilted down shows less forehead, and above the brows is then the fringe: the arc shrinks with
    # how short the face looks (brows to chin against the face width; about 0.85 for a level face)
    bc = brows.mean(0)
    span = float(abs((pts[8] - bc) @ up)) / max(Wf, 1e-6)
    if lift is None:
        lift = 0.33 * float(np.clip((span - 0.45) / 0.35, 0.25, 1.0))
    forehead = brows[::-1] + up * lift * Wf
    return np.vstack([jaw, forehead]).astype(np.float32)


FOREHEAD_PARSED = 0.55  # of the face width: how far above the brows the parser's skin is taken as forehead
EYE_BELOW = 0.18       # of the eye's width: how far below the eye's middle the eye area reaches
BROW_TH = 0.04        # brow band left out (0.075 left a visible unretouched band above the brows)
# v3 (fit to the hand-retouched reference of cut 069, which is retouched over the whole face: nose, upper lip,
# chin and the skin right up to the eyes and the lips): where the face parser is trusted its skin IS the skin,
# and the features are only left out as far as they really reach
EYE_RX, EYE_RY = 0.70, 0.32   # eye ellipse, of the eye's width (was 0.78 x 0.50)
FEAT_SOFT = 0.022      # of the face width: soft edge of the features in the mask (was 0.05, times 1.6)
FEAT_GAIN = 1.25
PF_FEAT = 0.02         # parser fill: how far the features are widened (was 0.06)
PF_FEAT_SOFT = 0.012   # (was 0.02)
PF_MOUTH = 0.035       # parser fill: the lips' hull widened by this (was 0.12: chin and upper lip fell out)
PF_MOUTH_SOFT = 0.015  # (was 0.03)
PF_EDGE = 0.025        # parser fill: kept this far inside the face outline (was 0.04)
PARSE_STRICT = (0.36, 0.62)   # the parser's skin + neck share from which a pixel counts as skin (None: the old, wide gate)
PARSE_OWN = False      # inside the face outline the parser's sure skin counts even where the person's matte misses it
PARSE_REACH = 0.25     # of the face width: how far past the landmark outline the parser's sure skin still counts
PARSE_SOFT = 0.008     # of the face width: smoothing of the parser's map before that (was 0.012)
NOSTRIL_RX, NOSTRIL_RY = 0.60, 0.26   # nostril ellipse, of the nostrils' span (was 0.66 x 0.36)


def _features_mask(lm: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """255 where facial features are (eyes incl. lashes, brows, lips, nostrils)."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    pts = lm[:, :2]
    Wf = float(np.hypot(*(pts[16] - pts[0]))) if np.isfinite(pts[[0, 16]]).all() else 0.0
    for idx in (EYE_R, EYE_L):
        e = pts[idx]
        if not np.isfinite(e).all():
            continue
        c = e.mean(0)
        ew = float(np.hypot(*(e[3] - e[0])))
        ang = float(np.degrees(np.arctan2(e[3][1] - e[0][1], e[3][0] - e[0][0])))
        em = np.zeros((h, w), np.uint8)
        cv2.ellipse(em, (int(round(c[0])), int(round(c[1]))), (max(2, int(ew * EYE_RX)), max(2, int(ew * EYE_RY))),
                    ang, 0, 360, 255, -1)
        # under the eye only as far as the lower lashes: the skin below them (fine lines, the bag under the
        # eye) is skin and is retouched; the whole ellipse used to leave it out
        ax = (e[3] - e[0]) / max(ew, 1e-6)
        dn = np.array([-ax[1], ax[0]])
        if np.isfinite(pts[[8, 27]]).all() and float(dn @ (pts[8] - pts[27])) < 0:
            dn = -dn
        p0 = c + dn * EYE_BELOW * ew
        q = [p0 - ax * 2 * ew, p0 + ax * 2 * ew, p0 + ax * 2 * ew + dn * 2 * ew, p0 - ax * 2 * ew + dn * 2 * ew]
        cv2.fillConvexPoly(em, np.round(np.array(q)).astype(np.int32), 0)
        m = np.maximum(m, em)
    for idx in (BROW_R, BROW_L):
        b = pts[idx]
        if not np.isfinite(b).all():
            continue
        th = max(2, int(BROW_TH * Wf)) if Wf else 3      # the brow's own hairs are also kept by the engine's edge hold
        cv2.polylines(m, [np.round(b).astype(np.int32)], False, 255, th)
    lp = pts[LIPS]
    if np.isfinite(lp).all():
        cv2.fillPoly(m, [np.round(lp).astype(np.int32)], 255)
        d = max(1, int(0.03 * Wf)) if Wf else 1
        m = cv2.dilate(m, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) if d > 1 else m
    ns = pts[NOSTRILS]
    if np.isfinite(ns).all():
        nw = float(np.hypot(*(ns[-1] - ns[0])))
        # the nostrils sit a little above the line of the lower nose points: cover them whole, so that face
        # smoothing never lifts them (a nose then looks flat)
        up = pts[27] - pts[33] if np.isfinite(pts[[27, 33]]).all() else np.array([0.0, -1.0])
        up = up / max(float(np.hypot(*up)), 1e-6)
        c = ns.mean(0) + up * 0.08 * nw
        ang = float(np.degrees(np.arctan2(ns[-1][1] - ns[0][1], ns[-1][0] - ns[0][0])))
        cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (max(1, int(nw * NOSTRIL_RX)), max(1, int(nw * NOSTRIL_RY))),
                    ang, 0, 360, 255, -1)
    return m


def build_skin_model(rgb_frames: List[np.ndarray], faces: List[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Robust Gaussian of (Cr, Cb) from face skin pixels (features removed)."""
    samples = []
    for rgb, lm in zip(rgb_frames, faces):
        poly = face_polygon(lm)
        if poly is None:
            continue
        h, w = rgb.shape[:2]
        reg = np.zeros((h, w), np.uint8)
        cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 255)
        feat = _features_mask(lm, (h, w))
        feat = cv2.dilate(feat, np.ones((5, 5), np.uint8))
        sel = (reg > 0) & (feat == 0)
        ycc = _ycrcb(rgb)
        y = ycc[..., 0]
        sel &= (y > 12) & (y < 245)
        px = ycc[sel][:, 0:3].astype(np.float32)
        if len(px) > 50:
            if len(px) > 4000:
                px = px[np.random.default_rng(0).choice(len(px), 4000, replace=False)]
            samples.append(px)
    if not samples:
        return None
    X = np.concatenate(samples)
    yref = float(np.median(X[:, 0]))
    X = X[:, 1:3]
    med = np.median(X, 0)
    d = np.linalg.norm(X - med, axis=1)
    keep = d <= np.percentile(d, 85)          # <=: flat chroma (clipped or one-colour grade) keeps its pixels
    if keep.sum() >= 20:
        X = X[keep]
    if len(X) < 20:
        return None
    mean = X.mean(0)
    # never tighter than a few code values: graded footage (one coloured light on everything) has very little
    # chroma spread, and a tight model turns 8-bit steps into a blotchy mask
    cov = np.cov(X.T) + np.eye(2) * 12.0
    return mean, np.linalg.inv(cov), yref


DEFAULT_MODEL = (np.array([150.0, 112.0]), np.linalg.inv(np.diag([90.0, 70.0])))


import threading as _thr_planes
_PLANE_TLS = _thr_planes.local()    # each thread keeps the planes of the frame it is on


def _chroma_planes(rgb_u8: np.ndarray):
    """Luma and the smoothed Cr, Cb of a frame, as float32. A crowd frame asks for them once per person (the colour
    model is per person, the planes are not): kept for the frame the thread is working on, the same values as before."""
    key = (rgb_u8.__array_interface__["data"][0], rgb_u8.shape, hash(rgb_u8[::7, ::11].tobytes()))
    ent = getattr(_PLANE_TLS, "planes", None)
    if ent is not None and ent[0] == key:
        return ent[1]
    ycc = _ycrcb(rgb_u8).astype(np.float32)
    # chroma a little smoothed: 8-bit chroma steps must not turn into a blotchy mask
    sig = max(0.8, 0.0012 * rgb_u8.shape[0])
    cr = cv2.GaussianBlur(ycc[..., 1], (0, 0), sig)
    cb = cv2.GaussianBlur(ycc[..., 2], (0, 0), sig)
    out = (np.ascontiguousarray(ycc[..., 0]), cr, cb)
    _PLANE_TLS.planes = (key, out)
    return out


def skin_probability(rgb_u8: np.ndarray, model) -> np.ndarray:
    m = model if model is not None else DEFAULT_MODEL
    mean, icov = m[0], m[1]
    yref = m[2] if len(m) > 2 else None
    y, cr, cb = _chroma_planes(rgb_u8)
    d0, d1 = cr - mean[0], cb - mean[1]
    m2 = icov[0, 0] * d0 * d0 + 2 * icov[0, 1] * d0 * d1 + icov[1, 1] * d1 * d1
    md = np.sqrt(np.maximum(m2, 0))
    prob = np.exp(-0.5 * np.maximum(md - 1.6, 0) ** 2)
    if yref is not None and yref > 8:
        # deep shadows, pupils, dark hair: dark against this face's own skin (dark grades stay skin)
        prob *= np.clip((y - 0.30 * yref) / (0.25 * yref), 0, 1)
    else:
        prob *= np.clip((y - 25) / 25, 0, 1)
    return prob.astype(np.float32)


def face_prior(lm: Optional[np.ndarray], rgb_u8: np.ndarray, model) -> Optional[np.ndarray]:
    """Skin by where it is, not by its colour: inside the face outline (features are taken out later by
    every mask that uses this), minus what is much darker than the face's own skin (hair, deep shadow).
    Under one strong coloured light, clothes, background and skin share a colour and the colour model alone
    leaves holes; the face geometry from the landmarks does not depend on the light."""
    if lm is None or not np.isfinite(lm[:, :2]).all():
        return None
    poly = face_polygon(lm)
    if poly is None:
        return None
    h, w = rgb_u8.shape[:2]
    reg = np.zeros((h, w), np.uint8)
    cv2.fillPoly(reg, [np.round(poly * 4).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=2)
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    # a little inside the jaw line and hairline: the outline is where skin meets background or hair
    k = max(1, int(0.03 * Wf))
    reg = cv2.erode(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    pr = cv2.GaussianBlur(reg.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.025 * Wf))
    yref = model[2] if model is not None and len(model) > 2 else None
    if yref is not None and yref > 8:
        y = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        y = cv2.GaussianBlur(y, (0, 0), max(1.0, 0.01 * Wf))
        pr *= np.clip((y - 0.35 * yref) / (0.25 * yref), 0, 1)
    return pr.astype(np.float32)


_TEX_MIN = 1.2          # floor of the cheeks' fine-structure level on a dark face (code values)


def _texture_gate(rgb_u8: np.ndarray, lm: np.ndarray) -> Optional[np.ndarray]:
    """Hair is strands: much more fine structure than the skin of the same face. Bleached or light brown hair
    has the skin's colour, and a fringe over a face tilted down sits right where the forehead would be; the
    colour model alone takes it for skin. Measured against this face's own cheeks, so a soft or motion-blurred
    frame (where hair is smooth too) is judged on its own terms. 1 = skin-like texture, 0 = hair-like."""
    L = lm[:, :2]
    if not np.isfinite(L).all():
        return None
    h, w = rgb_u8.shape[:2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    if Wf < 40:
        return None
    x0, y0 = np.floor(L.min(0) - 0.6 * Wf).astype(int)
    x1, y1 = np.ceil(L.max(0) + 0.6 * Wf).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    y = cv2.cvtColor(rgb_u8[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    s1 = max(0.7, 0.004 * Wf)
    hp = y - cv2.GaussianBlur(y, (0, 0), s1)
    E = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), max(1.5, 0.025 * Wf)))
    # reference: both cheeks and the nose bridge
    ref, yref = [], []
    for i, j in ((1, 31), (15, 35), (28, 29)):
        c = (L[i] + L[j]) / 2 - [x0, y0]
        r = max(2, int(0.05 * Wf))
        cy, cx = int(c[1]), int(c[0])
        if 0 <= cy < E.shape[0] and 0 <= cx < E.shape[1]:
            ref.append(E[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel())
            yref.append(y[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel())
    if not ref:
        return None
    e0 = float(np.median(np.concatenate(ref)))
    # never finer than 8-bit steps. On a dark, flat, compressed face the block edges and grain are a code value
    # or two and would read as hair everywhere (a blotchy gate); strands that matter stand out by more than that.
    # A well lit face keeps its own measure (blonde hair on a smooth bright face is only a little finer than skin)
    yr = float(np.median(np.concatenate(yref))) if yref else 128.0
    e0 = max(e0, 0.6 + (_TEX_MIN - 0.6) * float(np.clip((70.0 - yr) / 40.0, 0.0, 1.0)))
    g = np.clip((3.2 * e0 - E) / (1.6 * e0), 0, 1)
    out = np.ones((h, w), np.float32)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(g, (0, 0), max(1.0, 0.012 * Wf))
    return out


def _ridge_aniso(I: np.ndarray, s: float) -> np.ndarray:
    """0..1: how line-like the dark (or, for -I, bright) structure at each pixel is (Hessian eigenvalues)."""
    Ib = cv2.GaussianBlur(I, (0, 0), s)
    dxx = cv2.Sobel(Ib, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(Ib, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(Ib, cv2.CV_32F, 1, 1, ksize=3)
    tr = dxx + dyy
    det = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    l1, l2 = (tr + det) / 2, (tr - det) / 2
    return np.where(l1 > 0, (l1 - np.abs(l2)) / (l1 + np.abs(l2) + 1e-6), 0).astype(np.float32)


_PAINT_MIN_Y = 3.0      # code values of darkening that count as the skin's own texture at the least
_PAINT_MIN_B = 2.0      # the same for the blue (Cb) difference


def paint_mask(rgb_u8: np.ndarray, lm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Face paint, drawn veins, tattoos and ink strokes on the face and neck: 1 there, 0 elsewhere.

    They are part of the look and must come through the retouch untouched (not smoothed, not taken for
    blemishes). Two kinds:
    - lines: dark (or bluer) than the skin around them, much thinner than a face feature, long and line-shaped
      (Hessian), well above the skin's own pore texture; found from strong seeds, grown along weaker line
      pixels so a fine crack is kept whole;
    - ink strokes: sharply bluer (less red, more blue) than their own surroundings and not brighter; measured
      against the surroundings, so a blue light on one side of the face is not ink.
    Sizes follow the face width, so a zoom changes nothing."""
    if lm is None or not np.isfinite(lm[[0, 8, 16, 27]]).all():
        return None
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    if Wf < 60:                         # too small to see paint, and too small to retouch finely
        return None
    h, w = rgb_u8.shape[:2]
    x0, y0 = np.floor(L.min(0) - [0.3 * Wf, 0.7 * Wf]).astype(int)
    x1, y1 = np.ceil(L.max(0) + [0.3 * Wf, 0.9 * Wf]).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    roi = rgb_u8[y0:y1, x0:x1]
    ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    # the face skin for reference levels: inside the outline, eyes, brows and mouth out
    poly = face_polygon(lm)
    reg = np.zeros((h, w), np.uint8)
    if poly is not None:
        cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 1)
    reg = reg[y0:y1, x0:x1].astype(bool)
    feat = _features_mask(lm, (h, w))[y0:y1, x0:x1] > 0
    ref = reg & ~feat
    if ref.sum() < 200:
        return None
    area = np.ones_like(reg)                   # face and neck below it
    k = max(3, int(0.02 * Wf) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    r_y = cv2.morphologyEx(Y, cv2.MORPH_BLACKHAT, se) / (cv2.GaussianBlur(Y, (0, 0), 0.04 * Wf) + 4)
    r_b = cv2.morphologyEx(Cb, cv2.MORPH_TOPHAT, se)
    # the skin's own texture sets the scale, but never below a few code values: a dark, flat, heavily compressed
    # face has almost no texture, and its block edges and grain (one or two code values) read as drawn lines
    y_ref = float(np.median(Y[ref]))
    n_y = max(float(np.percentile(r_y[ref], 90)), _PAINT_MIN_Y / (y_ref + 4))
    n_b = max(float(np.percentile(r_b[ref], 90)) + 0.5, _PAINT_MIN_B)
    sc = np.maximum(r_y / (n_y + 1e-4), r_b / n_b)
    s = max(0.8, 0.004 * Wf)
    an = np.maximum(_ridge_aniso(Y, s), _ridge_aniso(-Cb, s))
    weak = ((sc > 1.6) & (an > 0.35) & area & ~feat).astype(np.uint8)
    strong = (sc > 2.4) & (an > 0.5)
    n, lab, stt, _ = cv2.connectedComponentsWithStats(weak, 8)
    keep = np.zeros_like(weak)
    if n > 1:
        seeds = np.unique(lab[strong & (lab > 0)])
        ok = np.zeros(n, bool)
        for i in seeds:
            ln = float(np.hypot(stt[i, 2], stt[i, 3]))
            ok[i] = ln > 0.04 * Wf and stt[i, 4] < 0.25 * ln * ln
        keep = ok[lab].astype(np.uint8)
    crb = cv2.GaussianBlur(Cr, (0, 0), 1.5)
    cbb = cv2.GaussianBlur(Cb, (0, 0), 1.5)
    ink = (cbb - float(np.median(Cb[ref]))) - (crb - float(np.median(Cr[ref])))
    ink_hp = ink - cv2.GaussianBlur(ink, (0, 0), 0.03 * Wf)
    y_hp = Y - cv2.GaussianBlur(Y, (0, 0), 0.03 * Wf)
    blob = ((ink_hp > 7) & (y_hp < 2) & area & ~feat).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = np.maximum(keep, blob)
    if not m.any():
        return None
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.004 * Wf)) * 1.5, 0, 1)
    return out


def parse_gate(parse: Optional[dict], Wf: float, with_neck: bool = True) -> Optional[np.ndarray]:
    """1 where the face parser sees skin (or neck) or cannot be trusted, down to 0.03 where it sees something
    else in front of or around the face: hair, a hand, a light stick, clothes. Outside the parser's crop: 1."""
    if parse is None or parse.get("trust", 0) <= 0:
        return None
    m = parse["skin"] + (parse["neck"] if with_neck else 0)
    mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, PARSE_SOFT * Wf))
    if PARSE_STRICT is not None:
        # v3: the parser's own border between skin and hair is taken as it is. The widened one (times 1.7) let a
        # blond fringe, bleached hair beside the cheek and hair lying on the neck through as skin wherever the
        # parser was only half sure, and their colour is the skin's (cuts 024, 025, 028, 030, 059, 095)
        g = np.clip((mb - PARSE_STRICT[0]) / (PARSE_STRICT[1] - PARSE_STRICT[0]), 0, 1)
    else:
        g = np.clip(mb * 1.7, 0, 1)
    cov = cv2.GaussianBlur(parse["cover"], (0, 0), max(1.0, 0.03 * Wf))
    return (1.0 - cov * parse["trust"] * (1.0 - g) * 0.97).astype(np.float32)


def _parse_fill(rgb_u8: np.ndarray, lm: np.ndarray, parse: dict) -> Optional[np.ndarray]:
    """Skin the face parser is confident about, inside the (slightly widened) face outline, with the eyes, brows,
    lips and the mouth left out. 0..1."""
    if parse.get("trust", 0) <= 0 or not np.isfinite(lm[[0, 16], :2]).all():
        return None
    poly = face_polygon(lm, FOREHEAD_PARSED)      # the parser itself tells forehead from fringe
    if poly is None:
        return None
    h, w = rgb_u8.shape[:2]
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    reg = np.zeros((h, w), np.uint8)
    cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 255)
    k = max(3, int(PF_EDGE * Wf) | 1)
    reg = cv2.erode(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))   # inside, not on the edge
    reg = cv2.GaussianBlur(reg.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.02 * Wf))
    feat = _features_mask(lm, (h, w))
    kf = max(3, int(PF_FEAT * Wf) | 1)
    feat = cv2.dilate(feat, np.ones((kf, kf), np.uint8))
    featf = cv2.GaussianBlur(feat.astype(np.float32) / 255.0, (0, 0), max(1.0, PF_FEAT_SOFT * Wf))
    mouth = np.zeros((h, w), np.uint8)
    lp = lm[LIPS, :2]
    if np.isfinite(lp).all():
        cv2.fillConvexPoly(mouth, cv2.convexHull(np.round(lp).astype(np.int32)), 255)
        d = max(3, int(PF_MOUTH * Wf) | 1)
        mouth = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    mz = cv2.GaussianBlur(mouth.astype(np.float32) / 255.0, (0, 0), max(1.0, PF_MOUTH_SOFT * Wf))
    sk = np.clip((parse["skin"].astype(np.float32) - 0.55) / 0.3, 0, 1)
    cov = parse.get("cover")
    if cov is not None:
        sk = sk * cov                                  # outside the parser's crop it knows nothing
    tex = parse.get("tex")
    tg = tex.astype(np.float32) if tex is not None else 1.0       # hair against skin: the texture gate keeps it out
    return (0.95 * float(np.clip(parse["trust"], 0, 1)) * sk * reg * (1.0 - featf) * (1.0 - mz) * tg).astype(np.float32)


def face_skin_probability(rgb_u8: np.ndarray, model, lm: Optional[np.ndarray], parse: Optional[dict] = None) -> np.ndarray:
    return face_skin_and_protect(rgb_u8, model, lm, parse)[0]


def face_skin_and_protect(rgb_u8: np.ndarray, model, lm: Optional[np.ndarray], parse: Optional[dict] = None):
    """(the face's skin probability, the soft map of what the retouch leaves alone on it: hair strands over the skin and face
    paint, 0..1, or None). The map is what has been cut out of the probability; the neck mask blurs the probability (a strand is
    a few pixels wide) and needs the map itself to keep a strand out."""
    gate = None
    if parse is not None and lm is not None and np.isfinite(lm[[0, 16], :2]).all():
        gate = parse_gate(parse, float(np.hypot(*(lm[16, :2] - lm[0, :2]))))
    out = _face_skin_probability(rgb_u8, model, lm, gate, parse.get("tex") if parse is not None else None)
    if gate is not None:
        out = out * gate
    # the colour model alone loses skin that sits in shadow or under a strongly coloured light; where the face
    # parser is sure it is skin (inside the face outline, off the features) it is skin
    if parse is not None and lm is not None and gate is not None:
        try:
            pf = _parse_fill(rgb_u8, lm, parse)
            if pf is not None:
                out = np.maximum(out, pf)
        except Exception:
            pass
    # hair strands over the skin, face paint, tattoos, drawn veins: never retouched (soft edged)
    pm = soft_paint(rgb_u8, lm) if lm is not None else None
    if pm is not None:
        out = out * (1.0 - pm)
    return out.astype(np.float32), pm


def _face_skin_probability(rgb_u8: np.ndarray, model, lm: Optional[np.ndarray], gate=None, tex=None) -> np.ndarray:
    """Colour model, held up inside the face by the face geometry only where the colour model fails.

    With the model re-centred on each frame's own face (frame_model_for) the colour alone finds the skin even
    on flash frames, and it is what keeps the lips, the inside of the mouth and the eyes out when the landmarks
    sit a little off (motion blur, a fast turn). The face outline is only a fallback for frames where the colour
    model loses much of the face, and even then the mouth is left to the colour model."""
    prob = skin_probability(rgb_u8, model)
    pr = face_prior(lm, rgb_u8, model)
    if pr is None:
        return prob
    # the face layer's gate is held steady through time; measured on one frame alone it doubled the mask's
    # frame-to-frame flicker (grain and compression noise read as texture)
    tg = tex if tex is not None else _texture_gate(rgb_u8, lm)
    if tg is not None:
        prob = prob * tg
        pr = pr * tg
    h, w = rgb_u8.shape[:2]
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    feat = _features_mask(lm, (h, w))
    k = max(3, int(0.06 * Wf) | 1)
    feat = cv2.dilate(feat, np.ones((k, k), np.uint8))
    core = (pr > 0.9) & (feat == 0)
    if core.sum() < 50:
        return prob
    cov = float(prob[core].mean())
    wgt = float(np.clip((0.80 - cov) / 0.25, 0.0, 1.0))
    if wgt <= 0.0:
        return prob
    mouth = np.zeros((h, w), np.uint8)
    lp = L[LIPS]
    if np.isfinite(lp).all():
        cv2.fillConvexPoly(mouth, cv2.convexHull(np.round(lp).astype(np.int32)), 255)
        d = max(3, int(0.12 * Wf) | 1)
        mouth = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    mz = cv2.GaussianBlur(mouth.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.03 * Wf))
    if gate is not None:
        pr = pr * gate
    return np.maximum(prob, 0.95 * wgt * pr * (1.0 - mz)).astype(np.float32)


# --------------------------------------------------------------------------- hair strands over the skin

_LK: dict = {}
STRAND_LO, STRAND_HI = 0.012, 0.026
STRAND_SEEN = 2.5      # code values (of 255) a strand must differ from the skin beside it to be kept
NECK_LINE_MAX = 0.15   # dark lines on the neck fainter than this (of the skin's brightness) are neck lines
# 2611 IWALY c050 (2026-11-22): a faint hair or a thin necklace chain lying diagonally on the neck was handed back
# to the neck line removal as a neck line and lifted (the chain went pale, a hair strand broke). A neck line is a short,
# soft, curved valley; a hair or a chain is a long straight thin line (its pieces are 8-12 times as long as wide, the
# neck lines and tendons it was confused with 2-5). A piece that long and that straight stays a strand.
NECK_STRAND_GUARD = True
NECK_STRAND_ELONG = 6.5    # at least this many times longer than wide (the ratio of the piece's two principal axes)
NECK_STRAND_SPAN = 0.30    # and at least this long (of the face width)
PAINT_SEEN = 5.0       # the same for a drawn line


def _line_kernels(L, n=12):
    key = (L, n)
    if key not in _LK:
        ks = []
        for i in range(n):
            a = np.pi * i / n
            k = np.zeros((L, L), np.float32)
            c = (L - 1) / 2.0
            p0 = (int(round(c - np.cos(a) * c)), int(round(c - np.sin(a) * c)))
            p1 = (int(round(c + np.cos(a) * c)), int(round(c + np.sin(a) * c)))
            cv2.line(k, p0, p1, 1.0, 1, cv2.LINE_AA)
            ks.append(k / k.sum())
        _LK[key] = ks
    return _LK[key]


def _strand_map(rgb_u8: np.ndarray, Wf: float, dark_only: bool = False, gain: float = 1.0) -> np.ndarray:
    """Hair strands lying over skin (very thin ones too): 0..1, soft edged.

    A strand is a thin line darker (or lighter) than the skin on BOTH sides of it and much longer than it is
    wide. Thin structures are taken with a black-hat / top-hat a few pixels wide (the edge between a lit cheek
    and its shadow does not answer to it, a hair does), relative to the local brightness. What makes it a
    strand is its length: the thin-structure map is averaged along short straight segments in twelve directions,
    and only pixels that lie on a line in some direction are kept, so pores, grain and moles drop out while a
    faint single hair stays whole. The result is widened by a pixel and feathered: the retouch leaves the strand
    alone, takes the skin right beside it, and the border between the two never shows as a line."""
    h, w = rgb_u8.shape[:2]
    # a 2-D float array (0..255) is taken as the brightness itself
    y = rgb_u8.astype(np.float32) if rgb_u8.ndim == 2 else cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    y = cv2.GaussianBlur(y, (0, 0), 0.6)
    base = cv2.GaussianBlur(y, (0, 0), max(2.0, 0.03 * Wf)) + 8.0
    k1 = max(3, int(0.012 * Wf) | 1)
    k2 = max(5, int(0.022 * Wf) | 1)
    kb = max(9, int(0.07 * Wf) | 1)
    def hat(k, op):
        return cv2.morphologyEx(y, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) / base
    Rd = np.maximum(hat(k1, cv2.MORPH_BLACKHAT), hat(k2, cv2.MORPH_BLACKHAT))
    Rl = np.maximum(hat(k1, cv2.MORPH_TOPHAT), hat(k2, cv2.MORPH_TOPHAT))
    # thin only: a hair is all there at the small scale; a shadow, a lip line, a nostril keeps growing with the
    # kernel (the wide hat sees much more of it than the narrow one)
    Bd, Bl = hat(kb, cv2.MORPH_BLACKHAT), hat(kb, cv2.MORPH_TOPHAT)
    thin_d = np.clip((Rd / (Bd + 1e-4) - 0.45) / 0.3, 0, 1)
    thin_l = np.clip((Rl / (Bl + 1e-4) - 0.45) / 0.3, 0, 1)
    if dark_only:
        Rl = Rl * 0.0
    R = np.minimum(np.maximum(Rd * thin_d, 0.8 * Rl * thin_l), 0.25)
    # a strand that can hardly be seen (a faint hair on the shadow side of a face) is not worth a hole in the
    # retouch: there the skin is cleaned and the hair goes with it
    R = R * np.clip((np.maximum(Rd, Rl) * base - STRAND_SEEN) / 3.0, 0, 1)
    L = max(9, int(0.065 * Wf) | 1)
    LR = np.zeros((h, w), np.float32)
    acc = np.zeros((h, w), np.float32)
    ks = _line_kernels(L)
    for k in ks:
        r_ = cv2.filter2D(R, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        LR = np.maximum(LR, r_)
        acc += r_
    # a line answers in one direction only; a mole, a nostril, a speck answers the same in all of them
    aniso = (LR - acc / len(ks)) / (LR + 1e-4)
    lo, hi = STRAND_LO / gain, STRAND_HI / gain          # gain > 1: fainter lines are taken too (skin lines)
    on_line = np.clip((LR - lo) / (hi - lo), 0, 1) * np.clip((aniso - 0.45) / 0.2, 0, 1)
    thin = np.clip((R - 0.5 * lo) / (1.0 * lo), 0, 1)
    core = on_line * thin
    # specks left in noisy shadow: a strand piece is at least a twentieth of the face wide
    n, lab, stats, _ = cv2.connectedComponentsWithStats((core > 0.25).astype(np.uint8), connectivity=8)
    span = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    ok = span >= max(7.0, 0.045 * Wf)
    ok[0] = False
    core = core * ok[lab]
    # a faint strand answers in pieces: each kept piece is followed along the weaker answer joined to it, so the
    # strand is taken whole (a weak answer with no sure piece on it stays out: that is skin texture)
    weak = np.clip((LR - 0.45 * lo) / (0.55 * lo), 0, 1) * np.clip((aniso - 0.35) / 0.2, 0, 1) * \
        np.clip((R - 0.2 * lo) / (0.6 * lo), 0, 1)
    n2, lab2 = cv2.connectedComponents(((weak > 0.3) | (core > 0.25)).astype(np.uint8), connectivity=8)
    if n2 > 1:
        keep = np.zeros(n2, bool)
        keep[np.unique(lab2[core > 0.25])] = True
        keep[0] = False
        core = np.maximum(core, 0.9 * weak * keep[lab2])
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    soft = cv2.GaussianBlur(core, (0, 0), max(0.9, 0.005 * Wf))
    return np.clip(soft * 1.5, 0, 1).astype(np.float32)


def strand_mask(rgb_u8: np.ndarray, lm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Hair strands over the face and neck of this person (0..1, soft), on the whole image; None without a face."""
    if lm is None or not np.isfinite(lm[[0, 8, 16, 27], :2]).all():
        return None
    L = lm[:, :2]
    Wf = float(np.hypot(*(L[16] - L[0])))
    if Wf < 60:
        return None
    h, w = rgb_u8.shape[:2]
    x0, y0 = np.floor(L.min(0) - [0.35 * Wf, 0.7 * Wf]).astype(int)
    x1, y1 = np.ceil(L.max(0) + [0.35 * Wf, 0.9 * Wf]).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = _strand_map(np.ascontiguousarray(rgb_u8[y0:y1, x0:x1]), Wf)
    return out


def soft_paint(rgb_u8: np.ndarray, lm: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """What the skin retouch leaves alone on a face: hair strands (thin, feathered) and face paint.

    The paint finder took strands lying over a cheek for drawn lines: in broken, wide, hard-edged pieces, so the
    cheek beside a strand lost its retouch and the border showed. Strands now have their own finder; a paint
    line that is really a strand is dropped from the paint, and what is left of the paint gets a soft edge too."""
    pm = paint_mask(rgb_u8, lm)
    sm = strand_mask(rgb_u8, lm)
    if sm is None:
        return pm
    if pm is None:
        return sm
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pm > 0.5).astype(np.uint8), connectivity=8)
    if n > 1:
        near = cv2.dilate((sm > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8))
        hit = np.bincount(lab.ravel(), weights=near.ravel().astype(np.float64), minlength=n) / np.maximum(1, stats[:, cv2.CC_STAT_AREA])
        drop = hit > 0.2
        drop[0] = False
        pm = pm * (~drop[lab])
    # what the paint finder takes on the shadow side of a face is the edge of the shadow and hairs that can
    # hardly be seen, not paint: a drawn line is thin (the narrow hat holds most of what the wide one sees) and
    # visibly darker or bluer than the skin beside it. Pieces that are neither are skin, and are retouched
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pm > 0.5).astype(np.uint8), connectivity=8)
    if n > 1:
        ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        yy = cv2.GaussianBlur(ycc[..., 0], (0, 0), 0.6)
        k2 = max(5, int(0.022 * Wf) | 1)
        kb = max(9, int(0.07 * Wf) | 1)
        se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
        nar = cv2.morphologyEx(yy, cv2.MORPH_BLACKHAT, se2)
        wid = cv2.morphologyEx(yy, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
        blu = cv2.morphologyEx(ycc[..., 2], cv2.MORPH_TOPHAT, se2)
        cnt = np.maximum(1, stats[:, cv2.CC_STAT_AREA]).astype(np.float64)
        lr = lab.ravel()
        m_n = np.bincount(lr, weights=nar.ravel().astype(np.float64), minlength=n) / cnt
        m_w = np.bincount(lr, weights=wid.ravel().astype(np.float64), minlength=n) / cnt
        m_b = np.bincount(lr, weights=blu.ravel().astype(np.float64), minlength=n) / cnt
        ink = m_b >= 3.0
        drop = ~ink & ((m_n < PAINT_SEEN) | (m_n < 0.3 * m_w))
        drop[0] = False
        gone = cv2.dilate(drop[lab].astype(np.uint8), np.ones((5, 5), np.uint8))
        pm = pm * (1 - gone)
    pm = np.clip(cv2.GaussianBlur(pm.astype(np.float32), (0, 0), max(0.9, 0.005 * Wf)) * 1.3, 0, 1)
    out = np.maximum(pm, sm).astype(np.float32)
    return _neck_lines_back(out, rgb_u8, lm, Wf)


def _elongation(sub: np.ndarray) -> float:
    """How many times longer than wide a piece is (the ratio of the two principal axes of its pixels; 1 = round)."""
    ys, xs = np.nonzero(sub)
    if len(xs) < 4:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64)))
    return float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))


def _neck_lines_back(out: np.ndarray, rgb_u8: np.ndarray, lm: np.ndarray, Wf: float) -> np.ndarray:
    """Below the jaw a faint thin dark line is a neck line, not a hair. Taken for a strand it was cut out of the
    skin, and the neck line removal never saw it."""
    L = lm[:, :2]
    if not np.isfinite(L[[8, 27]]).all():
        return out
    ax = L[8] - L[27]
    n_ = float(np.hypot(*ax))
    if n_ < 1e-3:
        return out
    ax = ax / n_
    h, w = out.shape
    n, lab, stats, cen = cv2.connectedComponentsWithStats((out > 0.3).astype(np.uint8), connectivity=8)
    if n <= 1:
        return out
    hull = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.round(L).astype(np.int32)), 1)
    y = cv2.GaussianBlur(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 0.6)
    k2 = max(5, int(0.022 * Wf) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    loc = cv2.GaussianBlur(y, (0, 0), max(2.0, 0.03 * Wf)) + 4.0
    bh = cv2.morphologyEx(y, cv2.MORPH_BLACKHAT, se) / loc
    th = cv2.morphologyEx(y, cv2.MORPH_TOPHAT, se) / loc
    drop = np.zeros(n, bool)
    for i in range(1, n):
        x0, y0, bw, bh_, area = stats[i]
        if area < 6:
            continue
        if float((cen[i] - L[8]) @ ax) < -0.12 * Wf:            # above the chin: the face
            continue
        sub = lab[y0:y0 + bh_, x0:x0 + bw] == i
        if hull[y0:y0 + bh_, x0:x0 + bw][sub].mean() > 0.3:
            continue
        dk = float(np.percentile(bh[y0:y0 + bh_, x0:x0 + bw][sub], 75))
        lt = float(np.percentile(th[y0:y0 + bh_, x0:x0 + bw][sub], 75))
        # a neck line is a faint dark line (a tenth of the skin's brightness); a hair over the neck is
        # either light (blond, a highlight) or clearly darker than that. A necklace chain can be as faint, but
        # it is long (it goes round the neck or hangs down the chest) and beads or links glint along it
        span = float(np.hypot(bw, bh_))
        gl = float(np.percentile(th[y0:y0 + bh_, x0:x0 + bw][cv2.dilate(sub.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0], 97))
        if dk > lt and dk < NECK_LINE_MAX and span < 0.6 * Wf and gl < 0.25:
            if NECK_STRAND_GUARD and span >= NECK_STRAND_SPAN * Wf and _elongation(sub) >= NECK_STRAND_ELONG:
                continue                                       # a long straight thin line: a hair, a chain; it stays a strand
            drop[i] = True
    if drop.any():
        gone = cv2.dilate(drop[lab].astype(np.uint8), np.ones((5, 5), np.uint8))
        out = out * (1 - gone)
    return out


# --------------------------------------------------------------------------- mask

DARK_GUARD = (0.30, 0.46)   # of the face's own skin brightness: round the face the mask fades out between these
DARK_ZONE = 1.7             # face widths round the face where the guard works (not on a leg or an arm in shadow)


def _dark_guard(m: np.ndarray, rgb_u8: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Hair lying over the cheek or beside the neck, a black top under it: far darker than the face's skin, and the
    colour model lets it through when its hue is the skin's (a log clip seen in Rec.709, dark brown hair). Round
    the face the mask is taken out where the picture is much darker than the skin the mask sits on."""
    h, w = m.shape
    Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
    c = lm[:17, :2][np.isfinite(lm[:17, :2]).all(1)].mean(0) if np.isfinite(lm[:17, :2]).all(1).any() else None
    if c is None or Wf < 8:
        return m
    Y = rgb_u8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    zone = np.zeros((h, w), np.uint8)
    cv2.circle(zone, (int(round(c[0])), int(round(c[1]))), int(DARK_ZONE * Wf), 255, -1)
    zf = cv2.GaussianBlur(zone.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.15 * Wf))
    sel = (zone > 0) & (m > 0.6)
    if sel.sum() < 40:
        return m
    yref = float(np.median(Y[sel]))
    Ys = cv2.GaussianBlur(Y, (0, 0), max(1.0, 0.012 * Wf))
    g = _sstep(DARK_GUARD[0] * yref, DARK_GUARD[1] * yref, Ys)
    return (m * (1.0 - zf * (1.0 - g))).astype(np.float32)


def skin_mask(job: SkinJob, rgb_proxy: np.ndarray, sx: float, sy: float, person_alpha: Optional[np.ndarray],
              model, parse: Optional[dict] = None, body_region: Optional[np.ndarray] = None,
              cover: Optional[np.ndarray] = None, prob0: Optional[np.ndarray] = None) -> np.ndarray:
    """float32 0..1 at proxy resolution. prob0: the face's skin probability the caller already has for this very
    face, model and parse (the engine's frame state): not computed a second time. body_region: the chosen body regions (neck, shoulders, arms, hands,
    legs) of this person, 0..1, added to the face. cover: an arm, leg or hand in front of the face (taken out)."""
    h, w = rgb_proxy.shape[:2]
    lm = None
    if job.face is not None:
        lm = job.face.copy()
        lm[:, 0] *= sx
        lm[:, 1] *= sy
    prob = prob0 if prob0 is not None else face_skin_probability(rgb_proxy, model, lm if job.face_w > 0.3 else None,
                                                                 parse if job.face_w > 0.3 else None)
    if cover is not None:
        prob = prob * (1.0 - cover)
    from ..core.project import scope_set
    regions = scope_set(job.scope)
    region = np.zeros((h, w), np.float32)
    if "face" in regions:
        poly = face_polygon(lm) if lm is not None else None
        if poly is not None:
            reg = np.zeros((h, w), np.uint8)
            cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 255)
            Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
            k = max(3, int(0.08 * Wf) | 1)
            region = cv2.GaussianBlur(reg.astype(np.float32) / 255.0, (k, k), 0) * job.face_w
            # the forehead: the arc above is a guess from the head's tilt and stops low on faces seen from
            # below or upside down (cuts 046, 049, 051, 052, 058). Where the face parser is trusted it sees
            # forehead skin and hair apart, so up to a full forehead its skin is taken
            if parse is not None and parse.get("trust", 0) > 0 and job.face_w > 0.3:
                tall = face_polygon(lm, FOREHEAD_PARSED)
                if tall is not None:
                    rt = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(rt, [np.round(tall).astype(np.int32)], 255)
                    ps = np.clip((parse["skin"].astype(np.float32) - 0.4) / 0.3, 0, 1) * float(parse["trust"])
                    ext = cv2.GaussianBlur(rt.astype(np.float32) / 255.0 * ps, (k, k), 0) * job.face_w
                    region = np.maximum(region, ext)
                    if PARSE_REACH > 0:
                        # a face seen from below or turned far: the outline from the landmarks sits beside the real
                        # face and half of it stayed outside (cut 112: forehead and one cheek untouched). The parser
                        # sees the face itself; its sure skin counts up to a quarter of a face width past the outline
                        kr = max(3, int(PARSE_REACH * Wf) | 1)
                        wide = cv2.dilate(rt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr)))
                        ps2 = np.clip((parse["skin"].astype(np.float32) - 0.6) / 0.25, 0, 1) * float(parse["trust"])
                        region = np.maximum(region, cv2.GaussianBlur(wide.astype(np.float32) / 255.0 * ps2, (k, k), 0) * job.face_w)
        # the nose and the skin between it and the upper lip: on a face in profile or turned well away the
        # nose stands out past the face outline and was left untouched (cut 037: a smooth face, a raw nose)
        if lm is not None and np.isfinite(lm[27:36, :2]).all() and np.isfinite(lm[48:55, :2]).all():
            Wf2 = max(8.0, float(np.hypot(*(lm[16, :2] - lm[0, :2]))), float(np.hypot(*(lm[8, :2] - lm[27, :2]))))
            nm = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(nm, cv2.convexHull(np.round(np.vstack([lm[27:36, :2], lm[48:55, :2]])).astype(np.int32)), 255)
            kn = max(3, int(0.05 * Wf2) | 1)
            nm = cv2.dilate(nm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kn, kn)))
            region = np.maximum(region, cv2.GaussianBlur(nm.astype(np.float32) / 255.0, (kn, kn), 0) * job.face_w)
    if body_region is not None:
        region = np.maximum(region, body_region)
    if region.max() <= 0.0:
        return np.zeros((h, w), np.float32)
    if person_alpha is not None:
        a_ = person_alpha
        if PARSE_OWN and parse is not None and parse.get("trust", 0) > 0 and lm is not None and job.face_w > 0.3:
            # the person's matte can miss part of its own face (cut 112, seen from below: half the face was outside
            # the matte and stayed raw). Inside the face's own outline, what the parser is sure is skin belongs to it
            try:
                own = face_polygon(lm, FOREHEAD_PARSED)
                if own is not None:
                    oz = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(oz, [np.round(own).astype(np.int32)], 255)
                    Wo = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
                    oz = cv2.GaussianBlur(oz.astype(np.float32) / 255.0, (0, 0), max(1.0, 0.02 * Wo))
                    sure = np.clip((parse["skin"].astype(np.float32) - 0.6) / 0.25, 0, 1) * float(parse["trust"])
                    a_ = np.maximum(a_, oz * sure)
            except Exception:
                pass
        region = region * a_
    m = prob * region
    if lm is not None and np.isfinite(lm[:, :2]).any():
        feat = _features_mask(lm, (h, w))
        Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2]))) if np.isfinite(lm[[0, 16], :2]).all() else 20.0
        k = max(3, int(FEAT_SOFT * Wf) | 1)
        featf = cv2.GaussianBlur(feat.astype(np.float32) / 255.0, (k, k), 0)
        featf = np.clip(featf * FEAT_GAIN, 0, 1)
        m *= 1.0 - featf
    elif job.head is not None:
        # no face landmarks (face turned away or too small): leave the whole head alone
        hd = job.head.copy()
        hd[:, 0] *= sx
        hd[:, 1] *= sy
        ok = np.isfinite(hd).all(1)
        if ok.sum() >= 2:
            c = hd[ok].mean(0)
            spread = float(np.max(np.linalg.norm(hd[ok] - c, axis=1)))
            rad = max(spread * 1.9, 0.06 * max(job.scale, 200.0) * sx)
            hm = np.zeros((h, w), np.uint8)
            cv2.circle(hm, (int(round(c[0])), int(round(c[1]))), int(rad), 255, -1)
            k = max(3, int(rad * 0.3) | 1)
            m *= 1.0 - cv2.GaussianBlur(hm.astype(np.float32) / 255.0, (k, k), 0)
    if DARK_GUARD is not None and lm is not None and np.isfinite(lm[[0, 16], :2]).all():
        m = _dark_guard(m, rgb_proxy, lm)
    # edge softening follows the face size (a fixed pixel blur washed a small face's mask out: during a zoom
    # the retouch faded in instead of holding its strength)
    sig = max(0.8, 0.0025 * h)
    if lm is not None and np.isfinite(lm[[0, 16], :2]).all():
        Wf_ = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
        sig = max(0.6, min(sig, 0.008 * Wf_))
    m = cv2.GaussianBlur(m, (0, 0), sig)
    return np.clip(m, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- filter

TEX_MODE = True                    # False: the two wide passes of before v3
TEX_W = (0.79, 1.21, 1.07)        # share of each pass's texture taken out, per unit of the skin slider (70 = the reference)
TEX_R = (0.004, 0.011, 0.026)     # radius of each pass, of the face width
TEX_EPS = (0.004, 0.006, 0.006)   # edge threshold of each pass (at mid brightness; follows the local brightness)
TEX_GRAIN_KEEP = 0.5               # share of the grain band given back; fades out on the largest faces
TEX_GRAIN_PX = 1.8                 # px: what is finer than this is the picture's grain
PIT = 0.0                         # 0 = off. Acne-scar like pits (user, 2026-11-14, cut 069): what is left of them after the passes is filled, softly
PIT_S = (0.0023, 0.0056)          # of the face width: the pit scale (fine blur, wide blur); a pit is darker than its surroundings at this scale
PIT_C = (1.0 / 255, 6.0 / 255, 6.0 / 255)   # contrast: from where it counts, from where painted marks (strong contrast) fade out, over how much
TEX_SHAPE = (1.0, 0.35, 0.0)      # share of each pass left where the engine marks shape (the nose)
# v3b (user, 2026-11-14, cut 069: "the texture is not alive, and it is not a question of strength"). Fitted band by band
# on the hand retouch of 069 (frame 01:01:56:19): the retouch scales every band of the skin by about the same factor
# everywhere on the skin (no edge-keeping threshold): the finest grain is kept (even a little crisper), the bands of
# pores, pits and bumps are taken down to about a third, wider ones less, shading kept. Linear: no threshold that can
# switch a pit in or out from one frame to the next; the band widths follow the face width without steps
TEX_FS = True
FS_S = (0.0006, 0.0012, 0.0024, 0.0048, 0.0096, 0.0192)   # band upper edges (Gaussian sigma), of the face width
FS_G = (1.35, 0.70, 0.36, 0.33, 0.45, 0.80)               # gain of each band at the reference strength
FS_SHAPE = (1.0, 1.0, 1.0, 0.7, 0.42, 0.6)                # share of each band's change left where the engine marks shape (nose)
FS_NORM = 0.0     # > 0: the bands are taken with blurs of the skin alone (normalised by the skin area's own weight, full from this value of the
                  # mask up): the background, hair or a lip beside the skin is not averaged in, so the skin along the face's outline is not darkened
                  # (or lightened) by the step to what lies beyond it (2611 IWALY 11/24: a dark band along the cheek's contour, ENDING f26, INT_c006 f90).
                  # 0 = the plain blurs, as before
FS_REF = 0.70                                              # skin slider of the reference
# keep the picture's own grain (user 2026-11-14 22:51): in every band the part of each pixel's value up to FS_GRAIN times
# that band's grain level is left as it is; only what stands out above the grain (pores, pits, bumps) is scaled.
# Grain level of a band = its typical small value on the skin of this frame (25th percentile of |band| / 0.3186, which is
# sigma for Gaussian grain). Clean footage: grain is low and nearly everything is scaled as before; noisy or dark
# footage: the grain stays and only the skin structure above it is taken down.
FS_GRAIN = 2.0
FS_GRAIN_PX = 2.0          # only bands up to this Gaussian width in pixels carry grain (pores and pits are wider on a big face)
# small face (user 2026-11-15): on a face only a few hundred pixels wide the pores are as fine as the grain, so the wide
# grain range kept the pores too and the retouch came out weaker than v3. The range narrows with the face width, without
# a step: FS_GRAIN_PX_SMALL px up to FS_GRAIN_WF[0] face width, FS_GRAIN_PX from FS_GRAIN_WF[1] on
DBG_SN = None              # a list: the grain level of each band is appended (diagnosis only)
FS_GRAIN_PX_SMALL = 1.0
FS_GRAIN_WF = (350.0, 480.0)
# follow the sharpness (user 2026-11-15, cut 069 56:07..56:12: the source goes soft and sharp again and the retouch showed it).
# The engine passes sq = this frame's skin sharpness / the sharpness round it (1 = as usual). The bands from FS_SQ_PX up
# that the retouch takes down keep a share of 1/sq of what is left: a sharper frame is thinned more, a softer one less,
# so the skin that comes out has the same structure on both. Never more than the source has (the gain stays below 1)
FS_SQ = True
FS_SQ_PX = 0.7
FS_SQ_DEAD = 0.05          # a change in sharpness below this is not followed (the measure's own noise)
FS_SQ_CLIP = (0.4, 1.0)
import threading as _threading
_TLS = _threading.local()         # .shape: 0 on shape .. 1 elsewhere, set by the engine for the next luma smoothing of this thread


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_self(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    m = _box(I, r)
    var = _box(I * I, r) - m * m
    a = var / (var + eps)
    b = m - a * m
    return _box(a, r) * I + _box(b, r)


# lines lying on the skin (face paint, a tattoo, a hair strand): the retouch took the contrast out of them (cut 053, the
# paint on the cheek; cut 018, the fringe over the forehead). Found by the picture's own second derivative at a few
# pixels' width: one direction strongly curved (the line's cross section), the other not (along it). A pore, a pit or a
# scar's blob is curved both ways and is not held. The band scaling is held back on them for the bands from FS_LINE_BAND on
FS_LINE = False   # off: by luma alone it also held skin pits and scars (the 069 reference went from 0.60 to 0.63 at best) and barely helped paint or fair hair
FS_LINE_CON = (0.10, 0.18)    # peak contrast of the line over the skin's level: from where it counts .. fully a line
FS_LINE_ANI = (0.55, 0.80)      # 1 - (curvature along) / (curvature across)
FS_LINE_LEN = 0.08              # of the face width: shorter than this is no line
FS_LINE_BAND = 1                # index of the first band that is held
FS_LINE_KEEP = 0.85             # share of the band's change taken away at a full line


def _line_hold(roi: np.ndarray, wf: float) -> Optional[np.ndarray]:
    try:
        sg = max(1.2, 0.0040 * wf)
        g = cv2.GaussianBlur(roi, (0, 0), sg)
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        tr = 0.5 * (gxx + gyy)
        dt = np.sqrt(0.25 * (gxx - gyy) ** 2 + gxy ** 2)
        l1, l2 = tr + dt, tr - dt                       # l1 >= l2
        bright = -l2 > l1                               # a bright line is a ridge: l2 strongly negative
        pm = np.where(bright, -l2, l1) * (sg * sg)
        other = np.where(bright, np.maximum(0.0, -l1), np.maximum(0.0, l2)) * (sg * sg)
        ani = 1.0 - np.clip(other / np.maximum(pm, 1e-6), 0.0, 1.0)
        lvl = cv2.GaussianBlur(roi, (0, 0), max(2.0, 0.06 * wf)) + 0.03
        rel = pm / lvl
        t = _sstep(FS_LINE_CON[0], FS_LINE_CON[1], rel) * _sstep(FS_LINE_ANI[0], FS_LINE_ANI[1], ani)
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats((t > 0.3).astype(np.uint8), connectivity=8)
        if n_ <= 1:
            return None
        ln_ = np.hypot(st_[:, cv2.CC_STAT_WIDTH], st_[:, cv2.CC_STAT_HEIGHT])
        keep_ = ln_ >= FS_LINE_LEN * wf
        keep_[0] = False
        t = (t * keep_[lab_]).astype(np.float32)
        if not t.any():
            return None
        kd = max(3, int(round(0.012 * wf)) | 1)
        t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        return np.clip(cv2.GaussianBlur(t, (0, 0), max(1.0, 0.005 * wf)) * 1.3, 0.0, 1.0)
    except Exception:
        return None


def _sstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_channel(I: np.ndarray, mask: np.ndarray, strength: float, keep: float, r: int,
                   eps: float = 0.0012, relative: bool = False) -> np.ndarray:
    """I float32 (0..1-ish), mask same size. Returns the smoothed channel (only where mask > 0).
    relative: the edge threshold follows the skin's own brightness (luma), so a dark grade and a flash
    frame of the same face are smoothed alike."""
    sel = mask > 0.004
    ys = np.flatnonzero(sel.any(1))         # (the rows and columns with something: listing every pixel of the mask
    if not len(ys):                         # took longer than the smoothing of a small area on a 4K plane)
        return I
    xs = np.flatnonzero(sel.any(0))
    pad = 3 * r + 4
    y0, y1 = max(0, int(ys[0]) - pad), min(I.shape[0], int(ys[-1]) + pad + 1)
    x0, x1 = max(0, int(xs[0]) - pad), min(I.shape[1], int(xs[-1]) + pad + 1)
    roi = I[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1]
    if relative:
        # per pixel, from the skin's own local brightness: one level for the whole face let the lit side and
        # the side in shadow be smoothed differently, and the strength changed as the light moved over the
        # face (a zoom through a light change): now the same slider does the same everywhere
        loc = cv2.GaussianBlur(roi, (0, 0), max(2.0, 3.0 * r))
        eps = eps * np.clip(loc / 0.45, 0.15, 2.0) ** 2
    # above the middle of the slider the edge threshold rises too: strong retouches (hard light, a beauty
    # reference that leaves about half of the texture) reach further into contrasty skin. Below 50 % nothing
    # changes, so earlier settings and presets look the same.
    if relative and TEX_FS:
        wfc = getattr(_TLS, "wf", None)
        _TLS.wf = None
        wf_ = float(wfc) if wfc else float(r) / 0.022
        shp = getattr(_TLS, "shape", None)
        _TLS.shape = None
        _TLS.pitm = None
        if shp is not None:
            shp = shp[y0:y1, x0:x1] if shp.shape == I.shape else None
        k_ = float(strength) / FS_REF
        sq_ = getattr(_TLS, "sq", None)
        _TLS.sq = None
        if not FS_SQ or sq_ is None or not np.isfinite(sq_):
            sq_ = 1.0
        sq_ = float(np.clip(sq_, FS_SQ_CLIP[0], FS_SQ_CLIP[1]))
        d_ = abs(sq_ - 1.0)
        sq_ = 1.0 + np.sign(sq_ - 1.0) * max(d_ - FS_SQ_DEAD, 0.0)
        gpx_ = FS_GRAIN_PX_SMALL + (FS_GRAIN_PX - FS_GRAIN_PX_SMALL) * float(
            np.clip((wf_ - FS_GRAIN_WF[0]) / (FS_GRAIN_WF[1] - FS_GRAIN_WF[0]), 0.0, 1.0))
        lh_ = _line_hold(roi, wf_) if FS_LINE else None
        prev = roi
        ps = 0.0
        acc = np.zeros_like(roi)
        wsup = np.clip(m / FS_NORM, 0.0, 1.0).astype(np.float32) if FS_NORM > 0 else None
        for i_, (fs_, g_) in enumerate(zip(FS_S, FS_G)):
            s_ = fs_ * wf_
            if s_ < ps + 0.15:
                continue
            sg_ = float(np.sqrt(s_ * s_ - ps * ps))
            if wsup is not None:
                # the blur of the skin alone: what lies beyond the skin area does not enter the skin's bands
                den_ = cv2.GaussianBlur(wsup, (0, 0), sg_)
                cur = np.where(den_ > 1e-3, cv2.GaussianBlur(prev * wsup, (0, 0), sg_) / np.maximum(den_, 1e-3), prev).astype(np.float32)
            else:
                cur = cv2.GaussianBlur(prev, (0, 0), sg_)
            b_ = prev - cur
            prev = cur
            ps = s_
            if g_ > 1.0:
                # crisper only where the band is wider than the picture's own pixel grain
                g_ = 1.0 + (g_ - 1.0) * float(np.clip((s_ - 0.45) / 0.35, 0.0, 1.0))
            if g_ < 1.0 and s_ >= FS_SQ_PX and sq_ != 1.0:
                # what is left of the band follows the sharpness: sharper frame, thinner; softer frame, left more
                f_ = -float(min(k_ * (1.0 - g_) * sq_, 1.0))
            else:
                f_ = k_ * (g_ - 1.0)
            if FS_GRAIN > 0 and s_ <= gpx_:
                sel_ = m > 0.5
                ab_ = np.abs(b_[sel_]) if sel_.any() else np.abs(b_).ravel()
                if ab_.size > 400:
                    sn_ = float(np.percentile(ab_[:: max(1, ab_.size // 200000)], 25)) / 0.3186
                    if DBG_SN is not None:
                        DBG_SN.append((i_, sn_))
                    lim_ = FS_GRAIN * sn_
                    if lim_ > 0:
                        b_ = b_ - b_ * np.minimum(1.0, lim_ / np.maximum(np.abs(b_), 1e-9))   # the part above the grain
            if lh_ is not None and i_ >= FS_LINE_BAND:
                b_ = b_ * (1.0 - FS_LINE_KEEP * lh_)
            if shp is not None and FS_SHAPE[i_] < 1.0:
                acc += f_ * (FS_SHAPE[i_] + (1.0 - FS_SHAPE[i_]) * shp) * b_
            else:
                acc += f_ * b_
        out = I.astype(np.float32, copy=True)
        out[y0:y1, x0:x1] = roi + m * acc
        return out
    if relative and TEX_MODE:
        # v3. Three narrow edge-keeping passes instead of two wide ones, each taking its own share of the texture
        # (pores; the grain between them; blotches up to a few pores wide). Fitted on the hand-retouched reference
        # of cut 069: there the texture is thinned evenly over the whole face, right up to the eyes, the nose's
        # edge and the lips, and nothing wider than a few pores is changed. The wide passes left a band untouched
        # around every feature (their window saw the feature and held back) and flattened the cheeks more than
        # the reference, which is what read as "blurred here, raw there"; and being wide they moved the nose's
        # shading and the edges of shadows, which then had to be protected by masks that flickered
        wf_ = float(r) / 0.022
        shp = getattr(_TLS, "shape", None)
        _TLS.shape = None
        if shp is not None:
            shp = shp[y0:y1, x0:x1] if shp.shape == I.shape else None
        pitm = getattr(_TLS, "pitm", None)
        _TLS.pitm = None
        if pitm is not None:
            pitm = pitm[y0:y1, x0:x1] if pitm.shape == I.shape else None
        res = roi
        prev = roi
        rp = 0
        acc = np.zeros_like(roi)
        for i_, (wi, ri, ei) in enumerate(zip(TEX_W, TEX_R, TEX_EPS)):
            r_ = max(rp + 1, int(round(ri * wf_)))
            rp = r_
            b_ = guided_self(prev, r_, ei * (eps / 0.0012))
            w_ = float(np.clip(wi * strength * ((1.0 - keep) if i_ == 0 else 1.0), 0.0, 1.0))
            if shp is not None and TEX_SHAPE[i_] < 1.0:
                # on the nose the wider passes are held back: its ridge light and the shade of its sides are as
                # soft and as wide as a blotch, and a smooth face lost them (cut 024: a flat, blurred nose)
                acc += w_ * (TEX_SHAPE[i_] + (1.0 - TEX_SHAPE[i_]) * shp) * (b_ - prev)
            else:
                acc += w_ * (b_ - prev)
            prev = b_
        if TEX_GRAIN_KEEP > 0:
            # the picture's own grain (about a pixel wide whatever the face's size) is not skin texture: part of what
            # the passes took from that band is given back, or the face comes out cleaner than the picture around it
            # (a soft patch on grainy footage). On the largest faces that band is pores, and it is left as fitted
            # (the reference keeps its finest band as strong as the next one)
            r1_ = max(1, int(round(TEX_R[0] * wf_)))
            gk_ = TEX_GRAIN_KEEP * float(np.clip((6.0 - r1_) / 4.0, 0.0, 1.0))
            if gk_ > 1e-3:
                acc = acc - gk_ * (acc - cv2.GaussianBlur(acc, (0, 0), TEX_GRAIN_PX))
        if PIT > 0:
            # what the edge-keeping passes left of the pits: a dark spot of pit size, low in contrast. Soft in both
            # directions (no threshold that can switch from frame to frame); strong marks (face paint, moles) are above the cap
            cur = roi + (m if pitm is None else np.maximum(m, pitm)) * acc
            s0_ = max(0.6, PIT_S[0] * wf_); s1_ = max(1.2, PIT_S[1] * wf_)
            Bp = np.clip(cv2.GaussianBlur(cur, (0, 0), s1_) - cv2.GaussianBlur(cur, (0, 0), s0_), 0.0, None)
            bs = np.clip(loc / 0.45, 0.3, 2.0)
            g_ = np.clip(Bp / (PIT_C[0] * bs), 0.0, 1.0) * (1.0 - np.clip((Bp - PIT_C[1] * bs) / (PIT_C[2] * bs), 0.0, 1.0))
            pit_add = PIT * 1.5 * g_ * Bp
            if pitm is not None:
                # the dark-spot hold of the main smoothing keeps pits out of `m`; the fill has its own mask
                out = I.astype(np.float32, copy=True)
                out[y0:y1, x0:x1] = roi + m * acc + pitm * pit_add
                return out
            acc = acc + pit_add
        out = I.astype(np.float32, copy=True)
        out[y0:y1, x0:x1] = roi + m * acc
        return out
    eps = eps * (1.0 + 14.0 * max(0.0, float(strength) - 0.5) ** 2)
    base = guided_self(roi, r, eps)
    tone = guided_self(base, 2 * r, eps * 1.6)
    target = tone + (roi - base) * keep
    res = roi + m * strength * (target - roi)
    out = I.astype(np.float32, copy=True)
    out[y0:y1, x0:x1] = res
    return out


def radius_px(job: SkinJob, res_scale: float) -> int:
    """Filter radius in pixels of the image being processed (res_scale = image px per source px)."""
    if job.face is not None and np.isfinite(job.face[[0, 16], :2]).all():
        Wf = float(np.hypot(*(job.face[16, :2] - job.face[0, :2]))) * res_scale
        return int(max(1, round(0.022 * Wf)))
    return int(max(1, round(0.006 * max(job.scale, 200.0) * res_scale)))


# --------------------------------------------------------------------------- per-person helpers on the tracker

def person_alpha(tracker, t: int, p: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
    a = tracker.matte.get(t)
    lab = tracker.labels(t)
    if a is None or lab is None:
        return None
    m = (np.asarray(a).astype(np.float32) / 255.0) * (lab == p)
    return cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)


_MODEL_LOCK = threading.RLock()     # the people of a frame are prepared side by side (engine.PREP_THREADS): the models kept on the tracker are made
                                    # and let go of by one at a time


def model_for(tracker, p: int):
    with _MODEL_LOCK:
        return _model_for(tracker, p)


def _model_for(tracker, p: int):
    cache = getattr(tracker, "_skin_models", None)
    if cache is None:
        cache = tracker._skin_models = {}
    if p in cache:
        return cache[p]
    st = tracker.store
    px = tracker.proxy
    frames, faces = [], []
    if st is not None and st.has_face:
        cand = np.nonzero(st.face_vis[p] > 0.5)[0]
        if len(cand):
            pick = cand[np.linspace(0, len(cand) - 1, min(10, len(cand))).astype(int)]
            for t in pick:
                lm = st.face[p, t].copy()
                if not np.isfinite(lm[:, :2]).any():
                    continue
                lm[:, 0] *= px.sx
                lm[:, 1] *= px.sy
                frames.append(np.asarray(px.frame(int(t))))
                faces.append(lm)
    cache[p] = build_skin_model(frames, faces) if frames else None
    return cache[p]


def frame_model_for(tracker, p: int, t: int, rgb_u8: np.ndarray, lm_px: Optional[np.ndarray],
                    centre: Optional[np.ndarray] = None):
    """The person's skin colour model, re-centred on this frame's own face skin when the face is in view.

    Under strobes, flashes or coloured light sweeping over the set, the skin's colour changes from frame to
    frame; a model learned once for the shot would drop the skin on those frames (the retouch blinks on and
    off with the light). The spread stays the shot's own, only the centre follows the light."""
    base = model_for(tracker, p)
    if centre is not None:
        # the face layer's centre: measured on the parser's skin and held steady over the shot
        icov = base[1] if base is not None else DEFAULT_MODEL[1]
        return (np.asarray(centre[:2], np.float64), icov, float(centre[2]))
    if lm_px is None or not np.isfinite(lm_px[:, :2]).all():
        return base
    with _MODEL_LOCK:
        cache = getattr(tracker, "_skin_fmodels", None)
        if cache is None:
            cache = tracker._skin_fmodels = {}
        key = (p, t)
        if key in cache:
            return cache[key]
    fm = build_skin_model([rgb_u8], [lm_px])
    if fm is None:
        res = base
    else:
        icov = base[1] if base is not None else fm[1]
        # never tighter than this frame's own spread
        if base is not None:
            c_base = np.linalg.inv(base[1])
            c_fr = np.linalg.inv(fm[1])
            grow = max(1.0, float(np.trace(c_fr) / max(np.trace(c_base), 1e-6)))
            icov = np.linalg.inv(c_base * grow)
        res = (fm[0], icov, fm[2])
    with _MODEL_LOCK:
        cache[key] = res
        while len(cache) > 600:
            try:
                cache.pop(next(iter(cache)))
            except (StopIteration, KeyError, RuntimeError):
                break
    return res


def masks_for_frame(cut, tracker, t: int) -> List[Tuple[SkinJob, np.ndarray]]:
    jobs = skin_jobs(cut, tracker, t)
    if not jobs:
        return []
    px = tracker.proxy
    rgb = np.asarray(px.frame(t))
    out = []
    for job in jobs:
        alpha = person_alpha(tracker, t, job.p, (px.w, px.h))
        lm = None
        if job.face is not None:
            lm = job.face.copy()
            lm[:, 0] *= px.sx
            lm[:, 1] *= px.sy
        m = skin_mask(job, rgb, px.sx, px.sy, alpha, frame_model_for(tracker, job.p, t, rgb, lm))
        out.append((job, m))
    return out


def apply_rgb(img: np.ndarray, masks: List[Tuple[SkinJob, np.ndarray]], res_scale: float) -> np.ndarray:
    """Preview: uint8 RGB at proxy resolution."""
    if not masks:
        return img
    f = img.astype(np.float32) / 255.0
    for job, m in masks:
        r = radius_px(job, res_scale)
        for c in range(3):
            f[..., c] = smooth_channel(f[..., c], m, job.strength, job.keep, r, relative=True)
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_planes(planes: List[np.ndarray], subsample, max_value: float, is_yuv: bool,
                 masks: List[Tuple[SkinJob, np.ndarray]], res_scale: float) -> List[np.ndarray]:
    """Render: native planes. YUV: luma at full strength, chroma gently. RGB: every channel."""
    if not masks:
        return planes
    out = [np.array(p, copy=True) for p in planes]
    for job, m_proxy in masks:
        for i, (pl, (ssx, ssy)) in enumerate(zip(out, subsample)):
            h, w = pl.shape[:2]
            m = cv2.resize(m_proxy, (w, h), interpolation=cv2.INTER_LINEAR)
            r = max(1, int(round(radius_px(job, res_scale) / ssx)))
            I = pl.astype(np.float32) / max_value if np.issubdtype(pl.dtype, np.integer) else pl.astype(np.float32)
            if is_yuv and i > 0:
                res = smooth_channel(I, m, job.strength * 0.55, 0.0, r, eps=0.0004)
            else:
                res = smooth_channel(I, m, job.strength, job.keep, r, relative=True)
            if np.issubdtype(pl.dtype, np.integer):
                out[i] = np.clip(np.rint(res * max_value), 0, max_value).astype(pl.dtype)
            else:
                out[i] = res.astype(pl.dtype)
    return out
