"""Skin smoothing that never touches eyes, brows, lips, nostrils or hair.

Mask  = skin colour probability (learned per person from their own face)
        x the person's matte (only the selected person)
        x face area (scope 'face') or the whole body (scope 'all')
        x NOT (eyes, lashes, brows, lips, nostrils) built from the tracked 68 face landmarks.
Filter = edge-preserving guided filter: the large-scale tone is evened out, a chosen share of the fine
        texture (pores) is put back so skin does not turn plastic. Works on luma and, more gently, chroma.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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


def face_polygon(lm: np.ndarray) -> Optional[np.ndarray]:
    """Jaw line plus a forehead arc (brows lifted by a third of the face width)."""
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
    lift = 0.33 * float(np.clip((span - 0.45) / 0.35, 0.25, 1.0))
    forehead = brows[::-1] + up * lift * Wf
    return np.vstack([jaw, forehead]).astype(np.float32)


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
        cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (max(2, int(ew * 0.78)), max(2, int(ew * 0.50))),
                    ang, 0, 360, 255, -1)
    for idx in (BROW_R, BROW_L):
        b = pts[idx]
        if not np.isfinite(b).all():
            continue
        th = max(2, int(0.075 * Wf)) if Wf else 3
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
        cv2.ellipse(m, (int(round(c[0])), int(round(c[1]))), (max(1, int(nw * 0.66)), max(1, int(nw * 0.36))),
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
    X = X[d < np.percentile(d, 85)]
    mean = X.mean(0)
    # never tighter than a few code values: graded footage (one coloured light on everything) has very little
    # chroma spread, and a tight model turns 8-bit steps into a blotchy mask
    cov = np.cov(X.T) + np.eye(2) * 12.0
    return mean, np.linalg.inv(cov), yref


DEFAULT_MODEL = (np.array([150.0, 112.0]), np.linalg.inv(np.diag([90.0, 70.0])))


def skin_probability(rgb_u8: np.ndarray, model) -> np.ndarray:
    m = model if model is not None else DEFAULT_MODEL
    mean, icov = m[0], m[1]
    yref = m[2] if len(m) > 2 else None
    ycc = _ycrcb(rgb_u8).astype(np.float32)
    # chroma a little smoothed: 8-bit chroma steps must not turn into a blotchy mask
    sig = max(0.8, 0.0012 * rgb_u8.shape[0])
    cr = cv2.GaussianBlur(ycc[..., 1], (0, 0), sig)
    cb = cv2.GaussianBlur(ycc[..., 2], (0, 0), sig)
    d0, d1 = cr - mean[0], cb - mean[1]
    m2 = icov[0, 0] * d0 * d0 + 2 * icov[0, 1] * d0 * d1 + icov[1, 1] * d1 * d1
    md = np.sqrt(np.maximum(m2, 0))
    prob = np.exp(-0.5 * np.maximum(md - 1.6, 0) ** 2)
    y = ycc[..., 0]
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
    ref = []
    for i, j in ((1, 31), (15, 35), (28, 29)):
        c = (L[i] + L[j]) / 2 - [x0, y0]
        r = max(2, int(0.05 * Wf))
        cy, cx = int(c[1]), int(c[0])
        if 0 <= cy < E.shape[0] and 0 <= cx < E.shape[1]:
            ref.append(E[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel())
    if not ref:
        return None
    e0 = float(np.median(np.concatenate(ref)))
    e0 = max(e0, 0.6)                                  # never finer than 8-bit steps
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
    sc = np.maximum(r_y / (np.percentile(r_y[ref], 90) + 1e-4), r_b / (np.percentile(r_b[ref], 90) + 0.5))
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
    g = np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, 0.012 * Wf)) * 1.7, 0, 1)
    cov = cv2.GaussianBlur(parse["cover"], (0, 0), max(1.0, 0.03 * Wf))
    return (1.0 - cov * parse["trust"] * (1.0 - g) * 0.97).astype(np.float32)


def face_skin_probability(rgb_u8: np.ndarray, model, lm: Optional[np.ndarray], parse: Optional[dict] = None) -> np.ndarray:
    gate = None
    if parse is not None and lm is not None and np.isfinite(lm[[0, 16], :2]).all():
        gate = parse_gate(parse, float(np.hypot(*(lm[16, :2] - lm[0, :2]))))
    out = _face_skin_probability(rgb_u8, model, lm, gate, parse.get("tex") if parse is not None else None)
    if gate is not None:
        out = out * gate
    # face paint, tattoos, drawn veins: never retouched
    pm = paint_mask(rgb_u8, lm) if lm is not None else None
    if pm is not None:
        out = out * (1.0 - pm)
    return out.astype(np.float32)


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


# --------------------------------------------------------------------------- mask

def skin_mask(job: SkinJob, rgb_proxy: np.ndarray, sx: float, sy: float, person_alpha: Optional[np.ndarray],
              model, parse: Optional[dict] = None, body_region: Optional[np.ndarray] = None) -> np.ndarray:
    """float32 0..1 at proxy resolution. body_region: the chosen body regions (neck, shoulders, arms, hands,
    legs) of this person, 0..1, added to the face."""
    h, w = rgb_proxy.shape[:2]
    lm = None
    if job.face is not None:
        lm = job.face.copy()
        lm[:, 0] *= sx
        lm[:, 1] *= sy
    prob = face_skin_probability(rgb_proxy, model, lm if job.face_w > 0.3 else None,
                                 parse if job.face_w > 0.3 else None)
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
    if body_region is not None:
        region = np.maximum(region, body_region)
    if region.max() <= 0.0:
        return np.zeros((h, w), np.float32)
    if person_alpha is not None:
        region = region * person_alpha
    m = prob * region
    if lm is not None and np.isfinite(lm[:, :2]).any():
        feat = _features_mask(lm, (h, w))
        Wf = float(np.hypot(*(lm[16, :2] - lm[0, :2]))) if np.isfinite(lm[[0, 16], :2]).all() else 20.0
        k = max(3, int(0.05 * Wf) | 1)
        featf = cv2.GaussianBlur(feat.astype(np.float32) / 255.0, (k, k), 0)
        featf = np.clip(featf * 1.6, 0, 1)
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
    # edge softening follows the face size (a fixed pixel blur washed a small face's mask out: during a zoom
    # the retouch faded in instead of holding its strength)
    sig = max(0.8, 0.0025 * h)
    if lm is not None and np.isfinite(lm[[0, 16], :2]).all():
        Wf_ = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
        sig = max(0.6, min(sig, 0.008 * Wf_))
    m = cv2.GaussianBlur(m, (0, 0), sig)
    return np.clip(m, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- filter

def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_self(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    m = _box(I, r)
    var = _box(I * I, r) - m * m
    a = var / (var + eps)
    b = m - a * m
    return _box(a, r) * I + _box(b, r)


def smooth_channel(I: np.ndarray, mask: np.ndarray, strength: float, keep: float, r: int,
                   eps: float = 0.0012, relative: bool = False) -> np.ndarray:
    """I float32 (0..1-ish), mask same size. Returns the smoothed channel (only where mask > 0).
    relative: the edge threshold follows the skin's own brightness (luma), so a dark grade and a flash
    frame of the same face are smoothed alike."""
    ys, xs = np.nonzero(mask > 0.004)
    if not len(xs):
        return I
    pad = 3 * r + 4
    y0, y1 = max(0, ys.min() - pad), min(I.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(I.shape[1], xs.max() + pad + 1)
    roi = I[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1]
    if relative:
        sel = m > 0.3
        lvl = float(np.median(roi[sel])) if sel.sum() > 20 else float(np.median(roi))
        eps = eps * float(np.clip(lvl / 0.45, 0.15, 2.0)) ** 2
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


def model_for(tracker, p: int):
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
    cache[key] = res
    while len(cache) > 600:
        cache.pop(next(iter(cache)))
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
