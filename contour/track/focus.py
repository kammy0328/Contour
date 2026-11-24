"""Faces out of focus: people the shot is not about (2611 IWALY, 11/24: the dancers at the sides of c057, soft in the whole
shot, took analysis and render time like the one in focus, and their retouch is never seen).

On a few frames spread through the work range every face is cut out of the analysis picture by its landmarks (eyes, brows,
nose, mouth: little hair, no background), brought to one width (only ever made smaller, never enlarged) and measured: how
steep its steepest edges are for its contrast (measure()). That does not depend on the face's size; the grain is taken by a
slight blur first.

A person is called out of focus only when, on every part of the shot, they are clearly softer than the sharpest face of the
same frames and soft on their own measure too; one stretch in focus (a focus pull, someone walking into the plane) keeps them.
A face too small or seen on too few frames to tell is never called out of focus. The call is a suggestion the user can undo:
the person is protected (not retouched) with hold "defocus"."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

SAMPLES = 16          # frames looked at in the work range
NORM_W = 96.0         # faces wider than this (analysis px) are made this wide before the measure
MIN_W = 48.0          # faces narrower than this (analysis px) are not judged (small faces measure low in focus too: c034, c040)
MIN_SEEN = 4          # samples a person needs before they can be called out of focus
REL = 0.62            # out of focus: under this share of the sharpest face's measure on the same frames ...
ABS = 0.80            # ... and under this on their own
SHARP_REF = 0.95      # the sharpest face must be at least this sharp (a soft or blurred shot calls nobody out)
KEEP_Q = 85           # a person's measure: this percentile of their samples (one sharp stretch keeps them)
INNER = slice(17, 68) # 68-point face: brows, nose, eyes, mouth


def _box(lm68: np.ndarray, sx: float, sy: float, W: int, H: int):
    pts = lm68[INNER, :2]
    if not np.isfinite(pts).all() or not np.isfinite(lm68[:17, :2]).all():
        return None
    pts = pts * [sx, sy]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w = float(x1 - x0)
    h = float(y1 - y0)
    if w < 4 or h < 4:
        return None
    # landmarks that sit on the face: the inner points about as wide as tall, and as wide as the jaw (c040: a track whose points
    # spread over hair and a neighbour gave a "face" of the frame's width, sharp, the reference of everybody else)
    jw = float(np.ptp(lm68[:17, 0] * sx))
    if not (0.45 <= w / h <= 2.2) or not (0.5 <= w / max(jw, 1e-3) <= 1.15):
        return None
    m = 0.06 * w
    x0, y0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    x1, y1 = min(W, int(np.ceil(x1 + m))), min(H, int(np.ceil(y1 + m)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1, w


def measure(gray: np.ndarray, box) -> Optional[float]:
    """How steep the face's steepest edges are (eyes, brows, lips, nostrils) for its contrast: the 97th percentile of the gradient
    (after a slight blur that takes the grain) over the range of its brightness (3rd to 97th percentile), in face widths (the face
    brought to NORM_W). In focus 1 and more; far out of focus (the dancers at the sides of c057) about 0.6."""
    x0, y0, x1, y1, w = box
    if w < MIN_W:
        return None
    c = gray[y0:y1, x0:x1].astype(np.float32) / 255.0
    s = min(1.0, NORM_W / w)
    if s < 1.0:
        c = cv2.resize(c, (max(4, int(round(c.shape[1] * s))), max(4, int(round(c.shape[0] * s)))), interpolation=cv2.INTER_AREA)
    if min(c.shape) < 12:
        return None
    rng = float(np.percentile(c, 97) - np.percentile(c, 3))
    if rng < 0.04:                       # a face in the dark or washed out: nothing to tell by
        return None
    g = cv2.GaussianBlur(c, (0, 0), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    m = np.hypot(gx, gy)
    return float(np.percentile(m, 97) / rng * (w * s) / 10.0)


def landmarks(st, p: int, t: int) -> Optional[np.ndarray]:
    for arr in (getattr(st, "face", None), getattr(st, "face_raw", None)):
        if arr is None:
            continue
        lm = arr[p, t]
        if np.isfinite(lm[INNER, :2]).all() and (not np.isfinite(lm[:, 2]).any() or np.nanmedian(lm[INNER, 2]) >= 0.3):
            return lm
    return None


def scores(st, gray: Callable[[int], np.ndarray], sx: float, sy: float, rng: Tuple[int, int],
           samples: int = SAMPLES, persons: Optional[List[int]] = None, crops: Optional[list] = None) -> Dict[int, list]:
    """p -> [(t, measure)] on frames spread through rng. crops (a list, filled when given): (p, t, measure, crop) for a sheet."""
    a, b = int(rng[0]), int(rng[1])
    if b <= a:
        return {}
    ts = np.unique(np.linspace(a, b - 1, min(samples, b - a)).round().astype(int))
    P = len(st.pids)
    persons = list(range(P)) if persons is None else persons
    out: Dict[int, list] = {p: [] for p in persons}
    for t in ts:
        here = [p for p in persons if st.present[p, t]]
        lms = {p: landmarks(st, p, t) for p in here}
        here = [p for p in here if lms[p] is not None]
        if not here:
            continue
        g = np.asarray(gray(int(t)))
        H, W = g.shape[:2]
        for p in here:
            bx = _box(lms[p], sx, sy, W, H)
            if bx is None:
                continue
            m = measure(g, bx)
            if m is None:
                continue
            out[p].append((int(t), m))
            if crops is not None:
                x0, y0, x1, y1, _w = bx
                crops.append((p, int(t), m, g[y0:y1, x0:x1].copy()))
    return out


def defocused(sc: Dict[int, list]) -> Tuple[set, dict]:
    """(the persons out of focus, p -> dict(own, ref, n)) from scores()."""
    by_t: Dict[int, float] = {}
    for p, rows in sc.items():
        for t, m in rows:
            by_t[t] = max(by_t.get(t, 0.0), m)
    info, out = {}, set()
    if not by_t:
        return out, info
    best = max(by_t.values())
    for p, rows in sc.items():
        if not rows:
            info[p] = dict(own=None, ref=None, n=0)
            continue
        own = float(np.percentile([m for _, m in rows], KEEP_Q))
        rel = [m / max(by_t[t], 1e-6) for t, m in rows]
        relq = float(np.percentile(rel, KEEP_Q))
        ref = float(np.median([by_t[t] for t, _ in rows]))
        info[p] = dict(own=round(own, 3), rel=round(relq, 3), ref=round(ref, 3), n=len(rows))
        if len(rows) >= MIN_SEEN and best >= SHARP_REF and ref >= SHARP_REF and own < ABS and relq < REL:
            out.add(p)
    return out, info
