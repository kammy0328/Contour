"""Face landmarks held by the picture itself: anchored optical-flow tracking for skin retouch.

The landmark models look at one frame at a time. A head tilted down under a fringe, a hand passing over the
face, a whip pan or a strobe frame, and they either lose the face (the retouch blinks off for a few frames)
or, worse, place it confidently in the wrong spot (the skin mask slides onto hair or the jaw shadow).

A retoucher's planar tracker does the opposite: it follows the pixels of the face from frame to frame and
only needs a good starting position. Both are combined here:
- between every two frames the face region's motion is measured from dense optical flow (DIS) and fitted
  with a similarity transform (move, turn, scale) by RANSAC, so a hand or hair crossing part of the face
  does not drag it; strobe frames are compared on local structure, not brightness;
- a forward and a backward pass carry the landmarks along that motion and take each frame's own landmarks
  in only where they agree with where the picture says the face went (then expressions, a mouth opening,
  come from the landmarks, and the position from both);
- where a frame's landmarks disagree with both passes and the passes agree with each other, the frame is
  wrong and is replaced; where a frame has none, the passes fill it (up to MAX_DRIFT frames from the last
  agreeing frame, with the visibility fading as the carried position ages);
- where neither the frame nor the picture can be trusted, the face edit fades out instead of guessing.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

AGREE = 0.05          # landmarks agree with the carried face: median distance below 5 % of the face width
MAX_DRIFT = 12        # frames a face is carried by motion alone
REANCHOR = 6          # frames the landmarks must keep disagreeing before the carried face gives way
FLOW_W = 960          # flow resolution (long side)


class _Flow:
    def __init__(self, fs, scale: float):
        self.fs = fs
        self.k = scale
        self._g: Dict[int, np.ndarray] = {}
        self._f: Dict[Tuple[int, int], np.ndarray] = {}
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def gray(self, t: int) -> np.ndarray:
        g = self._g.get(t)
        if g is None:
            g0 = self.fs.gray(t)
            if g0.dtype != np.uint8:
                g0 = np.clip(g0 * (255.0 if g0.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
            h, w = g0.shape[:2]
            g = cv2.resize(g0, (int(round(w * self.k)), int(round(h * self.k))), interpolation=cv2.INTER_AREA)
            self._g[t] = g
            if len(self._g) > 6:
                self._g.pop(next(iter(self._g)))
        return g

    def flow(self, t0: int, t1: int) -> np.ndarray:
        key = (t0, t1)
        f = self._f.get(key)
        if f is None:
            from .camera import light_changed, local_norm
            g0, g1 = self.gray(t0), self.gray(t1)
            if light_changed(g0, g1):
                g0, g1 = local_norm(g0), local_norm(g1)
            f = self.dis.calc(g0, g1, None)
            self._f[key] = f
            if len(self._f) > 4:
                self._f.pop(next(iter(self._f)))
        return f


def _samples(X: np.ndarray, n: int = 9) -> np.ndarray:
    """Landmarks plus a grid inside the face (cheeks, forehead): what the flow is read at."""
    c = X.mean(0)
    lo, hi = X.min(0), X.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1] - 0.15 * (hi[1] - lo[1]), hi[1], n))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    # keep the grid inside the jaw outline (and the forehead above it)
    hull = cv2.convexHull(np.vstack([X[:17], X[17:27] - [0, 0.25 * (hi[1] - lo[1])]]).astype(np.float32))
    inside = np.array([cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0 for p in G])
    return np.vstack([X, G[inside]])


def _carry(fl: _Flow, X: np.ndarray, t0: int, t1: int, to_flow: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """X (68,2) in source px at t0 -> t1 along the picture's motion. Returns (X1 or None, quality 0..1)."""
    S = _samples(X)
    k = to_flow
    P = S * k
    F = fl.flow(t0, t1)
    h, w = F.shape[:2]
    xi = np.clip(np.round(P[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(P[:, 1]).astype(int), 0, h - 1)
    Q = P + F[yi, xi]
    # forward-backward check on the same samples
    B = fl.flow(t1, t0)
    qi = np.clip(np.round(Q[:, 0]).astype(int), 0, w - 1)
    qj = np.clip(np.round(Q[:, 1]).astype(int), 0, h - 1)
    back = Q + B[qj, qi]
    wf = float(np.hypot(*(X[16] - X[0]))) * k[0]
    fb_ok = np.linalg.norm(back - P, axis=1) < max(1.0, 0.03 * wf)
    if fb_ok.sum() < 8:
        return None, 0.0
    M, inl = cv2.estimateAffinePartial2D(P[fb_ok].astype(np.float32), Q[fb_ok].astype(np.float32),
                                         method=cv2.RANSAC, ransacReprojThreshold=max(1.0, 0.02 * wf),
                                         maxIters=500, confidence=0.99)
    if M is None or inl is None:
        return None, 0.0
    q = float(inl.mean()) * float(fb_ok.mean())
    if q < 0.25:
        return None, q
    X1 = (np.hstack([X * k, np.ones((len(X), 1))]) @ M.T) / k
    return X1, q


def _pair(fl: _Flow, X: np.ndarray, t0: int, t1: int, to_flow: np.ndarray) -> Optional[np.ndarray]:
    """The face's motion t0 -> t1 as a 3x3 similarity in source px (None when the picture cannot tell)."""
    X1, q = _carry(fl, X, t0, t1, to_flow)
    if X1 is None:
        return None
    # recover the transform from the carried points (exact: _carry applies one similarity to all of them)
    M, _ = cv2.estimateAffinePartial2D(X.astype(np.float32), X1.astype(np.float32), method=cv2.LMEDS)
    if M is None:
        return None
    return np.vstack([M, [0, 0, 1]])


def _apply(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ A[:2, :2].T + A[:2, 2]


STAB_R = 3            # frames each side
STAB_SIGMA = 1.4      # frames
STAB_REJECT = 0.035   # a neighbour's point further than this (face widths) from the median is an expression change


def stabilize(fl: _Flow, X: np.ndarray, keep: np.ndarray, to_flow: np.ndarray, breaks: set) -> np.ndarray:
    """Takes the frame-to-frame shake out of the landmarks without slowing down any real movement.

    Each frame's neighbours (up to STAB_R each side) are brought onto this frame along the picture's own motion
    (so the head's movement, a pan or a zoom is not averaged away), then averaged per point. A neighbour point
    that moved relative to the face (a mouth opening, a blink) is left out of the average, so expressions keep
    their timing. What remains to be averaged is the landmark model's own frame-to-frame noise."""
    T = len(X)
    out = X.copy()
    Mf: list = [None] * T                       # t -> t+1
    for t in range(T - 1):
        if keep[t] and keep[t + 1] and t not in breaks:
            Mf[t] = _pair(fl, X[t], t, t + 1, to_flow)
    for t in range(T):
        if not keep[t]:
            continue
        cand = [X[t]]
        wts = [1.0]
        A = np.eye(3)
        for s in range(t + 1, min(T, t + STAB_R + 1)):          # later frames, mapped back to t
            if Mf[s - 1] is None:
                break
            A = Mf[s - 1] @ A
            if keep[s]:
                cand.append(_apply(np.linalg.inv(A), X[s]))
                wts.append(float(np.exp(-0.5 * ((s - t) / STAB_SIGMA) ** 2)))
        B = np.eye(3)
        for s in range(t - 1, max(-1, t - STAB_R - 1), -1):     # earlier frames, mapped forward to t
            if Mf[s] is None:
                break
            B = B @ Mf[s]
            if keep[s]:
                cand.append(_apply(B, X[s]))
                wts.append(float(np.exp(-0.5 * ((s - t) / STAB_SIGMA) ** 2)))
        if len(cand) < 3:
            continue
        C = np.stack(cand)                                       # (n, 68, 2)
        w = np.array(wts)[:, None]
        wf = max(1.0, float(np.hypot(*(X[t, 16] - X[t, 0]))))
        med = np.median(C, 0)
        ok = np.linalg.norm(C - med, axis=2) < STAB_REJECT * wf  # (n, 68)
        ok[0] |= ~ok.any(0)                                      # a point nobody agrees on keeps its own value
        W = w * ok
        out[t] = (C * W[..., None]).sum(0) / W.sum(0)[:, None]
    return out


def _dist(A: np.ndarray, B: np.ndarray) -> float:
    wf = max(1.0, float(np.hypot(*(A[16] - A[0]))))
    return float(np.median(np.linalg.norm(A - B, axis=1))) / wf


def _pass(fl, D: np.ndarray, valid: np.ndarray, order, to_flow, breaks: set):
    """One direction. Returns carried landmarks (T,68,2), age (frames since an agreeing detection), and
    whether the frame's own landmarks agreed."""
    T = len(D)
    out = np.full((T, 68, 2), np.nan)
    age = np.full(T, 999, np.int32)
    agree = np.zeros(T, bool)
    X = None
    a = 999
    prev = None
    run = []                 # consecutive disagreeing detections that agree among themselves
    for t in order:
        Pd = None
        if X is not None and prev is not None and min(prev, t) not in breaks:
            Pd, q = _carry(fl, X, prev, t, to_flow)
        if valid[t]:
            if Pd is None:
                X, a = D[t].copy(), 0
                agree[t] = True
                run = []
            elif _dist(D[t], Pd) < AGREE:
                X, a = 0.65 * D[t] + 0.35 * Pd, 0
                agree[t] = True
                run = []
            else:
                # the frame says somewhere else; follow the picture, unless the frames keep saying so
                # (then the pose really changed faster than the flow could follow, e.g. a fast head turn)
                if run and _dist(D[t], D[run[-1]]) < 2 * AGREE:
                    run.append(t)
                else:
                    run = [t]
                if len(run) >= REANCHOR or (Pd is not None and a >= MAX_DRIFT // 2 and len(run) >= 2):
                    X, a = D[t].copy(), 0
                    agree[t] = True
                    run = []
                else:
                    X, a = Pd, a + 1
        else:
            X, a = (Pd, a + 1) if Pd is not None else (None, 999)
        if X is not None and a > MAX_DRIFT:
            X, a = None, 999
        if X is not None:
            out[t] = X
            age[t] = a
        prev = t
    return out, age, agree


def refine(store, fs, frame_range: Tuple[int, int], progress: Optional[Callable] = None, skip=()) -> dict:
    """Runs on store.face / store.face_vis after solve_face. Returns counts for the report. skip: store indices left as
    they are (people out of focus, track/focus.py: never retouched, their faces need no flow)."""
    if fs is None or not getattr(store, "has_face", False):
        return {}
    a, b = frame_range
    g0 = fs.gray(a)
    h, w = g0.shape[:2]
    k = min(1.0, FLOW_W / max(h, w))
    fl = _Flow(fs, k)
    to_flow = np.array([fs.sx * k, fs.sy * k])
    brk = set(store.breaks()) if hasattr(store, "breaks") else set()
    rep = dict(replaced=0, filled=0, faded=0)
    P = store.face.shape[0]
    T = b - a
    for p in range(P):
        if p in skip:
            continue
        pres = store.present[p, a:b]
        if pres.sum() < 2:
            continue
        D = store.face[p, a:b, :, :2].astype(np.float64)
        vis = store.face_vis[p, a:b].astype(np.float32)
        valid = pres & np.isfinite(D).all((1, 2)) & (vis >= 0.3)
        if valid.sum() < 1:
            continue
        loc_brk = {t - a for t in brk if a < t < b}
        fwd, af, okf = _pass(fl, D, valid, range(T), to_flow, loc_brk)
        bwd, ab, okb = _pass(fl, D, valid, range(T - 1, -1, -1), to_flow, loc_brk)
        new = D.copy()
        nv = vis.copy()
        for t in range(T):
            if not pres[t]:
                continue
            hf, hb = np.isfinite(fwd[t]).all(), np.isfinite(bwd[t]).all()
            if hf and hb:
                wfw, wbw = 1.0 / (1 + af[t]), 1.0 / (1 + ab[t])
                C = (fwd[t] * wfw + bwd[t] * wbw) / (wfw + wbw)
                both_ok = _dist(fwd[t], bwd[t]) < 1.5 * AGREE
            elif hf or hb:
                C = fwd[t] if hf else bwd[t]
                both_ok = False
            else:
                C = None
                both_ok = False
            age = min(af[t], ab[t])
            if valid[t] and (okf[t] or okb[t]):
                new[t] = C if (C is not None and okf[t] and okb[t]) else D[t]
                continue
            if valid[t] and not (okf[t] or okb[t]):
                if C is not None and both_ok:
                    new[t] = C
                    nv[t] = max(0.5, min(vis[t], 0.9))
                    rep["replaced"] += 1
                # the two carried positions disagree with each other too: the picture cannot tell, the frame's
                # own landmarks stay
                continue
            # no landmarks of its own
            if C is not None and (both_ok or age <= MAX_DRIFT // 2):
                new[t] = C
                nv[t] = float(np.clip(0.85 - 0.04 * age, 0.4, 0.85)) * (1.0 if both_ok else 0.8)
                rep["filled"] += 1
        keep = pres & np.isfinite(new).all((1, 2)) & (nv >= 0.08)
        if keep.sum() >= 3:
            new = stabilize(fl, new, keep, to_flow, loc_brk)
        sc = store.face[p, a:b, :, 2:3]
        sc = np.where(np.isfinite(sc), sc, 0.5)
        out = np.concatenate([new, sc], -1).astype(np.float32)
        out[~keep] = np.nan
        store.face[p, a:b] = out
        store.face_vis[p, a:b] = np.where(keep, nv, 0.0)
        if progress:
            progress((p + 1) / P, "얼굴 화면 추적")
    return rep
