"""Global (camera) motion between consecutive frames, estimated on the background only."""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


def local_norm(g: np.ndarray, sig: float = 6.0) -> np.ndarray:
    """Local contrast normalisation: each pixel against the brightness and contrast around it. What a flash,
    a strobe or a light sweeping across the set changes is mostly that local level, not the picture's
    structure, so optical flow on these images keeps following through the light change."""
    f = g.astype(np.float32)
    mu = cv2.GaussianBlur(f, (0, 0), sig)
    d = f - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sig)) + 4.0
    return np.clip(d / sd * 40.0 + 128.0, 0, 255).astype(np.uint8)


def light_changed(g0: np.ndarray, g1: np.ndarray) -> bool:
    """The light level changed between two frames (flash, strobe), more than any camera exposure drift."""
    m0, m1 = float(g0.mean()) + 2.0, float(g1.mean()) + 2.0
    return max(m0, m1) / min(m0, m1) > 1.12 or abs(m0 - m1) > 6.0


def frame_homography(g0: np.ndarray, g1: np.ndarray, mask0: Optional[np.ndarray], stats: Optional[dict] = None):
    """Homography mapping g0 -> g1 (proxy px). mask0: 255 where features may be used.
    stats (optional dict): gets "follow", the share of corners that track there and back (NaN if too few)."""
    if light_changed(g0, g1):
        g0, g1 = local_norm(g0), local_norm(g1)
    pts = cv2.goodFeaturesToTrack(g0, maxCorners=600, qualityLevel=0.01, minDistance=12, mask=mask0, blockSize=7)
    if pts is None or len(pts) < 12:
        return np.eye(3), False
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((pb - pts).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    if stats is not None:
        stats["follow"] = float(ok.mean()) if len(pts) >= 30 else float("nan")
    if ok.sum() < 12:
        return np.eye(3), False
    a = pts.reshape(-1, 2)[ok]
    b = p1.reshape(-1, 2)[ok]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 1.5, maxIters=2000, confidence=0.995)
    if H is None or inl is None or inl.sum() < 10:
        return np.eye(3), False
    # reject implausible (projective blow-ups)
    if abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
        A, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if A is None:
            return np.eye(3), False
        H = np.vstack([A, [0, 0, 1]])
    return H, True


def to_source(Hp: np.ndarray, sx: float, sy: float) -> np.ndarray:
    S_ = np.diag([sx, sy, 1.0])
    return np.linalg.inv(S_) @ Hp @ S_


def estimate_all(n: int, gray: Callable[[int], np.ndarray], person_mask: Callable[[int], Optional[np.ndarray]],
                 sx: float, sy: float, progress=None, cancel=None):
    Hs = np.tile(np.eye(3), (n, 1, 1))
    ok = np.zeros(n, bool)
    diff = np.zeros(n, np.float32)
    follow = np.full(n, np.nan, np.float32)
    thumbs = []
    prev = gray(0) if n else None
    for t in range(n - 1):
        if cancel and cancel():
            break
        nxt = gray(t + 1)
        pm = person_mask(t)
        mask = None
        if pm is not None:
            mask = np.where(pm > 20, 0, 255).astype(np.uint8)
            mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
        fst = {}
        Hp, good = frame_homography(prev, nxt, mask, fst)
        follow[t] = fst.get("follow", np.nan)
        Hs[t] = to_source(Hp, sx, sy)
        ok[t] = good
        if t == 0:
            thumbs.append(_thumb(prev))
        thumbs.append(_thumb(nxt))
        diff[t] = _tdiff(thumbs[t], thumbs[t + 1])
        prev = nxt
        if progress and t % 10 == 0:
            progress(t / max(1, n - 1), "카메라 움직임 분석")
    return Hs, ok, find_jumps(diff, ok, follow)


def _thumb(g: np.ndarray) -> np.ndarray:
    a = cv2.resize(g, (96, max(2, int(96 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (a - a.mean()) / (float(a.std()) + 1.0)


def _tdiff(an: np.ndarray, bn: np.ndarray) -> float:
    # 1 - correlation: 0 for the same picture at any brightness, about 1 for unrelated pictures; scaled so
    # the thresholds below (made for grey-level differences) keep their meaning
    return float(45.0 * (1.0 - (an * bn).mean()))


def _thumb_diff(g0: np.ndarray, g1: np.ndarray) -> float:
    """How much the picture changed, with brightness and contrast evened out first: a strobe, a flash or a
    light sweeping over the set changes the level of the frame, not what is in it, and must not count as a
    cut (tracking and smoothing would restart on every flash)."""
    a = _thumb(g0)
    b = cv2.resize(g1, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _tdiff(a, (b - b.mean()) / (float(b.std()) + 1.0))


FOLLOW_SAME = 0.25   # share of corners still followed across a change above which it is the same shot


def find_jumps(diff: np.ndarray, ok: np.ndarray, follow: Optional[np.ndarray] = None) -> np.ndarray:
    """A jump: the camera could not follow AND the picture changed far more than usual (punch-in zoom,
    hidden cut). Smoothing and tracking restart there instead of dragging joints across.
    A flash that returns to the picture before it is NOT told apart here: jump cuts between two similar
    shots of one setup (snare-hit edits) return the same way, and dragging tracking across a real cut is far
    worse than restarting on a flash. What does tell them apart is whether the picture's corners can still be
    followed across the change: over a real cut almost none track there and back (under 10 % on the snare
    plate's jump cuts), over a light switching on or a flash in the same shot a third or more still do.
    follow: that share per frame pair (NaN when unknown). Known edit points (an editing XML) override all this."""
    n = len(diff)
    jump = np.zeros(n, bool)
    if n < 3:
        return jump
    med = float(np.median(diff[:-1])) + 0.5
    for t in range(n - 1):
        loc = diff[max(0, t - 8):t + 9]
        loc = float(np.median(np.delete(loc, min(t, 8)))) + 0.5 if len(loc) > 1 else med
        ref = min(med, loc)
        # diff is 45 x (1 - correlation): about 45 between unrelated pictures, under 25 for flashes,
        # strobes and whip pans within one shot
        if diff[t] > max(27.0, 2.2 * ref) or (not ok[t] and diff[t] > max(16.0, 1.6 * ref)):
            if follow is not None and np.isfinite(follow[t]) and follow[t] >= FOLLOW_SAME:
                continue
            jump[t] = True
    return jump
