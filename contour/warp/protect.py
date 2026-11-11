"""Protected background lines and shapes, carried through the shot with the camera motion."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.project import ProtectLine, ProtectShape
from ..track.store import TrackStore


def _map(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = np.c_[pts, np.ones(len(pts))] @ H.T
    return a[:, :2] / a[:, 2:3]


def _propagate(keys: Dict[int, List[float]], t: int, store: Optional[TrackStore], to_pts, from_pts):
    if not keys:
        return None
    ks = sorted(keys)
    prev = [k for k in ks if k <= t]
    nxt = [k for k in ks if k >= t]
    cands = []
    for k in ([prev[-1]] if prev else []) + ([nxt[0]] if nxt and (not prev or nxt[0] != prev[-1]) else []):
        pts = to_pts(keys[k])
        if store is not None and k != t:
            H = store.homography(k, t)
            if H is None:            # key set on the other side of a picture jump
                continue
            pts = _map(H, pts)
        cands.append((k, pts))
    if not cands:
        return None
    if len(cands) == 1:
        return from_pts(cands[0][1])
    (k0, p0), (k1, p1) = cands
    w = (t - k0) / max(1, (k1 - k0))
    return from_pts(p0 * (1 - w) + p1 * w)


def line_at(line: ProtectLine, t: int, store: Optional[TrackStore]) -> Optional[np.ndarray]:
    """[x1,y1,x2,y2] in source px at frame t."""
    return _propagate(line.keys, t, store, lambda v: np.array(v, float).reshape(2, 2), lambda p: p.reshape(-1))


def shape_at(shape: ProtectShape, t: int, store: Optional[TrackStore]) -> Optional[np.ndarray]:
    """[cx,cy,rx,ry] in source px at frame t (radii follow the camera zoom)."""
    def to_pts(v):
        cx, cy, rx, ry = v
        return np.array([[cx, cy], [cx + rx, cy], [cx, cy + ry]], float)

    def from_pts(p):
        c = p[0]
        return np.array([c[0], c[1], np.linalg.norm(p[1] - c), np.linalg.norm(p[2] - c)])
    return _propagate(shape.keys, t, store, to_pts, from_pts)


def dist_to_segments(X: np.ndarray, Y: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Min distance from grid points to line segments (M,4)."""
    d = np.full(X.shape, np.inf)
    for x1, y1, x2, y2 in segs:
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 < 1e-9:
            dd = np.hypot(X - x1, Y - y1)
        else:
            t = np.clip(((X - x1) * vx + (Y - y1) * vy) / L2, 0, 1)
            dd = np.hypot(X - (x1 + t * vx), Y - (y1 + t * vy))
        d = np.minimum(d, dd)
    return d


def dist_outside_shapes(X: np.ndarray, Y: np.ndarray, shapes: List[tuple]) -> np.ndarray:
    """Approximate distance to the outside of ellipses/rects; 0 inside."""
    d = np.full(X.shape, np.inf)
    for kind, (cx, cy, rx, ry) in shapes:
        rx, ry = max(rx, 1.0), max(ry, 1.0)
        if kind == "rect":
            dx = np.maximum(np.abs(X - cx) - rx, 0)
            dy = np.maximum(np.abs(Y - cy) - ry, 0)
            dd = np.hypot(dx, dy)
        else:
            r = np.hypot((X - cx) / rx, (Y - cy) / ry)
            dd = np.maximum(r - 1, 0) * min(rx, ry)
        d = np.minimum(d, dd)
    return d
