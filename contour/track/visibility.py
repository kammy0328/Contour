"""Joint visibility.

Pose models always return all 17 joints, also for limbs that are out of frame or hidden behind someone or
something: they "hallucinate" plausible positions, often with confident-looking scores. Using those joints
makes the skeleton wrong and lets the warp bend background that has no body behind it.

A joint counts as visible only when
  * the models agree it is there (score, normalised per model), and
  * it is not a low-score guess squeezed against the frame edge, and
  * there is a person in the matte at that spot (unless the models are very sure).
Joints placed beyond the frame edge are kept as VIS_OUTSIDE: they only give a cut-off limb its direction.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .store import VIS_HIDDEN, VIS_OUTSIDE, VIS_VISIBLE, TrackStore

SCORE_HIDE = 0.33        # below: hidden, whatever else is true
SCORE_EDGE = 0.60        # near the frame edge a joint needs at least this
SCORE_SURE = 0.88        # this sure, the matte is not consulted (thin wrists, fast blur)
MATTE_MIN = 0.35


def body_score(store: TrackStore, norm: Dict[str, float]) -> np.ndarray:
    """(P,N,K) mean normalised score over the body models (face sources excluded)."""
    keys = [k for k in store.raw if store.raw[k].shape[2] == S.K and not k.startswith("face_")]
    if not keys:
        return np.zeros(store.meas.shape[:3], np.float32)
    sc = np.stack([np.nan_to_num(store.raw[k][..., 2] / norm[k], nan=np.nan) for k in keys])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(np.where(np.isfinite(sc), sc, np.nan), axis=0)
    return np.nan_to_num(m, nan=0.0).astype(np.float32)


def measure(store: TrackStore, norm: Dict[str, float], frame_wh: Tuple[int, int],
            alpha_at: Optional[Callable[[int], Optional[np.ndarray]]] = None,
            alpha_scale: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    W, H = frame_wh
    P, N = store.present.shape
    vis = np.zeros((P, N, S.K), np.uint8)
    score = body_score(store, norm)
    xy = store.meas[..., :2]
    margin = 0.012 * min(W, H)
    for t in range(N):
        pres = np.nonzero(store.present[:, t])[0]
        if not len(pres):
            continue
        alpha = alpha_at(t) if alpha_at is not None else None
        for p in pres:
            pts = xy[p, t]
            s = score[p, t]
            fin = np.isfinite(pts).all(1)
            inside = fin & (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            edge = inside & ((pts[:, 0] < margin) | (pts[:, 0] > W - margin) | (pts[:, 1] < margin) |
                             (pts[:, 1] > H - margin))
            v = np.full(S.K, VIS_VISIBLE, np.uint8)
            v[~fin] = VIS_HIDDEN
            v[fin & ~inside] = VIS_OUTSIDE
            v[inside & (s < SCORE_HIDE)] = VIS_HIDDEN
            v[edge & (s < SCORE_EDGE)] = VIS_HIDDEN
            if alpha is not None:
                sc_ = store.scale[p, t] if np.isfinite(store.scale[p, t]) else 300.0
                r = max(2, int(0.025 * sc_ * alpha_scale[0]))
                ah, aw = alpha.shape
                for j in np.nonzero((v == VIS_VISIBLE) & (s < SCORE_SURE))[0]:
                    cx = int(pts[j, 0] * alpha_scale[0])
                    cy = int(pts[j, 1] * alpha_scale[1])
                    x0, x1 = max(0, cx - r), min(aw, cx + r + 1)
                    y0, y1 = max(0, cy - r), min(ah, cy + r + 1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    support = float(alpha[y0:y1, x0:x1].max()) / 255.0
                    if support < MATTE_MIN:
                        v[j] = VIS_HIDDEN
            vis[p, t] = v
    limb_check(store, vis, score, (W, H))
    return clean(vis, store.present)


# --------------------------------------------------------------------------- cut-off limbs

# chains from the torso outwards; the first bone of a leg chain is the torso side (shoulder -> hip)
CHAINS = [(S.L_SH, S.L_HIP, S.L_KN, S.L_AN), (S.R_SH, S.R_HIP, S.R_KN, S.R_AN),
          (S.L_SH, S.L_EL, S.L_WR), (S.R_SH, S.R_EL, S.R_WR)]
# bone length / torso length (shoulder-hip), adult averages; torso bone is relative to shoulder width
BONE_DEFAULT = {(S.L_SH, S.L_HIP): 1.35, (S.R_SH, S.R_HIP): 1.35,
                (S.L_HIP, S.L_KN): 0.82, (S.R_HIP, S.R_KN): 0.82, (S.L_KN, S.L_AN): 0.80, (S.R_KN, S.R_AN): 0.80,
                (S.L_SH, S.L_EL): 0.58, (S.R_SH, S.R_EL): 0.58, (S.L_EL, S.L_WR): 0.50, (S.R_EL, S.R_WR): 0.50}
EDGE_BAND = 0.07         # of min(W, H): a joint this close to the edge may be a clamped guess
FOLLOW_BAND = 0.20       # children of a cut-off joint within this band are cut off too


def _mid(a, b):
    return (a + b) * 0.5


def _edge_dist(pt, W, H):
    return min(pt[0], W - pt[0], pt[1], H - pt[1])


def _beyond(pt, W, H):
    return max(-pt[0], pt[0] - W, -pt[1], pt[1] - H)


def _series_fill(v: np.ndarray, k: int = 5) -> np.ndarray:
    """Median-filter a sparse series (NaN = unknown) and fill the gaps from the nearest known value."""
    out = np.full_like(v, np.nan)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    N = len(v)
    for t in range(N):
        w = v[max(0, t - k):t + k + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[t] = np.median(w)
    ok = np.isfinite(out)
    if not ok.all():
        idx = np.nonzero(ok)[0]
        out = np.interp(np.arange(N), idx, out[idx])
    return out


def limb_check(store: TrackStore, vis: np.ndarray, score: np.ndarray, frame_wh: Tuple[int, int]) -> None:
    """Cut-off limbs. When a limb leaves the frame the models still put its joints inside the picture, piled
    up against the edge. Such a joint is too close to its parent for this person's proportions, and the
    limb, drawn at its true length, would end beyond the edge. Those joints are moved to where the limb
    really goes (outside the frame) and marked VIS_OUTSIDE, so the visible part of the limb keeps its
    direction and its reshape, and the warp never pulls on the edge of the picture."""
    W, H = frame_wh
    P, N = store.present.shape
    band = EDGE_BAND * min(W, H)
    follow = FOLLOW_BAND * min(W, H)
    xy = store.meas[..., :2]
    for p in range(P):
        pres = store.present[p]
        if not pres.any():
            continue
        V = vis[p]
        X = xy[p]
        def visible_inner(t, j):
            return V[t, j] == VIS_VISIBLE and np.isfinite(X[t, j]).all() and _edge_dist(X[t, j], W, H) > band
        # ---- units per frame: torso length, shoulder width
        T = np.full(N, np.nan, np.float32)
        SW = np.full(N, np.nan, np.float32)
        for t in np.nonzero(pres)[0]:
            if V[t, S.L_SH] == VIS_VISIBLE and V[t, S.R_SH] == VIS_VISIBLE:
                SW[t] = np.hypot(*(X[t, S.L_SH] - X[t, S.R_SH]))
                if visible_inner(t, S.L_HIP) and visible_inner(t, S.R_HIP):
                    T[t] = np.hypot(*(_mid(X[t, S.L_SH], X[t, S.R_SH]) - _mid(X[t, S.L_HIP], X[t, S.R_HIP])))
        # ---- this person's proportions, from frames where both ends are clearly in the picture
        ratio = {}
        for bone, dflt in BONE_DEFAULT.items():
            a, b = bone
            unit = SW if a in (S.L_SH, S.R_SH) and b in (S.L_HIP, S.R_HIP) else T
            vals = []
            for t in np.nonzero(pres & np.isfinite(unit))[0]:
                if visible_inner(t, a) and visible_inner(t, b) and score[p, t, a] > 0.6 and score[p, t, b] > 0.6:
                    vals.append(np.hypot(*(X[t, b] - X[t, a])) / max(unit[t], 1e-3))
            r = float(np.percentile(vals, 85)) if len(vals) >= 8 else dflt
            ratio[bone] = float(np.clip(r, 0.6 * dflt, 1.6 * dflt))
        t_sw = ratio[(S.L_SH, S.L_HIP)]
        sw_s = _series_fill(SW)
        T_s = _series_fill(T)
        # where the torso is not measurable, derive it from the shoulders
        T_s = np.where(np.isfinite(T_s), T_s, sw_s * t_sw)
        if not np.isfinite(T_s).any():
            continue
        for t in np.nonzero(pres)[0]:
            for chain in CHAINS:
                moved = False
                prev_dir = None
                for a, b in zip(chain[:-1], chain[1:]):
                    if V[t, a] == VIS_HIDDEN or not np.isfinite(X[t, a]).all() or not np.isfinite(X[t, b]).all():
                        moved = False
                        prev_dir = None
                        continue
                    torso_bone = a in (S.L_SH, S.R_SH) and b in (S.L_HIP, S.R_HIP)
                    unit = sw_s[t] * 1.0 if torso_bone else T_s[t]
                    if not np.isfinite(unit) or unit < 4:
                        break
                    L = ratio[(a, b)] * unit
                    seg = X[t, b] - X[t, a]
                    obs = float(np.hypot(*seg))
                    if moved:
                        # parent is cut off: a child still in the picture near the edge is cut off too,
                        # unless the models are sure of it (an arm bending back into the frame)
                        if V[t, b] == VIS_VISIBLE and (_edge_dist(X[t, b], W, H) < follow or score[p, t, b] < SCORE_SURE):
                            d = prev_dir
                            ext = X[t, a] + d * L
                            if _beyond(ext, W, H) > 0:
                                X[t, b] = ext
                                V[t, b] = VIS_OUTSIDE
                            else:
                                V[t, b] = VIS_HIDDEN
                                moved = False
                            prev_dir = d
                            continue
                        moved = False
                    if V[t, b] != VIS_VISIBLE or _edge_dist(X[t, b], W, H) > band:
                        prev_dir = seg / obs if obs > 1e-3 else None
                        continue
                    if obs > 0.2 * L:
                        d = seg / obs
                    elif prev_dir is not None:
                        d = prev_dir
                    else:
                        d = _toward_edge(X[t, a], W, H)
                    ext = X[t, a] + d * L
                    if obs < 0.92 * L and _beyond(ext, W, H) > 0.15 * L:
                        X[t, b] = ext
                        V[t, b] = VIS_OUTSIDE
                        moved = True
                    prev_dir = d
        store.meas[p, ..., :2] = X


def _toward_edge(pt, W, H):
    d = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.float32)
    dist = np.array([pt[0], W - pt[0], pt[1], H - pt[1]])
    return d[int(np.argmin(dist))]


def clean(vis: np.ndarray, present: np.ndarray, k: int = 2) -> np.ndarray:
    """Temporal clean-up: a joint's state may not flicker for a frame or two."""
    out = vis.copy()
    P, N, K = vis.shape
    for p in range(P):
        for j in range(K):
            v = vis[p, :, j]
            on = (v == VIS_VISIBLE).astype(np.float32)
            # majority over a 2k+1 window, only where the person is present
            pad = np.pad(on, k, mode="edge")
            maj = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1).mean(1) > 0.5
            new = np.where(maj, VIS_VISIBLE, np.where(v == VIS_OUTSIDE, VIS_OUTSIDE, VIS_HIDDEN)).astype(np.uint8)
            # a frame voted visible whose own estimate was beyond the edge stays outside
            new = np.where(maj & (v == VIS_OUTSIDE), VIS_OUTSIDE, new)
            out[p, :, j] = np.where(present[p], new, VIS_HIDDEN)
    return out


def apply_overrides(vis: np.ndarray, store: TrackStore, keys_by_index: Dict[int, Dict[int, Dict[int, list]]],
                    hidden_by_index: Dict[int, Dict[int, list]]) -> np.ndarray:
    """User decisions win: a keyframe makes a joint visible on that frame, a hidden range hides it."""
    out = vis.copy()
    for p, jd in hidden_by_index.items():
        for j, ranges in jd.items():
            for a, b in ranges:
                out[p, max(0, a):min(store.n, b + 1), j] = VIS_HIDDEN
    for p, jd in keys_by_index.items():
        for j, fr in jd.items():
            for f in fr:
                if 0 <= f < store.n:
                    out[p, f, j] = VIS_VISIBLE
    return out


def segment_weights(final: np.ndarray, vis: np.ndarray, present: np.ndarray, fade: int = 6):
    """For one person: geometry with short gaps held, and a 0..1 weight per segment that fades in and out.

    final (N,K,3), vis (N,K). Returns geo (N,K,3), segw (N,N_SEG), shoulder weight (N,)."""
    from ..warp.body import N_SEG, segment_endpoints
    N = final.shape[0]
    geo = final.copy()
    for j in range(S.K):
        ok = np.isfinite(final[:, j, :2]).all(1)
        if ok.all() or not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        miss = np.nonzero(~ok)[0]
        pos = np.searchsorted(idx, miss)
        lo = idx[np.clip(pos - 1, 0, len(idx) - 1)]
        hi = idx[np.clip(pos, 0, len(idx) - 1)]
        near = np.where(np.abs(miss - lo) <= np.abs(hi - miss), lo, hi)
        close = np.abs(near - miss) <= fade
        geo[miss[close], j] = final[near[close], j]
    act = np.zeros((N, N_SEG), np.float32)
    for t in range(N):
        if not present[t]:
            continue
        kp = final[t]
        ends = segment_endpoints(kp)
        act[t] = np.isfinite(ends).all(1)
    # a segment also needs at least one truly visible endpoint (two joints off-frame do nothing useful)
    from ..warp.body import SEGMENTS
    vis_any = np.zeros((N, N_SEG), bool)
    for i, s in enumerate(SEGMENTS):
        js = []
        for e in (s.a, s.b):
            if e.isdigit():
                js.append([int(e)])
            elif e == "neck":
                js.append([S.L_SH, S.R_SH])
            elif e == "pelvis":
                js.append([S.L_HIP, S.R_HIP])
            else:
                js.append([S.L_EAR, S.R_EAR, S.NOSE])
        flat = [j for grp in js for j in grp]
        vis_any[:, i] = (vis[:, flat] == VIS_VISIBLE).any(1)
    act *= vis_any
    kern = np.ones(2 * fade + 1, np.float32) / (2 * fade + 1)
    segw = np.zeros_like(act)
    for i in range(N_SEG):
        a = act[:, i]
        # erode then blur: weight reaches 0 exactly where the segment stops being valid
        sm = np.convolve(np.pad(a, fade, mode="edge"), kern, mode="valid")[:N]
        segw[:, i] = sm
    # never put weight where geometry is unavailable
    for i in range(N_SEG):
        for t in range(N):
            if segw[t, i] > 0 and not act[t, i]:
                ends = segment_endpoints(geo[t])
                if not np.isfinite(ends[i]).all():
                    segw[t, i] = 0.0
    sh = ((vis[:, S.L_SH] == VIS_VISIBLE) & (vis[:, S.R_SH] == VIS_VISIBLE)).astype(np.float32)
    shw = np.convolve(np.pad(sh, fade, mode="edge"), kern, mode="valid")[:N]
    shw = np.where(np.isfinite(geo[:, S.L_SH, :2]).all(1) & np.isfinite(geo[:, S.R_SH, :2]).all(1), shw, 0.0)
    return geo, segw.astype(np.float32), shw.astype(np.float32)
