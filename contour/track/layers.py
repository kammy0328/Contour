"""Who is in front of whom, per frame, for every pair of tracked people that overlap.

The person matte (segment.py) has to give an overlapped pixel to the person who is really in front. SAM masks alone
do not say: two masks both claim the overlap, and the larger logit wins by chance. The order is therefore worked out
from the tracking, without any mask, and then held over time:

  * a joint of B that tracking marks hidden but lies on A's body says B is behind A (a hand behind a back);
  * a visible joint of A that lies on B's body says A is in front (B would hide it otherwise);
  * lower in the frame (feet) and larger (shoulder width) mean closer to the camera, as a weaker cue;
  * the pair's order is smoothed with a two-state forward-backward pass: it changes only where the evidence says so
    for a while (people crossing), not from frame to frame.

`DepthOrder.at(t)` gives (front pid, back pid, confidence 0..1) for the pairs whose order is clear on frame t."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .store import VIS_HIDDEN, VIS_VISIBLE

CANVAS = 320                 # long side of the silhouette canvas
OVERLAP_MIN = 0.03           # share of the smaller box that two boxes must share to count as overlapping
SWITCH_OVERLAP = 4.0         # cost (nats) of the order changing on a frame where the two people overlap
SWITCH_APART = 1.0           # ... and where they do not (the order does not matter then)
CLEAR = 0.45                 # |2 p - 1| above this: the order is used
W_JOINT, W_GROUND, W_SCALE = 0.7, 1.2, 1.0


def _skeleton_silhouette(kp: np.ndarray, vis: np.ndarray, scale: float, g: float, shape) -> np.ndarray:
    """Rough body area of one person from the VISIBLE joints only: thick limbs, torso polygon, head disc."""
    ch, cw = shape
    img = np.zeros((ch, cw), np.uint8)
    ok = np.isfinite(kp[:, :2]).all(1) & (vis == VIS_VISIBLE)
    th = max(2, int(round(0.10 * scale * g)))
    P = (kp[:, :2] * g)
    for a, b in S.LIMB_BONES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.round(P[a]).astype(int)), tuple(np.round(P[b]).astype(int)), 255, th)
    tors = [j for j in (S.L_SH, S.R_SH, S.R_HIP, S.L_HIP) if ok[j]]
    if len(tors) >= 3:
        cv2.fillConvexPoly(img, np.round(P[tors]).astype(np.int32), 255)
    hd = [j for j in (S.NOSE, S.L_EYE, S.R_EYE, S.L_EAR, S.R_EAR) if ok[j]]
    if hd:
        c = P[hd].mean(0)
        cv2.circle(img, tuple(np.round(c).astype(int)), max(2, int(round(0.075 * scale * g))), 255, -1)
    return img > 0


def _shoulder_w(kp, vis):
    if vis[S.L_SH] == VIS_VISIBLE and vis[S.R_SH] == VIS_VISIBLE and np.isfinite(kp[[S.L_SH, S.R_SH], :2]).all():
        return float(np.hypot(*(kp[S.L_SH, :2] - kp[S.R_SH, :2])))
    return None


def _bottom(kp, vis, box, H):
    ys = [kp[j, 1] for j in (S.L_AN, S.R_AN) if vis[j] == VIS_VISIBLE and np.isfinite(kp[j, 1])]
    if ys:
        return float(max(ys))
    if np.isfinite(box).all() and box[3] < 0.985 * H:
        return float(box[3])
    return None


def _inside(img: np.ndarray, pt, g: float) -> bool:
    x, y = int(round(pt[0] * g)), int(round(pt[1] * g))
    return 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and bool(img[y, x])


def _score(kpa, va, ba, sca, kpb, vb, bb, scb, g, shape, H) -> Tuple[float, float]:
    """(score >0: A in front, overlap share of the smaller box)."""
    ix = max(0.0, min(ba[2], bb[2]) - max(ba[0], bb[0]))
    iy = max(0.0, min(ba[3], bb[3]) - max(ba[1], bb[1]))
    area = lambda b: max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    ov = ix * iy / min(area(ba), area(bb))
    if ov < OVERLAP_MIN:
        return 0.0, ov
    sa = _skeleton_silhouette(kpa, va, sca, g, shape)
    sb = _skeleton_silhouette(kpb, vb, scb, g, shape)
    s = 0.0
    n = 0.0
    for j in range(S.K):
        if np.isfinite(kpb[j, :2]).all():
            if vb[j] == VIS_HIDDEN and _inside(sa, kpb[j], g):
                s += 1.0; n += 1                     # B's joint hidden where A's body is: B is behind A
            elif vb[j] == VIS_VISIBLE and _inside(sa, kpb[j], g) and not _inside(sb, kpb[j], g):
                s -= 0.7; n += 1                     # B's visible joint on A's body, off B's own: A behind (rare)
        if np.isfinite(kpa[j, :2]).all():
            if va[j] == VIS_HIDDEN and _inside(sb, kpa[j], g):
                s -= 1.0; n += 1
            elif va[j] == VIS_VISIBLE and _inside(sb, kpa[j], g) and not _inside(sa, kpa[j], g):
                s += 0.7; n += 1
    out = W_JOINT * float(np.clip(s, -6, 6))
    ya, yb = _bottom(kpa, va, ba, H), _bottom(kpb, vb, bb, H)
    if ya is not None and yb is not None:
        h = max(1.0, 0.5 * (sca + scb))
        out += W_GROUND * float(np.tanh((ya - yb) / h / 0.06))
    wa, wb = _shoulder_w(kpa, va), _shoulder_w(kpb, vb)
    if wa and wb and min(wa, wb) > 0.03 * H:
        out += W_SCALE * float(np.tanh(np.log(wa / wb) / 0.25))
    return out, ov


def _forward_backward(em: np.ndarray, ov: np.ndarray) -> np.ndarray:
    """em: (T,2) log-likelihood of (A front, B front). Returns P(A front) per frame."""
    T = len(em)
    cost = np.where(ov >= OVERLAP_MIN, SWITCH_OVERLAP, SWITCH_APART)
    la = np.zeros((T, 2))
    la[0] = em[0] - np.log(2)
    for t in range(1, T):
        stay = np.log1p(-np.exp(-cost[t]))             # tiny leakage into the other state
        sw = -cost[t]
        a0 = np.logaddexp(la[t - 1, 0] + stay, la[t - 1, 1] + sw)
        a1 = np.logaddexp(la[t - 1, 1] + stay, la[t - 1, 0] + sw)
        la[t] = np.array([a0, a1]) + em[t]
    lb = np.zeros((T, 2))
    for t in range(T - 2, -1, -1):
        stay = np.log1p(-np.exp(-cost[t + 1]))
        sw = -cost[t + 1]
        b0 = np.logaddexp(stay + em[t + 1, 0] + lb[t + 1, 0], sw + em[t + 1, 1] + lb[t + 1, 1])
        b1 = np.logaddexp(stay + em[t + 1, 1] + lb[t + 1, 1], sw + em[t + 1, 0] + lb[t + 1, 0])
        lb[t] = np.array([b0, b1])
    lp = la + lb
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return (p[:, 0] / p.sum(1)).astype(np.float32)


class DepthOrder:
    def __init__(self):
        self.pairs: Dict[Tuple[int, int], Tuple[int, np.ndarray, np.ndarray]] = {}   # (pidA,pidB) -> (t0, P(A front), overlap)

    def at(self, t: int, min_clear: float = CLEAR) -> List[Tuple[int, int, float]]:
        out = []
        for (pa, pb), (t0, post, ov) in self.pairs.items():
            i = t - t0
            if 0 <= i < len(post) and ov[i] >= OVERLAP_MIN:
                conf = abs(2.0 * float(post[i]) - 1.0)
                if conf >= min_clear:
                    out.append((pa, pb, conf) if post[i] > 0.5 else (pb, pa, conf))
        return out


def depth_order(st, a: int, b: int, frame_wh: Tuple[float, float], use_final: bool = True) -> DepthOrder:
    """Order of every overlapping pair of people over frames a..b (source px; frame_wh = source frame size)."""
    W, H = float(frame_wh[0]), float(frame_wh[1])
    g = CANVAS / max(W, H)
    shape = (int(np.ceil(H * g)), int(np.ceil(W * g)))
    kps = st.final if use_final else st.meas
    n = b - a
    P = len(st.pids)
    res = DepthOrder()
    sc_all = st.scale
    for i in range(P):
        for j in range(i + 1, P):
            if not (st.present[i, a:b].any() and st.present[j, a:b].any()):
                continue
            sc = np.zeros(n, np.float32)
            ov = np.zeros(n, np.float32)
            for k in range(n):
                t = a + k
                if not (st.present[i, t] and st.present[j, t]):
                    continue
                ba, bb = st.boxes[i, t], st.boxes[j, t]
                if not (np.isfinite(ba).all() and np.isfinite(bb).all()):
                    continue
                hs = []
                for p_ in (i, j):
                    s_ = sc_all[p_, t]
                    hs.append(float(s_) if np.isfinite(s_) else float(st.boxes[p_, t, 3] - st.boxes[p_, t, 1]))
                sc[k], ov[k] = _score(kps[i, t], st.vis[i, t], ba, hs[0], kps[j, t], st.vis[j, t], bb, hs[1], g, shape, H)
            if not (ov >= OVERLAP_MIN).any():
                continue
            s = np.clip(sc, -5, 5)
            em = np.stack([-np.log1p(np.exp(-s)), -np.log1p(np.exp(s))], 1)
            em[ov < OVERLAP_MIN] = 0.0
            post = _forward_backward(em, ov)
            res.pairs[(int(st.pids[i]), int(st.pids[j]))] = (a, post, ov)
    return res
