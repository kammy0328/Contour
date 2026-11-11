"""People who turn around: facing direction and left/right consistency.

Pose models label left and right from the picture. When a dancer turns her back, models often keep labelling
as if she still faced the camera, flip halfway through the turn, or disagree with each other; averaged
together that puts both wrists in the middle of the body, and the joint tracks jump across the body.

1. Facing per frame (front +1 .. back -1) from the face detector and the head joint scores.
2. Within a frame, every model's left/right labels are aligned to one reference model (per group: head,
   arms, legs), so the ensemble never averages a left wrist with a right one.
3. Through the shot, a two-state path (keep / swap) per group is chosen with dynamic programming: joints may
   not jump between frames, and where the facing is clear, the person's left must be on the correct side
   (on the image right when she faces the camera, on the image left when she shows her back). The labels
   then follow the body through the whole turn.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import skeleton as S

GROUPS: Dict[str, List[Tuple[int, int]]] = {
    "head": [(S.L_EYE, S.R_EYE), (S.L_EAR, S.R_EAR)],
    "arms": [(S.L_SH, S.R_SH), (S.L_EL, S.R_EL), (S.L_WR, S.R_WR)],
    "legs": [(S.L_HIP, S.R_HIP), (S.L_KN, S.R_KN), (S.L_AN, S.R_AN)],
}
# the pair whose left-right order tells the facing for each group
ORDER_PAIR = {"head": (S.L_EAR, S.R_EAR), "arms": (S.L_SH, S.R_SH), "legs": (S.L_HIP, S.R_HIP)}
REF_KEYS = ("pose_rtmw_x", "pose_rtmpose_x", "pose_dwpose_l", "pose_rtmpose_m")


def _perm(group: str) -> np.ndarray:
    p = np.arange(S.K)
    for a, b in GROUPS[group]:
        p[a], p[b] = b, a
    return p


PERM = {g: _perm(g) for g in GROUPS}
PERM_ALL = np.arange(S.K)
for _g in GROUPS:
    for _a, _b in GROUPS[_g]:
        PERM_ALL[_a], PERM_ALL[_b] = _b, _a


def swap_all(kp: np.ndarray) -> np.ndarray:
    """Every left/right pair exchanged (for swap-invariant comparisons)."""
    return kp[..., PERM_ALL, :]


def _norm_scores(a: np.ndarray) -> float:
    s = a[..., 2]
    s = s[np.isfinite(s) & (s > 0)]
    return max(float(np.percentile(s, 90)), 1e-6) if s.size > 20 else (float(s.max()) if s.size else 1.0)


def _body_keys(st) -> List[str]:
    return [k for k in st.raw if k.startswith("pose_") and st.raw[k].shape[2] == S.K]


def _consensus(st, keys, p, t) -> Optional[np.ndarray]:
    arrs = [st.raw[k][p, t] for k in keys if np.isfinite(st.raw[k][p, t, :, :2]).any()]
    if not arrs:
        return None
    A = np.stack(arrs)
    w = np.where(np.isfinite(A[..., 2]) & np.isfinite(A[..., :2]).all(-1), np.clip(A[..., 2], 0, None), 0.0)
    ws = w.sum(0)
    xy = np.where((ws > 1e-6)[:, None], (np.nan_to_num(A[..., :2]) * w[..., None]).sum(0) / np.maximum(ws, 1e-9)[:, None],
                  np.nan)
    sc = ws / max(len(arrs), 1)
    return np.column_stack([xy, sc])


def _scale(kp: np.ndarray) -> float:
    v = S.person_scale(kp)
    return float(v) if np.isfinite(v) and v > 1 else float("nan")


def _up_right(kp: np.ndarray):
    sh = [kp[j, :2] for j in (S.L_SH, S.R_SH) if np.isfinite(kp[j, :2]).all()]
    hp = [kp[j, :2] for j in (S.L_HIP, S.R_HIP) if np.isfinite(kp[j, :2]).all()]
    if sh and hp:
        up = np.mean(sh, 0) - np.mean(hp, 0)
        n = np.hypot(*up)
        up = up / n if n > 1e-6 else np.array([0.0, -1.0])
    else:
        up = np.array([0.0, -1.0])
    right = np.array([-up[1], up[0]])            # image right when the body is upright
    return up, right


# --------------------------------------------------------------------------- facing

def facing(st, rng: Tuple[int, int]) -> np.ndarray:
    """(P, N) facing in -1 (back to the camera) .. +1 (facing it); NaN where unknown.

    Pose models (and even a face detector, on the sliver of cheek seen past the hair) keep reporting a face on
    the back of the head, so their scores say little. What does change is the geometry of the 68 face
    landmarks: on a face seen from the front the eyes are a known distance apart (for this person, relative to
    the torso); turned away, the landmark model collapses them into a small cluster. A side view sits in
    between and gives no verdict (the left/right order of a side view is ambiguous anyway)."""
    from .face import face_size
    a, b = rng
    P, N = st.present.shape
    out = np.full((P, N), np.nan, np.float32)
    keys = _body_keys(st)
    if not keys:
        return out
    fr = getattr(st, "face_raw", None)
    yn = st.raw.get("face_yunet")
    for p in range(P):
        er = np.full(N, np.nan)
        fz = np.full(N, np.nan)
        for t in range(a, b):
            if not st.present[p, t]:
                continue
            c = _consensus(st, keys, p, t)
            if c is None:
                continue
            sh = [c[j, :2] for j in (S.L_SH, S.R_SH) if np.isfinite(c[j, :2]).all()]
            hp = [c[j, :2] for j in (S.L_HIP, S.R_HIP) if np.isfinite(c[j, :2]).all()]
            if not sh or not hp:
                continue
            torso = float(np.hypot(*(np.mean(sh, 0) - np.mean(hp, 0))))
            if torso < 8 or fr is None:
                continue
            lm = fr[p, t]
            if not np.isfinite(lm[36:48, :2]).all():
                continue
            ed = float(np.hypot(*(lm[42:48, :2].mean(0) - lm[36:42, :2].mean(0))))
            er[t] = ed / torso
            fs = face_size(lm)
            fz[t] = fs / torso if np.isfinite(fs) else np.nan
        ok = np.isfinite(er)
        if ok.sum() < 3:
            continue
        # this person's eye distance and face size when seen from the front (never less than a typical adult's)
        ref = max(float(np.nanpercentile(er[ok], 80)), 0.09)
        zref = max(float(np.nanpercentile(fz[np.isfinite(fz)], 80)) if np.isfinite(fz).any() else 0.2, 0.18)
        r = er / ref
        z = fz / zref
        f = np.clip((r - 0.4) / 0.25, -1.0, 1.0)
        # eyes close together but the face still spread out: a side view, not the back of the head
        side = (f < 0) & np.isfinite(z) & (z > 0.75)
        f = np.where(side, np.maximum(f, -0.2), f)
        if yn is not None:
            # a face detector agreeing with a wide eye distance confirms a front view
            ys = yn[p, :, S.NOSE, 2]
            f = np.where(np.isfinite(ys) & (ys > 0.6) & (r > 0.45), np.maximum(f, 0.9), f)
            # a confident face detection is a front view whatever the torso says: the eye distance against the
            # torso misleads when the camera moves in (the torso leaves the frame, the reference was taken on a
            # wide shot) and the person walks out of a silhouette into the light
            f = np.where(np.isfinite(ys) & (ys > 0.8), np.maximum(f, 0.8), f)
        f = np.where(ok, f, np.nan)
        # light smoothing (rolling median of 5)
        idx = np.arange(N)
        filled = np.interp(idx, idx[ok], f[ok])
        pad = np.pad(filled, 2, mode="edge")
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
        out[p] = np.where(ok, med, np.nan)
    return out


# --------------------------------------------------------------------------- alignment within a frame

def _pair_costs(kp: np.ndarray, ref: np.ndarray, pairs) -> Tuple[float, float, int]:
    """Summed distances of kp to ref as labelled (keep) and with left/right exchanged (swap)."""
    d0 = d1 = 0.0
    n = 0
    for a_, b_ in pairs:
        for j, jj in ((a_, b_), (b_, a_)):
            if np.isfinite(kp[j, :2]).all() and np.isfinite(ref[j, :2]).all() and np.isfinite(ref[jj, :2]).all():
                d0 += np.hypot(*(kp[j, :2] - ref[j, :2]))
                d1 += np.hypot(*(kp[j, :2] - ref[jj, :2]))
                n += 1
    return d0, d1, n


def _align_to_ref(st, keys: List[str], p: int, t: int) -> None:
    """Give every model the same left/right labelling as the majority (per group)."""
    cand = [k for k in keys if k in st.raw and np.isfinite(st.raw[k][p, t, :, :2]).sum() >= 12]
    if len(cand) < 2:
        return
    sc = _scale(st.raw[cand[0]][p, t])
    if not np.isfinite(sc):
        return
    for g, pairs in GROUPS.items():
        # the reference is the model that agrees best with all the others (allowing each a swap)
        best, ref_key = None, None
        for r in cand:
            ref = st.raw[r][p, t]
            tot = 0.0
            for k in cand:
                if k == r:
                    continue
                d0, d1, n = _pair_costs(st.raw[k][p, t], ref, pairs)
                tot += min(d0, d1) / max(n, 1)
            if best is None or tot < best:
                best, ref_key = tot, r
        ref = st.raw[ref_key][p, t]
        for k in cand:
            if k == ref_key:
                continue
            kp = st.raw[k][p, t]
            d0, d1, n = _pair_costs(kp, ref, pairs)
            if n >= 2 and d1 < 0.6 * d0 and d0 / n > 0.04 * sc:
                st.raw[k][p, t] = kp[PERM[g]]


# --------------------------------------------------------------------------- consistency through the shot

def _order(kp: np.ndarray, g: str, scale: float) -> float:
    """+1 when the person's left of this group is on the image right (as seen from the front)."""
    a, b = ORDER_PAIR[g]
    if g == "head" and not (np.isfinite(kp[a, :2]).all() and np.isfinite(kp[b, :2]).all()):
        a, b = S.L_EYE, S.R_EYE
    if not (np.isfinite(kp[a, :2]).all() and np.isfinite(kp[b, :2]).all()):
        return 0.0
    _, right = _up_right(kp)
    d = float((kp[a, :2] - kp[b, :2]) @ right)
    return float(np.clip(d / max(0.12 * scale, 1e-6), -1, 1))


def _motion(x0: np.ndarray, x1: np.ndarray, joints: List[int], scale: float) -> Optional[float]:
    ds = []
    for j in joints:
        if np.isfinite(x0[j, :2]).all() and np.isfinite(x1[j, :2]).all():
            ds.append(np.hypot(*(x1[j, :2] - x0[j, :2])))
    if len(ds) < 2:
        return None
    return float(np.mean(ds)) / max(scale, 1e-6)


def consistent_swaps(st, rng: Tuple[int, int], face_dir: np.ndarray) -> Dict[str, np.ndarray]:
    """Per group a (P, N) bool: swap left and right on that frame (relative to the aligned models)."""
    a, b = rng
    P, N = st.present.shape
    keys = _body_keys(st)
    out = {g: np.zeros((P, N), bool) for g in GROUPS}
    if not keys:
        return out
    TRUST, FACE_W, MOT_S, CAP = 1.0, 3.0, 0.08, 30.0
    for p in range(P):
        frames = [t for t in range(a, b) if st.present[p, t]]
        cons = {}
        scales = {}
        for t in frames:
            c = _consensus(st, keys, p, t)
            if c is None:
                continue
            s = _scale(c)
            if not np.isfinite(s):
                continue
            cons[t] = c
            scales[t] = s
        ts = sorted(cons)
        if not ts:
            continue
        for g, pairs in GROUPS.items():
            joints = [j for pr in pairs for j in pr]
            n = len(ts)
            cost = np.zeros((n, 2))
            back = np.zeros((n, 2), int)
            for i, t in enumerate(ts):
                c = cons[t]
                sc = scales[t]
                f = face_dir[p, t] if np.isfinite(face_dir[p, t]) else 0.0
                o = _order(c, g, sc)
                un = np.array([0.0, TRUST])
                if abs(f) > 0.3:
                    # swapping reverses the order; a front view wants order > 0, a back view order < 0
                    for s_ in (0, 1):
                        oo = o if s_ == 0 else -o
                        un[s_] += FACE_W * max(0.0, -np.sign(f) * oo) * min(1.0, abs(f))
                if i == 0:
                    cost[i] = un
                    continue
                tp = ts[i - 1]
                if t - tp > 3 or (st.jump is not None and st.jump[tp + 1:t + 1].any()):
                    best = cost[i - 1].min()
                    cost[i] = un + best
                    back[i] = int(np.argmin(cost[i - 1]))
                    continue
                xp = cons[tp]
                for s_ in (0, 1):
                    x1 = c[PERM[g]] if s_ else c
                    cands = []
                    for sp in (0, 1):
                        x0 = xp[PERM[g]] if sp else xp
                        m = _motion(x0, x1, joints, sc)
                        tc = 0.0 if m is None else min(CAP, (m / MOT_S) ** 2)
                        cands.append(cost[i - 1, sp] + tc)
                    back[i, s_] = int(np.argmin(cands))
                    cost[i, s_] = un[s_] + min(cands)
            s_ = int(np.argmin(cost[-1]))
            for i in range(n - 1, -1, -1):
                out[g][p, ts[i]] = bool(s_)
                s_ = back[i, s_]
    return out


def fix_left_right(st, rng: Tuple[int, int]) -> np.ndarray:
    """Align every model's left/right labels within each frame and through the shot (in place, on st.raw).
    Returns the facing (P, N)."""
    a, b = rng
    keys = _body_keys(st)
    head_keys = [k for k in st.raw if k.startswith("face_") and st.raw[k].shape[2] == S.K]
    P = st.present.shape[0]
    for p in range(P):
        for t in range(a, b):
            if st.present[p, t]:
                _align_to_ref(st, keys + head_keys, p, t)
    face_dir = facing(st, rng)
    sw = consistent_swaps(st, rng, face_dir)
    for g, m in sw.items():
        if not m.any():
            continue
        perm = PERM[g]
        for k in keys + head_keys:
            arr = st.raw[k]
            idx = np.nonzero(m)
            arr[idx] = arr[idx][:, perm]
        if g == "arms" and getattr(st, "hand_raw", None) is not None:
            hr = st.hand_raw
            idx = np.nonzero(m)
            hr[idx] = np.concatenate([hr[idx][:, 21:42], hr[idx][:, 0:21]], 1)
    # facing in the store: with the labels now consistent, the shoulder order adds evidence on side views
    st.facing = face_dir
    return face_dir
