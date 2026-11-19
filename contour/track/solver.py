"""Turning raw model output into stable joint tracks.

1. Ensemble fusion: per joint, robust (IRLS / Cauchy) combination of every pose model, both crop passes and the
   face landmarks; model disagreement lowers confidence and is flagged.
2. Bidirectional point-tracker fusion: a per-joint Kalman filter whose motion model is pyramidal Lucas-Kanade
   optical flow (with forward-backward checking). A forward and a backward pass are combined (two-filter
   smoother); where they disagree the frame is flagged for review.
3. User keyframes are hard measurements; their residual is spread smoothly to neighbouring frames.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import skeleton as S
from .store import TrackStore

LK_PARAMS = dict(winSize=(25, 25), maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))


# --------------------------------------------------------------------------- 1. ensemble fusion

def normalize_scores(raw: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Per-source score normaliser (90th percentile of confident outputs)."""
    out = {}
    for k, a in raw.items():
        s = a[..., 2]
        s = s[np.isfinite(s) & (s > 0)]
        out[k] = float(np.percentile(s, 90)) if s.size > 20 else (float(s.max()) if s.size else 1.0)
        out[k] = max(out[k], 1e-6)
    return out


def _smooth_series(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Rolling median for a 1-D series with NaNs (NaNs are filled by interpolation, then restored)."""
    n = len(x)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    idx = np.arange(n)
    f = np.interp(idx, idx[ok], x[ok])
    pad = np.pad(f, k, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1), axis=1)
    med[~ok] = np.nan if (~ok).all() else med[~ok]
    return med


BLUR_RULE = "scaled"
"""How motion blur lowers a fused joint's trust (a frame whose sharpness dips under 0.55 of its neighbours'):
- "scaled": always, by the sharpness (down to 0.3 x). With the models' own scores already low on a blurred frame,
  a joint the models agree on can fall under the solver's 0.05 and be dropped (PC check 2026-11-19: fused joints
  missing 2.8 % on blurred frames against 1.5 % for each model alone);
- "agree": only where the models disagree (or one model alone saw it); where two or more agree closely the joint
  keeps its trust;
- "agree_low": as "agree", and on a blurred frame a model's point counts from a normalised score of 0.1 (not 0.2)
  when at least two models agree on it (blur lowers every model's score: a joint all of them still place in the
  same spot was left with no measurement at all);
- "none": flagged only."""


def fuse_measurements(store: TrackStore, conf_threshold: float = 0.55, frame_wh=None, alpha_at=None,
                      alpha_scale=(1.0, 1.0), blur_rule: Optional[str] = None, norm: Optional[dict] = None) -> None:
    keys = [k for k in store.raw if store.raw[k].shape[2] == S.K]
    if not keys:
        return
    blur_rule = blur_rule or BLUR_RULE
    norm = norm or normalize_scores({k: store.raw[k] for k in keys})
    xy = np.stack([store.raw[k][..., :2] for k in keys])            # (M,P,N,K,2)
    sc = np.stack([store.raw[k][..., 2] / norm[k] for k in keys])   # (M,P,N,K)
    sc = np.clip(np.nan_to_num(sc, nan=0.0), 0, 1.2)
    is_face = np.array([k.startswith("face_") for k in keys])
    blurred_pn = np.nan_to_num(store.blur, nan=1.0) < 0.55                  # (P,N)
    low_ok = (blur_rule == "agree_low") & blurred_pn[None, :, :, None] & ~is_face[:, None, None, None]
    valid = ((sc > 0.2) | (low_ok & (sc > 0.1))) & np.isfinite(xy).all(-1)
    w = sc * valid
    # Face landmarks are precise for the head joints but carry no body information.
    w[is_face] *= 1.3
    P, N = store.present.shape
    # self-learned trust per model, joint and blur condition (learn/fusion.py); none until it has proven
    # itself on earlier cuts
    try:
        from ..learn.fusion import active_table, blur_bin
        tab = active_table()
    except Exception:
        tab = None
    if tab:
        bb = blur_bin(np.nan_to_num(store.blur, nan=1.0))                 # (P,N)
        for i, k in enumerate(keys):
            r = tab.get(k)
            if r is not None:
                r = np.asarray(r, np.float32)                               # (K,NB)
                w[i] *= np.moveaxis(r[:, bb], 0, -1)                        # (P,N,K)
    # person scale per frame from body models
    scale = np.full((P, N), np.nan, np.float32)
    body_keys = [i for i, k in enumerate(keys) if not is_face[i]]
    for p in range(P):
        for t in np.nonzero(store.present[p])[0]:
            ss = []
            for m in body_keys:
                if valid[m, p, t].sum() >= 5:
                    kp = np.where(valid[m, p, t][:, None], xy[m, p, t], np.nan)
                    v = S.person_scale(kp)
                    if np.isfinite(v):
                        ss.append(v)
            if ss:
                scale[p, t] = float(np.median(ss))
        scale[p] = _smooth_series(scale[p], 7)
    store.scale = scale
    sc_ = np.where(np.isfinite(scale), scale, 400.0)[None, :, :, None]   # (1,P,N,1)
    tau = 0.03 * sc_
    # IRLS with Cauchy weights
    wsum = w.sum(0)
    mu = np.where(wsum[..., None] > 0, (np.nan_to_num(xy) * w[..., None]).sum(0) / np.maximum(wsum, 1e-9)[..., None], np.nan)
    for _ in range(4):
        d = np.linalg.norm(np.nan_to_num(xy) - np.nan_to_num(mu)[None], axis=-1)
        rw = w / (1.0 + (d / tau) ** 2)
        rs = rw.sum(0)
        mu = np.where(rs[..., None] > 0, (np.nan_to_num(xy) * rw[..., None]).sum(0) / np.maximum(rs, 1e-9)[..., None], np.nan)
    d = np.linalg.norm(np.nan_to_num(xy) - np.nan_to_num(mu)[None], axis=-1)
    inlier = valid & (d < 2.5 * tau)
    n_valid = valid[~is_face].sum(0)
    n_in = (inlier[~is_face]).sum(0)
    q_in = np.where(inlier, sc, 0).sum(0) / np.maximum(inlier.sum(0), 1)
    agree = np.where(n_valid > 0, n_in / np.maximum(n_valid, 1), 0)
    spread = np.sqrt(np.where(inlier, d ** 2, 0).sum(0) / np.maximum(inlier.sum(0), 1)) / sc_[0]
    conf = np.clip(q_in, 0, 1) * (0.45 + 0.55 * agree) * np.exp(-(spread / 0.05) ** 2)
    single = (n_valid == 1)
    conf = np.where(single, conf * 0.8, conf)
    # head joints confirmed by face landmarks get a lift
    face_in = inlier[is_face].any(0) if is_face.any() else np.zeros_like(conf, bool)
    conf = np.where(face_in, np.minimum(1.0, conf * 1.15), conf)
    conf = np.where(np.isfinite(mu).all(-1), conf, 0.0)
    if blur_rule == "agree_low":
        # a joint held up only by points under the usual score: kept when two or more of them agree
        strong = (inlier & (sc > 0.2)).any(0)
        conf = np.where(~strong & (n_in < 2), 0.0, conf)
    flags = np.zeros(conf.shape, np.uint8)
    flags |= np.where((n_valid >= 2) & ((agree < 0.67) | (spread > 0.045)), S.FLAG_DISAGREE, 0).astype(np.uint8)
    # blur
    blur_w = np.clip(store.blur, 0.3, 1.0)[..., None]
    blurred = store.blur[..., None] < 0.55
    if blur_rule in ("agree", "agree_low"):
        sure = (n_valid >= 2) & (agree >= 0.67) & (spread <= 0.03)
        conf = conf * np.where(blurred & ~sure, blur_w, 1.0)
    elif blur_rule != "none":
        conf = conf * np.where(blurred, blur_w, 1.0)
    flags |= np.where(store.blur[..., None] < 0.55, S.FLAG_BLUR, 0).astype(np.uint8)
    # bone length jumps
    for a, b in S.LIMB_BONES:
        L = np.linalg.norm(mu[:, :, a] - mu[:, :, b], axis=-1)   # (P,N)
        for p in range(P):
            ref = _smooth_series(L[p], 3)
            r = L[p] / np.maximum(ref, 1e-3)
            bad = np.isfinite(r) & ((r < 0.72) | (r > 1.38))
            for j in (a, b):
                conf[p, bad, j] *= 0.6
                flags[p, bad, j] |= S.FLAG_BONE
    absent = ~store.present[..., None] | ~np.isfinite(mu).all(-1)
    flags |= np.where(absent & store.present[..., None], S.FLAG_MISSING, 0).astype(np.uint8)
    mu = np.where(store.present[..., None, None], mu, np.nan)
    conf = np.where(store.present[..., None], conf, 0)
    store.meas = np.concatenate([mu, conf[..., None]], -1).astype(np.float32)
    store.disagree = spread.astype(np.float32)
    store.flags = flags
    if frame_wh is not None:
        from .visibility import measure
        from .store import VIS_HIDDEN, VIS_OUTSIDE
        vis = measure(store, norm, frame_wh, alpha_at, alpha_scale)
        store.vis_meas = vis
        store.has_vis = True
        c = store.meas[..., 2]
        c = np.where(vis == VIS_HIDDEN, 0.0, c)
        c = np.where(vis == VIS_OUTSIDE, c * 0.35, c)
        store.meas[..., 2] = c
        # person scale from the joints that are really there
        for p in range(P):
            for t in np.nonzero(store.present[p])[0]:
                kp = np.where((vis[p, t] != VIS_HIDDEN)[:, None], store.meas[p, t, :, :2], np.nan)
                v = S.person_scale(kp)
                if np.isfinite(v):
                    scale[p, t] = v
            scale[p] = _smooth_series(scale[p], 7)
        store.scale = scale


# --------------------------------------------------------------------------- 2. bidirectional LK Kalman

class FrameSource:
    """Grey proxy frames + coordinate conversion."""

    def __init__(self, gray_fn: Callable[[int], np.ndarray], sx: float, sy: float):
        self.gray = gray_fn
        self.sx, self.sy = sx, sy


def _lk_step(g0: np.ndarray, g1: np.ndarray, pts: np.ndarray):
    if len(pts) == 0:
        return pts, np.zeros(0, bool), np.zeros(0, np.float32)
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    from .camera import light_changed, local_norm
    if light_changed(g0, g1):
        # a flash or strobe between the two frames: follow structure, not brightness
        g0, g1 = local_norm(g0), local_norm(g1)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK_PARAMS)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK_PARAMS)
    fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
    h, w = g0.shape
    p1 = p1.reshape(-1, 2)
    inb = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & inb & np.isfinite(p1).all(1)
    return p1, ok, fb.astype(np.float32)


def _pass(fs: FrameSource, z: np.ndarray, R: np.ndarray, present: np.ndarray, scale: np.ndarray,
          frames: Sequence[int], use_flow: bool, progress=None, prog_base=0.0, prog_span=1.0, max_coast: int = 5,
          breaks: Optional[set] = None):
    """One directional pass.

    z (Q,T,2) source px, R (Q,T) variance (inf = missing), present (Q,T), scale (Q,T).
    Returns posterior x,P and prior xp,Pp, all indexed by position in `frames`."""
    Q, T = R.shape
    x = np.full((Q, T, 2), np.nan, np.float64)
    Pv = np.full((Q, T), np.inf)
    xp = np.full((Q, T, 2), np.nan, np.float64)
    Pp = np.full((Q, T), np.inf)
    cur = np.full((Q, 2), np.nan)
    curP = np.full(Q, np.inf)
    active = np.zeros(Q, bool)
    miss = np.zeros(Q, np.int32)
    prev_gray = None
    for i, t in enumerate(frames):
        g = fs.gray(t) if use_flow else None
        sc = np.where(np.isfinite(scale[:, i]), scale[:, i], 300.0)
        if i > 0 and breaks and min(int(frames[i - 1]), int(t)) in breaks:
            # the picture jumps here: start over from the measurements
            active[:] = False
            cur[:] = np.nan
            curP[:] = np.inf
            miss[:] = 0
            prev_gray = None
        if i > 0 and active.any():
            idx = np.nonzero(active)[0]
            if use_flow and prev_gray is not None and g is not None:
                pts = cur[idx] * [fs.sx, fs.sy]
                p1, ok, fb = _lk_step(prev_gray, g, pts)
                fb_src = fb / fs.sx
                newp = np.where(ok[:, None], p1 / [fs.sx, fs.sy], cur[idx])
                qstd = np.where(ok, 0.5 + 0.004 * sc[idx] + 1.2 * fb_src, 0.07 * sc[idx])
            else:
                newp = cur[idx]
                qstd = 0.5 + 0.03 * sc[idx]
            cur[idx] = newp
            curP[idx] = curP[idx] + qstd ** 2
        # deactivate where the person is absent or the joint has gone unseen for too long
        zi_, Ri_ = z[:, i], R[:, i]
        has_ = np.isfinite(Ri_) & np.isfinite(zi_).all(1) & present[:, i]
        miss = np.where(has_, 0, miss + 1)
        gone = active & (~present[:, i] | (miss > max_coast))
        active[gone] = False
        cur[gone] = np.nan
        curP[gone] = np.inf
        xp[active, i] = cur[active]
        Pp[active, i] = curP[active]
        zi, Ri = z[:, i], R[:, i]
        has = np.isfinite(Ri) & np.isfinite(zi).all(1) & present[:, i]
        init = has & ~active
        cur[init] = zi[init]
        curP[init] = Ri[init]
        active[init] = True
        upd = has & ~init & active
        if upd.any():
            u = np.nonzero(upd)[0]
            innov = zi[u] - cur[u]
            S_ = curP[u] + Ri[u]
            d2 = (innov ** 2).sum(1) / S_
            Reff = Ri[u] * np.maximum(1.0, d2 / 9.0)      # soft gating against glitches
            K = curP[u] / (curP[u] + Reff)
            cur[u] = cur[u] + K[:, None] * innov
            curP[u] = (1 - K) * curP[u]
        x[active, i] = cur[active]
        Pv[active, i] = curP[active]
        prev_gray = g
        if progress and i % 10 == 0:
            progress(prog_base + prog_span * i / max(1, T), "")
    return x, Pv, xp, Pp


def one_euro_zero_phase(x: np.ndarray, valid: np.ndarray, scale: np.ndarray, strength: float) -> np.ndarray:
    """Zero-phase adaptive smoothing (forward and backward One-Euro, averaged). x (T,2)."""
    if strength <= 0.01:
        return x
    mincut = 4.0 * (1 - strength) + 0.25 * strength   # Hz at 1 frame = 1/30 s
    beta = 0.8
    dcut = 1.0
    te = 1 / 30.0

    def alpha(cut):
        r = 2 * np.pi * cut * te
        return r / (r + 1)

    def run(seq, ok, sc):
        out = seq.copy()
        prev = None
        dprev = np.zeros(2)
        for i in range(len(seq)):
            if not ok[i]:
                prev = None
                continue
            if prev is None:
                prev = seq[i].copy()
                dprev = np.zeros(2)
                out[i] = prev
                continue
            dx = (seq[i] - prev) / te
            a_d = alpha(dcut)
            dhat = a_d * dx + (1 - a_d) * dprev
            speed = np.linalg.norm(dhat) / max(sc[i], 1.0)      # heights per second
            cut = mincut + beta * speed * 10
            a = alpha(cut)
            prev = a * seq[i] + (1 - a) * prev
            dprev = dhat
            out[i] = prev
        return out

    sc = np.where(np.isfinite(scale), scale, 300.0)
    f = run(x, valid, sc)
    b = run(x[::-1], valid[::-1], sc[::-1])[::-1]
    return np.where(valid[:, None], (f + b) / 2, x)


def relative_homographies(homog: Optional[np.ndarray], frames: np.ndarray, radius: int,
                          breaks: Optional[set] = None) -> np.ndarray:
    """rel[i, k] maps points of frame (frames[i] + k - radius) into frame frames[i].
    Neighbours on the other side of a picture jump get NaN (not usable)."""
    T = len(frames)
    K2 = 2 * radius + 1
    rel = np.tile(np.eye(3), (T, K2, 1, 1))
    breaks = breaks or set()
    if breaks:
        for i, t in enumerate(frames):
            for k in range(1, radius + 1):
                if (t + k - 1) in breaks:
                    rel[i, radius + k:] = np.nan
                    break
            for k in range(1, radius + 1):
                if (t - k) in breaks:
                    rel[i, :radius - k + 1] = np.nan
                    break
    if homog is None:
        return rel
    N = len(homog)
    for i, t in enumerate(frames):
        # forward chain: frame t+k -> t  = inv(H_{t+k-1}) ... inv(H_t)
        M = np.eye(3)
        for k in range(1, radius + 1):
            s = t + k - 1
            if s >= N - 1 or s < 0 or s in breaks:
                break
            try:
                M = M @ np.linalg.inv(homog[s])
            except np.linalg.LinAlgError:
                break
            rel[i, radius + k] = M
        M = np.eye(3)
        for k in range(1, radius + 1):
            s = t - k
            if s < 0 or s in breaks:
                break
            M = M @ homog[s]
            rel[i, radius - k] = M
    return rel


def stabilized_smooth(x: np.ndarray, valid: np.ndarray, scale: np.ndarray, rel: np.ndarray, strength: float,
                      locked: Optional[np.ndarray] = None) -> np.ndarray:
    """Speed-adaptive zero-phase smoothing in camera-stabilised coordinates.

    x (Q,T,2), valid (Q,T), scale (Q,T), rel (T,2R+1,3,3). Camera shake survives untouched because every
    neighbour is first mapped into the current frame; only body-relative jitter is smoothed."""
    if strength <= 0.01:
        return x
    Q, T, _ = x.shape
    R = (rel.shape[1] - 1) // 2
    sig_max = 0.5 + 3.0 * strength
    xs = np.nan_to_num(x)
    acc = np.zeros((Q, T, 2))
    wsum = np.zeros((Q, T))
    # speed in stabilised coords (central difference)
    sp = np.zeros((Q, T))
    for k in (-1, 1):
        idx = np.clip(np.arange(T) + k, 0, T - 1)
        M = rel[:, R + k]                                   # (T,3,3)
        okM = np.isfinite(M).all((1, 2))
        M = np.where(okM[:, None, None], M, np.eye(3))
        nb = np.where(okM[None, :, None], xs[:, idx], xs)   # across a jump: no speed from that side
        hom = np.einsum("tij,qtj->qti", M, np.concatenate([nb, np.ones((Q, T, 1))], -1))
        mapped = hom[..., :2] / hom[..., 2:3]
        sp += np.linalg.norm(mapped - xs, axis=-1) * 0.5
    sc = np.where(np.isfinite(scale), scale, 300.0)
    v = sp / np.maximum(sc, 1.0)                            # heights per frame
    sigma = sig_max / (1.0 + v / 0.008)
    for k in range(-R, R + 1):
        idx = np.arange(T) + k
        inside = (idx >= 0) & (idx < T)
        idc = np.clip(idx, 0, T - 1)
        M = rel[:, R + k]
        okM = np.isfinite(M).all((1, 2))
        M = np.where(okM[:, None, None], M, np.eye(3))
        nb = xs[:, idc]
        hom = np.einsum("tij,qtj->qti", M, np.concatenate([nb, np.ones((Q, T, 1))], -1))
        mapped = hom[..., :2] / hom[..., 2:3]
        w = np.exp(-0.5 * (k / np.maximum(sigma, 1e-3)) ** 2) * inside[None, :] * valid[:, idc] * okM[None, :]
        acc += mapped * w[..., None]
        wsum += w
    out = np.where((wsum > 0)[..., None], acc / np.maximum(wsum, 1e-9)[..., None], x)
    out = np.where(valid[..., None], out, x)
    if locked is not None:
        out = np.where(locked[..., None], x, out)
    return out


def spread_residuals(track: np.ndarray, keys: Dict[int, np.ndarray], frames: np.ndarray, reach: int = 8) -> np.ndarray:
    """Make `track` (T,2) pass exactly through keyframes, distributing the residual smoothly."""
    if not keys:
        return track
    f0, f1 = int(frames[0]), int(frames[-1])
    pos = {int(f) - f0: np.asarray(v, float) for f, v in keys.items() if f0 <= int(f) <= f1}
    if not pos:
        return track
    T = len(track)
    ks = sorted(pos)
    res = {k: pos[k] - track[k] for k in ks if np.isfinite(track[k]).all()}
    if not res:
        out = track.copy()
        for k in ks:
            out[k] = pos[k]
        return out
    ks = sorted(res)
    corr = np.zeros((T, 2))
    for i in range(T):
        if i <= ks[0]:
            w = max(0.0, 1 - (ks[0] - i) / reach)
            corr[i] = res[ks[0]] * (w * w * (3 - 2 * w))
        elif i >= ks[-1]:
            w = max(0.0, 1 - (i - ks[-1]) / reach)
            corr[i] = res[ks[-1]] * (w * w * (3 - 2 * w))
        else:
            j = np.searchsorted(ks, i)
            a, b = ks[j - 1], ks[j]
            if a == i:
                corr[i] = res[a]
            else:
                tt = (i - a) / (b - a)
                corr[i] = res[a] * (1 - tt) + res[b] * tt
    out = track + corr
    for k in pos:
        out[k] = pos[k]
    return out


def solve(store: TrackStore, fs: Optional[FrameSource], person_idx: Sequence[int], frame_range: Tuple[int, int],
          user_keys: Optional[Dict[int, Dict[int, Dict[int, List[float]]]]] = None,
          extra: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
          smoothing: float = 0.35, use_flow: bool = True, conf_threshold: float = 0.55,
          progress: Optional[Callable[[float, str], None]] = None,
          offset: Optional[np.ndarray] = None, vis: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve tracks for persons `person_idx` over [a, b).

    user_keys: p_index -> joint -> frame -> [x, y]
    extra: list of (xy (P,N,K,2), var (P,N,K)) additional measurements (clip adaptation, refiner).
    Returns (track (len(idx), T, K, 3), frame_conf (len(idx), T), flags (len(idx), T, K))."""
    a, b = frame_range
    frames = np.arange(a, b)
    T = len(frames)
    idx = list(person_idx)
    nP = len(idx)
    K = S.K
    meas = store.meas[idx][:, a:b]                  # (nP,T,K,3)
    scale = store.scale[idx][:, a:b]
    sc = np.where(np.isfinite(scale), scale, 300.0)
    present = store.present[idx][:, a:b]
    z = meas[..., :2].astype(np.float64)
    if offset is not None:
        z = z + np.nan_to_num(offset[idx][:, a:b])
    c = np.nan_to_num(meas[..., 2], nan=0.0)
    sigma0 = 0.012 * sc[..., None]
    R = np.where(c > 0.05, (sigma0 / np.maximum(c, 0.05)) ** 2, np.inf)
    R = np.where(np.isfinite(z).all(-1), R, np.inf)
    vw = None
    if vis is not None:
        from .store import VIS_HIDDEN, VIS_OUTSIDE
        vw = vis[idx][:, a:b]
        R = np.where(vw == VIS_HIDDEN, np.inf, R)
        R = np.where(vw == VIS_OUTSIDE, R * 9.0, R)
    flags = store.flags[idx][:, a:b].copy()
    # extra measurements (information-form fusion)
    if extra:
        for exy, evar in extra:
            ez = exy[idx][:, a:b].astype(np.float64)
            ev = evar[idx][:, a:b].astype(np.float64)
            ok = np.isfinite(ez).all(-1) & np.isfinite(ev) & (ev > 0)
            both = ok & np.isfinite(R)
            only = ok & ~np.isfinite(R)
            info = np.where(both, 1 / np.where(both, R, 1) + 1 / np.where(both, ev, 1), 0)
            zz = np.where(both[..., None], (np.nan_to_num(z) / np.where(both, R, 1)[..., None]
                                            + np.nan_to_num(ez) / np.where(both, ev, 1)[..., None])
                          / np.where(both, info, 1)[..., None], z)
            z = np.where(only[..., None], ez, zz)
            R = np.where(both, 1 / np.where(both, info, 1), np.where(only, ev, R))
            flags[ok] |= S.FLAG_ADAPT
    # user keyframes: hard measurements
    keys_by_q: Dict[int, Dict[int, np.ndarray]] = {}
    if user_keys:
        for qi, p in enumerate(idx):
            for j, fr in (user_keys.get(p) or {}).items():
                for f, xy in fr.items():
                    if a <= f < b:
                        ti = f - a
                        z[qi, ti, j] = xy
                        R[qi, ti, j] = 0.25
                        present[qi, ti] = True
                        flags[qi, ti, j] |= S.FLAG_USER
                        keys_by_q.setdefault(qi * K + j, {})[f] = np.asarray(xy, float)
    # flatten persons x joints into points
    Q = nP * K
    zq = z.transpose(0, 2, 1, 3).reshape(Q, T, 2)
    Rq = R.transpose(0, 2, 1).reshape(Q, T)
    pq = np.repeat(present[:, None, :], K, axis=1).reshape(Q, T)
    sq = np.repeat(sc[:, None, :], K, axis=1).reshape(Q, T)

    def prog(f, msg):
        if progress:
            progress(f, msg or "추적 융합")

    brk = store.breaks()
    xf, Pf, _, _ = _pass(fs, zq, Rq, pq, sq, frames, use_flow and fs is not None, prog, 0.0, 0.5, breaks=brk)
    rev = frames[::-1]
    xb, Pb, xbp, Pbp = _pass(fs, zq[:, ::-1], Rq[:, ::-1], pq[:, ::-1], sq[:, ::-1], rev,
                             use_flow and fs is not None, prog, 0.5, 0.5, breaks=brk)
    xbp, Pbp = xbp[:, ::-1], Pbp[:, ::-1]
    xb, Pb = xb[:, ::-1], Pb[:, ::-1]
    # combine forward posterior with backward prior (two-filter smoother)
    fin_f = np.isfinite(Pf)
    fin_b = np.isfinite(Pbp)
    wf = np.where(fin_f, 1 / np.where(fin_f, Pf, 1), 0)
    wb = np.where(fin_b, 1 / np.where(fin_b, Pbp, 1), 0)
    ws = wf + wb
    x = np.where((ws > 0)[..., None], (np.nan_to_num(xf) * wf[..., None] + np.nan_to_num(xbp) * wb[..., None])
                 / np.maximum(ws, 1e-12)[..., None], np.nan)
    # posterior-only frames (e.g. last frame of backward) fall back to backward posterior
    only_b = (ws == 0) & np.isfinite(Pb)
    x[only_b] = xb[only_b]
    var = np.where(ws > 0, 1 / np.maximum(ws, 1e-12), np.where(only_b, Pb, np.inf))
    mism = np.where(fin_f & fin_b, ((xf - xbp) ** 2).sum(-1) / np.maximum(Pf + Pbp, 1e-9), 0)
    # extra smoothing in camera-stabilised coordinates, then exact keyframes
    if smoothing > 0.01:
        rel = relative_homographies(store.homog if store.homog_ok.any() else None, frames, 6, brk)
        okall = np.isfinite(x).all(-1)
        x = stabilized_smooth(x, okall, sq, rel, smoothing)
    for q in keys_by_q:
        x[q] = spread_residuals(x[q], keys_by_q[q], frames)
    x = x.reshape(nP, K, T, 2).transpose(0, 2, 1, 3)
    var = var.reshape(nP, K, T).transpose(0, 2, 1)
    mism = mism.reshape(nP, K, T).transpose(0, 2, 1)
    s_ok = (0.015 * sc)[..., None]
    jconf = np.sqrt(s_ok ** 2 / (s_ok ** 2 + np.where(np.isfinite(var), var, 1e12)))
    jconf = np.where(np.isfinite(x).all(-1), jconf, 0.0)
    meas_c = np.nan_to_num(meas[..., 2], nan=0.0)
    jconf = 0.5 * jconf + 0.5 * np.maximum(meas_c, jconf * 0.6)
    fbbad = mism > 16
    flags = flags | np.where(fbbad, S.FLAG_FB, 0).astype(np.uint8)
    jconf = np.where(fbbad, jconf * 0.6, jconf)
    jconf = np.where(flags & S.FLAG_DISAGREE, jconf * 0.8, jconf)
    jconf = np.where(flags & S.FLAG_USER, 1.0, jconf)
    jconf = np.clip(jconf, 0, 1)
    if vw is not None:
        from .store import VIS_HIDDEN, VIS_VISIBLE
        keyed = (flags & S.FLAG_USER) > 0
        hide = (vw == VIS_HIDDEN) & ~keyed
        x = np.where(hide[..., None], np.nan, x)
        jconf = np.where(hide, 0.0, jconf)
        seen = (vw == VIS_VISIBLE) | keyed
    else:
        seen = np.isfinite(x).all(-1)
    low = (jconf < conf_threshold) & np.isfinite(x).all(-1) & seen
    flags = (flags & ~np.uint8(S.FLAG_LOW)) | np.where(low, S.FLAG_LOW, 0).astype(np.uint8)
    fconf = frame_conf(jconf, seen, present)
    track = np.concatenate([x, jconf[..., None]], -1).astype(np.float32)
    track[~present] = np.nan
    return track, fconf.astype(np.float32), flags


def frame_conf(jconf: np.ndarray, seen: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Per-frame confidence over the joints that are actually visible (hidden ones are not 'uncertain').

    Uses the core joints when at least three are visible, otherwise every visible joint (close-ups), and
    ignores the one or two worst ones (brief occlusion, motion blur)."""
    K = jconf.shape[-1]
    core_mask = np.zeros(K, bool)
    core_mask[S.CORE_JOINTS] = True
    n_core = (seen & core_mask).sum(-1)
    sel = seen & np.where((n_core >= 3)[..., None], core_mask, True)
    vals = np.where(sel, np.nan_to_num(jconf), np.inf)
    srt = np.sort(vals, axis=-1)
    n = sel.sum(-1)
    k = np.where(n >= 6, 2, np.where(n >= 4, 1, 0))
    fin = np.where(np.isfinite(srt), srt, 0.0)
    tot = fin.sum(-1)
    worst = np.where(k >= 1, fin[..., 0], 0.0) + np.where(k >= 2, fin[..., 1], 0.0)
    out = np.where(n > k, (tot - worst) / np.maximum(n - k, 1), 0.0)
    return np.where(present & (n > 0), out, 0.0).astype(np.float32)


def frame_conf_from_track(track: np.ndarray, present: np.ndarray, vis: Optional[np.ndarray] = None,
                          flags: Optional[np.ndarray] = None) -> np.ndarray:
    jconf = np.nan_to_num(track[..., 2])
    if vis is not None:
        from .store import VIS_VISIBLE
        seen = (vis == VIS_VISIBLE) | (((flags & S.FLAG_USER) > 0) if flags is not None else False)
        seen = seen & np.isfinite(track[..., :2]).all(-1)
    else:
        seen = np.isfinite(track[..., :2]).all(-1)
    return frame_conf(jconf, seen, present)
