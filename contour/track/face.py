"""Face landmarks (68) and hands (2 x 21) tracked through the shot.

Same idea as the body: per-frame model landmarks fused with optical flow in a forward-backward filter, then
camera-stabilised smoothing, so face shaping and skin masks stay locked to the face without shimmer.

Faces: RTMW (whole body model) + a dedicated 106-point face model on a face crop, fused by score.
Hands: RTMW hand points + a dedicated 21-point hand model on hand crops.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from . import solver as SV
from .store import NF, NH, TrackStore

# 106-point face model (LaPa layout) -> iBUG 68
MAP106_68 = np.array(list(range(0, 33, 2)) + [33, 34, 35, 36, 37, 42, 43, 44, 45, 46] + [51, 52, 53, 54]
                     + [58, 59, 60, 61, 62] + [66, 67, 69, 70, 71, 73] + [75, 76, 78, 79, 80, 82]
                     + list(range(84, 96)) + list(range(96, 104)))

L_EYE_IDX = list(range(42, 48))     # subject's left eye (image right when facing camera)
R_EYE_IDX = list(range(36, 42))
JAW_IDX = list(range(0, 17))
BROW_L = list(range(22, 27))
BROW_R = list(range(17, 22))
NOSE_IDX = list(range(27, 36))
MOUTH_IDX = list(range(48, 60))
INNER_MOUTH_IDX = list(range(60, 68))


def face_size(lm: np.ndarray) -> float:
    """Jaw width in px (or a fallback from the eyes)."""
    a, b = lm[0, :2], lm[16, :2]
    if np.isfinite(a).all() and np.isfinite(b).all():
        return float(np.hypot(*(b - a)))
    e1, e2 = lm[36, :2], lm[45, :2]
    if np.isfinite(e1).all() and np.isfinite(e2).all():
        return float(np.hypot(*(e2 - e1)) * 2.1)
    return float("nan")


REGIONS = {"jaw": JAW_IDX, "eye_r": R_EYE_IDX, "eye_l": L_EYE_IDX, "mouth": MOUTH_IDX + INNER_MOUTH_IDX}


def _axes(a, b, toward=None):
    u = b - a
    w = float(np.hypot(*u))
    if not np.isfinite(w) or w < 1e-6:
        return None
    u = u / w
    n = np.array([-u[1], u[0]])
    if toward is not None and n @ (toward - (a + b) / 2) < 0:
        n = -n
    return u, n, w


def bad_regions(lm: np.ndarray) -> set:
    """Which parts of one face's 68 landmarks cannot be a face: on hard frames (motion blur, a crop it does not
    like) a landmark model can return a tangle, every point on some facial feature but not its own, so that the
    mouth outline crosses itself or the jaw runs back and forth. Such a part is no measurement at all."""
    P = lm[:, :2]
    out = set()
    if not np.isfinite(P).all():
        return out
    fs = face_size(lm)
    ax = _axes(P[0], P[16])
    if ax is None or (np.diff(P[JAW_IDX] @ ax[0]) < -0.02 * fs).sum() > 2:
        out.add("jaw")
    down = P[8]
    ax = _axes(P[48], P[54], down)
    if ax is None:
        out.add("mouth")
    else:
        u, n, w = ax
        # in units of the mouth width, but never of less than a quarter face: a mouth seen from the side is
        # narrow, and small slips would look huge
        m = max(w, 0.25 * fs)
        pu = (P[48:60] - P[48]) @ u / m
        gap = (P[[59, 58, 57, 56, 55]] - P[[49, 50, 51, 52, 53]]) @ n / m
        # along the mouth the lips run from corner to corner (an open "O" mouth bulges past its corners, so the
        # corners themselves are left out of this)
        su = np.diff(pu[1:6])
        sl = np.diff(pu[7:12])
        if (w > 0.85 * fs or pu.min() < -0.4 * w / m - 0.1 or pu.max() > 1.4 * w / m + 0.1 or gap.min() < -0.03
                or gap.mean() < 0.05 or su.min() < -0.1 or sl.max() > 0.1):
            out.add("mouth")
    for name, idx in (("eye_r", R_EYE_IDX), ("eye_l", L_EYE_IDX)):
        E = P[idx]
        ax = _axes(E[0], E[3], down)
        if ax is None:
            continue
        u, n, w = ax
        if w < 0.04 * fs:
            continue            # the far eye of a face in profile: hidden, nothing to judge
        m = max(w, 0.12 * fs)
        pu = (E - E[0]) @ u / m
        gap = (E[[5, 4]] - E[[1, 2]]) @ n / m
        if w > 0.45 * fs or pu.min() < -0.15 or pu.max() > w / m + 0.15 or gap.min() < -0.06:
            out.add(name)
    return out


def clean_sources(a: Optional[np.ndarray], b: Optional[np.ndarray]):
    """Two landmark sources (P,T,68,3): where a part of one is a tangle (bad_regions) and the same part of the
    other is a face, the good one stands in for it, so the fused landmarks follow it alone. Copies."""
    if a is None or b is None:
        return a, b
    a = a.copy()
    b = b.copy()
    P, T = a.shape[:2]

    def q90(x):
        sc = x[..., 2]
        fin = np.isfinite(sc) & (sc > 0)
        return float(np.percentile(sc[fin], 90)) if fin.sum() > 50 else 1.0
    # the two models score on different scales: a stand-in point carries its score in the other's scale
    ab = q90(a) / max(q90(b), 1e-6)
    for p in range(P):
        for t in range(T):
            fa, fb = np.isfinite(a[p, t, :, :2]).all(), np.isfinite(b[p, t, :, :2]).all()
            if not (fa and fb):
                continue
            ba, bb = bad_regions(a[p, t]), bad_regions(b[p, t])
            if not ba and not bb:
                continue
            for name, idx in REGIONS.items():
                if name in ba and name not in bb:
                    a[p, t, idx] = b[p, t, idx]
                    a[p, t, idx, 2] *= ab
                elif name in bb and name not in ba:
                    b[p, t, idx] = a[p, t, idx]
                    b[p, t, idx, 2] /= ab
    return a, b


def fuse_sources(sources) -> np.ndarray:
    """sources: [(raw (P,N,K,3), weight)]. Each source's scores are normalised by its own 90th
    percentile; positions are averaged by weight x score. Returns (P,N,K,3) with scores ~0..1."""
    acc = None
    for raw, w in sources:
        if raw is None or not np.isfinite(raw[..., :2]).any():
            continue
        sc = raw[..., 2]
        fin = np.isfinite(sc) & (sc > 0)
        q90 = float(np.percentile(sc[fin], 90)) if fin.sum() > 50 else (float(sc[fin].max()) if fin.any() else 1.0)
        s = np.where(np.isfinite(raw[..., :2]).all(-1), np.nan_to_num(sc / max(q90, 1e-6)), 0.0)
        s = np.clip(s, 0, 1.2)
        ww = s * w
        if acc is None:
            acc = [np.nan_to_num(raw[..., :2]) * ww[..., None], ww.copy(), s * w,
                   np.where(np.isfinite(raw[..., :2]).all(-1), w, 0.0)]
        else:
            acc[0] += np.nan_to_num(raw[..., :2]) * ww[..., None]
            acc[1] += ww
            acc[2] += s * w
            # a source that has nothing at this point (the whole-body model's face points dropped on a close-up)
            # does not count against the one that has
            acc[3] += np.where(np.isfinite(raw[..., :2]).all(-1), w, 0.0)
    if acc is None:
        return None
    pos = np.where((acc[1] > 1e-6)[..., None], acc[0] / np.maximum(acc[1], 1e-9)[..., None], np.nan)
    score = acc[2] / np.maximum(acc[3], 1e-9)
    return np.concatenate([pos, score[..., None]], -1).astype(np.float32)


def _track_points(store: TrackStore, fs, frames: np.ndarray, z, R, ok, size, smoothing: float, progress, max_coast=3):
    """Forward-backward Kalman + optical flow + stabilised smoothing for K points per person.
    z (P,T,K,2), R (P,T,K) variances (inf = no measurement), ok (P,T), size (P,T). Returns (P,T,K,2)."""
    P, T, K = R.shape
    Q = P * K
    zq = z.transpose(0, 2, 1, 3).reshape(Q, T, 2)
    Rq = R.transpose(0, 2, 1).reshape(Q, T)
    pq = np.repeat(ok[:, None, :], K, axis=1).reshape(Q, T)
    sq = np.repeat((np.nan_to_num(size, nan=100.0) * 3.0)[:, None, :], K, axis=1).reshape(Q, T)
    use_flow = fs is not None
    brk = store.breaks()
    xf, Pf, _, _ = SV._pass(fs, zq, Rq, pq, sq, frames, use_flow, progress, 0.0, 0.5, max_coast=max_coast, breaks=brk)
    xb, Pb, xbp, Pbp = SV._pass(fs, zq[:, ::-1], Rq[:, ::-1], pq[:, ::-1], sq[:, ::-1], frames[::-1], use_flow,
                                progress, 0.5, 0.5, max_coast=max_coast, breaks=brk)
    xbp, Pbp = xbp[:, ::-1], Pbp[:, ::-1]
    xb, Pb = xb[:, ::-1], Pb[:, ::-1]
    wf = np.where(np.isfinite(Pf), 1 / np.where(np.isfinite(Pf), Pf, 1), 0)
    wb = np.where(np.isfinite(Pbp), 1 / np.where(np.isfinite(Pbp), Pbp, 1), 0)
    ws = wf + wb
    x = np.where((ws > 0)[..., None], (np.nan_to_num(xf) * wf[..., None] + np.nan_to_num(xbp) * wb[..., None])
                 / np.maximum(ws, 1e-12)[..., None], np.nan)
    only_b = (ws == 0) & np.isfinite(Pb)
    x[only_b] = xb[only_b]
    if smoothing > 0.01:
        rel = SV.relative_homographies(store.homog if store.homog_ok.any() else None, frames, 6, brk)
        x = SV.stabilized_smooth(x, np.isfinite(x).all(-1), sq, rel, smoothing)
    return x.reshape(P, K, T, 2).transpose(0, 2, 1, 3)


def solve_face(store: TrackStore, fs: Optional[SV.FrameSource], frame_range: Tuple[int, int],
               smoothing: float = 0.35, progress: Optional[Callable[[float, str], None]] = None) -> None:
    a, b = frame_range
    fr, f106 = clean_sources(store.face_raw, getattr(store, "face106_raw", None))
    fused = fuse_sources([(fr, 1.0), (f106, 2.0)])
    if fused is None or not np.isfinite(fused[..., :2]).any():
        store.has_face = False
        return
    raw = fused
    P = raw.shape[0]
    frames = np.arange(a, b)
    T = len(frames)
    z = raw[:, a:b, :, :2].astype(np.float64)                 # (P,T,68,2)
    s = np.nan_to_num(raw[:, a:b, :, 2], nan=0.0)             # (P,T,68)
    size = np.full((P, T), np.nan)
    frontal = np.zeros((P, T))
    med = np.zeros((P, T))
    for p in range(P):
        for i in range(T):
            lm = raw[p, a + i]
            if not np.isfinite(lm[:, :2]).any():
                continue
            fsz = face_size(lm)
            size[p, i] = fsz
            # judged on the eyes, nose and mouth: under hard side light half the jaw line is in black shadow
            # (or under hair) with low scores, while the features that the edits need are clear
            med[p, i] = float(np.median(s[p, i, 27:68]))
            e1, e2 = lm[36, :2], lm[45, :2]
            if np.isfinite(fsz) and fsz > 0 and np.isfinite(e1).all() and np.isfinite(e2).all():
                frontal[p, i] = np.hypot(*(e2 - e1)) / fsz
        size[p] = SV._smooth_series(size[p], 5)
    present = store.present[:, a:b]
    ok = present & np.isfinite(size) & (size > 10) & (med > 0.4)
    # landmarks of a tiny face are not precise enough for eye-level edits: fade in from 16 to 40 px wide
    size_w = np.clip((np.nan_to_num(size, nan=0.0) - 16.0) / 24.0, 0, 1)
    vis_raw = np.clip((med - 0.4) / 0.3, 0, 1) * np.clip((frontal - 0.18) / 0.15, 0, 1) * size_w * ok
    # a person with her back to the camera has no face to edit, whatever the models make of the back of the head
    fd = getattr(store, "facing", None)
    if fd is not None:
        fdir = fd[:, a:b]
        vis_raw = vis_raw * np.where(np.isfinite(fdir), np.clip((fdir + 0.25) / 0.6, 0, 1), 1.0)
        ok = ok & ~(np.isfinite(fdir) & (fdir < -0.3))
    # landmarks of a face that is turned away carry no information
    R = np.where(ok[..., None] & (s > 0.15),
                 (0.010 * np.nan_to_num(size, nan=100.0)[..., None] / np.maximum(s, 0.05)) ** 2, np.inf)
    x = _track_points(store, fs, frames, z, R, ok, size, smoothing, progress)
    # where even the tracked landmarks are a tangle (both models lost on a dark, blurred or turned face in the
    # crowd), face edits and skin masks fade out rather than follow them
    for p in range(P):
        for i in range(T):
            if vis_raw[p, i] > 0 and np.isfinite(x[p, i]).all():
                nb = len(bad_regions(x[p, i]))
                if nb:
                    vis_raw[p, i] *= 0.6 if nb == 1 else 0.0
    # visibility: smoothed, and landmarks dropped where the face is not usable
    k = 3
    vis = np.zeros((P, T), np.float32)
    for p in range(P):
        v = np.pad(vis_raw[p], k, mode="edge")
        vis[p] = np.lib.stride_tricks.sliding_window_view(v, 2 * k + 1).mean(1)
    vis = np.where(present, vis, 0.0)
    out = np.concatenate([x, np.clip(s, 0, 1)[..., None]], -1).astype(np.float32)
    out[vis < 0.08] = np.nan
    store.face[:, a:b] = out
    store.face_vis[:, a:b] = vis
    store.has_face = True


def hand_size(h: np.ndarray) -> float:
    """Wrist to middle finger knuckle (21-point hand), px."""
    w, m = h[0, :2], h[9, :2]
    if np.isfinite(w).all() and np.isfinite(m).all():
        return float(np.hypot(*(m - w)))
    pts = h[np.isfinite(h[:, :2]).all(1), :2]
    if len(pts) >= 4:
        return float(np.hypot(*(pts.max(0) - pts.min(0)))) * 0.45
    return float("nan")


def solve_hands(store: TrackStore, fs: Optional[SV.FrameSource], frame_range: Tuple[int, int],
                smoothing: float = 0.25, progress: Optional[Callable[[float, str], None]] = None) -> None:
    a, b = frame_range
    fused = fuse_sources([(store.hand_raw, 1.0), (store.hand21_raw, 2.0)])
    if fused is None:
        store.has_hands = False
        return
    P = fused.shape[0]
    frames = np.arange(a, b)
    T = len(frames)
    out = np.full((P, T, NH, 3), np.nan, np.float32)
    for h in range(2):
        sl = slice(h * 21, (h + 1) * 21)
        raw = fused[:, a:b, sl]
        z = raw[..., :2].astype(np.float64)
        s = np.nan_to_num(raw[..., 2], nan=0.0)
        size = np.full((P, T), np.nan)
        med = np.zeros((P, T))
        for p in range(P):
            for i in range(T):
                hs = hand_size(raw[p, i])
                size[p, i] = hs
                # judged on the eyes, nose and mouth: under hard side light half the jaw line is in black shadow
            # (or under hair) with low scores, while the features that the edits need are clear
            med[p, i] = float(np.median(s[p, i, 27:68]))
            size[p] = SV._smooth_series(size[p], 5)
        ok = store.present[:, a:b] & np.isfinite(size) & (size > 4) & (med > 0.35)
        R = np.where(ok[..., None] & (s > 0.2), (0.04 * np.nan_to_num(size, nan=40.0)[..., None] / np.maximum(s, 0.05)) ** 2,
                     np.inf)
        x = _track_points(store, fs, frames, z, R, ok, size, smoothing, None, max_coast=2)
        sc = np.where(ok[..., None], np.clip(s, 0, 1), 0.0)
        res = np.concatenate([x, sc[..., None]], -1).astype(np.float32)
        res[~ok] = np.nan
        out[:, :, sl] = res
        if progress:
            progress(0.5 * (h + 1), "손 추적")
    store.hand[:, a:b] = out
    store.has_hands = bool(np.isfinite(out[..., :2]).any())


def occlusion_fade(store: TrackStore, frame_range: Tuple[int, int], min_size: float, r: int = 3) -> None:
    """Fade a face out in the frames something passes in front of it (a hand, a leg in a high kick, a light
    stick, another dancer). The landmark models then still draw a face, on whatever covers it, and its skin
    colour (a leg, an arm) passes the skin model: the retouch would sit on the cover. The separate face
    detector (YuNet) does not see a face there while it sees one in the frames around (it keeps seeing heavily
    motion-blurred faces, so blur alone does not fade), and the 106-point model is much less sure of itself
    than it is on this face elsewhere.

    min_size: face width (source px) below which the face detector is not expected to find the face."""
    yn = getattr(store, "raw", {}).get("face_yunet")
    f106 = getattr(store, "face106_raw", None)
    if yn is None or f106 is None:
        return
    a, b = frame_range
    P = store.face.shape[0]
    for p in range(P):
        det = np.nan_to_num(yn[p, :, 0, 2]) > 0
        sc = np.where(np.isfinite(f106[p, :, :, 2]).all(1), np.nanmean(f106[p, :, 27:68, 2], 1), np.nan)
        typ = sc[det & np.isfinite(sc)]
        if len(typ) < 3:
            continue
        typ = float(np.median(typ))
        if typ <= 0:
            continue
        for t in range(a, b):
            if det[t] or store.face_vis[p, t] <= 0 or not np.isfinite(store.face[p, t, :, :2]).all():
                continue
            F = store.face[p, t, :, :2]
            if float(np.hypot(*(F[16] - F[0]))) < min_size:
                continue
            nb = [s for s in range(max(0, t - r), min(store.n, t + r + 1))
                  if s != t and det[s] and store.connected(min(s, t), max(s, t))]
            if not nb:
                continue                      # never seen by the detector around here: nothing to compare with
            ratio = float(sc[t] / typ) if np.isfinite(sc[t]) else 0.0
            occ = float(np.clip((0.7 - ratio) / 0.4, 0.0, 1.0))
            if occ > 0:
                store.face_vis[p, t] *= (1.0 - occ)
                if store.face_vis[p, t] < 0.08:
                    store.face[p, t] = np.nan
