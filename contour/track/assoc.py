"""Multi-person association: pose similarity + box overlap + appearance, then tracklet re-identification."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from . import skeleton as S

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _greedy_assign(cost: np.ndarray):
    rows, cols = [], []
    c = cost.copy()
    while c.size and np.isfinite(c).any():
        i, j = np.unravel_index(np.nanargmin(np.where(np.isfinite(c), c, np.inf)), c.shape)
        if not np.isfinite(c[i, j]):
            break
        rows.append(i)
        cols.append(j)
        c[i, :] = np.inf
        c[:, j] = np.inf
    return np.array(rows, int), np.array(cols, int)


def assign(cost: np.ndarray, max_cost: float):
    if cost.size == 0:
        return []
    ok = np.isfinite(cost) & (cost <= max_cost)
    if linear_sum_assignment is not None:
        # a pair that may not match costs as much as leaving both unmatched (not "anything goes, but match as
        # many as possible": that would trade one clearly right pair for two doubtful ones)
        big = np.where(ok, cost, max_cost + 1e-3)
        r, c = linear_sum_assignment(big)
    else:
        big = np.where(ok, cost, 1e6)
        r, c = _greedy_assign(np.where(ok, cost, np.inf))
    return [(i, j) for i, j in zip(r, c) if ok[i, j]]


# --------------------------------------------------------------------------- appearance

def appearance(rgb: np.ndarray, kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Hellinger-normalised HSV histograms of torso and legs (72 dims)."""
    Hf, Wf = rgb.shape[:2]
    xy = kp[:, :2]
    ok = np.isfinite(xy).all(1) & (kp[:, 2] > 0.3)
    # work on the person's part of the frame only (same pixels, a fraction of the cost)
    pts = [np.asarray(box[:2], np.float64), np.asarray(box[2:4], np.float64)]
    if ok.any():
        pts += [xy[ok].min(0), xy[ok].max(0)]
    P = np.array(pts)
    x0, y0 = np.floor(np.clip(P.min(0) - 2, 0, [Wf, Hf])).astype(int)
    x1, y1 = np.ceil(np.clip(P.max(0) + 2, 0, [Wf, Hf])).astype(int)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros(72, np.float32)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]
    kp = kp.copy()
    kp[:, 0] -= x0
    kp[:, 1] -= y0
    box = np.asarray(box, np.float64) - [x0, y0, x0, y0]
    feats = []
    xy = kp[:, :2]

    def region_mask(poly_idx, fallback_frac):
        m = np.zeros((H, W), np.uint8)
        if all(ok[i] for i in poly_idx):
            # convex hull: the same region whatever way left and right are labelled (people turning around)
            pts = cv2.convexHull(np.round(xy[poly_idx]).astype(np.int32))
            cv2.fillConvexPoly(m, pts, 255)
        else:
            x1, y1, x2, y2 = box
            hh = y2 - y1
            a, b = fallback_frac
            cv2.rectangle(m, (int(x1 + (x2 - x1) * 0.25), int(y1 + hh * a)), (int(x2 - (x2 - x1) * 0.25), int(y1 + hh * b)), 255, -1)
        return m

    regions = [([S.L_SH, S.R_SH, S.R_HIP, S.L_HIP], (0.2, 0.5)), ([S.L_HIP, S.R_HIP, S.R_KN, S.L_KN], (0.5, 0.75))]
    for poly, fb in regions:
        m = region_mask(poly, fb)
        if m.sum() < 255 * 30:
            feats.append(np.zeros(36, np.float32))
            continue
        h = cv2.calcHist([hsv], [0, 1], m, [8, 4], [0, 180, 0, 256]).ravel()
        v = cv2.calcHist([hsv], [2], m, [4], [0, 256]).ravel()
        f = np.concatenate([h / max(h.sum(), 1), v / max(v.sum(), 1)])
        feats.append(np.sqrt(f).astype(np.float32))
    return np.concatenate(feats)


def app_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or not a.any() or not b.any():
        return 0.5
    sims = []
    for s in (slice(0, 36), slice(36, 72)):
        x, y = a[s], b[s]
        if x.any() and y.any():
            # Bhattacharyya coefficient of the two sub-histograms (each sums to 2 after sqrt-normalising halves)
            sims.append(float(np.dot(x, y) / 2.0))
    return float(np.mean(sims)) if sims else 0.5


# --------------------------------------------------------------------------- similarity

def oks(kp_a: np.ndarray, kp_b: np.ndarray, area: float) -> float:
    """Keypoint similarity, the same whether or not left and right got swapped (a turning dancer)."""
    from .orient import swap_all
    return max(_oks(kp_a, kp_b, area), _oks(kp_a, swap_all(kp_b), area))


def _oks(kp_a: np.ndarray, kp_b: np.ndarray, area: float) -> float:
    ok = (kp_a[:, 2] > 0.3) & (kp_b[:, 2] > 0.3) & np.isfinite(kp_a[:, :2]).all(1) & np.isfinite(kp_b[:, :2]).all(1)
    if ok.sum() < 3:
        return 0.0
    d2 = ((kp_a[ok, :2] - kp_b[ok, :2]) ** 2).sum(1)
    k2 = (2 * S.OKS_SIGMAS[ok]) ** 2
    return float(np.mean(np.exp(-d2 / (2 * max(area, 1.0) * k2))))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- tracker
#
# Frame to frame: pose similarity (left/right swaps allowed), box overlap, body re-ID and colour.
# Then, over the whole shot (analysis is offline, so the future is known):
#   1. where two people overlapped (crossing), who each was before is compared with who each is after, by
#      body re-ID, face and where each was heading; tails are swapped when crossed fits better
#   2. tracklets of the same person (lost behind someone, out of frame, turned away) are joined, all candidate
#      links solved together, by where each was heading, how each looks and, decisively, by face.

# frame to frame the body geometry decides; re-ID mostly matters when joining pieces (identical outfits make it
# noisy between neighbours), so it only nudges here
W_OKS, W_IOU, W_REID, W_APP = 0.6, 0.25, 0.05, 0.10
REID_LO, REID_HI = 0.35, 0.80          # cosine mapped to 0..1
CROSS_FIX = True
# SFace galleries are compared by their mean pairwise cosine: the best single pair is unreliable (in a group of
# similar faces some pair of different people always scores high). Calibrated on small (~50 px) group-dance
# faces: same person p5 ~0.31, different people p95 ~0.35 for galleries of 6.
FACE_SAME, FACE_DIFF, FACE_MIN, FACE_MAX = 0.40, 0.22, 4, 24
FACE_SURE_SAME, FACE_SURE_DIFF, FACE_SURE_N = 0.45, 0.15, 6     # decisive only well beyond the overlap


def _rs(c):
    return float(np.clip((c - REID_LO) / (REID_HI - REID_LO), 0, 1))


class _Track:
    def __init__(self, tid, t, det, iso):
        self.id = tid
        self.frames: Dict[int, int] = {t: det['i']}
        self.kp = det['kp'].copy()
        self.vel = np.zeros((det['kp'].shape[0], 2))
        self.box = det['box'].copy()
        self.box0 = det['box'].copy()
        self.app = det['app'].copy()
        self.app_first = det['app'].copy()
        self.app_recent = det['app'].copy()
        self.n_app = 1
        r = det.get('reid')
        self.reid = r.copy() if r is not None and iso else (r.copy() * 0.3 if r is not None else None)
        self.reid_first = self.reid.copy() if self.reid is not None else None
        self.feats = {}              # frame -> reid (isolated detections only)
        self.faces = {}              # frame -> face feat
        if r is not None and iso:
            self.feats[t] = r
        if det.get('face') is not None:
            self.faces[t] = det['face']['feat']
        self.last = t
        self.rec_run = 0
        self.conf_ref = _core_conf(det['kp'])
        self.fbox = det.get('fbox')          # last clearly seen face (x, y, w, h)
        self.fbox_t = t if self.fbox is not None else -999

    def predict(self, t):
        gap = t - self.last
        kp = self.kp.copy()
        kp[:, :2] += self.vel * min(gap, 5)
        return kp

    def update(self, t, det, iso):
        gap = max(1, t - self.last)
        v = (det['kp'][:, :2] - self.kp[:, :2]) / gap
        good = np.isfinite(v).all(1)
        self.vel[good] = 0.6 * v[good] + 0.4 * self.vel[good]
        self.kp = det['kp'].copy()
        self.box = det['box'].copy()
        if det['app'].any():
            self.app = (self.app * self.n_app + det['app']) / (self.n_app + 1)
            if len(self.frames) < 5:
                self.app_first = (self.app_first * len(self.frames) + det['app']) / (len(self.frames) + 1)
            self.app_recent = 0.7 * self.app_recent + 0.3 * det['app'] if self.app_recent.any() else det['app'].copy()
            self.n_app = min(self.n_app + 1, 30)
        r = det.get('reid')
        if r is not None and r.any():
            w = 0.15 if iso else 0.02
            self.reid = r.copy() if self.reid is None or not self.reid.any() else (1 - w) * self.reid + w * r
            self.reid /= np.linalg.norm(self.reid) + 1e-9
            if iso:
                self.feats[t] = r
        if det.get('face') is not None:
            self.faces[t] = det['face']['feat']
        if det.get('fbox') is not None:
            self.fbox, self.fbox_t = det['fbox'], t
        self.frames[t] = det['i']
        self.last = t
        if det.get('recovered'):
            self.rec_run += 1
        else:
            self.rec_run = 0
            c = _core_conf(det['kp'])
            if np.isfinite(c):
                self.conf_ref = c if not np.isfinite(self.conf_ref) else 0.8 * self.conf_ref + 0.2 * c

    def pose_box(self, t):
        """Where to look for this person at frame t: the last box moved along with the body."""
        gap = t - self.last
        v = self.vel[[S.L_SH, S.R_SH, S.L_HIP, S.R_HIP]]
        v = v[np.isfinite(v).all(1)]
        d = v.mean(0) * gap if len(v) else np.zeros(2)
        return (self.box + np.r_[d, d]).astype(np.float32)


def _core_conf(kp):
    c = kp[[S.L_SH, S.R_SH, S.L_HIP, S.R_HIP], 2]
    c = c[np.isfinite(c)]
    return float(np.median(c)) if len(c) else float("nan")


def _kp_extent(kp):
    ok = np.isfinite(kp[:, :2]).all(1) & (kp[:, 2] > 0.3)
    if ok.sum() < 4:
        return None
    return np.r_[kp[ok, :2].min(0), kp[ok, :2].max(0)]


def _carry_box(tr, kp):
    """Box for a pose found without a detection: the last box, moved and scaled with the skeleton."""
    a, b = _kp_extent(tr.kp), _kp_extent(kp)
    if a is None or b is None:
        return None
    ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    s = float(np.clip((b[3] - b[1]) / max(a[3] - a[1], 1.0), 0.85, 1.18))
    c = (tr.box[:2] + tr.box[2:]) / 2 + (cb - ca)
    half = (tr.box[2:] - tr.box[:2]) / 2 * s
    return np.r_[c - half, c + half].astype(np.float32)


def _isolated(frame_dets):
    n = len(frame_dets)
    iso = [True] * n
    for i in range(n):
        for j in range(n):
            if i != j and iou(frame_dets[i]['box'], frame_dets[j]['box']) > 0.25:
                iso[i] = False
    return iso


REC_RUN = 30          # frames in a row a person may be carried by pose alone (no detection)
REC_KEEP = 0.6        # ... while the pose is at least this sure, relative to the person's usual
REC_MOVE = 0.25       # ... and moves on from where the person was heading
REC_CLEAR = 0.2       # ... and nobody else's box overlaps the place more than this


ONLINE_GAP = 3        # frame-to-frame matching only bridges short misses; longer ones go to the linking, which
                      # weighs all candidates together (on group-dance tests a long online gap swapped people)


GLOBAL = False         # identities solved over the whole shot (members.py) instead of greedy linking


def track_people(dets: List[List[dict]], max_gap: int = ONLINE_GAP, recover=None, members: Optional[int] = None):
    """dets: per frame, the detections (dicts with i, box, kp, score, app, optional reid and face).

    recover(t, boxes) -> per box a dict(kp=(17,3), app=..., raw=...) or None: pose estimation on a box where a
    person is expected but the detector found nobody (motion blur, crouching, strobe, half hidden). Accepted
    poses are appended to dets[t] with recovered=True."""
    active, done = [], []
    nid = 0
    for t, frame_dets in enumerate(dets):
        active_now = [tr for tr in active if t - tr.last <= max_gap]
        done += [tr for tr in active if tr not in active_now]
        active = active_now
        if frame_dets:
            nid = _step(t, frame_dets, active, nid)
        if recover is not None:
            _recover(t, frame_dets, active, recover)
    done += active
    tracks = [t for t in done if len(t.frames) >= 3]
    if CROSS_FIX:
        tracks = _resolve_crossings(tracks, dets)
        tracks = _resolve_hops(tracks, dets)
    if FACE_SPLIT:
        tracks = _split_by_face(tracks, dets)
    if GLOBAL:
        from . import members as MB
        try:
            return MB.assign([t.frames for t in tracks], dets, members)
        except Exception:
            import traceback
            traceback.print_exc()
    return _merge(tracks, dets)


FACE_SPLIT = True
SPLIT_WIN = 10        # faces on each side of a candidate switch


def _split_by_face(tracks, dets):
    """A track whose faces before some point clearly belong to someone else than after it followed the wrong
    person somewhere in between (a crossing the overlap test missed, a long gap). Cut it there; the linking
    then joins each piece to the right person."""
    out = []
    for tr in tracks:
        out += _split_one(tr, dets)
    return out


def _split_one(tr, dets):
    ff = sorted(tr.faces)
    n = len(ff)
    if n < 2 * SPLIT_WIN:
        return [tr]
    F = np.stack([tr.faces[f] for f in ff])
    best, best_k = 0.0, None
    for k in range(SPLIT_WIN, n - SPLIT_WIN + 1):
        A_, B_ = F[k - SPLIT_WIN:k], F[k:k + SPLIT_WIN]
        cross = float((A_ @ B_.T).mean())
        if cross > FACE_DIFF:
            continue
        sa = float((A_ @ A_.T)[np.triu_indices(SPLIT_WIN, 1)].mean())
        sb = float((B_ @ B_.T)[np.triu_indices(SPLIT_WIN, 1)].mean())
        margin = min(sa, sb) - cross
        if min(sa, sb) >= FACE_SAME and margin > best:
            best, best_k = margin, k
    if best_k is None:
        return [tr]
    lo, hi = ff[best_k - 1], ff[best_k]
    cut = _switch_frame(tr, dets, lo, hi)
    a_fr = {f: i for f, i in tr.frames.items() if f < cut}
    b_fr = {f: i for f, i in tr.frames.items() if f >= cut}
    if len(a_fr) < 3 or len(b_fr) < 3:
        return [tr]
    pieces = []
    for fr in (a_fr, b_fr):
        t0 = min(fr)
        p = _Track(tr.id, t0, dets[t0][fr[t0]], False)
        p.frames = dict(fr)
        p.feats = {f: v for f, v in tr.feats.items() if f in fr}
        p.faces = {f: v for f, v in tr.faces.items() if f in fr}
        apps = [dets[f][i]['app'] for f, i in fr.items() if dets[f][i]['app'].any()]
        if apps:
            p.app = np.mean(apps, 0)
            p.app_first = np.mean(apps[:5], 0)
            p.app_recent = np.mean(apps[-5:], 0)
        lf = max(fr)
        p.last, p.box, p.kp = lf, dets[lf][fr[lf]]['box'], dets[lf][fr[lf]]['kp']
        p.box0 = dets[t0][fr[t0]]['box']
        pieces.append(p)
    # a piece may hold another switch
    return _split_one(pieces[0], dets) + _split_one(pieces[1], dets)


def _switch_frame(tr, dets, lo, hi):
    """Where between two face samples the track most likely jumped to someone else: the biggest jump of the
    torso relative to body size (a gap, a crossing), else the middle."""
    fr = [f for f in sorted(tr.frames) if lo <= f <= hi]
    best, cut = 0.0, (lo + hi + 1) // 2
    for f0, f1 in zip(fr, fr[1:]):
        d0, d1 = dets[f0][tr.frames[f0]], dets[f1][tr.frames[f1]]
        h = max(1.0, d0['box'][3] - d0['box'][1])
        j = np.hypot(*(_center(d1) - _center(d0))) / h / (f1 - f0) ** 0.5 + 0.02 * (f1 - f0)
        if j > best:
            best, cut = j, f1
    return cut


def _recover(t, frame_dets, active, recover):
    cands = [tr for tr in active if tr.last == t - 1 and len(tr.frames) >= 5 and tr.rec_run < REC_RUN
             and np.isfinite(tr.conf_ref)]
    if not cands:
        return
    boxes = [tr.pose_box(t) for tr in cands]
    # only a person standing clear of the others: where people overlap, a pose found without a detection is as
    # likely to be the neighbour (that case is left to the linking of tracklets, which compares identities)
    others = [d['box'] for d in frame_dets]
    clear = []
    for i, b in enumerate(boxes):
        o = others + [boxes[j] for j in range(len(boxes)) if j != i]
        if all(iou(b, x) < REC_CLEAR for x in o):
            clear.append(i)
    if not clear:
        return
    cands = [cands[i] for i in clear]
    boxes = np.array([boxes[i] for i in clear], np.float32)
    res = recover(t, boxes)
    if not res:
        return
    for tr, r in zip(cands, res):
        if r is None:
            continue
        kp = r['kp']
        c = _core_conf(kp)
        if not np.isfinite(c) or c < REC_KEEP * tr.conf_ref:
            continue
        box = _carry_box(tr, kp)
        if box is None:
            continue
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        if oks(tr.predict(t), kp, area) < REC_MOVE:
            continue
        # the pose found there belongs to someone already seen in this frame: the person is hidden behind them
        if any(oks(kp, d['kp'], area) > 0.5 for d in frame_dets):
            continue
        d = dict(i=len(frame_dets), box=box, kp=kp, score=0.0, app=r.get('app', np.zeros(72, np.float32)),
                 reid=None, face=None, recovered=True)
        if 'raw' in r:
            d['raw'] = r['raw']
        frame_dets.append(d)
        tr.update(t, d, False)


def _face_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / u) if u > 0 else 0.0


def _step(t, frame_dets, active, nid):
    iso = _isolated(frame_dets)
    cost = np.full((len(active), len(frame_dets)), np.inf)
    for i, tr in enumerate(active):
        pk = tr.predict(t)
        for j, d in enumerate(frame_dets):
            b = d['box']
            area = (b[2] - b[0]) * (b[3] - b[1])
            o = oks(pk, d['kp'], area)
            u = iou(tr.box, b)
            a = max(app_similarity(tr.app, d['app']), app_similarity(tr.app_recent, d['app']))
            # close-ups: the same clearly seen face a frame or two ago settles it, whatever the body box does
            fb = d.get('fbox')
            fu = -1.0
            if fb is not None and tr.fbox is not None and t - tr.fbox_t <= 3:
                fu = _face_iou(fb, tr.fbox)
            if o < 0.05 and u < 0.05 and fu <= 0.3:
                continue
            r = d.get('reid')
            rs = _rs(float(tr.reid @ r)) if (tr.reid is not None and r is not None and r.any() and tr.reid.any()) else 0.5
            gap_pen = 0.02 * (t - tr.last - 1)
            sim = W_OKS * o + W_IOU * u + W_REID * rs + W_APP * a
            if fu > 0.3:
                sim = max(sim, 0.45 + 0.5 * fu)
            elif 0 <= fu < 0.02:
                sim = min(sim, 0.15)
            cost[i, j] = 1 - sim + gap_pen
    matched = assign(cost, 0.8)
    mi = {j for _, j in matched}
    for i, j in matched:
        active[i].update(t, frame_dets[j], iso[j])
    for j, d in enumerate(frame_dets):
        if j not in mi:
            active.append(_Track(nid, t, d, iso[j]))
            nid += 1
    return nid


def _center(d):
    k = d['kp']
    idx = [5, 6, 11, 12]
    pts = [k[j, :2] for j in idx if np.isfinite(k[j, :2]).all() and k[j, 2] > 0.2]
    if len(pts) >= 2:
        return np.mean(pts, 0)
    b = d['box']
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _fit(tr, dets, frames):
    """Linear fit of the torso centre over the given frames: (t_mean, position at t_mean, velocity) or None."""
    fr = [f for f in frames if f in tr.frames]
    if len(fr) < 3:
        return None
    P = np.array([_center(dets[f][tr.frames[f]]) for f in fr])
    T = np.array(fr, float)
    tm = T.mean()
    A = np.vstack([T - tm, np.ones_like(T)]).T
    sol, *_ = np.linalg.lstsq(A, P, rcond=None)
    return tm, sol[1], sol[0]


def _at(fit, t):
    tm, p, v = fit
    return p + v * (t - tm)


MOT_W = 1.0


def _mean_feat(tr, frames):
    fs = [tr.feats[f] for f in frames if f in tr.feats]
    if not fs:
        return None
    m = np.mean(fs, 0)
    return m / (np.linalg.norm(m) + 1e-9)


def _faces(tr, frames):
    fs = [tr.faces[f] for f in frames if f in tr.faces]
    if len(fs) > FACE_MAX:
        fs = [fs[i] for i in np.linspace(0, len(fs) - 1, FACE_MAX).round().astype(int)]
    return fs


def face_mean(a_faces, b_faces) -> float:
    """Mean pairwise cosine of two face galleries (NaN when either is empty)."""
    if not a_faces or not b_faces:
        return float("nan")
    return float((np.stack(a_faces) @ np.stack(b_faces).T).mean())


def _id_sim(a_feat, b_feat, a_faces, b_faces):
    """(similarity 0..1, decisive) from face galleries when both have enough faces, else body re-ID."""
    body = _rs(float(a_feat @ b_feat)) if (a_feat is not None and b_feat is not None) else 0.5
    if len(a_faces) >= FACE_MIN and len(b_faces) >= FACE_MIN:
        fs = face_mean(a_faces, b_faces)
        if min(len(a_faces), len(b_faces)) >= FACE_SURE_N:
            if fs >= FACE_SURE_SAME:
                return 1.0, True
            if fs <= FACE_SURE_DIFF:
                return 0.0, True
        f = float(np.clip((fs - FACE_DIFF) / (FACE_SAME - FACE_DIFF), 0, 1))
        return 0.7 * f + 0.3 * body, False
    return body, False


def _resolve_crossings(tracks, dets, win=25):
    """Two tracks whose boxes overlapped (people crossing) may have swapped people. Compare who each one was
    before the overlap with who it is after it, by re-ID and face; swap the tails when crossed fits better."""
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        for ia in range(len(tracks)):
            for ib in range(ia + 1, len(tracks)):
                A, B = tracks[ia], tracks[ib]
                common = sorted(set(A.frames) & set(B.frames))
                if len(common) < 2:
                    continue
                ov = [f for f in common if iou(dets[f][A.frames[f]]['box'], dets[f][B.frames[f]]['box']) > 0.3]
                if not ov:
                    continue
                # contiguous overlap episodes
                eps = []
                s = ov[0]
                prev = ov[0]
                for f in ov[1:]:
                    if f - prev > 3:
                        eps.append((s, prev))
                        s = f
                    prev = f
                eps.append((s, prev))
                for (s, e) in eps:
                    bef = range(s - win, s)
                    aft = range(e + 1, e + 1 + win)
                    fa0, fa1 = _mean_feat(A, bef), _mean_feat(A, aft)
                    fb0, fb1 = _mean_feat(B, bef), _mean_feat(B, aft)
                    if any(x is None for x in (fa0, fa1, fb0, fb1)):
                        continue
                    straight = float(fa0 @ fa1) + float(fb0 @ fb1)
                    crossed = float(fa0 @ fb1) + float(fb0 @ fa1)
                    fs_s = fs_c = 0.0
                    A0, A1, B0, B1 = _faces(A, bef), _faces(A, aft), _faces(B, bef), _faces(B, aft)
                    if min(len(A0), len(A1), len(B0), len(B1)) >= 2:
                        fs_s = face_mean(A0, A1) + face_mean(B0, B1)
                        fs_c = face_mean(A0, B1) + face_mean(B0, A1)
                    # motion: where each was heading before the overlap vs where each comes out of it
                    k = 8
                    ra = _fit(A, dets, range(s - k, s)); rb = _fit(B, dets, range(s - k, s))
                    qa = _fit(A, dets, range(e + 1, e + 1 + k)); qb = _fit(B, dets, range(e + 1, e + 1 + k))
                    mot = 0.0
                    if all(x is not None for x in (ra, rb, qa, qb)):
                        m = (s + e) / 2.0
                        hb = np.mean([dets[f][A.frames[f]]['box'][3] - dets[f][A.frames[f]]['box'][1] for f in common[:3]])
                        err_s = np.hypot(*(_at(ra, m) - _at(qa, m))) + np.hypot(*(_at(rb, m) - _at(qb, m)))
                        err_c = np.hypot(*(_at(ra, m) - _at(qb, m))) + np.hypot(*(_at(rb, m) - _at(qa, m)))
                        mot = MOT_W * float(np.clip((err_s - err_c) / max(hb, 1.0), -1.5, 1.5))
                    if (crossed - straight) + 2 * (fs_c - fs_s) + mot > 0.12:
                        # swap everything after the episode start
                        cut = s
                        ta = {f: i for f, i in A.frames.items() if f >= cut}
                        tb = {f: i for f, i in B.frames.items() if f >= cut}
                        for f in ta:
                            del A.frames[f]
                        for f in tb:
                            del B.frames[f]
                        A.frames.update(tb)
                        B.frames.update(ta)
                        for attr in ('feats', 'faces'):
                            da, db = getattr(A, attr), getattr(B, attr)
                            ta2 = {f: v for f, v in da.items() if f >= cut}
                            tb2 = {f: v for f, v in db.items() if f >= cut}
                            for f in ta2:
                                del da[f]
                            for f in tb2:
                                del db[f]
                            da.update(tb2)
                            db.update(ta2)
                        A.last, B.last = max(A.frames), max(B.frames)
                        changed = True
    for tr in tracks:
        lf = max(tr.frames)
        tr.box = dets[lf][tr.frames[lf]]['box']
        ff = min(tr.frames)
        tr.box0 = dets[ff][tr.frames[ff]]['box']
    return tracks


HOP_K = 5             # frames after one track ends in which the other person's new track may start
HOP_MIN = 0.25        # how much better the handed-over version must explain motion and identity


def _resolve_hops(tracks, dets, win=25):
    """The other way two people get swapped where they cross: track A jumps from its person onto person Y, Y's
    own track B ends right there, and A's person carries on in a new track E. Compare A-before -> E and
    B -> A-after with how things are now, by motion, re-ID and face; hand the pieces over when crossed fits."""
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        starts = {}
        for E in tracks:
            starts.setdefault(min(E.frames), []).append(E)
        for B in list(tracks):
            eb = max(B.frames)
            bb = dets[eb][B.frames[eb]]['box']
            hb = max(1.0, bb[3] - bb[1])
            for A in tracks:
                if A is B or eb not in A.frames or (eb + 1) not in A.frames and (eb + 2) not in A.frames:
                    continue
                if iou(dets[eb][A.frames[eb]]['box'], bb) < 0.3:
                    continue
                s = eb + 1
                cands = [E for t in range(s, s + HOP_K + 1) for E in starts.get(t, []) if E is not A and E is not B]
                best = None
                for E in cands:
                    e0 = min(E.frames)
                    ce = _center(dets[e0][E.frames[e0]])
                    if np.hypot(*(ce - _center(dets[eb][A.frames[eb]]))) > 0.8 * hb:
                        continue
                    sc = _hop_score(A, B, E, dets, s, win)
                    if sc is not None and sc > HOP_MIN and (best is None or sc > best[0]):
                        best = (sc, E)
                if best is None:
                    continue
                E = best[1]
                # A keeps its past and takes E's future; B takes A's future
                a_post = {f: i for f, i in A.frames.items() if f >= s}
                for f in a_post:
                    del A.frames[f]
                A.frames.update(E.frames)
                B.frames.update(a_post)
                for attr in ('feats', 'faces'):
                    da, db, de = getattr(A, attr), getattr(B, attr), getattr(E, attr)
                    moved = {f: v for f, v in da.items() if f >= s}
                    for f in moved:
                        del da[f]
                    da.update(de)
                    db.update(moved)
                A.last, B.last = max(A.frames), max(B.frames)
                tracks = [t for t in tracks if t is not E]
                changed = True
                break
            if changed:
                break
    for tr in tracks:
        lf = max(tr.frames)
        tr.box = dets[lf][tr.frames[lf]]['box']
    return tracks


def _hop_score(A, B, E, dets, s, win):
    """> 0 when 'A-before + E, B + A-after' explains the shot better than 'A, B ended, E new'."""
    bef = range(s - win, s)
    aft_a = range(s, s + win)
    e0 = min(E.frames)
    aft_e = range(e0, e0 + win)
    fa0, fb0 = _mean_feat(A, bef), _mean_feat(B, bef)
    fa1, fe1 = _mean_feat(A, aft_a), _mean_feat(E, aft_e)
    sc = 0.0
    if not any(x is None for x in (fa0, fb0, fa1, fe1)):
        sc += (float(fa0 @ fe1) + float(fb0 @ fa1)) - (float(fa0 @ fa1) + float(fb0 @ fe1))
    A0, B0, A1, E1 = _faces(A, bef), _faces(B, bef), _faces(A, aft_a), _faces(E, aft_e)
    if min(len(A0), len(B0), len(A1), len(E1)) >= 2:
        sc += 2 * ((face_mean(A0, E1) + face_mean(B0, A1)) - (face_mean(A0, A1) + face_mean(B0, E1)))
    k = 8
    ra, rb = _fit(A, dets, range(s - k, s)), _fit(B, dets, range(s - k, s))
    qa, qe = _fit(A, dets, range(s, s + k)), _fit(E, dets, range(e0, e0 + k))
    if any(x is None for x in (ra, rb, qa, qe)):
        return sc if sc != 0.0 else None
    m = s + 0.5 * (e0 - s)
    hb = max(1.0, float(np.mean([dets[f][A.frames[f]]['box'][3] - dets[f][A.frames[f]]['box'][1]
                                 for f in range(s - k, s) if f in A.frames] or [1.0])))
    err_s = np.hypot(*(_at(ra, m) - _at(qa, m))) + np.hypot(*(_at(rb, m) - _at(qe, m)))
    err_c = np.hypot(*(_at(ra, m) - _at(qe, m))) + np.hypot(*(_at(rb, m) - _at(qa, m)))
    sc += MOT_W * float(np.clip((err_s - err_c) / hb, -1.5, 1.5))
    return sc


def _end_state(tr, dets, end=True, k=8):
    fr = sorted(tr.frames)
    sel = fr[-k:] if end else fr[:k]
    fit = _fit(tr, dets, sel) if len(sel) >= 3 else None
    f0 = sel[-1] if end else sel[0]
    d = dets[f0][tr.frames[f0]]
    c = _center(d)
    h = d['box'][3] - d['box'][1]
    v = fit[2] if fit is not None else np.zeros(2)
    return f0, c, v, h


LINK_W_POS, LINK_W_APP, LINK_W_ID, LINK_MAX = 1.0, 0.6, 0.6, 1.25


def _link_cost(a, b, dets, max_gap):
    a_end = max(a.frames)
    b_start = min(b.frames)
    gap = b_start - a_end
    if gap <= 0 or gap > max_gap:
        return None
    fa = _mean_feat(a, sorted(a.frames)[-30:])
    fb = _mean_feat(b, sorted(b.frames)[:30])
    # faces near the link: a long track may still hold someone else far from it
    ids, decisive = _id_sim(fa, fb, _faces(a, sorted(a.frames)[-90:]), _faces(b, sorted(b.frames)[:90]))
    if decisive and ids < 0.5:
        return None
    col = max(app_similarity(a.app, b.app), app_similarity(a.app_recent, b.app_first))
    _, ca, va, ha = _end_state(a, dets, True)
    _, cb, vb, hb = _end_state(b, dets, False)
    h = max(1.0, 0.5 * (ha + hb))
    # meet in the middle: a carried forward, b carried back, both for at most ~half a second of motion
    g = min(gap, 12)
    pa = ca + va * g * 0.5
    pb = cb - vb * g * 0.5
    dist = np.hypot(*(pa - pb)) / (h * (0.35 + 0.025 * gap))
    if dist > 2.5 and not (decisive and ids >= 1.0):
        return None
    cost = LINK_W_POS * min(dist, 2.5) + LINK_W_APP * (1 - col) + LINK_W_ID * (1 - ids) + 0.002 * gap
    if decisive and ids >= 1.0:
        cost -= 0.8
    return cost


def _merge(tracks, dets, max_gap=150):
    """Join tracklets of the same person across gaps: all candidate links solved together (Hungarian), so two
    people who vanish and reappear at the same time are paired by where each was heading and how each looks."""
    from scipy.optimize import linear_sum_assignment
    while True:
        n = len(tracks)
        C = np.full((n, n), 1e6)
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i == j:
                    continue
                c = _link_cost(a, b, dets, max_gap)
                if c is not None and c < LINK_MAX:
                    C[i, j] = c
        if not (C < 1e6).any():
            break
        r, c = linear_sum_assignment(C)
        pairs = [(i, j) for i, j in zip(r, c) if C[i, j] < 1e6]
        if not pairs:
            break
        # apply links (chains are fine: a->b and b->c)
        target = {i: j for i, j in pairs}
        absorbed = set()
        for i in sorted(target, key=lambda x: min(tracks[x].frames)):
            j = target[i]
            if i in absorbed or j in absorbed:
                continue
            a, b = tracks[i], tracks[j]
            a.frames.update(b.frames)
            a.feats.update(b.feats)
            a.faces.update(b.faces)
            a.app = (a.app + b.app) / 2
            a.app_recent = b.app_recent
            a.box, a.kp, a.last = b.box, b.kp, b.last
            absorbed.add(j)
        tracks = [t for k, t in enumerate(tracks) if k not in absorbed]
    return [t.frames for t in tracks]


def link_tracklets(tracklets: List[Dict[int, int]], dets: List[List[dict]]) -> List[Dict[int, int]]:
    """Join pieces of one person found by another tracker (SAM 2 video): the same linking of tracklets across
    gaps as for the frame-to-frame tracker (where each was heading, how each looks, face)."""
    tracks = []
    for k, fr in enumerate(tracklets):
        fs = sorted(fr)
        if not fs:
            continue
        tr = _Track(k, fs[0], dets[fs[0]][fr[fs[0]]], False)
        for f in fs[1:]:
            tr.update(f, dets[f][fr[f]], True)
        tracks.append(tr)
    return _merge(tracks, dets)


# --------------------------------------------------------------------------- gaps

def fill_gaps(tracklets: List[Dict[int, int]], dets: List[List[dict]], recover, max_len: int = 90) -> int:
    """Carry each person through the frames where the detector lost them (motion blur, crouching, a strobe, a
    dark moment) by pose estimation on where they should be, forwards from before the gap and backwards from
    after it. Identities are already settled; this only fills presence. Returns the number of frames filled."""
    owner = {}
    for k, tr in enumerate(tracklets):
        for f, i in tr.items():
            owner[(f, i)] = k
    filled = 0
    for k, tr in enumerate(tracklets):
        fr = sorted(tr)
        for f0, f1 in zip(fr, fr[1:]):
            if f1 - f0 <= 1 or f1 - f0 > max_len:
                continue
            got = _carry(tr, dets, f0, f1, +1, recover)
            for t, d in got:
                d['i'] = len(dets[t])
                dets[t].append(d)
                tr[t] = d['i']
            last = got[-1][0] if got else f0
            back = _carry(tr, dets, f1, last, -1, recover)
            for t, d in back:
                d['i'] = len(dets[t])
                dets[t].append(d)
                tr[t] = d['i']
            filled += len(got) + len(back)
    return filled


def _carry(tr, dets, start, stop, step, recover):
    """Pose-follow one person from frame start towards stop (exclusive)."""
    d0 = dets[start][tr[start]]
    prev = dets[start - step][tr[start - step]] if (start - step) in tr else None
    st = _Track(-1, start, d0, False)
    st.conf_ref = _core_conf(d0['kp'])
    if prev is not None:
        v = d0['kp'][:, :2] - prev['kp'][:, :2]          # per step of the walk, either direction
        ok = np.isfinite(v).all(1)
        st.vel[ok] = v[ok]
    if not np.isfinite(st.conf_ref):
        return []
    out = []
    t = start + step
    while (t < stop) if step > 0 else (t > stop):
        # predict() and pose_box() count frames forwards; walking backwards the velocity already points back
        st.last = t - 1
        box = st.pose_box(t)
        others = [d['box'] for d in dets[t]]
        if any(iou(box, x) >= REC_CLEAR for x in others):
            break
        res = recover(t, np.array([box], np.float32))
        r = res[0] if res else None
        if r is None:
            break
        kp = r['kp']
        c = _core_conf(kp)
        if not np.isfinite(c) or c < REC_KEEP * st.conf_ref:
            break
        nb = _carry_box(st, kp)
        if nb is None:
            break
        area = float((nb[2] - nb[0]) * (nb[3] - nb[1]))
        if oks(st.predict(t), kp, area) < REC_MOVE or any(oks(kp, d['kp'], area) > 0.5 for d in dets[t]):
            break
        d = dict(i=-1, box=nb, kp=kp, score=0.0, app=r.get('app', np.zeros(72, np.float32)), reid=None, face=None,
                 recovered=True)
        if 'raw' in r:
            d['raw'] = r['raw']
        v = (kp[:, :2] - st.kp[:, :2])
        ok = np.isfinite(v).all(1)
        st.vel[ok] = 0.6 * v[ok] + 0.4 * st.vel[ok]
        st.kp, st.box = kp.copy(), nb
        out.append((t, d))
        t += step
    return out
