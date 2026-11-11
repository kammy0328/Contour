"""Who is who over the whole shot: group-dance identities solved globally instead of link by link.

The frame-to-frame tracker is good at following someone while nobody comes near, and that is where it is cut:
its tracks are split wherever the person's box overlaps someone else's ("contested"), leaving pure pieces.
Then, knowing that the shot has a fixed number of members and that two pieces seen at the same moment can never
be the same person:

1. Appearance adapted to this shot. Body re-ID and colour features of every clean piece are projected so that
   frames of one piece agree and different pieces differ (linear discriminant over the pieces): the projection
   learns what tells these particular dancers apart (hair colour, one jacket) and what does not (the uniform).
   Similarities are turned into evidence (log-likelihood ratios) calibrated on this shot: pieces seen at the
   same time are surely different people, halves of one piece surely the same.
2. Members. The long clean pieces are joined into at most K member paths by a minimum-cost flow over the whole
   shot: every possible join costs how far apart the two ends are in motion, minus the evidence that they look
   like the same person, minus a bonus where the frame-to-frame tracker itself carried the person from one
   piece to the next (on group-dance tests it was right there far more often than any later override); every
   piece covered earns its length. All joins are decided together, so two people who leave and come back at
   the same time are paired by everything known about both, not one after the other.
3. The rest. Short and contested pieces go into the gaps of the member paths, the most certain first: a piece
   fits a member when it continues where that member was heading and leads into where the member reappears,
   with a bonus where the frame-to-frame tracker had already followed it there.
4. Anyone else (more people than K, extras at the back) is kept as separate tracks.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import assoc as A

CONTEST = 0.3          # box overlap (IoU) with another detection that makes a frame contested
P = dict(
    R=0.15,            # reward per covered frame
    E=1.0,             # cost of starting or ending a member path (cheap at the ends of the shot)
    G=0.004,           # cost per frame of gap in a join
    GMAX=300,          # longest join (frames)
    W_MOT=1.0, DMAX=3.0,
    W_APP=0.5, W_FACE=0.5,
    MAXSUCC=10,        # candidate joins kept per piece
    AN_MIN=8,          # frames for a clean piece to take part in the member flow
    SHRINK=0.5,        # same-person similarity taken halfway from different-people to halves-of-a-piece
    DIM=32, REG=0.3,   # projection size and regularisation
    ITER=0,            # self-training rounds (re-learning from the members found; off: it amplified early mistakes)
    INS_MAX=2.5, INS_GAP=60, B_INS=2.0,
    LLR_CLIP=5.0,
    EXTRA_MIN=10,      # frames for an extra (non-member) track to be kept
    B_ON=2.0,          # bonus for joining a piece to the frame-to-frame tracker's own next piece
    POS="halves",      # sure-same pairs: consecutive pieces of one frame-to-frame track ("online") or halves
    PAIRS=1,           # projection learnt from people seen together (1) or from pieces as classes (0)
    CUT=1,             # cut the frame-to-frame tracks where boxes overlap (0: keep them whole)
)


class _Piece:
    __slots__ = ("fr", "c", "tid", "t0", "t1", "n", "pe", "ve", "he", "ps", "vs", "hs", "X", "xf", "faces", "fg", "e")


# --------------------------------------------------------------------------- pieces

def _contested(dets, f, i):
    bb = dets[f][i]["box"]
    return any(A.iou(bb, d["box"]) > CONTEST for j, d in enumerate(dets[f]) if j != i)


def _cut(tracks: List[Dict[int, int]], dets):
    if not P["CUT"]:
        return [(dict(tr), False, tid) for tid, tr in enumerate(tracks)]
    out = []
    for tid, tr in enumerate(tracks):
        cur, state = {}, None
        for f in sorted(tr):
            i = tr[f]
            c = _contested(dets, f, i)
            if state is not None and (c != state or (cur and f - max(cur) > 3)):
                out.append((cur, state, tid))
                cur = {}
            cur[f] = i
            state = c
        if cur:
            out.append((cur, state, tid))
    return out


def _state(fr, dets, end, k=8):
    sel = sorted(fr)
    sel = sel[-k:] if end else sel[:k]
    pts = np.array([A._center(dets[f][fr[f]]) for f in sel])
    h = float(np.mean([dets[f][fr[f]]["box"][3] - dets[f][fr[f]]["box"][1] for f in sel]))
    if len(sel) >= 3:
        T = np.array(sel, float)
        tm = T.mean()
        M = np.vstack([T - tm, np.ones_like(T)]).T
        sol, *_ = np.linalg.lstsq(M, pts, rcond=None)
        v = sol[0]
        p = sol[1] + v * ((sel[-1] if end else sel[0]) - tm)
        v = v * min(1.0, (len(sel) - 1) / 6.0)          # few frames: a noisy direction, trust it less
    else:
        v = np.zeros(2)
        p = pts[-1] if end else pts[0]
    return p, v, max(1.0, h)


def _feat(d, use_reid):
    a = d.get("app")
    if a is None or not np.any(a):
        return None
    a = a / (np.linalg.norm(a) + 1e-9)
    if not use_reid:
        return a
    r = d.get("reid")
    if r is None or not np.any(r):
        return None
    return np.concatenate([r, a])


def _build(raw, dets, use_reid):
    ps = []
    dim = 584 if use_reid else 72
    for fr, contested, tid in raw:
        p = _Piece()
        p.fr, p.c, p.tid = fr, contested, tid
        p.t0, p.t1, p.n = min(fr), max(fr), len(fr)
        p.pe, p.ve, p.he = _state(fr, dets, True)
        p.ps, p.vs, p.hs = _state(fr, dets, False)
        X = [] if contested else [(f, _feat(dets[f][i], use_reid)) for f, i in sorted(fr.items())
                                  if P["CUT"] or not _contested(dets, f, i)]
        X = [(f, x) for f, x in X if x is not None]
        p.xf = [f for f, _ in X]
        p.X = np.array([x for _, x in X], np.float64) if X else np.zeros((0, dim))
        p.faces = [dets[f][i]["face"]["feat"] for f, i in sorted(fr.items())
                   if dets[f][i].get("face") is not None and dets[f][i]["face"].get("feat") is not None]
        p.fg = p.faces[-A.FACE_MAX:] if len(p.faces) > A.FACE_MAX else p.faces
        p.e = None
        ps.append(p)
    return ps


# --------------------------------------------------------------------------- appearance, adapted to the shot

def _adapt(groups):
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 3:
        return None
    from scipy.linalg import eigh
    allX = np.vstack(groups)
    mu = allX.mean(0)
    d = allX.shape[1]
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for G in groups:
        m = G.mean(0)
        D = G - m
        Sw += D.T @ D
        Sb += len(G) * np.outer(m - mu, m - mu)
    Sw /= len(allX)
    Sb /= len(allX)
    Sw += P["REG"] * np.trace(Sw) / d * np.eye(d)
    try:
        _, V = eigh(Sb, Sw)
    except Exception:
        return None
    return mu, V[:, ::-1][:, :P["DIM"]]


def _embed(X, tf):
    if tf is None or len(X) == 0:
        return None
    mu, V = tf
    Z = (X - mu) @ V
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    e = Z.mean(0)
    return e / (np.linalg.norm(e) + 1e-9)


class _Calib:
    """Log-likelihood ratio 'same person' of a similarity, from this shot's sure-same and sure-different pairs."""

    def __init__(self, pos, neg, shrink=None):
        self.ok = len(pos) >= 4 and len(neg) >= 4
        if self.ok:
            mp, sp = float(np.mean(pos)), max(0.03, float(np.std(pos)))
            self.mn, self.sn = float(np.mean(neg)), max(0.03, float(np.std(neg)))
            self.ok = mp > self.mn
            # halves of one piece look more alike than one person a minute apart: positives taken halfway
            self.mp = self.mn + (P["SHRINK"] if shrink is None else shrink) * (mp - self.mn)
            self.sp = max(sp, self.sn)

    def __call__(self, s):
        if not self.ok or s is None or not np.isfinite(s):
            return 0.0
        s = float(np.clip(s, self.mn, self.mp))           # beyond the means the tails would decide arbitrarily
        lp = -0.5 * ((s - self.mp) / self.sp) ** 2 - np.log(self.sp)
        ln = -0.5 * ((s - self.mn) / self.sn) ** 2 - np.log(self.sn)
        return float(np.clip(lp - ln, -P["LLR_CLIP"], P["LLR_CLIP"]))


def _adapt_pairs(ps, dets, use_reid):
    """Projection from what differs between people seen in the same frame (in a synchronised dance they are in
    the same pose and light, so what differs is who they are) against what varies within one piece (pose,
    motion blur, light over time). Pieces of one person at different times are never pushed apart."""
    from scipy.linalg import eigh
    clean = [p for p in ps if len(p.X) >= 5]
    if len(clean) < 2:
        return None
    d = clean[0].X.shape[1]
    Sw = np.zeros((d, d))
    nw = 0
    for p in clean:
        D = p.X - p.X.mean(0)
        Sw += D.T @ D
        nw += len(D)
    # per frame, the clean frames of different pieces
    at = {}
    for k, p in enumerate(clean):
        for f, x in zip(p.xf, p.X):
            at.setdefault(f, []).append((k, x))
    Sb = np.zeros((d, d))
    nb = 0
    for f, lst in at.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] == lst[j][0]:
                    continue
                v = lst[i][1] - lst[j][1]
                Sb += np.outer(v, v)
                nb += 1
    if nb < 20 or nw < 20:
        return None
    Sw /= nw
    Sb /= nb
    Sw += P["REG"] * np.trace(Sw) / d * np.eye(d)
    try:
        _, V = eigh(Sb, Sw)
    except Exception:
        return None
    mu = np.vstack([p.X for p in clean]).mean(0)
    return mu, V[:, ::-1][:, :P["DIM"]]


def _appearance(ps, groups=None, dets=None):
    if groups is None and P["PAIRS"]:
        tf = _adapt_pairs(ps, dets, True)
    else:
        tf = _adapt(groups if groups is not None else [p.X for p in ps])
    for p in ps:
        p.e = _embed(p.X, tf)
    pos_a, neg_a, pos_f, neg_f = [], [], [], []
    if P["POS"] == "online":
        # one person at two moments: consecutive clean pieces of one frame-to-frame track, on either side of an
        # overlap (right about 9 times in 10). Unlike two halves of one piece, these differ by what really
        # changes over time in a dance: pose, the light where the person now stands
        by_tid = {}
        for p in ps:
            if not p.c:
                by_tid.setdefault(p.tid, []).append(p)
        for lst in by_tid.values():
            lst.sort(key=lambda p: p.t0)
            for a, b in zip(lst, lst[1:]):
                if a.e is not None and b.e is not None and len(a.X) >= 5 and len(b.X) >= 5:
                    pos_a.append(float(a.e @ b.e))
                if len(a.faces) >= 4 and len(b.faces) >= 4:
                    pos_f.append(A.face_mean(a.fg, b.fg))
    if len(pos_a) < 4 or len(pos_f) < 4:
        ha, hf = [], []
        for p in ps:
            if tf is not None and len(p.X) >= 16:
                k = len(p.X) // 3
                ha.append(float(_embed(p.X[:k], tf) @ _embed(p.X[-k:], tf)))
            if len(p.faces) >= 8:
                k = len(p.faces) // 2
                hf.append(A.face_mean(p.faces[:k], p.faces[k:]))
        if len(pos_a) < 4:
            pos_a = ha
        if len(pos_f) < 4:
            pos_f = hf
    cand = sorted([p for p in ps if (p.e is not None and len(p.X) >= 5) or len(p.faces) >= 4], key=lambda p: p.t0)
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if b.t0 > a.t1:
                break
            if a.e is not None and b.e is not None and len(a.X) >= 5 and len(b.X) >= 5:
                neg_a.append(float(a.e @ b.e))
            if len(a.faces) >= 4 and len(b.faces) >= 4:
                neg_f.append(A.face_mean(a.fg, b.fg))
    sh = 1.0 if P["POS"] == "online" else None
    return _Calib(pos_a, neg_a, sh), _Calib(pos_f, neg_f, sh)


def _app_llr(a, b, ca, cf):
    s = 0.0
    if a.e is not None and b.e is not None:
        s += P["W_APP"] * min(1.0, min(len(a.X), len(b.X)) / 10.0) * ca(float(a.e @ b.e))
    if len(a.faces) >= 3 and len(b.faces) >= 3:
        s += P["W_FACE"] * min(1.0, min(len(a.faces), len(b.faces)) / 8.0) * cf(A.face_mean(a.fg, b.fg))
    return s


# --------------------------------------------------------------------------- motion and joins

def _motion(a, b):
    """Distance (body heights, more slack for longer gaps) between where a was heading and where b came from."""
    gap = b.t0 - a.t1
    g = min(gap, 12)
    pa = a.pe + a.ve * g * 0.5
    pb = b.ps - b.vs * g * 0.5
    return float(np.hypot(*(pa - pb)) / (0.5 * (a.he + b.hs) * (0.30 + 0.02 * gap)))


def _join(a, b, ca, cf):
    gap = b.t0 - a.t1
    if gap <= 0 or gap > P["GMAX"]:
        return None
    on = P["B_ON"] if a.tid == b.tid and _next_an.get(id(a)) is b else 0.0
    return P["W_MOT"] * min(_motion(a, b), P["DMAX"]) - _app_llr(a, b, ca, cf) + P["G"] * gap - on


_next_an: Dict[int, "_Piece"] = {}


def _flow(ps, K, n_frames, ca, cf):
    """At most K disjoint paths through the pieces, minimum total cost (linear programme, integral)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n = len(ps)
    if n == 0 or K <= 0:
        return []
    S, T = 0, 1
    arcs = []
    for i, p in enumerate(ps):
        arcs.append((2 + 2 * i, 3 + 2 * i, -P["R"] * p.n, 1))
        arcs.append((S, 2 + 2 * i, P["E"] * (0.1 if p.t0 <= 2 else 1.0), 1))
        arcs.append((3 + 2 * i, T, P["E"] * (0.1 if p.t1 >= n_frames - 3 else 1.0), 1))
    order = sorted(range(n), key=lambda i: ps[i].t0)
    starts = np.array([ps[i].t0 for i in order])
    for i in range(n):
        a = ps[i]
        lo = int(np.searchsorted(starts, a.t1 + 1))
        hi = int(np.searchsorted(starts, a.t1 + P["GMAX"] + 1))
        cands = []
        for j in order[lo:hi]:
            c = _join(a, ps[j], ca, cf)
            if c is not None:
                cands.append((c, j))
        cands.sort()
        for c, j in cands[:P["MAXSUCC"]]:
            arcs.append((3 + 2 * i, 2 + 2 * j, c, 1))
    arcs.append((S, T, 0.0, K))
    m, N = len(arcs), 2 + 2 * n
    rows = np.empty(2 * m, np.int64)
    cols = np.repeat(np.arange(m), 2)
    vals = np.tile([1.0, -1.0], m)
    rows[0::2] = [a[0] for a in arcs]
    rows[1::2] = [a[1] for a in arcs]
    Aeq = coo_matrix((vals, (rows, cols)), shape=(N, m)).tocsr()
    beq = np.zeros(N)
    beq[S], beq[T] = K, -K
    cost = np.array([a[2] for a in arcs])
    ub = np.array([a[3] for a in arcs], float)
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=np.column_stack([np.zeros(m), ub]), method="highs")
    if res.x is None:
        return []
    x = np.round(res.x).astype(int)
    nxt, heads = {}, []
    for k, (u, v, _, _) in enumerate(arcs):
        if x[k] <= 0 or (u == S and v == T):
            continue
        if u == S:
            heads.append((v - 2) // 2)
        elif v != T and u % 2 == 1 and v % 2 == 0:
            nxt[(u - 3) // 2] = (v - 2) // 2
    paths = []
    for s in heads:
        path = [s]
        while path[-1] in nxt and len(path) <= n:
            path.append(nxt[path[-1]])
        paths.append(path)
    return paths


def _online(a, b):
    return P["B_INS"] if a.tid == b.tid and 0 < b.t0 - a.t1 <= 4 else 0.0


def _insert(paths, ps, rest, ca, cf):
    """Short and contested pieces into the gaps of the member paths, the most certain first."""
    paths = [sorted(p, key=lambda i: ps[i].t0) for p in paths]
    todo = set(rest)
    if not paths:
        return paths, todo

    def cost(path, q):
        Q = ps[q]
        prev = nxt = None
        for i in path:
            p = ps[i]
            if p.t1 >= Q.t0 and p.t0 <= Q.t1:
                return None
            if p.t1 < Q.t0:
                prev = p
            elif nxt is None:
                nxt = p
        cs = []
        if prev is not None and Q.t0 - prev.t1 <= P["INS_GAP"]:
            cs.append(min(_motion(prev, Q), P["DMAX"]) - 0.5 * _app_llr(prev, Q, ca, cf)
                      + P["G"] * (Q.t0 - prev.t1) - _online(prev, Q))
        if nxt is not None and nxt.t0 - Q.t1 <= P["INS_GAP"]:
            cs.append(min(_motion(Q, nxt), P["DMAX"]) - 0.5 * _app_llr(Q, nxt, ca, cf)
                      + P["G"] * (nxt.t0 - Q.t1) - _online(Q, nxt))
        if not cs:
            return None
        return float(np.mean(cs)) + 0.3 * float(np.max(cs))

    C = {q: [cost(p, q) for p in paths] for q in todo}
    while todo:
        best = None
        for q in todo:
            cs = [c if c is not None else 1e6 for c in C[q]]
            o = np.argsort(cs)
            c1 = cs[o[0]]
            if c1 >= P["INS_MAX"]:
                continue
            c2 = cs[o[1]] if len(cs) > 1 else 1e6
            key = min(c2, P["INS_MAX"] + 1) - c1 + 0.02 * ps[q].n
            if best is None or key > best[0]:
                best = (key, q, int(o[0]))
        if best is None:
            break
        _, q, k = best
        paths[k].append(q)
        paths[k].sort(key=lambda i: ps[i].t0)
        todo.discard(q)
        Q = ps[q]
        for r in todo:
            R_ = ps[r]
            if abs(R_.t0 - Q.t0) <= P["INS_GAP"] + Q.n + R_.n:
                C[r][k] = cost(paths[k], r)
    return paths, todo


def _extras(ps, rest, ca, cf):
    """People beyond the members (or pieces nobody could place): chained by the same joins, kept when long."""
    rest = sorted(rest, key=lambda i: ps[i].t0)
    chains: List[List[int]] = []
    for q in rest:
        best = None
        for ch in chains:
            last = ps[ch[-1]]
            c = _join(last, ps[q], ca, cf)
            if c is not None and c < P["INS_MAX"] and (best is None or c < best[0]):
                best = (c, ch)
        if best is None:
            chains.append([q])
        else:
            best[1].append(q)
    return [ch for ch in chains if sum(ps[i].n for i in ch) >= P["EXTRA_MIN"]]


def estimate_members(dets) -> int:
    c = np.array([len(f) for f in dets])
    return int(np.percentile(c, 90)) if len(c) else 0


def assign(tracks: List[Dict[int, int]], dets, members: Optional[int] = None) -> List[Dict[int, int]]:
    """Frame-to-frame tracks (frame -> detection index) -> one track per person over the whole shot."""
    if not tracks:
        return []
    n_frames = len(dets)
    n_reid = sum(1 for fd in dets for d in fd if d.get("reid") is not None and np.any(d["reid"]))
    n_all = max(1, sum(len(fd) for fd in dets))
    ps = _build(_cut(tracks, dets), dets, use_reid=n_reid >= 0.5 * n_all)
    ca, cf = _appearance(ps, dets=dets)
    K = int(members) if members else estimate_members(dets)
    K = max(1, K)
    an = [i for i, p in enumerate(ps) if not p.c and p.n >= P["AN_MIN"]]
    sub = [ps[i] for i in an]
    # the frame-to-frame tracker's own continuation of each long piece (across the overlaps it went through)
    _next_an.clear()
    last = {}
    for p in sorted(sub, key=lambda p: p.t0):
        if p.tid in last:
            _next_an[id(last[p.tid])] = p
        last[p.tid] = p
    paths = _flow(sub, K, n_frames, ca, cf)
    for _ in range(P["ITER"]):
        # self-training: the members found so far teach the projection what tells them apart in this shot
        groups = [np.vstack([sub[i].X for i in path if len(sub[i].X)]) if any(len(sub[i].X) for i in path)
                  else np.zeros((0, sub[0].X.shape[1])) for path in paths]
        groups = [g for g in groups if len(g) >= 10]
        if len(groups) < 2:
            break
        ca, cf = _appearance(ps, groups)
        paths = _flow(sub, K, n_frames, ca, cf)
    paths = [[an[i] for i in path] for path in paths]
    used = {i for p in paths for i in p}
    rest = [i for i in range(len(ps)) if i not in used]
    paths, left = _insert(paths, ps, rest, ca, cf)
    paths += _extras(ps, left, ca, cf)
    out = []
    for path in paths:
        fr: Dict[int, int] = {}
        for i in path:
            fr.update(ps[i].fr)
        if fr:
            out.append(fr)
    return out
