"""Who is who across the cuts of an edited timeline.

Every analysed cut has its own people (인물 1, 인물 2 ...). Here they are joined into members over the whole
timeline, so that a member's beauty settings and name can follow them into every cut.

Evidence between two people of different cuts: face similarity (the strongest: faces survive outfit and
hair changes), body re-ID similarity (same outfit within one scene), both through the self-learned
projection when it has proven itself (learn/identity.py). People on screen in the same cut are never the
same member. Joining is agglomerative (average linkage), the most certain joins first, and stops where
the evidence runs out. Every person gets a confidence: how much better its member fits than the next best;
low ones are marked 확인 필요 in the people panel.

What the user decided (put someone into a member, split them out, merge two members) is kept as fixed
links and wins over everything automatic; the next automatic run works around them.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FACE_SAME = 0.42        # face cosine where "same person" becomes more likely than not
REID_SAME = 0.80        # body re-ID cosine (raw features), only within similar outfits
CONF_LOW = 0.55
MEMBER_COLORS = ["#4F8BFF", "#5CC8A0", "#E8B04A", "#F0709A", "#B48CF2", "#F28C5C", "#4FD1D9", "#D9D24F",
                 "#F25C78", "#8CD96F", "#A3A3F5", "#F5A3D3"]


def _norm(X):
    X = np.asarray(X, np.float32)
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


class Inst:
    __slots__ = ("cut", "pid", "faces", "reid", "weight", "name")

    def __init__(self, cut, pid, faces, reid, weight, name):
        self.cut, self.pid, self.faces, self.reid, self.weight, self.name = cut, pid, faces, reid, weight, name


def _instances(project) -> List[Inst]:
    from ..learn import identity as ID
    Wf, Wr = ID.active("face"), ID.active("reid")
    out = []
    for c in project.timeline_cuts() or project.cuts:
        if not c.analyzed:
            continue
        idf = project.cut_dir(c) / "identity.npz"
        extra = None
        if idf.exists():
            try:
                with np.load(idf) as d:
                    extra = {k: d[k] for k in ("face", "face_pid", "reid", "reid_pid")}
            except Exception:
                extra = None
        metrics = {r["pid"]: r for r in (c.status or {}).get("metrics", {}).get("persons", [])}
        for q in c.persons:
            if q.role == "ignore":
                continue
            faces = [np.asarray(f, np.float32) for f in q.faces]
            reid = [np.asarray(q.reid, np.float32)] if q.reid else []
            if extra is not None:
                if extra["face"].size:
                    faces = list(extra["face"][extra["face_pid"] == q.id]) or faces
                if extra["reid"].size:
                    reid = list(extra["reid"][extra["reid_pid"] == q.id]) or reid
            if not faces and not reid:
                continue
            m = metrics.get(q.id, {})
            w = float(m.get("frames", max(1, q.last - q.first + 1))) * (1.0 + float(m.get("face_px", 0)) / 200.0)
            out.append(Inst(c, q.id, ID.project(np.stack(faces), Wf) if faces else None,
                            ID.project(np.stack(reid), Wr) if reid else None, w, q.name))
    return out


_THR = {"face": FACE_SAME, "reid": REID_SAME}


def _sim(a: Inst, b: Inst) -> Tuple[float, float]:
    """(evidence for "same": positive = same, negative = different, weight of that evidence)."""
    FACE_SAME, REID_SAME = _THR["face"], _THR["reid"]
    if a.faces is not None and b.faces is not None:
        S = a.faces @ b.faces.T
        k = min(3, S.size)
        s = float(np.sort(S.ravel())[-k:].mean())
        conf = min(1.0, min(len(a.faces), len(b.faces)) / 4.0)
        ev = (s - FACE_SAME) / 0.12
        if a.reid is not None and b.reid is not None:
            r = float(a.reid.mean(0) @ b.reid.mean(0) / (np.linalg.norm(a.reid.mean(0)) * np.linalg.norm(b.reid.mean(0)) + 1e-9))
            ev += 0.3 * (r - REID_SAME) / 0.08
        return ev, 0.5 + 0.5 * conf
    if a.reid is not None and b.reid is not None:
        ma, mb = a.reid.mean(0), b.reid.mean(0)
        r = float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9))
        return (r - REID_SAME) / 0.08, 0.35
    return 0.0, 0.0


def link(project, keep_user: bool = True) -> dict:
    """Assigns Person.member / member_conf on every analysed cut and rebuilds project.members."""
    from ..learn import identity as ID
    _THR["face"] = ID.threshold("face", FACE_SAME)
    _THR["reid"] = ID.threshold("reid", REID_SAME)
    insts = _instances(project)
    n = len(insts)
    old = project.members or {}
    # fixed links from the user: (cut id, pid) -> member id
    fixed: Dict[Tuple[str, int], str] = {}
    if keep_user:
        for mid, m in old.items():
            for cid, pid in m.get("fixed", []):
                fixed[(cid, int(pid))] = mid
    groups: List[List[int]] = []
    gid_of: Dict[int, int] = {}
    # start: user members as groups, everyone else alone
    by_mid: Dict[str, List[int]] = {}
    for i, it in enumerate(insts):
        mid = fixed.get((it.cut.id, it.pid))
        if mid:
            by_mid.setdefault(mid, []).append(i)
    for mid, idx in by_mid.items():
        gid_of.update({i: len(groups) for i in idx})
        groups.append(idx)
    for i in range(n):
        if i not in gid_of:
            gid_of[i] = len(groups)
            groups.append([i])
    # pairwise evidence
    E = np.zeros((n, n), np.float32)
    Wt = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if insts[i].cut is insts[j].cut:
                E[i, j] = E[j, i] = -50.0        # same cut: never one member
                Wt[i, j] = Wt[j, i] = 1.0
                continue
            e, w = _sim(insts[i], insts[j])
            E[i, j] = E[j, i] = e
            Wt[i, j] = Wt[j, i] = w
    user_mids = set(by_mid)
    pos = {(it.cut.id, it.pid): i for i, it in enumerate(insts)}
    for a_, b_ in getattr(project, "member_cannot", []) or []:
        i, j = pos.get((a_[0], int(a_[1]))), pos.get((b_[0], int(b_[1])))
        if i is not None and j is not None:
            E[i, j] = E[j, i] = -50.0
            Wt[i, j] = Wt[j, i] = 1.0

    def gscore(a: List[int], b: List[int]) -> float:
        sub = E[np.ix_(a, b)]
        w = Wt[np.ix_(a, b)]
        if (sub <= -49).any():
            return -99.0
        if w.sum() <= 0:
            return -99.0
        return float((sub * w).sum() / w.sum())

    alive = [True] * len(groups)
    mid_of_group = {g: mid for g, mid in enumerate(by_mid)}
    while True:
        best = (0.0, None, None)
        for a in range(len(groups)):
            if not alive[a]:
                continue
            for b in range(a + 1, len(groups)):
                if not alive[b]:
                    continue
                if a in mid_of_group and b in mid_of_group:
                    continue                   # two user members stay apart unless the user merges them
                s = gscore(groups[a], groups[b])
                if s > best[0]:
                    best = (s, a, b)
        if best[1] is None:
            break
        a, b = best[1], best[2]
        if b in mid_of_group and a not in mid_of_group:
            a, b = b, a
        groups[a] += groups[b]
        alive[b] = False
    groups = [g for k, g in enumerate(groups) if alive[k]]
    mids = []
    for k, g in enumerate(groups):
        mid = next((fixed.get((insts[i].cut.id, insts[i].pid)) for i in g if (insts[i].cut.id, insts[i].pid) in fixed), None)
        mids.append(mid)
    # stable ids and names: reuse an old member when most of its people are here
    members = {}
    used = set()
    order = sorted(range(len(groups)), key=lambda k: -sum(insts[i].weight for i in groups[k]))
    for rank, k in enumerate(order):
        g = groups[k]
        mid = mids[k]
        if mid is None:
            votes: Dict[str, int] = {}
            for i in g:
                for om, m in old.items():
                    if [insts[i].cut.id, insts[i].pid] in m.get("people", []) and om not in used:
                        votes[om] = votes.get(om, 0) + 1
            mid = max(votes, key=votes.get) if votes else None
        if mid is None or mid in used:
            mid = "m" + uuid.uuid4().hex[:6]
        used.add(mid)
        om = old.get(mid, {})
        members[mid] = dict(name=om.get("name") or f"멤버 {chr(65 + rank) if rank < 26 else rank + 1}",
                            color=om.get("color") or MEMBER_COLORS[rank % len(MEMBER_COLORS)],
                            user_named=bool(om.get("user_named")), fixed=om.get("fixed", []),
                            people=[[insts[i].cut.id, insts[i].pid] for i in g],
                            cuts=len({insts[i].cut.id for i in g}))
    # confidence per person: own member against the best other member
    gidx = {i: k for k, g in enumerate(groups) for i in g}
    k2mid = {k: [m for m, v in members.items() if [insts[groups[k][0]].cut.id, insts[groups[k][0]].pid] in v["people"]][0]
             for k in range(len(groups))}
    low = 0
    for i, it in enumerate(insts):
        k = gidx[i]
        own = [j for j in groups[k] if j != i]
        s_own = float(np.average(E[i, own], weights=np.maximum(Wt[i, own], 1e-3))) if own else 0.0
        s_other = max([float(np.average(E[i, g], weights=np.maximum(Wt[i, g], 1e-3)))
                       for kk, g in enumerate(groups) if kk != k and not (E[i, g] <= -49).any()] or [-5.0])
        if (it.cut.id, it.pid) in fixed:
            conf = 1.0
        elif not own:
            conf = float(np.clip(0.5 - s_other / 6.0, 0, 1))      # alone: sure if nobody else is close
        else:
            conf = float(1 / (1 + np.exp(-(s_own - s_other) / 1.5)))
        q = it.cut.person(it.pid)
        if q is not None:
            q.member = k2mid[k]
            q.member_conf = round(conf, 3)
            m = members[q.member]
            if m.get("user_named") or len(m["people"]) > 1:
                q.name = m["name"]
        if conf < CONF_LOW:
            low += 1
    project.members = members
    return dict(people=n, members=len(members), low=low,
                multi=sum(1 for m in members.values() if m["cuts"] > 1))


# ------------------------------------------------------------------ user decisions
def _pair_feats(project, a: Tuple[str, int], b: Tuple[str, int]):
    out = []
    ca, cb = project.cut(a[0]), project.cut(b[0])
    if ca is None or cb is None:
        return out
    pa, pb = ca.person(a[1]), cb.person(b[1])
    if pa is None or pb is None:
        return out
    if pa.faces and pb.faces:
        out.append(("face", np.mean(_norm(pa.faces), 0), np.mean(_norm(pb.faces), 0)))
    if pa.reid and pb.reid:
        out.append(("reid", np.asarray(pa.reid, np.float32), np.asarray(pb.reid, np.float32)))
    return out


def _teach(project, a, b, same: bool):
    from ..learn import identity as ID
    for kind, fa, fb in _pair_feats(project, a, b):
        ID.add_links([(fa, fb, same)], kind)


def assign(project, cut_id: str, pid: int, mid: Optional[str]) -> str:
    """The user says: this person is member `mid` (None: a new member of their own)."""
    members = project.members
    for om, m in members.items():
        if [cut_id, pid] in m.get("people", []) and om != mid:
            # taken out of a member: never put back with the people it was with
            for o in m["people"]:
                if o != [cut_id, pid]:
                    project.member_cannot.append([[cut_id, pid], list(o)])
                    _teach(project, (cut_id, pid), tuple(o), False)
        m["fixed"] = [x for x in m.get("fixed", []) if x != [cut_id, pid]]
        m["people"] = [x for x in m.get("people", []) if x != [cut_id, pid]]
        m["cuts"] = len({x[0] for x in m["people"]})
    if mid is None or mid not in members:
        mid = "m" + uuid.uuid4().hex[:6]
        members[mid] = dict(name=f"멤버 {len(members) + 1}", color=MEMBER_COLORS[len(members) % len(MEMBER_COLORS)],
                            user_named=False, fixed=[], people=[], cuts=0)
    m = members[mid]
    others = [tuple(x) for x in m["people"]]
    m["fixed"].append([cut_id, pid])
    m["people"].append([cut_id, pid])
    m["cuts"] = len({x[0] for x in m["people"]})
    for o in others[:6]:
        _teach(project, (cut_id, pid), o, True)
    # the members it was not: evidence for "different"
    for om, mm in members.items():
        if om != mid:
            for o in mm.get("people", [])[:2]:
                _teach(project, (cut_id, pid), tuple(o), False)
    c = project.cut(cut_id)
    q = c.person(pid) if c else None
    if q is not None:
        q.member, q.member_conf = mid, 1.0
        q.name = m["name"]
    # everyone already in the member is confirmed too
    for x in m["people"]:
        if x not in m["fixed"]:
            m["fixed"].append(x)
    return mid


def merge(project, keep: str, drop: str) -> None:
    ms = project.members
    if keep not in ms or drop not in ms or keep == drop:
        return
    for x in list(ms[drop]["people"]):
        assign(project, x[0], int(x[1]), keep)
    ms.pop(drop, None)


def rename(project, mid: str, name: str) -> None:
    m = project.members.get(mid)
    if m is None:
        return
    m["name"] = name
    m["user_named"] = True
    for cid, pid in m["people"]:
        c = project.cut(cid)
        q = c.person(int(pid)) if c else None
        if q is not None:
            q.name = name


def copy_settings(project, src_cut_id: str, pid: int, scope: str = "beauty") -> int:
    """This person's sliders onto every other appearance of the same member. scope: beauty | all."""
    import copy as _copy
    from ..core.params import PARAMS
    groups = {p.key: p.group for p in PARAMS}
    c = project.cut(src_cut_id)
    q = c.person(pid) if c else None
    if q is None or not q.member:
        return 0
    src = c.params_for(pid)
    n = 0
    for cid, p2 in project.members.get(q.member, {}).get("people", []):
        if cid == src_cut_id and int(p2) == pid:
            continue
        c2 = project.cut(cid)
        if c2 is None or c2.person(int(p2)) is None:
            continue
        dst = c2.params_for(int(p2))
        for k, cv in src.amounts.items():
            if scope == "all" or groups.get(k) in ("skin", "color"):
                dst.amounts[k] = _copy.deepcopy(cv)
                dst.amounts[k].keys = {}           # frame keys belong to the source cut's frames
        dst.skin_detail = src.skin_detail
        dst.skin_basic = dict(getattr(src, "skin_basic", {}) or {})
        c2.person(int(p2)).role = q.role if q.role == "target" else c2.person(int(p2)).role
        n += 1
    return n


def face_thumb(project, cut, pid: int, size: int = 96) -> Optional[np.ndarray]:
    """A square crop of this person's face (or upper body) from the cut's proxy, cached on disk."""
    import cv2
    from ..core.proxy import Proxy
    from ..core.settings import settings
    from .store import TrackStore
    d = project.folder / "timeline" / "people"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{cut.id}_{pid}.jpg"
    if f.exists():
        im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    px = Proxy(project.proxy_dir(cut), cut.info, settings().proxy_long_side)
    if not px.has_frames():
        return None
    try:
        from .analyze import CutTracker
        st = CutTracker(project, cut, px).store
    except Exception:
        st = None
    if st is None or not st.has(pid):
        return None
    p = st.index(pid)
    box = None
    t_best = None
    if getattr(st, "has_face", False):
        F = st.face[p, :, :, :2]
        ok = st.present[p] & np.isfinite(F).all((1, 2)) & (st.face_vis[p] >= 0.5)
        if ok.any():
            w = np.where(ok, np.hypot(*(F[:, 16] - F[:, 0]).T), 0)
            t_best = int(np.argmax(w))
            c = F[t_best].mean(0)
            r = 0.9 * float(w[t_best])
            box = (c[0] - r, c[1] - 1.1 * r, c[0] + r, c[1] + 0.9 * r)
    if box is None:
        ts = np.nonzero(st.present[p])[0]
        if not len(ts):
            return None
        t_best = int(ts[len(ts) // 2])
        k = st.final[p, t_best, :, :2]
        k = k[np.isfinite(k).all(1)]
        if not len(k):
            return None
        x0, y0 = k.min(0)
        x1, y1 = k.max(0)
        h = (y1 - y0) * 0.45
        cx = (x0 + x1) / 2
        box = (cx - h / 2, y0 - 0.1 * h, cx + h / 2, y0 + 0.9 * h)
    rgb = np.asarray(px.frame(t_best))
    x0, y0, x1, y1 = [v * s for v, s in zip(box, (px.sx, px.sy, px.sx, px.sy))]
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        f.write_bytes(buf.tobytes())
    return crop
