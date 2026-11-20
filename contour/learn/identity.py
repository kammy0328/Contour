"""Identity self-learning: what tells K-pop members apart, learned from every analysed cut, no labels needed.

The body re-ID and face recognition models were trained on street and celebrity photos. In music videos
members wear the same outfit, the same make-up and change hair between scenes, and the lights change
colour every beat. Every analysed cut teaches, without anyone labelling it:
- one tracked person seen at different moments of a shot is the same person (different pose, light, angle)
- two people on screen at the same time are different people
plus every link the user confirms or corrects in the people panel (the strongest evidence, weighted x5).

From those, a linear projection is learned for each feature kind (face, body) that shrinks the differences
within one person and keeps those between people: regularised linear discriminant analysis, closed form,
no GPU, retrained in a second whenever new cuts come in. It is used to link people across cuts.

Safety: a projection is used only if, on cuts it was not learned from, it separates same-person from
different-person pairs better than the raw features (ROC AUC, at least +0.005 and never worse on the
user's own corrections). Otherwise the raw features stay.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.settings import settings

MIN_CLASSES = 12
SHRINK = 0.3


def _dir() -> Path:
    d = settings().learning_path / "identity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(project, cut) -> str:
    """The same shot analysed in two projects is one sample (it would otherwise check itself)."""
    src = Path(str(cut.source)).stem
    return re.sub(r"[^\w\-]+", "_", f"{src}__{int(getattr(cut.info, 'offset', 0) or 0)}_{cut.info.frames}")[:120]


def collect(project, cut) -> bool:
    """Copies the cut's identity samples into the learning set (once per analysis)."""
    from . import allowed
    if not (allowed(project, cut) and getattr(settings(), "self_learning", True)):
        return False
    src = project.cut_dir(cut) / "identity.npz"
    dst = _dir() / f"{_key(project, cut)}.npz"
    if not src.exists():
        return _from_persons(cut, dst)
    try:
        dst.write_bytes(src.read_bytes())
        return True
    except OSError:
        return False


def _from_persons(cut, dst: Path) -> bool:
    """Cuts analysed before 1.7.0 keep only a few faces per person and one body vector: still usable."""
    F, fp, co = [], [], []
    for q in cut.persons:
        for f in q.faces:
            F.append(np.asarray(f, np.float32))
            fp.append(q.id)
    ps = cut.persons
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if min(ps[i].last, ps[j].last) - max(ps[i].first, ps[j].first) >= 3:
                co.append((ps[i].id, ps[j].id))
    if len(F) < 4:
        return False
    R = [np.asarray(q.reid, np.float32) for q in ps if q.reid]
    rp = [q.id for q in ps if q.reid]
    np.savez_compressed(dst, face=np.stack(F), face_pid=np.array(fp, np.int32),
                        reid=np.stack(R) if R else np.zeros((0, 0), np.float32), reid_pid=np.array(rp, np.int32),
                        cooccur=np.array(co, np.int32).reshape(-1, 2))
    return True


def add_links(pairs: List[Tuple[np.ndarray, np.ndarray, bool]], kind: str) -> None:
    """User decisions from the people panel: (feature a, feature b, same?)."""
    if not pairs:
        return
    f = _dir() / f"user_{kind}.npz"
    A, B, S = [], [], []
    if f.exists():
        try:
            with np.load(f) as d:
                A, B, S = list(d["a"]), list(d["b"]), list(d["same"])
        except Exception:
            pass
    for a, b, same in pairs:
        A.append(np.asarray(a, np.float32))
        B.append(np.asarray(b, np.float32))
        S.append(bool(same))
    A, B, S = A[-4000:], B[-4000:], S[-4000:]
    np.savez_compressed(f, a=np.stack(A), b=np.stack(B), same=np.array(S))


def _load(kind: str):
    """[(features (n,d), class ids (n,), cannot-link class pairs)] per cut."""
    out = []
    for f in sorted(_dir().glob("*.npz")):
        if f.name.startswith("user_"):
            continue
        try:
            with np.load(f) as d:
                X = d[kind]
                y = d[f"{kind}_pid"]
                co = d["cooccur"]
        except Exception:
            continue
        if X.ndim != 2 or len(X) < 4:
            continue
        out.append((f.stem, X.astype(np.float32), y.astype(int), co))
    return out


def _norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def fit_lda(cuts, dim: Optional[int] = None) -> Optional[np.ndarray]:
    """Projection W (d,k): within-person scatter whitened, between-person kept."""
    Xs, ys = [], []
    base = 0
    for name, X, y, _ in cuts:
        Xs.append(_norm(X))
        ys.append(y + base)
        base += int(y.max()) + 1 if len(y) else 0
    if not Xs:
        return None
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    cls = [c for c in np.unique(y) if (y == c).sum() >= 2]
    if len(cls) < MIN_CLASSES:
        return None
    d = X.shape[1]
    mu = X.mean(0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in cls:
        Xc = X[y == c]
        m = Xc.mean(0)
        Z = Xc - m
        Sw += Z.T @ Z
        Sb += len(Xc) * np.outer(m - mu, m - mu)
    Sw /= max(1, len(X))
    Sb /= max(1, len(X))
    Sw = (1 - SHRINK) * Sw + SHRINK * np.eye(d) * np.trace(Sw) / d
    # whiten Sw, then the principal directions of Sb
    ev, V = np.linalg.eigh(Sw)
    Wh = V / np.sqrt(np.maximum(ev, 1e-8))
    Sb2 = Wh.T @ Sb @ Wh
    ev2, U = np.linalg.eigh(Sb2)
    k = dim or min(d, max(16, min(len(cls) - 1, 128)))
    U = U[:, ::-1][:, :k]
    return (Wh @ U).astype(np.float32)


def project(X: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    X = _norm(np.asarray(X, np.float32))
    if W is None or X.shape[-1] != W.shape[0]:
        return X
    return _norm(X @ W)


def _pairs(cuts, rng, n_max=4000):
    """Same-person pairs (two moments of one tracked person) and different-person pairs (on screen
    together) from these cuts."""
    pos, neg = [], []
    for name, X, y, co in cuts:
        for c in np.unique(y):
            idx = np.nonzero(y == c)[0]
            if len(idx) >= 2:
                h = len(idx) // 2
                pos.append((X[idx[:h]].mean(0), X[idx[h:]].mean(0)))
        for a, b in co:
            ia, ib = np.nonzero(y == a)[0], np.nonzero(y == b)[0]
            if len(ia) and len(ib):
                neg.append((X[ia].mean(0), X[ib].mean(0)))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_max], neg[:n_max]


def _auc(pos, neg, W) -> Optional[float]:
    if len(pos) < 10 or len(neg) < 10:
        return None
    sp = np.array([float(project(a, W) @ project(b, W)) for a, b in pos])
    sn = np.array([float(project(a, W) @ project(b, W)) for a, b in neg])
    # probability a random same pair scores above a random different pair
    allv = np.concatenate([sp, sn])
    r = allv.argsort().argsort() + 1
    return float((r[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


def _threshold(pos, neg, W) -> Optional[float]:
    """Cosine where same and different pairs are best told apart (balanced accuracy)."""
    if len(pos) < 10 or len(neg) < 10:
        return None
    sp = np.array([float(project(a, W) @ project(b, W)) for a, b in pos])
    sn = np.array([float(project(a, W) @ project(b, W)) for a, b in neg])
    best = (0.0, None)
    for t in np.unique(np.quantile(np.concatenate([sp, sn]), np.linspace(0.02, 0.98, 97))):
        acc = 0.5 * ((sp >= t).mean() + (sn < t).mean())
        if acc > best[0]:
            best = (acc, float(t))
    return best[1]


def _user_auc(kind, W) -> Optional[float]:
    f = _dir() / f"user_{kind}.npz"
    if not f.exists():
        return None
    try:
        with np.load(f) as d:
            A, B, S = d["a"], d["b"], d["same"].astype(bool)
    except Exception:
        return None
    if S.sum() < 5 or (~S).sum() < 5:
        return None
    pos = [(a, b) for a, b, s in zip(A, B, S) if s]
    neg = [(a, b) for a, b, s in zip(A, B, S) if not s]
    return _auc(pos, neg, W)


def refit() -> dict:
    """For face and body: learn on 80 % of the cuts, check on the rest, keep only when it helps."""
    res = {}
    rng = np.random.default_rng(0)
    for kind in ("face", "reid"):
        cuts = _load(kind)
        r = dict(cuts=len(cuts), active=False)
        if len(cuts) < 5:
            r["reason"] = f"영상 {len(cuts)}개 (최소 5개)"
            res[kind] = r
            continue
        # 5-fold over cuts: every cut is checked once by a projection that never saw it
        order = rng.permutation(len(cuts))
        folds = np.array_split(order, 5)
        g0, g1 = [], []
        for fo in folds:
            va = [cuts[i] for i in fo]
            tr = [cuts[i] for i in order if i not in set(fo.tolist())]
            W = fit_lda(tr)
            pos, neg = _pairs(va, rng)
            a0, a1 = _auc(pos, neg, None), _auc(pos, neg, W) if W is not None else None
            if a0 is not None and a1 is not None:
                g0.append(a0)
                g1.append(a1)
        if len(g0) < 2:
            r["reason"] = "확인할 인물 쌍 부족"
            res[kind] = r
            continue
        a0, a1 = float(np.mean(g0)), float(np.mean(g1))
        worst = float(min(b - a for a, b in zip(g0, g1)))
        Wall = fit_lda(cuts)
        u0, u1 = _user_auc(kind, None), (_user_auc(kind, Wall) if Wall is not None else None)
        r.update(auc_raw=round(a0, 4), auc_learned=round(a1, 4), worst_fold=round(worst, 4), user_raw=u0, user_learned=u1)
        ok = a1 >= a0 + 0.005 and worst >= -0.02 and (u0 is None or (u1 is not None and u1 >= u0 - 0.002))
        f = _dir() / f"active_{kind}.npz"
        allpos, allneg = _pairs(cuts, rng)
        r["thr_raw"] = _threshold(allpos, allneg, None)
        if ok:
            if Wall is not None:
                thr = _threshold(allpos, allneg, Wall)
                np.savez_compressed(f, W=Wall, thr=np.array(thr if thr is not None else np.nan),
                                    meta=np.array(json.dumps(dict(created=time.time(), cuts=len(cuts),
                                                                  auc_raw=a0, auc_learned=a1))))
                r["active"] = True
                r["thr_learned"] = thr
        else:
            r["reason"] = "처음 보는 영상에서 나아지지 않아 원래 특징 사용"
            if f.exists():
                f.unlink()
        res[kind] = r
    _cache.clear()
    (_dir() / "status.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


_cache: Dict[str, tuple] = {}


def active(kind: str) -> Optional[np.ndarray]:
    if not getattr(settings(), "self_learning", True):
        return None
    f = _dir() / f"active_{kind}.npz"
    if not f.exists():
        return None
    m = f.stat().st_mtime
    hit = _cache.get(kind)
    if hit and hit[0] == m:
        return hit[1]
    try:
        with np.load(f) as d:
            W = d["W"].astype(np.float32)
    except Exception:
        W = None
    _cache[kind] = (m, W)
    return W


def threshold(kind: str, default: float) -> float:
    """The same/different cosine for the features xlink will see (learned projection or raw)."""
    st = status().get(kind, {})
    # the pairs measured here are two moments of one shot, easier than the same person in another scene:
    # only the shift between raw and learned features is taken over, the cross-cut level stays
    if active(kind) is not None and st.get("thr_learned") is not None and st.get("thr_raw") is not None:
        return float(default + st["thr_learned"] - st["thr_raw"])
    return default


def status() -> dict:
    f = _dir() / "status.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
