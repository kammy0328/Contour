"""Self-learned trust in each pose model (no user input needed).

Every analysed cut has, for each joint, the per-frame guesses of every pose model (RTMPose-x, RTMW-x,
RTMPose-m, a second pass of each, the face models for the head) and the final automatic track, which was
solved over the whole shot in both directions. Where that track is certain, it tells how far off each
model was in that frame. Over many cuts this shows which model to trust for which joint and in which
condition (sharp / soft / motion blurred frame): for example that one model's wrists drift in motion
blur while another's hold. Those reliabilities become weights in the per-frame fusion of the next
analyses, so the fused measurement starts closer to the truth, fewer frames are flagged, and the tracker
needs less smoothing.

Safety: a new table is only used if, cut by cut, a table learned from all the other cuts makes that cut's
fused measurement closer to its own certain track (leave-one-cut-out), on average by at least 1 % and on
no cut worse by more than 3 %. Otherwise the previous table (or none) stays.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.settings import settings
from ..track import skeleton as S

MAX_POINTS = 4000
NB = 3                        # blur bins: sharp, soft, motion blurred
SHRINK = 150.0                # samples needed before a learned weight moves half way from 1
TAU = 0.03
MIN_CUTS = 3


def _dir() -> Path:
    d = settings().learning_path / "fusion"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(cut) -> str:
    return re.sub(r"[^\w\-]+", "_", Path(str(cut.source)).stem)[:80] + "__" + str(cut.info.frames or 0)


def blur_bin(blur: np.ndarray) -> np.ndarray:
    return np.where(blur >= 0.8, 0, np.where(blur >= 0.55, 1, 2)).astype(np.int8)


def teacher_mask(st, cut, reach: int = 4) -> np.ndarray:
    """(P,N,K) joints where the solved track can serve as the answer, including hard frames.

    The useful cases are frames where the models disagree or the frame is blurred, but the shot settles
    it: a confident frame within `reach` frames before AND after (same shot), forward and backward passes
    that agree at this frame, plausible bone lengths and a visible joint. Easy frames (all models already
    agree) are kept too, so a model is never taught to move where nothing is wrong."""
    P, N = st.present.shape
    thr = float(getattr(cut.tracking, "conf_threshold", 0.55))
    hi = st.present & (st.auto_conf >= min(0.9, thr + 0.1))
    shot = np.cumsum(np.concatenate([[0], st.jump[:-1].astype(int)]))      # shot index per frame
    before = np.zeros((P, N), bool)
    after = np.zeros((P, N), bool)
    for d in range(0, min(reach, N - 1) + 1):
        same_b = np.concatenate([np.zeros(d, bool), shot[d:] == shot[:N - d]]) if d else np.ones(N, bool)
        same_a = np.concatenate([shot[:N - d] == shot[d:], np.zeros(d, bool)]) if d else np.ones(N, bool)
        hb = np.concatenate([np.zeros((P, d), bool), hi[:, :N - d]], 1) if d else hi
        ha = np.concatenate([hi[:, d:], np.zeros((P, d), bool)], 1) if d else hi
        before |= hb & same_b[None]
        after |= ha & same_a[None]
    anchored = before & after
    bad = S.FLAG_FB | S.FLAG_BONE | S.FLAG_MISSING
    near_jump = np.zeros(N, bool)
    for t in np.nonzero(st.jump)[0]:
        near_jump[max(0, t - 1):t + 3] = True
    return st.present[..., None] & anchored[..., None] & (st.vis == 1) & ((st.flags & bad) == 0) & \
        ~near_jump[None, :, None] & np.isfinite(st.auto[..., :2]).all(-1) & \
        np.isfinite(st.scale)[..., None] & (st.scale[..., None] > 40)


def collect(project, cut, tracker, seed: int = 0) -> int:
    """Stores the certain points of one analysed cut: each model's offset from the solved track."""
    from ..track.solver import normalize_scores
    st = tracker.store
    f = _dir() / f"{_key(cut)}.npz"
    if st is None or not (project.learn and cut.learn and getattr(settings(), "self_learning", True)):
        return 0
    keys = [k for k in st.raw if st.raw[k].shape[2] == S.K]
    if not keys:
        return 0
    norm = normalize_scores({k: st.raw[k] for k in keys})
    P, N = st.present.shape
    ok = teacher_mask(st, cut)
    for pid, jd in cut.joint_keys.items():          # corrected by hand: the user samples cover those
        if st.has(pid):
            p = st.index(pid)
            for j, fr in jd.items():
                for t in fr:
                    if 0 <= int(t) < N:
                        ok[p, int(t), int(j)] = False
    idx = np.argwhere(ok)
    if len(idx) < 50:
        return 0
    rng = np.random.default_rng(seed + len(idx))
    if len(idx) > MAX_POINTS:
        hard = (st.flags[tuple(idx.T)] & (S.FLAG_DISAGREE | S.FLAG_BLUR)) > 0
        hi_ = np.nonzero(hard)[0]
        lo_ = np.nonzero(~hard)[0]
        nh = min(len(hi_), MAX_POINTS // 2)
        pick = np.concatenate([rng.choice(hi_, nh, replace=False),
                               rng.choice(lo_, min(len(lo_), MAX_POINTS - nh), replace=False)])
        idx = idx[pick]
    p, t, j = idx.T
    sc = st.scale[p, t][:, None, None]
    tgt = st.auto[p, t, j, :2][:, None, :]
    D = np.stack([st.raw[k][p, t, j, :2] for k in keys], 1)            # (n,M,2)
    D = ((D - tgt) / sc).astype(np.float32)
    Sm = np.stack([np.clip(np.nan_to_num(st.raw[k][p, t, j, 2] / norm[k], nan=0.0), 0, 1.2) for k in keys], 1)
    np.savez_compressed(f, D=D, S=Sm.astype(np.float32), joint=j.astype(np.int8),
                        blur=blur_bin(st.blur[p, t]), keys=np.array(json.dumps(keys)),
                        meta=np.array(json.dumps(dict(cut=cut.name, project=project.name, created=time.time()))))
    return int(len(idx))


# ------------------------------------------------------------------ fitting and checking
def _load_all() -> List[dict]:
    out = []
    for f in sorted(_dir().glob("*.npz")):
        try:
            with np.load(f) as d:
                out.append(dict(name=f.stem, D=d["D"], S=d["S"], joint=d["joint"].astype(int),
                                blur=d["blur"].astype(int), keys=json.loads(str(d["keys"]))))
        except Exception:
            continue
    return out


def _fit(clips: List[dict]) -> Dict[str, list]:
    """{model key: (K, NB) weight multipliers}."""
    acc: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for c in clips:
        e2 = np.minimum((np.nan_to_num(c["D"], nan=1.0) ** 2).sum(-1), 0.01)    # (n,M), capped (outliers)
        valid = (c["S"] > 0.2) & np.isfinite(c["D"]).all(-1)
        for mi, k in enumerate(c["keys"]):
            s_, n_ = acc.setdefault(k, (np.zeros((S.K, NB)), np.zeros((S.K, NB))))
            v = valid[:, mi]
            np.add.at(s_, (c["joint"][v], c["blur"][v]), e2[v, mi])
            np.add.at(n_, (c["joint"][v], c["blur"][v]), 1)
    if not acc:
        return {}
    keys = list(acc)
    mse = np.stack([np.where(acc[k][1] > 0, acc[k][0] / np.maximum(acc[k][1], 1), np.nan) for k in keys])
    cnt = np.stack([acc[k][1] for k in keys])
    inv = 1.0 / np.maximum(mse, 1e-5)
    # relative to the other models for the same joint and condition (geometric mean = 1)
    lg = np.log(inv)
    ref = np.nanmean(np.where(cnt > 0, lg, np.nan), 0, keepdims=True)
    r = np.exp(lg - np.nan_to_num(ref))
    shrink = cnt / (cnt + SHRINK)
    r = np.where(np.isfinite(r), 1.0 + (r - 1.0) * shrink, 1.0)
    r = np.clip(r, 0.4, 2.5)
    return {k: r[i].round(4).tolist() for i, k in enumerate(keys)}


def _fused_err(c: dict, table: Optional[Dict[str, list]]) -> Tuple[float, float]:
    """Mean distance (person scale) of the IRLS fusion to the certain track: all points, blurred points."""
    D = np.nan_to_num(c["D"], nan=0.0)
    valid = (c["S"] > 0.2) & np.isfinite(c["D"]).all(-1)
    w = c["S"] * valid
    is_face = np.array([k.startswith("face_") for k in c["keys"]])
    w[:, is_face] *= 1.3
    if table:
        for mi, k in enumerate(c["keys"]):
            if k in table:
                w[:, mi] *= np.asarray(table[k])[c["joint"], c["blur"]]
    ws = w.sum(1)
    mu = (D * w[..., None]).sum(1) / np.maximum(ws, 1e-9)[:, None]
    for _ in range(4):
        d = np.linalg.norm(D - mu[:, None], axis=-1)
        rw = w / (1.0 + (d / TAU) ** 2)
        mu = (D * rw[..., None]).sum(1) / np.maximum(rw.sum(1), 1e-9)[:, None]
    e = np.linalg.norm(mu, axis=-1)
    keep = ws > 0
    hard = keep & (c["blur"] > 0)
    return float(e[keep].mean()) if keep.any() else np.nan, float(e[hard].mean()) if hard.sum() >= 20 else np.nan


def refit() -> dict:
    """Leave-one-cut-out check, then (if it helps) the table from all cuts becomes active."""
    clips = _load_all()
    res = dict(cuts=len(clips), active=False)
    if len(clips) < MIN_CUTS:
        res["reason"] = f"자가학습 영상 {len(clips)}개 (최소 {MIN_CUTS}개)"
        return res
    gains, hard = [], []
    for i, c in enumerate(clips):
        tab = _fit(clips[:i] + clips[i + 1:])
        b, bh = _fused_err(c, None)
        a, ah = _fused_err(c, tab)
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            gains.append(1 - a / b)
        if np.isfinite(ah) and np.isfinite(bh) and bh > 0:
            hard.append(1 - ah / bh)
    res.update(gain=float(np.mean(gains)) if gains else 0.0, worst=float(np.min(gains)) if gains else 0.0,
               gain_blur=float(np.mean(hard)) if hard else None)
    ok = gains and res["gain"] >= 0.01 and res["worst"] >= -0.03
    f = _dir() / "active.json"
    if ok:
        tab = _fit(clips)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(version=1, created=time.time(), cuts=len(clips), gain=res["gain"],
                                       worst=res["worst"], gain_blur=res["gain_blur"], table=tab),
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f)
        res["active"] = True
    else:
        res["reason"] = "다른 영상으로 확인했을 때 나아지지 않아 이전 상태 유지"
        # an earlier table stays only if it still passes on the data we have now
        if f.exists():
            try:
                old = json.loads(f.read_text(encoding="utf-8"))["table"]
                g = [1 - _fused_err(c, old)[0] / max(1e-9, _fused_err(c, None)[0]) for c in clips]
                if np.mean(g) < 0.0:
                    f.unlink()
            except Exception:
                pass
    _cache.clear()
    return res


_cache: Dict[str, tuple] = {}


def active_table() -> Optional[Dict[str, list]]:
    if not getattr(settings(), "self_learning", True):
        return None
    f = _dir() / "active.json"
    if not f.exists():
        return None
    m = f.stat().st_mtime
    hit = _cache.get("t")
    if hit and hit[0] == m:
        return hit[1]
    try:
        tab = json.loads(f.read_text(encoding="utf-8"))["table"]
    except Exception:
        tab = None
    _cache["t"] = (m, tab)
    return tab


def status() -> Optional[dict]:
    f = _dir() / "active.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        d.pop("table", None)
        return d
    except Exception:
        return None
