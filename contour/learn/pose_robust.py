"""Body joints that hold under motion blur, coloured light and flash, learned from the footage alone.

Two kinds of self-made answers, both from analysed cuts (no hand labels needed to start):

- degraded: a sharp frame whose solved track is sure (both passes agree, a sure frame before and after, no other
  person over this one, not fast, no flash) is made hard (motion blur, defocus, under/over exposure, one-colour light,
  flash, heavy compression, or several together). The pose model is run on the hard picture; the answer is the
  sure track of the clean frame. This is what the current model does wrong in such frames, measured exactly.
- flow: from a sharp sure frame the joints are carried to the next blurred frames of the same person with optical
  flow, kept only where the flow comes back to the start (round trip) and the point stays on that person (the
  person label map). The answer is the carried point, the measurement the frame's own fused one.

Each sample is stored like the refiner's (contour/learn/refiner.py) so the refiner can learn from them, with what
the record needs to be checked: source file, cut, frame, person, crop centre and size, label kind and method,
confidence, review state. Frames and their degraded copies stay in one group, and groups never cross the
training / check split (group = a music video, or inside one video a camera roll: repeated angles stay together).

Checks (evaluate): joint error relative to the person's size, left/right swaps, and on real sequences the
frame-to-frame jitter and jumps of the per-frame measurement."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..track import skeleton as S
from . import refiner as RF

KINDS = ("motion", "defocus", "dark", "bright", "colour", "flash", "compress", "mixed")
LR_PAIRS = [(S.L_EYE, S.R_EYE), (S.L_EAR, S.R_EAR), (S.L_SH, S.R_SH), (S.L_EL, S.R_EL), (S.L_WR, S.R_WR),
            (S.L_HIP, S.R_HIP), (S.L_KN, S.R_KN), (S.L_AN, S.R_AN)]
MAX_TARGET = 0.45          # of the network window: farther than this a local correction cannot reach (counted apart)
SWAP_MIN = 0.10            # of the person's size: a pair this far off can be a swap
LABEL_DEGRADED = "degraded_v1"
LABEL_FLOW = "flow_v1"
FLOW_RT = 0.012            # round-trip error allowed, share of the person's size
SAMPLE_W = 0.6


# --------------------------------------------------------------------------- groups

def roll_group(name: str) -> str:
    """Camera roll of a source clip name (A_0006C009_261110_... -> A_0006): a setup's repeated angles together."""
    m = re.match(r"^([A-Za-z])_?(\d{3,4})C\d+", Path(str(name or "")).stem)
    if m:
        return f"{m.group(1).upper()}_{m.group(2)}"
    return Path(str(name or "")).stem[:24] or "clip"


# --------------------------------------------------------------------------- sure frames

def sure_mask(st, cut) -> Tuple[np.ndarray, dict]:
    """(P,N,K) joints whose solved track is the answer on sharp, sure frames (refiner.pseudo_mask plus sharp)."""
    ok, stat = RF.pseudo_mask(st, cut)
    sharp = st.present & (st.blur >= 0.8) & (st.auto_conf >= 0.9)
    ok = ok & sharp[..., None]
    if getattr(st, "has_vis", False):
        ok &= st.vis_meas == 1
    for pid, jd in (getattr(cut, "joint_keys", None) or {}).items():   # corrected by hand: the user's samples
        if st.has(pid):
            p = st.index(pid)
            for j, fr in jd.items():
                for t in fr:
                    if 0 <= int(t) < st.n:
                        ok[p, int(t), int(j)] = False
    stat["sharp_kept"] = int(ok.sum())
    return ok, stat


# --------------------------------------------------------------------------- making a frame hard

def _motion_kernel(L: int, angle: float) -> np.ndarray:
    L = max(3, int(L) | 1)
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0
    R = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, R, (L, L))
    return k / max(1e-6, k.sum())


def degrade_frame(rgb: np.ndarray, kind: str, rng, size_px: float) -> Tuple[np.ndarray, dict]:
    """A hard copy of rgb (uint8) and what was done. size_px: the person's height in this picture (blur lengths
    follow the person, not the frame)."""
    x = rgb.astype(np.float32)
    prm: dict = dict(kind=kind)
    kinds = [kind] if kind != "mixed" else list(rng.choice(["motion", "defocus", "dark", "colour", "flash", "compress"],
                                                           size=int(rng.integers(2, 4)), replace=False))
    for k in kinds:
        if k == "motion":
            L = float(rng.uniform(0.03, 0.12)) * size_px
            a = float(rng.uniform(0, 180))
            x = cv2.filter2D(x, -1, _motion_kernel(int(round(L)), a), borderType=cv2.BORDER_REFLECT)
            prm.update(motion_px=round(L, 1), angle=round(a, 1))
        elif k == "defocus":
            s = float(rng.uniform(0.008, 0.03)) * size_px
            x = cv2.GaussianBlur(x, (0, 0), max(0.5, s))
            prm.update(defocus_sigma=round(s, 2))
        elif k == "dark":
            g = float(rng.uniform(0.18, 0.45))
            x = 255.0 * (np.clip(x / 255.0 * g, 0, 1) ** float(rng.uniform(1.0, 1.4)))
            x = x + rng.normal(0, rng.uniform(1.5, 5), x.shape).astype(np.float32)
            prm.update(gain=round(g, 2))
        elif k == "bright":
            g = float(rng.uniform(1.8, 3.2))
            x = 255.0 * np.clip(x / 255.0 * g, 0, 1)
            prm.update(gain=round(g, 2))
        elif k == "colour":
            gains = rng.uniform(0.04, 0.35, 3).astype(np.float32)
            c = int(rng.integers(3))
            gains[c] = 1.0
            if rng.random() < 0.4:                       # magenta / cyan / amber: two channels lit
                gains[(c + 1) % 3] = float(rng.uniform(0.6, 1.0))
            x = x * gains
            x = x * float(rng.uniform(1.0, 1.8))
            prm.update(gains=[round(float(v), 2) for v in gains])
        elif k == "flash":
            H, W = x.shape[:2]
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.exp(-(((yy - cy) / (0.6 * H)) ** 2 + ((xx - cx) / (0.6 * W)) ** 2))
            amt = float(rng.uniform(90, 220))
            x = x * float(rng.uniform(1.3, 2.4)) + amt * r[..., None]
            prm.update(flash=round(amt, 1))
        elif k == "compress":
            f = float(rng.uniform(2.0, 4.0))
            h, w = x.shape[:2]
            u = np.clip(x, 0, 255).astype(np.uint8)
            small = cv2.resize(u, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
            ok_, b_ = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(6, 22))])
            if ok_:
                small = cv2.imdecode(b_, cv2.IMREAD_UNCHANGED).reshape(small.shape)
            x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            prm.update(down=round(f, 2))
    if kind == "mixed":
        prm["parts"] = [str(k) for k in kinds]
    return np.clip(x, 0, 255).astype(np.uint8), prm


def hide_others(rgb: np.ndarray, labels: Optional[np.ndarray], p: int) -> np.ndarray:
    """Other tracked people greyed (as the analysis shows a pose model one person at a time)."""
    if labels is None:
        return rgb
    H, W = rgb.shape[:2]
    lab = cv2.resize(labels.astype(np.int16), (W, H), interpolation=cv2.INTER_NEAREST)
    other = (lab >= 0) & (lab != p)
    if not other.any():
        return rgb
    other = cv2.erode(other.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = rgb.copy()
    out[other] = 114
    return out


# --------------------------------------------------------------------------- measures

def lr_swaps(pred: np.ndarray, gt: np.ndarray, scale: float, ok: np.ndarray) -> Tuple[int, int]:
    """(pairs checked, pairs swapped): a pair is swapped when crossing the two points fits the answer clearly
    better (half the distance) and the pair is off by at least SWAP_MIN of the person's size."""
    n = s = 0
    for a, b in LR_PAIRS:
        if not (ok[a] and ok[b]) or not (np.isfinite(pred[a]).all() and np.isfinite(pred[b]).all()):
            continue
        n += 1
        d = np.hypot(*(pred[a] - gt[a])) + np.hypot(*(pred[b] - gt[b]))
        x = np.hypot(*(pred[a] - gt[b])) + np.hypot(*(pred[b] - gt[a]))
        if d / max(scale, 1e-6) >= 2 * SWAP_MIN and x < 0.5 * d:
            s += 1
    return n, s


def jitter(xy: np.ndarray, scale: np.ndarray, ok: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """(median second difference, share of jumps) of a (P,N,K,2) track over frames, relative to the person's size.
    A jump: a joint moving more than 8 % of the person's size from one frame to the next and back."""
    if xy.shape[1] < 3:
        return None, None
    a2 = xy[:, 2:] + xy[:, :-2] - 2 * xy[:, 1:-1]
    sc = scale[:, 1:-1, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.hypot(a2[..., 0], a2[..., 1]) / sc
    m = ok[:, 1:-1] & ok[:, 2:] & ok[:, :-2] & np.isfinite(j)
    if not m.any():
        return None, None
    return float(np.median(j[m])), float((j[m] > 0.16).mean())


# --------------------------------------------------------------------------- collecting

def _pose_call(pose, img: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kp, sc = pose(img, box[None].astype(np.float32))
    return np.asarray(kp[0][:S.K], np.float32), np.asarray(sc[0][:S.K], np.float32)


def _empty():
    return dict(crops=[], joints=[], targets=[], frames=[], persons=[], kinds=[], conf=[], centre=[], side=[],
                gt=[], meas=[], scale=[], rec=[])


def collect_degraded(project, cut, tracker, pose, rng, frames: int = 24, kinds_per_frame: int = 3,
                     kinds: Sequence[str] = KINDS, cancel: Optional[Callable[[], bool]] = None) -> Tuple[dict, list]:
    """Samples (refiner layout plus record fields) and per-record checks [(kind, p, t, gt (K,2), pred (K,2),
    ok (K,), scale, clean_pred (K,2))] from sure sharp frames of one analysed cut, made hard."""
    st = tracker.store
    px = tracker.proxy
    out, recs = _empty(), []
    if st is None:
        return out, recs
    ok, _ = sure_mask(st, cut)
    per = ok.sum(-1)                                    # (P,N) sure joints
    cand = np.argwhere(per >= 8)
    if not len(cand):
        return out, recs
    flash = RF.flash_frames(sorted({int(t) for _, t in cand}), lambda t: px.frame(t), st.n)
    cand = np.array([c for c in cand if int(c[1]) not in flash])
    if not len(cand):
        return out, recs
    pick = cand[rng.choice(len(cand), size=min(frames, len(cand)), replace=False)]
    sx, sy = float(px.sx), float(px.sy)
    for p, t in sorted(pick.tolist(), key=lambda a: a[1]):
        if cancel and cancel():
            break
        sc = float(st.scale[p, t])
        b = st.boxes[p, t]
        if not np.isfinite(b).all() or not np.isfinite(sc):
            continue
        rgb = np.ascontiguousarray(np.asarray(px.frame(t)))
        lab = tracker.labels(t) if hasattr(tracker, "labels") else None
        view = hide_others(rgb, lab, p)
        box = np.array([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], np.float32)
        size_px = float(box[3] - box[1])
        gt = st.auto[p, t, :, :2].astype(np.float32)
        okj = ok[p, t]
        kc, _ = _pose_call(pose, view, box)
        clean = kc / [sx, sy]
        for kind in rng.choice(list(kinds), size=min(kinds_per_frame, len(kinds)), replace=False):
            img, prm = degrade_frame(view, str(kind), rng, size_px)
            kp, scr = _pose_call(pose, img, box)
            pred = kp / [sx, sy]
            r = len(recs)
            recs.append(dict(kind=str(kind), p=int(p), t=int(t), gt=gt.copy(), pred=pred.astype(np.float32),
                             ok=okj.copy(), scale=sc, clean=clean.astype(np.float32), prm=prm,
                             conf=float(st.auto_conf[p, t])))
            side = RF.SIDE_FRAC * sc
            for j in np.nonzero(okj)[0]:
                if scr[j] < 0.05 or not np.isfinite(pred[j]).all():
                    continue
                tg = (gt[j] - pred[j]) / side
                if np.abs(tg).max() > MAX_TARGET:
                    continue
                c = kp[j]
                out["crops"].append(RF.crop_at(img, c, side * 1.5 * sx, RF.STORE))
                out["joints"].append(int(j))
                out["targets"].append(tg.astype(np.float32))
                out["frames"].append(int(t))
                out["persons"].append(int(st.pids[p]))
                out["kinds"].append(str(kind))
                out["conf"].append(float(scr[j]))
                out["centre"].append([float(c[0]), float(c[1])])
                out["side"].append(float(side * 1.5 * sx))
                out["gt"].append(gt[j])
                out["meas"].append(pred[j].astype(np.float32))
                out["scale"].append(sc)
                out["rec"].append(r)
    return out, recs


def flow_points(prev: np.ndarray, nxt: np.ndarray, pts: np.ndarray, rt_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """pts (M,2) in prev carried into nxt with pyramidal Lucas-Kanade, kept where the way back lands within rt_px."""
    if not len(pts):
        return pts, np.zeros(0, bool)
    g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY) if prev.ndim == 3 else prev
    g1 = cv2.cvtColor(nxt, cv2.COLOR_RGB2GRAY) if nxt.ndim == 3 else nxt
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    pb, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    back = np.hypot(*(pb - p0).reshape(-1, 2).T)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (back <= rt_px)
    return p1.reshape(-1, 2), ok


def collect_flow(project, cut, tracker, rng, starts: int = 16, steps: int = 3,
                 cancel: Optional[Callable[[], bool]] = None) -> Tuple[dict, list]:
    """Samples on real blurred frames: sure joints of a sharp frame carried frame by frame (both ways) to frames
    of the same person that are blurred or flagged, while the round trip holds and the point stays on the person."""
    st = tracker.store
    px = tracker.proxy
    out, recs = _empty(), []
    if st is None or st.n < 2:
        return out, recs
    ok, _ = sure_mask(st, cut)
    shot = np.cumsum(np.concatenate([[0], st.jump[:-1].astype(int)]))
    hard = st.present & ((st.blur < 0.7) | (((st.flags & (S.FLAG_BLUR | S.FLAG_DISAGREE)) > 0).sum(-1) >= 2))
    cand = []
    for p, t in np.argwhere(ok.sum(-1) >= 8):
        near = [t + d for d in range(-steps, steps + 1) if d and 0 <= t + d < st.n and shot[t + d] == shot[t]]
        if any(hard[p, u] for u in near):
            cand.append((int(p), int(t)))
    if not cand:
        return out, recs
    cand = [cand[i] for i in rng.choice(len(cand), size=min(starts, len(cand)), replace=False)]
    sx, sy = float(px.sx), float(px.sy)
    frame = {}

    def F(t):
        if t not in frame:
            frame[t] = np.ascontiguousarray(np.asarray(px.frame(t)))
        return frame[t]
    for p, t0 in cand:
        if cancel and cancel():
            break
        js = np.nonzero(ok[p, t0])[0]
        for direction in (-1, 1):
            pts = st.auto[p, t0, js, :2] * [sx, sy]
            alive = np.ones(len(js), bool)
            t = t0
            for _ in range(steps):
                u = t + direction
                if not (0 <= u < st.n) or shot[u] != shot[t0] or not st.present[p, u]:
                    break
                sc = float(st.scale[p, u])
                if not np.isfinite(sc):
                    break
                nxt, good = flow_points(F(t), F(u), pts, FLOW_RT * sc * sy)
                lab = tracker.labels(u) if hasattr(tracker, "labels") else None
                if lab is not None:
                    H, W = F(u).shape[:2]
                    lx = np.clip((nxt[:, 0] * lab.shape[1] / W).astype(int), 0, lab.shape[1] - 1)
                    ly = np.clip((nxt[:, 1] * lab.shape[0] / H).astype(int), 0, lab.shape[0] - 1)
                    good &= lab[ly, lx] == p
                alive &= good
                pts, t = nxt, u
                if not alive.any():
                    break
                if not hard[p, u]:
                    continue
                gt = np.full((S.K, 2), np.nan, np.float32)
                gt[js[alive]] = (pts[alive] / [sx, sy]).astype(np.float32)
                okj = np.zeros(S.K, bool)
                okj[js[alive]] = True
                meas = st.meas[p, u, :, :2].astype(np.float32)
                r = len(recs)
                recs.append(dict(kind="flow", p=int(p), t=int(u), gt=gt, pred=meas.copy(), ok=okj, scale=sc,
                                 clean=None, prm=dict(from_frame=int(t0)), conf=float(st.auto_conf[p, u])))
                side = RF.SIDE_FRAC * sc
                rgb = F(u)
                for j in np.nonzero(okj)[0]:
                    m = st.meas[p, u, j]
                    if not np.isfinite(m).all() or m[2] < 0.05:
                        continue
                    tg = (gt[j] - m[:2]) / side
                    if np.abs(tg).max() > MAX_TARGET:
                        continue
                    c = m[:2] * [sx, sy]
                    out["crops"].append(RF.crop_at(rgb, c, side * 1.5 * sx, RF.STORE))
                    out["joints"].append(int(j))
                    out["targets"].append(tg.astype(np.float32))
                    out["frames"].append(int(u))
                    out["persons"].append(int(st.pids[p]))
                    out["kinds"].append("flow")
                    out["conf"].append(float(m[2]))
                    out["centre"].append([float(c[0]), float(c[1])])
                    out["side"].append(float(side * 1.5 * sx))
                    out["gt"].append(gt[j])
                    out["meas"].append(m[:2].astype(np.float32))
                    out["scale"].append(sc)
                    out["rec"].append(r)
        frame = {k: v for k, v in frame.items() if abs(k - t0) <= steps}
    return out, recs


# --------------------------------------------------------------------------- storing

def write_shard(path: Path, smp: dict, recs: list, meta: dict) -> int:
    """Refiner layout (crops, joints, targets, weights, easy, frames, meta) plus the record fields and the checks."""
    n = len(smp["crops"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    R = len(recs)
    K = S.K
    rg = np.full((R, K, 2), np.nan, np.float32)
    rp = np.full((R, K, 2), np.nan, np.float32)
    rc = np.full((R, K, 2), np.nan, np.float32)
    ro = np.zeros((R, K), bool)
    for i, r in enumerate(recs):
        rg[i], rp[i], ro[i] = r["gt"], r["pred"], r["ok"]
        if r.get("clean") is not None:
            rc[i] = r["clean"]
    meta = dict(meta, n=n, records=R, created=time.time(), pseudo=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            crops=np.stack(smp["crops"]) if n else np.zeros((0, RF.STORE, RF.STORE, 3), np.uint8),
            joints=np.array(smp["joints"], np.int64), targets=np.array(smp["targets"], np.float32).reshape(-1, 2),
            weights=np.full(n, SAMPLE_W, np.float32), easy=np.zeros(n, bool), frames=np.array(smp["frames"], np.int32),
            persons=np.array(smp["persons"], np.int32), kinds=np.array(smp["kinds"]), conf=np.array(smp["conf"], np.float32),
            centre=np.array(smp["centre"], np.float32).reshape(-1, 2), side=np.array(smp["side"], np.float32),
            gt=np.array(smp["gt"], np.float32).reshape(-1, 2), meas=np.array(smp["meas"], np.float32).reshape(-1, 2),
            scale=np.array(smp["scale"], np.float32), rec=np.array(smp["rec"], np.int32),
            rec_kind=np.array([r["kind"] for r in recs]), rec_p=np.array([r["p"] for r in recs], np.int32),
            rec_t=np.array([r["t"] for r in recs], np.int32), rec_gt=rg, rec_pred=rp, rec_clean=rc, rec_ok=ro,
            rec_scale=np.array([r["scale"] for r in recs], np.float32),
            rec_conf=np.array([r["conf"] for r in recs], np.float32),
            rec_prm=np.array(json.dumps([r["prm"] for r in recs])),
            review=np.array(["unreviewed"] * n), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    tmp.replace(path)
    return n


def record_errors(rg: np.ndarray, rp: np.ndarray, ro: np.ndarray, rs: np.ndarray) -> dict:
    """Error (relative to the person's size) over the sure joints, left/right swaps, share out of a local
    correction's reach."""
    e = np.hypot(*(rp - rg).transpose(2, 0, 1)) / rs[:, None]
    m = ro & np.isfinite(e)
    pairs = swaps = 0
    for i in range(len(rg)):
        n, s = lr_swaps(rp[i], rg[i], float(rs[i]), ro[i])
        pairs += n
        swaps += s
    far = (e > MAX_TARGET * RF.SIDE_FRAC) & m
    return dict(records=int(len(rg)), joints=int(m.sum()), err_mean=float(e[m].mean()) if m.any() else None,
                err_median=float(np.median(e[m])) if m.any() else None,
                err_p90=float(np.percentile(e[m], 90)) if m.any() else None,
                pck05=float((e[m] <= 0.05).mean()) if m.any() else None,
                far=float(far.sum() / max(1, m.sum())), pairs=int(pairs), swaps=int(swaps),
                swap_rate=float(swaps / pairs) if pairs else None)


def apply_refiner(rt, crops: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Offsets (M,2) in network windows where the refiner is sure enough (its gate), NaN elsewhere."""
    if not len(crops):
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)
    o0 = (RF.STORE - RF.CROP) // 2
    c = crops[:, o0:o0 + RF.CROP, o0:o0 + RF.CROP]
    off, var = rt.predict(np.ascontiguousarray(c), joints)
    use = (var <= rt.gate) & (rt.gate > 0) & np.isin(joints, list(rt.joints))
    return off.astype(np.float32), use
