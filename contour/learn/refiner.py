"""Cross-project learning: a small CNN that corrects joint estimates, trained on the user's own corrections.

Data   : every corrected keyframe (and sampled frames of ranges marked as confirmed) becomes a sample:
         a crop around the automatic estimate and the offset to where the user put the joint.
Model  : 64 px crop + joint id -> offset (in crop units) + log-variance. Trained with PyTorch (optional),
         exported to ONNX so inference never needs PyTorch.
Safety : each training run becomes a new version; it is only switched on when it beats the current
         version on held-out data and on the user's benchmark cuts. Any version can be re-activated.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.settings import settings
from ..track import skeleton as S

CROP = 64            # network input
STORE = 96           # stored crop (1.5x context for shift augmentation)
SIDE_FRAC = 0.16     # crop side (network window) relative to person height


# --------------------------------------------------------------------------- crops

def crop_at(rgb: np.ndarray, center: np.ndarray, side: float, out: int) -> np.ndarray:
    """Axis-aligned crop of `side` px (image units) centred at `center`, resampled to out x out."""
    s = out / max(side, 1.0)
    M = np.array([[s, 0, out / 2 - s * center[0]], [0, s, out / 2 - s * center[1]]], np.float32)
    return cv2.warpAffine(rgb, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- dataset

def samples_dir() -> Path:
    d = settings().learning_path / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect_cut(project, cut, tracker) -> int:
    """(Re)build the sample shard for one cut. Returns the number of samples."""
    st = tracker.store
    shard = samples_dir() / f"{project.name}__{cut.id}.npz"
    if st is None or not (project.learn and cut.learn and settings().collect_learning_data):
        if shard.exists():
            shard.unlink()
        return 0
    proxy = tracker.proxy
    crops, joints, targets, weights, frames = [], [], [], [], []
    todo: Dict[int, List[Tuple[int, int, np.ndarray, float]]] = {}
    for pid, jd in cut.joint_keys.items():
        if not st.has(pid):
            continue
        p = st.index(pid)
        for j, fr in jd.items():
            for f, xy in fr.items():
                todo.setdefault(f, []).append((p, j, np.asarray(xy, float), 1.0))
    for pid, ranges in cut.confirmed.items():
        if not st.has(pid):
            continue
        p = st.index(pid)
        for a, b in ranges:
            for f in range(a, b + 1, 6):
                if 0 <= f < st.n and st.present[p, f]:
                    for j in range(S.K):
                        v = st.final[p, f, j, :2]
                        if np.isfinite(v).all() and st.vis[p, f, j] == 1 and \
                                not (cut.joint_keys.get(pid, {}).get(j, {}).get(f)):
                            todo.setdefault(f, []).append((p, j, v.astype(float), 0.5))
    for f, items in sorted(todo.items()):
        rgb = np.asarray(proxy.frame(f))
        for p, j, xy, w in items:
            m = st.meas[p, f, j, :2]
            sc = st.scale[p, f]
            if not (np.isfinite(m).all() and np.isfinite(sc)):
                continue
            side = SIDE_FRAC * sc
            tgt = (xy - m) / side
            if np.abs(tgt).max() > 0.45:
                continue        # too far off: the crop would not contain the joint
            c = m * [proxy.sx, proxy.sy]
            crops.append(crop_at(rgb, c, side * 1.5 * proxy.sx, STORE))
            joints.append(j)
            targets.append(tgt.astype(np.float32))
            weights.append(w)
            frames.append(f)
    if not crops:
        if shard.exists():
            shard.unlink()
        return 0
    meta = dict(project=project.name, project_path=project.path, cut=cut.name, cut_id=cut.id, client=project.client,
                created=time.time(), n=len(crops), source=Path(str(cut.source)).name)
    tmp = shard.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, crops=np.stack(crops), joints=np.array(joints, np.int64),
                            targets=np.stack(targets), weights=np.array(weights, np.float32),
                            frames=np.array(frames, np.int32), meta=np.array(json.dumps(meta)))
    tmp.replace(shard)
    return len(crops)


# --------------------------------------------------------------------------- self-learning (no user input)
PSEUDO_W = 0.3        # weight of a self-labelled sample (user correction 1.0, confirmed range 0.5)
AUTO_CAP = 400        # self-labelled samples kept per cut
AUTO_SHARDS = 60      # most recent cuts kept for self-learning
BAD = S.FLAG_DISAGREE | S.FLAG_FB | S.FLAG_BONE | S.FLAG_MISSING


def auto_shard(project, cut) -> Path:
    return samples_dir() / f"{project.name}__{cut.id}__auto.npz"


def collect_auto(project, cut, tracker, seed: int = 0) -> int:
    """Self-labelled samples from a freshly analysed cut (teacher-student self-training).

    Teacher: the automatic track, solved over the whole shot in both directions with optical flow, the
    pose ensemble, the face models and the person's own limb lengths. Student: the refiner, which only sees
    one frame. Only joints the teacher is sure about are used: visible, all pose models agree, the forward
    and backward passes agree, bone lengths plausible, a confident frame, no cut within 2 frames, not
    corrected by hand (those are user samples already). Frames where one frame alone got it wrong but the
    whole shot settled it (motion blur, a flash, a passing arm) are exactly what the student learns from,
    so samples with a larger teacher-measurement gap are preferred."""
    st = tracker.store
    shard = auto_shard(project, cut)
    if st is None or not (project.learn and cut.learn and settings().collect_learning_data
                          and getattr(settings(), "self_learning", True)):
        if shard.exists():
            shard.unlink()
        return 0
    proxy = tracker.proxy
    from .fusion import teacher_mask
    ok = teacher_mask(st, cut)
    for pid, jd in cut.joint_keys.items():
        if st.has(pid):
            p_ = st.index(pid)
            for j_, fr in jd.items():
                for t_ in fr:
                    if 0 <= int(t_) < st.n:
                        ok[p_, int(t_), int(j_)] = False
    if st.has_vis:
        ok &= st.vis_meas == 1
    cand = []
    for p, t, j in np.argwhere(ok):
        sc = st.scale[p, t]
        side = SIDE_FRAC * sc
        m = st.meas[p, t, j]
        a_ = st.auto[p, t, j]
        if not np.isfinite(m).all() or m[2] < 0.1:
            continue
        tgt = (a_[:2] - m[:2]) / side
        g = float(np.abs(tgt).max())
        if g > 0.45:
            continue
        cand.append((g, p, t, j, tgt.astype(np.float32), side))
    if len(cand) < 20:
        if shard.exists():
            shard.unlink()
        return 0
    rng = np.random.default_rng(seed + len(cand))
    gaps = np.array([c[0] for c in cand])
    # half of the budget for the hardest cases (largest teacher-measurement gap), half spread at random
    hard = np.argsort(-gaps)[:AUTO_CAP // 2]
    rest = np.setdiff1d(np.arange(len(cand)), hard)
    pick = np.concatenate([hard, rng.choice(rest, size=min(len(rest), AUTO_CAP - len(hard)), replace=False)])
    by_t: Dict[int, list] = {}
    for i in pick:
        by_t.setdefault(cand[i][2], []).append(cand[i])
    crops, joints, targets, frames = [], [], [], []
    for t, items in sorted(by_t.items()):
        rgb = np.asarray(proxy.frame(t))
        for g, p, _, j, tgt, side in items:
            c = st.meas[p, t, j, :2] * [proxy.sx, proxy.sy]
            crops.append(crop_at(rgb, c, side * 1.5 * proxy.sx, STORE))
            joints.append(j)
            targets.append(tgt)
            frames.append(t)
    meta = dict(project=project.name, project_path=project.path, cut=cut.name, cut_id=cut.id, client=project.client,
                created=time.time(), n=len(crops), pseudo=True, source=Path(str(cut.source)).name)
    tmp = shard.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, crops=np.stack(crops), joints=np.array(joints, np.int64),
                            targets=np.stack(targets), weights=np.full(len(crops), PSEUDO_W, np.float32),
                            frames=np.array(frames, np.int32), meta=np.array(json.dumps(meta)))
    tmp.replace(shard)
    # keep the most recent cuts only (memory during training stays bounded)
    autos = sorted(samples_dir().glob("*__auto.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in autos[AUTO_SHARDS:]:
        try:
            f.unlink()
        except OSError:
            pass
    return len(crops)


def new_auto_since_last() -> Tuple[int, int]:
    """(self-labelled samples, cuts) added after the newest trained version."""
    vers = list_versions()
    last = max([v.get("created", 0) for v in vers], default=0)
    n = k = 0
    for f in samples_dir().glob("*__auto.npz"):
        if f.stat().st_mtime > last:
            try:
                with np.load(f) as d:
                    n += len(d["joints"])
                k += 1
            except Exception:
                pass
    return n, k


def dataset_summary() -> dict:
    n = 0
    cuts = 0
    projects = set()
    rows = []
    for f in sorted(samples_dir().glob("*.npz")):
        try:
            with np.load(f) as d:
                meta = json.loads(str(d["meta"]))
                k = int(len(d["joints"]))
        except Exception:
            continue
        n += k
        cuts += 1
        projects.add(meta.get("project"))
        rows.append(dict(file=f.name, project=meta.get("project"), cut=meta.get("cut"), n=k,
                         created=meta.get("created", 0), pseudo=bool(meta.get("pseudo"))))
    pseudo = sum(r["n"] for r in rows if r["pseudo"])
    return dict(samples=n, cuts=cuts, projects=len(projects), rows=rows, pseudo=pseudo, human=n - pseudo)


def remove_project_samples(project_name: str) -> int:
    k = 0
    for f in samples_dir().glob(f"{project_name}__*.npz"):
        f.unlink()
        k += 1
    return k


def load_dataset():
    X, J, T, Wt, G = [], [], [], [], []
    gid: Dict[str, int] = {}
    for f in sorted(samples_dir().glob("*.npz")):
        try:
            with np.load(f) as d:
                try:
                    mt = json.loads(str(d["meta"]))
                    key = mt.get("source") or mt.get("cut_id") or f.stem
                except Exception:
                    key = f.stem
                gi = gid.setdefault(key, len(gid))     # a cut's user and self-labelled samples stay together
                X.append(d["crops"])
                J.append(d["joints"])
                T.append(d["targets"])
                Wt.append(d["weights"])
                G.append(np.full(len(d["joints"]), gi))
        except Exception:
            continue
    if not X:
        return None
    return np.concatenate(X), np.concatenate(J), np.concatenate(T), np.concatenate(Wt), np.concatenate(G)


# --------------------------------------------------------------------------- registry

def registry_dir() -> Path:
    d = settings().models_path / "refiner"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_versions() -> List[dict]:
    out = []
    for d in sorted(registry_dir().glob("v*")):
        mf = d / "meta.json"
        if mf.exists() and (d / "refiner.onnx").exists():
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                m["path"] = str(d)
                out.append(m)
            except Exception:
                pass
    out.sort(key=lambda m: m.get("version", 0))
    return out


def active_version() -> int:
    f = registry_dir() / "active.json"
    if f.exists():
        try:
            return int(json.loads(f.read_text())["version"])
        except Exception:
            return 0
    return 0


def set_active(version: int) -> None:
    f = registry_dir() / "active.json"
    f.write_text(json.dumps({"version": int(version), "changed": time.time()}))
    global _runtime
    _runtime = None


def update_meta(version: int, **kw) -> None:
    d = registry_dir() / f"v{version:03d}"
    mf = d / "meta.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        m.update(kw)
        mf.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def delete_version(version: int) -> None:
    if version == active_version():
        set_active(0)
    shutil.rmtree(registry_dir() / f"v{version:03d}", ignore_errors=True)


# --------------------------------------------------------------------------- runtime (ONNX)

class RefinerRuntime:
    def __init__(self, path: Path, version: int):
        from ..track.models import session
        self.sess = session(path)
        self.version = version
        self.gate = 0.02
        self.joints = set(range(S.K))
        mf = Path(path).parent / "meta.json"
        if mf.exists():
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                self.gate = float(m.get("gate", 0.02))
                self.joints = set(int(j) for j in m.get("joints", range(S.K)))
            except Exception:
                pass

    def predict(self, crops: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """crops (M,64,64,3) uint8 RGB -> offsets (M,2) in crop units, variance (M,) in crop units^2."""
        if len(crops) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
        x = (crops.astype(np.float32) / 255.0 - 0.5).transpose(0, 3, 1, 2)
        oh = np.eye(S.K, dtype=np.float32)[joints]
        outs = []
        for i in range(0, len(x), 512):
            outs.append(self.sess.run(None, {"crop": x[i:i + 512], "joint": oh[i:i + 512]})[0])
        o = np.concatenate(outs)
        return o[:, :2], np.exp(np.clip(o[:, 2], -12, 4))


_runtime: Optional[RefinerRuntime] = None


def active_runtime() -> Optional[RefinerRuntime]:
    global _runtime
    v = active_version()
    if v <= 0:
        return None
    if _runtime is not None and _runtime.version == v:
        return _runtime
    p = registry_dir() / f"v{v:03d}" / "refiner.onnx"
    if not p.exists():
        return None
    try:
        _runtime = RefinerRuntime(p, v)
    except Exception:
        _runtime = None
    return _runtime


def refine_measurements(store, proxy, rt: RefinerRuntime, frame_getter=None, sx=None, sy=None):
    """Extra measurement (xy (P,N,K,2), var (P,N,K)) from the refiner at every fused measurement."""
    P, N = store.present.shape
    xy = np.full((P, N, S.K, 2), np.nan, np.float32)
    var = np.full((P, N, S.K), np.inf, np.float32)
    get = frame_getter or (lambda t: np.asarray(proxy.frame(t)))
    sx = proxy.sx if sx is None else sx
    sy = proxy.sy if sy is None else sy
    for t in range(N):
        pres = np.nonzero(store.present[:, t])[0]
        if not len(pres):
            continue
        rgb = None
        crops, idx = [], []
        for p in pres:
            sc = store.scale[p, t]
            if not np.isfinite(sc):
                continue
            side = SIDE_FRAC * sc
            for j in range(S.K):
                if j not in rt.joints:       # only joints the model has actually been taught
                    continue
                m = store.meas[p, t, j]
                if not np.isfinite(m[:2]).all() or m[2] < 0.05:
                    continue
                if store.has_vis and store.vis_meas[p, t, j] != 1:     # hidden or beyond the frame edge
                    continue
                if rgb is None:
                    rgb = get(t)
                crops.append(crop_at(rgb, m[:2] * [sx, sy], side * sx, CROP))
                idx.append((p, j, side))
        if not crops:
            continue
        off, v = rt.predict(np.stack(crops), np.array([j for _, j, _ in idx]))
        for (p, j, side), o, vv in zip(idx, off, v):
            if vv > rt.gate or rt.gate <= 0:     # the model is unsure: no measurement
                continue
            xy[p, t, j] = store.meas[p, t, j, :2] + o * side
            var[p, t, j] = max(vv, 1e-4) * side * side
    return xy, var


# --------------------------------------------------------------------------- training (PyTorch)

def degrade(c: np.ndarray, rng, p: float = 0.5) -> np.ndarray:
    """What makes music video frames hard, applied to a crop: strobe/flash (gain and gamma), a single
    coloured club light, motion blur, sensor noise. Each with probability p."""
    x = c.astype(np.float32)
    if rng.random() < p:                                    # flash / dark frame
        x = 255.0 * (np.clip(x / 255.0 * rng.uniform(0.35, 2.2), 0, 1) ** rng.uniform(0.6, 1.6))
    if rng.random() < p * 0.6:                              # one-colour light
        g = rng.uniform(0.25, 1.0, 3).astype(np.float32)
        g[rng.integers(3)] = 1.0
        x = x * g
    if rng.random() < p * 0.6:                              # motion blur along a random direction
        L = int(rng.integers(3, 10))
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1.0 / L
        R = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        k = cv2.warpAffine(k, R, (L, L))
        k /= max(1e-6, k.sum())
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)
    if rng.random() < p * 0.5:
        x = x + rng.normal(0, rng.uniform(2, 9), x.shape).astype(np.float32)
    if rng.random() < p:
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    return np.clip(x, 0, 255).astype(np.uint8)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def train_new_version(progress: Optional[Callable[[float, str], None]] = None, epochs: int = 40,
                      cancel: Optional[Callable[[], bool]] = None) -> dict:
    import torch
    import torch.nn as nn

    data = load_dataset()
    if data is None or len(data[0]) < 40:
        raise RuntimeError("학습 데이터가 부족합니다. 관절 수정이나 확정 구간이 더 쌓이면 다시 시도하세요 (최소 40개).")
    X, J, T, Wt, G = data
    rng = np.random.default_rng(0)
    groups = np.unique(G)
    if len(groups) >= 3:
        val_groups = set(rng.choice(groups, size=max(1, len(groups) // 5), replace=False).tolist())
        val = np.isin(G, list(val_groups))
    else:
        val = rng.random(len(G)) < 0.2
    if val.sum() < 5 or (~val).sum() < 20:
        val = rng.random(len(G)) < 0.2
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                     nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
            self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128))
            self.j = nn.Linear(S.K, 32)
            self.h = nn.Sequential(nn.Linear(128 * 16 + 32, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                   nn.Linear(256, 3))

        def forward(self, crop, joint):
            a = self.f(crop).flatten(1)
            return self.h(torch.cat([a, torch.relu(self.j(joint))], 1))

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    tr_idx = np.nonzero(~val)[0]
    va_idx = np.nonzero(val)[0]
    eye = np.eye(S.K, dtype=np.float32)
    shift_px = (STORE - CROP) // 2

    def batch(ix, augment):
        crops, tg = [], []
        drng = np.random.default_rng(7)
        for i in ix:
            img = X[i]
            if augment is True:
                dx, dy = rng.integers(-shift_px, shift_px + 1, 2)
            elif augment == "degrade":
                # a hard frame: the per-frame estimate is off by up to 1/8 of the window as well
                dx, dy = drng.integers(-CROP // 8, CROP // 8 + 1, 2)
            else:
                dx = dy = 0
            x0, y0 = shift_px + dx, shift_px + dy
            # the stored crop covers 1.5 network windows
            c = img[y0:y0 + CROP, x0:x0 + CROP]
            t = T[i] - np.array([dx, dy], np.float32) / CROP
            if augment is True:
                c = degrade(c, rng, 0.5)
            elif augment == "degrade":
                c = degrade(c, drng, 1.0)
            crops.append(c)
            tg.append(t)
        x = torch.from_numpy((np.stack(crops).astype(np.float32) / 255.0 - 0.5).transpose(0, 3, 1, 2)).to(dev)
        j = torch.from_numpy(eye[J[ix]]).to(dev)
        t = torch.from_numpy(np.stack(tg).astype(np.float32)).to(dev)
        w = torch.from_numpy(Wt[ix]).to(dev)
        return x, j, t, w

    def nll(out, t, w):
        # the mean is trained directly; the variance head learns to predict the remaining error
        lv = out[:, 2].clamp(-12, 4)
        err = ((out[:, :2] - t) ** 2).sum(1)
        reg = torch.nn.functional.smooth_l1_loss(out[:, :2], t, reduction="none", beta=0.02).sum(1)
        return (w * (reg + 0.1 * (0.5 * torch.exp(-lv) * err.detach() + 0.5 * lv))).mean()

    bs = 64
    per_epoch = max(1, int(np.ceil(len(tr_idx) / bs)))
    epochs = int(np.clip(np.ceil(1500 / per_epoch), epochs, 300))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = None
    for ep in range(epochs):
        if cancel and cancel():
            raise RuntimeError("학습이 취소되었습니다.")
        net.train()
        rng.shuffle(tr_idx)
        for k in range(0, len(tr_idx), bs):
            x, j, t, w = batch(tr_idx[k:k + bs], True)
            opt.zero_grad()
            loss = nll(net(x, j), t, w)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            x, j, t, w = batch(va_idx, False)
            o = net(x, j)
            score = float(nll(o, t, torch.ones_like(w)))
            e_model = float((o[:, :2] - t).norm(dim=1).mean())
            # also on the same frames made hard (flash, colour light, blur, off by a few px): a model that
            # only ever says "stay" looks fine on clean frames but helps nowhere
            xh, jh, th, wh = batch(va_idx, "degrade")
            score += float(nll(net(xh, jh), th, torch.ones_like(wh)))
        if best is None or score < best[0]:
            best = (score, {k_: v.detach().cpu().clone() for k_, v in net.state_dict().items()})
        if progress:
            progress((ep + 1) / epochs, f"학습 {ep + 1}/{epochs} · 검증 오차 {e_model * 100:.1f}% (창 크기 대비)")
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        x, j, t, w = batch(va_idx, False)
        o = net(x, j)
        tn = t.norm(dim=1).cpu().numpy()
        em = (o[:, :2] - t).norm(dim=1).cpu().numpy()
        pv = torch.exp(o[:, 2].clamp(-12, 4)).cpu().numpy()
    human = Wt[va_idx] >= 0.5
    with torch.no_grad():
        xd, jd, td, _ = batch(va_idx, "degrade")
        od = net(xd, jd)
        em_d = (od[:, :2] - td).norm(dim=1).cpu().numpy()
        tn_d = td.norm(dim=1).cpu().numpy()
        pv_d = torch.exp(od[:, 2].clamp(-12, 4)).cpu().numpy()
    base = float(tn.mean())
    # choose the uncertainty gate that helps most on held-out data
    # (clean held-out frames and the same frames made hard; the clean ones must not get worse)
    best_gate, applied, best_tot = 0.0, base, float(tn.mean() + tn_d.mean())
    for g in np.unique(np.quantile(np.concatenate([pv, pv_d]), np.linspace(0.05, 1.0, 20))):
        app = float(np.where(pv <= g, em, tn).mean())
        tot = app + float(np.where(pv_d <= g, em_d, tn_d).mean())
        if app <= base * 1.005 and tot < best_tot - 1e-6:
            best_gate, applied, best_tot = float(g), app, tot
    coverage = float((pv <= best_gate).mean()) if best_gate > 0 else 0.0

    def gain(e, t_, v_, sel=None):
        if sel is not None:
            e, t_, v_ = e[sel], t_[sel], v_[sel]
        if len(e) == 0 or best_gate <= 0:
            return None
        b = float(t_.mean())
        a = float(np.where(v_ <= best_gate, e, t_).mean())
        return (1 - a / b) if b > 0 else None

    improve_human = gain(em, tn, pv, human) if human.sum() >= 10 else None
    improve_degraded = gain(em_d, tn_d, pv_d)
    versions = list_versions()
    ver = (max([m["version"] for m in versions]) + 1) if versions else 1
    out_dir = registry_dir() / f"v{ver:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    net_cpu = net.to("cpu")
    dummy = (torch.zeros(1, 3, CROP, CROP), torch.zeros(1, S.K))
    torch.onnx.export(net_cpu, dummy, str(out_dir / "refiner.onnx"), input_names=["crop", "joint"],
                      output_names=["out"], dynamic_axes={"crop": {0: "n"}, "joint": {0: "n"}, "out": {0: "n"}},
                      opset_version=17, dynamo=False)
    meta = dict(version=ver, created=time.time(), samples=int(len(X)), train=int(len(tr_idx)), val=int(len(va_idx)),
                cuts=int(len(groups)), val_baseline=base, val_model=applied, coverage=coverage, gate=best_gate,
                joints=[int(j_) for j_ in range(S.K) if int((J[tr_idx] == j_).sum()) >= 8],
                improvement=(1 - applied / base) if base > 0 else 0.0, bench={},
                improve_human=improve_human, improve_degraded=improve_degraded, n_human_val=int(human.sum()),
                pseudo=int((Wt < 0.5).sum()),
                val_raw=dict(clean_base=float(tn.mean()), clean_model=float(em.mean()),
                             hard_base=float(tn_d.mean()), hard_model=float(em_d.mean())))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
