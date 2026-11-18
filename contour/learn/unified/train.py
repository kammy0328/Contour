"""Training the combined model on teacher labels (teach.py). A sample teaches only the heads it has labels for
(a face crop: the face head; a person crop: clothing and alpha; a frame: alpha). Whole projects are held out for
the check, fixed once in <out>/val_fixed.json, so similar cuts of one music video are never on both sides.
The exported file is checked (loads, runs, shapes, finite, agrees with torch) and kept in <out>; putting it into
the app is a separate, explicit step after the comparison (bench.py)."""
from __future__ import annotations

import json
import math
import os
import random
import time
import zlib
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import teach as T
from .net import CLOTH_CLASSES, FACE_CLASSES

FACE_FLIP = [0, 1, 2, 3, 4, 5]                 # MediaPipe multiclass has no left / right classes
CLOTH_FLIP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 13, 12, 15, 14, 16, 17]
FACE_KEY = (2, 3)               # body skin (the neck), face skin: what the retouch needs most
CLOTH_KEY = (4, 5, 6, 7)        # top, skirt, trousers, dress: what 옷 그대로 keeps


# ------------------------------------------------------------------ split
def split(items: list, out_dir: str, n_val: int = 1) -> tuple:
    """(train, val, held projects). The held projects are chosen once (from their names) and kept."""
    f = os.path.join(out_dir, "val_fixed.json")
    from ..safe_install import atomic_json, read_json
    fixed = read_json(f, {})
    projects = sorted({it[2] for it in items})
    held = [p for p in fixed.get("projects", []) if p in projects]
    if not held:
        full = [p for p in projects if {it[1] for it in items if it[2] == p} >= {"face", "person", "frame"}]
        pool = full or projects
        held = sorted(pool, key=lambda p: zlib.crc32(p.encode("utf-8")))[:n_val]
        fixed["projects"] = sorted(set(fixed.get("projects", [])) | set(held))
        atomic_json(f, fixed)
    tr = [it for it in items if it[2] not in held]
    va = [it for it in items if it[2] in held]
    return tr, va, held


# ------------------------------------------------------------------ samples
def load(path: str) -> dict:
    with np.load(path) as d:
        s = {k: d[k] for k in d.files}
    s["img"] = T.unjpg(s["img"])
    s["meta"] = json.loads(str(s["meta"]))
    return s


def degrade(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Input only: dark / washed out / coloured light / small and soft / blur / noise / JPEG."""
    f = x.astype(np.float32) / 255.0
    r = rng.random()
    if r < 0.3:
        f = f ** rng.uniform(1.4, 2.6) * rng.uniform(0.35, 0.8)
    elif r < 0.5:
        f = np.clip(f * rng.uniform(1.3, 2.0) + rng.uniform(0.0, 0.12), 0, 1) ** rng.uniform(0.6, 0.9)
    if rng.random() < 0.4:
        f = f * np.array([rng.uniform(0.75, 1.2), rng.uniform(0.85, 1.1), rng.uniform(0.75, 1.25)], np.float32)
    f = np.clip(f, 0, 1)
    h, w = f.shape[:2]
    if rng.random() < 0.4:
        k = rng.uniform(0.25, 0.7)
        f = cv2.resize(cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.6, 2.0))
    if rng.random() < 0.4:
        f = f + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0.004, 0.025), f.shape).astype(np.float32)
    u = np.clip(f * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, b = cv2.imencode(".jpg", u, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 85)])
        u = cv2.imdecode(b, cv2.IMREAD_UNCHANGED).reshape(u.shape)
    return u


def _warp_all(s: dict, M: np.ndarray) -> dict:
    """The same small move on the picture and every label (class maps at 1/4 size)."""
    out = dict(s)
    h, w = s["img"].shape[:2]
    out["img"] = cv2.warpAffine(s["img"], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if "alpha" in s:
        out["alpha"] = cv2.warpAffine(s["alpha"], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M4 = M.copy()
    M4[:, 2] /= 4.0
    for k in ("face", "cloth"):
        if k in s:
            c, hh, ww = s[k].shape
            out[k] = np.stack([cv2.warpAffine(s[k][i], M4, (ww, hh), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT) for i in range(c)])
    return out


def augment(s: dict, rng: random.Random) -> dict:
    s = dict(s)
    if rng.random() < 0.5:
        s["img"] = s["img"][:, ::-1].copy()
        if "alpha" in s:
            s["alpha"] = s["alpha"][:, ::-1].copy()
        if "face" in s:
            s["face"] = s["face"][FACE_FLIP][:, :, ::-1].copy()
        if "cloth" in s:
            s["cloth"] = s["cloth"][CLOTH_FLIP][:, :, ::-1].copy()
    if rng.random() < 0.6:
        h, w = s["img"].shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
        M[:, 2] += (rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h)
        s = _warp_all(s, M)
    s["img"] = degrade(s["img"], rng) if rng.random() < 0.8 else s["img"]
    return s


def to_batch(samples: List[dict], dev):
    import torch
    x = torch.from_numpy(np.stack([T.to_input(s["img"])[0] for s in samples])).to(dev)
    lab = {}
    for k in ("face", "cloth", "alpha"):
        if all(k in s for s in samples):
            a = np.stack([s[k] for s in samples]).astype(np.float32) / 255.0
            lab[k] = torch.from_numpy(a[:, None] if k == "alpha" else a).to(dev)
    return x, lab


# ------------------------------------------------------------------ losses
def soft_ce(logits, prob, key_classes):
    import torch
    import torch.nn.functional as F
    prob = prob / prob.sum(1, keepdim=True).clamp(min=1e-6)
    conf, _ = prob.max(1)
    pw = 0.3 + 0.7 * conf
    w = torch.ones(prob.shape[1], device=logits.device)
    w[list(key_classes)] = 2.0
    ce = -(prob * F.log_softmax(logits, 1) * w.view(1, -1, 1, 1)).sum(1)
    return (ce * pw).sum() / pw.sum().clamp(min=1)


def alpha_loss(pred, tgt):
    import torch
    import torch.nn.functional as F
    l1 = (pred - tgt).abs()
    edge = ((tgt > 0.02) & (tgt < 0.98)).float()
    edge = F.max_pool2d(edge, 7, 1, 3)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3) / 8
    g = lambda z: torch.cat([F.conv2d(z, kx, padding=1), F.conv2d(z, kx.transpose(2, 3), padding=1)], 1)
    return l1.mean() + 2.0 * (l1 * edge).sum() / edge.sum().clamp(min=1) + (g(pred) - g(tgt)).abs().mean()


# ------------------------------------------------------------------ check during training
def quick_eval(model, items: list, dev, n: int = 120) -> dict:
    """Agreement with the teachers on held-out samples (no degradation): skin+nose / neck IoU, garment IoU,
    alpha mean absolute difference."""
    import torch
    model.eval()
    r = random.Random(11)
    pick = r.sample(items, min(n, len(items)))
    acc = dict(skin=[], neck=[], garment=[], alpha_mad=[])
    with torch.no_grad():
        for it in pick:
            s = load(it[0])
            x, _ = to_batch([s], dev)
            f, c, a = model(x)
            if "face" in s:
                pf = torch.softmax(f.float(), 1)[0].cpu().numpy()
                tf = s["face"].astype(np.float32) / 255.0
                for name, ids in (("skin", [3]), ("neck", [2])):
                    pm, tm = pf[ids].sum(0) > 0.5, tf[ids].sum(0) > 0.5
                    if tm.sum() > 20:
                        acc[name].append((pm & tm).sum() / max(1, (pm | tm).sum()))
            if "cloth" in s:
                pc = torch.softmax(c.float(), 1)[0].cpu().numpy().argmax(0)
                tc = s["cloth"].argmax(0)
                g_p, g_t = np.isin(pc, CLOTH_KEY), np.isin(tc, CLOTH_KEY)
                if g_t.sum() > 50:
                    acc["garment"].append((g_p & g_t).sum() / max(1, (g_p | g_t).sum()))
            if "alpha" in s:
                acc["alpha_mad"].append(float(np.abs(a[0, 0].float().cpu().numpy() - s["alpha"] / 255.0).mean()))
    model.train()
    return {k: (round(float(np.mean(v)), 4) if v else None) for k, v in acc.items()}


def _score(e: dict) -> float:
    s = 0.0
    for k in ("skin", "neck", "garment"):
        if e.get(k) is not None:
            s += e[k]
    if e.get("alpha_mad") is not None:
        s += 1.0 - 5.0 * e["alpha_mad"]
    return s


def train(root: str, out_dir: str, steps: int = 8000, batch: int = 8, lr: float = 4e-4, log=print,
          seed: int = 0) -> dict:
    import torch
    from . import net as N
    os.makedirs(out_dir, exist_ok=True)
    items = T.index(root)
    tr, va, held = split(items, out_dir)
    if not tr or not va:
        raise RuntimeError(f"학습 {len(tr)}개 / 검증 {len(va)}개: 프로젝트가 2개 이상 필요합니다")
    by_kind = {k: [it for it in tr if it[1] == k] for k in ("face", "person", "frame")}
    kinds = [k for k in by_kind if by_kind[k]]
    log(f"학습 {len(tr)} (얼굴 {len(by_kind['face'])}, 인물 {len(by_kind['person'])}, 프레임 {len(by_kind['frame'])}), "
        f"검증 {len(va)} (프로젝트 {held})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = random.Random(seed)
    with np.load(next(it[0] for it in tr if it[1] == "face")) as d0:
        fc = int(d0["face"].shape[0]) if "face" in d0.files else N.FACE_CLASSES
    model = N.build(pretrained=True, face_classes=fc).to(dev)
    enc = [p for n_, p in model.named_parameters() if n_.startswith(("stem", "l1", "l2", "l3", "l4"))]
    dec = [p for n_, p in model.named_parameters() if not n_.startswith(("stem", "l1", "l2", "l3", "l4"))]
    opt = torch.optim.AdamW([dict(params=enc, lr=lr * 0.3), dict(params=dec, lr=lr)], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, steps)))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, hist = None, []
    t0 = time.time()
    run = {k: 0.0 for k in ("face", "cloth", "alpha")}
    model.train()
    for step in range(steps):
        kind = kinds[step % len(kinds)]
        want = batch if kind != "frame" else max(2, batch // 2)
        pick = [load(rng.choice(by_kind[kind])[0])]
        tries = 0
        while len(pick) < want and tries < 8 * want:
            s_ = load(rng.choice(by_kind[kind])[0])
            tries += 1
            if s_["img"].shape == pick[0]["img"].shape:          # frames of other aspect ratios: another batch
                pick.append(s_)
        x, lab = to_batch([augment(s, rng) for s in pick], dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            f, c, a = model(x)
        loss = 0.0
        # only the heads this sample has answers for
        if "face" in lab:
            lf = soft_ce(f.float(), lab["face"], FACE_KEY)
            loss = loss + lf
            run["face"] = 0.98 * run["face"] + 0.02 * lf.item()
        if "cloth" in lab:
            lc = soft_ce(c.float(), lab["cloth"], CLOTH_KEY)
            loss = loss + lc
            run["cloth"] = 0.98 * run["cloth"] + 0.02 * lc.item()
        if "alpha" in lab:
            la = alpha_loss(a.float(), lab["alpha"])
            loss = loss + la
            run["alpha"] = 0.98 * run["alpha"] + 0.02 * la.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if (step + 1) % max(1, steps // 16) == 0 or step + 1 == steps:
            e = quick_eval(model, va, dev)
            el = time.time() - t0
            hist.append(dict(step=step + 1, minutes=round(el / 60, 1), loss={k: round(v, 4) for k, v in run.items()}, check=e))
            log(f"스텝 {step + 1}/{steps} {el / 60:.0f}분 loss {run} 검증 {e}")
            if best is None or _score(e) > best[0]:
                best = (_score(e), step + 1, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}, e)
    model.load_state_dict(best[2])
    pt = os.path.join(out_dir, "unified_diap.pt")
    torch.save(model.state_dict(), pt)
    onnx_path = os.path.join(out_dir, "unified_diap.onnx")
    chk = N.export(model, onnx_path)
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), steps=steps, best_step=best[1], check=best[3], held=held,
               train=len(tr), val=len(va), onnx=onnx_path if chk["ok"] else None, onnx_check=chk["ok"],
               onnx_detail={k: v for k, v in chk.items() if isinstance(v, dict)}, hist=hist)
    from ..safe_install import atomic_json
    atomic_json(os.path.join(out_dir, "train.json"), rec)
    return rec
