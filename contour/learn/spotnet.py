"""Spot detector learned from hand-retouched footage (experiment; DIAP's own detector stays the default).

Labels: on skin_samples (contour/learn/skinnet.py) the places where the hand removed a spot: a compact change of the
hand that undoes a local flaw of the plate, well above this face's usual texture change (a dark spot lightened, a red
spot made less red, a raised spot flattened). Kept away from the skin mask's border (the hairline, where hair edges
changed) and from what DIAP's protection map marks (hair strands, jewellery, strong edges).

Network: a small U-Net on the sample's picture (face width 300) and skin mask, three outputs (dark, red, raised).
Evaluation on held-out shots, against DIAP's rule detector at the same number of marks per face: how many of the
hand's spots each finds, and how many of its marks the hand left alone.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from . import skinnet as SN

KINDS = ("dark", "red", "bump")
EDGE = 0.03          # of the face width: labels this close to the skin mask's border are left out (hairline)
WF = float(SN.FACE_W)


def _ycc(rgb):
    from ..retouch import engine as RE
    return RE.rgb_to_ycc(np.clip(rgb, 0, 1).astype(np.float32))


def labels(s: dict) -> Optional[Dict[str, np.ndarray]]:
    """{'lab': uint8 0 none / 1 dark / 2 red / 3 raised, 'valid': where labels mean something} for one sample."""
    from ..retouch import detect as DT
    b, d, sk = SN._f(s["before"]), SN._f(s["delta"]), SN._f(s["skin"])
    on = sk > 0.5
    r = max(2, int(round(EDGE * WF)))
    on = cv2.erode(on.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    if on.sum() < 3000:
        return None
    Yb, Cbb, Crb = _ycc(b)
    prot = DT.protect(Yb, Cbb, Crb, sk.astype(np.float32), WF) > 0.5
    on &= ~cv2.dilate(prot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    Ya, _, Cra = _ycc(np.clip(b + d, 0, 1))
    dog = lambda z: cv2.GaussianBlur(z, (0, 0), 1.2) - cv2.GaussianBlur(z, (0, 0), 6.0)
    hY, hC, pY, pC = dog(Ya - Yb), dog(Cra - Crb), dog(Yb), dog(Crb)
    rs = lambda z: 1.4826 * float(np.median(np.abs(z[on] - np.median(z[on])))) + 1e-5
    tY, tC = max(0.012, 4 * rs(hY)), max(0.006, 4 * rs(hC))
    qY, qC = max(0.008, 3 * rs(pY)), max(0.004, 3 * rs(pC))
    lab = np.zeros(sk.shape, np.uint8)
    lab[(hY > tY) & (pY < -qY) & on] = 1
    lab[(hC < -tC) & (pC > qC) & on & (lab == 0)] = 2
    lab[(hY < -tY) & (pY > qY) & on & (lab == 0)] = 3
    out = np.zeros_like(lab)
    for code in (1, 2, 3):
        n, cc, st, _ = cv2.connectedComponentsWithStats((lab == code).astype(np.uint8), connectivity=8)
        for j in range(1, n):
            if st[j, 4] >= 6 and max(st[j, 2], st[j, 3]) <= 0.08 * WF:
                out[cc == j] = code
    return dict(lab=out, valid=on)


def rule_scores(s: dict) -> np.ndarray:
    """DIAP's rule detector on the sample: (3, H, W) dark, red, raised, protection taken out."""
    from ..retouch import detect as DT
    b, sk = SN._f(s["before"]), SN._f(s["skin"]).astype(np.float32)
    Y, Cb, Cr = _ycc(b)
    d = DT.spots(Y, Cr, sk, (0.012 + 0.05 * 0.5) * WF)
    p = DT.protect(Y, Cb, Cr, sk, WF)
    return np.stack([d[k] * (1 - p) for k in KINDS]).astype(np.float32)


def build_net(width: int = 24):
    import torch
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU(inplace=True),
                             nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU(inplace=True))

    class SpotNet(nn.Module):
        def __init__(s, w=width):
            super().__init__()
            c = [w, 2 * w, 4 * w]
            s.e0, s.e1, s.e2 = blk(4, c[0]), blk(c[0], c[1]), blk(c[1], c[2])
            s.pool = nn.AvgPool2d(2)
            s.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            s.d1, s.d0 = blk(c[2] + c[1], c[1]), blk(c[1] + c[0], c[0])
            s.head = nn.Conv2d(c[0], 3, 1)
            nn.init.constant_(s.head.bias, -4.0)

        def forward(s, x):
            z0 = s.e0(torch.cat([x[:, :3] - 0.5, x[:, 3:4]], 1))
            z1 = s.e1(s.pool(z0))
            z2 = s.e2(s.pool(z1))
            y = s.d1(torch.cat([s.up(z2), z1], 1))
            y = s.d0(torch.cat([s.up(y), z0], 1))
            return s.head(y)                     # logits

    return SpotNet()


def _batch(items: List[dict], idx, rng, size: int, train: bool):
    xs, ys, ws = [], [], []
    for i in idx:
        it = items[i]
        H = it["x"].shape[1]
        if train and size < H:
            y0, x0 = rng.integers(0, H - size + 1, 2)
        else:
            y0 = x0 = (H - size) // 2 if size < H else 0
        x = it["x"][:, y0:y0 + size, x0:x0 + size].astype(np.float32)
        lab = it["lab"][y0:y0 + size, x0:x0 + size]
        val = it["valid"][y0:y0 + size, x0:x0 + size]
        if train:
            if rng.random() < 0.5:
                x, lab, val = x[:, :, ::-1], lab[:, ::-1], val[:, ::-1]
            g = float(rng.uniform(0.8, 1.2))
            x = np.concatenate([np.clip(x[:3] * g, 0, 1), x[3:]], 0)
        y = np.stack([(lab == k + 1) for k in range(3)]).astype(np.float32)
        y = np.stack([cv2.dilate(c, np.ones((3, 3), np.uint8)) for c in y])     # a pixel off is still the spot
        xs.append(np.ascontiguousarray(x))
        ys.append(y)
        ws.append(val[None].astype(np.float32))
    return np.stack(xs), np.stack(ys), np.stack(ws)


def item(s: dict) -> Optional[dict]:
    L = labels(s)
    if L is None:
        return None
    x = np.concatenate([SN._f(s["before"]).transpose(2, 0, 1), SN._f(s["skin"])[None]], 0).astype(np.float16)
    return dict(x=x, lab=L["lab"], valid=L["valid"], meta=s["meta"])


def train(items: List[dict], val: List[int], steps=6000, batch=12, size=256, lr=2e-3, width=24, device=None,
          log=print, seed=0):
    import torch
    import torch.nn.functional as F
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vs = set(val)
    tr = [i for i in range(len(items)) if i not in vs]
    net = build_net(width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(2, steps), pct_start=0.1)
    pos_w = torch.tensor([20.0, 20.0, 20.0], device=dev).view(1, 3, 1, 1)
    hist = []
    for it in range(steps):
        net.train()
        idx = rng.choice(tr, size=min(batch, len(tr)), replace=len(tr) < batch)
        x, y, w = (torch.from_numpy(a).to(dev) for a in _batch(items, idx, rng, size, True))
        l = F.binary_cross_entropy_with_logits(net(x), y, pos_weight=pos_w, reduction="none")
        loss = (l * w).sum() / (w.sum() * 3).clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps // 10) == 0:
            hist.append(dict(step=it + 1, loss=float(loss)))
            log(f"step {it + 1}/{steps} loss {float(loss):.4f}")
    return net, hist


def predict(net, it: dict, device=None) -> np.ndarray:
    import torch
    dev = device or next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(it["x"].astype(np.float32)[None]).to(dev)
        return torch.sigmoid(net(x)).cpu().numpy()[0]


def score(maps: List[np.ndarray], items: List[dict], thr: float) -> dict:
    """Found share of the hand's spots (any kind), and marks per face / share the hand left alone, at threshold thr."""
    hand = found = marks = alone = 0
    for m, it in zip(maps, items):
        det = (m.max(0) > thr) & (SN._f(np.asarray(it["x"][3])) > 0.5)
        det &= it["valid"]
        dd = cv2.dilate(det.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        lab = it["lab"] > 0
        n, cc, st, _ = cv2.connectedComponentsWithStats(lab.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            hand += 1
            found += bool(dd[cc == j].any())
        hd = cv2.dilate(lab.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        n, cc, st, _ = cv2.connectedComponentsWithStats(det.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            marks += 1
            alone += not hd[cc == j].any()
    return dict(hand=hand, found=round(found / max(1, hand), 3), marks_per_face=round(marks / max(1, len(items)), 2),
                left_alone=round(alone / max(1, marks), 3))


def matched(maps_a: List[np.ndarray], maps_b: List[np.ndarray], items: List[dict], thr_a: float = 0.5) -> dict:
    """Compare b to a at the same marks per face: b's threshold is chosen so it marks as much as a does at thr_a."""
    a = score(maps_a, items, thr_a)
    best = None
    for t in np.linspace(0.05, 0.95, 19):
        s = score(maps_b, items, float(t))
        if best is None or abs(s["marks_per_face"] - a["marks_per_face"]) < abs(best[1]["marks_per_face"] - a["marks_per_face"]):
            best = (float(t), s)
    return dict(rule=a, net=dict(best[1], thr=round(best[0], 2)))
