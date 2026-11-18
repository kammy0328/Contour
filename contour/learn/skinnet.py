"""Skin retouch network learned from hand-retouched footage (before / after pairs of the same frames).

What the hand result teaches is only the skin retouch. Everything else it carries is taken out first:
- a grade over the whole picture (colour fit, style.fit_grade, given by the caller from the full frames),
- liquify and face reshaping (dense flow from the plate to the hand result, style.align_crop; what moved a lot is
  left out),
- skin tone (the broad part of the difference, wider than TONE_SIGMA of the face width, measured on the skin only),
- cleaning and anything else that changed a lot in a few places (robust outliers),
- acne and spots: DIAP removes them itself (the user: its own removal can do better than the hand), so the spots
  DIAP's detector finds are left out of the loss,
- grain: the finest band of the difference is dropped; grain is always the plate's own.

A sample is a square around one face and its neck, scaled so the face is FACE_W pixels wide: the plate (RGB), the
skin mask, the hand's retouch on it as a correction field ("delta"), how much of the band finer than the sample's
pixels the hand kept ("keep": pores and fine texture at the plate's own resolution), a loss weight, and the same
terms for DIAP's current retouch of the frame (for the comparison). The network reads the plate and the skin mask
and gives delta and keep; it is run at the sample's scale and its delta is laid on the plate at full resolution,
so the plate's grain is never touched.

Training data, the split by shots (adjacent frames of a shot never in both training and validation), training and
evaluation run on the user's PC (tools/train_skinnet.py); nothing here changes the app's retouch by itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

K = np.array([0.2126, 0.7152, 0.0722], np.float32)
FACE_W = 300            # face width in a sample (px)
CROP = 640              # sample side (px): the face, the hairline and the neck
TONE_SIGMA = 0.15       # of the face width: broader parts of the difference are skin tone (not learned)
GRAIN_SIGMA = 0.8       # full resolution px: the finest band of the difference is grain (never learned)
OUTLIER = (6.0, 0.03)   # robust sigmas and absolute floor (0..1) from which a change is cleaning, not retouch
DELTA_MAX = 0.2
KEEP_MAX = 1.2
MIN_WF = 150            # px of the plate: smaller faces teach nothing about skin


# ------------------------------------------------------------------------------------------------ geometry
def face_width(lm: np.ndarray) -> float:
    f = np.asarray(lm, np.float32)[:, :2]
    return float(np.hypot(*(f[16] - f[0])))


def crop_geom(lm: np.ndarray) -> Tuple[float, float, float, float]:
    """(centre x, centre y, side in plate px, scale sample px per plate px) of the square around face and neck."""
    f = np.asarray(lm, np.float32)[:, :2]
    wf = face_width(lm)
    c = f[:68].mean(0)
    down = f[8] - f[27]
    down = down / max(1e-6, float(np.hypot(*down)))
    cx, cy = c + down * 0.35 * wf
    scale = FACE_W / max(wf, 1e-6)
    return float(cx), float(cy), CROP / scale, scale


def cut_square(img: np.ndarray, cx: float, cy: float, side: float, out: Optional[int] = None,
               border=cv2.BORDER_REFLECT) -> np.ndarray:
    """The square (cx, cy, side) of img at plate resolution (out None) or resized to out x out."""
    s = int(round(side))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    H, W = img.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x0 + s - W), max(0, y0 + s - H)
    a = img[max(0, y0):min(H, y0 + s), max(0, x0):min(W, x0 + s)]
    if px0 or py0 or px1 or py1:
        if a.size == 0:
            a = np.zeros((s, s) + img.shape[2:], img.dtype)
        else:
            a = cv2.copyMakeBorder(a, py0, py1, px0, px1, border, value=0)
    if out is not None and a.shape[0] != out:
        a = cv2.resize(a, (out, out), interpolation=cv2.INTER_AREA if a.shape[0] > out else cv2.INTER_LINEAR)
    return a


# ------------------------------------------------------------------------------------------------ terms
def _mblur(x: np.ndarray, w: np.ndarray, sig: float) -> np.ndarray:
    """Blur of x weighted by w (skin only), plain blur where there is no weight around."""
    sig = max(0.5, float(sig))
    if x.ndim == 3:
        return np.stack([_mblur(x[..., i], w, sig) for i in range(x.shape[2])], -1)
    num = cv2.GaussianBlur(x * w, (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), cv2.GaussianBlur(x, (0, 0), sig)).astype(np.float32)


def retouch_terms(b: np.ndarray, x: np.ndarray, skin: np.ndarray, wf: float, scale: float):
    """What x does to b on the skin (plate px, float RGB): (delta without tone and grain, keep ratio of the band
    finer than the sample's pixels, |change| for outliers). All at plate resolution."""
    w = np.clip(skin, 0, 1).astype(np.float32)
    D = (x - b).astype(np.float32)
    D = D - _mblur(D, w, TONE_SIGMA * wf)                      # skin tone out
    Dg = cv2.GaussianBlur(D, (0, 0), GRAIN_SIGMA)              # grain out
    sc = max(GRAIN_SIGMA + 0.5, 1.0 / max(scale, 1e-6))        # plate px per sample px
    yb, yx = b @ K, x @ K

    def band(y):
        return cv2.GaussianBlur(y, (0, 0), GRAIN_SIGMA) - cv2.GaussianBlur(y, (0, 0), sc)

    def energy(y):
        return np.sqrt(cv2.GaussianBlur(band(y) ** 2, (0, 0), 4 * sc)) + 1e-4

    keep = np.clip(energy(yx) / energy(yb), 0, KEEP_MAX).astype(np.float32)
    return Dg, keep, cv2.GaussianBlur(np.abs(D @ K), (0, 0), 1.0)


def make_sample(b: np.ndarray, a: np.ndarray, lm: np.ndarray, skin: np.ndarray, G: Optional[np.ndarray] = None,
                diap: Optional[np.ndarray] = None, spots: Optional[Callable] = None) -> Optional[dict]:
    """b, a: the plate and the hand result of one frame (float RGB 0..1, full frame, same size); lm: 68 face
    landmarks (plate px); skin: the skin mask of this person (plate size, 0..1: DIAP's skin and neck masks); G: the
    hand result's grade (style.fit_grade on the whole frames); diap: DIAP's retouch of the frame (for the
    comparison); spots(crop rgb, landmarks, mask) -> bool map of the spots to leave out."""
    from . import style as SL
    wf = face_width(lm)
    if not np.isfinite(wf) or wf < MIN_WF:
        return None
    cx, cy, side, scale = crop_geom(lm)
    bc = cut_square(b, cx, cy, side).astype(np.float32)
    ac = cut_square(a, cx, cy, side).astype(np.float32)
    sk = cut_square(skin.astype(np.float32), cx, cy, side, border=cv2.BORDER_CONSTANT)
    if float((sk > 0.5).mean()) < 0.03:
        return None
    s = bc.shape[0]
    loc = np.asarray(lm, np.float32)[:, :2] - [cx - s / 2.0, cy - s / 2.0]
    bg = SL.apply_grade(G, bc) if G is not None else bc
    al, ok, wpx = SL.align_crop(bg, SL.apply_grade(None, ac), sk > 0.5)
    # the hand's change measured on the graded plate, laid on the plate itself
    T, keep, mag = retouch_terms(bg, al, sk * ok, wf, scale)
    on = (sk > 0.5) & ok
    if on.sum() < 2000:
        return None
    med = float(np.median(mag[on]))
    mad = 1.4826 * float(np.median(np.abs(mag[on] - med))) + 1e-5
    out = mag > max(med + OUTLIER[0] * mad, OUTLIER[1])
    r = max(1, int(round(0.015 * wf)))
    out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    sp = spots(bc, loc, sk > 0.5) if spots is not None else np.zeros(on.shape, bool)
    wgt = (on & ~out & ~sp).astype(np.float32)
    # not near the crop's border (reflected padding is no picture)
    e = max(2, int(0.02 * s))
    wgt[:e], wgt[-e:], wgt[:, :e], wgt[:, -e:] = 0, 0, 0, 0
    R = lambda z: cv2.resize(z, (CROP, CROP), interpolation=cv2.INTER_AREA if z.shape[0] > CROP else cv2.INTER_LINEAR)
    smp = dict(before=np.clip(R(bc), 0, 1), skin=np.clip(R(sk), 0, 1), delta=np.clip(R(T), -DELTA_MAX, DELTA_MAX),
               keep=R(keep), weight=(R(wgt) > 0.5).astype(np.float32),
               meta=dict(wf=wf, scale=scale, side=side, cx=cx, cy=cy, warp_px=float(wpx),
                         outlier_share=float((out & on).sum()) / max(1, int(on.sum())),
                         spot_share=float((sp & on).sum()) / max(1, int(on.sum())), graded=G is not None))
    if diap is not None:
        dc = cut_square(diap, cx, cy, side).astype(np.float32)
        Td, kd, _ = retouch_terms(bc, dc, sk, wf, scale)
        smp["diap_delta"] = np.clip(R(Td), -DELTA_MAX, DELTA_MAX)
        smp["diap_keep"] = R(kd)
    return smp


def save_sample(path, smp: dict) -> None:
    arr = dict(before=np.round(smp["before"] * 65535).astype(np.uint16),
               skin=np.round(smp["skin"] * 255).astype(np.uint8),
               delta=smp["delta"].astype(np.float16), keep=smp["keep"].astype(np.float16),
               weight=smp["weight"].astype(np.uint8), meta=np.array(json.dumps(smp["meta"])))
    for k in ("diap_delta", "diap_keep"):
        if k in smp:
            arr[k] = smp[k].astype(np.float16)
    tmp = str(path) + ".part.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, str(path))


def load_sample(path, compact: bool = False, diap: bool = True) -> dict:
    """compact: the arrays as stored (uint16 / float16 / uint8, a third of the memory; read with _f), for training on
    a thousand samples in the PC's memory."""
    z = np.load(str(path))
    keys = ["before", "skin", "delta", "keep", "weight"] + ([k for k in ("diap_delta", "diap_keep") if k in z.files]
                                                            if diap else [])
    d = {k: z[k] for k in keys}
    d["meta"] = json.loads(str(z["meta"]))
    if not compact:
        d = {k: (_f(v) if k != "meta" else v) for k, v in d.items()}
    return d


def _f(a: np.ndarray) -> np.ndarray:
    """A sample array as float32 (0..1 for the stored integer pictures and masks)."""
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    if a.dtype == np.uint8:
        return a.astype(np.float32) / (255.0 if a.max(initial=0) > 1 else 1.0)
    return a.astype(np.float32)


# ------------------------------------------------------------------------------------------------ split
def split_by_shot(shots: List[int], val_share: float = 0.2, block: int = 6, seed: int = 0) -> set:
    """Validation shots: whole runs of `block` consecutive shots (a person, a set, a light stays over a few shots;
    a shot's own frames never go to both sides)."""
    u = sorted(set(int(s) for s in shots))
    if len(u) < 3:
        return set(u[-1:])
    blocks = [u[i:i + block] for i in range(0, len(u), block)]
    rng = np.random.default_rng(seed)
    n = max(1, int(round(val_share * len(blocks))))
    pick = rng.choice(len(blocks), size=n, replace=False)
    return {s for i in pick for s in blocks[int(i)]}


# ------------------------------------------------------------------------------------------------ network
def build_net(width: int = 32):
    import torch
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU(inplace=True),
                             nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU(inplace=True))

    class SkinNet(nn.Module):
        """U-Net: plate RGB + skin mask -> delta RGB (bounded) and keep (0..KEEP_MAX)."""

        def __init__(self, w=width):
            super().__init__()
            c = [w, 2 * w, 4 * w, 8 * w]
            self.e0, self.e1, self.e2, self.e3 = blk(4, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3])
            self.pool = nn.AvgPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d2, self.d1, self.d0 = blk(c[3] + c[2], c[2]), blk(c[2] + c[1], c[1]), blk(c[1] + c[0], c[0])
            self.head = nn.Conv2d(c[0], 4, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            rgb, m = x[:, :3], x[:, 3:4]
            z0 = self.e0(torch.cat([rgb - 0.5, m], 1))
            z1 = self.e1(self.pool(z0))
            z2 = self.e2(self.pool(z1))
            z3 = self.e3(self.pool(z2))
            y = self.d2(torch.cat([self.up(z3), z2], 1))
            y = self.d1(torch.cat([self.up(y), z1], 1))
            y = self.d0(torch.cat([self.up(y), z0], 1))
            h = self.head(y)
            delta = DELTA_MAX * torch.tanh(h[:, :3]) * m
            # 0..KEEP_MAX, exactly 1 (the plate's texture as it is) for the untrained head
            keep = KEEP_MAX * torch.sigmoid(h[:, 3:4] + float(np.log(1.0 / (KEEP_MAX - 1.0))))
            keep = m * keep + (1 - m)
            return torch.cat([delta, keep], 1)

    return SkinNet()


def _batch(samples: List[dict], idx, rng, size: int, train: bool):
    xs, ys, ws = [], [], []
    for i in idx:
        s = samples[i]
        H = s["before"].shape[0]
        if train and size < H:
            y0, x0 = rng.integers(0, H - size + 1, 2)
        else:
            y0 = x0 = (H - size) // 2 if size < H else 0
        sl = (slice(y0, y0 + size), slice(x0, x0 + size))
        b, m, d, k, w = (_f(s[q][sl]) for q in ("before", "skin", "delta", "keep", "weight"))
        if train:
            if rng.random() < 0.5:
                b, m, d, k, w = b[:, ::-1], m[:, ::-1], d[:, ::-1], k[:, ::-1], w[:, ::-1]
            g = float(rng.uniform(0.8, 1.2))           # exposure: the hand's change goes with it
            b, d = np.clip(b * g, 0, 1), d * g
        xs.append(np.concatenate([b, m[..., None]], -1).transpose(2, 0, 1))
        ys.append(np.concatenate([d, k[..., None]], -1).transpose(2, 0, 1))
        ws.append(w[None])
    return (np.ascontiguousarray(np.stack(xs), np.float32), np.ascontiguousarray(np.stack(ys), np.float32),
            np.ascontiguousarray(np.stack(ws), np.float32))


def _loss(pred, y, w):
    import torch
    n = w.sum().clamp(min=1.0)
    ld = (torch.abs(pred[:, :3] - y[:, :3]) * w).sum() / (3 * n)
    lk = (torch.abs(pred[:, 3:] - y[:, 3:]) * w).sum() / n
    return 20.0 * ld + 0.5 * lk


def train(samples: List[dict], val: List[int], steps: int = 4000, batch: int = 6, size: int = 384, lr: float = 1e-3,
          width: int = 32, device: Optional[str] = None, log: Callable = print, seed: int = 0):
    """samples: load_sample() dicts; val: indices held out. Returns (net, history)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = [i for i in range(len(samples)) if i not in set(val)]
    net = build_net(width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(2, steps), pct_start=0.1)
    hist = []
    best = (1e9, None)
    for it in range(steps):
        net.train()
        idx = rng.choice(tr, size=min(batch, len(tr)), replace=len(tr) < batch)
        x, y, w = (torch.from_numpy(a).to(dev) for a in _batch(samples, idx, rng, size, True))
        loss = _loss(net(x), y, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps // 20) == 0 or it == steps - 1:
            vl = evaluate(net, samples, val, dev)["net"]["loss"] if val else float(loss)
            hist.append(dict(step=it + 1, train=float(loss), val=vl))
            log(f"step {it + 1}/{steps} train {float(loss):.4f} val {vl:.4f}")
            if vl < best[0]:
                best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    return net, hist


def evaluate(net, samples: List[dict], idx: List[int], device: Optional[str] = None) -> dict:
    """On held-out samples: how far from the hand's retouch (luma, x1000) are no retouch, DIAP's current retouch and
    the network; and the keep error. Lower is closer to the hand."""
    import torch
    dev = device or next(net.parameters()).device
    net.eval()
    acc = {k: dict(err=[], keep=[], loss=[]) for k in ("none", "diap", "net")}
    with torch.no_grad():
        for i in idx:
            s = samples[i]
            x, y, w = _batch(samples, [i], None, s["before"].shape[0], False)
            p = net(torch.from_numpy(x).to(dev)).cpu().numpy()
            cands = {"none": np.concatenate([np.zeros_like(y[:, :3]), np.ones_like(y[:, 3:])], 1), "net": p}
            if "diap_delta" in s:
                cands["diap"] = np.concatenate([_f(s["diap_delta"]).transpose(2, 0, 1)[None],
                                                _f(s["diap_keep"])[None, None]], 1)
            ww = w[0, 0] > 0.5
            if ww.sum() < 50:
                continue
            for k, c in cands.items():
                r = cv2.GaussianBlur(((c[0, :3] - y[0, :3]).transpose(1, 2, 0) @ K).astype(np.float32), (0, 0), 1.0)
                acc[k]["err"].append(float(np.sqrt((r[ww] ** 2).mean())) * 1000)
                acc[k]["keep"].append(float(np.abs(c[0, 3] - y[0, 3])[ww].mean()))
                acc[k]["loss"].append(float(_loss(torch.from_numpy(c), torch.from_numpy(y), torch.from_numpy(w))))
    return {k: dict(err=float(np.mean(v["err"])) if v["err"] else None, keep=float(np.mean(v["keep"])) if v["keep"] else None,
                    loss=float(np.mean(v["loss"])) if v["loss"] else float("nan"), n=len(v["err"]))
            for k, v in acc.items()}


def export_onnx(net, path) -> None:
    import torch
    net = net.cpu().eval()
    x = torch.zeros(1, 4, 256, 256)
    kw = dict(input_names=["x"], output_names=["y"], opset_version=17,
              dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False            # the TorchScript exporter: the new one needs onnxscript, not in the app (j2742)
    torch.onnx.export(net, x, str(path), **kw)
