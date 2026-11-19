"""Fine-tuning the whole-body pose model (RTMW-x) on frames made blurred, compressed, dark or flashing
(2026-11-19, step 1 of 'both': the joint refiner cannot learn these, j2770c gate 0).

Data (made from analysed projects, nothing in them changed): sharp frames the analysis was sure about, one person
crop each, the way the model sees it (PoseEstimator: box padded 1.25, 288x384), with the solved body joints as
the answer. The degradations are made fresh at every training step from the clean crop, so the model never sees
the same blur twice; the clean crop also goes through the model as it is now (frozen), and its whole-body output
(face, hands, feet) is what the trained model is held to on the degraded crop, so those points are not forgotten.

Nothing here touches the app: the trained model is exported to ONNX beside the training data and judged against
the one in use on held-out cuts (another camera roll of the same MV, and a different MV). Switching the app's
model is a separate, approved step."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

SIZE = (288, 384)              # model input (w, h)
PAD = 1.25
RATIO = 2.0                    # simcc bins per input pixel
K_BODY = 17
K_ALL = 133
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
KINDS = ("clean", "motion", "defocus", "compress", "dark", "colour", "flash", "mixed")
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


# ---------------------------------------------------------------------- crops
def crop_matrix(box) -> np.ndarray:
    """The affine the pose model's own input uses for this box (infer.PoseEstimator._prep, no rotation)."""
    from ..track.infer import _warp_matrix
    x1, y1, x2, y2 = [float(v) for v in box]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale = np.array([(x2 - x1) * PAD, (y2 - y1) * PAD], np.float32)
    w, h = SIZE
    ar = w / h
    if scale[0] > scale[1] * ar:
        scale = np.array([scale[0], scale[0] / ar], np.float32)
    else:
        scale = np.array([scale[1] * ar, scale[1]], np.float32)
    return _warp_matrix(center, scale, (w, h), 0.0)


def crop(rgb: np.ndarray, box) -> Tuple[np.ndarray, np.ndarray]:
    m = crop_matrix(box)
    return cv2.warpAffine(rgb, m, SIZE, flags=cv2.INTER_LINEAR), m


def to_crop(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float32)
    return p @ m[:, :2].T + m[:, 2]


def normalise(img: np.ndarray) -> np.ndarray:
    return ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)


# ---------------------------------------------------------------------- degradations (crop space, 384 px tall)
def degrade(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """One degradation of a clean crop, with random strength."""
    if kind == "clean":
        return img
    if kind == "mixed":
        out = img
        for k in rng.choice(["motion", "defocus", "compress", "dark", "colour"], size=2, replace=False):
            out = degrade(out, str(k), rng)
        return out
    h = img.shape[0]
    if kind == "motion":
        L = int(rng.uniform(0.02, 0.09) * h) | 1
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        k = cv2.warpAffine(k, M, (L, L))
        return cv2.filter2D(img, -1, k / max(1e-6, k.sum()))
    if kind == "defocus":
        return cv2.GaussianBlur(img, (0, 0), float(rng.uniform(1.5, 6.0)))
    if kind == "compress":
        s = float(rng.uniform(0.35, 0.8))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 35))])
        out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if kind == "dark":
        g = float(rng.uniform(1.6, 3.0))
        x = (img.astype(np.float32) / 255.0) ** g * float(rng.uniform(0.35, 0.8))
        x = x + rng.normal(0, float(rng.uniform(0.005, 0.03)), x.shape)
        return np.clip(x * 255, 0, 255).astype(np.uint8)
    if kind == "colour":
        tint = rng.uniform(0.3, 1.3, 3).astype(np.float32)
        x = img.astype(np.float32) * tint[None, None]
        return np.clip(x, 0, 255).astype(np.uint8)
    if kind == "flash":
        x = img.astype(np.float32)
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]].astype(np.float32)
        cx, cy = rng.uniform(0, img.shape[1]), rng.uniform(0, img.shape[0])
        r = float(rng.uniform(0.3, 0.8)) * img.shape[0]
        glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
        x = x + glow * float(rng.uniform(80, 200))
        return np.clip(x, 0, 255).astype(np.uint8)
    raise ValueError(kind)


# ---------------------------------------------------------------------- SimCC targets, decoding, scoring
def simcc_targets(kp: np.ndarray, w: np.ndarray, sigma: float = 6.0):
    """Gaussian bin targets for points in crop px: (K, 576), (K, 768), weights (K,)."""
    nx, ny = int(SIZE[0] * RATIO), int(SIZE[1] * RATIO)
    bx = np.arange(nx, dtype=np.float32)[None]
    by = np.arange(ny, dtype=np.float32)[None]
    x = kp[:, :1] * RATIO
    y = kp[:, 1:2] * RATIO
    tx = np.exp(-((bx - x) ** 2) / (2 * sigma ** 2))
    ty = np.exp(-((by - y) ** 2) / (2 * sigma ** 2))
    inside = (kp[:, 0] >= 0) & (kp[:, 0] < SIZE[0]) & (kp[:, 1] >= 0) & (kp[:, 1] < SIZE[1])
    ok = np.isfinite(kp).all(1) & inside & (np.asarray(w) > 0)
    tx = np.where(ok[:, None], np.nan_to_num(tx), 0).astype(np.float32)
    ty = np.where(ok[:, None], np.nan_to_num(ty), 0).astype(np.float32)
    return tx, ty, ok.astype(np.float32)


def decode(sx: np.ndarray, sy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(N,K,bins) -> points in crop px (N,K,2) and scores (N,K): argmax as the app does (no sub-bin step)."""
    x = sx.argmax(-1).astype(np.float32) / RATIO
    y = sy.argmax(-1).astype(np.float32) / RATIO
    return np.stack([x, y], -1), 0.5 * (sx.max(-1) + sy.max(-1))


def score(pred: np.ndarray, gt: np.ndarray, ok: np.ndarray, height: np.ndarray) -> Dict[str, float]:
    """Body joints: pck@0.05 / 0.1 of the person's height, mean error, and left/right swaps (a pair closer to
    each other's answer)."""
    e = np.linalg.norm(pred - gt, axis=-1) / height[:, None]
    m = ok.astype(bool)
    swaps = n = 0
    for a, b in LR_PAIRS:
        both = m[:, a] & m[:, b]
        if not both.any():
            continue
        d_same = np.linalg.norm(pred[:, a] - gt[:, a], axis=-1) + np.linalg.norm(pred[:, b] - gt[:, b], axis=-1)
        d_swap = np.linalg.norm(pred[:, a] - gt[:, b], axis=-1) + np.linalg.norm(pred[:, b] - gt[:, a], axis=-1)
        gap = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1) / height
        sel = both & (gap > 0.04)
        swaps += int((sel & (d_swap < 0.6 * d_same)).sum())
        n += int(sel.sum())
    ev = e[m]
    return dict(n=int(m.sum()), pck05=float((ev < 0.05).mean()) if ev.size else float("nan"),
                pck10=float((ev < 0.10).mean()) if ev.size else float("nan"),
                err=float(ev.mean()) if ev.size else float("nan"), swap=swaps / max(1, n))


# ---------------------------------------------------------------------- torch side (training venv)
def kl_loss(px, py, tx, ty, w, beta: float = 10.0):
    """KL between the model's bins and the targets (mmpose KLDiscretLoss with label softmax), per point weight."""
    import torch
    import torch.nn.functional as F

    def one(p, t):
        lp = F.log_softmax(p * beta, -1)
        lt = F.softmax(t * beta, -1)
        return (lt * (torch.log(lt + 1e-12) - lp)).sum(-1)
    lx, ly = one(px, tx), one(py, ty)
    return ((lx + ly) * w).sum() / w.sum().clamp(min=1.0)


def teacher_targets(tx_logits, ty_logits):
    """The frozen model's own bins on the clean crop, as targets (already on the logit scale)."""
    return tx_logits.detach(), ty_logits.detach()


def fix_onnx_clip(model) -> int:
    """Clip nodes with an empty min/max input (allowed by ONNX, not by onnx2torch): the bound as a constant."""
    from onnx import numpy_helper
    fixed = 0
    for n in model.graph.node:
        if n.op_type != "Clip":
            continue
        ins = list(n.input) + [""] * (3 - len(n.input))
        for i, v in ((1, -3.4e38), (2, 3.4e38)):
            if ins[i] == "":
                nm = f"_clip_bound_{fixed}_{i}"
                model.graph.initializer.append(numpy_helper.from_array(np.array(v, np.float32), nm))
                ins[i] = nm
                fixed += 1
        del n.input[:]
        n.input.extend(ins)
    return fixed
