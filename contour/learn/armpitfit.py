"""The armpit fold retouch learned from a hand-retouched shot (before / after pairs of the same frames).

What the retoucher did to the armpit's folds (c006: measured on 8 frames) is a blur laid on at an opacity inside the
armpit: its change follows a Gaussian blur of 0.027..0.04 face widths with a correlation of 0.92..0.99 above the grain,
at an opacity of 0.6..1.2 (median 0.84). fit_blur finds the blur width (in face widths, so that it carries to other faces
and resolutions) and the opacity from before / after pairs, over the core of the area the hand retouched (strong single
changes such as a finger the blur ran over left out). Grain is never learned: the change is low-passed by GRAIN_REL
before it is laid on, so the plate keeps its own grain (the hand's blur took it away).

The model is a small JSON: {"kind": "blur", "sigma_rel", "opacity", "grain_rel", "pairs": [...]}. The engine
(ARMPIT_LEARNED) lays it on inside the armpit stage's own mask.

2611 IWALY c006 (user 2026-11-25: "IWALY_c006_BTv2.mov에서 겨드랑이 주름을 수동으로 보정했는데 겨드랑이 주름 보정 학습해")."""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

K709 = np.array([0.2126, 0.7152, 0.0722], np.float32)
GRAIN_REL = 0.006          # of the face width: finer than this is grain and texture at the plate's own scale (gain 1)
EDGES_REL = (0.006, 0.012, 0.024, 0.048, 0.096)   # band edges (Gaussian sigmas) in face widths; the last band runs to the broad shape (kept)
OUTLIER = 0.012            # a change of the hand's larger than this (0..1, blurred 2 px) is cleaning or a slip (a finger), not the fold retouch


def ycc(rgb: np.ndarray) -> np.ndarray:
    Y = rgb @ K709
    return np.stack([Y, (rgb[..., 2] - Y) / 1.8556, (rgb[..., 0] - Y) / 1.5748], -1).astype(np.float32)


def bands(x: np.ndarray, sig_px: Iterable[float]) -> List[np.ndarray]:
    """x split into detail bands: [x - G(s0), G(s0) - G(s1), ..., G(s_last)] (the last is the broad rest)."""
    L = [x] + [cv2.GaussianBlur(x, (0, 0), float(s)) for s in sig_px]
    return [L[i] - L[i + 1] for i in range(len(L) - 1)] + [L[-1]]


def hand_area(d: np.ndarray, thr: float = 0.0012, core: float = 0.6) -> np.ndarray:
    """Where the hand retouched (bool): its change soft-dense, eroded to the core, strong single changes left out."""
    act = cv2.GaussianBlur((np.abs(cv2.GaussianBlur(d, (0, 0), 1.0)) > thr).astype(np.float32), (0, 0), 8)
    m = act > core
    out = np.abs(cv2.GaussianBlur(d, (0, 0), 2)) > OUTLIER
    m &= ~cv2.dilate(out.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return m


def fit_blur(pairs, sigmas_rel=(0.013, 0.02, 0.027, 0.033, 0.04, 0.053), grain_rel: float = GRAIN_REL) -> Dict:
    """The hand's fold retouch as a blur laid on at an opacity: for every pair (before RGB, after RGB, face width px) the
    blur width (in face widths) whose change follows the hand's best above the grain, and the opacity (least squares)
    over the core of the area the hand retouched. The model is the median over the pairs."""
    rows = []
    for before, after, wf in pairs:
        b = ycc(before.astype(np.float32)); a = ycc(after.astype(np.float32))
        d = a[..., 0] - b[..., 0]
        area = hand_area(d, core=0.9)
        if area.sum() < 500:
            continue
        sg = max(0.5, grain_rel * wf)
        gl = lambda z: cv2.GaussianBlur(z, (0, 0), sg)
        y = gl(d)[area]
        best = None
        for sr in sigmas_rel:
            x = gl(cv2.GaussianBlur(b[..., 0], (0, 0), sr * wf) - b[..., 0])[area]
            den = float((x * x).sum())
            if den <= 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1]); al = float((x * y).sum() / den)
            if best is None or r > best[0]:
                best = (r, sr, al)
        if best is not None:
            rows.append(dict(sigma_rel=best[1], opacity=round(best[2], 3), corr=round(best[0], 3), px=int(area.sum())))
    if not rows:
        return {}
    return dict(kind="blur", sigma_rel=float(np.median([r["sigma_rel"] for r in rows])), opacity=float(np.clip(np.median([r["opacity"] for r in rows]), 0, 1)),
                grain_rel=grain_rel, pairs=rows)


def apply(Y: np.ndarray, m: np.ndarray, model: Dict, wf: float, k: float = 1.0, chroma: Optional[List[np.ndarray]] = None,
          m_c: Optional[np.ndarray] = None, wf_c: Optional[float] = None, support: bool = True):
    """The learned retouch laid on inside m (0..1) at strength k: the blur's change low-passed by the grain's width, so the
    plate's grain (and the hand's loss of it) is never taken; chroma planes at their own size with m_c / wf_c."""
    def one(P, mm, w):
        P = P.astype(np.float32)
        s = max(0.5, float(model["sigma_rel"]) * w)
        sg = max(0.5, float(model.get("grain_rel", GRAIN_REL)) * w)
        if support:
            # the blur reads the armpit's own skin only (the mask's support): the garment, the wall and a hand next to it stay out of it
            wt = np.clip(mm * 4.0, 0.0, 1.0).astype(np.float32)
            bl = cv2.GaussianBlur(P * wt, (0, 0), s) / np.maximum(cv2.GaussianBlur(wt, (0, 0), s), 1e-4)
            bl = np.where(wt > 1e-3, bl, P)
        else:
            bl = cv2.GaussianBlur(P, (0, 0), s)
        dP = cv2.GaussianBlur(bl - P, (0, 0), sg)
        return P + float(k) * float(model["opacity"]) * mm * dP
    Yo = one(Y, m, wf).astype(Y.dtype)
    if chroma is None:
        return Yo
    mc = m if m_c is None else m_c
    wc = wf if wf_c is None else wf_c
    return Yo, [one(C, mc, wc).astype(C.dtype) for C in chroma]


def save(path, model: Dict) -> None:
    json.dump(model, open(path, "w", encoding="utf-8"), indent=1)


def load(path) -> Dict:
    return json.load(open(path, encoding="utf-8"))
