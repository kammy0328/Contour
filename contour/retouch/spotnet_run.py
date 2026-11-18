"""The learned spot detector (contour/learn/spotnet.py, trained on LNGSHOT Bananas hand retouching) in the app.

Used for large faces only (SPOTNET_WF: face width in source pixels): on held-out shots it found 44 % of the hand's
spots against the rule detector's 10 % on faces of 500 px and more, at the same number of marks; on smaller faces
there were too few spots to tell. Everything else (the neck lines, the protection map, the fill) stays the rule
detector's. No model file, or any failure: the rule detector's maps stand."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ON = True
SPOTNET_WF = 500.0          # face width in source pixels from which the net finds the spots
NET_WF = 300.0              # the face width the net was trained at
THR = 0.7                   # the net's probability that marks as much as the rule detector's 0.5 (j2750)
KINDS = ("dark", "red", "bump")
EDGE = 0.03                 # of the face width: nothing this close to the skin's border (as the training labels)
SPOT_MAX = 0.08             # of the face width: the largest spot (as the training labels)
MODEL = Path(__file__).resolve().parent.parent / "track" / "data" / "spotnet_v1.onnx"

_sess = None
_failed = False


def available() -> bool:
    return ON and not _failed and MODEL.is_file()


def _session():
    global _sess, _failed
    if _sess is None and not _failed:
        try:
            from ..track import models as MD
            _sess = MD.session(MODEL)
        except Exception as e:
            sys.stderr.write(f"SPOTNET OFF: {str(e)[:200]}\n")
            _failed = True
    return _sess


def to_score(p: np.ndarray) -> np.ndarray:
    """The net's probability on the rule detector's scale: THR becomes 0.5 (the same marks), 0 and 1 stay."""
    p = np.clip(p, 0, 1).astype(np.float32)
    return np.where(p < THR, p * (0.5 / THR), 0.5 + (p - THR) * (0.5 / (1 - THR))).astype(np.float32)


def run(rgb: np.ndarray, region: np.ndarray, wf: float, sess=None,
        max_size: Optional[float] = None) -> Optional[Dict[str, np.ndarray]]:
    """dark, red, bump scores (0..1, rule scale) on this picture. rgb float 0..1 (H,W,3); region: where skin is
    (face skin and neck); wf: face width in this picture's pixels; max_size: the largest spot's diameter in this
    picture's pixels (the 잡티 크기 slider, as the rule detector and the fill get it; None: the model's own)."""
    sess = sess or _session()
    if sess is None or wf < 8:
        return None
    H, W = region.shape
    f = min(1.0, NET_WF / float(wf))
    w_, h_ = max(8, int(round(W * f))), max(8, int(round(H * f)))
    inter = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    x = np.concatenate([cv2.resize(np.clip(rgb, 0, 1).astype(np.float32), (w_, h_), interpolation=inter),
                        cv2.resize(region.astype(np.float32), (w_, h_), interpolation=inter)[..., None]], -1)
    ph, pw = (-h_) % 4, (-w_) % 4
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")
    inp = sess.get_inputs()[0].name
    lg = sess.run(None, {inp: np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0, :, :h_, :w_]
    p = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))
    # where the net was taught: inside the skin, away from its border (the hairline, the lips, the collar: no labels
    # were made there, and there it marked the border itself)
    r = max(1, int(round(EDGE * wf)))
    inner = cv2.erode((region > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    out = {}
    for i, k in enumerate(KINDS):
        m = cv2.resize(to_score(p[i]), (W, H), interpolation=cv2.INTER_LINEAR) * inner
        out[k] = _spots_only(m.astype(np.float32), wf, max_size)
    return out


def _spots_only(m: np.ndarray, wf: float, max_size: Optional[float] = None) -> np.ndarray:
    """Marks that are spots: compact, no larger than SPOT_MAX of the face width (the largest the model was taught;
    a line or a long edge is not a spot), and no larger than the user's largest spot (max_size, diameter)."""
    on = (m > 0.25).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
    if n <= 1:
        return m
    own = max(3.0, SPOT_MAX * wf)
    big = own if max_size is None else min(own, max(2.0, float(max_size)))
    bad = np.zeros(n, bool)
    for j in range(1, n):
        w_, h_, a_ = st[j, 2], st[j, 3], st[j, 4]
        if max(w_, h_) > big or a_ < 0.3 * w_ * h_ and max(w_, h_) > 0.25 * own:
            bad[j] = True
    return np.where(bad[cc], 0.0, m).astype(np.float32)
