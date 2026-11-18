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


def run(rgb: np.ndarray, region: np.ndarray, wf: float, sess=None) -> Optional[Dict[str, np.ndarray]]:
    """dark, red, bump scores (0..1, rule scale) on this picture. rgb float 0..1 (H,W,3); region: where skin is
    (face skin and neck); wf: face width in this picture's pixels."""
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
    out = {}
    for i, k in enumerate(KINDS):
        m = cv2.resize(to_score(p[i]), (W, H), interpolation=cv2.INTER_LINEAR)
        out[k] = (m * (region > 0.5)).astype(np.float32)
    return out
