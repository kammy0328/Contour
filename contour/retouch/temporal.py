"""The retouch held steady through time (1.27.1, LNGSHOT BEEP BEEP 11/27).

The engine retouches every frame on its own. Where something on one frame only fools a stage (the hard shadow of a hand crossing a chin for one frame
of a whip move: the chin's stages took its edge for pebbles, c087 f32: +14 / -12 levels; a fold stage switched on for one frame, c056 f21; a stage that
goes by the face's turn going off on one frame), the retouch has a one-frame change that the eye sees as a blink.

Here a frame's retouch (what the engine changed: out - src) is compared with the retouch of the frames before and after it, carried onto this frame by
the optical flow of the source pictures, and each pixel takes the middle one of the three (a median). What all three frames do stays; what one frame
does alone goes; a stage that is on for one frame only is taken back to what its neighbours do. Where the flow cannot be trusted (forward and backward
flow disagree: something comes out from behind something else, a fast blur) the neighbour counts as this frame's own retouch, so nothing is carried there.

Only the retouch is touched, never the picture's own content: outside the places the engine changed on these three frames the frame is written as it
came. The first and the last frame of a render have one neighbour and are written as they came. STEADY = False: off."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

STEADY = True
STEADY_FLOW_W = 960          # the flow is made on the source's luma at most this wide (the retouched region's box)
STEADY_FB = 1.0              # px of that flow picture: forward and backward flow disagreeing by more, the neighbour is not used there
STEADY_EPS = 0.25 / 1023.0   # of the plane's range: a change smaller is no change (where the engine changed nothing)
STEADY_PAD = 24              # px of the source round the changed region


def _luma(planes, maxv: float) -> np.ndarray:
    return np.asarray(planes[0], np.float32) / float(maxv if maxv else 1.0)


def _box(mask: np.ndarray, pad: int) -> Optional[Tuple[int, int, int, int]]:
    ys = np.flatnonzero(mask.any(1))
    if ys.size == 0:
        return None
    xs = np.flatnonzero(mask.any(0))
    h, w = mask.shape
    return max(0, ys[0] - pad), min(h, ys[-1] + 1 + pad), max(0, xs[0] - pad), min(w, xs[-1] + 1 + pad)


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


_dis = None


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """the flow from a to b (where each pixel of a is found in b), both uint8"""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(a, b, None)


def _pull(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """img (of the other frame) sampled where flow says each pixel of this frame lies there"""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _trust(fwd: np.ndarray, bwd: np.ndarray, thr: float) -> np.ndarray:
    """1 where the backward flow, taken where the forward flow points, brings the pixel back (within thr px)"""
    back = _pull(bwd, fwd)
    err = np.hypot(fwd[..., 0] + back[..., 0], fwd[..., 1] + back[..., 1])
    return (err < thr).astype(np.float32)


def steady(src: List[List[np.ndarray]], out: List[List[np.ndarray]], subs, maxv: float) -> List[np.ndarray]:
    """The middle frame of three (src, out: [before, this, after], each a list of planes) with its retouch made the median of its own and the
    neighbours' (carried by the flow). Returns the planes to write for it."""
    S1, O1 = src[1], out[1]
    if not STEADY:
        return O1
    eps = STEADY_EPS * float(maxv if maxv else 1.0)
    # where any of the three frames was changed (luma plane, full size)
    ch = np.zeros(np.asarray(S1[0]).shape, bool)
    for k in range(3):
        ch |= np.abs(np.asarray(out[k][0], np.float32) - np.asarray(src[k][0], np.float32)) > eps
    bx = _box(ch, STEADY_PAD)
    if bx is None:
        return O1
    y0, y1, x0, x1 = bx
    H, W = ch.shape
    f = min(1.0, STEADY_FLOW_W / float(max(1, x1 - x0)))
    hw = (max(8, int(round((x1 - x0) * f))), max(8, int(round((y1 - y0) * f))))
    L = [cv2.resize(_luma(src[k], maxv)[y0:y1, x0:x1], hw, interpolation=cv2.INTER_AREA) for k in range(3)]
    L = [_u8(cv2.GaussianBlur(x, (0, 0), 0.7)) for x in L]
    pulls = []                                       # per neighbour: (flow at full scale of the box, trust)
    for k in (0, 2):
        fw = _flow(L[1], L[k])
        bw = _flow(L[k], L[1])
        tr = _trust(fw, bw, STEADY_FB)
        tr = cv2.erode(tr, np.ones((3, 3), np.uint8))
        pulls.append((fw, tr))
    res = []
    for i, (P1, Q1) in enumerate(zip(S1, O1)):
        P1 = np.asarray(P1)
        sx, sy = subs[i] if subs and i < len(subs) else (1, 1)
        py0, py1, px0, px1 = y0 // sy, -(-y1 // sy), x0 // sx, -(-x1 // sx)
        pw, ph = px1 - px0, py1 - py0
        d1 = np.asarray(Q1, np.float32)[py0:py1, px0:px1] - P1[py0:py1, px0:px1].astype(np.float32)
        ds = [None, d1, None]
        for j, k in enumerate((0, 2)):
            fw, tr = pulls[j]
            fl = cv2.resize(fw, (pw, ph), interpolation=cv2.INTER_LINEAR)
            fl[..., 0] *= pw / float(fw.shape[1])
            fl[..., 1] *= ph / float(fw.shape[0])
            trp = cv2.resize(tr, (pw, ph), interpolation=cv2.INTER_NEAREST)
            dk = np.asarray(out[k][i], np.float32)[py0:py1, px0:px1] - np.asarray(src[k][i], np.float32)[py0:py1, px0:px1]
            dk = _pull(dk, fl)
            ds[k] = trp * dk + (1.0 - trp) * d1
        a, b, c = ds
        med = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
        o = np.array(Q1, copy=True)
        v = P1[py0:py1, px0:px1].astype(np.float32) + med
        if np.issubdtype(o.dtype, np.integer):
            v = np.clip(np.round(v), 0, float(maxv))
        o[py0:py1, px0:px1] = v.astype(o.dtype)
        res.append(o)
    return res


class Steadier:
    """Frames in order -> the same frames out, each (but the first and the last) with steady() laid on it. push(t, src, out) gives back the frames
    ready to write [(t, planes)], flush() the rest."""

    def __init__(self, subs, maxv: float):
        self.subs, self.maxv = subs, maxv
        self.buf = []                                # [(t, src, out)]
        self.first = True

    def push(self, t: int, src, out) -> List[Tuple[int, list]]:
        self.buf.append((t, src, out))
        ready = []
        if self.first:
            self.first = False
            ready.append((t, out))                   # the first frame: as it came (kept for the next one's neighbour)
            return ready
        if len(self.buf) >= 3:
            (ta, sa, oa), (tb, sb, ob), (tc, sc, oc) = self.buf[-3:]
            if tb == ta + 1 and tc == tb + 1:
                try:
                    ready.append((tb, steady([sa, sb, sc], [oa, ob, oc], self.subs, self.maxv)))
                except Exception:
                    ready.append((tb, ob))
            else:
                ready.append((tb, ob))
            self.buf = self.buf[-2:]
        return ready

    def flush(self) -> List[Tuple[int, list]]:
        if len(self.buf) >= 2:
            t, s, o = self.buf[-1]
            self.buf = []
            return [(t, o)]
        self.buf = []
        return []
