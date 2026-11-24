"""Retouch for footage that is not display-referred (S-Log3, ARRI LogC3/LogC4, ...).

The calculation is on the original. The Rec.709 view (the same OCIO transform as the viewer and the analysis proxy)
is only the place where the retouch looks for what to hold and where: dark spots, moles, paint, strands, folds, colour
keys. Those thresholds were fitted on a display picture and mean nothing on log code values (log compresses the
shadows). So, for such footage:

  1. The skin texture, the main retouch, is thinned on the original luma plane (it scales bands of the picture, so it
     needs no display values). The masks and holds it uses come from the view.
  2. The view follows that change through the local slope of the display transform, so the later steps (folds, creases,
     colour work) find on it what they would find on a display-referred file.
  3. What those later steps change is a change of the view. It is carried back into the original through the local
     Jacobian of the display transform (Newton, bounded; only an improvement is kept).

Pixels the retouch did not touch keep their exact code values. Chroma that is subsampled in the file receives the
area average of the change.

"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ..core.colour import PASSTHROUGH
from . import engine as RE

_KR, _KB = 0.2126, 0.0722
_EPS = 2.0e-3          # finite difference step in the source encoding (code value units)
_THRESH = 1.0e-6       # display delta below this counts as untouched
_EPS_L = 2.0e-3        # luma step (source code values) for the view's slope
_STEPS = 3             # Newton iterations
_STEP_MAX = 0.05       # per step and channel, source code value units
_TOTAL_MAX = 0.12      # total change allowed at one pixel
CROP = True            # transform only the area the retouch can touch (union of the masks plus a margin)
CROP_MARGIN = 0.8      # of the face width


def wanted(cm, vs) -> bool:
    """Only for scene-referred encodings (log, linear). Display-referred footage keeps the plain code-value path."""
    if not (cm is not None and getattr(cm, "available", False) and vs is not None
            and vs.input_cs and vs.input_cs != PASSTHROUGH):
        return False
    try:
        import PyOpenColorIO as O
        cs = cm.config.getColorSpace(vs.input_cs)
        return cs is not None and cs.getReferenceSpaceType() == O.REFERENCE_SPACE_SCENE
    except Exception:
        return True


def _eval(cm, vs, pts: np.ndarray) -> np.ndarray:
    """pts: N x 3 float32 in the source encoding -> N x 3 display values (unclipped)."""
    return cm.to_display(pts.reshape(-1, 1, 3), vs).reshape(-1, 3)


def display_delta_to_source(cm, vs, src: np.ndarray, disp0: np.ndarray, disp1: np.ndarray) -> np.ndarray:
    """H x W x 3 deltas. src/disp0/disp1 are H x W x 3 float32. Returns the source-encoding delta (zeros where untouched)."""
    d = disp1 - disp0
    ch = np.max(np.abs(d), axis=-1) > _THRESH
    out = np.zeros_like(src)
    if not ch.any():
        return out
    ys, xs = np.nonzero(ch)
    p = src[ys, xs].astype(np.float64)
    target = disp1[ys, xs].astype(np.float64)
    cur = p.copy()
    best = p.copy()
    f0 = _eval(cm, vs, p.astype(np.float32)).astype(np.float64)
    best_err = np.linalg.norm(target - f0, axis=1)       # doing nothing leaves the whole requested change unmet
    for _ in range(_STEPS):
        f0 = _eval(cm, vs, cur.astype(np.float32)).astype(np.float64)
        J = np.empty((cur.shape[0], 3, 3))
        for k in range(3):
            q = cur.copy()
            q[:, k] += _EPS
            J[:, :, k] = (_eval(cm, vs, q.astype(np.float32)).astype(np.float64) - f0) / _EPS
        r = target - f0
        # damped least squares; where the display clips or compresses hard the full change cannot be reached and the
        # step would run away, so the step is bounded (trust region) and only an improvement is kept
        JT = np.transpose(J, (0, 2, 1))
        A = JT @ J
        lam = 1e-3 * np.trace(A, axis1=1, axis2=2)[:, None, None] + 1e-9
        A = A + lam * np.eye(3)[None]
        step = np.linalg.solve(A, (JT @ r[:, :, None]))[:, :, 0]
        step = np.clip(step, -_STEP_MAX, _STEP_MAX)
        nxt = np.clip(cur + step, p - _TOTAL_MAX, p + _TOTAL_MAX)
        err = np.linalg.norm(target - _eval(cm, vs, nxt.astype(np.float32)).astype(np.float64), axis=1)
        better = err < best_err
        best[better] = nxt[better]
        best_err = np.where(better, err, best_err)
        cur = nxt
    cur = best
    out[ys, xs] = (cur - p).astype(np.float32)
    return out


def _box(rf, W: int, H: int):
    """Source-pixel box around everything the retouch of this frame can change, or None (use the whole frame)."""
    try:
        u = rf.union()
        if u is None:
            return None
        ph, pw = u.shape[:2]
        ys, xs = np.nonzero(u > 1e-4)
        if not len(xs):
            return None
        sx, sy = W / pw, H / ph
        wf = max((j.wf_src for j in rf.jobs if j.wf_src > 0), default=0.11 * max(W, H))
        m = CROP_MARGIN * wf
        x0, x1 = int(max(0, xs.min() * sx - m)), int(min(W, (xs.max() + 1) * sx + m))
        y0, y1 = int(max(0, ys.min() * sy - m)), int(min(H, (ys.max() + 1) * sy + m))
        if (x1 - x0) * (y1 - y0) > 0.7 * W * H or x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    except Exception:
        return None


def _to_ycc(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    kg = 1.0 - _KR - _KB
    Y = _KR * r + kg * g + _KB * b
    return Y, (b - Y) / (2 * (1 - _KB)), (r - Y) / (2 * (1 - _KR))


def _from_ycc(Y, Cb, Cr):
    kg = 1.0 - _KR - _KB
    r = Y + 2 * (1 - _KR) * Cr
    b = Y + 2 * (1 - _KB) * Cb
    g = (Y - _KR * r - _KB * b) / kg
    return np.stack([r, g, b], -1)


def _yuv_to_rgb(planes, max_value: float, rng: str) -> np.ndarray:
    y, u, v = (p.astype(np.float32) for p in planes[:3])
    h, w = y.shape
    if u.shape != y.shape:
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    s = (max_value + 1) / 256.0
    if rng == "tv":
        yn, un, vn = (y - 16 * s) / (219 * s), (u - 128 * s) / (224 * s), (v - 128 * s) / (224 * s)
    else:
        yn, un, vn = y / max_value, (u - 128 * s) / max_value, (v - 128 * s) / max_value
    return _from_ycc(yn, un, vn).astype(np.float32)


def _dbg(*a):
    import os as _o
    if _o.environ.get("IWALY_ROI_DEBUG"):
        print("APPLY_ROI:", *a, flush=True)


def _apply_roi(planes, subsample, max_value, pix_fmt, rf, y_per_src, cm, vs, rng):
    """engine.APPLY_ROI: each person on their own piece of the picture, one after the other (a later person's piece is read from
    the planes the earlier ones left). None when it does not apply (a body effect, no face): the whole frame then."""
    if not pix_fmt.startswith("yuv") or len(planes) < 3:
        _dbg("not yuv planes", pix_fmt)
        return None
    H, W = planes[0].shape[:2]
    bad = [(j.pid, RE.roi_job_ok(j, why=True)) for j in rf.jobs]
    bad = [b for b in bad if b[1] is not True]
    if bad:
        _dbg("jobs not on their piece", bad)
        return None
    al = 2
    for pl in planes[1:3]:
        if W % pl.shape[1] or H % pl.shape[0]:
            return None
        al = max(al, W // pl.shape[1], H // pl.shape[0])
    boxes = []
    for j in rf.jobs:
        w_ = []
        b = RE.roi_job_box(j, (W, H), rf.proxy_wh, y_per_src, float(getattr(RE, "APPLY_ROI_MARGIN", 1.5)), al, w_)
        _dbg("job", j.pid, "masks' boxes (proxy px)", w_)
        if b is None:
            _dbg("no box", j.pid)
            return None
        boxes.append(b)
    if sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) > 0.8 * W * H:
        _dbg("pieces cover most of the picture", boxes)
        return None                      # the pieces are most of the picture: once for all is cheaper
    _dbg("on pieces", boxes)
    out = [np.array(p, copy=True) for p in planes[:3]]
    for j, (x0, y0, x1, y1) in zip(rf.jobs, boxes):
        rfj = RE.roi_frame(rf, j, (x0, y0, x1, y1), (W, H), y_per_src)
        sub = []
        for pl in out:
            kx, ky = W // pl.shape[1], H // pl.shape[0]
            sub.append(pl[y0 // ky:y1 // ky, x0 // kx:x1 // kx])
        res = apply_planes_display(sub, subsample, max_value, pix_fmt, rfj, y_per_src, cm, vs, rng, _roi=False)
        for pl, r in zip(out, res[:3]):
            kx, ky = W // pl.shape[1], H // pl.shape[0]
            pl[y0 // ky:y1 // ky, x0 // kx:x1 // kx] = r
        j.naso_pres = rfj.jobs[0].naso_pres
    return out + list(planes[3:])


def apply_planes_display(planes: List[np.ndarray], subsample, max_value: float, pix_fmt: str, rf,
                         y_per_src: float, cm, vs, rng: str = "tv", _roi: bool = True) -> List[np.ndarray]:
    """Same contract as engine.apply_planes, retouching in the display picture of `vs`."""
    if rf is None or rf.empty():
        return planes
    if _roi and float(getattr(RE, "APPLY_ROI", 0) or 0) > 0:
        try:
            r_ = _apply_roi(planes, subsample, max_value, pix_fmt, rf, y_per_src, cm, vs, rng)
        except Exception:
            import traceback
            traceback.print_exc()
            r_ = None
        if r_ is not None:
            return r_
    if pix_fmt.startswith("yuv"):
        src = _yuv_to_rgb(planes, max_value, rng)
    else:
        g, b, r = planes[0], planes[1], planes[2]
        src = np.stack([r, g, b], -1).astype(np.float32)
    Hs, Ws = src.shape[:2]
    box = _box(rf, Ws, Hs) if CROP else None
    if box is None:
        disp0 = cm.to_display(src, vs)
    else:
        x0, y0, x1, y1 = box
        disp0 = np.empty_like(src)
        crop_d = cm.to_display(src[y0:y1, x0:x1], vs)
        disp0[:] = crop_d.reshape(-1, 3).mean(0)          # outside the box the retouch does not look
        disp0[y0:y1, x0:x1] = crop_d
    Y, Cb, Cr = _to_ycc(disp0)
    # the original luma and what a unit change of it does to the view (a change of luma at fixed colour = the same
    # amount on R, G and B)
    Ys = (_KR * src[..., 0] + (1.0 - _KR - _KB) * src[..., 1] + _KB * src[..., 2]).astype(np.float32)
    if box is None:
        up = cm.to_display(src + _EPS_L, vs)
    else:
        up = np.zeros_like(disp0)
        up[y0:y1, x0:x1] = cm.to_display(src[y0:y1, x0:x1] + _EPS_L, vs)
    Jv = (up - disp0) / _EPS_L
    Jy, Jcb, Jcr = _to_ycc(Jv)
    sd = {"Y": Ys, "dY": np.zeros_like(Ys), "Jy": Jy.astype(np.float32), "Jcb": Jcb.astype(np.float32),
          "Jcr": Jcr.astype(np.float32)}
    Y, Cb, Cr = RE.process_ycc(Y.copy(), Cb.copy(), Cr.copy(), rf, y_per_src, src=sd)
    disp1 = _from_ycc(Y, Cb, Cr).astype(np.float32)
    # 1: the texture, already in the original's units
    dS = sd["dY"]
    src2 = src + dS[..., None]
    # 3: whatever else the later steps changed in the view = the view they ended with, less the view the texture alone
    # predicts (so a pixel only the texture touched gets nothing carried back, and keeps exactly the texture's change)
    rest = disp1 - (disp0 + Jv * dS[..., None])
    if box is None:
        dsp2 = cm.to_display(src2, vs) if dS.any() else disp0
        dlog = display_delta_to_source(cm, vs, src2, dsp2, dsp2 + rest)
    else:
        dlog = np.zeros_like(src)
        d2 = cm.to_display(src2[y0:y1, x0:x1], vs) if dS.any() else disp0[y0:y1, x0:x1]
        dlog[y0:y1, x0:x1] = display_delta_to_source(cm, vs, src2[y0:y1, x0:x1], d2, d2 + rest[y0:y1, x0:x1])
    dlog = dlog + sd["dY"][..., None]
    if not dlog.any():
        return planes
    if pix_fmt.startswith("yuv"):
        dY, dCb, dCr = _to_ycc(dlog)
        s = (max_value + 1) / 256.0
        if rng == "tv":
            kY, kC = 219 * s, 224 * s
        else:
            kY = kC = max_value
        h, w = dY.shape
        deltas = [dY * kY, dCb * kC, dCr * kC]
        out = []
        for pl, dl in zip(planes[:3], deltas):
            if dl.shape != pl.shape:
                dl = cv2.resize(dl, (pl.shape[1], pl.shape[0]), interpolation=cv2.INTER_AREA)
            if np.issubdtype(pl.dtype, np.integer):
                out.append(np.clip(np.rint(pl.astype(np.float32) + dl), 0, max_value).astype(pl.dtype))
            else:
                out.append((pl + dl).astype(pl.dtype))
        return out + list(planes[3:])
    o = src + dlog
    return [o[..., 1].astype(planes[0].dtype), o[..., 2].astype(planes[1].dtype), o[..., 0].astype(planes[2].dtype)] \
        + list(planes[3:])
