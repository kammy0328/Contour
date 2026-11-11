"""Separation mode: warp the person layer only and rebuild the background it uncovers.

The uncovered strip is filled from neighbouring frames (aligned with the camera homographies, using pixels
where nobody stands) and only what is never visible falls back to spatial inpainting.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .field import FieldResult, warp_planes

FILL_ORDER = [-2, 2, -4, 4, -7, 7, -11, 11, -16, 16, -24, 24]


def _plane_mask(mask_full: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask_full.shape == shape:
        return mask_full
    return cv2.resize(mask_full.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _resize_alpha(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if a.shape == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fill_background(planes: List[np.ndarray], subsample, hole: np.ndarray, t: int,
                    neighbour: Callable[[int], Optional[Tuple[List[np.ndarray], np.ndarray]]],
                    homography: Callable[[int, int], np.ndarray], scale_xy: Tuple[float, float],
                    max_value: float) -> Tuple[List[np.ndarray], int]:
    """Fill `hole` (bool, luma res). neighbour(k) -> (planes, people_alpha float luma res) or None.

    homography(src_frame, dst_frame) is in source px; scale_xy converts source px to luma px.
    Returns (filled planes, number of luma pixels that needed spatial inpainting)."""
    out = [np.array(p, copy=True) for p in planes]
    if not hole.any():
        return out, 0
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    pad = 8
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0 -= x0 % 2
    y0 -= y0 % 2
    need = hole[y0:y1, x0:x1].copy()
    S_ = np.diag([scale_xy[0], scale_xy[1], 1.0])
    T_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    for k in FILL_ORDER:
        if not need.any():
            break
        nb = neighbour(t + k)
        if nb is None:
            continue
        nplanes, nalpha = nb
        Hs = homography(t + k, t)                       # source px, frame t+k -> t
        if Hs is None or not np.isfinite(Hs).all():     # other side of a picture jump
            continue
        Hl = T_roi @ S_ @ Hs @ np.linalg.inv(S_)        # luma px of t+k -> ROI px of t
        roi_w, roi_h = x1 - x0, y1 - y0
        wa = cv2.warpPerspective(nalpha.astype(np.float32), Hl, (roi_w, roi_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        valid = need & (wa < 0.04)
        if not valid.any():
            continue
        for i, (pl, (ssx, ssy)) in enumerate(zip(nplanes, subsample)):
            Sp = np.diag([1.0 / ssx, 1.0 / ssy, 1.0])
            Hp = Sp @ Hl @ np.linalg.inv(Sp)
            pw, ph = -(-roi_w // ssx), -(-roi_h // ssy)
            src = pl.astype(np.float32) if pl.dtype != np.float32 else pl
            wp = cv2.warpPerspective(src, Hp, (pw, ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            vm = _plane_mask(valid, (ph, pw)) if (ssx, ssy) != (1, 1) else valid
            cy0, cx0 = y0 // ssy, x0 // ssx
            tgt = out[i][cy0:cy0 + ph, cx0:cx0 + pw]
            vm = vm[:tgt.shape[0], :tgt.shape[1]]
            wp = wp[:tgt.shape[0], :tgt.shape[1]]
            if np.issubdtype(tgt.dtype, np.integer):
                wp = np.clip(np.rint(wp), 0, max_value)
            tgt[vm] = wp[vm].astype(tgt.dtype)
        need &= ~valid
    remaining = int(need.sum())
    if remaining:
        m8 = need.astype(np.uint8) * 255
        for i, (pl, (ssx, ssy)) in enumerate(zip(out, subsample)):
            cy0, cx0 = y0 // ssy, x0 // ssx
            ph, pw = -(-(y1 - y0) // ssy), -(-(x1 - x0) // ssx)
            region = pl[cy0:cy0 + ph, cx0:cx0 + pw]
            mm = cv2.resize(m8, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST) \
                if (ssx, ssy) != (1, 1) else m8[:region.shape[0], :region.shape[1]]
            if region.ndim == 3:
                chans = [region[..., c] for c in range(region.shape[2])]
            else:
                chans = [region]
            filled = []
            for ch in chans:
                src = ch if ch.dtype in (np.uint8, np.uint16, np.float32) else ch.astype(np.float32)
                filled.append(cv2.inpaint(np.ascontiguousarray(src), mm, 5, cv2.INPAINT_TELEA))
            res = np.dstack(filled) if region.ndim == 3 else filled[0]
            region[...] = res.astype(region.dtype)
    return out, remaining


def composite(planes: List[np.ndarray], subsample, fr: FieldResult, target_alpha: np.ndarray, t: int,
              neighbour, homography, scale_xy, max_value: float):
    """target_alpha: float32 luma-res alpha of the reshaped people (unwarped).

    Returns (output planes, warped target alpha, inpainted pixel count)."""
    warped = warp_planes(planes, subsample, fr, max_value)
    a_w = warp_planes([target_alpha.astype(np.float32)], [(1, 1)], fr, 1.0)[0]
    a_w = np.clip(a_w, 0, 1)
    # Rebuild the whole original silhouette (dilated) so spatial inpainting only ever sees background
    # pixels on its boundary; the warped person covers the inside again afterwards.
    k = max(5, int(round(min(target_alpha.shape) * 0.006)) | 1)
    hole = cv2.dilate((target_alpha > 0.03).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    bg, missing = fill_background(planes, subsample, hole, t, neighbour, homography, scale_xy, max_value)
    out = []
    for w_, b_, (ssx, ssy) in zip(warped, bg, subsample):
        a = _resize_alpha(a_w, w_.shape[:2]) if (ssx, ssy) != (1, 1) else a_w
        if w_.ndim == 3:
            a = a[..., None]
        o = w_.astype(np.float32) * a + b_.astype(np.float32) * (1 - a)
        if np.issubdtype(w_.dtype, np.integer):
            o = np.clip(np.rint(o), 0, max_value)
        out.append(o.astype(w_.dtype))
    return out, a_w, missing
