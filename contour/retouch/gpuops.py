"""GPU (torch / CUDA) versions of the image operations the retouch's mask making and apply spend their time in.

The engine, skin and detect modules call OpenCV through `cv2`; `proxy()` hands them an object that is OpenCV for everything
except the operations ported here, which run on the GPU when it pays (a big enough picture, a wide enough kernel) and fall back
to OpenCV otherwise (no CUDA, a dtype or border mode not ported, an error). The results match OpenCV's to float rounding:
the same Gaussian taps (cv2.getGaussianKernel), the same border (BORDER_REFLECT_101 = torch "reflect").

Switch: engine.GPU_MASKS (0 / 1; control.json engine_cfg can set it), or the environment IWALY_GPU_MASKS before import.
(2611 IWALY, 11/24: renders were 9 to 21 s a frame with the GPU 26 % busy; the mask making on the CPU was the time.)
"""
from __future__ import annotations

import os
import sys
import threading

import cv2 as _cv2
import numpy as np

MIN_PIX = int(os.environ.get("IWALY_GPU_MIN_PIX", "120000"))   # pictures smaller than this stay on the CPU (the copy to the GPU costs more)
MIN_K = int(os.environ.get("IWALY_GPU_MIN_K", "13"))            # Gaussian kernels narrower than this stay on the CPU
MIN_MK = int(os.environ.get("IWALY_GPU_MIN_MK", "9"))          # structuring elements smaller than this stay on the CPU
_ENV = os.environ.get("IWALY_GPU_MASKS")

_lock = threading.Lock()
_state = {"ok": None, "torch": None, "F": None, "dev": None, "kern": {}, "calls": 0, "gpu": 0, "fail": 0}


def _flag() -> bool:
    eng = sys.modules.get("contour.retouch.engine")
    v = getattr(eng, "GPU_MASKS", None) if eng is not None else None
    if v is None:
        v = _ENV
    try:
        return bool(float(v)) if v is not None else False
    except Exception:
        return False


def _init() -> bool:
    if _state["ok"] is not None:
        return _state["ok"]
    with _lock:
        if _state["ok"] is not None:
            return _state["ok"]
        try:
            import torch
            import torch.nn.functional as F
            if not torch.cuda.is_available():
                _state["ok"] = False
                return False
            _state.update(torch=torch, F=F, dev=torch.device("cuda"))
            _state["ok"] = True
        except Exception:
            _state["ok"] = False
    return _state["ok"]


def available() -> bool:
    return _flag() and _init()


def _taps(k: int, sigma: float):
    key = (k, round(float(sigma), 6))
    t = _state["kern"].get(key)
    if t is None:
        g = _cv2.getGaussianKernel(k, float(sigma), _cv2.CV_32F).reshape(-1).astype(np.float32)
        t = _state["torch"].from_numpy(g).to(_state["dev"])
        _state["kern"][key] = t
    return t


def _ksize(sigma: float) -> int:
    # OpenCV's own choice for a float picture and ksize (0, 0): round(sigma * 4 * 2 + 1) | 1
    return int(round(float(sigma) * 8.0 + 1.0)) | 1


def gaussian_blur(src, ksize, sigmaX, dst=None, sigmaY=0, borderType=None):
    """cv2.GaussianBlur on the GPU when it pays; OpenCV otherwise (same arguments, same result)."""
    _state["calls"] += 1
    try:
        if (dst is not None or (borderType is not None and borderType not in (_cv2.BORDER_DEFAULT, _cv2.BORDER_REFLECT_101))
                or not isinstance(src, np.ndarray) or src.dtype != np.float32 or src.ndim not in (2, 3)
                or src.shape[0] * src.shape[1] < MIN_PIX or not available()):
            raise LookupError
        sx = float(sigmaX)
        sy = float(sigmaY) if sigmaY else sx
        kx, ky = (int(ksize[0]), int(ksize[1])) if ksize and (ksize[0] or ksize[1]) else (0, 0)
        if kx <= 0:
            if sx <= 0:
                raise LookupError
            kx = _ksize(sx)
        if ky <= 0:
            ky = kx if (not sigmaY and ksize and ksize[0]) else _ksize(sy)
        if sx <= 0:
            sx = 0.3 * ((kx - 1) * 0.5 - 1) + 0.8
        if sy <= 0:
            sy = 0.3 * ((ky - 1) * 0.5 - 1) + 0.8
        h, w = src.shape[:2]
        if max(kx, ky) < MIN_K or kx // 2 >= w or ky // 2 >= h:
            raise LookupError
        torch, F = _state["torch"], _state["F"]
        a = np.ascontiguousarray(src)
        with torch.no_grad():
            x = torch.from_numpy(a).to(_state["dev"], non_blocking=False)
            x = x[None, None] if a.ndim == 2 else x.permute(2, 0, 1)[None]
            c = x.shape[1]
            gx = _taps(kx, sx).view(1, 1, 1, kx).expand(c, 1, 1, kx)
            gy = _taps(ky, sy).view(1, 1, ky, 1).expand(c, 1, ky, 1)
            x = F.conv2d(F.pad(x, (kx // 2, kx // 2, 0, 0), mode="reflect"), gx, groups=c)
            x = F.conv2d(F.pad(x, (0, 0, ky // 2, ky // 2), mode="reflect"), gy, groups=c)
            x = x[0, 0] if a.ndim == 2 else x[0].permute(1, 2, 0)
            out = x.contiguous().cpu().numpy()
        _state["gpu"] += 1
        return out
    except LookupError:
        pass
    except Exception:
        _state["fail"] += 1
    if dst is not None:
        return _cv2.GaussianBlur(src, ksize, sigmaX, dst, sigmaY, borderType if borderType is not None else _cv2.BORDER_DEFAULT)
    if borderType is not None:
        return _cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=sigmaY, borderType=borderType)
    return _cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=sigmaY)


def _morph_gpu(src, kernel, anchor, iterations, borderType, op):
    """dilate (op = "max") / erode ("min") of a 2-D picture by a structuring element whose rows are each one run of ones (rectangles,
    ellipses, crosses): each row is a running max / min along x, the rows' results shifted along y and combined. The border is OpenCV's
    default for morphology (it never wins). Raises LookupError where this does not apply."""
    if (not isinstance(src, np.ndarray) or src.ndim != 2 or src.dtype not in (np.uint8, np.float32)
            or (anchor is not None and tuple(anchor) != (-1, -1)) or (borderType is not None and borderType != _cv2.BORDER_CONSTANT)
            or src.shape[0] * src.shape[1] < MIN_PIX or not available()):
        raise LookupError
    k = np.ones((3, 3), np.uint8) if kernel is None else np.asarray(kernel)
    if k.ndim != 2 or k.size == 0:
        raise LookupError
    kh, kw = k.shape
    if max(kh, kw) < MIN_MK:
        raise LookupError
    ay, ax = kh // 2, kw // 2
    runs = []
    for i in range(kh):
        nz = np.flatnonzero(k[i])
        if nz.size == 0:
            continue
        if nz[-1] - nz[0] + 1 != nz.size:
            raise LookupError
        runs.append((i - ay, int(nz[0]) - ax, int(nz.size)))
    if not runs:
        raise LookupError
    torch, F = _state["torch"], _state["F"]
    h, w = src.shape
    big = float("inf")
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(src)).to(_state["dev"]).float()
        if op == "min":
            x = -x
        for _ in range(max(1, int(iterations or 1))):
            xs = x[None]                                    # (1, H, W)
            hcache = {}
            P = max(max(abs(s0), abs(s0 + n0 - 1)) for _, s0, n0 in runs)
            xp = F.pad(xs, (P, P), value=-big)
            out = None
            Q = max(abs(dy) for dy, _, _ in runs)
            for dy, s0, n0 in runs:
                hk = (s0, n0)
                hrow = hcache.get(hk)
                if hrow is None:
                    m = F.max_pool1d(xp, n0, stride=1) if n0 > 1 else xp
                    hrow = m[:, :, s0 + P: s0 + P + w]
                    hrow = F.pad(hrow, (0, 0, Q, Q), value=-big)[0]          # (H + 2Q, W)
                    hcache[hk] = hrow
                part = hrow[Q + dy: Q + dy + h]
                out = part if out is None else torch.maximum(out, part)
            x = out
        if op == "min":
            x = -x
        r = x.cpu().numpy()
    _state["gpu"] += 1
    if src.dtype == np.uint8:
        return np.clip(np.rint(r), 0, 255).astype(np.uint8)
    return r.astype(np.float32)


def _morph(op, src, kernel, dst=None, anchor=None, iterations=1, borderType=None, borderValue=None):
    _state["calls"] += 1
    if dst is None and borderValue is None:
        try:
            return _morph_gpu(src, kernel, anchor, iterations, borderType, op)
        except LookupError:
            pass
        except Exception:
            _state["fail"] += 1
    f = _cv2.dilate if op == "max" else _cv2.erode
    kw = {}
    if anchor is not None:
        kw["anchor"] = anchor
    if iterations is not None and iterations != 1:
        kw["iterations"] = iterations
    if borderType is not None:
        kw["borderType"] = borderType
    if borderValue is not None:
        kw["borderValue"] = borderValue
    if dst is not None:
        return f(src, kernel, dst, **kw)
    return f(src, kernel, **kw)


def dilate(src, kernel, dst=None, anchor=None, iterations=1, borderType=None, borderValue=None):
    return _morph("max", src, kernel, dst, anchor, iterations, borderType, borderValue)


def erode(src, kernel, dst=None, anchor=None, iterations=1, borderType=None, borderValue=None):
    return _morph("min", src, kernel, dst, anchor, iterations, borderType, borderValue)


def morphology_ex(src, op, kernel, dst=None, anchor=None, iterations=1, borderType=None, borderValue=None):
    """cv2.morphologyEx for OPEN / CLOSE / TOPHAT / BLACKHAT / GRADIENT out of dilate and erode (OpenCV composes them the same way)."""
    if dst is None and borderValue is None and borderType is None and (anchor is None or tuple(anchor) == (-1, -1)) and iterations in (None, 1):
        try:
            if op == _cv2.MORPH_OPEN:
                return dilate(erode(src, kernel), kernel)
            if op == _cv2.MORPH_CLOSE:
                return erode(dilate(src, kernel), kernel)
            if op in (_cv2.MORPH_TOPHAT, _cv2.MORPH_BLACKHAT, _cv2.MORPH_GRADIENT) and isinstance(src, np.ndarray) and src.dtype == np.float32:
                if op == _cv2.MORPH_TOPHAT:
                    return (src - dilate(erode(src, kernel), kernel)).astype(np.float32)
                if op == _cv2.MORPH_BLACKHAT:
                    return (erode(dilate(src, kernel), kernel) - src).astype(np.float32)
                return (dilate(src, kernel) - erode(src, kernel)).astype(np.float32)
        except Exception:
            _state["fail"] += 1
    kw = {}
    if anchor is not None:
        kw["anchor"] = anchor
    if iterations is not None and iterations != 1:
        kw["iterations"] = iterations
    if borderType is not None:
        kw["borderType"] = borderType
    if borderValue is not None:
        kw["borderValue"] = borderValue
    if dst is not None:
        return _cv2.morphologyEx(src, op, kernel, dst, **kw)
    return _cv2.morphologyEx(src, op, kernel, **kw)


class _CV2:
    """OpenCV, with the ported operations swapped in."""

    def __getattr__(self, name):
        return getattr(_cv2, name)

    GaussianBlur = staticmethod(gaussian_blur)
    dilate = staticmethod(dilate)
    erode = staticmethod(erode)
    morphologyEx = staticmethod(morphology_ex)


_PROXY = _CV2()


def proxy():
    return _PROXY


def stats() -> dict:
    return {k: _state[k] for k in ("ok", "calls", "gpu", "fail")}
