"""GPU warp for the render: the whole per-plane warp (field upsampling + bicubic resampling) as one small ONNX
graph run by ONNX Runtime on CUDA (or DirectML). Only the coarse field and the planes cross the bus.

The render checks it against the CPU warp on its first frame (same picture within a fraction of a code value)
and times both; whichever is faster on this machine is used. Any error falls back to the CPU."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import field as F

_ASSET = Path(__file__).resolve().parents[2] / "assets" / "onnx" / "warp_plane.onnx"


class GpuWarper:
    def __init__(self):
        from ..track import models as M
        if not _ASSET.exists():
            raise FileNotFoundError(str(_ASSET))
        self.sess = M.session(_ASSET)
        from ..track import backend as BK
        acc = [BK.BACKEND[p] for p in self.sess.get_providers() if BK.BACKEND.get(p, "cpu") != "cpu"]
        # the device the session was opened for (CUDA, DirectML, Core ML); whether it is used is decided below by
        # running it against the CPU warp on a real frame (choose), not by this name
        self.device = BK.LABEL[acc[0]] if acc else "CPU"

    def warp_planes(self, planes: List[np.ndarray], subsample: List[Tuple[int, int]], fr: F.FieldResult,
                    max_value: float) -> List[np.ndarray]:
        if fr.empty():
            return planes
        D = np.ascontiguousarray(fr.D.transpose(2, 0, 1)[None].astype(np.float32))
        size = np.array([1, 2, fr.out_h, fr.out_w], np.int64)
        out = []
        for pl, (ssx, ssy) in zip(planes, subsample):
            P = np.ascontiguousarray(pl.astype(np.float32)[None, None])
            O = self.sess.run(["O"], {"D": D, "P": P, "size": size, "steps": np.array([ssy, ssx], np.int64)})[0][0, 0]
            if np.issubdtype(pl.dtype, np.integer):
                O = np.clip(np.rint(O), 0, max_value).astype(pl.dtype)
            else:
                O = O.astype(pl.dtype)
            out.append(O)
        return out


def choose(planes, subsample, fr: F.FieldResult, max_value: float) -> Tuple[Optional[GpuWarper], str]:
    """(warper or None for the CPU, a line for the render report). Compared on a real frame."""
    try:
        g = GpuWarper()
    except Exception as e:
        return None, f"가속: CPU (GPU 워프를 쓸 수 없음: {e.__class__.__name__})"
    if g.device == "CPU":
        return None, "가속: CPU (GPU 실행 장치 없음)"
    try:
        g.warp_planes(planes, subsample, fr, max_value)            # warm-up (kernels, memory)
        t0 = time.perf_counter()
        a = g.warp_planes(planes, subsample, fr, max_value)
        tg = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = F.warp_planes(planes, subsample, fr, max_value)
        tc = time.perf_counter() - t0
        # same picture away from the frame border (the border rule differs by a pixel)
        bad = 0.0
        for x, y in zip(a, b):
            h, w = x.shape[:2]
            m = (slice(4, h - 4), slice(4, w - 4))
            bad = max(bad, float(np.percentile(np.abs(x[m].astype(np.float64) - y[m].astype(np.float64)), 99.9)))
        if bad > 0.01 * max_value:
            return None, f"가속: CPU (GPU 결과 차이 {bad / max_value * 100:.2f}%로 사용 안 함)"
        if tg < tc * 0.9:
            return g, f"가속: {g.device} 워프 (프레임당 워프 {tc * 1000:.0f}ms → {tg * 1000:.0f}ms)"
        return None, f"가속: CPU (이 컴퓨터에서는 CPU 워프가 더 빠름 {tc * 1000:.0f}ms / GPU {tg * 1000:.0f}ms)"
    except Exception as e:
        return None, f"가속: CPU (GPU 워프 오류: {e.__class__.__name__})"
