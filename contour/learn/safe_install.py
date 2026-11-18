"""Putting a trained model where the app reads it, safely (face parser step training, the combined model).

- check_onnx: the exported file loads in ONNX Runtime, runs, gives the expected output shape, only finite values,
  and agrees with the training framework's answer within a tolerance. Nothing is installed that fails this.
- install: a temporary copy next to the target (same file system), checked again, then swapped in with os.replace
  (atomic): the app never sees half a file, and on any failure the model in use stays as it was.
- atomic_json: the training history written the same way (a crash while writing never loses the earlier records).
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Dict, Optional, Sequence

import numpy as np


def check_onnx(path: str, feed: Dict[str, np.ndarray], ref: Optional[np.ndarray] = None,
               shape: Optional[Sequence[int]] = None, atol: float = 1e-2) -> dict:
    """{'ok', 'reason', 'shape', 'max_diff'} for the first output of the model at path on feed."""
    out = dict(ok=False, reason="", shape=None, max_diff=None)
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        o = s.run(None, feed)[0]
    except Exception as e:
        out["reason"] = f"ONNX 로드·추론 실패: {type(e).__name__}: {str(e)[:200]}"
        return out
    out["shape"] = list(o.shape)
    if shape is not None and tuple(o.shape) != tuple(shape):
        out["reason"] = f"출력 형태 {tuple(o.shape)} != {tuple(shape)}"
        return out
    if not np.isfinite(o).all():
        out["reason"] = "출력에 NaN/Inf"
        return out
    if ref is not None:
        if tuple(np.shape(ref)) != tuple(o.shape):
            out["reason"] = f"출력 형태 {tuple(o.shape)}가 학습 모델 {tuple(np.shape(ref))}와 다름"
            return out
        d = float(np.abs(o.astype(np.float64) - np.asarray(ref, np.float64)).max())
        out["max_diff"] = d
        if not d <= atol:
            out["reason"] = f"학습 모델과 차이 {d:.4g} > 허용 {atol:g}"
            return out
    out["ok"] = True
    return out


def install(src: str, dst: str, check: Optional[Callable[[str], dict]] = None) -> dict:
    """Copy src to dst atomically. check(path) -> {'ok', 'reason'} runs on the temporary copy before the swap.
    Returns {'ok', 'reason', 'path'}; on failure dst is untouched and no temporary file is left."""
    if os.path.abspath(src).lower() == os.path.abspath(dst).lower():
        # already where the app reads it (the models folder could not be found and Cloud\model was used)
        r = check(dst) if check is not None else dict(ok=True)
        return dict(ok=bool(r.get("ok")), reason=r.get("reason", ""), path=dst)
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(dst)}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        if os.path.getsize(tmp) != os.path.getsize(src):
            raise OSError("복사본 크기가 다릅니다")
        if check is not None:
            r = check(tmp)
            if not r.get("ok"):
                raise RuntimeError(r.get("reason") or "검사 실패")
        os.replace(tmp, dst)
        return dict(ok=True, reason="", path=dst)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return dict(ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}", path=dst)


def atomic_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
