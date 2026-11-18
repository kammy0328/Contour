"""The app side of DIAP's combined model (contour/learn/unified): one ONNX file in place of face_parse,
parse_clothes and matte_rvm. Off unless switched on (settings.unified_model or DIAP_UNIFIED=1) and the file is in
the models folder (models/onnx/unified_diap.onnx).

While it is on, the three models it stands in for are neither downloaded nor loaded. If the file does not load,
or a run fails, that feature goes back to its own model (downloaded then, if need be) for the rest of the session;
the other features keep the combined model.
It is stateless: no memory between frames, so nothing carries over a cut or a change of person (RVM's recurrent
state is what reset() clears; here there is none to clear)."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

KEY = "unified_diap"
REPLACES = {"face": "face_parse", "cloth": "parse_clothes", "matte": "matte_rvm"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FRAME_W = 512

_lock = threading.Lock()
_rt = None
_failed: set = set()            # features that went back to their own model this session ('load' = the file itself)


def path() -> Path:
    from . import models as M
    return M.model_path(KEY)


def switched_on() -> bool:
    if os.environ.get("DIAP_UNIFIED", "") == "1":
        return True
    if os.environ.get("DIAP_UNIFIED", "") == "0":
        return False
    try:
        from ..core.settings import settings
        return bool(getattr(settings(), "unified_model", False))
    except Exception:
        return False


def active(feature: str) -> bool:
    """The combined model serves this feature ('face', 'cloth', 'matte') now."""
    return switched_on() and "load" not in _failed and feature not in _failed and path().is_file()


def fail(feature: str, err: Exception) -> None:
    """This feature goes back to its own model for the rest of the session."""
    _failed.add(feature)
    sys.stderr.write(f"UNIFIED OFF for {feature}: {type(err).__name__}: {str(err)[:200]}\n")


def reset_session() -> None:
    global _rt
    with _lock:
        _rt = None
        _failed.clear()


def runtime() -> Optional["Runtime"]:
    global _rt
    if not switched_on() or "load" in _failed or not path().is_file():
        return None
    with _lock:
        if _rt is None:
            try:
                _rt = Runtime(path())
            except Exception as e:
                fail("load", e)
                return None
        return _rt


def provider(feature: str) -> str:
    """The model key that makes this feature's masks now (recorded with the analysis: a cut whose masks were made
    by another model can have just those masks made again)."""
    return KEY if active(feature) else REPLACES[feature]


def stale(models_used: Dict[str, str]) -> list:
    """Features whose masks in a cut were made by a model other than the one that would make them now (by the
    fingerprints recorded at analysis). A cut analysed before this was recorded is taken as made by the three."""
    from . import models as M
    out = []
    for feat, orig in REPLACES.items():
        now = provider(feat)
        if now == KEY:
            # made by the three (or by another version of the combined model)
            if models_used.get(KEY) != M.fingerprint(KEY):
                out.append(feat)
        elif KEY in models_used:
            out.append(feat)                # made by the combined model, which is off now
        elif orig in models_used and M.fingerprint(orig) and models_used[orig] != M.fingerprint(orig):
            out.append(feat)                # another version of its own model
    return out


def _norm(img_u8: np.ndarray) -> np.ndarray:
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class Runtime:
    def __init__(self, p: Path):
        from . import models as M
        self.sess = M.session(p)
        self.inp = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]
        for k in ("face", "cloth", "alpha"):
            if k not in self.outs:
                raise RuntimeError(f"{p.name}: '{k}' 출력이 없습니다")

    def _run(self, x: np.ndarray, name: str) -> np.ndarray:
        return self.sess.run([name], {self.inp: x})[0]

    def face_logits(self, crop512: np.ndarray) -> np.ndarray:
        """(19, 128, 128) like face_parse on the same 512 px face crop."""
        return self._run(_norm(crop512), "face")[0]

    def cloth_logits(self, crop512: np.ndarray) -> np.ndarray:
        """(18, 128, 128) like parse_clothes on the same 512 x 512 person crop."""
        return self._run(_norm(crop512), "cloth")[0]

    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        """Person alpha (H, W) 0..1 of a whole (exposure-evened) frame, like RVM's."""
        h, w = rgb.shape[:2]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        fh = max(32, int(round(FRAME_W * h / max(1, w) / 32.0)) * 32)
        small = cv2.resize(rgb, (FRAME_W, fh), interpolation=cv2.INTER_AREA)
        a = self._run(_norm(small), "alpha")[0, 0]
        return np.clip(cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
