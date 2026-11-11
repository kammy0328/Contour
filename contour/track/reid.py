"""Who is who: person re-identification features.

Body: OSNet-IBN x1.0 trained for person re-identification on MSMT17 (Kaiyang Zhou, MIT; converted to ONNX by
tools/build_reid_onnx.py and shipped in assets/onnx). Instance-batch normalisation makes it less sensitive to
the colour and brightness of stage lighting than plain colour histograms.
Face: OpenCV SFace (Apache-2.0) on the YuNet face found on the person's head, aligned on its 5 landmarks.
Members in the same outfit are told apart by face whenever the face is seen."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import models as M
from . import skeleton as S

_ASSET = Path(__file__).resolve().parents[2] / "assets" / "onnx" / "reid_osnet_ibn_x1_0.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FACE_SAME = 0.36        # SFace cosine above which two faces are the same person (OpenCV's threshold 0.363)


class ReID:
    def __init__(self):
        self.sess = M.session(_ASSET)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(n, 512) unit vectors (zeros where the box is unusable)."""
        n = len(boxes)
        out = np.zeros((n, 512), np.float32)
        if n == 0:
            return out
        h, w = rgb.shape[:2]
        crops, idx = [], []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 6 or y2 - y1 < 12:
                continue
            c = cv2.resize(rgb[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            crops.append(((c - MEAN) / STD).transpose(2, 0, 1))
            idx.append(i)
        if not crops:
            return out
        for s in range(0, len(crops), 16):
            f = self.sess.run(None, {self.inp: np.stack(crops[s:s + 16]).astype(np.float32)})[0]
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
            out[idx[s:s + 16]] = f
        return out


class FaceID:
    def __init__(self, face_detector=None):
        from .infer import FaceDetector
        self.det = face_detector or FaceDetector()
        self.rec = cv2.FaceRecognizerSF.create(str(M.ensure_model("face_sface")), "")

    def __call__(self, rgb: np.ndarray, kp: np.ndarray) -> Optional[dict]:
        """Face embedding of the person whose joints are kp (17,3), or None when no usable face."""
        head = kp[[S.NOSE, S.L_EYE, S.R_EYE, S.L_EAR, S.R_EAR]]
        ok = np.isfinite(head[:, :2]).all(1)
        if ok.sum() < 2:
            return None
        c = head[ok, :2].mean(0)
        sc = S.person_scale(kp)
        size = max(24.0, 0.22 * sc) if np.isfinite(sc) else 48.0
        faces = self.det.detect(rgb, np.array([c[0] - size, c[1] - size, c[0] + size, c[1] + size]))
        if not len(faces):
            return None
        cen = faces[:, 0:2] + faces[:, 2:4] / 2
        k = int(np.argmin(np.linalg.norm(cen - c, axis=1)))
        f = faces[k]
        # the person's own eyes and nose must sit on this face: in a tight group the nearest face can be the
        # neighbour's when this person has turned away
        if np.linalg.norm(cen[k] - c) > 0.6 * f[2]:
            return None
        # profile and far-away faces carry little identity
        if f[14] < 0.7 or f[2] < 24:
            return None
        eyes = np.hypot(f[6] - f[4], f[7] - f[5])
        if eyes < 0.25 * f[2]:
            return None
        al = self.rec.alignCrop(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f[:15].reshape(1, -1).astype(np.float32))
        fe = self.rec.feature(al).ravel().astype(np.float32)
        fe /= np.linalg.norm(fe) + 1e-9
        return dict(feat=fe, score=float(f[14]), w=float(f[2]))


def cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or not a.any() or not b.any():
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def face_sim(fa: List[np.ndarray], fb: List[np.ndarray]) -> float:
    """Best cosine between two small face galleries (NaN when either has none)."""
    if not fa or not fb:
        return float("nan")
    A = np.stack(fa)
    Bm = np.stack(fb)
    return float((A @ Bm.T).max())
