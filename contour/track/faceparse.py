"""Face parsing: what each pixel around a face is (skin, neck, hair, clothes, a hand in front of it).

SegFormer (MiT-B5) fine-tuned on CelebAMask-HQ, 19 classes (jonathandinu/face-parsing, ONNX export). Licence:
non-commercial research and educational use, like the CelebAMask-HQ data. Downloaded once with the other
models (about 340 MB) into the data folder.

The face is cut out level (eyes horizontal) at the framing the model was trained on, parsed at 512 px, and
the class probabilities that the retouch needs are kept at a small size together with the crop's placement:
- skin: face skin and nose (ears are left out: they are rarely retouched and catch hair and earrings);
- neck: the neck itself (not the clothes, not a necklace);
- hair, cloth: for diagnostics.
Anything else in front of the face (a hand, a light stick, a microphone) is none of these: CelebAMask-HQ marks
occluders as background, so they fall out of the skin and neck by themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MODEL_KEY = "face_parse"
SIZE = 512
KEEP = 160                                   # stored size of each class map
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# class ids of this model (config.json id2label)
SKIN = (1, 2)                                # skin, nose
NECK = (17,)
HAIR = (13,)
CLOTH = (18, 16)                             # cloth, necklace
FEATURES = (3, 4, 5, 6, 7, 10, 11, 12)       # glasses, eyes, brows, mouth, lips


def available() -> bool:
    """Usable now (downloaded) or downloadable."""
    from . import models as M
    return MODEL_KEY in M.MODELS


def ready() -> bool:
    from . import models as M
    return M.is_downloaded(MODEL_KEY)


class FaceParser:
    def __init__(self):
        from . import models as M
        self.s = M.session(M.ensure_model(MODEL_KEY))
        self.inp = self.s.get_inputs()[0].name

    @staticmethod
    def crop_affine(lm: np.ndarray) -> Optional[np.ndarray]:
        """2x3 affine image px -> crop px (SIZE). Framing like CelebAMask-HQ: face about 45 % of the width,
        eyes level, a little more room below the chin than above the forehead (the neck is in)."""
        L = lm[:, :2]
        if not np.isfinite(L[[0, 8, 16, 36, 39, 42, 45]]).all():
            return None
        le, re_ = L[36:42].mean(0), L[42:48].mean(0)
        d = re_ - le
        ang = np.arctan2(d[1], d[0])
        wf = float(np.hypot(*(L[16] - L[0])))
        if wf < 8:
            return None
        eye_c = (le + re_) / 2
        down = np.array([-np.sin(ang), np.cos(ang)])
        right = np.array([np.cos(ang), np.sin(ang)])
        # a turned head (profile): the jaw width collapses while the face keeps its height, which cut the crop
        # to half a face. Frontal eye-to-chin is about 0.85 of the jaw width; size by the larger of the two,
        # and centre sideways on the visible landmarks.
        hf = float(np.hypot(*(L[8] - eye_c)))
        wf = min(max(wf, hf / 0.85), 2.2 * wf)
        c = eye_c + down * 0.45 * wf
        if hf > 0.85 * float(np.hypot(*(L[16] - L[0]))) * 1.08:
            off = float((L[:, :2].mean(0) - eye_c) @ right)
            c = c + right * np.clip(off, -0.35 * wf, 0.35 * wf)
        side = wf / 0.45
        s = SIZE / side
        ca, sa = np.cos(-ang) * s, np.sin(-ang) * s
        M = np.array([[ca, -sa, 0.0], [sa, ca, 0.0]])
        M[:, 2] = np.array([SIZE / 2, SIZE / 2]) - M[:, :2] @ c
        return M

    def parse(self, rgb_u8: np.ndarray, lm: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """lm in the same px as rgb_u8. Returns dict(M=affine img->crop(SIZE), skin, neck, hair, cloth, feat)
        with class maps float16 KEEP x KEEP in crop space."""
        M = self.crop_affine(lm)
        if M is None:
            return None
        crop = cv2.warpAffine(rgb_u8, M, (SIZE, SIZE), flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        lo = self.s.run(None, {self.inp: x})[0][0]                     # (19, 128, 128) logits, 1/4 size
        lo = lo - lo.max(0, keepdims=True)
        e = np.exp(lo)
        pr = e / e.sum(0, keepdims=True)

        def cls(ids):
            m = pr[list(ids)].sum(0)
            return cv2.resize(m, (KEEP, KEEP), interpolation=cv2.INTER_LINEAR).astype(np.float16)
        return dict(M=M.astype(np.float32), skin=cls(SKIN), neck=cls(NECK), hair=cls(HAIR), cloth=cls(CLOTH),
                    feat=cls(FEATURES))


def to_image(m: np.ndarray, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """A KEEP x KEEP crop map back onto the image (h, w): outside the crop is 0."""
    h, w = shape
    S = np.array([[KEEP / SIZE, 0, 0], [0, KEEP / SIZE, 0]], np.float64)
    A = np.vstack([M.astype(np.float64), [0, 0, 1]])
    A2 = S @ A                                                          # image -> KEEP crop
    return cv2.warpAffine(m.astype(np.float32), A2, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
