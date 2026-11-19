"""Face parsing: what each pixel around a face is (skin, neck, hair, clothes, a hand in front of it).

Two models, chosen by face_model():
- 'mp': Google MediaPipe Selfie Multiclass (Apache-2.0, fit for commercial use; ONNX conversion shipped in
  data/, licence in data/MEDIAPIPE_LICENSE.txt), with eyes, brows, lips and the neck's place from the landmarks;
- 'face_parse': SegFormer (MiT-B5) fine-tuned on CelebAMask-HQ, 19 classes (jonathandinu/face-parsing, ONNX
  export). Licence: non-commercial research and educational use, like the CelebAMask-HQ data. Downloaded once
  with the other models (about 340 MB) into the data folder.

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


MP_FILE = Path(__file__).resolve().parent / "data" / "mp_selfie_multiclass_256.onnx"
MP_SIZE = 256
MP_BG, MP_HAIR, MP_BODY, MP_FACE, MP_CLOTHES, MP_OTHERS = range(6)
DEFAULT_MODEL = "mp"             # commercial use (2026-11-19, j2758); face_parse: non-commercial, on request


def face_model() -> str:
    """Which model parses faces: 'mp' (Google MediaPipe Selfie Multiclass, Apache-2.0, shipped with the app: fit for
    commercial use) or 'face_parse' (SegFormer-B5 on CelebAMask-HQ: non-commercial research and education only).
    settings.face_model, DIAP_FACE_MODEL overrides."""
    import os
    v = os.environ.get("DIAP_FACE_MODEL", "")
    if not v:
        try:
            from ..core.settings import settings
            v = getattr(settings(), "face_model", "") or ""
        except Exception:
            v = ""
    return v if v in ("mp", "face_parse") else DEFAULT_MODEL


def available() -> bool:
    """Usable now (downloaded or shipped) or downloadable."""
    from . import models as M
    return (face_model() == "mp" and MP_FILE.is_file()) or MODEL_KEY in M.MODELS


def ready() -> bool:
    from . import models as M
    from . import unified as U
    if face_model() == "mp":
        return MP_FILE.is_file()
    return U.active("face") or M.is_downloaded(MODEL_KEY)


def file_fp(path) -> str:
    """Short fingerprint of a model file (size and the SHA-256 of its first and last MB): which file made a mask."""
    import hashlib
    try:
        p = Path(path)
        stt = p.stat()
        n = stt.st_size
        ck = (str(p), n, stt.st_mtime)
        if ck in _FP_CACHE:
            return _FP_CACHE[ck]
        h = hashlib.sha256(str(n).encode())
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if n > (1 << 20):
                f.seek(max(0, n - (1 << 20)))
                h.update(f.read(1 << 20))
        _FP_CACHE[ck] = h.hexdigest()[:8]
        return _FP_CACHE[ck]
    except OSError:
        return ""


_FP_CACHE: dict = {}


def label(provider: str) -> str:
    """A provider key in words, for the inspector."""
    if not provider:
        return "생성 모델 미확인"
    names = {"mp": "MediaPipe", "face_parse": "SegFormer 얼굴(비상업)", "face_parse_own": "자체 얼굴 분할(비상업 교사)",
             "face_parse_diap": "추가 학습 얼굴 분할(비상업)", "unified": "DIAP 통합 모델",
             "sam31": "SAM 3.1 옷 단어", "parse_clothes": "SegFormer 옷(비상업)"}
    parts = [names.get(x.split("@")[0], x.split("@")[0]) for x in provider.split("+")]
    return " + ".join(parts) + (" (대체 포함)" if "+" in provider else "")


def make_parser():
    """The face parser in use (face_model())."""
    return MPFaceParser() if face_model() == "mp" else FaceParser()


class FaceParser:
    def __init__(self):
        from . import unified as U
        self.unified = None
        self.tuned = False
        self.own = False
        self.s = None
        self.used: set = set()            # the models that actually made masks with this parser
        if U.active("face"):
            # DIAP's combined model taught by this parser's classes: face_parse is neither downloaded nor loaded
            rt = U.runtime()
            if rt is not None and getattr(rt, "face_classes", 19) == 19:
                self.unified = rt
                return
        self._load_own()

    def _load_own(self):
        from . import models as M
        import os
        base = M.ensure_model(MODEL_KEY)
        # the parser trained further on the user's own footage (Cloud\step_train.bat) sits next to the original;
        # delete that file (or set DIAP_NO_TUNED) to go back
        tuned = Path(str(base)).with_name("face_parse_diap.onnx")
        # DIAP's own parser (a different network taught by the one above, Cloud\step_train.bat installs it once it
        # is good enough) comes first; DIAP_NO_OWN goes back
        own = Path(str(base)).with_name("face_parse_own.onnx")
        self.tuned = False
        self.own = False
        self.s = None
        if own.exists() and not os.environ.get("DIAP_NO_OWN"):
            try:
                self.s = M.session(own)
                self.own = True
            except Exception:
                self.s = None
        if self.s is None and tuned.exists() and not os.environ.get("DIAP_NO_TUNED"):
            try:
                self.s = M.session(tuned)
                self.tuned = True
            except Exception:
                self.s = None
        if self.s is None:
            self.s = M.session(base)
        self.inp = self.s.get_inputs()[0].name
        self._own_key = ("face_parse_own@" + file_fp(own)) if self.own else \
            ("face_parse_diap@" + file_fp(tuned)) if self.tuned else ("face_parse@" + file_fp(base))

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

    def provider(self) -> str:
        """The models that made this parser's masks so far ('+' when one fell back to another)."""
        return "+".join(sorted(self.used))

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
        lo = None
        if self.unified is not None:
            try:
                lo = self.unified.face_logits(crop)
            except Exception as e:
                from . import unified as U
                U.fail("face", e)                  # back to face_parse for the rest of the session
                self.unified = None
                self._load_own()
        if lo is None:
            lo = self.s.run(None, {self.inp: x})[0][0]                 # (19, 128, 128) logits, 1/4 size
            self.used.add(self._own_key)
        else:
            from . import unified as U
            self.used.add("unified@" + file_fp(U.path()))
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


# ------------------------------------------------------------------ MediaPipe Selfie Multiclass (Apache-2.0)
def feature_mask(lmc: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Eyes, brows and lips drawn from the 68 landmarks in crop pixels (the multiclass model counts them as face
    skin; the SegFormer parser has them as classes of their own). float32 (size, size) 0..1."""
    m = np.zeros((size, size), np.uint8)
    L = np.asarray(lmc, np.float64)[:, :2]
    if not np.isfinite(L[17:68]).all():
        return m.astype(np.float32)
    wf = float(np.hypot(*(L[16] - L[0]))) if np.isfinite(L[[0, 16]]).all() else size * 0.45
    for a, b in ((36, 42), (42, 48)):                       # eyes, a little wider than the lids
        c = L[a:b].mean(0)
        poly = c + (L[a:b] - c) * 1.35
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    th = max(2, int(round(0.035 * wf)))
    for a, b in ((17, 22), (22, 27)):                       # brows as thick lines
        cv2.polylines(m, [np.round(L[a:b]).astype(np.int32)], False, 1, thickness=th)
    cv2.fillPoly(m, [np.round(L[48:60]).astype(np.int32)], 1)   # lips and the open mouth
    k = max(1, int(round(0.01 * wf)))
    m = cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.8, 0.006 * wf)), 0, 1)


def neck_zone(lmc: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Where the neck is in the crop: under the jaw line, about as wide as the jaw, down to about 0.65 face widths
    below the chin (the multiclass model has one class for all body skin: neck, chest, arms, a hand)."""
    L = np.asarray(lmc, np.float64)[:, :2]
    z = np.zeros((size, size), np.float32)
    if not np.isfinite(L[:17]).all():
        return z
    wf = float(np.hypot(*(L[16] - L[0])))
    jaw = L[:17]
    bottom = L[8, 1] + 0.65 * wf
    left, right = jaw[:, 0].min() - 0.12 * wf, jaw[:, 0].max() + 0.12 * wf
    poly = np.vstack([jaw, [[right, jaw[16, 1]], [right, bottom], [left, bottom], [left, jaw[0, 1]]]])
    cv2.fillPoly(z, [np.round(poly).astype(np.int32)], 1.0)
    return cv2.GaussianBlur(z, (0, 0), max(1.0, 0.03 * wf))


def limb_occluders(st, t: int, sx: float, sy: float) -> list:
    """Hands and forearms of everybody in frame t, in picture pixels: [(points (n, 2), thickness px)], a polygon's
    hull when thickness is 0. The multiclass model calls a hand under the chin body skin, like the neck; these
    shapes take it out of the neck (the SegFormer parser called a hand background by itself)."""
    from . import skeleton as S
    out = []
    P = len(st.pids)
    for p in range(P):
        if not st.present[p, t]:
            continue
        sc = float(st.scale[p, t]) if np.isfinite(st.scale[p, t]) else 0.0
        if getattr(st, "has_hands", False):
            for h0 in (0, 21):
                hp = st.hand[p, t, h0:h0 + 21, :2].astype(np.float64)
                ok = np.isfinite(hp).all(1)
                if ok.sum() >= 5:
                    out.append((hp[ok] * [sx, sy], 0.0))
        kp = st.final[p, t, :, :2].astype(np.float64)
        for a, b in ((S.L_EL, S.L_WR), (S.R_EL, S.R_WR)):
            if np.isfinite(kp[[a, b]]).all() and st.vis[p, t, b] == 1:
                seg = kp[[a, b]] * [sx, sy]
                # the hand beyond the wrist (a third of the forearm) when the hand points were not found
                ext = seg[1] + (seg[1] - seg[0]) * 0.45
                out.append((np.vstack([seg, ext]), max(4.0, 0.045 * sc * sx)))
    return out


def occluder_mask(occ: list, M_: np.ndarray, size: int = SIZE) -> np.ndarray:
    """The shapes of limb_occluders in a crop (affine M_), widened a little and softened. float32 0..1."""
    m = np.zeros((size, size), np.uint8)
    s = float(np.sqrt(abs(np.linalg.det(M_[:, :2]))))
    for pts, th in occ:
        q = np.c_[pts, np.ones(len(pts))] @ M_.T
        if not np.isfinite(q).all():
            continue
        if th <= 0:
            hull = cv2.convexHull(np.round(q).astype(np.int32))
            cv2.fillConvexPoly(m, hull, 1)
            r = max(2, int(round(0.25 * np.sqrt(max(1.0, cv2.contourArea(hull))))))
            cv2.polylines(m, [hull], True, 1, thickness=r)
        else:
            cv2.polylines(m, [np.round(q).astype(np.int32)], False, 1, thickness=max(2, int(round(th * s))))
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0) * 1.5, 0, 1)


class MPFaceParser:
    """Google MediaPipe Selfie Multiclass (Apache-2.0; converted to ONNX and shipped with the app) on the same
    landmark-aligned face crop as FaceParser. It gives background, hair, body skin, face skin, clothes and
    accessories; the retouch's classes are made from them with the landmarks:
    skin = face skin without eyes, brows and lips; neck = body skin under the jaw; hair; cloth = clothes and
    accessories. A hand in front of the face is body skin, so it stays out of the face's skin as before."""

    own = False
    tuned = False

    def __init__(self):
        from . import models as M
        from . import unified as U
        self.unified = None
        if U.active("face"):
            rt = U.runtime()
            if rt is not None and getattr(rt, "face_classes", 0) == 6:
                self.unified = rt           # DIAP's combined model, taught by this one
        self.s = M.session(MP_FILE)
        self.inp = self.s.get_inputs()[0].name
        self.used: set = set()

    def provider(self) -> str:
        return "+".join(sorted(self.used))

    def probs(self, crop512: np.ndarray) -> np.ndarray:
        """(6, MP_SIZE, MP_SIZE) class probabilities of a 512 px face crop."""
        if self.unified is not None:
            try:
                lo = self.unified.face_logits(crop512)
                from . import unified as U
                self.used.add("unified@" + file_fp(U.path()))
                lo = np.stack([cv2.resize(c, (MP_SIZE, MP_SIZE), interpolation=cv2.INTER_LINEAR) for c in lo])
                lo = lo - lo.max(0, keepdims=True)
                e = np.exp(lo)
                return e / e.sum(0, keepdims=True)
            except Exception as e_:
                from . import unified as U
                U.fail("face", e_)
                self.unified = None
        x = cv2.resize(crop512, (MP_SIZE, MP_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        lo = self.s.run(None, {self.inp: x[None]})[0][0].transpose(2, 0, 1)
        self.used.add("mp@" + file_fp(MP_FILE))
        lo = lo - lo.max(0, keepdims=True)
        e = np.exp(lo)
        return e / e.sum(0, keepdims=True)

    takes_occluders = True

    def parse(self, rgb_u8: np.ndarray, lm: np.ndarray, occ: Optional[list] = None) -> Optional[Dict[str, np.ndarray]]:
        """As FaceParser.parse; occ: limb_occluders of this frame (hands and forearms kept out of the neck)."""
        M_ = FaceParser.crop_affine(lm)
        if M_ is None:
            return None
        crop = cv2.warpAffine(rgb_u8, M_, (SIZE, SIZE), flags=cv2.INTER_AREA if M_[0, 0] < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        pr = self.probs(crop)
        lmc = np.c_[np.asarray(lm, np.float64)[:, :2], np.ones(len(lm))] @ M_.T
        up = lambda a: cv2.resize(a.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        feat = feature_mask(lmc)
        skin = up(pr[MP_FACE]) * (1.0 - feat)
        neck = up(pr[MP_BODY]) * neck_zone(lmc)
        if occ:
            om = occluder_mask(occ, M_)
            neck = neck * (1.0 - om)
            skin = skin * (1.0 - om)
        keep = lambda a: cv2.resize(a, (KEEP, KEEP), interpolation=cv2.INTER_AREA).astype(np.float16)
        return dict(M=M_.astype(np.float32), skin=keep(skin), neck=keep(neck), hair=keep(up(pr[MP_HAIR])),
                    cloth=keep(up(pr[MP_CLOTHES] + pr[MP_OTHERS])), feat=keep(feat))
