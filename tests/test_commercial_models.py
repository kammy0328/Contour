"""Commercially usable models in place of the non-commercial ones: faces with Google MediaPipe Selfie Multiclass
(Apache-2.0, shipped) plus the landmarks, clothing with SAM 3.1's garment phrases (SAM License). Chosen by
settings.face_model / cloth_model; the classes the retouch and the clothing store read stay the same."""
import types

import cv2
import numpy as np
import pytest

from contour.track import faceparse as FP
from contour.track import clothing as CL


def _lm(cx=256.0, top=180.0, w=230.0):
    """68 landmarks of a frontal face in a 512 picture (the crop is then about the picture itself)."""
    lm = np.zeros((68, 3), np.float64)
    a = np.linspace(np.pi, 0, 17)
    lm[:17, 0] = cx - np.cos(a) * w / 2
    lm[:17, 1] = top + np.sin(a) * w * 0.55
    lm[17:22, :2] = np.c_[np.linspace(cx - 90, cx - 30, 5), np.full(5, top - 60)]
    lm[22:27, :2] = np.c_[np.linspace(cx + 30, cx + 90, 5), np.full(5, top - 60)]
    lm[27:31, :2] = np.c_[np.full(4, cx), np.linspace(top - 40, top + 10, 4)]
    lm[31:36, :2] = np.c_[np.linspace(cx - 20, cx + 20, 5), np.full(5, top + 20)]
    for a0, ex in ((36, cx - 60), (42, cx + 60)):
        ang = np.linspace(0, 2 * np.pi, 7)[:-1]
        lm[a0:a0 + 6, :2] = np.c_[ex + 18 * np.cos(ang), top - 35 + 7 * np.sin(ang)]
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    lm[48:60, :2] = np.c_[cx + 35 * np.cos(ang), top + 60 + 12 * np.sin(ang)]
    lm[60:68, :2] = np.c_[cx + 25 * np.cos(ang[::3].repeat(2)[:8]), top + 60 + 6 * np.sin(ang[::3].repeat(2)[:8])]
    lm[:, 2] = 1
    return lm


def test_settings_choose_the_models(monkeypatch):
    monkeypatch.delenv("DIAP_FACE_MODEL", raising=False)
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    assert FP.face_model() == FP.DEFAULT_MODEL and CL.cloth_model() == CL.DEFAULT_MODEL
    monkeypatch.setenv("DIAP_FACE_MODEL", "mp")
    assert FP.face_model() == "mp" and FP.ready() and isinstance(FP.make_parser(), FP.MPFaceParser)
    monkeypatch.setenv("DIAP_FACE_MODEL", "nonsense")
    assert FP.face_model() == FP.DEFAULT_MODEL


def test_shipped_multiclass_model_runs():
    p = FP.MPFaceParser()
    pr = p.probs(np.full((512, 512, 3), 128, np.uint8))
    assert pr.shape == (6, FP.MP_SIZE, FP.MP_SIZE) and np.allclose(pr.sum(0), 1, atol=1e-4)


def test_classes_from_multiclass_and_landmarks(monkeypatch):
    """skin = face skin without eyes, brows, lips; neck = body skin under the jaw only; a hand under the chin
    (body skin too) is kept out of the neck."""
    p = FP.MPFaceParser()
    lm = _lm()
    M_ = FP.FaceParser.crop_affine(lm)

    def probs(crop):
        pr = np.zeros((6, FP.MP_SIZE, FP.MP_SIZE), np.float32)
        pr[FP.MP_FACE] = 1.0                                   # face skin everywhere ...
        lmc = (np.c_[lm[:, :2], np.ones(68)] @ M_.T) * FP.MP_SIZE / FP.SIZE
        chin = int(lmc[8, 1])
        pr[FP.MP_FACE, chin:] = 0
        pr[FP.MP_BODY, chin:] = 1.0                            # ... body skin under the chin, all the way across
        return pr
    monkeypatch.setattr(p, "probs", probs)
    rgb = np.zeros((512, 512, 3), np.uint8)
    r = p.parse(rgb, lm)
    K = FP.KEEP
    lmc = (np.c_[lm[:, :2], np.ones(68)] @ M_.T) * K / FP.SIZE
    ey, ex = lmc[36:42].mean(0).astype(int)[::-1]
    my, mx = lmc[48:60].mean(0).astype(int)[::-1]
    cy, cx = lmc[30].astype(int)[::-1]
    assert r["skin"][ey, ex] < 0.2 and r["skin"][my, mx] < 0.2 and r["skin"][cy, cx] > 0.8
    chin = lmc[8]
    wfk = abs(lmc[16, 0] - lmc[0, 0])
    below = (int(chin[1] + 0.15 * wfk), int(chin[0]))
    assert r["neck"][below] > 0.5
    far_side = (below[0], min(K - 1, int(max(lmc[0, 0], lmc[16, 0]) + 0.4 * wfk)))
    assert r["neck"][far_side] < 0.1                            # shoulder / arm skin is not the neck
    # a hand under the chin
    hand = np.c_[np.linspace(lm[8, 0] - 40, lm[8, 0] + 40, 21), np.full(21, lm[8, 1] + 35)]
    hand[::2, 1] += 25
    r2 = p.parse(rgb, lm, [(hand, 0.0)])
    assert r2["neck"][below] < 0.2


def test_hands_and_forearms_from_the_track():
    from contour.track.store import TrackStore
    st = TrackStore(3, [1])
    st.present[0] = True
    st.scale[0] = 300
    st.final[0, :, :, :2] = np.nan
    st.final[0, 1, 7, :2] = [100, 200]
    st.final[0, 1, 9, :2] = [150, 220]
    st.vis[0, 1, 9] = 1
    occ = FP.limb_occluders(st, 1, 2.0, 2.0)
    assert len(occ) == 1 and occ[0][1] > 0 and np.allclose(occ[0][0][0], [200, 400])
    m = FP.occluder_mask(occ, np.array([[0.5, 0, 0], [0, 0.5, 0]]), 512)
    assert m[int(210), int(125)] > 0.5 and m[50, 450] == 0


class _Enc:
    pass


def test_garment_phrases_make_the_clothes_classes(monkeypatch, tmp_path):
    """Each phrase's masks become that garment's class; where two claim a pixel the surer one takes it; the map is
    in the clothes model's classes so keep_mask stays as it is."""
    f = tmp_path / "g.npz"
    np.savez(f, skirt__text=np.zeros((32, 1, 256), np.float32), skirt__text_pad=np.zeros((1, 32), bool),
             dress__text=np.ones((32, 1, 256), np.float32), dress__text_pad=np.zeros((1, 32), bool))
    monkeypatch.setattr(CL, "GARMENT_FILE", f)
    import contour.track.infer as INF

    class Seg:
        def __init__(self, *a):
            pass

        def encode(self, rgb):
            return dict(emb={}, rect=(0, 0, rgb.shape[1], rgb.shape[0]))

    class Find:
        def __init__(self, *a):
            self.const = dict(text=None, text_pad=None, box_tokens=None)

        def find(self, enc, thr):
            lg = np.full((16, 16), -5.0, np.float32)
            if float(self.const["text"].max()) == 0.0:        # skirt: lower half
                lg[8:] = 5.0
                return [dict(score=0.9, lg=lg, rect=enc["rect"], box=None)]
            lg[6:12] = 3.0                                       # dress: a band, less sure
            return [dict(score=0.6, lg=lg, rect=enc["rect"], box=None)]
    monkeypatch.setattr(INF, "Segmenter31", Seg)
    monkeypatch.setattr(INF, "Finder31", Find)
    gf = CL.GarmentFinder()
    cm = gf.classes(np.zeros((64, 64, 3), np.uint8), (32, 32))
    assert cm[28, 5] == 5                     # skirt
    assert cm[13, 5] == 7                     # dress (only it)
    assert cm[20, 5] == 5                     # both: skirt is surer
    assert cm[2, 5] == 0


def test_catch_all_clothing_only_fills_what_garment_words_left(monkeypatch, tmp_path):
    """'clothing' (a denim jacket no garment word found) counts as a top, but never over a found garment."""
    f = tmp_path / "g.npz"
    np.savez(f, pants__text=np.zeros((32, 1, 256), np.float32), pants__text_pad=np.zeros((1, 32), bool),
             clothing__text=np.ones((32, 1, 256), np.float32), clothing__text_pad=np.zeros((1, 32), bool))
    monkeypatch.setattr(CL, "GARMENT_FILE", f)
    import contour.track.infer as INF

    class Seg:
        def __init__(self, *a):
            pass

        def encode(self, rgb):
            return dict(emb={}, rect=(0, 0, rgb.shape[1], rgb.shape[0]))

    class Find:
        def __init__(self, *a):
            self.const = dict(text=None, text_pad=None, box_tokens=None)

        def find(self, enc, thr):
            lg = np.full((16, 16), -5.0, np.float32)
            if float(self.const["text"].max()) == 0.0:          # pants: lower half, not very sure
                lg[8:] = 1.0
                return [dict(score=0.5, lg=lg, rect=enc["rect"], box=None)]
            lg[:] = 6.0                                            # clothing: everything, very sure
            return [dict(score=0.99, lg=lg, rect=enc["rect"], box=None)]
    monkeypatch.setattr(INF, "Segmenter31", Seg)
    monkeypatch.setattr(INF, "Finder31", Find)
    cm = CL.GarmentFinder().classes(np.zeros((64, 64, 3), np.uint8), (32, 32))
    assert cm[28, 5] == 6 and cm[4, 5] == 4


def test_defaults_are_the_commercially_usable_models(monkeypatch):
    monkeypatch.delenv("DIAP_FACE_MODEL", raising=False)
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    assert FP.face_model() == "mp" and CL.cloth_model() == "sam31"
    monkeypatch.setattr(CL, "GARMENT_FILE", CL.GARMENT_FILE.with_name("missing.npz"))
    assert CL.cloth_model() == "sam31"            # a missing file never switches to parse_clothes by itself
