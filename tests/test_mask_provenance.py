"""Face/neck and clothing masks record the model that really made them (a fallback shows), are made again per cut
on their own (one never touches the other), are swapped in only when complete (cancel / failure: the masks in use
stay and nothing new is marked ready), and with SAM 3.1 garments as the setting a failing combined model goes back
to SAM 3.1, never to parse_clothes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from contour.core.proxy import MatteStore
from contour.track import clothing as CL
from contour.track import facelayer as FL
from contour.track import faceparse as FP
from contour.track import models as M
from contour.track.store import TrackStore
from tests.test_commercial_models import _lm


class _Proxy:
    sx = sy = 1.0
    w, h = 512, 512
    n = 6

    def frame(self, t):
        rng = np.random.default_rng(t)
        return (rng.random((512, 512, 3)) * 255).astype(np.uint8)


def _tracker(tmp_path, N=6):
    st = TrackStore(N, [7])
    st.present[0] = True
    st.has_face = True
    lm = _lm()
    for t in range(N):
        st.face[0, t] = lm.astype(np.float32)
        st.face_vis[0, t] = 1.0
        st.boxes[0, t] = [100, 60, 420, 500]
        st.final[0, t, :, :2] = 200
        st.scale[0, t] = 300
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cut = NS(work_range=(0, N), analysis_version=3, persons=[], id="c")
    tr = NS(store=st, proxy=_Proxy(), cut=cut, project=NS(cache_dir=lambda c: cache),
            matte=NS(w=128, h=128), labels=lambda t: np.zeros((128, 128), np.int16),
            _virtual_face=lambda p, t: None)
    tr.cloth = MatteStore(cache, N, 128, 128, prefix="cloth")
    return tr


@pytest.fixture
def mp(monkeypatch):
    monkeypatch.setenv("DIAP_FACE_MODEL", "mp")
    monkeypatch.setenv("DIAP_UNIFIED", "0")


def test_face_masks_record_their_model_and_old_ones_are_unknown(tmp_path, mp):
    tr = _tracker(tmp_path)
    assert FL.info(tr)["state"] == "none"
    assert FL.build(tr) is not None
    inf = FL.info(tr)
    assert inf["model"].startswith("mp@") and inf["state"] == "current" and inf["seconds"] is not None
    # a layer from before 1.22.1 (no model recorded): kept and said so, not deleted
    f = FL.path_for(tr)
    with np.load(f) as d:
        old = {k: d[k] for k in d.files if k not in ("model", "made", "seconds")}
    np.savez_compressed(f, **old)
    tr._facelayer = None
    assert FL.info(tr)["state"] == "unknown" and f.exists()
    assert FP.label("") == "생성 모델 미확인"


def test_fallback_is_told_apart():
    assert FL.provider_state("mp@1", {"mp"}) == "current"
    assert FL.provider_state("mp@1+unified@2", {"mp", "unified"}) == "fallback"
    assert FL.provider_state("face_parse@3", {"mp"}) == "other"
    assert FL.provider_state("", {"mp"}) == "unknown"
    assert "대체" in FP.label("mp@1+unified@2")


def test_cancelled_face_rebuild_keeps_the_masks_in_use(tmp_path, mp):
    tr = _tracker(tmp_path)
    FL.build(tr)
    f = FL.path_for(tr)
    before = f.read_bytes()
    calls = [0]

    def cancel():
        calls[0] += 1
        return calls[0] > 2
    assert FL.build(tr, None, cancel) is None
    assert f.read_bytes() == before and not list(f.parent.glob("*.tmp.npz"))


def _garments(monkeypatch, tmp_path):
    f = tmp_path / "g.npz"
    np.savez(f, shirt__text=np.zeros((32, 1, 256), np.float32), shirt__text_pad=np.zeros((1, 32), bool))
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
            lg = np.full((16, 16), 5.0, np.float32)
            return [dict(score=0.9, lg=lg, rect=enc["rect"], box=None)]
    monkeypatch.setattr(INF, "Segmenter31", Seg)
    monkeypatch.setattr(INF, "Finder31", Find)


def _no_parse_clothes(monkeypatch):
    real = M.ensure_model

    def ensure(key, *a, **k):
        if key == "parse_clothes":
            raise AssertionError("parse_clothes would be downloaded / loaded")
        return real(key, *a, **k)
    monkeypatch.setattr(M, "ensure_model", ensure)


def test_cloth_masks_by_sam31_record_model_calls_and_time(tmp_path, monkeypatch):
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    monkeypatch.setenv("DIAP_UNIFIED", "0")
    _garments(monkeypatch, tmp_path)
    _no_parse_clothes(monkeypatch)
    tr = _tracker(tmp_path)
    assert CL.run(tr, [0])
    inf = CL.info(tr)
    assert inf["model"].startswith("sam31@") and inf["state"] == "current"
    assert inf["stats"]["frames"] == 6 and inf["stats"]["find_calls"] == 6
    assert CL.ready_for(tr, 7)


def test_failing_combined_model_goes_back_to_sam31_not_parse_clothes(tmp_path, monkeypatch):
    from contour.track import unified as U
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    monkeypatch.setenv("DIAP_UNIFIED", "1")
    U.reset_session()
    _garments(monkeypatch, tmp_path)
    _no_parse_clothes(monkeypatch)
    monkeypatch.setattr(U, "active", lambda f: f == "cloth" and "cloth" not in U._failed)

    class BadRT:
        face_classes = 6

        def cloth_logits(self, c):
            raise RuntimeError("cuda error")
    monkeypatch.setattr(U, "runtime", lambda: BadRT())
    tr = _tracker(tmp_path)
    try:
        assert CL.run(tr, [0])
        assert CL.info(tr)["model"].startswith("sam31@")
    finally:
        U.reset_session()


def test_cancelled_cloth_run_is_not_ready_and_keeps_the_old_masks(tmp_path, monkeypatch):
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    monkeypatch.setenv("DIAP_UNIFIED", "0")
    _garments(monkeypatch, tmp_path)
    tr = _tracker(tmp_path)
    assert CL.run(tr, [0])
    before = tr.cloth.path.read_bytes()
    state = CL._state_path(tr).read_text()
    n = [0]

    def cancel():
        n[0] += 1
        return n[0] > 2
    assert not CL.run(tr, [0], None, cancel)
    assert tr.cloth.path.read_bytes() == before and CL._state_path(tr).read_text() == state
    assert not tr.cloth.tmp_path.exists()
    # a first run cancelled: nothing marked ready, keep_mask gives nothing
    tr2 = _tracker(tmp_path / "x")
    assert not CL.run(tr2, [0], None, lambda: True)
    assert not CL.ready_for(tr2, 7) and CL.keep_mask(tr2, 1, 0, "keep:top") is None


def test_missing_garment_words_leave_clothing_undecided(tmp_path, monkeypatch):
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    monkeypatch.setenv("DIAP_UNIFIED", "0")
    monkeypatch.setattr(CL, "GARMENT_FILE", tmp_path / "missing.npz")
    _no_parse_clothes(monkeypatch)
    tr = _tracker(tmp_path)
    with pytest.raises(RuntimeError, match="옷 단어 파일"):
        CL.run(tr, [0])
    assert not CL.ready_for(tr, 7)


def test_face_and_cloth_masks_are_made_again_separately(tmp_path, monkeypatch, mp):
    monkeypatch.delenv("DIAP_CLOTH_MODEL", raising=False)
    _garments(monkeypatch, tmp_path)
    tr = _tracker(tmp_path)
    FL.build(tr)
    CL.run(tr, [0])
    face_b, cloth_b = FL.path_for(tr).read_bytes(), tr.cloth.path.read_bytes()
    cstate = CL._state_path(tr).read_text()
    FL.build(tr)                                         # face again: clothing untouched
    assert tr.cloth.path.read_bytes() == cloth_b and CL._state_path(tr).read_text() == cstate
    f_mt = FL.path_for(tr).stat().st_mtime_ns
    CL.run(tr, [0])                                      # clothing again: face untouched
    assert FL.path_for(tr).stat().st_mtime_ns == f_mt


def test_inspector_words():
    from contour.ui.inspector import ReshapeTab as Inspector
    t = Inspector.mask_text("의상 마스크", dict(state="current", label="SAM 3.1 옷 단어", seconds=42.3))
    assert "SAM 3.1" in t and "현재 설정과 같음" in t and "42초" in t
    assert "생성 모델 미확인" in Inspector.mask_text("얼굴·목 마스크", dict(state="unknown", label="생성 모델 미확인"))
