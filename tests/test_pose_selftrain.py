"""Self-learning from originals alone (no BEFORE/AFTER): the automatic track is the answer only where it is sure,
and not where people overlap (occlusion, numbers that can swap), where the forward and backward passes disagree
over a stretch, where a joint moves very fast, or on a flash frame. Easy sharp frames are kept and made hard in
training; a turned and scaled crop carries its answer with it; the check holds out whole projects."""
import json
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from contour.learn import refiner as RF
from contour.track import skeleton as S
from contour.track.store import TrackStore


def _store(N=40, P=2, overlap_at=None, fb_at=None, fast_at=None):
    st = TrackStore(N, list(range(1, P + 1)))
    rng = np.random.default_rng(0)
    for p in range(P):
        x0 = 100 + 400 * p
        st.present[p] = True
        st.scale[p] = 300.0
        st.blur[p] = 1.0
        st.auto_conf[p] = 0.95
        st.conf[p] = 0.95
        base = np.stack([np.linspace(x0, x0 + 100, S.K), np.linspace(100, 400, S.K)], -1)
        for t in range(N):
            st.auto[p, t, :, :2] = base + [t * 1.0, 0]
            st.auto[p, t, :, 2] = 0.9
            st.meas[p, t, :, :2] = st.auto[p, t, :, :2] + rng.normal(0, 3, (S.K, 2))
            st.meas[p, t, :, 2] = 0.9
            st.boxes[p, t] = [x0 + t - 20, 80, x0 + t + 120, 420]
    st.final[:] = st.auto
    if overlap_at is not None:
        st.boxes[1, overlap_at] = st.boxes[0, overlap_at] + [30, 0, 30, 0]
    if fb_at is not None:
        st.flags[0, fb_at, :5] |= S.FLAG_FB
    if fast_at is not None:
        st.auto[0, fast_at + 1, 3, :2] += 200.0
    return st


def _cut():
    return NS(tracking=NS(conf_threshold=0.55), joint_keys={}, confirmed={}, learn=True, id="c1", name="c1",
              source="src.mov", info=NS(frames=40))


def test_overlap_takes_the_stretch_around_it_out():
    st = _store(overlap_at=20)
    ok, stat = RF.pseudo_mask(st, _cut())
    assert not ok[0, 20 - RF.REACH:20 + RF.REACH + 1].any() and not ok[1, 20].any()
    assert ok[0, 5].any() and ok[0, 35].any()
    assert stat["occluded"] > 0


def test_forward_backward_conflict_takes_its_stretch_out():
    st = _store(fb_at=15)
    ok, stat = RF.pseudo_mask(st, _cut())
    assert not ok[0, 15 - RF.REACH:15 + RF.REACH + 1].any()
    assert ok[1, 15].any()                       # the other person is not affected
    assert stat["conflict"] > 0


def test_fast_joint_is_not_an_answer():
    st = _store(fast_at=25)
    ok, stat = RF.pseudo_mask(st, _cut())
    assert not ok[0, 25, 3] and not ok[0, 27, 3]
    assert ok[0, 25, 4]
    assert stat["fast"] > 0


def test_flash_frame_found():
    frames = {t: np.full((32, 32, 3), 80, np.uint8) for t in range(10)}
    frames[5] = np.full((32, 32, 3), 200, np.uint8)
    got = RF.flash_frames(range(10), lambda t: frames[t], 10)
    assert 5 in got and 3 not in got


def test_turned_crop_keeps_its_answer():
    rng = np.random.default_rng(1)
    img = np.zeros((RF.STORE, RF.STORE, 3), np.uint8)
    t = np.array([0.2, -0.1], np.float32)
    c = np.array([RF.STORE / 2, RF.STORE / 2]) + t * RF.CROP
    cv2.circle(img, (int(round(c[0])), int(round(c[1]))), 2, (255, 255, 255), -1)
    for _ in range(5):
        out, t2 = RF.geo_aug(img, t, rng)
        q = np.array([RF.STORE / 2, RF.STORE / 2]) + t2 * RF.CROP
        ys, xs = np.nonzero(out[..., 0] > 100)
        assert np.hypot(xs.mean() - q[0], ys.mean() - q[1]) < 1.5


def test_jpeg_in_degrade_keeps_shape():
    c = np.random.default_rng(2).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    for s in range(10):
        out = RF.degrade(c, np.random.default_rng(s), 1.0)
        assert out.shape == c.shape and out.dtype == np.uint8


class _Proxy:
    sx = sy = 1.0

    def frame(self, t):
        return np.full((480, 960, 3), 90, np.uint8)


def test_collect_from_original_alone(tmp_path, monkeypatch):
    """A freshly analysed cut, no hand retouch, no corrections: samples come out, with how they were made."""
    from contour.core import settings as SE
    SE._settings = SE.AppSettings(data_dir=str(tmp_path / "data"))
    st = _store(N=40, overlap_at=None)
    prj = NS(name="MV_A", path=str(tmp_path / "MV_A"), client="", learn=True)
    n = RF.collect_auto(prj, _cut(), NS(store=st, proxy=_Proxy()))
    assert n >= 20
    f = RF.auto_shard(prj, _cut())
    with np.load(f) as d:
        meta = json.loads(str(d["meta"]))
        assert d["easy"].dtype == bool and d["easy"].any()
        assert len(d["crops"]) == n
    assert meta["label"] == RF.LABEL and meta["pseudo"] and meta["excluded"]["kept"] > 0
    SE._settings = None


def test_check_holds_out_whole_projects(tmp_path):
    from contour.core import settings as SE
    SE._settings = SE.AppSettings(data_dir=str(tmp_path / "data"))
    try:
        d = RF.samples_dir()
        for k, prj in enumerate(["A", "A", "B", "C", "D"]):
            m = dict(project=prj, source=f"s{k}.mov", cut_id=f"c{k}")
            np.savez(d / f"{prj}__c{k}__auto.npz", crops=np.zeros((5, RF.STORE, RF.STORE, 3), np.uint8),
                     joints=np.zeros(5, np.int64), targets=np.zeros((5, 2), np.float32),
                     weights=np.full(5, 0.3, np.float32), frames=np.zeros(5, np.int32), meta=np.array(json.dumps(m)))
        X, J, T, W, G, E, grouping = RF.load_dataset()
        assert grouping == "project" and len(set(G.tolist())) == 4
        assert len(set(G[:10].tolist())) == 1          # the two cuts of project A are one group
        assert not E.any()                             # shards from before have no easy flag
        *_, g2 = RF.load_dataset("source")
        assert g2 == "source"
    finally:
        SE._settings = None
