"""Pose stability learning from originals (Ballin MV instruction, 2026-11-19): the self-made answers are exact on a
made-up cut (degraded frames keep the clean track as answer; flow keeps only round-trip points on the person), left/
right swaps and jitter are counted right, and the shards load as refiner data with their record fields."""
import json
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from contour.learn import pose_robust as PR
from contour.learn import refiner as RF
from contour.track import skeleton as S
from contour.track.store import TrackStore

H, W = 360, 480


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)


def _cut(n=24, blur=None, shift=(0.0, 0.0)):
    """One person standing still (or moving by `shift` px a frame), sure everywhere."""
    st = TrackStore(n, [7])
    base = np.array([[240, 60], [230, 52], [250, 52], [222, 56], [258, 56], [210, 100], [270, 100], [200, 150],
                     [280, 150], [195, 195], [285, 195], [220, 200], [260, 200], [218, 260], [262, 260],
                     [216, 320], [264, 320]], np.float32)
    for t in range(n):
        pts = base + np.array(shift, np.float32) * t
        st.present[0, t] = True
        st.auto[0, t, :, :2] = pts
        st.auto[0, t, :, 2] = 1.0
        st.meas[0, t, :, :2] = pts + 1.5
        st.meas[0, t, :, 2] = 0.9
        st.final[0, t] = st.auto[0, t]
        st.boxes[0, t] = [pts[:, 0].min() - 20, pts[:, 1].min() - 20, pts[:, 0].max() + 20, pts[:, 1].max() + 20]
        st.scale[0, t] = 300.0
        st.auto_conf[0, t] = 0.95
    if blur is not None:
        st.blur[0] = blur
    cut = NS(tracking=NS(conf_threshold=0.55), joint_keys={}, name="c", id="cid", source="A_0006C009_x.mov")
    return st, cut


class _Proxy:
    sx = sy = 1.0
    w, h = W, H

    def __init__(self, frames):
        self.frames = frames

    def frame(self, t):
        return self.frames[t]


def test_roll_group_keeps_a_camera_roll_together():
    assert PR.roll_group("A_0006C009_261110_030227_p1C62.mxf") == "A_0006"
    assert PR.roll_group("A_0006C011_261110_031000_p1C62.mxf") == "A_0006"
    assert PR.roll_group("B_0002C001_261110.mov") == "B_0002"
    assert PR.roll_group("intro_title.mov") == "intro_title"


@pytest.mark.parametrize("kind", PR.KINDS)
def test_every_degradation_changes_the_picture_and_keeps_its_shape(kind):
    rgb = _texture()
    out, prm = PR.degrade_frame(rgb, kind, np.random.default_rng(3), 300.0)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - rgb).mean() > 1.0
    assert prm["kind"] == kind
    if kind == "mixed":
        assert 2 <= len(prm["parts"]) <= 3


def test_lr_swap_is_counted_only_when_crossing_fits_clearly():
    st, _ = _cut()
    gt = st.auto[0, 0, :, :2].copy()
    ok = np.ones(S.K, bool)
    assert PR.lr_swaps(gt, gt, 300.0, ok) == (8, 0)
    pred = gt.copy()
    pred[[S.L_WR, S.R_WR]] = gt[[S.R_WR, S.L_WR]]
    assert PR.lr_swaps(pred, gt, 300.0, ok) == (8, 1)
    near = gt.copy()
    near[S.L_EYE], near[S.R_EYE] = gt[S.R_EYE], gt[S.L_EYE]      # 20 px apart on a 300 px person: not a swap
    assert PR.lr_swaps(near, gt, 300.0, ok)[1] == 0


def test_jitter_and_jumps():
    xy = np.zeros((1, 10, S.K, 2), np.float32)
    sc = np.full((1, 10), 100.0, np.float32)
    ok = np.ones((1, 10, S.K), bool)
    j, jumps = PR.jitter(xy, sc, ok)
    assert j == 0.0 and jumps == 0.0
    xy[0, 5, :, 0] = 20.0                    # one frame 20 % of the person away and back
    j2, jumps2 = PR.jitter(xy, sc, ok)
    assert jumps2 > 0


def test_flow_carries_points_and_drops_the_ones_that_do_not_come_back():
    a = _texture(1)
    M = np.float32([[1, 0, 3], [0, 1, 2]])
    b = cv2.warpAffine(a, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    pts = np.array([[100, 100], [200, 150], [300, 200]], np.float32)
    got, ok = PR.flow_points(a, b, pts, 1.0)
    assert ok.all()
    assert np.abs(got - (pts + [3, 2])).max() < 0.5
    other = _texture(9)
    _, ok2 = PR.flow_points(a, other, pts, 1.0)
    assert ok2.sum() <= 1


class _ShiftPose:
    """A pose model that is off by (dx, dy) px wherever the picture differs from the clean one."""

    def __init__(self, clean, gt, d=(6.0, -4.0)):
        self.clean, self.gt, self.d = clean, gt, np.array(d, np.float32)

    def __call__(self, img, boxes):
        off = self.d if np.abs(img.astype(int) - self.clean).mean() > 0.5 else np.zeros(2, np.float32)
        kp = (self.gt + off)[None]
        return kp, np.ones((1, S.K), np.float32)


def test_degraded_samples_answer_with_the_clean_track():
    st, cut = _cut()
    frames = [_texture(0)] * st.n
    tr = NS(store=st, proxy=_Proxy(frames), labels=lambda t: None)
    pose = _ShiftPose(frames[0], st.auto[0, 0, :, :2])
    smp, recs = PR.collect_degraded(None, cut, tr, pose, np.random.default_rng(0), frames=4, kinds_per_frame=2)
    assert recs and smp["crops"]
    side = RF.SIDE_FRAC * 300.0
    tg = np.array(smp["targets"])
    assert np.allclose(tg, -np.array([6.0, -4.0]) / side, atol=1e-5)    # back to the clean track
    for r in recs:
        assert np.allclose(r["clean"], st.auto[0, 0, :, :2])             # the model agrees on the clean frame
        assert r["ok"].sum() >= 8
    e = PR.record_errors(np.stack([r["gt"] for r in recs]), np.stack([r["pred"] for r in recs]),
                         np.stack([r["ok"] for r in recs]), np.array([r["scale"] for r in recs]))
    assert e["err_mean"] == pytest.approx(np.hypot(6, 4) / 300.0, rel=1e-4)
    assert e["swaps"] == 0


def test_flow_samples_only_on_hard_frames_of_the_same_shot():
    blur = np.ones(24, np.float32)
    blur[11:14] = 0.4                                         # three blurred frames in the middle
    st, cut = _cut(blur=blur, shift=(2.0, 0.0))
    a = _texture(2)
    frames = [cv2.warpAffine(a, np.float32([[1, 0, 2 * t], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)
              for t in range(st.n)]
    tr = NS(store=st, proxy=_Proxy(frames), labels=lambda t: None)
    smp, recs = PR.collect_flow(None, cut, tr, np.random.default_rng(0), starts=40, steps=3)
    assert recs, "the blurred frames got carried answers"
    for r in recs:
        assert 11 <= r["t"] <= 13
        okj = r["ok"]
        assert np.abs(r["gt"][okj] - st.auto[0, r["t"], okj, :2]).max() < 1.0
    tg = np.array(smp["targets"]) * RF.SIDE_FRAC * 300.0
    assert np.abs(tg + 1.5).max() < 1.0                        # meas was 1.5 px off the track


def test_flow_stops_at_a_shot_change():
    blur = np.full(24, 0.4, np.float32)
    blur[:12] = 1.0
    st, cut = _cut(blur=blur)
    st.jump[11] = True                                        # 11 -> 12 is a cut
    frames = [_texture(0)] * st.n
    tr = NS(store=st, proxy=_Proxy(frames), labels=lambda t: None)
    _, recs = PR.collect_flow(None, cut, tr, np.random.default_rng(0), starts=40, steps=3)
    assert all(r["t"] < 12 for r in recs)


def test_shard_loads_as_refiner_data_with_its_record(tmp_path, monkeypatch):
    st, cut = _cut()
    frames = [_texture(0)] * st.n
    tr = NS(store=st, proxy=_Proxy(frames), labels=lambda t: None)
    pose = _ShiftPose(frames[0], st.auto[0, 0, :, :2])
    smp, recs = PR.collect_degraded(None, cut, tr, pose, np.random.default_rng(0), frames=3, kinds_per_frame=2)
    monkeypatch.setattr(RF, "samples_dir", lambda: tmp_path)
    n = PR.write_shard(tmp_path / "Ballin__cid__degraded.npz", smp, recs,
                       dict(project="Ballin", cut="c", cut_id="cid", source="A_0006C009_x.mov", group="A_0006",
                            label=PR.LABEL_DEGRADED, method="auto"))
    data = RF.load_dataset("source")
    assert data is not None and len(data[0]) == n
    with np.load(tmp_path / "Ballin__cid__degraded.npz") as d:
        assert d["crops"].shape[1:] == (RF.STORE, RF.STORE, 3)
        assert len(d["rec_kind"]) == len(recs)
        assert set(d["review"].tolist()) == {"unreviewed"}
        m = json.loads(str(d["meta"]))
        assert m["group"] == "A_0006" and m["label"] == PR.LABEL_DEGRADED
        e = PR.record_errors(d["rec_gt"], d["rec_pred"], d["rec_ok"], d["rec_scale"])
        assert e["records"] == len(recs)
