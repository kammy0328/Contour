"""Instruction 2026-11-19 §3: '구간 다시 분석' (face and neck first). One person's selected frames are made again,
compared with what is in use on the same frames, and only applied when the user says so: then exactly that
person's range changes in the face layer (and the face points the parser aligned there); the other people, every
frame outside, the joints, ids, retouch values and keys stay. Cancel or failure writes nothing. The preview and the
render read the same file. Before 1.22.2 the face and neck masks could only be made again for the whole cut, with
no comparison before they replaced the ones in use."""
import copy
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from contour.core.project import Person
from contour.track import facelayer as FL
from contour.track import faceparse as FP
from contour.track import rangefix as RF
from contour.track.store import TrackStore
from tests.test_commercial_models import _lm

N = 30


class _Proxy:
    sx = sy = 1.0
    w, h = 512, 512
    n = N

    def frame(self, t):
        rng = np.random.default_rng(int(t))
        return (rng.random((512, 512, 3)) * 255).astype(np.uint8)


class FakeParser:
    """Skin: a disc whose size is set by `radius` (one value per run), neck: a band below; the crop is the
    picture itself. fail_at: raise on that frame."""
    radius = 40
    fail_at = None
    seen = []

    def parse(self, rgb, lm, occ=None):
        cx = float(np.mean(lm[:17, 0]))
        if FakeParser.fail_at is not None and len(FakeParser.seen) >= FakeParser.fail_at:
            raise RuntimeError("parser failed")
        FakeParser.seen.append(cx)
        K = FP.KEEP
        k = K / FP.SIZE
        sk = np.zeros((K, K), np.float32)
        cv2.circle(sk, (int(cx * k), int(200 * k)), int(FakeParser.radius), 1.0, -1)
        nk = np.zeros((K, K), np.float32)
        nk[int(300 * k):int(360 * k), int((cx - 50) * k):int((cx + 50) * k)] = 1.0
        return dict(M=np.array([[1, 0, 0], [0, 1, 0]], np.float32), skin=sk.astype(np.float16),
                    neck=nk.astype(np.float16))

    def provider(self):
        return "fake@1"


@pytest.fixture
def fake(monkeypatch):
    FakeParser.radius, FakeParser.fail_at, FakeParser.seen = 40, None, []
    monkeypatch.setattr(FP, "available", lambda: True)
    monkeypatch.setattr(FP, "make_parser", lambda: FakeParser())
    monkeypatch.setattr(FL, "ALIGN_ON", False)
    return FakeParser


def _tracker(tmp_path):
    st = TrackStore(N, [7, 9])
    st.present[:] = True
    st.has_face = True
    for p, cx in ((0, 150.0), (1, 380.0)):
        lm = _lm(cx=cx, w=180.0)
        for t in range(N):
            st.face[p, t] = lm.astype(np.float32)
            st.face_vis[p, t] = 1.0
            st.boxes[p, t] = [cx - 110, 60, cx + 110, 500]
            st.final[p, t, :, :2] = [cx, 300]
            st.scale[p, t] = 400
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cut = NS(work_range=(0, N), id="c", persons=[Person(id=7, name="A", color="#f80", role="target"),
                                                  Person(id=9, name="B", color="#0af", role="target")],
             params={7: {"waist": 10}}, joint_keys={7: {9: {3: [1.0, 2.0]}}}, analysis_version=3)
    cut.person = lambda pid: next((p for p in cut.persons if p.id == pid), None)
    tr = NS(store=st, proxy=_Proxy(), cut=cut, project=NS(cache_dir=lambda c: cache), saved=[0])
    tr._virtual_face = lambda p, t: None
    tr.save = lambda: tr.saved.__setitem__(0, tr.saved[0] + 1)
    return tr


def _arrays(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in d.files}


def test_one_persons_range_compared_then_applied_and_nothing_else_moves(tmp_path, fake):
    tr = _tracker(tmp_path)
    assert FL.build(tr) is not None                       # the layer in use (radius 40)
    path = FL.path_for(tr)
    old = _arrays(path)
    face0 = tr.store.face.copy()
    cut0 = copy.deepcopy((tr.cut.params, tr.cut.joint_keys, [p.id for p in tr.cut.persons]))
    fake.radius = 55                                      # what a new run makes
    fake.seen = []
    cand = RF.candidate(tr, 7, 10, 14)
    # the parser read a few frames each side of the range, for person 7 only
    assert len(fake.seen) == (14 + RF.MARGIN) - (10 - RF.MARGIN) and all(abs(x - 150) < 1 for x in fake.seen)
    assert cand["skin"].shape[0] == 4 and cand["model"] == "fake@1"
    assert _arrays(path)["skin"].tobytes() == old["skin"].tobytes(), "nothing written before it is applied"
    cmp = RF.compare(tr, cand)
    assert cmp["frames"] == 4 and cmp["changed"] == 4 and cmp["mean_skin"] < 0.8
    assert "10~13 프레임" in RF.effects(tr.cut, cand) and "그대로인 것" in RF.effects(tr.cut, cand)
    RF.apply(tr, cand)
    new = _arrays(path)
    p = [int(x) for x in new["pids"]].index(7)
    q = [int(x) for x in new["pids"]].index(9)
    assert np.array_equal(new["skin"][p, 10:14], cand["skin"]) and not np.array_equal(new["skin"][p, 10:14],
                                                                                  old["skin"][p, 10:14])
    for k in ("skin", "neck", "M", "rel", "have", "tex"):
        assert np.array_equal(new[k][q], old[k][q], equal_nan=k in ("M",)), f"{k}: the other person changed"
        assert np.array_equal(new[k][p, :10], old[k][p, :10], equal_nan=k == "M"), k
        assert np.array_equal(new[k][p, 14:], old[k][p, 14:], equal_nan=k == "M"), k
    assert new["ranges"].tolist()[0][:3] == [7, 10, 14]
    assert np.array_equal(tr.store.face, face0) and tr.saved[0] == 1
    assert copy.deepcopy((tr.cut.params, tr.cut.joint_keys, [p.id for p in tr.cut.persons])) == cut0
    # the preview's layer and a render's (another tracker reading the same file) are the same
    other = NS(project=tr.project, cut=tr.cut)
    a_, b_ = FL.load(tr), FL.load(other)
    m1, m2 = a_.maps(7, 11, (512, 512)), b_.maps(7, 11, (512, 512))
    assert np.array_equal(m1["skin"], m2["skin"]) and m1["skin"].sum() > 0


def test_candidate_does_not_move_the_tracking(tmp_path, fake, monkeypatch):
    """The parser's landmark alignment is kept in the candidate; the live face points move only on apply."""
    tr = _tracker(tmp_path)
    FL.build(tr)
    face0 = tr.store.face.copy()
    monkeypatch.setattr(FL, "ALIGN_ON", True)
    monkeypatch.setattr(FL, "_align", lambda r, lm: np.array([6.0, 0.0]))
    cand = RF.candidate(tr, 7, 2, 6)
    assert np.array_equal(tr.store.face, face0) and cand["moved"] == 4
    RF.apply(tr, cand)
    p = tr.store.index(7)
    assert np.allclose(tr.store.face[p, 2:6, :, 0] - face0[p, 2:6, :, 0], 6.0, atol=1e-3)
    assert np.array_equal(tr.store.face[p, :2], face0[p, :2]) and np.array_equal(tr.store.face[1], face0[1])


def test_cancel_and_failure_write_nothing(tmp_path, fake):
    tr = _tracker(tmp_path)
    FL.build(tr)
    path = FL.path_for(tr)
    before = path.read_bytes()
    calls = [0]

    def cancel():
        calls[0] += 1
        return calls[0] > 2
    fake.radius = 60
    assert RF.candidate(tr, 7, 3, 9, cancel=cancel) is None
    fake.fail_at = 2
    fake.seen = []
    with pytest.raises(RuntimeError):
        RF.candidate(tr, 7, 3, 9)
    assert path.read_bytes() == before and not list(path.parent.glob("*.tmp.npz"))


def test_a_range_on_a_cut_without_a_layer_yet(tmp_path, fake):
    tr = _tracker(tmp_path)
    cand = RF.candidate(tr, 9, 0, 3)
    assert RF.current(tr, 9, 0, 3) is None
    cmp = RF.compare(tr, cand)
    assert cmp["changed"] == 3
    RF.apply(tr, cand)
    lay = FL.load(tr)
    assert lay.have[lay.index(9), :3].all() and not lay.have[lay.index(9), 3:].any()
    assert not lay.have[lay.index(7)].any()


def test_dialog_items_and_compare_view(qapp, tmp_path, fake):
    from contour.ui.rangefix_dialog import CompareDialog, RangeDialog, compose
    tr = _tracker(tmp_path)
    FL.build(tr)
    d = RangeDialog(None, tr.cut, 7, 2, 6)
    assert d.item() == "face" and d.buttons["face"].isEnabled()
    assert not d.buttons["track"].isEnabled() and not d.buttons["body"].isEnabled()
    fake.radius = 55
    cand = RF.candidate(tr, 7, 2, 6)
    dlg = CompareDialog(None, tr, tr.cut, cand, RF.compare(tr, cand))
    both = dlg.image(3, "both")
    old, new = dlg.image(3, "old"), dlg.image(3, "new")
    assert both.shape == old.shape == new.shape and not np.array_equal(old, new)
    # red outline of the old skin, green of the new
    assert ((both[..., 0] == 235) & (both[..., 1] == 70)).any() and ((both[..., 1] == 220) & (both[..., 0] == 80)).any()
    dlg.reject()
    rgb = np.zeros((20, 20, 3), np.uint8)
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert compose(rgb, (m, None), (None, None), "new").sum() == 0
