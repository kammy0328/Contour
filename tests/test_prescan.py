"""Quick look for people before the analysis (instruction 2026-11-19 §5): three answers, only a clear 'nobody'
skips a cut. Frames that could not be read, a small or weak or one-frame person, a dark or flashing picture are
'unsure' and analysed. One detector for every cut; the cuts of a source read in their order. Before 1.22.2 a cut
whose frames could not be read was recorded as 'nobody' and never analysed."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from contour.track import prescan as PS


def F(*dets):
    return [tuple(d) for d in dets]


def test_answers():
    none = PS.judge([F()] * 12, [80.0] * 12, 12)
    assert none["verdict"] == PS.NONE and "12프레임" in none["reason"]
    assert PS.judge([F((0.9, 0.4))] * 3 + [F()] * 9, [80.0] * 12, 12)["verdict"] == PS.PRESENT
    one = PS.judge([F((0.9, 0.4))] + [F()] * 11, [80.0] * 12, 12)
    assert one["verdict"] == PS.UNSURE and one["reason"] == "한 프레임에서만 보인 인물"
    small = PS.judge([F((0.6, 0.05))] * 4 + [F()] * 8, [80.0] * 12, 12)
    assert small["verdict"] == PS.UNSURE and small["reason"] == "작은 인물"
    weak = PS.judge([F((0.3, 0.3))] + [F()] * 11, [80.0] * 12, 12)
    assert weak["verdict"] == PS.UNSURE and weak["reason"] == "낮은 검출 신뢰도"
    dark = PS.judge([F()] * 12, [8.0] * 12, 12)
    assert dark["verdict"] == PS.UNSURE and dark["reason"] == "어두운 화면"
    flash = PS.judge([F()] * 12, [60.0] * 11 + [230.0], 12)
    assert flash["verdict"] == PS.UNSURE
    unread = PS.judge([None] * 12, [], 12)
    assert unread["verdict"] == PS.UNSURE and unread["failed"] == 12
    few = PS.judge([F(), F(), None, None, None, None], [80.0, 80.0], 6)
    assert few["verdict"] == PS.UNSURE


def test_plan_short_dense_long_spread():
    assert PS.plan(9) == [0, 2, 4, 6, 8]
    assert PS.plan(10) == [0, 2, 4, 6, 8, 9]
    p = PS.plan(500)
    assert p[0] == 0 and p[-1] == 499 and 249 in p or 250 in p
    more = PS.plan(500, more=True)
    assert len(more) > 10 and not set(more) & set(p)


class _Det:
    """Finds a person where the picture's red channel says so (frame value 200: a person, 120: a small one)."""
    made = 0

    def __init__(self, *a, **k):
        _Det.made += 1
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        v = int(img[0, 0, 0])
        H = img.shape[0]
        if v == 200:
            return np.array([[10, 0, 60, 0.5 * H]], np.float32), np.array([0.9], np.float32)
        if v == 120:
            return np.array([[10, 0, 20, 0.05 * H]], np.float32), np.array([0.6], np.float32)
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)


def _frames(values):
    return lambda t: None if values[t] is None else np.full((90, 160, 3), values[t], np.uint8)


def test_a_long_cut_with_someone_between_the_first_samples_is_looked_at_closer():
    n = 300
    vals = [60] * n
    p0 = set(PS.plan(n))
    hit = [t for t in PS.plan(n, more=True)][3]
    vals[hit] = 120                                    # someone small, in a frame the first round skips
    r = PS.Prescan(detector=_Det()).look(_frames(vals), n)
    assert hit not in p0
    assert r["verdict"] == PS.UNSURE and r["looked"] > len(p0)


def test_unreadable_frames_are_not_nobody():
    r = PS.Prescan(detector=_Det()).look(_frames([None] * 40), 40)
    assert r["verdict"] == PS.UNSURE and r["reason"] == "프레임을 충분히 읽지 못함"


def test_batch_analyze_never_skips_what_it_could_not_read(monkeypatch):
    """batch.prescan_people (used inside each cut's analysis) on frames that cannot be read: analysed."""
    from contour import batch
    import contour.track.infer as IN
    monkeypatch.setattr(IN, "Detector", _Det)
    cut = NS(work_range=(0, 30))
    r = batch.prescan_people(cut, NS(frame=lambda t: None))
    assert r["verdict"] != "none"
    r2 = batch.prescan_people(cut, NS(frame=lambda t: np.full((90, 160, 3), 60, np.uint8)))
    assert r2["verdict"] == "none"


def test_scan_opens_one_detector_and_reads_each_source_in_order(monkeypatch):
    _Det.made = 0
    order = []

    class Fr:
        def __init__(self, c):
            self.c = c

        def frame(self, t):
            order.append((self.c.source, self.c.info.offset))
            return np.full((90, 160, 3), 200 if self.c.id == "b2" else 60, np.uint8)

        def close(self):
            pass
    mk = lambda cid, src, off, n: NS(id=cid, source=src, info=NS(offset=off, path=src), work_range=(0, n))
    cuts = [mk("b2", "B.mov", 100, 20), mk("a1", "A.mov", 50, 20), mk("b1", "B.mov", 0, 20), mk("a0", "A.mov", 0, 20)]
    out = PS.scan(None, cuts, prescan=PS.Prescan(detector=_Det()), frames_for=lambda p, c: Fr(c))
    assert _Det.made == 1
    assert {k: v["verdict"] for k, v in out.items()} == {"a0": "none", "a1": "none", "b1": "none", "b2": "present"}
    seq = [o for i, o in enumerate(order) if i == 0 or order[i - 1] != o]
    assert seq == [("A.mov", 0), ("A.mov", 50), ("B.mov", 0), ("B.mov", 100)]


def test_runner_skips_nobody_and_does_not_look_twice(qapp, monkeypatch):
    from contour.ui.batchrun import BatchRunner
    spawned = []

    class P:
        def __init__(self, args):
            self.args = args

        def start(self):
            pass

        def deleteLater(self):
            pass

        def processId(self):
            return 0
    r = BatchRunner()
    monkeypatch.setattr(r, "_spawn", lambda args, name: (spawned.append(list(args)), P(args))[1])
    skipped = []
    r.skipped.connect(lambda cid, info: skipped.append((cid, json.loads(info)["reason"])))
    r.start("prj", ["c1", "c2", "c3"], prescan=True)
    assert spawned[-1][:2] == ["prescan", "prj"] and spawned[-1][2:] == ["c1", "c2", "c3"]
    for cid, v in (("c1", "present"), ("c2", "none"), ("c3", "unsure")):
        r._prescan_line(f"PRESCAN {cid} {v} " + json.dumps(dict(reason="인물 없음 (빠른 검사: 10프레임 확인)"
                                                                if v == "none" else v), ensure_ascii=False))
    assert skipped == [("c2", "인물 없음 (빠른 검사: 10프레임 확인)")]
    r._finished(0, None)                       # the quick look ended: analyses follow, without looking again
    assert spawned[-1][:3] == ["analyze", "prj", "c1"] and spawned[-1][-2:] == ["--prescan", "0"]
    r.proc = None
    r.queue.clear()


def test_picked_cuts_are_not_looked_at_first(qapp, monkeypatch):
    from contour.ui.batchrun import BatchRunner
    spawned = []
    r = BatchRunner()
    monkeypatch.setattr(r, "_spawn", lambda args, name: (spawned.append(list(args)),
                                                          NS(start=lambda: None, deleteLater=lambda: None))[1])
    r.start("prj", ["c9"], prescan=False)
    assert spawned[0][:3] == ["analyze", "prj", "c9"] and spawned[0][-2:] == ["--prescan", "1"]
    r.proc = None
    r.queue.clear()


def test_a_cut_with_people_stops_looking_at_two_sure_frames():
    """1.22.2 PC check: reading every planned frame of a 4K cut cost 2.8 s a cut (422 person cuts: 1194 s against
    217 s before). Two sure frames settle 'someone is there'."""
    det = _Det()
    r = PS.Prescan(detector=det).look(_frames([200] * 300), 300)
    assert r["verdict"] == PS.PRESENT and det.calls == 2 and r["looked"] == 2
    det2 = _Det()
    r2 = PS.Prescan(detector=det2).look(_frames([60] * 300), 300)
    assert r2["verdict"] == PS.NONE and det2.calls > 13, "an empty cut is still looked at in full"
