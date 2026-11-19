"""Instruction 2026-11-19 §4: the person matte follows the edits by itself. An edit marks only the frames whose
matte prompts it changed (a swap of two ids moves nobody), edits close together are matted again in one run, the
frames outside are kept, alpha and owners are written together from one run, and a run that a cut switch, a
cancel or a newer edit has outdated is never written (not even when it was already queued for writing). The old
matte stays in the preview meanwhile; the marks stay with the cut until written; a render asks first. Before
1.22.2 resolve_local() changed the tracking but the matte and owners stayed as the analysis made them until the
whole cut was matted again by hand ('인물 매트만 다시 만들기')."""
import json
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PySide6.QtCore import QObject

from contour.core.proxy import MatteStore
from contour.track import matte_refresh as MR
from contour.track import segment as SG
from contour.track.store import TrackStore

N, MW, MH = 40, 16, 12
OLD = 7


class FT:
    """A tracker with real matte and owner stores (old value 7 everywhere) and a track store."""

    def __init__(self, tmp, cut):
        self.cut = cut
        st = TrackStore(N, [1, 2])
        st.present[:] = True
        for p in range(2):
            for t in range(N):
                st.final[p, t, :, :2] = [[100 + 300 * p + j, 200 + 10 * j] for j in range(17)]
                st.final[p, t, :, 2] = 0.9
                st.boxes[p, t] = [80 + 300 * p, 150, 260 + 300 * p, 900]
        self.store = st
        self.proxy = NS(w=MW * 2, h=MH * 2, sx=1.0, sy=1.0, n=N)
        self.matte = MatteStore(tmp, N, MW, MH)
        self.seglab = MatteStore(tmp, N, MW, MH, prefix="seglab")
        for s in (self.matte, self.seglab):
            mm = s.create()
            mm[:] = OLD
            mm.flush()
            del mm
        self.meta = {}
        self.widths = []
        self.saved = 0
        self._label_cache = {}

    def has_seg(self):
        return self.seglab.exists()

    def invalidate_labels(self, frames=None):
        pass

    def update_widths(self, persons=None, rng=None):
        self.widths.append(tuple(rng))

    def seg_meta(self):
        return self.meta

    def write_seg_meta(self, model="", ranges=None, full=False):
        self.meta.setdefault("refreshed", []).extend([[a, b, time.time()] for a, b in ranges or []])
        self.meta["model"] = model

    def save(self):
        self.saved += 1


def _cut(cid="c1"):
    return NS(id=cid, status={}, work_range=(0, N), info=NS(height=1080), name=cid)


@pytest.fixture
def fake_sam(monkeypatch):
    """segment.run that writes, for each frame, the first joint's x of person 1 at that moment as both the alpha
    and the owner (so a frame tells which tracking it was made from); it can be held on a gate."""
    ctl = NS(gate=None, runs=[], entered=threading.Event())

    def run(st, proxy, matte, seglab, a, b, key, use_final, progress=None, cancel=None, matter=None, models=None,
            **k):
        ctl.runs.append((a, b))
        ctl.entered.set()
        mm, lm = matte.open_rw(), seglab.open_rw()
        for t in range(a, b):
            while ctl.gate is not None and not ctl.gate.wait(0.01):
                if cancel and cancel():
                    from contour.track.analyze import Cancelled
                    raise Cancelled()
            if cancel and cancel():
                from contour.track.analyze import Cancelled
                raise Cancelled()
            v = int(st.final[0, t, 0, 0]) % 250
            mm[t] = np.full((MH, MW), v, np.uint8)
            lm[t] = np.full((MH, MW), v, np.uint8)
        if models is not None:
            models["used"] = "SAM 3.1"
        return {}
    monkeypatch.setattr(SG, "run", run)
    return ctl


# ---------------------------------------------------------------------- which frames
def test_only_frames_whose_prompts_moved_are_marked(tmp_path):
    tr = FT(tmp_path, _cut())
    before = MR.prompt_state(tr)
    tr.store.final[0, 10:15, 3, 0] += 30                  # a corrected wrist, 30 px
    tr.store.final[1, 30, 5, 1] += 1                      # a 1 px wobble: the matte would not change
    assert MR.changed(tr, before) == [[10, 15]]
    before = MR.prompt_state(tr)
    tr.store.vis[1, 22, 4] = 2                            # a joint marked hidden: its point leaves the prompts
    assert MR.changed(tr, before) == [[22, 23]]
    # two ids swapped (or split off): the owners move with them, nobody moves, nothing to matte again
    before = MR.prompt_state(tr)
    tr.store.swap_persons(1, 2, 5, 25)
    assert MR.changed(tr, before) == []
    assert MR.changed(tr, before, (0, 3)) == []


def test_merge_and_subtract():
    assert MR.merge([[10, 20], [24, 30], [50, 52]]) == [[10, 30], [50, 52]]
    assert MR.subtract([[0, 40]], [[10, 20]]) == [[0, 10], [20, 40]]
    assert MR.subtract([[10, 20]], [[7, 23]]) == []


# ---------------------------------------------------------------------- written: range only, alpha + owners
def test_frames_outside_are_kept_and_alpha_and_owners_come_from_one_run(tmp_path, fake_sam):
    tr = FT(tmp_path, _cut())
    tr.store.final[0, :, 0, 0] = 120
    res = MR.compute(tr, [[10, 15]], models={"matter": None})
    assert res["ranges"] == [[10 - MR.MARGIN, 15 + MR.MARGIN]] and res["model"] == "SAM 3.1"
    # nothing written yet: the preview reads the old matte while it is computed
    assert int(tr.matte.get(12)[0, 0]) == OLD and int(np.asarray(tr.seglab.open_rw()[12])[0, 0]) == OLD
    MR.commit(tr, res)
    al = np.asarray(np.memmap(tr.matte.path, np.uint8, "r", shape=(N, MH, MW)))
    lb = np.asarray(np.memmap(tr.seglab.path, np.uint8, "r", shape=(N, MH, MW)))
    inside = range(7, 18)
    assert all(al[t].max() == 120 and lb[t].max() == 120 for t in inside)
    assert all(al[t].max() == OLD and lb[t].max() == OLD for t in range(N) if t not in inside)
    assert tr.widths == [(7, 18)] and tr.saved == 1
    assert tr.meta["model"] == "SAM 3.1" and tr.meta["refreshed"][0][:2] == [7, 18]


def test_nothing_to_refresh_without_a_sam_matte(tmp_path, fake_sam):
    tr = FT(tmp_path, _cut())
    tr.seglab.remove()
    assert MR.compute(tr, [[0, 5]], models={"matter": None}) is None and not fake_sam.runs


# ---------------------------------------------------------------------- the refresher
class Win(QObject):
    def __init__(self, cut, tr, delay=20):
        super().__init__()
        from contour.ui.matte_refresh import MatteRefresher
        from contour.ui.workers import SerialWorker
        self.cuts = {cut.id: cut}
        self.session = NS(cut=cut, tracker=tr, project=NS(cut=lambda cid: self.cuts.get(cid)), dirty=False,
                          touch=lambda *a: None)
        self.edit_worker = SerialWorker(self)
        self.edit_worker.done.connect(self._edit_done)
        self.job = None
        self.batch = NS(busy=False)
        self._opening = None
        self.toasts = []
        self.ref = MatteRefresher(self, delay_ms=delay)
        self.ref.models["matter"] = None

    def toast(self, text, kind=""):
        self.toasts.append(text)

    def _edit_done(self, tag, res):
        if tag[0] == "matte":
            self.ref.committed(res)

    def stop(self):
        self.ref.cancel()
        self.edit_worker.stop()
        self.edit_worker.wait(2000)


def _until(qapp, cond, timeout=5.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    return cond()


def _alpha(tr, t):
    return int(np.asarray(np.memmap(tr.matte.path, np.uint8, "r", shape=(N, MH, MW)))[t].max())


def _owner(tr, t):
    return int(np.asarray(np.memmap(tr.seglab.path, np.uint8, "r", shape=(N, MH, MW)))[t].max())


def test_close_edits_are_one_run_and_marks_clear_when_written(qapp, tmp_path, fake_sam):
    cut = _cut()
    tr = FT(tmp_path, cut)
    w = Win(cut, tr)
    try:
        tr.store.final[0, :, 0, 0] = 130
        w.ref.mark(cut, 10, 20)
        w.ref.mark(cut, 24, 30)                          # a second edit close by, before the first ran
        assert cut.status["matte_dirty"] == [[10, 30]] and w.ref.state(cut) == "waiting"
        assert _until(qapp, lambda: not cut.status.get("matte_dirty"))
        assert fake_sam.runs == [(7, 33)], "one run for both edits"
        assert _alpha(tr, 20) == 130 and _owner(tr, 20) == 130 and _alpha(tr, 40 - 1) == OLD
        assert w.ref.state(cut) == ""
    finally:
        w.stop()


def test_cut_switch_cancel_and_newer_edit_drop_the_run(qapp, tmp_path, fake_sam):
    cut = _cut()
    tr = FT(tmp_path, cut)
    w = Win(cut, tr)
    try:
        # 1) the user goes to another cut while the refresh runs: dropped, marks stay with the cut
        fake_sam.gate = threading.Event()
        tr.store.final[0, :, 0, 0] = 140
        w.ref.mark(cut, 10, 12)
        assert _until(qapp, lambda: fake_sam.entered.is_set())
        assert w.ref.state(cut) == "running"
        other = _cut("c2")
        w.cuts["c2"] = other
        w.ref.cancel()
        w.session.cut, w.session.tracker = other, None
        fake_sam.gate.set()
        assert _until(qapp, lambda: w.ref.running is None)
        time.sleep(0.1)
        qapp.processEvents()
        assert _alpha(tr, 11) == OLD and _owner(tr, 11) == OLD
        assert cut.status["matte_dirty"] == [[10, 12]]
        # back on the cut: its marks are taken up again
        w.session.cut, w.session.tracker = cut, tr
        fake_sam.gate = None
        fake_sam.entered.clear()
        w.ref.kick()
        assert _until(qapp, lambda: not cut.status.get("matte_dirty"))
        assert _alpha(tr, 11) == 140

        # 2) a newer edit while it runs: the run made from the older tracking is never written; the next run
        #    covers both edits, from the newer tracking
        fake_sam.gate = threading.Event()
        fake_sam.entered.clear()
        tr.store.final[0, :, 0, 0] = 150
        w.ref.mark(cut, 20, 22)
        assert _until(qapp, lambda: fake_sam.entered.is_set())
        tr.store.final[0, :, 0, 0] = 160
        w.ref.mark(cut, 21, 26)
        fake_sam.gate.set()
        fake_sam.gate = None
        assert _until(qapp, lambda: not cut.status.get("matte_dirty"))
        assert all(_alpha(tr, t) == 160 and _owner(tr, t) == 160 for t in range(20, 26))
        assert 150 not in {_alpha(tr, t) for t in range(N)}

        # 3) cancel (closing the project): nothing written, marks kept
        fake_sam.gate = threading.Event()
        fake_sam.entered.clear()
        tr.store.final[0, :, 0, 0] = 170
        w.ref.mark(cut, 30, 31)
        assert _until(qapp, lambda: fake_sam.entered.is_set())
        w.ref.cancel()
        fake_sam.gate.set()
        assert _until(qapp, lambda: w.ref.running is None)
        assert _alpha(tr, 30) != 170 and cut.status["matte_dirty"] == [[30, 31]]
    finally:
        w.stop()


def test_a_run_queued_for_writing_is_checked_again_on_the_worker(qapp, tmp_path, fake_sam):
    """The run finished, its write is queued behind a joint re-solve: the re-solve (an edit) outdates it, so
    neither alpha nor owners of it are written."""
    cut = _cut()
    tr = FT(tmp_path, cut)
    w = Win(cut, tr)
    try:
        tr.store.final[0, :, 0, 0] = 180
        gate = threading.Event()
        w.ref.mark(cut, 5, 8)
        w.ref.timer.stop()
        w.ref._go()                                       # the run starts (the worker is idle) ...
        w.edit_worker.submit(("block", 0), gate.wait)      # ... then the worker gets busy before its write
        assert _until(qapp, lambda: w.ref.committing == 1)
        w.ref.hold(cut)                                   # an edit is queued meanwhile
        w.ref.enabled = False
        gate.set()
        assert _until(qapp, lambda: w.ref.committing == 0)
        time.sleep(0.05)
        qapp.processEvents()
        assert _alpha(tr, 6) == OLD and _owner(tr, 6) == OLD
        assert cut.status.get("matte_dirty") == [[5, 8]], "still marked: made again by the next run"
    finally:
        w.stop()


def test_a_job_holds_the_refresh(qapp, tmp_path, fake_sam):
    cut = _cut()
    tr = FT(tmp_path, cut)
    w = Win(cut, tr)
    try:
        w.job = object()                                  # merge / render / analysis running
        w.ref.mark(cut, 1, 3)
        time.sleep(0.15)
        qapp.processEvents()
        assert not fake_sam.runs and cut.status["matte_dirty"] == [[1, 3]]
        w.job = None
        assert _until(qapp, lambda: not cut.status.get("matte_dirty"))
    finally:
        w.stop()


# ---------------------------------------------------------------------- the edits that mark
def test_resolve_task_returns_the_changed_frames(tmp_path):
    from contour.ui.main_window import resolve_task
    cut = _cut()
    tr = FT(tmp_path, cut)

    def resolve_local(pid, t, frame_end=None):
        tr.store.final[0, 12:16, 2, 0] += 25
        return (2, 30)
    tr.resolve_local = resolve_local
    s = NS(tracker=tr, touch=lambda *a: None)
    res = resolve_task(s, 1, 14)()
    assert res == dict(cut="c1", rng=(2, 30), marks=[[12, 16]])


def test_longer_work_range_mattes_tracked_frames_and_names_unanalysed_ones(tmp_path):
    cut = _cut()
    tr = FT(tmp_path, cut)
    mm = tr.matte.open_rw()
    mm[30:34] = 0                                          # tracked, never matted (outside the old range)
    mm.flush()
    del mm
    tr.matte.close()
    tr.store.present[:, 34:] = False                      # never analysed
    tr.meta["range"] = [5, 30]
    to_matte, unseen = MR.extension(tr, (5, 30), (5, 40))
    assert to_matte == [[30, 34]] and unseen == [[34, 40]]
    assert MR.extension(tr, (5, 30), (10, 30)) == ([], [])


def test_analysis_takes_the_marks_off_and_records_the_model(tmp_path):
    from contour.track.analyze import CutTracker
    t = object.__new__(CutTracker)
    t.cut = NS(status=dict(auto="good", matte_dirty=[[3, 9], [50, 60]]))
    t.project = NS(cache_dir=lambda c: tmp_path)
    t._matte_marks_done(0, 40)
    assert t.cut.status == dict(auto="good", matte_dirty=[[50, 60]])
    t._matte_marks_done(0, 100)
    assert t.cut.status == dict(auto="good")
    t.write_seg_meta(model="SAM 3.1", ranges=[[0, 40]], full=True)
    t.write_seg_meta(model="SAM 3.1", ranges=[[3, 9]])
    m = t.seg_meta()
    assert m["model"] == "SAM 3.1" and m["range"] == [0, 40] and m["refreshed"][0][:2] == [3, 9]


def test_inspector_line_names_the_model_and_the_refresh(tmp_path):
    from contour.ui.inspector import matte_line
    cut = _cut()
    cut.analyzed = True
    tr = FT(tmp_path, cut)
    tr.meta = {"model": "SAM 2.1"}
    ref = NS(state=lambda c: "waiting", last_error="")
    cut.status = {"matte_dirty": [[0, 12]]}
    line = matte_line(NS(refresher=ref), tr, cut)
    assert "SAM 2.1" in line and "12프레임 갱신 대기" in line and "SAM 2 " not in line + " "
    ref.state = lambda c: ""
    cut.status = {}
    assert matte_line(NS(refresher=ref), tr, cut) == "이 컷의 매트: SAM 2.1 · 최신"
    tr.seglab.remove()
    assert "RVM" in matte_line(NS(refresher=ref), tr, cut)
    import inspect
    from contour.ui import inspector as INS, main_window as MW
    for mod in (INS, MW):
        src = inspect.getsource(mod)
        assert "SAM 2로" not in src and "이 컷의 매트: SAM 2\"" not in src and "인물 매트만 다시 만들기" not in src
    assert "인물 매트 강제 갱신" in inspect.getsource(INS)


# ---------------------------------------------------------------------- render asks
@pytest.fixture
def win(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from contour.core.settings import settings
    from contour.ui import main_window as MW
    settings().ask_proxy = False
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    w = MW.MainWindow()
    monkeypatch.setattr(w, "show_error", lambda title, text: None)
    yield w
    w.proxy_builder.cancel()
    w.thumbcache.stop()
    w.session.dirty = False
    w._discard_on_close = True
    w.close()


def test_render_of_a_cut_waiting_for_its_matte_asks(win, qapp, tmp_path, monkeypatch):
    from conftest import make_video
    from contour.core import media as M
    from contour.core.project import Project
    from contour.ui.pages import QueueItem
    v = tmp_path / "r.mp4"
    make_video(v)
    prj = Project.create(str(tmp_path), "r")
    cut = prj.add_cut(M.probe(str(v)), "c1")
    cut.status = {"matte_dirty": [[2, 5]]}
    prj.save()
    win.open_project(prj.path, wait=True)
    it = QueueItem(id="x", project_path=prj.path, project_name="r", cut_id=cut.id, cut_name="c1", output={},
                   frames=10, fmt_label="")
    answers = []
    monkeypatch.setattr(win, "_ask_matte", lambda names, adding: answers.pop(0))
    answers.append("cancel")
    assert win.before_render(it) is None, "stopped: the item waits, not failed"
    answers.append("old")
    assert win.before_render(it) is True and it.matte == "old"
    assert win.before_render(it) is True, "asked once per item"
    # refresh first: a job, then the queue goes on
    it.matte = ""
    started = []
    monkeypatch.setattr(win.queue_page, "start_queue", lambda: started.append(1))
    answers.append("refresh")
    assert win.before_render(it) is None
    assert _until(qapp, lambda: win.job is None and started, 10)
    c = win.session.project.cut(cut.id)
    assert not (c.status or {}).get("matte_dirty")
    # the render queue: a None answer stops it without failing the item
    from contour.ui import pages as PG
    monkeypatch.setattr(win, "before_render", lambda item: None)
    win.queue_page.items = [it]
    it.status = "wait"
    PG.RenderQueuePage.start_queue(win.queue_page)
    assert it.status == "wait"
