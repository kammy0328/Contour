"""Instruction 2026-11-20 §1-2: the person matte as one revision.

Until 1.22.2 a refresh wrote the new alpha and owners straight into the files in use, then measured the widths,
then saved; an analysis or '인물 매트 강제 갱신' overwrote the matte in use frame by frame long before its tracks were
saved. A failure, a cancel or the app closing in between left the matte of one revision with the widths (or the
whole tracking) of another; a joint moved three times by a little never made the matte again (each move was
compared with the one before); and clothing cut with the old owners went on being used after the owners changed.

- alpha, owners, widths, outlines (and the clothing of the frames whose owners change) are finished first and put
  in use together; an error while they are put in use, or the app ending there, leaves the revision before;
- an analysis or a whole matte made again is written beside what is in use; cancelled or failed, nothing in use was
  touched; put in use, the tracks go with it;
- a move is compared with what the matte was made from: 3 moves of 10 px at 4K are 30 px;
- clothing is taken out of use only in the frames whose owners changed, by a state the preview and a render read
  from the same file;
- an analysis, a matte made again and the tracking edits all leave the record the comparison needs, and a cut
  opened with tracking that no longer matches its matte is marked."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from conftest import make_video

N, W, H = 24, 64, 48
MW, MH = 32, 24


def _project(tmp_path, height=None):
    """An analysed cut with two people, a SAM matte (alpha 255 and owners on two boxes), widths and a record of
    its meta, as an analysis leaves it."""
    from contour.core import media as M
    from contour.core.project import Person, Project
    from contour.track.store import TrackStore
    v = tmp_path / "v.mp4"
    make_video(v, n=N, w=W, h=H)
    prj = Project.create(str(tmp_path), "mr")
    cut = prj.add_cut(M.probe(str(v)), "c1")
    cut.analyzed = True
    cut.analysis_version = 3
    cut.persons = [Person(id=0, name="A", color="#ff8800", role="target"),
                   Person(id=1, name="B", color="#00aaff", role="target")]
    st = TrackStore(N, [0, 1])
    st.has_vis = st.has_face = True
    st.present[:] = True
    base = np.array([[32, 6], [33, 5], [31, 5], [34, 6], [30, 6], [36, 12], [28, 12], [38, 20], [26, 20],
                     [39, 27], [25, 27], [35, 28], [29, 28], [35, 37], [29, 37], [35, 45], [29, 45]], np.float32)
    for p, dx in ((0, -14.0), (1, 14.0)):
        for t in range(N):
            st.final[p, t, :, :2] = base + [dx, 0]
            st.final[p, t, :, 2] = 0.9
            st.boxes[p, t] = [24 + dx, 3, 41 + dx, 46]
    st.scale[:] = 44.0
    st.conf[:] = 0.9
    st.widths[:] = 5.0
    st.contour_raw[:] = 3.0
    st.contour[:] = 3.0
    st.has_contour = True
    st.save(prj.cut_dir(cut) / "tracks.npz")
    cache = prj.cache_dir(cut)
    mm = np.memmap(cache / f"matte_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    lm = np.memmap(cache / f"seglab_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    mm[:, 2:23, 4:13] = 255
    lm[:, 2:23, 4:13] = 1
    mm[:, 2:23, 19:28] = 255
    lm[:, 2:23, 19:28] = 2
    mm.flush()
    lm.flush()
    del mm, lm
    (cache / "seg_meta.json").write_text(json.dumps(dict(model="SAM 3.1", made=1.0, refreshed=[], revision=4)),
                                         encoding="utf-8")
    if height:
        cut.info.height = height
    prj.save()
    return prj, cut


def _tracker(prj, cut):
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    return CutTracker(prj, cut, frames_for(prj, cut))


def _files(prj, cut):
    """Bytes of everything that belongs to a matte revision, as the disk has it."""
    cache = Path(prj.cache_dir(cut))
    out = {}
    for f in sorted(cache.glob("*.u8")) + [cache / "seg_meta.json", Path(prj.cut_dir(cut)) / "tracks.npz"]:
        out[f.name] = f.read_bytes() if f.exists() else None
    for f in sorted(cache.glob("cloth_*.json")):
        out[f.name] = f.read_bytes()
    return out


def _leftovers(prj, cut):
    cache = Path(prj.cache_dir(cut))
    names = [f.name for f in list(cache.iterdir()) + list(Path(prj.cut_dir(cut)).iterdir()) if f.is_file()]
    return sorted(n for n in names if ".stage" in n or n.endswith(".old") or ".tmp" in n
                  or n in ("matte_undo.npz", "matte_swap.json", "matte_stage.json"))


@pytest.fixture
def fake_sam(monkeypatch):
    """segment.run: alpha 200 and the owners moved two pixels to the right, in every frame it is asked for.
    fail_at: raise there (a model error); cancel_at: as if cancel had been pressed there."""
    from contour.track import segment as SG
    ctl = NS(fail_at=None, cancel_at=None, runs=[], value=200)

    def run(st, proxy, matte, seglab, a, b, key, use_final, progress=None, cancel=None, matter=None, models=None,
            **k):
        ctl.runs.append((a, b))
        mm, lm = matte.open_rw(), seglab.open_rw()
        try:
            for t in range(a, b):
                if ctl.cancel_at is not None and t >= ctl.cancel_at:
                    from contour.track.analyze import Cancelled
                    raise Cancelled()
                if ctl.fail_at is not None and t >= ctl.fail_at:
                    raise RuntimeError("model failed")
                al = np.zeros((MH, MW), np.uint8)
                lb = np.zeros((MH, MW), np.uint8)
                al[2:23, 6:15] = ctl.value
                lb[2:23, 6:15] = 1
                al[2:23, 21:30] = ctl.value
                lb[2:23, 21:30] = 2
                mm[t] = al
                lm[t] = lb
            mm.flush()
            lm.flush()
        finally:
            del mm, lm
            matte.close()
            seglab.close()
        if models is not None:
            models["used"] = "SAM 3.1"
        return {}
    monkeypatch.setattr(SG, "run", run)
    monkeypatch.setattr(SG, "model_for", lambda acc: "sam", raising=False)
    import contour.track.infer as IN
    monkeypatch.setattr(IN, "Matter", lambda *a, **k: None, raising=False)
    return ctl


def _refresh(tr, ranges=((8, 12),)):
    from contour.track import matte_refresh as MR
    return MR.compute(tr, [list(r) for r in ranges], models={"matter": None})


# ---------------------------------------------------------------------------------------------- a refresh
def test_refresh_is_finished_before_anything_is_written_then_put_in_use_together(tmp_path, fake_sam):
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    before = _files(prj, cut)
    w0, c0 = tr.store.widths.copy(), tr.store.contour.copy()
    res = _refresh(tr)
    a, b = res["ranges"][0]
    assert (a, b) == (8 - MR.MARGIN, 12 + MR.MARGIN)
    # computed and measured, nothing in use changed: files, widths, outlines
    assert _files(prj, cut) == before and np.array_equal(tr.store.widths, w0, equal_nan=True)
    stg = res["stage"]
    assert set(stg["widths"]["persons"]) == {0, 1} and stg["owners_changed"] == list(range(a, b))
    assert not np.array_equal(stg["widths"]["widths"][0][a:b], w0[0, a:b], equal_nan=True), "measured on the new matte"
    MR.commit(tr, res)
    assert MRV.revision(tr) == 5 and tr.seg_meta()["refreshed"][-1][:2] == [a, b]
    al = np.array(np.memmap(tr.matte.path, np.uint8, "r", shape=(N, MH, MW)))
    lb = np.array(np.memmap(tr.seglab.path, np.uint8, "r", shape=(N, MH, MW)))
    assert al[a:b].max() == 200 and (al[:a].max(), al[b:].max()) == (255, 255)
    assert lb[a:b, 5, 14].tolist() == [1] * (b - a) and lb[a - 1, 5, 14] == 0
    for p in (0, 1):
        assert np.array_equal(tr.store.widths[p], stg["widths"]["widths"][p], equal_nan=True)
        assert np.array_equal(tr.store.contour[p], stg["widths"]["contour"][p], equal_nan=True)
    assert tr.store.matte_rev == 5 and not np.array_equal(tr.store.contour, c0, equal_nan=True)
    # the tracks on disk carry the same revision; nothing of the transaction is left
    from contour.track.store import TrackStore
    assert TrackStore.load(tr.path).matte_rev == 5 and _leftovers(prj, cut) == []
    rec = MRV.load_prompts(tr)
    assert rec is not None and rec["known"][a:b].all()


@pytest.mark.parametrize("where", ["frames", "widths", "save", "meta"])
def test_failure_in_the_middle_of_saving_leaves_the_revision_before(tmp_path, fake_sam, monkeypatch, where):
    mp = pytest.MonkeyPatch()
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    from contour.track.store import TrackStore
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    cut.status = dict(matte_dirty=[[8, 12]])
    before = _files(prj, cut)
    w0, c0, r0 = tr.store.widths.copy(), tr.store.contour.copy(), tr.store.contour_raw.copy()
    res = _refresh(tr)
    if where == "frames":
        # the owners cannot be written after the alpha of the same frames went in
        orig = tr.seglab.open_rw
        calls = {"n": 0}

        class Half:
            def __init__(self, mm):
                self.mm = mm

            def __setitem__(self, t, v):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise OSError("disk error while writing owners")
                self.mm[t] = v

            def flush(self):
                self.mm.flush()
        mp.setattr(tr.seglab, "open_rw", lambda: Half(orig()))
    elif where == "widths":
        real_fd, once = MRV._forget_derived, {"n": 0}

        def after_frames(*a, **k):
            once["n"] += 1
            if once["n"] == 1:
                raise RuntimeError("after the frames went in")
            return real_fd(*a, **k)
        mp.setattr(MRV, "_forget_derived", after_frames)
    elif where == "save":
        mp.setattr(tr, "save", lambda: (_ for _ in ()).throw(OSError("no space left")))
    else:
        real = tr.write_seg_meta

        def meta(model="", ranges=None, full=False, revision=None):
            if revision is not None:
                raise OSError("cannot write seg_meta")
            return real(model=model, ranges=ranges, full=full)
        mp.setattr(tr, "write_seg_meta", meta)
    try:
        with pytest.raises((OSError, RuntimeError)):
            MR.commit(tr, res)
    finally:
        mp.undo()
    # what the disk has is what it had: alpha, owners, tracks (with their widths), seg_meta and its revision
    now = _files(prj, cut)
    for k in before:
        if k == "tracks.npz":
            a_, b_ = TrackStore.load(tr.path), tr.store
            assert a_.matte_rev == 0 and np.array_equal(a_.widths, w0, equal_nan=True)
            assert np.array_equal(a_.contour, c0, equal_nan=True)
        else:
            assert now[k] == before[k], f"{k} changed"
    assert MRV.revision(tr) == 4 and _leftovers(prj, cut) == []
    # and in memory
    assert np.array_equal(tr.store.widths, w0, equal_nan=True) and np.array_equal(tr.store.contour, c0, equal_nan=True)
    assert np.array_equal(tr.store.contour_raw, r0, equal_nan=True) and tr.store.matte_rev == 0
    assert int(tr.matte.get(9).max()) == 255 and cut.status["matte_dirty"] == [[8, 12]]
    # the same refresh goes through afterwards
    MR.commit(tr, _refresh(tr))
    assert MRV.revision(tr) == 5 and int(tr.matte.get(9).max()) == 200


@pytest.mark.parametrize("stop", ["after_frames", "after_save", "after_meta"])
def test_app_ending_while_a_refresh_is_put_in_use(tmp_path, fake_sam, monkeypatch, stop):
    """Nothing gets to put the old data back (the process is gone): the next opening of the cut does."""
    mp = pytest.MonkeyPatch()
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    before = _files(prj, cut)
    w0 = tr.store.widths.copy()
    res = _refresh(tr)

    class Killed(BaseException):
        pass
    mp.setattr(MRV, "_undo_frames", lambda *a, **k: None)          # the process does not live to undo
    if stop == "after_frames":
        mp.setattr(MRV, "_forget_derived", lambda *a, **k: (_ for _ in ()).throw(Killed()))
    elif stop == "after_save":
        def gone(model="", ranges=None, full=False, revision=None):
            raise Killed()
        mp.setattr(tr, "write_seg_meta", gone)
    else:
        mp.setattr(MRV, "save_prompts", lambda *a, **k: (_ for _ in ()).throw(Killed()))
    try:
        with pytest.raises(Killed):
            MR.commit(tr, res)
    finally:
        mp.undo()
    assert (Path(prj.cache_dir(cut)) / MRV.UNDO).exists()
    if stop != "after_meta":
        assert _files(prj, cut) != before, "the disk is between two revisions"
    tr2 = _tracker(prj, cut)                                               # the cut opened again
    assert _leftovers(prj, cut) == []
    if stop == "after_meta":
        # seg_meta was written: the new revision is the one in use, only the undo file was left
        assert tr2.recovered == "cleaned" and MRV.revision(tr2) == 5 and int(tr2.matte.get(9).max()) == 200
        assert tr2.store.matte_rev == 5
    else:
        assert tr2.recovered == "rolled_back" and MRV.revision(tr2) == 4
        now = _files(prj, cut)
        assert all(now[k] == before[k] for k in before if k != "tracks.npz")
        assert int(tr2.matte.get(9).max()) == 255 and int(tr2.seglab.get(9)[5, 14]) == 0
        assert np.array_equal(tr2.store.widths, w0, equal_nan=True) and tr2.store.matte_rev == 0
        assert _tracker(prj, cut).recovered == "", "once"


def test_undo_file_of_another_running_app_is_left_alone(tmp_path, fake_sam, monkeypatch):
    mp = pytest.MonkeyPatch()
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    cache = Path(prj.cache_dir(cut))
    np.savez_compressed(cache / MRV.UNDO, rev=np.array(5), base=np.array(4), pid=np.array(os.getppid()),
                        ts=np.array([3], np.int64), cts=np.array([], np.int64),
                        matte=np.zeros((1, MH, MW), np.uint8), labels=np.zeros((1, MH, MW), np.uint8))
    tr = _tracker(prj, cut)
    assert tr.recovered == "" and (cache / MRV.UNDO).exists() and int(tr.matte.get(3).max()) == 255
    mp.setattr(MRV, "_alive", lambda pid: pid == os.getpid())      # that app is gone now
    try:
        tr = _tracker(prj, cut)
    finally:
        mp.undo()
    assert tr.recovered == "rolled_back" and int(tr.matte.get(3).max()) == 0


# ---------------------------------------------------------------------------------------------- small moves add up
def test_three_moves_of_ten_pixels_at_4k_make_the_matte_again(tmp_path, fake_sam):
    """4K: the limit is 1 % of 2160 = 21.6 px. Each move alone is under it; against what the matte was made from
    the third is 30 px."""
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    from contour.track import skeleton as S
    prj, cut = _project(tmp_path, height=2160)
    tr = _tracker(prj, cut)
    assert MRV.ensure_prompts(tr), "a cut matted by an older version: the tracking now is taken as its source"
    marks = []
    for i in range(3):
        before = MR.prompt_state(tr)
        tr.store.final[0, 10:15, S.L_WR, 0] += 10.0
        marks.append(MR.changed(tr, before, (6, 19)))
        # what 1.22.2 compared with: the tracking just before this edit
        assert MR._differs(MR._people(before, 12), MR._people(MR.prompt_state(tr), 12), 21.6) is False
    assert marks == [[], [], [[10, 15]]]
    assert MR.stale(tr) == [[10, 15]], "told without any edit saying so (a cut opened again)"
    # matted again: that is what it is made from now, and the count starts over
    MR.commit(tr, _refresh(tr, [(10, 15)]))
    assert MR.stale(tr) == []
    before = MR.prompt_state(tr)
    tr.store.final[0, 10:15, S.L_WR, 0] += 10.0
    assert MR.changed(tr, before) == []
    tr.store.final[0, 10:15, S.L_WR, 0] += 12.0
    assert MR.changed(tr, MR.prompt_state(tr)) == [[10, 15]]


def test_frames_nothing_is_recorded_for_fall_back_on_the_edit(tmp_path, fake_sam):
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    from contour.track import skeleton as S
    prj, cut = _project(tmp_path)
    cut.status = dict(matte_dirty=[[4, 8]])            # still to be matted: not a source for comparison
    tr = _tracker(prj, cut)
    assert MR.stale(tr) == [] and MRV.load_prompts(tr) is None
    assert MRV.ensure_prompts(tr)
    rec = MRV.load_prompts(tr)
    assert not rec["known"][4:8].any() and rec["known"][:4].all() and rec["known"][8:].all()
    before = MR.prompt_state(tr)
    tr.store.final[1, 5, S.R_WR, 1] += 30.0
    assert MR.changed(tr, before) == [[5, 6]]
    assert MR.stale(tr) == [], "nothing recorded there: only an edit can say"


def test_a_deleted_person_does_not_count_as_a_change(tmp_path, fake_sam):
    from contour.track import matte_refresh as MR
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    MRV.ensure_prompts(tr)
    tr.delete_person(1)
    assert tr.store.pids == [0] and MR.stale(tr) == []
    assert _tracker(prj, cut).store.pids == [0] and MR.stale(_tracker(prj, cut)) == []


# ---------------------------------------------------------------------------------------------- the whole matte
def test_matte_made_again_is_written_beside_the_one_in_use(tmp_path, fake_sam):
    from contour.track import matte_rev as MRV
    from contour.track.analyze import Cancelled
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    cut.status = dict(matte_dirty=[[3, 6]])
    before = _files(prj, cut)
    w0 = tr.store.widths.copy()
    seen = {}
    real = fake_sam.runs

    # cancelled half way: the files in use were never opened for writing
    fake_sam.cancel_at = 12
    with pytest.raises(Cancelled):
        tr.rebuild_matte()
    assert _files(prj, cut) == before and _leftovers(prj, cut) == []
    assert np.array_equal(tr.store.widths, w0, equal_nan=True) and int(tr.matte.get(2).max()) == 255
    assert not tr.matte.staged and cut.status["matte_dirty"] == [[3, 6]]
    # a model error: the same
    fake_sam.cancel_at, fake_sam.fail_at = None, 20
    tr.cut.tracking.seg = "sam2"
    import contour.track.analyze as AZ
    with pytest.raises(Exception):
        # (the RVM fallback has no model here either)
        tr.rebuild_matte()
    assert _files(prj, cut)["seg_meta.json"] == before["seg_meta.json"]
    assert _files(prj, cut)[f"matte_{MW}x{MH}.u8"] == before[f"matte_{MW}x{MH}.u8"] and _leftovers(prj, cut) == []
    # done: matte, owners, widths and the tracks' revision together; the record of what it was made from; marks off
    fake_sam.fail_at = None
    tr.rebuild_matte()
    assert MRV.revision(tr) == 5 and tr.seg_meta()["range"] == [0, N] and tr.seg_meta()["model"] == "SAM 3.1"
    assert int(tr.matte.get(2).max()) == 200 and int(tr.seglab.get(2)[5, 14]) == 1
    from contour.track.store import TrackStore
    on_disk = TrackStore.load(tr.path)
    assert on_disk.matte_rev == 5 and np.array_equal(on_disk.widths, tr.store.widths, equal_nan=True)
    assert not np.array_equal(tr.store.widths, w0, equal_nan=True)
    assert "matte_dirty" not in cut.status and _leftovers(prj, cut) == []
    assert MRV.load_prompts(tr)["known"].all()


def test_whole_matte_that_cannot_be_moved_in_leaves_the_one_before(tmp_path, fake_sam, monkeypatch):
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    before = _files(prj, cut)
    w0 = tr.store.widths.copy()
    real = MRV._move_in
    n = {"k": 0}

    def second_fails(live, new):
        n["k"] += 1
        if n["k"] == 2:
            raise OSError("owners cannot be moved in")
        return real(live, new)
    monkeypatch.setattr(MRV, "_move_in", second_fails)
    with pytest.raises(MRV.ActivationError):
        tr.rebuild_matte()
    assert n["k"] == 2, "the alpha had been moved in"
    assert _files(prj, cut) == before and _leftovers(prj, cut) == []
    assert np.array_equal(tr.store.widths, w0, equal_nan=True) and tr.store.matte_rev == 0
    assert int(tr.matte.get(2).max()) == 255 and not tr.matte.staged


def test_matte_file_someone_has_mapped_is_replaced_in_place(tmp_path, fake_sam):
    """Windows: a file another reader has mapped (a render of the same cut) cannot be renamed. The new content is
    written into it instead, the old content kept beside it until the revision is in use (or put back from
    there)."""
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    before = _files(prj, cut)
    inode = os.stat(tr.matte.live_path).st_ino
    real = os.replace

    def locked(src, dst):
        s_, d_ = str(src), str(dst)
        if (s_.endswith(".u8") and d_.endswith(".u8.old")) or (s_.endswith(".u8.old") and d_.endswith(".u8")):
            raise PermissionError(32, "being used by another process")
        return real(src, dst)
    mp = pytest.MonkeyPatch()
    mp.setattr(MRV, "_retry", lambda fn, tries=40, wait=0.05: fn())         # (no 2 s of retries in the test)
    mp.setattr(os, "replace", locked)
    try:
        tr.rebuild_matte()
        assert os.stat(tr.matte.live_path).st_ino == inode, "the same file, new content"
        assert MRV.revision(tr) == 5 and int(tr.matte.get(2).max()) == 200 and int(tr.seglab.get(2)[5, 14]) == 1
        assert _leftovers(prj, cut) == []
        # and back the same way when the revision cannot be put in use
        after = _files(prj, cut)
        real_meta = tr.write_seg_meta

        def meta(model="", ranges=None, full=False, revision=None):
            if revision is not None:
                raise OSError("cannot write seg_meta")
            return real_meta(model=model, ranges=ranges, full=full)
        mp.setattr(tr, "write_seg_meta", meta)
        fake_sam.value = 120                                             # another matte this time
        with pytest.raises(MRV.ActivationError):
            tr.rebuild_matte()
        now = _files(prj, cut)
        assert int(tr.matte.get(2).max()) == 200, "the content before, written back into the same file"
        assert all(now[k] == after[k] for k in after if k != "tracks.npz") and _leftovers(prj, cut) == []
        assert os.stat(tr.matte.live_path).st_ino == inode and MRV.revision(tr) == 5
    finally:
        mp.undo()
    assert before != after


def test_app_ending_while_a_whole_matte_is_moved_in(tmp_path, fake_sam, monkeypatch):
    mp = pytest.MonkeyPatch()
    from contour.track import matte_rev as MRV
    from contour.track.store import TrackStore
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    before = _files(prj, cut)

    class Killed(BaseException):
        pass
    real = MRV._move_in
    n = {"k": 0}

    def killed_on_second(live, new):
        n["k"] += 1
        if n["k"] == 2:
            raise Killed()
        return real(live, new)
    mp.setattr(MRV, "_move_in", killed_on_second)
    mp.setattr(MRV, "_roll_back_swap", lambda *a, **k: None)       # nobody left to move it back
    try:
        with pytest.raises(Killed):
            tr.rebuild_matte()
    finally:
        mp.undo()
    cache = Path(prj.cache_dir(cut))
    assert (cache / MRV.SWAP).exists() and (cache / f"matte_{MW}x{MH}.u8.old").exists()
    MRV._ACTIVE.clear()                                                    # (a new process knows of none)
    tr2 = _tracker(prj, cut)
    assert tr2.recovered == "rolled_back" and _files(prj, cut) == before and _leftovers(prj, cut) == []
    assert TrackStore.load(tr2.path).matte_rev == 0 and MRV.revision(tr2) == 4


def test_stage_files_of_an_analysis_that_is_gone_are_removed_and_a_running_one_is_not_touched(tmp_path, fake_sam,
                                                                                               monkeypatch):
    from contour.track import matte_rev as MRV
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    MRV.begin(tr, copy=True)
    assert tr.matte.staged and tr.matte.path.exists() and tr.matte.path != tr.matte.live_path
    stage = tr.matte.path
    # another tracker of the same cut made meanwhile (a thumbnail, a render): the stage is someone's work
    other = _tracker(prj, cut)
    assert stage.exists() and other.recovered == "" and int(other.matte.get(2).max()) == 255
    assert not other.matte.staged, "it reads the matte in use"
    # the process that was writing it is gone
    MRV._ACTIVE.clear()
    monkeypatch.setattr(MRV, "_alive", lambda pid: False)
    again = _tracker(prj, cut)
    assert again.recovered == "cleaned" and not stage.exists() and _leftovers(prj, cut) == []


def test_analysis_cancelled_or_failed_keeps_tracks_people_and_matte(tmp_path, fake_sam, monkeypatch):
    """The analysis itself needs the models; what is checked here is that everything it writes goes beside what is
    in use, and that a cancel after the new tracks and people were set puts all of it back."""
    from contour.core.project import Person
    from contour.track import facelayer as FL
    from contour.track import matte_rev as MRV
    from contour.track.analyze import Cancelled, CutTracker
    from contour.track.store import TrackStore
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    (Path(prj.cache_dir(cut)) / "facelayer.npz").write_bytes(b"old face layer")
    before = _files(prj, cut)
    old_store, old_people = tr.store, [p.id for p in cut.persons]

    def body(self, progress=None, cancel=None, accuracy="high", finish=True):
        a, b = self.cut.work_range
        assert self.matte.staged and FL.path_for(self).name == "facelayer.npz", "read: the layer in use"
        st = TrackStore(N, [0, 1, 7])
        st.present[:] = True
        st.final[:] = old_store.final[0, 0]
        st.boxes[:] = [10, 3, 30, 46]
        self._run_matte(a, b, lambda *x: None, st=st, accuracy=accuracy, cancel=cancel)
        self.store = st
        self.cut.persons = self.cut.persons + [Person(id=7, name="new", color="#fff", role="target")]
        out = FL.path_for(self, write=True)
        assert out.name == "facelayer.stage.npz"
        out.write_bytes(b"new face layer")
        assert FL.path_for(self) == out, "read from the stage once it is written"
        self.cut.analyzed, self.cut.analysis_version = True, self.cut.analysis_version + 1
        if not finish:
            raise Cancelled()
        MRV.commit(self, (a, b))
        return st
    monkeypatch.setattr(CutTracker, "_analyze", lambda self, p=None, c=None, acc="high": body(self, p, c, acc, False))
    with pytest.raises(Cancelled):
        tr.analyze()
    assert _files(prj, cut) == before and _leftovers(prj, cut) == []
    assert (Path(prj.cache_dir(cut)) / "facelayer.npz").read_bytes() == b"old face layer"
    assert tr.store is old_store and [p.id for p in cut.persons] == old_people and cut.analysis_version == 3
    assert not tr.matte.staged and FL.path_for(tr).name == "facelayer.npz"
    # finished: tracks, matte, owners and face layer of the new analysis, one revision
    monkeypatch.setattr(CutTracker, "_analyze", lambda self, p=None, c=None, acc="high": body(self, p, c, acc, True))
    st = tr.analyze()
    assert tr.store is st and TrackStore.load(tr.path).pids == [0, 1, 7] and MRV.revision(tr) == 5
    assert TrackStore.load(tr.path).matte_rev == 5 and int(tr.matte.get(2).max()) == 200
    assert (Path(prj.cache_dir(cut)) / "facelayer.npz").read_bytes() == b"new face layer"
    assert [p.id for p in cut.persons] == [0, 1, 7] and _leftovers(prj, cut) == []
    assert MRV.load_prompts(tr)["known"].all() and MRV.load_prompts(tr)["present"].shape[0] == 3


# ---------------------------------------------------------------------------------------------- clothing
def _with_cloth(prj, cut, tr):
    """Clothing parsed for both people: class 5 (a top, stored + 1) on the pixels each owns."""
    from contour.track import clothing as CL
    mm = tr.cloth.create()
    lab = np.array(np.memmap(tr.seglab.path, np.uint8, "r", shape=(N, MH, MW)))
    mm[:] = np.where(lab > 0, 5, 0).astype(np.uint8)
    mm.flush()
    del mm
    tr.cloth.close()
    CL._write_state(tr, [0, 1], CL.cloth_model(), 1.0)
    cut.params_for(0).cloth = "keep:top"
    return CL


@pytest.fixture
def fake_garments(monkeypatch):
    """The clothing model: class 4 (a top) on every pixel a present person owns. fail: it cannot be loaded."""
    from contour.track import clothing as CL
    ctl = NS(fail=False, frames=[])

    def maker(tracker, models=None):
        if ctl.fail:
            raise RuntimeError("no clothing model")

        def make(pres, t, frame, lab, rgb):
            ctl.frames.append(int(t))
            for p in pres:
                frame[lab == p] = 5
            return frame
        return make
    monkeypatch.setattr(CL, "_frame_maker", maker)
    monkeypatch.setattr(CL, "info", lambda tracker: dict(state="current", model="m", label="m"))
    return ctl


def test_clothing_is_made_again_only_where_the_owners_changed(tmp_path, fake_sam, fake_garments, monkeypatch):
    from contour.track import matte_refresh as MR
    from contour.track import segment as SG
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    CL = _with_cloth(prj, cut, tr)
    cloth0 = tr.cloth.path.read_bytes()
    run = SG.run

    def some_frames_same(st, proxy, matte, seglab, a, b, *x, **k):
        """The new matte gives other owners in frames 9 and 10 only (elsewhere exactly the ones in use)."""
        run(st, proxy, matte, seglab, a, b, *x, **k)
        lm = seglab.open_rw()
        for t in range(a, b):
            if t not in (9, 10):
                lm[t] = np.array(tr.seglab.get(t))
        return {}
    monkeypatch.setattr(SG, "run", some_frames_same)
    res = _refresh(tr, [(8, 12)])
    assert res["stage"]["owners_changed"] == [9, 10] and fake_garments.frames == [9, 10]
    assert tr.cloth.path.read_bytes() == cloth0, "nothing written while it is computed"
    MR.commit(tr, res)
    cl = np.array(np.memmap(tr.cloth.path, np.uint8, "r", shape=(N, MH, MW)))
    old = np.frombuffer(cloth0, np.uint8).reshape(N, MH, MW)
    for t in range(N):
        if t in (9, 10):
            assert cl[t, 5, 14] == 5 and cl[t, 5, 5] == 5, "cut with the new owners"
        else:
            assert np.array_equal(cl[t], old[t]), f"frame {t} was not the matte's business"
    assert CL.stale_ranges(tr) == [] and CL.keep_mask(tr, 9, 0, "keep:top") is not None


def test_clothing_that_cannot_be_made_again_is_out_of_use_there_for_preview_and_render_alike(tmp_path, fake_sam,
                                                                                            fake_garments):
    from contour.track import matte_refresh as MR
    from contour.warp import field as F
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    CL = _with_cloth(prj, cut, tr)
    fake_garments.fail = True
    res = _refresh(tr, [(8, 12)])
    a, b = res["ranges"][0]
    assert res["stage"]["cloth"]["stale"] == list(range(a, b)) and res["stage"]["cloth"]["frames"] == {}
    MR.commit(tr, res)
    assert CL.stale_ranges(tr) == [[a, b]]
    pp = cut.params_for(0)
    # the session's tracker (the preview) and one made for a render read the same file
    render = _tracker(prj, cut)
    for who in (tr, render):
        assert CL.stale_ranges(who) == [[a, b]] and CL.info(who) is not None
        assert F._cloth_keep(who, a, 0, pp) is None and F._cloth_keep(who, b - 1, 0, pp) is None
        assert F._cloth_keep(who, a - 1, 0, pp) is not None and F._cloth_keep(who, b, 0, pp) is not None
        assert CL.ready_for(who, 0)
    # made again there: in use again, for both
    fake_garments.fail = False
    fake_garments.frames.clear()
    assert CL.refresh_stale(tr) == [[a, b]] and fake_garments.frames == list(range(a, b))
    for who in (tr, render):
        assert CL.stale_ranges(who) == [] and F._cloth_keep(who, a, 0, pp) is not None
    assert int(np.asarray(tr.cloth.get(a))[5, 14]) == 5


def test_clothing_and_its_state_come_back_with_the_matte_when_putting_in_use_fails(tmp_path, fake_sam,
                                                                                  fake_garments):
    mp = pytest.MonkeyPatch()
    from contour.track import matte_refresh as MR
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    CL = _with_cloth(prj, cut, tr)
    before = _files(prj, cut)
    res = _refresh(tr, [(8, 12)])
    assert res["stage"]["cloth"]["frames"]
    mp.setattr(tr, "save", lambda: (_ for _ in ()).throw(OSError("no space left")))
    try:
        with pytest.raises(OSError):
            MR.commit(tr, res)
    finally:
        mp.undo()
    now = _files(prj, cut)
    assert all(now[k] == before[k] for k in before if k != "tracks.npz") and CL.stale_ranges(tr) == []
    assert _leftovers(prj, cut) == []


def test_clothing_state_of_an_older_version_and_a_whole_run(tmp_path, fake_sam, fake_garments):
    """1.22.2 only showed the refreshed ranges in the inspector and went on using the masks there."""
    from contour.track import clothing as CL0
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    CL = _with_cloth(prj, cut, tr)
    s = json.loads(CL.state_text(tr))
    s.pop("stale")
    s["made"] = 100.0
    CL.put_state_text(tr, json.dumps(s))
    tr.write_seg_meta(model="SAM 3.1", ranges=[[5, 9]])                    # refreshed after the clothes were cut
    assert CL.stale_ranges(tr) == [[5, 9]] and CL.keep_mask(tr, 6, 0, "keep:top") is None
    assert CL.keep_mask(tr, 12, 0, "keep:top") is not None
    CL._write_state(tr, [0, 1], CL.cloth_model(), 1.0)                     # the whole cut parsed again
    assert CL.stale_ranges(tr) == [] and CL.keep_mask(tr, 6, 0, "keep:top") is not None


# ---------------------------------------------------------------------------------------------- the window
def test_opening_a_cut_whose_tracking_left_its_matte_marks_it_and_keeps_the_playback_cache(qapp, tmp_path,
                                                                                         fake_sam, monkeypatch):
    """A cut saved with a joint moved but without its mark (the app ended before the mark was saved): opened, the
    frames are marked from what is recorded with the matte. Looking alone changes nothing a frame is made from:
    the cut's kept frames stay good."""
    from PySide6.QtWidgets import QMessageBox
    from contour.core.settings import settings
    from contour.track import matte_rev as MRV
    from contour.track import skeleton as S
    from contour.ui import main_window as MW
    from contour.ui.matte_refresh import pending
    settings().ask_proxy = False
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No))
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    assert MRV.ensure_prompts(tr)
    tr.store.final[0, 10:14, S.L_WR, 0] += 30.0
    tr.save()
    w = MW.MainWindow()
    try:
        monkeypatch.setattr(w, "show_error", lambda title, text: None)
        w.refresher.enabled = True
        w.refresher.timer.setInterval(3600_000)                    # marks only: no refresh run in this test
        w.cache_worker.enabled = False
        w.open_project(prj.path, wait=True)
        t0 = time.time()
        while not pending(w.session.cut) and time.time() - t0 < 10:
            qapp.processEvents()
            time.sleep(0.01)
        assert pending(w.session.cut) == [[10, 14]]
        # a cut that matches its matte: opened, nothing is marked and nothing is touched
        vers = dict(w.session.vers)
        w.refresher.kick()
        w.edit_worker.drain(10)
        for _ in range(20):
            qapp.processEvents()
        assert w.session.vers == vers, "comparing is not a change"
    finally:
        w.refresher.cancel()
        w.proxy_builder.cancel()
        w.thumbcache.stop()
        w.session.dirty = False
        w._discard_on_close = True
        w.close()


def test_refresh_that_cannot_be_put_in_use_says_so_and_keeps_the_marks(qapp, tmp_path, fake_sam, monkeypatch):
    from PySide6.QtCore import QObject
    from contour.track import matte_refresh as MR
    from contour.ui.matte_refresh import MatteRefresher, pending
    from contour.ui.workers import SerialWorker
    prj, cut = _project(tmp_path)
    tr = _tracker(prj, cut)
    toasts = []
    win = QObject()
    win.session = NS(cut=cut, tracker=tr, project=prj, dirty=False)
    win.job = None
    win.batch = NS(busy=False)
    win.edit_worker = SerialWorker()
    win.toast = lambda text, kind="": toasts.append(text)
    r = MatteRefresher(win, delay_ms=10)
    got = []
    win.edit_worker.done.connect(lambda tag, res: (got.append(res), r.committed(res)) if tag[0] == "matte" else None)
    monkeypatch.setattr(tr, "save", lambda: (_ for _ in ()).throw(OSError("no space left")))
    try:
        r.mark(cut, 8, 12)
        t0 = time.time()
        while not got and time.time() - t0 < 15:
            qapp.processEvents()
            time.sleep(0.01)
        assert got and got[0]["error"] and pending(cut) == [[8, 12]]
        assert any("이전 매트를 그대로" in t for t in toasts)
        assert int(tr.matte.get(9).max()) == 255
        qapp.processEvents()
        assert not r.timer.isActive(), "not tried again by itself every second and a half"
    finally:
        r.cancel()
        win.edit_worker.stop()
        win.edit_worker.wait(3000)
