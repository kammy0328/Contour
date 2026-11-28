"""(11/28) The render's analysis saved per frame (retouch/prepcache.py) and a render that takes it from there: the same planes as a render that
prepares every frame itself; a folder for other values or another range is not used; a frame missing from it is prepared as ever."""
import os
import tempfile
from pathlib import Path

import numpy as np

import baseline_util as BU
from contour.core import perf
from contour.retouch import engine as E
from contour.retouch import prepcache as PC


def _project():
    from contour.core.project import Project
    pdir, src = BU.copy_fixture(Path(__file__).resolve().parent / "data" / "baseline_1222", Path(tempfile.mkdtemp()))
    return Project.load(str(pdir))


def _save(prj, cut, root):
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    BU.steady_rng()
    fr = frames_for(prj, cut)
    tr = CutTracker(prj, cut, fr)
    try:
        a, b = cut.work_range
        return PC.save_cut(cut, tr, a, b, root, tag="t1")
    finally:
        fr.close()


def test_masks_are_kept_to_the_bit_in_their_own_type():
    rng = np.random.default_rng(3)
    m32 = np.zeros((90, 120), np.float32)
    m32[20:50, 30:70] = rng.random((30, 40), dtype=np.float32)
    m16 = np.zeros((90, 120), np.float16)
    m16[5:9, 100:110] = 0.25
    j = E.Job(p=0, pid=3, v={"skin": 0.4}, keep=0.5, scope="face", grain_size=1.0, face=None, face_w=100.0, scale=1.0,
              masks={"skin": m32, "dark": m16, "none": np.zeros((90, 120), np.float32)}, det={"box": (1, 2, 3, 4)})
    rf = E.RetouchFrame(t=7, jobs=[j], proxy_wh=(120, 90), src_wh=(240, 180))
    back = PC._unpack(PC._pack(rf))
    assert back.t == 7 and back.jobs[0].det == {"box": (1, 2, 3, 4)} and back.jobs[0].v == {"skin": 0.4}
    for k, v in rf.jobs[0].masks.items():
        w = back.jobs[0].masks[k]
        assert w.dtype == v.dtype and w.shape == v.shape and np.array_equal(w, v), k
    assert rf.jobs[0].masks["skin"] is m32                       # the frame packed is not changed
    assert PC._unpack(PC._pack(None)) is None


import pytest


@pytest.mark.parametrize("threads", ["1", "3"])
def test_a_render_from_the_saved_analysis_is_the_same_render(monkeypatch, threads):
    monkeypatch.setenv("CONTOUR_RENDER_THREADS", threads)        # the render's one-frame-at-a-time loop and its frames side by side
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    root = tempfile.mkdtemp()
    prj = _project()
    cut = prj.cuts[0]
    with BU.NoModels():
        base = BU.render_planes(prj, cut)
        r = _save(prj, cut, root)
        assert r["complete"] and r["frames"] == cut.work_range[1] - cut.work_range[0]
        called = []
        prep = E.prepare
        monkeypatch.setattr(E, "prepare", lambda *a, **k: (called.append(a[2]), prep(*a, **k))[1])
        monkeypatch.setenv("CONTOUR_PREP_CACHE", tempfile.mkdtemp() + ";" + root)      # (the second of two folders has it)
        monkeypatch.setenv("CONTOUR_PREP_CACHE_MODE", "read")
        cached = BU.render_planes(prj, cut)
        assert called == []                                       # nothing prepared again
        assert np.array_equal(base, cached)
        # a frame missing from the folder: prepared as ever, the same render
        d = PC.folder(root, cut, *cut.work_range)
        os.remove(os.path.join(d, "t%05d.pz" % (cut.work_range[0] + 2)))
        again = BU.render_planes(prj, cut)
        assert called and np.array_equal(base, again)


def test_a_folder_for_other_values_or_another_range_is_not_used(monkeypatch):
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    root = tempfile.mkdtemp()
    prj = _project()
    cut = prj.cuts[0]
    with BU.NoModels():
        _save(prj, cut, root)
    a, b = cut.work_range
    r, why = PC.Reader.open(root, cut, a, b)
    assert r is not None and why == ""
    assert PC.Reader.open(root, cut, a, b, tag="t2")[0] is None                       # another code's analysis
    assert PC.Reader.open(root, cut, a, b - 1)[0] is None                              # another range
    pid = cut.targets()[0].id
    from contour.core.project import Curve
    cut.params_for(pid).amounts["skin"] = Curve(base=cut.params_for(pid).amount("skin", a) + 7.0)
    r, why = PC.Reader.open(root, cut, a, b)
    assert r is None and "no saved" in why                                             # (another folder: the values are in its name)
    w = []
    monkeypatch.setenv("CONTOUR_PREP_CACHE", root)
    monkeypatch.setenv("CONTOUR_PREP_CACHE_MODE", "read")
    assert PC.reader_for_render(cut, a, b, w) is None and w and "no saved" in w[0]
    monkeypatch.setenv("CONTOUR_PREP_CACHE_MODE", "")
    assert PC.reader_for_render(cut, a, b, w) is None


def test_a_folder_being_written_is_not_read(monkeypatch):
    root = tempfile.mkdtemp()
    prj = _project()
    cut = prj.cuts[0]
    a, b = cut.work_range
    w = PC.Writer(PC.folder(root, cut, a, b), PC._meta(cut, a, b, ""))
    w.put(a, None)
    assert PC.Reader.open(root, cut, a, b)[0] is None              # no meta.json until it is closed
    w.close(ok=False)
    assert PC.Reader.open(root, cut, a, b)[0] is None              # an unfinished save stays unread


def test_check_grade():
    a = np.full((40, 40, 3), 120, np.uint8)
    b = a.copy()
    b[:10, :10, 0] = 140                                           # the lips
    m = np.zeros((40, 40), np.float32)
    m[20:, 20:] = 1.0                                              # the skin
    assert PC.check_grade(a, b, m)["p99"] == 0.0
    assert PC.check_grade(a, b, None)["p99"] == 20.0


@pytest.mark.parametrize("threads", ["1", "3"])
def test_a_render_that_saves_its_analysis_gives_the_same_file_and_a_folder_a_later_render_reads(monkeypatch, threads):
    monkeypatch.setenv("CONTOUR_RENDER_THREADS", threads)
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    monkeypatch.setattr(PC, "MIN_FREE_GB", 0.0)
    root = tempfile.mkdtemp()
    prj = _project()
    cut = prj.cuts[0]
    a, b = cut.work_range
    with BU.NoModels():
        base = BU.render_planes(prj, cut)
        monkeypatch.setenv("CONTOUR_PREP_CACHE", root)
        monkeypatch.setenv("CONTOUR_PREP_CACHE_MODE", "write")
        saving = BU.render_planes(prj, cut)
        assert np.array_equal(base, saving)
        r, why = PC.Reader.open(root, cut, a, b)
        assert r is not None, why
        called = []
        prep = E.prepare
        monkeypatch.setattr(E, "prepare", lambda *a_, **k: (called.append(a_[2]), prep(*a_, **k))[1])
        monkeypatch.setenv("CONTOUR_PREP_CACHE_MODE", "read")
        cached = BU.render_planes(prj, cut)
        assert called == [] and np.array_equal(base, cached)


def test_a_cancelled_render_leaves_no_folder_to_read(monkeypatch):
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    monkeypatch.setattr(PC, "MIN_FREE_GB", 0.0)
    root = tempfile.mkdtemp()
    prj = _project()
    cut = prj.cuts[0]
    a, b = cut.work_range
    d = PC.folder(root, cut, a, b)
    w = PC.Writer(d, PC._meta(cut, a, b, ""))
    w.put(a, None)
    w.close(ok=True)                                               # one frame of several: not complete
    assert PC.Reader.open(root, cut, a, b)[0] is None


def test_the_app_setting_saves_in_the_cut_folder_and_the_next_render_reads_it(monkeypatch):
    """(1.28) 설정 > 렌더할 때 분석 저장: the first render saves, the next takes the saved analysis (nothing prepared, the same planes); other
    values are prepared again (and saved anew); a saved folder whose pictures were another shot's is not read."""
    from contour.core.settings import settings
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    monkeypatch.setattr(PC, "MIN_FREE_GB", 0.0)
    monkeypatch.delenv("CONTOUR_PREP_CACHE", raising=False)
    monkeypatch.delenv("CONTOUR_PREP_CACHE_MODE", raising=False)
    monkeypatch.setattr(settings(), "keep_analysis", True)
    prj = _project()
    cut = prj.cuts[0]
    a, b = cut.work_range
    with BU.NoModels():
        base = BU.render_planes(prj, cut)                     # saves
        d = PC.folder(str(prj.cut_dir(cut) / "prep_cache"), cut, a, b)
        assert os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(os.path.join(d, "look.npz"))
        called = []
        prep = E.prepare
        monkeypatch.setattr(E, "prepare", lambda *a_, **k: (called.append(a_[2]), prep(*a_, **k))[1])
        again = BU.render_planes(prj, cut)
        assert called == [] and np.array_equal(base, again)
        # another shot's pictures: not the same picture
        saved = np.load(os.path.join(d, "look.npz"))
        rng = np.random.default_rng(1)
        other = (rng.standard_normal(saved["outline"].shape).astype(np.float32), saved["pic"])
        ok, why = PC.same_picture((saved["outline"], saved["pic"]), other)
        assert not ok and "다른 화면" in why
        brighter = (saved["outline"], np.clip(saved["pic"].astype(np.int16) + 30, 0, 255).astype(np.uint8))
        ok, why = PC.same_picture((saved["outline"], saved["pic"]), brighter)
        assert not ok and "색" in why
        assert PC.same_picture((saved["outline"], saved["pic"]), (saved["outline"], saved["pic"]))[0]


def test_the_app_setting_off_does_nothing(monkeypatch):
    from contour.core.settings import settings
    monkeypatch.delenv("CONTOUR_PREP_CACHE", raising=False)
    monkeypatch.setattr(settings(), "keep_analysis", False)
    prj = _project()
    cut = prj.cuts[0]
    assert PC.for_project_render(prj, cut, *cut.work_range) == (None, None)
    assert not (prj.cut_dir(cut) / "prep_cache").exists()


def test_a_changed_analysis_is_not_read(monkeypatch):
    from contour.core.settings import settings
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    monkeypatch.setattr(PC, "MIN_FREE_GB", 0.0)
    monkeypatch.delenv("CONTOUR_PREP_CACHE", raising=False)
    monkeypatch.setattr(settings(), "keep_analysis", True)
    prj = _project()
    cut = prj.cuts[0]
    a, b = cut.work_range
    with BU.NoModels():
        BU.render_planes(prj, cut)
    r, w = PC.for_project_render(prj, cut, a, b)
    assert r is not None and w is None
    tp = prj.cut_dir(cut) / "tracks.npz"
    if tp.exists():
        os.utime(tp, ns=(tp.stat().st_atime_ns, tp.stat().st_mtime_ns + 10 ** 9))      # the analysis saved again (tracks edited)
        warn = []
        r, w = PC.for_project_render(prj, cut, a, b, warn)
        assert r is None and w is not None and "분석이 바뀜" in warn[0]
        w.close(ok=False)
