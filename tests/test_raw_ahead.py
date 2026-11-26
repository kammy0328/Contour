"""1.27: the next frames' own masks made ahead of the render's serial stage in threads (retouch/rawahead.py): a render gets
the frames it would have made itself, the helper asks only inside the render's range, a frame it cannot make is made by the
render, and which computers use it (core/perf.raw_ahead)."""
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

import baseline_util as BU
from contour.core import perf
from contour.retouch import engine as E
from contour.retouch import rawahead as RA


def _alive():
    return [th for th in threading.enumerate() if th.name.startswith("raw-ahead") and th.is_alive()]


def test_which_computers(monkeypatch):
    monkeypatch.delenv("CONTOUR_RAW_AHEAD", raising=False)
    assert perf.raw_ahead(4, ram=31, cores=18) == (3, 2)
    # one thread ahead made the render slower (j2895c: 9.98 s a frame against 9.41 without): no tier with one
    assert perf.raw_ahead(4, ram=31, cores=8) == (0, 0)
    assert perf.raw_ahead(4, ram=24, cores=18) == (0, 0) and perf.raw_ahead(4, ram=31, cores=6) == (0, 0)
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", "4:3")
    assert perf.raw_ahead(4, ram=8, cores=2) == (4, 3)
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", "2")
    assert perf.raw_ahead(4, ram=8, cores=2) == (2, 2)
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", "0")
    assert perf.raw_ahead(4, ram=64, cores=32) == (0, 0)
    monkeypatch.delenv("CONTOUR_RAW_AHEAD")
    from contour.core.settings import settings
    settings().render_parallel = "off"
    assert perf.raw_ahead(4, ram=64, cores=32) == (0, 0)


def test_frames_are_asked_ahead_inside_the_range_and_taken_once(monkeypatch):
    made = []
    lock = threading.Lock()

    def raw(cut, tracker, s, cache=None, only=None):
        with lock:
            made.append(s)
        if s == 7:
            raise RuntimeError("no picture")
        return types.SimpleNamespace(t=s, jobs=[types.SimpleNamespace(masks={"skin": np.full((4, 4), s, np.float32)})])
    monkeypatch.setattr(E, "_prepare_raw", raw)
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    ra = RA.RawAhead("cut", "tracker", 2, 9, ahead=2, threads=2)
    assert ra.take(4) == (False, None)                          # the first frame asked was not made ahead: the caller makes it
    ok, rf = ra.take(5)
    assert ok and rf.t == 5 and float(E._mask_unpack(rf.jobs[0].masks["skin"])[0, 0]) == 5.0     # kept as _raw_cached keeps its own
    assert ra.take(3) == (False, None)                          # a frame behind: not made ahead, nothing asked twice
    assert ra.take(6)[0] and ra.take(7) == (False, None)        # 7 failed here: the render makes it and meets the error itself
    assert ra.take(8)[0]
    assert ra.take(9) == (False, None)                          # past the range: never asked
    st = ra.close()
    assert sorted(made) == [5, 6, 7, 8] and st["hit"] == 3 and st["failed"] == 1
    assert ra.take(8) == (False, None) and not _alive()         # closed: the caller is on its own


def test_nothing_is_started_ahead_while_the_computer_is_short_of_memory(monkeypatch):
    """What the other programs leave is not what is installed (the 31 GB PC with 16 GB in use: a render with frames
    made ahead took it to 30.7 GB, j2895b): below RAW_AHEAD_MIN_FREE_GB the render makes its frames itself."""
    made, free = [], [1.5]
    monkeypatch.setattr(E, "_prepare_raw", lambda cut, tr, s, cache=None, only=None: (made.append(s), types.SimpleNamespace(t=s, jobs=[]))[1])
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: free[0])
    ra = RA.RawAhead("cut", "tracker", 0, 20, ahead=2, threads=1)
    assert ra.take(0) == (False, None) and ra.take(1) == (False, None) and ra.take(2) == (False, None)
    assert made == [] and ra.held == 3
    free[0] = 9.0                                               # memory again: ahead from here
    assert ra.take(3) == (False, None)
    assert ra.take(4)[0] and ra.take(5)[0]
    free[0] = -1.0                                              # cannot be read: as the installed memory said
    assert ra.take(6)[0]
    monkeypatch.setattr(perf, "RAW_AHEAD_MIN_FREE_GB", 0.0)     # 0: never held
    free[0] = 0.1
    assert ra.take(7)[0]
    st = ra.close()
    assert st["held"] == 3 and st["hit"] == 4 and not _alive()


def test_memory_now_is_read(monkeypatch):
    total, avail = perf._memory()
    assert total > 0.5 and (avail == -1.0 or 0 < avail <= total)
    perf._FREE[1] = -1.0
    a = perf.free_gb()
    assert a == avail or abs(a - avail) < 1.0
    monkeypatch.setattr(perf, "_memory", lambda: (8.0, 0.25))
    assert perf.free_gb(max_age=60.0) == a                      # asked a moment ago: not again
    assert perf.free_gb(max_age=0.0) == 0.25
    perf._FREE[1] = -1.0


def _render(monkeypatch, ahead: str):
    from contour.core.project import Project
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", ahead)
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: 64.0)
    stats = []
    stop = RA.stop_for_render
    monkeypatch.setattr(RA, "stop_for_render", lambda tr: (stats.append(stop(tr)) or stats[-1]))
    pdir, src = BU.copy_fixture(Path(__file__).resolve().parent / "data" / "baseline_1222", Path(tempfile.mkdtemp()))
    prj = Project.load(str(pdir))
    with BU.NoModels():
        out = BU.render_planes(prj, prj.cuts[0])
    return out, [s for s in stats if s]


def test_a_render_with_frames_made_ahead_is_the_same_render(monkeypatch):
    base, s0 = _render(monkeypatch, "0")
    ahead, s1 = _render(monkeypatch, "3:2")
    assert s0 == [] and s1 and s1[0]["hit"] >= 3 and s1[0]["failed"] == 0
    assert np.array_equal(base, ahead)
    assert not _alive()


# ------------------------------------------------------------------------------------------------ short of memory
def _render_threads(monkeypatch, threads: str, free):
    """The fixture rendered with `threads` frames at once; free(): the computer's free memory in GB as the render sees it."""
    from contour.core.project import Project
    from contour.warp import render as RD
    monkeypatch.setenv("CONTOUR_RENDER_THREADS", threads)
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", "0")
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: free())
    reps = []
    inner = RD._render_cut
    monkeypatch.setattr(RD, "_render_cut", lambda *a, **k: (reps.append(inner(*a, **k)) or reps[-1]))
    pdir, src = BU.copy_fixture(Path(__file__).resolve().parent / "data" / "baseline_1222", Path(tempfile.mkdtemp()))
    prj = Project.load(str(pdir))
    with BU.NoModels():
        out = BU.render_planes(prj, prj.cuts[0])
    return out, reps[-1]


def test_short_of_memory_the_render_finishes_its_frames_before_it_starts_another(monkeypatch):
    """Four frames at once filled the 31 GB PC on a cut of six people (j2896a). With little memory free a frame is started
    only when the ones before it are written: the same file, told in the render's summary."""
    one, r1 = _render_threads(monkeypatch, "1", lambda: 64.0)
    many, r3 = _render_threads(monkeypatch, "3", lambda: 64.0)
    assert np.array_equal(one, many) and "기다림" not in r3.accel and "프레임 3개 동시 처리" in r3.accel
    short, rs = _render_threads(monkeypatch, "3", lambda: 0.5)
    assert np.array_equal(one, short)
    assert "메모리가 부족해" in rs.accel and "기다림" in rs.accel
    # cannot be read, or switched off: never held
    for free, env in ((lambda: -1.0, None), (lambda: 0.5, "0")):
        if env is not None:
            monkeypatch.setenv("CONTOUR_MIN_FREE_GB", env)
        out, r = _render_threads(monkeypatch, "3", free)
        assert np.array_equal(one, out) and "기다림" not in r.accel
    monkeypatch.delenv("CONTOUR_MIN_FREE_GB")
    assert perf.short_of_memory() is True and perf.short_of_memory(0.1) is False
