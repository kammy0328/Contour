"""1.27: the next frames' own masks made ahead of the render's serial stage in threads (retouch/rawahead.py): a render gets
the frames it would have made itself, the helper asks only inside the render's range, a frame it cannot make is made by the
render, and which computers use it (core/perf.raw_ahead)."""
import tempfile
import threading
import time
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
class Mem:
    """The computer as a gate sees it: `free` GB free and the process holding `held` GB (both set by the test)."""

    def __init__(self, free, held=1.0):
        self.free, self.held, self.now = float(free), float(held), 0.0


def _gate(mem, workers=4, people=6, mp=8.3):
    return perf.FrameGate(workers, people, mp, free=lambda: mem.free, held=lambda: mem.held, clock=lambda: mem.now)


def test_frame_gate_gives_the_slots_the_memory_has_room_for(monkeypatch):
    monkeypatch.delenv("CONTOUR_FRAME_GATE", raising=False)
    monkeypatch.delenv("CONTOUR_RESERVE_GB", raising=False)
    # IWALY c004 on the 31 GB PC with 12.5 GB in use by other programs: 17.7 GB free at the start, six people, the whole 4K
    # frame worked on (3.5 GB a frame, 6.4 GB between frames, 3 GB left to the computer): two frames at once
    g = _gate(Mem(free=17.7))
    assert g.on and g.slots == 1                                # one, until a frame has said what kind they are
    g.hint(True)
    assert g.slots == 2 and abs(g.per - 0.42 * 8.3) < 1e-6
    g.hint(False)                                               # said once
    assert g.slots == 2
    # the same computer, faces only (the people's pieces): four
    g = _gate(Mem(free=17.7), people=4)
    g.hint(False)
    assert g.slots == 4
    # plenty of memory: as many as the render's workers; hardly any: one
    for free, want in ((64.0, 4), (9.0, 1), (0.5, 1)):
        g = _gate(Mem(free=free))
        g.hint(True)
        assert g.slots == want
    # an 8K frame with a body effect on the 31 GB PC: one at a time
    g = _gate(Mem(free=20.0), mp=33.2)
    g.hint(True)
    assert g.slots == 1
    # memory that cannot be read: as many as asked for
    g = perf.FrameGate(4, 4, 8.3, free=lambda: -1.0, held=lambda: -1.0)
    g.hint(True)
    assert g.slots == 4


def test_frame_gate_lets_that_many_retouch_and_one_more_prepare(monkeypatch):
    monkeypatch.delenv("CONTOUR_FRAME_GATE", raising=False)
    m = Mem(free=17.7)
    g = _gate(m)
    order, holds = [], [threading.Event() for _ in range(4)]

    def frame(i):
        with g.frame():
            order.append(("frame", i))
            g.hint(True)
            with g.apply():
                order.append(("apply", i))
                holds[i].wait(5.0)
        order.append(("out", i))
    ths = [threading.Thread(target=frame, args=(i,)) for i in range(4)]
    for th in ths:
        th.start()
    t0 = time.time()
    while (len([o for o in order if o[0] == "frame"]) < 3 or len([o for o in order if o[0] == "apply"]) < 2) and time.time() - t0 < 5:
        time.sleep(0.01)
    time.sleep(0.7)
    assert len([o for o in order if o[0] == "apply"]) == 2      # two retouched at once (the slots)
    assert len([o for o in order if o[0] == "frame"]) == 3      # a third prepared meanwhile, the fourth not started
    assert g.applies == 2 and g.frames == 3 and g.most == 2 and g.waited == 1
    for h in holds:
        h.set()
    for th in ths:
        th.join(5.0)
    assert g.applies == 0 and g.frames == 0 and len([o for o in order if o[0] == "out"]) == 4
    assert "한 번에 2프레임까지" in g.summary()
    g.close()


def test_frame_gate_lets_the_waiting_frames_go_in_their_own_order(monkeypatch):
    """A frame that waited was passed by the ones that came after it, again and again (IWALY c004 with two slots, j2907b: prepared
    eight places late, its neighbours' masks gone from the table of kept frames and made a second time: 29 made for 20)."""
    monkeypatch.delenv("CONTOUR_FRAME_GATE", raising=False)
    g = _gate(Mem(free=12.0))
    g.hint(True)
    assert g.slots == 1
    order = []
    held = g.apply(0)
    held.__enter__()                                            # the slot is taken

    def one(t):
        with g.apply(t):
            order.append(t)
            time.sleep(0.02)
    ths = []
    for t in (7, 5, 6, 3):                                      # they come in another order than their own
        ths.append(threading.Thread(target=one, args=(t,)))
        ths[-1].start()
        t0 = time.time()
        while len(g._line["applies"]) < len(ths) and time.time() - t0 < 5:
            time.sleep(0.005)
    assert g.applies == 1 and order == [] and g.waited == 4
    late = threading.Thread(target=one, args=(4,))             # one that comes while the others wait takes its own place, not the first
    late.start()
    t0 = time.time()
    while len(g._line["applies"]) < 5 and time.time() - t0 < 5:
        time.sleep(0.005)
    held.__exit__(None, None, None)
    for th in ths + [late]:
        th.join(5.0)
    assert order == [3, 4, 5, 6, 7] and g.applies == 0 and g._line["applies"] == [] and g.most == 1
    # without a number: in the order they came
    order.clear()
    held = g.frame()
    h2 = g.frame()
    held.__enter__()
    h2.__enter__()                                              # (a slot and one more: both in)

    def two(i):
        with g.frame():
            order.append(i)
    ths = []
    for i in range(3):
        ths.append(threading.Thread(target=two, args=(i,)))
        ths[-1].start()
        t0 = time.time()
        while len(g._line["frames"]) < len(ths) and time.time() - t0 < 5:
            time.sleep(0.005)
    held.__exit__(None, None, None)
    h2.__exit__(None, None, None)
    for th in ths:
        th.join(5.0)
    assert order == [0, 1, 2] and g.frames == 0
    # a frame that gives up while it waits (an error in its thread) leaves the line
    held = g.apply(0)
    held.__enter__()
    wait0 = g._cond.wait
    calls = []

    def bad(timeout=None):
        calls.append(1)
        raise RuntimeError("stopped")
    g._cond.wait = bad
    try:
        g._enter("applies", 9)
        raise AssertionError("not raised")
    except RuntimeError:
        pass
    g._cond.wait = wait0
    assert g._line["applies"] == [] and g.applies == 1
    held.__exit__(None, None, None)
    with g.apply(1):
        assert g.applies == 1
    g.close()


def test_frame_gate_takes_a_slot_away_when_memory_is_short_and_gives_it_back_after_plenty(monkeypatch):
    monkeypatch.delenv("CONTOUR_FRAME_GATE", raising=False)
    m = Mem(free=40.0)
    g = _gate(m)
    g.hint(True)
    assert g.slots == 4
    g.applies = 4                                               # (as if four were being retouched)
    m.now, m.free = 5.0, 1.0                                    # the computer's memory nearly gone
    g.look()
    assert g.slots == 3 and g.shrunk == 1
    m.now = 6.0
    g.look()                                                    # not again within three seconds
    assert g.slots == 3
    m.now = 9.0
    g.look()
    assert g.slots == 2 and g.shrunk == 2
    g.applies = 2
    m.now, m.free = 12.0, 30.0                                  # plenty again: only after GROW_S seconds of it
    g.look()
    m.now = 20.0
    g.look()
    assert g.slots == 2
    m.now = 9.0 + g.GROW_S + 0.1
    g.look()
    assert g.slots == 3 and g.grown == 1
    m.now += g.GROW_S + 1
    m.free = 6.0                                                # free, but less than the reserve and a frame and a half: no more
    g.look()
    m.now += g.GROW_S + 1
    g.look()
    assert g.slots == 3
    for _ in range(10):                                         # never under one
        m.now += 4.0
        m.free = 0.2
        g.look()
    assert g.slots == 1
    assert "도중에" in g.summary()


def test_frame_gate_off_and_reads_this_process(monkeypatch):
    monkeypatch.setenv("CONTOUR_FRAME_GATE", "0")
    g = _gate(Mem(free=0.2))
    assert not g.on and g.slots == 4
    with g.frame():
        with g.apply():
            assert g.applies == 0 and g.summary() == ""
    monkeypatch.delenv("CONTOUR_FRAME_GATE")
    assert perf.FrameGate(1, 4).on is False                    # one frame at a time anyway: nothing to hold
    assert perf.process_gb() > 0.05                            # this process's memory is read here
    monkeypatch.setenv("CONTOUR_RESERVE_GB", "10")
    g = _gate(Mem(free=17.7), people=4)
    g.hint(False)
    assert g.reserve == 10.0 and g.slots == 1                  # (17.7 - 10 - 5.6) // 2.1


def _render_threads(monkeypatch, threads: str, free):
    """The fixture rendered with `threads` frames at once; free(): the computer's free memory in GB as the render sees it."""
    from contour.core.project import Project
    from contour.warp import render as RD
    monkeypatch.setenv("CONTOUR_RENDER_THREADS", threads)
    monkeypatch.setenv("CONTOUR_RAW_AHEAD", "0")
    monkeypatch.setattr(perf, "free_gb", lambda *a, **k: free())
    reps, said = [], []
    inner = getattr(RD._render_cut, "inner", RD._render_cut)      # (the render itself, not this helper's wrapper of an earlier call)

    def with_progress(project, cut, tracker, out, progress=None, *a, **k):
        reps.append(inner(project, cut, tracker, out, lambda f, m="": said.append(m), *a, **k))
        return reps[-1]
    with_progress.inner = inner
    monkeypatch.setattr(RD, "_render_cut", with_progress)
    pdir, src = BU.copy_fixture(Path(__file__).resolve().parent / "data" / "baseline_1222", Path(tempfile.mkdtemp()))
    prj = Project.load(str(pdir))
    with BU.NoModels():
        out = BU.render_planes(prj, prj.cuts[0])
    reps[-1].said = said
    return out, reps[-1]


def test_short_of_memory_a_render_retouches_fewer_frames_at_once_and_makes_the_same_file(monkeypatch):
    """Four frames at once filled the 31 GB PC on a cut of six people (j2896a). With little memory free the frames are
    retouched one at a time: the same file, told in the render's summary."""
    monkeypatch.delenv("CONTOUR_FRAME_GATE", raising=False)
    one, r1 = _render_threads(monkeypatch, "1", lambda: 64.0)
    many, r3 = _render_threads(monkeypatch, "3", lambda: 64.0)
    assert np.array_equal(one, many) and "메모리에 맞춰" not in r3.accel and "프레임 3개 동시 처리" in r3.accel
    short, rs = _render_threads(monkeypatch, "3", lambda: 0.5)
    assert np.array_equal(one, short)
    assert "메모리에 맞춰 한 번에 1프레임까지 보정" in rs.accel
    # ... and while it renders, in the progress line (a render with room says nothing of it)
    assert any("메모리에 맞춰 한 번에 1프레임까지 보정" in m for m in rs.said) and all(m.startswith("렌더 ") for m in rs.said if "메모리" in m)
    assert not any("메모리에 맞춰" in m for m in r3.said) and any(m.startswith("렌더 ") for m in r3.said)
    monkeypatch.setenv("CONTOUR_FRAME_GATE", "0")               # switched off: never held
    out, r = _render_threads(monkeypatch, "3", lambda: 0.5)
    assert np.array_equal(one, out) and "메모리에 맞춰" not in r.accel
