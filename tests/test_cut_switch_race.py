"""2. Work queued for one cut must not land on the next one: a joint re-solve submitted on cut A that runs after the
user went to cut B must re-solve A's tracker (and keep it), never B's; a preview frame being computed for A when the
cut changes must not come back as B's frame."""
import threading
import time

import numpy as np
import pytest


class FakeTracker:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.saved = 0
        self.store = object()

    def resolve_local(self, pid, t, frame_end=None):
        self.calls.append((pid, t, frame_end))
        return (t, t)

    def resolve_all(self):
        self.calls.append("all")

    def save(self):
        self.saved += 1


def test_queued_resolve_stays_on_its_cut(qapp):
    from contour.ui.session import EditSession
    from contour.ui.workers import SerialWorker
    s = EditSession()
    A, B = FakeTracker("A"), FakeTracker("B")
    s.tracker = A
    w = SerialWorker()
    gate = threading.Event()
    w.submit(("block", 0), gate.wait)                 # the worker is busy with something before
    from contour.ui.main_window import resolve_task
    w.submit(("resolve", 3), resolve_task(s, 3, 10))  # submitted while cut A is open
    s.tracker = B                                     # the user goes to cut B
    gate.set()
    t0 = time.time()
    while w.pending() and time.time() - t0 < 5:
        time.sleep(0.02)
    time.sleep(0.2)
    w.stop()
    w.wait(2000)
    assert B.calls == [], "cut B's tracker must not be re-solved with cut A's joint"
    assert A.calls == [(3, 10, None)], "cut A's re-solve still happens"
    assert A.saved == 1, "and is written, since cut A was saved before it ran"


class SlowProxy:
    """A proxy whose frame() waits for a signal (a frame being read when the cut changes)."""
    def __init__(self, value, gate=None):
        self.value = value
        self.gate = gate
        self.w, self.h, self.n, self.sx, self.sy = 8, 6, 10, 1.0, 1.0

    def frame(self, t):
        if self.gate is not None:
            self.gate.wait()
        return np.full((self.h, self.w, 3), self.value, np.uint8)

    def close(self):
        pass


def test_frame_read_for_old_cut_is_not_cached_for_new(qapp, monkeypatch):
    from contour.ui.session import EditSession
    s = EditSession()
    gate = threading.Event()
    s.proxy = SlowProxy(10, gate)
    s.tracker = None
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("a", s.retouched(5)))
    th.start()
    time.sleep(0.1)
    # the cut changes while frame 5 of the old cut is being read (open_cut, without a project: by hand)
    with s._lock:
        s.cut = None
        s.proxy = SlowProxy(200)
        s._cache.clear()
        s._frames.clear()
        s.version += 1
        if hasattr(s, "_new_epoch"):
            s._new_epoch()
    gate.set()
    th.join(5)
    got = s.retouched(5)
    assert int(got[0, 0, 0]) == 200, "the new cut's frame 5, not the old cut's"
