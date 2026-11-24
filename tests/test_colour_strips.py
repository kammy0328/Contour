"""A picture converted for the display in strips of rows by several threads (colour.py, 1.27) is the picture one call over the whole of it makes: the same
floats, the same 8 bit values, whatever the number of threads."""
import threading

import numpy as np
import pytest

from contour.core import colour as C

pytestmark = pytest.mark.skipif(C.OCIO is None, reason="OpenColorIO is not installed")


def _vs():
    return C.ViewSettings(input_cs="ARRI LogC4")


def _picture(h=1100, w=1000, seed=1):
    r = np.random.default_rng(seed)
    a = r.random((h, w, 3), dtype=np.float32) * 0.8
    a[:40, :40] = 0.0                     # blacks and
    a[40:80, :40] = 1.0                   # whites, where a log curve is most steep or flat
    a[80:120, :40] = np.float32(-0.02)    # and under black
    return a


def _manager():
    cm = C.manager()
    if cm.config is None:
        pytest.skip("no OCIO config")
    return cm


@pytest.mark.parametrize("n", [2, 3, 8])
def test_float_conversion_is_the_same_in_strips(monkeypatch, n):
    cm, a = _manager(), _picture()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "1")
    one = cm.to_display(a, _vs())
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", str(n))
    many = cm.to_display(a, _vs())
    assert one.dtype == np.float32 and many.dtype == np.float32
    assert np.array_equal(one, many)


@pytest.mark.parametrize("n", [2, 5, 8])
def test_u8_conversion_is_the_same_in_strips(monkeypatch, n):
    cm, a = _manager(), _picture(seed=2)
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "1")
    one = cm.to_display_u8(a, _vs())
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", str(n))
    many = cm.to_display_u8(a, _vs())
    assert one.dtype == np.uint8 and many.dtype == np.uint8 and one.shape == a.shape
    assert np.array_equal(one, many)


def test_the_input_is_not_changed(monkeypatch):
    cm, a = _manager(), _picture(seed=3)
    b = a.copy()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    cm.to_display(a, _vs())
    cm.to_display_u8(a, _vs())
    assert np.array_equal(a, b)


def test_strips_are_used_for_a_big_picture_and_not_for_a_small_one(monkeypatch):
    cm = _manager()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    jobs = []
    real = C._strips

    class Spy:
        def __init__(self, pool):
            self.pool = pool

        def submit(self, fn, *a):
            jobs.append(a)
            return self.pool.submit(fn, *a)
    monkeypatch.setattr(C, "_strips", lambda n: Spy(real(n)))
    cm.to_display_u8(_picture(), _vs())
    assert len(jobs) == 4
    jobs.clear()
    cm.to_display_u8(np.zeros((100, 100, 3), np.float32) + 0.3, _vs())          # 10 000 pixels: one call
    assert jobs == []


def test_odd_shapes_and_contiguity(monkeypatch):
    cm = _manager()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "1")
    a = _picture(1500, 760, seed=4)
    ref = cm.to_display_u8(a, _vs())
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "7")
    assert np.array_equal(cm.to_display_u8(a, _vs()), ref)
    # a view that is not contiguous (every other column) goes through the one call path, with the same answer as a contiguous copy of it
    v = a[:, ::2]
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "1")
    ref2 = cm.to_display_u8(np.ascontiguousarray(v), _vs())
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "7")
    assert np.array_equal(cm.to_display_u8(v, _vs()), ref2)
    # not three channels: left to the one call (or to OCIO's own complaint), not split
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    assert not C._can_strip(np.zeros((2000, 1000, 4), np.float32), 4)
    assert not C._can_strip(np.zeros((2000, 3), np.float32), 4)


def test_passthrough_and_exposure_are_as_before(monkeypatch):
    cm = _manager()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    a = _picture(1100, 1000, seed=5)
    vs = C.ViewSettings(input_cs=C.PASSTHROUGH, exposure=1.0)
    assert np.allclose(cm.to_display(a, vs), a * 2.0)
    u8 = cm.to_display_u8(a, vs)
    assert np.array_equal(u8, (np.clip(a * 2.0, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def test_many_threads_asking_at_once_each_get_their_own_right_answer(monkeypatch):
    cm = _manager()
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    pics = [_picture(1100, 1000, seed=10 + i) for i in range(4)]
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "1")
    want = [cm.to_display_u8(p, _vs()) for p in pics]
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    got = [None] * 4
    errs = []

    def run(i):
        try:
            for _ in range(2):
                got[i] = cm.to_display_u8(pics[i], _vs())
        except Exception as e:           # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    assert not errs
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_a_failing_strip_raises_and_none_is_left_running(monkeypatch):
    monkeypatch.setenv("CONTOUR_COLOUR_THREADS", "4")
    done = []

    def job(y0, y1):
        if y0 == 0:
            raise ValueError("first strip")
        done.append((y0, y1))
    with pytest.raises(ValueError):
        C._run_strips(4, 400, job)
    assert len(done) == 3
