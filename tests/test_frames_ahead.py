"""1.27: the frames of a pass read ahead of the work on them (track/ahead.py), and the SAM pass with the whole frame encoded
ahead (segment.run): what the pass gets is what it would have read and made itself, in the pass's order, errors at their frame,
nothing running on after the pass ends."""
import threading
import time

import numpy as np
import pytest

from contour.track import ahead as AH
from contour.track.ahead import FramesAhead


class Px:
    """Frames whose every pixel is the frame's number."""
    w, h, sx, sy = 16, 12, 1.0, 1.0

    def __init__(self, n=12, slow=0.0, bad=None):
        self.n, self.slow, self.bad = n, slow, bad
        self.asked = []
        self.lock = threading.Lock()

    def frame(self, t):
        with self.lock:
            self.asked.append(t)
        if self.slow:
            time.sleep(self.slow)
        if self.bad is not None and t == self.bad:
            raise OSError(f"frame {t}")
        return np.full((self.h, self.w, 3), t, np.uint8)


def _alive():
    return [th for th in threading.enumerate() if th.name == "frames-ahead" and th.is_alive()]


def test_in_order_every_frame_once_with_its_extra():
    px = Px()
    with FramesAhead(px, 2, 10, n=2, extra=lambda t, rgb: (t, int(rgb[0, 0, 0]) * 2), view=None) as fa:
        for t in range(2, 10):
            rgb, x = fa.get(t)
            assert int(rgb[0, 0, 0]) == t and x == (t, 2 * t)
        assert fa.hits == 8 and fa.direct == 0
    assert sorted(px.asked) == list(range(2, 10))            # each read once, none twice
    assert not _alive()


def test_skipped_and_out_of_order_frames_are_right_too():
    px = Px()
    with FramesAhead(px, 0, 12, n=2, view=None) as fa:
        assert int(fa.get(0)[0][0, 0, 0]) == 0
        assert int(fa.get(3)[0][0, 0, 0]) == 3               # 1 and 2 skipped by the pass
        assert int(fa.get(1)[0][0, 0, 0]) == 1               # back: read on the spot
        assert int(fa.get(7)[0][0, 0, 0]) == 7
        assert fa.direct >= 1


def test_an_error_is_met_at_its_frame():
    px = Px(bad=4)
    with FramesAhead(px, 0, 8, n=2, view=None) as fa:
        for t in range(4):
            assert int(fa.get(t)[0][0, 0, 0]) == t
        with pytest.raises(OSError):
            fa.get(4)
    assert not _alive()


def test_extra_error_is_met_at_its_frame():
    def extra(t, rgb):
        if t == 2:
            raise RuntimeError("no memory")
        return t
    with FramesAhead(Px(), 0, 6, n=2, extra=extra, view=None) as fa:
        assert fa.get(0)[1] == 0 and fa.get(1)[1] == 1
        with pytest.raises(RuntimeError):
            fa.get(2)


def test_close_ends_the_reader_at_once_even_when_it_waits():
    px = Px(n=200, slow=0.01)
    fa = FramesAhead(px, 0, 200, n=2, view=None)
    fa.get(0)
    time.sleep(0.1)                                          # the reader stands with two frames ready
    t0 = time.time()
    fa.close()
    assert time.time() - t0 < 2.0 and not _alive()
    assert len(px.asked) < 20                                # it did not read on


def test_zero_ahead_is_as_before(monkeypatch):
    monkeypatch.setenv("DIAP_AHEAD", "0")
    px = Px()
    with FramesAhead(px, 0, 5, view=None) as fa:
        assert fa._th is None
        for t in range(5):
            assert int(fa.get(t)[0][0, 0, 0]) == t
        assert fa.direct == 5 and fa.hits == 0
    assert px.asked == list(range(5))
    monkeypatch.setenv("DIAP_AHEAD", "x")
    assert AH.frames_ahead() == AH.N


# ------------------------------------------------------------------------------------------------ the SAM pass
class Seg:
    """A segmenter that records what it encoded (by whom) and gives it back."""

    def __init__(self):
        self.calls = []

    def encode(self, rgb, region=None):
        self.calls.append((int(rgb[0, 0, 0]), region, threading.current_thread().name))
        return dict(emb=int(rgb[0, 0, 0]), rect=(0, 0, rgb.shape[1], rgb.shape[0]))


def _run(tmp_path, monkeypatch, ahead: str, cancel_at=None):
    from contour.core.proxy import MatteStore
    from contour.track import segment as SG
    monkeypatch.setenv("DIAP_AHEAD", ahead)
    px = Px(n=10)
    seg = Seg()
    got = []
    monkeypatch.setattr(SG, "analysis_view", lambda x: x)
    monkeypatch.setattr(AH, "analysis_view", lambda x: x)
    monkeypatch.setattr(SG, "prompts", lambda st, t, *a: ([1] if t % 3 else []))       # frames 0, 3, 6, 9: nobody to cut out
    monkeypatch.setattr(SG, "head_priors", lambda *a, **k: {})
    monkeypatch.setattr(SG, "combine", lambda logits, rvm, shape, heads, extra: (np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)))

    def decode(seg_, rgb, st, t, *a, full=None, **k):
        got.append((t, int(rgb[0, 0, 0]), None if full is None else full["emb"]))
        return {}, []
    monkeypatch.setattr(SG, "decode_frame", decode)
    m = MatteStore(tmp_path, px.n, 8, 6)
    lab = MatteStore(tmp_path, px.n, 8, 6, prefix="seglab")
    for x in (m, lab):
        x.create()
        x.close()
    st = type("St", (), dict(present=np.ones((1, px.n), bool), pids=[0]))()
    n = [0]

    def cancel():
        n[0] += 1
        return cancel_at is not None and n[0] > cancel_at
    try:
        SG.run(st, px, m, lab, 0, px.n, "seg_sam31", False, None, cancel, None, models={"seg": seg, "no_finder": True})
    except Exception as e:
        assert type(e).__name__ == "Cancelled"
    return got, seg


def test_sam_pass_gets_each_frame_and_its_encoding_made_ahead(tmp_path, monkeypatch):
    got, seg = _run(tmp_path, monkeypatch, "2")
    assert [g[0] for g in got] == list(range(10)) and all(g[0] == g[1] for g in got)
    assert [g[2] for g in got] == [None if t % 3 == 0 else t for t in range(10)]       # the frame's own encoding, none where nobody is
    assert sorted(c[0] for c in seg.calls) == [t for t in range(10) if t % 3]            # each encoded once
    assert all(c[2] == "frames-ahead" for c in seg.calls)
    assert not _alive()


def test_sam_pass_without_ahead_reads_and_encodes_on_the_pass_own_thread(tmp_path, monkeypatch):
    got, seg = _run(tmp_path, monkeypatch, "0")
    ahead, _ = _run(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, monkeypatch, "2")
    assert got == ahead                                                                 # the same frames, the same encodings
    assert sorted(c[0] for c in seg.calls) == [t for t in range(10) if t % 3]
    assert all(c[2] != "frames-ahead" for c in seg.calls)


def test_cancelled_sam_pass_leaves_no_reader(tmp_path, monkeypatch):
    got, seg = _run(tmp_path, monkeypatch, "2", cancel_at=3)
    assert len(got) == 3
    assert not _alive()
    assert not [th for th in threading.enumerate() if th.name.startswith("sam-finish") and th.is_alive()]


def test_mattes_are_written_behind_the_pass_in_order_and_an_error_there_is_the_pass_error(tmp_path, monkeypatch):
    from contour.core.proxy import MatteStore
    from contour.track import segment as SG
    monkeypatch.setenv("DIAP_AHEAD", "2")
    px = Px(n=8)
    monkeypatch.setattr(AH, "analysis_view", lambda x: x)
    monkeypatch.setattr(SG, "prompts", lambda st, t, *a: [1])
    monkeypatch.setattr(SG, "head_priors", lambda *a, **k: {})
    monkeypatch.setattr(SG, "decode_frame", lambda seg_, rgb, st, t, *a, **k: ({7: dict(canvas=t, src="click")}, []))
    order, bad = [], [None]

    def combine(logits, rvm, shape, heads, extra):
        t = logits[0][1]
        order.append((t, threading.current_thread().name))
        if bad[0] == t:
            raise ValueError("combine")
        return np.full(shape, 10 + t, np.uint8), np.full(shape, 1 + t, np.uint8)
    monkeypatch.setattr(SG, "combine", combine)

    def run(sub):
        d = tmp_path / sub
        d.mkdir()
        m, lab = MatteStore(d, px.n, 8, 6), MatteStore(d, px.n, 8, 6, prefix="seglab")
        for x in (m, lab):
            x.create()
            x.close()
        st = type("St", (), dict(present=np.ones((1, px.n), bool), pids=[7]))()
        SG.run(st, px, m, lab, 0, px.n, "seg_sam31", False, None, None, None, models={"seg": Seg(), "no_finder": True})
        return (np.fromfile(m.path, np.uint8).reshape(px.n, 6, 8), np.fromfile(lab.path, np.uint8).reshape(px.n, 6, 8))
    al, lb = run("a")
    assert [int(al[t, 0, 0]) for t in range(8)] == [10 + t for t in range(8)] and [int(lb[t, 0, 0]) for t in range(8)] == [1 + t for t in range(8)]
    assert [o[0] for o in order] == list(range(8)) and all(o[1].startswith("sam-finish") for o in order)
    bad[0] = 5
    with pytest.raises(ValueError):
        run("b")
    assert not [th for th in threading.enumerate() if th.name.startswith("sam-finish") and th.is_alive()]
