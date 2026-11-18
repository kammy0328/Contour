"""Edited-timeline export: cancel stops the file being written in every kind of shot (transparent, original,
retouched) and is reported as cancelled, not as a finished file or an error; a source that ends early stays an
error; the progress counts the frames written, once, so it only goes up and ends at 100 % (a shot with work range
[20, 80) ran it to 140 %); the shots finished before a cancel stay, the one being written goes."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from contour.core import media as M
from contour.warp import tlrender as TL
from tests.conftest import make_video


class FakeWriter:
    made = []

    def __init__(self, path, *a, **k):
        self.path, self.n, self.closed, self.aborted = path, 0, False, False
        FakeWriter.made.append(self)

    def write(self, planes):
        self.n += 1

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeWriter.made = []
    monkeypatch.setattr(M, "FrameWriter", FakeWriter)
    v = make_video(tmp_path / "src.mp4", n=100)
    info = M.probe(str(v))
    assert info.frames == 100
    prj = NS(timeline={}, path=str(tmp_path / "p" / "project.json"), name="T")
    return tmp_path, info, prj


def _cut(info, name="c1", wr=None, frames=None):
    inf = info
    if frames is not None:
        import copy
        inf = copy.copy(info)
        inf.frames = frames
    return NS(id=name, name=name, info=inf, work_range=wr or (0, inf.frames), version=1, source=inf.path)


def _run(monkeypatch, prj, rows, mode, out_dir, cancel_after=None, fmt="prores_4444_xq"):
    monkeypatch.setattr(TL, "plan", lambda p, e: rows)
    seen = []
    exp = TL.TimelineExport(mode=mode, format=fmt, out_dir=str(out_dir), audio=False)

    def cancel():
        return cancel_after is not None and sum(w.n for w in FakeWriter.made) >= cancel_after
    rep = TL.render_timeline(prj, exp, lambda f, m="": seen.append(f), cancel)
    return rep, seen


def test_cancel_in_transparent_shot(env, monkeypatch):
    tmp, info, prj = env
    rows = [dict(cut=_cut(info), action="alpha", status="good")]
    rep, _ = _run(monkeypatch, prj, rows, "full", tmp / "o", cancel_after=10)
    w = FakeWriter.made[0]
    assert rep.cancelled and rep.paths == [] and w.aborted and not w.closed
    assert w.n == 10 and rep.cancelled_path == w.path


def test_cancel_in_original_copy_is_not_an_error(env, monkeypatch):
    tmp, info, prj = env
    rows = [dict(cut=_cut(info), action="original", status="good")]
    rep, _ = _run(monkeypatch, prj, rows, "full", tmp / "o", cancel_after=10, fmt="prores_422_hq")
    w = FakeWriter.made[0]
    assert rep.cancelled and rep.paths == [] and w.aborted and not w.closed


def test_source_ending_early_stays_an_error(env, monkeypatch):
    tmp, info, prj = env
    rows = [dict(cut=_cut(info, frames=110), action="original", status="good")]
    with pytest.raises(M.MediaError):
        _run(monkeypatch, prj, rows, "full", tmp / "o", fmt="prores_422_hq")
    assert FakeWriter.made[0].aborted and not FakeWriter.made[0].closed


def _fake_render(monkeypatch):
    """render_cut writing the work range into the export's writer, reporting its own fraction of the work range."""
    import contour.core.frames as FR
    import contour.track.analyze as AN
    import contour.warp.render as R
    monkeypatch.setattr(FR, "source_present", lambda info: True)
    monkeypatch.setattr(FR, "frames_for", lambda p, c: None)
    monkeypatch.setattr(AN, "CutTracker", lambda p, c, px: None)
    monkeypatch.setattr(TL.OutputSettings, "from_dict", classmethod(lambda cls, d: NS()))

    def render_cut(prj, cut, tr, o, progress, cancel, ext_writer=None):
        a, b = cut.work_range
        rep = NS(frames=0, cancelled=False, warnings=[])
        z = np.zeros((cut.info.height, cut.info.width), np.uint16)
        for t in range(a, b):
            if cancel and cancel():
                rep.cancelled = True
                break
            ext_writer.write([z, z, z])
            rep.frames += 1
            progress(rep.frames / (b - a), "렌더")
        return rep
    monkeypatch.setattr(R, "render_cut", render_cut)


def test_partial_work_range_progress_is_100_percent(env, monkeypatch):
    tmp, info, prj = env
    _fake_render(monkeypatch)
    c = _cut(info, wr=(20, 80))
    c.output = NS(to_dict=lambda: {})
    rows = [dict(cut=c, action="retouch", status="good")]
    rep, seen = _run(monkeypatch, prj, rows, "full", tmp / "o", fmt="prores_422_hq")
    w = FakeWriter.made[0]
    assert not rep.cancelled and w.closed and w.n == 100 and rep.frames == 100
    assert seen and max(seen) == pytest.approx(1.0) and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_cancel_inside_retouch(env, monkeypatch):
    tmp, info, prj = env
    _fake_render(monkeypatch)
    c = _cut(info, wr=(20, 80))
    c.output = NS(to_dict=lambda: {})
    rows = [dict(cut=c, action="retouch", status="good")]
    rep, _ = _run(monkeypatch, prj, rows, "full", tmp / "o", cancel_after=30, fmt="prores_422_hq")
    assert rep.cancelled and FakeWriter.made[0].aborted and not FakeWriter.made[0].closed


def test_each_mode_keeps_finished_shots(env, monkeypatch):
    tmp, info, prj = env
    rows = [dict(cut=_cut(info, "a"), action="original", status="good"),
            dict(cut=_cut(info, "b"), action="original", status="good")]
    rep, seen = _run(monkeypatch, prj, rows, "each", tmp / "o", cancel_after=110, fmt="prores_422_hq")
    a, b = FakeWriter.made
    assert rep.cancelled and a.closed and not a.aborted and b.aborted and not b.closed
    assert rep.paths == [a.path] and rep.cancelled_path == b.path and rep.frames == 100
    assert max(seen) <= 1.0


def test_each_mode_all_done(env, monkeypatch):
    tmp, info, prj = env
    rows = [dict(cut=_cut(info, "a"), action="original", status="good"),
            dict(cut=_cut(info, "b"), action="original", status="good")]
    rep, seen = _run(monkeypatch, prj, rows, "each", tmp / "o", fmt="prores_422_hq")
    assert not rep.cancelled and len(rep.paths) == 2 and rep.frames == 200
    assert seen[-1] == pytest.approx(1.0) and all(b >= a for a, b in zip(seen, seen[1:]))
