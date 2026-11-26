"""1.26.3: the apply on the people's own pieces of the picture (engine.roi_pieces / apply_on_pieces) for display-referred footage too
(before: log footage only), and the pieces used unless their areas add up to more than APPLY_ROI_COVER of the picture times the people
(LNGSHOT 024, 4 faces: the retouch 230 -> 121 s over 12 frames, the file frame for frame the same: j2851)."""
import tempfile
import types
from pathlib import Path

import numpy as np

import baseline_util as BU
from contour.retouch import engine as E


def test_cover_follows_the_people(monkeypatch):
    planes = [np.zeros((100, 200), np.uint16)] * 3
    boxes = {1: (0, 0, 120, 100), 2: (80, 0, 200, 100)}            # each 0.6 of the picture, 1.2 together
    monkeypatch.setattr(E, "roi_job_ok", lambda j, why=False: True)
    monkeypatch.setattr(E, "roi_job_box", lambda j, *a, **k: boxes[j.pid])
    rf = types.SimpleNamespace(jobs=[types.SimpleNamespace(pid=1), types.SimpleNamespace(pid=2)], proxy_wh=(200, 100))
    assert E.roi_pieces(planes, rf, 1.0) == [boxes[1], boxes[2]]    # 1.2 < 0.8 x 2 people
    rf1 = types.SimpleNamespace(jobs=[types.SimpleNamespace(pid=1)], proxy_wh=(200, 100))
    boxes[1] = (0, 0, 190, 100)                                      # one person, 0.95 of the picture: the whole frame
    assert E.roi_pieces(planes, rf1, 1.0) is None
    monkeypatch.setattr(E, "APPLY_ROI", 0.0)
    assert E.roi_pieces(planes, rf, 1.0) is None


def test_display_referred_pieces_render_as_the_whole_frame(monkeypatch):
    from contour.core.project import Project
    out = {}
    real = E.apply_on_pieces
    calls = []
    monkeypatch.setattr(E, "apply_on_pieces", lambda *a, **k: calls.append(1) or real(*a, **k))
    for cover in (0.8, 100.0):                    # the fixture's piece is most of its small picture: on pieces only when the rule is lifted
        monkeypatch.setattr(E, "APPLY_ROI_COVER", cover)
        calls.clear()
        pdir, src = BU.copy_fixture(Path(__file__).resolve().parent / "data" / "baseline_1222", Path(tempfile.mkdtemp()))
        prj = Project.load(str(pdir))
        with BU.NoModels():
            out[cover] = BU.render_planes(prj, prj.cuts[0])
        assert bool(calls) == (cover > 1)
    assert np.array_equal(out[0.8], out[100.0])
