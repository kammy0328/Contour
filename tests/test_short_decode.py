"""3. A file that decodes to fewer frames than its metadata says (a truncated export): the proxy must not be marked
complete with a black tail, and a render must not end as a success with a shorter file (in an edited timeline every
later shot would move)."""
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video


def _short_cut(tmp_path):
    from contour.core import media as MD
    from contour.core.project import Project
    v = make_video(tmp_path / "short.mov", n=3, fmt="mov")
    info = MD.probe(str(v))
    assert info.frames == 3
    info.frames = 6                       # the metadata promises 6
    prj = Project.create(str(tmp_path), "S")
    c = prj.add_cut(info, "short")
    return prj, c


def test_proxy_short_decode_is_not_complete(tmp_path):
    from contour.core.colour import manager
    from contour.core.proxy import Proxy
    from contour.warp.render import _vs_for
    prj, c = _short_cut(tmp_path)
    px = Proxy(prj.proxy_dir(c), c.info, 1920)
    vs = _vs_for(c)
    with pytest.raises(Exception) as ei:
        px.build(manager(), vs)
    assert "3" in str(ei.value) and "6" in str(ei.value), "the message says how many frames were read"
    assert not px.is_valid(vs.key()), "a proxy with a black tail is never valid"


def test_render_short_decode_fails_and_leaves_no_file(tmp_path):
    from contour.warp.render import render_cut
    prj, c = _short_cut(tmp_path)
    c.output.out_dir = str(tmp_path / "out")
    c.output.format = "prores_4444"
    from contour.core.proxy import Proxy
    from contour.track.analyze import CutTracker
    tr = CutTracker(prj, c, Proxy(prj.proxy_dir(c), c.info, 1920))      # not analysed: no retouch, no warp
    with pytest.raises(Exception) as ei:
        render_cut(prj, c, tr, c.output)
    assert "3" in str(ei.value) and "6" in str(ei.value)
    outs = list(Path(tmp_path / "out").rglob("*.mov")) if (tmp_path / "out").exists() else []
    assert outs == [], f"no short file is left as if finished: {outs}"
