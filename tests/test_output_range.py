"""3 (1.16.1). A render without any retouch or warp keeps the picture: full or limited range source, BT.709 / BT.601 /
BT.2020, 4:2:2 or 4:4:4. A full-range source used to be written unchanged but tagged limited (darker, flatter)."""
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.mark.parametrize("rng", ["pc", "tv"])
@pytest.mark.parametrize("matrix", ["bt709", "smpte170m", "bt2020nc"])
@pytest.mark.parametrize("pix", ["yuv444p10le", "yuv422p10le"])
def test_plain_render_keeps_picture(tmp_path, rng, matrix, pix):
    from contour.core import media as M
    from contour.core.project import OutputSettings, Project
    from contour.warp.render import render_cut
    M.configure_ffmpeg()
    W, H = 64, 48
    src = tmp_path / "src.mkv"
    # a smooth colour gradient (ProRes keeps it; sharp test patterns differ by compression alone), encoded by
    # ffmpeg itself in the asked range and matrix
    subprocess.run([M.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=24", "-frames:v", "3",
                    "-vf", f"gblur=sigma=4,scale=out_range={rng}:out_color_matrix={matrix.replace('nc', '').replace('smpte170m', 'bt601')}",
                    "-pix_fmt", pix, "-color_range", rng, "-colorspace", matrix, "-c:v", "ffv1", "-level", "3", str(src)],
                   check=True)
    info = M.probe(str(src))
    assert info.color_range == rng
    prj = Project.create(str(tmp_path), "r")
    cut = prj.add_cut(info)
    tracker = NS(store=None, matte=NS(w=W // 2, h=H // 2))
    fmt = "prores_4444" if pix.startswith("yuv444") else "prores_422_hq"
    rep = render_cut(prj, cut, tracker, OutputSettings(format=fmt, out_dir=str(tmp_path / "out"), include_audio=False))
    out = M.probe(rep.out_path)
    a = M.read_frame_rgb(info, 1).astype(float) * 255
    b = M.read_frame_rgb(out, 1).astype(float) * 255
    d = np.abs(a - b)
    assert np.percentile(d, 99) <= 2.5, f"{rng} {matrix} {pix}: p99 diff {np.percentile(d, 99):.2f}, max {d.max():.2f}"
