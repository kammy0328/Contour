"""1.18 original-first: frames read straight from the original (frames.SourceFrames) are the frames a proxy holds,
pixel for pixel, at the same size and frame numbers, in any access order (forward, backwards, jumps, two readers
interleaved), for long-GOP H.264 / H.265, ProRes, image sequences, a stretch of a longer file (timeline offset) and
full / limited range; the decoded-frame memory is capped; readers end on close."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import make_video


def _pattern(i, w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx * 255 / w + i * 7) % 256
    g = (yy * 255 / h + i * 13) % 256
    b = ((xx + yy) * 2 + i * 29) % 256
    f = np.dstack([r, g, b]).astype(np.uint8)
    f[(i * 3) % h: (i * 3) % h + 4, :] = 255          # a moving bar: every frame is different
    return f


def _encode(path: Path, n=40, w=96, h=64, fps=24, args=()):
    from contour.core import media as MD
    MD.configure_ffmpeg()
    cmd = [MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", *args, str(path)]
    subprocess.run(cmd, input=b"".join(_pattern(i, w, h).tobytes() for i in range(n)), check=True)
    return path


def _kinds(tmp_path):
    out = {
        "h264_longgop": _encode(tmp_path / "a.mp4", args=("-c:v", "libx264", "-g", "40", "-bf", "3", "-pix_fmt", "yuv420p",
                                                          "-crf", "18")),
        "prores": _encode(tmp_path / "b.mov", args=("-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le")),
        "h264_fullrange": _encode(tmp_path / "c.mp4", args=("-c:v", "libx264", "-g", "40", "-pix_fmt", "yuvj420p",
                                                            "-color_range", "pc", "-crf", "18")),
    }
    try:
        out["h265_longgop"] = _encode(tmp_path / "d.mp4", args=("-c:v", "libx265", "-x265-params", "keyint=40:log-level=0",
                                                                "-pix_fmt", "yuv420p", "-crf", "20"))
    except subprocess.CalledProcessError:
        pass
    seq = tmp_path / "seq"
    seq.mkdir()
    import cv2
    for i in range(30):
        cv2.imwrite(str(seq / f"f.{1000 + i:04d}.png"), cv2.cvtColor(_pattern(i, 96, 64), cv2.COLOR_RGB2BGR))
    out["png_sequence"] = seq / "f.1000.png"
    return out



def _pair(tmp_path, name, path, offset=0, frames=None):
    from contour.core import media as MD
    from contour.core.colour import manager
    from contour.core.frames import SourceFrames
    from contour.core.proxy import Proxy
    from contour.core.project import Project
    from contour.warp.render import _vs_for
    info = MD.probe(str(path))
    if offset or frames:
        info.offset = offset
        info.frames = frames
    prj = Project.create(str(tmp_path / ("p_" + name)), name)
    c = prj.add_cut(info, name)
    vs = _vs_for(c)
    px = Proxy(prj.proxy_dir(c), c.info, 1920)
    assert px.build(manager(), vs)
    sf = SourceFrames(tmp_path / ("sf_" + name), c.info, (px.w, px.h), manager(), vs)
    return px, sf


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    return _kinds(tmp_path_factory.mktemp("media"))


@pytest.mark.parametrize("kind", ["h264_longgop", "prores", "h264_fullrange", "h265_longgop", "png_sequence"])
def test_same_pixels_as_proxy_in_any_order(tmp_path, media, kind):
    from contour.core import frames as FR
    if kind not in media:
        pytest.skip("encoder not in this ffmpeg")
    FR.pool().clear()
    px, sf = _pair(tmp_path, kind, media[kind])
    n = px.n
    assert (sf.n, sf.w, sf.h, sf.sx, sf.sy) == (px.n, px.w, px.h, px.sx, px.sy)
    rng = np.random.default_rng(1)
    orders = [list(range(n)), list(range(n - 1, -1, -1)), list(rng.permutation(n))]
    for order in orders:
        FR.pool().clear()
        sf.close()
        for i in order:
            a, b = sf.frame(i), px.frame(i)
            assert a.shape == b.shape and a.dtype == np.uint8
            assert np.array_equal(a, b), f"{kind} frame {i}: max diff {np.abs(a.astype(int) - b).max()}"
    # backwards: one reader start per block, not one per frame
    FR.pool().clear()
    sf.close()
    s0 = sf.seeks
    for i in range(n - 1, -1, -1):
        sf.frame(i)
    assert sf.seeks - s0 <= -(-n // FR.BLOCK) + 1
    assert np.array_equal(sf.gray(5), px.gray(5))
    sf.close()
    px.close()


def test_timeline_stretch_offset(tmp_path, media):
    """A shot that is frames 10..29 of a longer file (an edited timeline's bake)."""
    from contour.core import frames as FR
    FR.pool().clear()
    px, sf = _pair(tmp_path, "stretch", media["h264_longgop"], offset=10, frames=20)
    full_px, _ = _pair(tmp_path, "full", media["h264_longgop"])
    for i in (0, 7, 19, 3):
        assert np.array_equal(sf.frame(i), px.frame(i))
        assert np.array_equal(sf.frame(i), full_px.frame(10 + i))
    sf.close()


def test_interleaved_readers_do_not_restart_each_other(tmp_path, media):
    """Playback near the end while an analysis pass runs from the start: two readers, each going on."""
    from contour.core import frames as FR
    FR.pool().clear()
    px, sf = _pair(tmp_path, "inter", media["prores"])
    s0 = sf.seeks
    for k in range(15):
        assert np.array_equal(sf.frame(k), px.frame(k))
        assert np.array_equal(sf.frame(25 + k), px.frame(25 + k))
    assert sf.seeks - s0 <= 2, "one reader each (or one reading on), never one start per request"
    sf.close()


def test_pool_is_capped_and_frames_are_copies(tmp_path, media):
    from contour.core import frames as FR
    px, sf = _pair(tmp_path, "cap", media["prores"])
    one = sf.w * sf.h * 3
    old = FR.pool().limit
    FR.pool().clear()
    FR.pool().set_limit(one * 5)
    try:
        for i in range(px.n):
            sf.frame(i)
        assert FR.pool().bytes <= one * 5 and FR.pool().peak <= one * 5
        a = sf.frame(3)
        a[:] = 0                                    # a caller drawing on its frame changes nothing in the pool
        assert np.array_equal(sf.frame(3), px.frame(3))
        assert np.array_equal(sf.frame(0), px.frame(0)), "an evicted frame is decoded again"
    finally:
        FR.pool().set_limit(old)
        sf.close()


def test_close_ends_readers(tmp_path, media):
    from contour.core import frames as FR
    FR.pool().clear()
    _, sf = _pair(tmp_path, "close", media["prores"])
    sf.frame(0)
    sf.frame(30)
    procs = [c.reader._proc for c in sf._cursors]
    assert procs
    sf.close()
    assert sf._cursors == []
    for p in procs:
        assert p.poll() is not None, "no ffmpeg process is left running"


def test_view_change_is_another_cache(tmp_path, media):
    """A different view (input colour space, LUT) never finds the frames decoded for the old one."""
    from contour.core import frames as FR
    from contour.core.colour import ViewSettings, manager
    FR.pool().clear()
    px, sf = _pair(tmp_path, "view", media["prores"])
    sf.frame(2)
    vs2 = ViewSettings(input_cs=sf.vs.input_cs, display=sf.vs.display, view=sf.vs.view, lut_path="", exposure=1.0)
    sf2 = FR.SourceFrames(sf.dir, sf.info, (sf.w, sf.h), manager(), vs2)
    assert sf2.owner != sf.owner
    assert FR.pool().get((sf2.owner, 2)) is None
    sf.close()


def test_missing_frames_raise_instead_of_black(tmp_path):
    """Metadata promising more frames than the file has: an error, never a black frame analysed as a picture."""
    from contour.core import media as MD
    from contour.core import frames as FR
    from contour.core.colour import manager
    from contour.warp.render import _vs_for
    from contour.core.project import Project
    v = make_video(tmp_path / "s.mov", n=3, fmt="mov")
    info = MD.probe(str(v))
    info.frames = 6
    prj = Project.create(str(tmp_path), "S")
    c = prj.add_cut(info, "s")
    sf = FR.SourceFrames(tmp_path / "x", info, (64, 48), manager(), _vs_for(c))
    assert sf.frame(2).shape == (48, 64, 3)
    with pytest.raises(FR.FrameError):
        sf.frame(5)
    sf.close()


@pytest.mark.parametrize("cs", ["ARRI LogC4", "Apple Log"])
def test_log_input_same_as_proxy(tmp_path, media, cs):
    """A log input colour space (OCIO view transform in the conversion): same pixels as the proxy made in that view."""
    from contour.core import frames as FR
    from contour.core import media as MD
    from contour.core.colour import manager
    from contour.core.project import Project
    from contour.core.proxy import Proxy
    from contour.warp.render import _vs_for
    if cs not in [c[0] for c in manager().colorspaces()]:
        pytest.skip("not in this OCIO config")
    FR.pool().clear()
    info = MD.probe(str(media["prores"]))
    prj = Project.create(str(tmp_path / "log"), "log")
    c = prj.add_cut(info, "log")
    c.input_cs = cs
    vs = _vs_for(c)
    px = Proxy(prj.proxy_dir(c), c.info, 1920)
    assert px.build(manager(), vs)
    sf = FR.SourceFrames(tmp_path / "sf", c.info, (px.w, px.h), manager(), vs)
    plain = FR.SourceFrames(tmp_path / "sf2", c.info, (px.w, px.h), manager(), _vs_for(prj.add_cut(info, "plain")))
    for i in (0, 17, 9):
        assert np.array_equal(sf.frame(i), px.frame(i))
    assert not np.array_equal(sf.frame(9), plain.frame(9)), "the log view is applied"
    sf.close()
    plain.close()
