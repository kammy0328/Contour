"""1.19: pieces of the original at its own resolution (frames.SourceCrops) are the same pixels as the whole frame
decoded at full size and cut, at the same frame numbers, for long-GOP H.264, ProRes 4:2:2 and a timeline stretch;
a scaled piece has the size asked for; the reader goes on while the piece stays in its grid cell and ends on close."""
import numpy as np
import pytest

from test_source_frames import _encode


@pytest.fixture(scope="module")
def vids(tmp_path_factory):
    d = tmp_path_factory.mktemp("crops")
    return {"h264": _encode(d / "a.mp4", n=30, w=320, h=240, args=("-c:v", "libx264", "-g", "30", "-bf", "2", "-pix_fmt",
                                                                    "yuv420p", "-crf", "16")),
            "prores": _encode(d / "b.mov", n=30, w=320, h=240, args=("-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt",
                                                                     "yuv422p10le"))}


def _full(info, i, cm, vs):
    from contour.core import media as MD
    return cm.to_display_u8(MD.read_frame_rgb(info, i), vs)


@pytest.mark.parametrize("kind", ["h264", "prores"])
def test_piece_equals_cut_of_full_frame(vids, kind, tmp_path):
    from contour.core import frames as FR, media as MD
    from contour.core.colour import PASSTHROUGH, ViewSettings, manager
    FR.pool().clear()
    info = MD.probe(str(vids[kind]))
    vs = ViewSettings(input_cs=PASSTHROUGH)
    sc = FR.SourceCrops(info, manager(), vs)
    box = (50, 38, 190, 170)
    for i in (0, 1, 2, 3, 17, 5, 29):
        a = sc.piece(i, box)
        b = _full(info, i, manager(), vs)[box[1]:box[3], box[0]:box[2]]
        assert a.shape == b.shape
        if kind == "prores":
            assert np.array_equal(a, b), f"frame {i}"
        else:
            # 4:2:0: the chroma at the piece's edge is interpolated from inside the piece only
            d = np.abs(a.astype(int) - b)
            assert d[4:-4, 4:-4].max() <= 2, f"frame {i}: {d[4:-4, 4:-4].max()}"
    s0 = sc.seeks
    for i in range(6, 14):                       # sequential, the box moving a little: one reader
        sc.piece(i, (52 + i, 40, 188 + i, 168))
    assert sc.seeks - s0 == 1
    half = sc.piece(3, box, 0.5)
    assert abs(half.shape[0] - 66) <= 1 and abs(half.shape[1] - 70) <= 1
    proc = next(iter(sc._readers.values()))[0]._proc
    sc.close()
    assert proc.poll() is not None


def test_piece_of_a_timeline_stretch(vids):
    from contour.core import frames as FR, media as MD
    from contour.core.colour import PASSTHROUGH, ViewSettings, manager
    FR.pool().clear()
    info = MD.probe(str(vids["prores"]))
    vs = ViewSettings(input_cs=PASSTHROUGH)
    full = [_full(info, i, manager(), vs) for i in (10, 14)]
    info.offset, info.frames = 10, 12
    sc = FR.SourceCrops(info, manager(), vs)
    box = (20, 20, 120, 100)
    assert np.array_equal(sc.piece(0, box), full[0][20:100, 20:120])
    assert np.array_equal(sc.piece(4, box), full[1][20:100, 20:120])
    sc.close()
