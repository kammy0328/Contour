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
    proc = (sc._full[0] if FR.SHARED_DECODE else next(iter(sc._readers.values()))[0])._proc
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


def test_pieces_of_a_frame_share_one_decode(vids, monkeypatch):
    """1.26.3 (SHARED_DECODE): the pieces of every person of a frame are cut from one decode of the whole frame, asked for from several
    threads at once; the pieces are the pixels of the whole frame, and each frame is read once."""
    import threading
    from contour.core import frames as FR, media as MD
    from contour.core.colour import PASSTHROUGH, ViewSettings, manager
    FR.pool().clear()
    info = MD.probe(str(vids["prores"]))
    vs = ViewSettings(input_cs=PASSTHROUGH)
    boxes = [(4, 4, 60, 60), (230, 150, 300, 230), (100, 60, 200, 180)]
    full = [_full(info, i, manager(), vs) for i in (0, 1, 2)]
    reads = []
    read0 = FR.FrameReader.read
    monkeypatch.setattr(FR.FrameReader, "read", lambda self: reads.append(1) or read0(self))
    sc = FR.SourceCrops(info, manager(), vs)
    got, errs = {}, []

    def one(k):
        try:
            got[k] = [sc.piece(i, boxes[k]) for i in (0, 1, 2)]
        except Exception as e:                      # pragma: no cover
            errs.append(e)
    ths = [threading.Thread(target=one, args=(k,)) for k in range(3)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    assert not errs, errs
    for k, b in enumerate(boxes):
        for i in range(3):
            assert np.array_equal(got[k][i], full[i][b[1]:b[3], b[0]:b[2]]), (k, i)
    assert len(reads) == 3 and sc.seeks == 1                  # three frames, read once each, by one reader
    sc.close()


def test_two_regions_are_read_side_by_side(vids, monkeypatch):
    from contour.core import frames as FR
    monkeypatch.setattr(FR, "SHARED_DECODE", False)          # (the readers of before 1.26.3, one a region)
    """j2799 (three people of a 4K frame, body blemishes): each person's piece of the original waited for the one
    before it under one lock, a third of a second each. Different regions are read at the same time, one region by
    one thread at a time; the pieces are the same."""
    import threading
    import time
    from contour.core import frames as FR, media as MD
    from contour.core.colour import PASSTHROUGH, ViewSettings, manager
    FR.pool().clear()
    info = MD.probe(str(vids["prores"]))
    vs = ViewSettings(input_cs=PASSTHROUGH)
    boxes = [(4, 4, 60, 60), (230, 150, 300, 230)]
    ref = FR.SourceCrops(info, manager(), vs)
    assert ref._grid(boxes[0]) != ref._grid(boxes[1])
    want = [[ref.piece(i, b) for i in (0, 1, 2)] for b in boxes]
    ref.close()
    FR.pool().clear()
    read0 = FR.FrameReader.read
    now, peak, lk = [0], [0], threading.Lock()

    def slow(self):
        with lk:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        try:
            time.sleep(0.1)
            return read0(self)
        finally:
            with lk:
                now[0] -= 1
    monkeypatch.setattr(FR.FrameReader, "read", slow)
    sc = FR.SourceCrops(info, manager(), vs)
    got, errs = [None, None], []

    def one(k):
        try:
            got[k] = [sc.piece(i, boxes[k]) for i in (0, 1, 2)]
        except Exception as e:                      # pragma: no cover
            errs.append(e)
    ths = [threading.Thread(target=one, args=(k,)) for k in range(2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    assert not errs, errs
    assert peak[0] == 2, "the two regions were read one after the other"
    for k in range(2):
        for a, b in zip(got[k], want[k]):
            assert np.array_equal(a, b)
    assert sc.seeks == 2 and len(sc._readers) == 2
    # one region asked for by two threads at once: one reader, each frame decoded once
    FR.pool().clear()
    n0, s0 = sc.decoded, sc.seeks
    peak[0] = 0
    same = [None, None]

    def again(k):
        same[k] = sc.piece(4, boxes[0])
    ths = [threading.Thread(target=again, args=(k,)) for k in range(2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    assert np.array_equal(same[0], same[1]) and peak[0] == 1
    assert sc.seeks - s0 <= 1 and sc.decoded - n0 <= 5, (sc.seeks - s0, sc.decoded - n0)
    procs = [r[0]._proc for r in sc._readers.values()]
    sc.close()
    assert all(pr.poll() is not None for pr in procs) and not sc._readers


def test_pool_floor_keeps_the_newest_frames_of_a_kind_and_lets_go_when_told():
    """1.27: a kind of frames with a floor keeps its newest frames whatever the limit says (the whole frames of an 8K source:
    the limit holds three); other frames go first; unfloor makes them ordinary again."""
    from contour.core.frames import FramePool
    fr = lambda: np.zeros((10, 10, 10), np.uint8)             # 1000 bytes
    p = FramePool(2500)
    for i in range(8):
        p.put(("src", "whole", i), fr(), floor=5)
        p.put(("view", i), fr())                               # the viewer's frames of the same cut: no floor
    assert [i for i in range(8) if p.get(("src", "whole", i)) is not None] == [3, 4, 5, 6, 7]
    assert sum(p.get(("view", i)) is not None for i in range(8)) == 0 and p.bytes == 5000      # over the limit by the floor alone
    p.put(("other", "whole", 0), fr())                         # no floor of its own: it is what goes
    assert p.get(("other", "whole", 0)) is not None or p.bytes == 5000
    p.unfloor(("src", "whole"))
    assert p.bytes <= 2500 and p.get(("src", "whole", 7)) is not None
    q = FramePool(2500)
    for i in range(8):
        q.put(("src", "whole", i), fr())                       # as before 1.27: the limit alone
    assert q.bytes <= 2500


def test_whole_frames_stay_while_people_of_several_frames_are_cut_from_them(vids, monkeypatch):
    """A pool too small for the whole frames (an 8K source in the default pool): three frames worked on side by side, each by
    several people, still read once each by one reader; closing the source lets the frames go."""
    import threading
    from contour.core import frames as FR, media as MD
    from contour.core.colour import PASSTHROUGH, ViewSettings, manager
    info = MD.probe(str(vids["prores"]))
    vs = ViewSettings(input_cs=PASSTHROUGH)
    small = FR.FramePool(int(info.width * info.height * 6 * 1.5))        # room for one whole frame and a half
    monkeypatch.setattr(FR, "_pool", small)
    sc = FR.SourceCrops(info, manager(), vs)
    boxes = [(4, 4, 60, 60), (230, 150, 300, 230), (100, 60, 200, 180)]
    full = [_full(info, i, manager(), vs) for i in range(6)]
    got, errs = {}, []

    def one(k, i):
        try:
            got[(k, i)] = sc.piece(i, boxes[k])
        except Exception as e:                      # pragma: no cover
            errs.append(e)
    for i0 in (0, 3):                                           # frames 0..2 side by side, then 3..5
        one(0, i0)                                              # (the render's own frame is asked first, the ones ahead after it)
        ths = [threading.Thread(target=one, args=(k, i)) for i in (i0 + 2, i0, i0 + 1) for k in range(3)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(60)
    assert not errs, errs
    for (k, i), a in got.items():
        b = boxes[k]
        assert np.array_equal(a, full[i][b[1]:b[3], b[0]:b[2]]), (k, i)
    assert sc.seeks == 1                                        # never back for a frame that had been dropped
    assert small.bytes > small.limit
    sc.close()
    assert small.bytes <= small.limit
