"""1.26.3: the IWALY batch's speed settings in the app (the user 11/26: "기존에 worker로 렌더했을때는 빨랐는데 왜 렌더 속도가 느려졌어?"):
people prepared side by side as the RAM allows, ffmpeg's threads for the time of a render, the skin colour models' pictures read first."""
import types

import numpy as np

from contour.core import media as M
from contour.core import perf as PF
from contour.retouch import engine as E


def test_threads_follow_the_computer(monkeypatch):
    monkeypatch.delenv("CONTOUR_PREP_THREADS", raising=False)
    monkeypatch.delenv("CONTOUR_MASK_THREADS", raising=False)
    monkeypatch.setattr(PF, "_off", lambda: False)
    assert PF.prep_threads(ram=31.7, cores=18) == 4          # the user's PC (32 GB, 18 threads): the j2847 setting
    assert PF.prep_threads(ram=24, cores=8) == 2
    assert PF.prep_threads(ram=15.8, cores=8) == 1           # a 16 GB computer: one after the other, as before
    assert PF.prep_threads(ram=64, cores=4) == 2
    assert PF.mask_threads(4, ram=31.7, cores=18) == 3
    assert PF.mask_threads(6, ram=31.7, cores=18) == 2
    assert PF.mask_threads(1, ram=31.7, cores=18) == 4
    assert PF.mask_threads(4, ram=15.8, cores=18) == 0


def test_overrides(monkeypatch):
    monkeypatch.setenv("CONTOUR_PREP_THREADS", "6")
    monkeypatch.setenv("CONTOUR_MASK_THREADS", "1")
    assert PF.prep_threads(ram=8, cores=2) == 6 and PF.mask_threads(3, ram=8) == 1
    monkeypatch.delenv("CONTOUR_PREP_THREADS")
    monkeypatch.delenv("CONTOUR_MASK_THREADS")
    monkeypatch.setattr(PF, "_off", lambda: True)              # settings: render_parallel off
    assert PF.prep_threads(ram=64, cores=32) == 1 and PF.mask_threads(2, ram=64, cores=32) == 0


def test_engine_uses_them(monkeypatch):
    monkeypatch.setattr(E, "PREP_THREADS", None)
    monkeypatch.setattr(E, "MASK_THREADS", None)
    monkeypatch.setattr(PF, "prep_threads", lambda: 3)
    monkeypatch.setattr(PF, "mask_threads", lambda n: 12 // n)
    assert E._prep_n() == 3 and E._mask_n(4) == 3
    monkeypatch.setattr(E, "PREP_THREADS", 1)
    monkeypatch.setattr(E, "MASK_THREADS", 0)
    assert E._prep_n() == 1 and E._mask_n(4) == 0


def test_reader_and_writer_threads(monkeypatch):
    got = {}

    class P:
        def __init__(self, cmd, **k):
            got["cmd"] = list(cmd)
            self.stdout = types.SimpleNamespace(read=lambda n=0: b"")
            self.stderr = types.SimpleNamespace(read=lambda n=0: b"")

        def poll(self):
            return 0

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(M.subprocess, "Popen", P)
    info = M.MediaInfo(path="x.mov", width=64, height=32, fps_num=24, fps_den=1, frames=4, chroma="rgb")
    monkeypatch.setattr(M, "READER_THREADS", 1)
    r = M.FrameReader(info, 0, 2, "rgb48le", prefetch=1)
    cmd = got["cmd"]
    assert cmd[cmd.index("-threads") + 1] == "1" and cmd.index("-threads") < cmd.index("-i")
    try:
        r.close()
    except Exception:
        pass
    monkeypatch.setattr(M, "READER_THREADS", None)
    M.FrameReader(info, 0, 2, "rgb48le", prefetch=1)
    assert "-threads" not in got["cmd"]
    monkeypatch.setattr(M, "ENCODE_THREADS", 3)
    a = M.prores_args("prores_4444_xq", prefer="software")
    if "prores_ks" in a:
        assert a[a.index("-threads") + 1] == "3"
    monkeypatch.setattr(M, "ENCODE_THREADS", None)
    a = M.prores_args("prores_4444_xq", prefer="software")
    assert "-threads" not in a


def test_render_sets_and_restores(monkeypatch):
    from contour.warp import render as R
    seen = {}

    def fake(project, cut, tracker, out, progress, cancel, ext_writer, preview):
        seen["threads"] = (M.READER_THREADS, M.ENCODE_THREADS)
        return R.RenderReport()
    monkeypatch.setattr(R, "_render_cut", fake)
    monkeypatch.setattr(R.RE, "prewarm_models", lambda cut, tr: seen.setdefault("prewarm", True) and {})
    monkeypatch.setattr(M, "READER_THREADS", None)
    monkeypatch.setattr(M, "ENCODE_THREADS", None)
    R.render_cut(None, None, object(), None)
    assert seen["threads"] == (PF.ff_decode_threads(), PF.ff_encode_threads()) == (1, 3)
    assert seen.get("prewarm") and (M.READER_THREADS, M.ENCODE_THREADS) == (None, None)


def test_prewarm_reads_what_the_colour_models_read():
    T, P_ = 40, 2
    vis = np.zeros((P_, T), np.float32)
    vis[0, 5:30] = 0.9
    vis[1, :] = 0.2                                        # never clearly in view: no model pictures
    face = np.full((P_, T, 68, 3), np.nan, np.float32)
    face[0, 5:30, :, :2] = 10.0
    st = types.SimpleNamespace(has_face=True, face_vis=vis, face=face, pids=[1, 2], has=lambda pid: pid in (1, 2), index=lambda pid: pid - 1)
    read = []
    px = types.SimpleNamespace(w=8, h=4, sx=1.0, sy=1.0, one=lambda t: read.append(t), peek=lambda t: None)
    tr = types.SimpleNamespace(store=st, proxy=px)
    cut = types.SimpleNamespace(targets=lambda: [types.SimpleNamespace(id=1), types.SimpleNamespace(id=2)])
    out = E.prewarm_models(cut, tr, workers=3)
    cand = np.nonzero(vis[0] > 0.5)[0]
    want = sorted(set(int(t) for t in cand[np.linspace(0, len(cand) - 1, 10).astype(int)]))
    assert sorted(read) == want and out["frames"] == len(want)
    tr._skin_models = {0: "made"}                         # a person whose model exists: nothing read for them
    read.clear()
    assert E.prewarm_models(cut, tr, workers=3)["frames"] == 0 and not read
