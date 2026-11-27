"""(1.27.1, the user 11/27: the delivery's checks made by the app after every render) core.renderqa: the file, the frames in their places, the faces'
retouch frame by frame, what changes on one frame only."""
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from contour.core import media as MD
from contour.core import renderqa as RQ


def _frames(n=8, H=90, W=160, shift=2, seed=3):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.uniform(40, 220, (H, W + n * shift + 10)).astype(np.float32), (0, 0), 2.0)
    return [np.ascontiguousarray(base[:, 10 + i * shift:10 + i * shift + W]) for i in range(n)]


def test_a_blink_of_one_frame_is_found_and_a_steady_retouch_is_not():
    src = _frames()
    assert RQ.one_frame_changes(src, [s.copy() for s in src]) == []
    steady = [s + 5.0 for s in src]                              # the same retouch on every frame
    assert RQ.one_frame_changes(src, steady) == []
    blink = [s.copy() for s in src]
    blink[4][20:40, 60:100] += 8.0
    got = RQ.one_frame_changes(src, blink)
    assert [g["t"] for g in got] == [4]
    assert 60 <= got[0]["x"] <= 100 and 20 <= got[0]["y"] <= 40 and got[0]["level"] > 4


def test_a_face_whose_retouch_falls_out_for_a_frame():
    ser = {0: [0.5, 0.52, 0.49, 0.05, 0.5, 0.51, 0.5, None, 0.48]}
    f = RQ.series_findings(ser)
    assert any(e["kind"] == "drop" and e["t"] == 3 for e in f)
    assert not any(e["t"] in (0, 1, 2, 5, 6, 8) for e in f)
    assert RQ.series_findings({1: [0.05, 0.06, 0.05, None, 0.04]}) == [], "a face hardly retouched at all has nothing to fall out"


def test_face_series_reads_the_boxes():
    src = _frames(3)
    out = [s.copy() for s in src]
    out[1][10:20, 10:20] += 4.0
    ser = RQ.face_series(src, out, {7: [(10, 10, 20, 20), (10, 10, 20, 20), None]})
    assert ser[7][0] == 0.0 and ser[7][1] == pytest.approx(4.0, abs=0.01) and ser[7][2] is None


def _write(path, frames):
    H, W = frames[0].shape
    p = subprocess.Popen([MD.ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-framerate", "24000/1001",
                          "-i", "-", "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.clip(f + 0.5, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def test_check_of_a_file(tmp_path):
    src = _frames(10, 96, 160)
    _write(tmp_path / "src.mov", src)
    out = [s.copy() for s in src]
    out[5][20:50, 60:110] += 12.0
    _write(tmp_path / "ok.mov", src)
    _write(tmp_path / "blink.mov", out)
    _write(tmp_path / "short.mov", src[:9])
    cut = SimpleNamespace(info=MD.probe(str(tmp_path / "src.mov")), targets=lambda: [])
    prj = SimpleNamespace(cut_dir=lambda c: tmp_path / "cut")
    tr = SimpleNamespace(store=None)
    good = RQ.check(prj, cut, tr, str(tmp_path / "ok.mov"), 0, 10)
    assert good["ok"], good["problems"]
    bad = RQ.check(prj, cut, tr, str(tmp_path / "blink.mov"), 0, 10)
    assert not bad["ok"] and any("한 프레임" in p for p in bad["problems"]) and [e["t"] for e in bad["flicker"]] == [5]
    assert (tmp_path / "cut" / "qa" / "blink" / "report.json").exists() and list((tmp_path / "cut" / "qa" / "blink").glob("flicker_f00005.jpg"))
    short = RQ.check(prj, cut, tr, str(tmp_path / "short.mov"), 0, 10)
    assert any("프레임 수" in p for p in short["problems"])
