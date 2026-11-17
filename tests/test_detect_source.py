"""1.19: where the original has more pixels than the analysis picture, the flaws are looked for on a piece of the
original (not on the reduced picture made larger): a spot two source pixels across, gone from the half-size
analysis picture, is found; the piece carries the same frame number; without the original the analysis picture is
used."""
import subprocess

import cv2
import numpy as np
import pytest


W, H, N = 640, 480, 6
SPOT = (300, 250)           # source px


def _frame(i, rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.empty((H, W, 3), np.float32)
    f[..., 0], f[..., 1], f[..., 2] = 0.72, 0.56, 0.47                 # skin
    f *= (1.0 + 0.05 * (xx / W - 0.5))[..., None]
    f += rng.normal(0, 0.02, (H, W, 1)).astype(np.float32)
    # a small dark spot that moves one pixel per frame with the "face" (so a wrong frame number shows)
    g = np.exp(-((xx - (SPOT[0] + i)) ** 2 + (yy - SPOT[1]) ** 2) / (2 * 1.0 ** 2))
    f *= (1 - 0.12 * g)[..., None]
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


@pytest.fixture
def cut4k(tmp_path):
    from contour.core import media as MD
    from contour.core.project import Project
    from contour.core.settings import settings
    MD.configure_ffmpeg()
    rng = np.random.default_rng(0)
    p = tmp_path / "big.mp4"
    cmd = [MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(p)]
    subprocess.run(cmd, input=b"".join(_frame(i, rng).tobytes() for i in range(N)), check=True)
    settings().proxy_long_side = 320                    # the analysis picture is half the original
    prj = Project.create(str(tmp_path), "big")
    c = prj.add_cut(MD.probe(str(p)), "big")
    return prj, c


def _job(px):
    from contour.retouch import engine as RE
    v = {k: 0.0 for k in RE.RETOUCH_KEYS}
    v["blemish"] = 1.0
    face = np.zeros((68, 3), np.float32)                 # source px: a face 300 px wide round the spot
    face[:, 0] = np.linspace(150, 450, 68)
    face[:, 1] = np.linspace(120, 380, 68)
    face[0, :2], face[16, :2] = (150, 250), (450, 250)
    m = np.zeros((px.h, px.w), np.float32)
    m[50:200, 60:260] = 1.0
    job = RE.Job(0, 1, v, 0.45, "face", 1.0, face, 1.0, 600.0, masks={"skinx": m}, wf_src=300.0, blemish_size=0.5)
    lm = face.copy()
    lm[:, 0] *= px.sx
    lm[:, 1] *= px.sy
    return job, lm


def _score_at(det, x_src, y_src, px):
    """The dark score at a source position, from the maps and their box (analysis px)."""
    m = det["maps"]["dark"]
    x0, y0, x1, y1 = det["box"]
    fx, fy = m.shape[1] / (x1 - x0), m.shape[0] / (y1 - y0)
    xd, yd = int(round((x_src * px.sx - x0) * fx)), int(round((y_src * px.sy - y0) * fy))
    return float(m[max(0, yd - 3):yd + 4, max(0, xd - 3):xd + 4].max()) / 255.0


def test_small_spot_found_on_the_original_piece(cut4k):
    from contour.core import frames as FR
    from contour.retouch import engine as RE
    prj, c = cut4k
    FR.pool().clear()
    px = FR.frames_for(prj, c)
    assert (px.w, px.h) == (320, 240) and px.kind == "source"
    job, lm = _job(px)
    old = RE.DET_SRC
    try:
        s0, s1 = [], []
        for t in range(N):
            rgb = px.frame(t)
            RE.DET_SRC = False
            d0 = RE._det_raw(c, px, t, job, lm, rgb)
            RE.DET_SRC = True
            d1 = RE._det_raw(c, px, t, job, lm, rgb)
            assert not d0["src"] and d1["src"]
            # the piece is the original's own pixels: twice the analysis picture's, not that picture made larger
            assert d1["maps"]["dark"].shape[1] >= 1.9 * d0["maps"]["dark"].shape[1]
            here = (SPOT[0] + t, SPOT[1])                  # the spot moves a pixel a frame: a wrong frame misses it
            s0.append(_score_at(d0, *here, px))
            s1.append(_score_at(d1, *here, px))
            if t == 0:
                first = d1
        assert s0[0] < 0.4 and s1[0] > 0.6, f"frame 0: analysis picture {s0[0]:.2f}, original piece {s1[0]:.2f}"
        assert np.mean(s1) > np.mean(s0) + 0.3, f"over the frames: {np.mean(s0):.2f} against {np.mean(s1):.2f}"
        # the maps put on the analysis picture's plane land on the spot
        job.det = first
        rf = RE.RetouchFrame(0, [job], (px.w, px.h), (W, H))
        maps = RE._det_place(job, rf, (px.h, px.w))
        xs, ys = int(round(SPOT[0] * px.sx)), int(round(SPOT[1] * px.sy))
        assert float(maps["dark"][ys - 2:ys + 3, xs - 2:xs + 3].max()) > 0.4
    finally:
        RE.DET_SRC = old
        px.close()
    assert px.__dict__["_src_crops"]._reader is None, "the piece reader ends with the cut's frames"


def test_no_original_falls_back_to_the_analysis_picture(cut4k, tmp_path):
    from contour.core import frames as FR
    from contour.core.colour import manager
    from contour.retouch import engine as RE
    from contour.warp.render import _vs_for
    from pathlib import Path
    prj, c = cut4k
    pf = FR.proxy_for(prj, c)
    assert pf.build(manager(), _vs_for(c))
    Path(c.info.path).rename(tmp_path / "gone.mp4")
    px = FR.frames_for(prj, c)
    assert px.kind == "proxy"
    job, lm = _job(px)
    d = RE._det_raw(c, px, 0, job, lm, px.frame(0))
    assert d is not None and not d["src"]
    px.close()
