"""Review 6c3dcb3 (1.19 skin): (1) a strong red spot was detected and then fully protected as "another colour" (from
Cr +0.06 nothing was removed); (2) the original pieces of several faces were read with one decoder that restarted at
every face; (3) the neck's bump slider alone did not use the bump detection."""
import os
import subprocess
from collections import defaultdict

import numpy as np
import pytest

from test_detect import _skin, _g, WF, SIZE


def _red_removed(amp, cb_amp=0.0, s=3.0):
    from contour.retouch import detect as D, engine as R
    Y, Cb, Cr, m, xx, yy = _skin(seed=2, grain=.006)
    g = _g(xx, yy, 180, 180, s)
    c = Cr + amp * g
    b = Cb + cb_amp * g
    job = R.Job(0, 1, defaultdict(float, blemish=1.), .5, 'face', 1., None, 1., 300., masks={'skinx': m}, wf_src=WF,
                blemish_size=.5)
    rf = R.RetouchFrame(0, [job], (360, 360), (360, 360))
    _, _, co = R.process_ycc(Y, b, c, rf, 1.)
    return float((c[180, 180] - co[180, 180]) / amp), D.detect(Y, b, c, m, None, WF, SIZE)


@pytest.mark.parametrize("amp", [0.06, 0.08, 0.10])
def test_strong_red_spot_is_removed(amp):
    share, d = _red_removed(amp, -0.35 * amp)
    assert d["red"][180, 180] > 0.9
    assert share > 0.6, (amp, share, float(d["protect"][180, 180]))
    share, d = _red_removed(amp)                       # red with no blue change, as in the review
    assert share > 0.6, (amp, share)


def test_paint_and_other_hues_stay_protected():
    from contour.retouch import detect as D
    Y, Cb, Cr, m, xx, yy = _skin(seed=3, grain=.006)
    g1, g2, g3 = _g(xx, yy, 100, 120, 3.0), _g(xx, yy, 250, 120, 3.0), _g(xx, yy, 180, 260, 14.0)
    cr = Cr + 0.22 * g1 + 0.08 * (g3 > 0.5)          # a painted red dot; a wide red painted patch
    cb = Cb + 0.08 * g2                               # a blue thread, a blue gem
    d = D.detect(Y, cb, cr, m, None, WF, SIZE)
    assert d["protect"][120, 100] > 0.8, "far redder than any pimple: a thing"
    assert d["protect"][120, 250] > 0.8, "another hue: a thing"
    assert d["protect"][260, 180 + 12] > 0.5, "red over a wide patch: paint, not a pimple"


def test_faces_read_alternately_do_not_restart_the_decoder(tmp_path):
    from contour.core import media as M, frames as F, settings as S
    from contour.core.colour import manager, ViewSettings, PASSTHROUGH
    S._settings = S.AppSettings(data_dir=str(tmp_path / 'data'))
    try:
        src = tmp_path / 'src.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=640x480:rate=24', '-frames:v', '8',
                        '-c:v', 'libx264', '-threads', '2', str(src)], check=True)
        info = M.probe(str(src))
        F.pool().clear()
        s = F.SourceCrops(info, manager(), ViewSettings(input_cs=PASSTHROUGH))
        a, b = (32, 32, 160, 160), (320, 160, 448, 288)
        try:
            pa = [s.piece(i, a) for i in range(6)]
            F.pool().clear()
            s.close()
            s = F.SourceCrops(info, manager(), ViewSettings(input_cs=PASSTHROUGH))
            for i in range(6):
                x = s.piece(i, a)
                s.piece(i, b)
                assert np.array_equal(np.asarray(x), np.asarray(pa[i])), "the same picture as read alone"
            assert s.seeks <= 3, s.seeks                 # was 12: a restart at every face
            assert s.decoded <= 14
        finally:
            s.close()
            F.pool().clear()
    finally:
        S._settings = None


def test_neck_bump_slider_alone_uses_the_detection():
    from contour.retouch import engine as R
    Y0, Cb, Cr, m, xx, yy = _skin(seed=13)
    Y = (Y0 * (1 + 0.10 * _g(xx, yy, 180, 180, 2.0))).astype(np.float32)
    import cv2
    sm = lambda a: cv2.GaussianBlur(a, (0, 0), 1.2)

    def left(on):
        v = {k: 0.0 for k in R.RETOUCH_KEYS}
        v["neck_bumps"] = 1.0
        job = R.Job(0, 1, v, 0.45, "neck", 1.0, None, 1.0, 600.0, masks={"neck": m}, wf_src=WF, blemish_size=0.5)
        old = R.DET_ON
        R.DET_ON = on
        try:
            Y1, _, _ = R.process_ycc(Y, Cb, Cr, R.RetouchFrame(0, [job], (360, 360), (360, 360)), 1.0)
        finally:
            R.DET_ON = old
        return abs(float(sm(Y1 - Y0)[180, 180])) / abs(float(sm(Y - Y0)[180, 180]))
    assert left(True) < 0.5, left(True)
