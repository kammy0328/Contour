"""The learned spot detector is used on large faces only (≥ SPOTNET_WF source px), on the rule scale, inside the
skin, and the rule detector stays where it is off or fails."""
import types

import numpy as np

from contour.retouch import detect as DT
from contour.retouch import engine as RE
from contour.retouch import spotnet_run as SR


class _Sess:
    """Stands in for the ONNX session: a high 'red' logit at one place of the net's picture."""
    def __init__(self, yx):
        self.yx = yx
        self.shape = None

    def get_inputs(self):
        return [types.SimpleNamespace(name="x")]

    def run(self, _o, feed):
        x = feed["x"]
        self.shape = x.shape
        lg = np.full((1, 3) + x.shape[2:], -8.0, np.float32)
        y, xx = self.yx
        lg[0, 1, y - 2:y + 3, xx - 2:xx + 3] = 8.0
        return [lg]


def test_score_scale():
    s = SR.to_score(np.array([0.0, SR.THR, 1.0]))
    assert np.allclose(s, [0.0, 0.5, 1.0])


def test_run_at_net_width_and_inside_skin():
    H, W, wf = 600, 600, 600.0                 # picture at twice the net's face width
    rgb = np.full((H, W, 3), 0.6, np.float32)
    region = np.zeros((H, W), np.float32)
    region[100:500, 100:500] = 1
    ss = _Sess((100, 100))                     # net picture 300x300: the spot at (200, 200) here
    m = SR.run(rgb, region, wf, sess=ss)
    assert ss.shape[2] % 4 == 0 and ss.shape[3] % 4 == 0 and ss.shape[2] == 300
    assert m["red"][200, 200] > 0.9
    assert m["dark"].max() < 0.01
    assert m["red"][50, 50] == 0                 # outside the skin
    rgb2 = rgb.copy()
    region2 = np.zeros_like(region)
    region2[180:220, 300:400] = 1               # the spot outside this skin: nothing
    assert SR.run(rgb2, region2, wf, sess=ss)["red"].max() < 0.01


def test_only_large_faces(monkeypatch):
    monkeypatch.setattr(SR, "available", lambda: True)
    job = types.SimpleNamespace(v={"blemish": 0.8}, wf_src=400.0)
    assert not RE._spot_net_wanted(job)
    job.wf_src = 600.0
    assert RE._spot_net_wanted(job)
    job.wf_src = SR.SPOTNET_WF - 0.5
    assert not RE._spot_net_wanted(job)
    job.wf_src = SR.SPOTNET_WF
    assert RE._spot_net_wanted(job)
    job.v = {"blemish": 0.0}
    assert not RE._spot_net_wanted(job)
    monkeypatch.setattr(SR, "available", lambda: False)
    job.v = {"blemish": 0.8}
    assert not RE._spot_net_wanted(job)


def test_detect_takes_net_spots_and_keeps_rule_on_failure():
    H = W = 200
    rng = np.random.default_rng(0)
    Y = (0.55 + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
    Y[100:104, 100:104] -= 0.15                  # a dark spot the rule detector finds
    Cb = np.zeros_like(Y)
    Cr = np.full_like(Y, 0.04)
    skin = np.zeros_like(Y)
    skin[30:170, 30:170] = 1
    rule = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0)
    assert rule["dark"].max() > 0.2
    net_map = np.zeros_like(Y)
    net_map[60:63, 60:63] = 0.9
    got = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0,
                    net=lambda region, size: {"dark": net_map, "red": np.zeros_like(Y), "bump": np.zeros_like(Y)})
    assert got["dark"][61, 61] > 0.8 and got["dark"][102, 102] < 0.01
    assert np.allclose(got["protect"], rule["protect"])

    def boom(region, size):
        raise RuntimeError("no gpu")
    bad = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0, net=boom)
    assert np.allclose(bad["dark"], rule["dark"])


class _EdgeSess(_Sess):
    """High 'dark' logits along a line (an edge the net took for spots) and one compact spot."""
    def run(self, _o, feed):
        x = feed["x"]
        lg = np.full((1, 3) + x.shape[2:], -8.0, np.float32)
        lg[0, 0, 50:52, 60:240] = 8.0           # a long line inside the skin
        lg[0, 0, 20:22, 60:240] = 8.0           # the skin's border (region starts at 20 in the net picture)
        lg[0, 0, 150:154, 150:154] = 8.0        # a spot
        return [lg]


def test_edges_and_lines_are_not_spots():
    H = W = 600
    rgb = np.full((H, W, 3), 0.6, np.float32)
    region = np.zeros((H, W), np.float32)
    region[40:560, 40:560] = 1
    m = SR.run(rgb, region, 600.0, sess=_EdgeSess((0, 0)))
    assert m["dark"][302, 302] > 0.9            # the spot
    assert m["dark"][101, 200:400].max() < 0.01  # the line
    assert m["dark"][41, 200:400].max() < 0.01   # the border



class _SquareSess(_Sess):
    """A 20 x 20 px detection (in the picture: the net works at the same scale when wf <= NET_WF)."""
    def run(self, _o, feed):
        x = feed["x"]
        lg = np.full((1, 3) + x.shape[2:], -8.0, np.float32)
        lg[0, 0, 100:120, 100:120] = 8.0
        return [lg]


def test_user_spot_size_limits_the_net():
    """잡티 크기: a 20 x 20 px mark goes at a largest diameter of 6 px and stays at 31 px (review of 1.21.0: the
    net's marks ignored the slider)."""
    H = W = 240
    rgb = np.full((H, W, 3), 0.6, np.float32)
    region = np.zeros((H, W), np.float32)
    region[20:220, 20:220] = 1
    wf = 300.0                                  # the model's own largest spot: 0.08 * 300 = 24 px
    m6 = SR.run(rgb, region, wf, sess=_SquareSess((0, 0)), max_size=6.0)
    m31 = SR.run(rgb, region, wf, sess=_SquareSess((0, 0)), max_size=31.0)
    m_ = SR.run(rgb, region, wf, sess=_SquareSess((0, 0)))
    assert m6["dark"][110, 110] < 0.01
    assert m31["dark"][110, 110] > 0.9 and m_["dark"][110, 110] > 0.9
    # the model's own largest still holds over a larger user size (an edge the size of a cheek is no spot)
    m_small_face = SR.run(rgb, region, 200.0, sess=_SquareSess((0, 0)), max_size=31.0)   # own: 16 px
    assert m_small_face["dark"][110, 110] < 0.01


def test_detect_passes_the_spot_size_to_the_net():
    H = W = 120
    Y = np.full((H, W), 0.55, np.float32)
    Cb, Cr = np.zeros_like(Y), np.full_like(Y, 0.04)
    skin = np.zeros_like(Y)
    skin[10:110, 10:110] = 1
    got = {}

    def net(region, size):
        got["size"] = size
        return None
    DT.detect(Y, Cb, Cr, skin, None, 150.0, 7.5, net=net)
    assert got["size"] == 7.5


def test_engine_net_gets_the_slider_size(monkeypatch):
    seen = {}
    monkeypatch.setattr(SR, "available", lambda: True)
    monkeypatch.setattr(SR, "run", lambda rgb, region, wf, sess=None, max_size=None: seen.update(size=max_size, wf=wf))
    job = types.SimpleNamespace(v={"blemish": 0.8}, wf_src=600.0, blemish_size=0.2)
    f = RE._spot_net(job, np.zeros((10, 10, 3), np.float32), 300.0)
    f(np.ones((10, 10), np.float32), RE.spot_size(job, 300.0))
    assert seen["size"] == RE.spot_size(job, 300.0) and seen["wf"] == 300.0


def test_model_that_fails_to_load_leaves_the_rule_detector(monkeypatch):
    from contour.track import models as MD
    monkeypatch.setattr(SR, "_sess", None)
    monkeypatch.setattr(SR, "_failed", False)

    def bad(path):
        raise RuntimeError("no provider")
    monkeypatch.setattr(MD, "session", bad)
    H = W = 160
    rng = np.random.default_rng(3)
    Y = (0.55 + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
    Y[78:82, 78:82] -= 0.15
    Cb, Cr = np.zeros_like(Y), np.full_like(Y, 0.04)
    skin = np.zeros_like(Y)
    skin[20:140, 20:140] = 1
    rgb = np.dstack([Y, Y, Y])
    rule = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0)
    got = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0, net=lambda region, size: SR.run(rgb, region, 150.0, max_size=size))
    assert np.allclose(got["dark"], rule["dark"])
    assert not SR.available()                    # and it is not tried again
