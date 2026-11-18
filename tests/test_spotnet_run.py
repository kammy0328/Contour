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
                    net=lambda region: {"dark": net_map, "red": np.zeros_like(Y), "bump": np.zeros_like(Y)})
    assert got["dark"][61, 61] > 0.8 and got["dark"][102, 102] < 0.01
    assert np.allclose(got["protect"], rule["protect"])

    def boom(region):
        raise RuntimeError("no gpu")
    bad = DT.detect(Y, Cb, Cr, skin, None, 150.0, 10.0, net=boom)
    assert np.allclose(bad["dark"], rule["dark"])
