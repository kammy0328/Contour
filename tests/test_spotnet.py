"""Spot detector learned from hand results (experiment): labels from the hand's removals, away from the hairline,
and a network that learns them better than nothing on held-out samples."""
import numpy as np
import cv2
import pytest

from contour.learn import skinnet as SN, spotnet as SP


def _sample(seed, n=6):
    rng = np.random.default_rng(seed)
    H = 256
    yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, H)).astype(np.float32), (0, 0), 1.5)
    Y = 0.55 + 0.01 * tex / tex.std()
    spots = np.zeros((H, H), np.float32)
    for _ in range(n):
        x, y = rng.uniform(60, 196, 2)
        spots += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.5 ** 2))
    b = np.stack([Y * 1.12, Y * 0.95, Y * 0.82], -1) * (1 - 0.15 * spots[..., None])
    a = np.stack([Y * 1.12, Y * 0.95, Y * 0.82], -1)            # the hand took the spots out
    skin = (((xx - 128) / 110) ** 2 + ((yy - 128) / 120) ** 2 < 1).astype(np.float32)
    return dict(before=np.clip(b, 0, 1).astype(np.float32), delta=(a - b).astype(np.float32), skin=skin,
                meta=dict(shot=seed, pid=0, t=0, wf=400.0)), spots


def test_labels_find_the_removed_spots_and_not_the_border():
    s, spots = _sample(1)
    L = SP.labels(s)
    n, cc, st, _ = cv2.connectedComponentsWithStats((L["lab"] == 1).astype(np.uint8))
    assert n - 1 >= 4                                       # most of the six spots
    edge = cv2.erode((s["skin"] > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    assert not (L["lab"] > 0)[edge].any()


def test_network_learns_spots_on_held_out_samples():
    pytest.importorskip("torch")
    items = [SP.item(_sample(i)[0]) for i in range(10)]
    net, _ = SP.train(items, [8, 9], steps=200, batch=4, size=128, lr=3e-3, width=8, device="cpu", log=lambda *a: None)
    maps = [SP.predict(net, items[i], "cpu") for i in (8, 9)]
    sc = SP.score(maps, [items[8], items[9]], 0.5)
    assert sc["found"] > 0.5, sc
