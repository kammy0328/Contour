"""Skin retouch network (2026-11-18, learned from LNGSHOT Bananas BEFORE/AFTER): what a sample keeps of the hand's
retouch and what it leaves out (grade, liquify, skin tone, cleaning, spots, grain), the split by shots, and that the
network learns the hand's smoothing on held-out samples better than doing nothing."""
import numpy as np
import cv2
import pytest

from contour.learn import skinnet as SN

S = 900


def _face(seed=0, spot=True, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, wf = 450.0 + shift[0], 380.0 + shift[1], 300.0
    lm = np.zeros((68, 2), np.float32)
    ang = np.linspace(np.pi, 0, 17)
    lm[:17] = np.stack([cx + 0.5 * wf * np.cos(ang), cy + 0.6 * wf * np.sin(ang)], 1)
    lm[17:27] = np.stack([np.linspace(cx - 0.4 * wf, cx + 0.4 * wf, 10), np.full(10, cy - 0.25 * wf)], 1)
    lm[27:36] = np.stack([np.full(9, cx), np.linspace(cy - 0.2 * wf, cy + 0.15 * wf, 9)], 1)
    lm[36:48] = np.stack([np.r_[np.linspace(cx - 0.3 * wf, cx - 0.1 * wf, 6), np.linspace(cx + 0.1 * wf, cx + 0.3 * wf, 6)],
                          np.full(12, cy - 0.12 * wf)], 1)
    lm[48:68] = np.stack([np.linspace(cx - 0.18 * wf, cx + 0.18 * wf, 20), np.full(20, cy + 0.3 * wf)], 1)
    skin = (((xx - cx) / (0.5 * wf)) ** 2 + ((yy - cy - 0.1 * wf) / (0.75 * wf)) ** 2 < 1).astype(np.float32)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 3.0)
    tex = 0.035 * tex / tex.std()
    grain = 0.01 * rng.normal(0, 1, (S, S)).astype(np.float32)
    base = 0.55 + 0.0 * xx
    Y = base + tex * skin + grain
    if spot:
        Y = Y - 0.12 * np.exp(-((xx - cx - 60) ** 2 + (yy - cy - 40) ** 2) / (2 * 4.0 ** 2))
    b = np.clip(np.stack([Y * 1.12, Y * 0.95, Y * 0.82], -1), 0, 1)
    # the hand: texture smoothed by 70 %, a skin tone lift, the spot taken out, a grade over everything
    Ya = base + 0.3 * tex * skin + (1 - skin) * tex * 0 + grain
    tone = 0.05 * skin
    a = np.stack([(Ya + tone) * 1.12, (Ya + tone) * 0.95, (Ya + tone) * 0.82], -1)
    a = np.clip(a * [1.04, 1.0, 0.97], 0, 1).astype(np.float32)
    return b.astype(np.float32), a, lm, skin, tex


def test_sample_keeps_the_smoothing_and_drops_tone_grade_and_spot():
    from contour.learn import style as SL
    b, a, lm, skin, tex = _face()
    G = SL.fit_grade(b, a)
    s = SN.make_sample(b, a, lm, skin, G, spots=lambda o, loc, m: _spot_at(o.shape, loc))
    assert s is not None
    w = s["weight"] > 0.5
    # the hand's change on the skin is (about) minus 70 % of the texture: the sample's delta follows it
    cx, cy, side, scale = SN.crop_geom(lm)
    want = SN.cut_square((-0.7 * tex)[..., None].repeat(3, -1), cx, cy, side, SN.CROP) @ SN.K
    got = s["delta"] @ SN.K
    r = np.corrcoef(cv2.GaussianBlur(want, (0, 0), 1)[w], cv2.GaussianBlur(got, (0, 0), 1)[w])[0, 1]
    assert r > 0.8, r
    assert abs(float(got[w].mean())) < 0.01, "the skin tone lift is not learned"
    assert s["weight"][_spot_px(lm)].max() == 0, "the spot is DIAP's own"
    assert float(s["keep"][w].mean()) < 1.0, "the hand smoothed fine texture away"


def _spot_px(lm):
    cx, cy, side, scale = SN.crop_geom(lm)
    x = int((450 + 60 - (cx - side / 2)) * scale)
    y = int((380 + 40 - (cy - side / 2)) * scale)
    return slice(y - 2, y + 3), slice(x - 2, x + 3)


def _spot_at(shape, loc):
    m = np.zeros(shape[:2], bool)
    # loc is the landmarks in crop px: the spot is at (cx + 60, cy + 40) of the face
    cx = float(loc[:17, 0].mean())
    cy = float(loc[0, 1])
    yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
    m[(xx - cx - 60) ** 2 + (yy - cy - 40) ** 2 < 14 ** 2] = True
    return m


def test_liquify_and_cleaning_are_not_learned():
    from contour.learn import style as SL
    b, a, lm, skin, tex = _face(seed=3, spot=False)
    # cleaning: a patch painted over on the cheek; liquify: the hand result moved by 2 px on the jaw side
    a2 = a.copy()
    a2[430:470, 380:420] = 0.3
    M = np.float32([[1, 0, 2.0], [0, 1, 0]])
    a2 = cv2.warpAffine(a2, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    s = SN.make_sample(b, a2, lm, skin, SL.fit_grade(b, a2))
    cx, cy, side, scale = SN.crop_geom(lm)
    y0, x0 = int((430 - (cy - side / 2)) * scale), int((380 - (cx - side / 2)) * scale)
    patch = (slice(y0 + 3, y0 + int(40 * scale) - 3), slice(x0 + 3, x0 + int(40 * scale) - 3))
    assert s["weight"][patch].max() == 0, "the painted patch is cleaning"
    w = s["weight"] > 0.5
    want = SN.cut_square((-0.7 * tex)[..., None].repeat(3, -1), cx, cy, side, SN.CROP) @ SN.K
    r = np.corrcoef(cv2.GaussianBlur(want, (0, 0), 1)[w], cv2.GaussianBlur(s["delta"] @ SN.K, (0, 0), 1)[w])[0, 1]
    assert r > 0.6, r


def test_split_keeps_runs_of_shots_together():
    shots = list(range(60))
    val = SN.split_by_shot(shots, 0.2, 6, 0)
    assert 6 <= len(val) <= 18
    for s in val:
        blk = s // 6
        assert all(x in val for x in range(blk * 6, min(60, blk * 6 + 6)))


def test_network_learns_the_hand_smoothing_on_held_out_samples():
    torch = pytest.importorskip("torch")
    from contour.learn import style as SL
    samples = []
    for i in range(8):
        b, a, lm, skin, tex = _face(seed=10 + i, spot=False, shift=(i * 7.0, -i * 5.0))
        s = SN.make_sample(b, a, lm, skin, SL.fit_grade(b, a))
        for k in ("before", "skin", "delta", "keep", "weight"):
            s[k] = cv2.resize(s[k], (160, 160), interpolation=cv2.INTER_AREA)
        s["weight"] = (s["weight"] > 0.5).astype(np.float32)
        samples.append(s)
    net, hist = SN.train(samples, [6, 7], steps=150, batch=4, size=128, lr=3e-3, width=8, device="cpu",
                         log=lambda *a: None)
    ev = SN.evaluate(net, samples, [6, 7], "cpu")
    assert ev["net"]["err"] < 0.8 * ev["none"]["err"], ev


def test_saved_sample_reads_back_compact_and_full(tmp_path):
    from contour.learn import style as SL
    b, a, lm, skin, tex = _face(seed=4, spot=False)
    s = SN.make_sample(b, a, lm, skin, SL.fit_grade(b, a), diap=b)
    SN.save_sample(tmp_path / "x.npz", s)
    full = SN.load_sample(tmp_path / "x.npz")
    comp = SN.load_sample(tmp_path / "x.npz", compact=True)
    assert comp["before"].dtype == np.uint16 and comp["weight"].dtype == np.uint8
    assert np.abs(SN._f(comp["before"]) - s["before"]).max() < 1e-4
    assert np.abs(full["delta"] - s["delta"]).max() < 1e-3
    assert np.array_equal(SN._f(comp["weight"]) > 0.5, s["weight"] > 0.5)
    x, y, w = SN._batch([comp], [0], None, SN.CROP, False)
    x2, y2, w2 = SN._batch([full], [0], None, SN.CROP, False)
    assert np.abs(x - x2).max() < 1e-4 and np.abs(y - y2).max() < 1e-3


def test_export_needs_no_extra_package(tmp_path, monkeypatch):
    """j2742: torch 2.14's default exporter wants onnxscript, which the app does not ship."""
    torch = pytest.importorskip("torch")
    import builtins
    real = builtins.__import__

    def no_onnxscript(name, *a, **k):
        if name.split(".")[0] == "onnxscript":
            raise ModuleNotFoundError("No module named 'onnxscript'")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_onnxscript)
    net = SN.build_net(8)
    SN.export_onnx(net, tmp_path / "n.onnx")
    import onnxruntime as ort
    s = ort.InferenceSession(str(tmp_path / "n.onnx"), providers=["CPUExecutionProvider"])
    y = s.run(None, {"x": np.zeros((1, 4, 96, 128), np.float32)})[0]
    assert y.shape == (1, 4, 96, 128)
    assert np.allclose(y[0, 3], 1.0)                 # untrained: keep the texture as it is
