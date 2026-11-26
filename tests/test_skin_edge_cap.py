"""1.26.2 (LNGSHOT QA 11/26, 4K close-ups): the skin stage's bands softened the light falling off at the jaw's and the cheek's outline (a
light line inside the jaw in the change). FS_EDGE_GRAD: where the light changes steeply over FS_EDGE_SIG of the face width, the wider
bands are not scaled; the texture elsewhere is smoothed as before."""
import numpy as np

from contour.retouch import skin as SK


def _run(monkeypatch, grad, tex=True, edge=True):
    monkeypatch.setattr(SK, "FS_EDGE_GRAD", grad)
    rng = np.random.default_rng(1)
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    I = np.full((h, w), 0.55, np.float32)
    if edge:
        I = np.where(xx < 150, 0.55, 0.30).astype(np.float32)          # the jaw's light falling off
    if tex:
        I = I + 0.012 * np.sin(xx * 0.9) * np.sin(yy * 0.8)            # skin texture, a few px
    I = (I + rng.normal(0, 0.002, I.shape)).astype(np.float32)
    SK._TLS.wf = 400.0
    out = SK.smooth_channel(I, np.ones((h, w), np.float32), 0.7, 0.0, int(0.022 * 400), relative=True)
    d = out - I
    return float(np.abs(d[:, 135:165]).mean()), float(np.abs(d[:, 40:100]).mean())


def test_the_edge_keeps_its_contrast_the_texture_is_smoothed_as_before(monkeypatch):
    e0, _ = _run(monkeypatch, None, tex=False)
    e1, _ = _run(monkeypatch, (0.03, 0.08), tex=False)
    assert e1 < 0.35 * e0, (e0, e1)
    _, t0 = _run(monkeypatch, None, edge=False)
    _, t1 = _run(monkeypatch, (0.03, 0.08), edge=False)
    assert t0 > 1e-3 and abs(t1 - t0) < 0.05 * t0, (t0, t1)
