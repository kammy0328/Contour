"""4. The same retouch values must give the same picture in the preview (8-bit RGB) and in the render (the file's own
YUV planes the render reads: 10-bit 4:4:4 or 4:2:2, limited or full range). Before 1.16 the render read limited-range YUV as if it were full
range: grey RGB 51 with skin brightness 50 % became 86 in the preview and about 92-95 in the render."""
from collections import defaultdict

import numpy as np
import pytest


def _rf(h, w, **vals):
    from contour.retouch import engine as RE
    v = defaultdict(float, vals)
    job = RE.Job(p=0, pid=0, v=v, keep=0.5, scope="face", grain_size=1.0, face=None, face_w=1.0, scale=400.0,
                 masks={"tone": np.ones((h, w), np.float32)}, wf_src=100.0)
    return RE.RetouchFrame(t=0, jobs=[job], proxy_wh=(w, h), src_wh=(w, h))


def _yuv_planes(rgb_u8, bits, rng, sub="444"):
    """Encode RGB the way a BT.709 video file holds it."""
    from contour.core import media as MD
    f = rgb_u8.astype(np.float64) / 255.0
    kr, kb = 0.2126, 0.0722
    Y = kr * f[..., 0] + (1 - kr - kb) * f[..., 1] + kb * f[..., 2]
    Cb = (f[..., 2] - Y) / (2 * (1 - kb))
    Cr = (f[..., 0] - Y) / (2 * (1 - kr))
    mx = (1 << bits) - 1
    s = (mx + 1) / 256.0
    if rng == "tv":
        y, u, v = 16 * s + 219 * s * Y, 128 * s + 224 * s * Cb, 128 * s + 224 * s * Cr
    else:
        y, u, v = mx * Y, 128 * s + mx * Cb, 128 * s + mx * Cr
    dt = np.uint16 if bits > 8 else np.uint8
    pl = [np.clip(np.rint(x), 0, mx).astype(dt) for x in (y, u, v)]
    pf = {("444", 10): "yuv444p10le", ("422", 10): "yuv422p10le"}[(sub, bits)]     # what the render reads
    h, w = Y.shape
    layout = MD.plane_layout(pf, w, h)
    if sub != "444":
        import cv2
        ch, cw = pl[1].shape[0], pl[1].shape[1] // 2
        pl[1] = cv2.resize(pl[1], (cw, ch), interpolation=cv2.INTER_AREA)
        pl[2] = cv2.resize(pl[2], (cw, ch), interpolation=cv2.INTER_AREA)
    return pl, layout, pf


@pytest.mark.parametrize("bits,rng,sub", [(10, "tv", "444"), (10, "pc", "444"), (10, "tv", "422")])
@pytest.mark.parametrize("vals", [dict(skin_bright=0.5), dict(skin_bright=-0.4), dict(skin_warm=0.6),
                                  dict(skin_sat=-0.5)])
def test_render_matches_preview(bits, rng, sub, vals):
    from contour.core import media as MD
    from contour.retouch import engine as RE
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (51, 51, 51)
    rgb[:, 4:] = (200, 150, 120)                   # a skin-like colour next to the grey
    rf = _rf(h, w, **vals)
    prev = RE.apply_rgb(rgb, rf, 1.0).astype(np.float64)
    pl, layout, pf = _yuv_planes(rgb, bits, rng, sub)
    src_back = MD.yuv_to_rgb(pl, layout, "bt709", rng) * 255.0
    out = RE.apply_planes([p.copy() for p in pl], layout.subsample, layout.max_value, pf, rf, 1.0, None, rng)
    ren = MD.yuv_to_rgb(out, layout, "bt709", rng) * 255.0
    # what the file itself could hold (quantisation of the source) is allowed, nothing more
    q = np.abs(src_back - rgb).max() + 1.0
    d = np.abs(ren - prev)
    assert d.max() <= q + 1.0, f"preview {prev[0, 0]} / render {ren[0, 0]} (grey), {prev[0, -1]} / {ren[0, -1]} (skin)"


def test_reported_case():
    """Grey 51, skin brightness 50 %: preview 86; the render must agree within a code value."""
    from contour.core import media as MD
    from contour.retouch import engine as RE
    rgb = np.full((4, 4, 3), 51, np.uint8)
    rf = _rf(4, 4, skin_bright=0.5)
    prev = RE.apply_rgb(rgb, rf, 1.0)
    pl, layout, pf = _yuv_planes(rgb, 10, "tv")
    out = RE.apply_planes([p.copy() for p in pl], layout.subsample, layout.max_value, pf, rf, 1.0, None, "tv")
    ren = MD.yuv_to_rgb(out, layout, "bt709", "tv") * 255.0
    assert int(prev[0, 0, 0]) == 86
    assert abs(float(ren[0, 0, 0]) - 86) <= 1.0, f"render {ren[0, 0]}"
