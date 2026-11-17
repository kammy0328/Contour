"""2611 cut 024: '강하게' (redness 30) took the blush off the cheeks. The redness step evens out small red patches
and leaves a broad cheek colour alone."""
import numpy as np


def _face(w=400, h=400):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    Cr = np.full((h, w), 0.04, np.float32)
    Cb = np.full((h, w), -0.03, np.float32)
    blush = np.exp(-(((xx - 110) ** 2 + (yy - 220) ** 2) / (2 * 38.0 ** 2)))      # broad, on the left cheek
    Cr += 0.05 * blush
    spot = np.exp(-(((xx - 290) ** 2 + (yy - 200) ** 2) / (2 * 4.0 ** 2)))        # a small red spot, right cheek
    Cr += 0.05 * spot
    Y = np.full((h, w), 0.6, np.float32)
    m = np.zeros((h, w), np.float32)
    m[40:360, 40:360] = 1.0
    return Y, Cb, Cr, m, blush, spot


def test_blush_kept_spot_evened():
    from contour.retouch import engine as RE
    Y, Cb, Cr, m, blush, spot = _face()
    wf = 320.0
    Y1, Cb1, Cr1 = RE._redness(Cb, Cr, Y, m, m, 1.0, wf, 1.0)          # k = 1 (strongest)
    b = blush > 0.6
    s = spot > 0.6
    blush0 = float((Cr[b] - 0.04).mean())
    blush1 = float((Cr1[b] - 0.04).mean())
    spot0 = float((Cr[s] - 0.04).mean())
    spot1 = float((Cr1[s] - Cr1[(spot < 0.01) & (blush < 0.01) & (m > 0)].mean()).mean())
    assert blush1 > 0.8 * blush0, f"the cheek colour is kept ({blush1:.4f} of {blush0:.4f})"
    assert spot1 < 0.5 * spot0, f"the small red spot is evened out ({spot1:.4f} of {spot0:.4f})"
