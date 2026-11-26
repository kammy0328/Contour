"""1.26.2 (LNGSHOT BEEP BEEP QA, 11/26, 4K close-ups with 요철 and 턱주름 at 100): the chin's stages leave the lip-chin crease and the
chin's outline alone (CHIN_KEEP_LIP, CHIN_KEEP_EDGE), and the raised-spot fill looks for bumps on the skin mask only, a bump's size inside
its edge (FINAL_BUMP_EDGE: a beanie's knit above the brows and the light side of an outline were filled as bumps)."""
import numpy as np

from contour.retouch import engine as RE


def _face(cx=300.0, w=300.0):
    """68 landmarks of a frontal face of width w (only what the zones read: the jaw, the brows, the lips, the chin): the lower lip's
    bottom at y 400, the chin's bottom at 500 (a third of the face width below it, as on a real face)."""
    lm = np.full((68, 3), np.nan, np.float32)
    a_ = np.linspace(np.pi, 0, 17)
    lm[0:17, 0] = cx - 0.5 * w * np.cos(a_)
    lm[0:17, 1] = 300.0 + 200.0 * np.sin(a_)               # the jaw: from the ears (y 300) down to the chin (y 500)
    lm[17:27, 0] = np.linspace(cx - 0.4 * w, cx + 0.4 * w, 10)
    lm[17:27, 1] = 200.0
    t_ = np.linspace(np.pi, -np.pi, 12, endpoint=False)
    lm[48:60, 0] = cx + 0.17 * w * np.cos(t_)
    lm[48:60, 1] = 385.0 + 15.0 * np.sin(t_)
    lm[57, :2] = (cx, 400.0)
    lm[8, :2] = (cx, 500.0)
    lm[:, 2] = 1.0
    return lm


def test_chin_zone_keeps_off_the_lip_crease_and_the_outline(monkeypatch):
    lm = _face()
    shape = (700, 600)
    z = RE.chin_zone(shape, lm[:, :2], 300.0)
    ball = z[465, 300]                    # the chin's ball, where it pebbles
    crease = z[430, 300]                  # the lip-chin crease, a tenth of the face width under the lip
    edge = z[497, 300]                    # at the chin's outline
    assert ball > 0.5, ball
    assert crease < 0.2 and edge < 0.2, (crease, edge)
    monkeypatch.setattr(RE, "CHIN_KEEP_LIP", 0.0)
    monkeypatch.setattr(RE, "CHIN_KEEP_EDGE", 0.0)
    z0 = RE.chin_zone(shape, lm[:, :2], 300.0)
    assert z0[430, 300] > crease + 0.3 and z0[497, 300] > edge + 0.3     # as before: both were in
    assert abs(float(z0[465, 300]) - float(ball)) < 0.1


def _bump(Y, x, y, r=2.0, a=0.12):
    """A raised spot: a small light top and its darker side beside it."""
    yy, xx = np.mgrid[0:Y.shape[0], 0:Y.shape[1]].astype(np.float32)
    top = a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))
    side = 0.06 * np.exp(-((xx - x - 2 * r) ** 2 + (yy - y) ** 2) / (2 * r * r))
    return (Y * (1 + top - side)).astype(np.float32)


def test_bumps_are_looked_for_inside_the_skin_only():
    rng = np.random.default_rng(3)
    Y = np.full((300, 300), 0.5, np.float32) + rng.normal(0, 0.002, (300, 300)).astype(np.float32)
    region = np.zeros((300, 300), np.float32)
    region[60:240, 60:240] = 1.0
    wf = 250.0
    Y = _bump(Y, 150, 150)            # well inside
    Y = _bump(Y, 64, 150)             # on the region's edge (its top inside, by 4 px)
    out = RE._final_bumps(Y, region, wf, None, None)
    assert out is not None
    m, _ = out
    assert m[150, 150] > 0.5
    assert m[150, 64] < 0.1 and m[150, 66] < 0.1
