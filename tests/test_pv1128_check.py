"""The check of the 11/28 (client's 1st PV feedback) renders: a neck seen from below, the lip flakes by the teeth, the face's skin the colour model missed.

  - neck_mask: a head thrown back is as short from the brows to the chin as a head bowed, and the bowed head's guard (under the jaw is the chest
    then) took the whole neck out (LNGSHOT BEEP BEEP c034 f40). Where the parser is sure of the neck it stays (NECK_UP_PARSED).
  - _lip_flakes: with the mouth open, the teeth's tips by the lips' inner edge (inside the lips' zone when the inner lip points sit a few pixels into
    the teeth) were taken down as light flakes (c046 f5 / f8). A light speck that runs on out of the zone, or stands out far more than a flake, stays.
  - lip_zone: an open mouth's opening is kept LIP_OPEN_MARGIN of the face width out of the zone."""
import numpy as np
import cv2

from contour.retouch import engine as RE


def _lm_short(h=420, w=360, brows_y=200.0):
    lm = np.zeros((68, 3), np.float32)
    for i in range(17):
        a = np.pi * i / 16
        lm[i, :2] = (180 - 100 * np.cos(a), 120 + 140 * np.sin(a))
    lm[17:27, :2] = [(100 + 18 * k, brows_y) for k in range(10)]
    lm[27:, 0] = 180
    lm[27:, 1] = np.linspace(110, 230, 41)
    return lm


def test_a_neck_under_a_short_face_stays_where_the_parser_is_sure():
    h, w = 420, 360
    lm = _lm_short()
    prob = np.ones((h, w), np.float32)
    parse = dict(neck=np.zeros((h, w), np.float32), skin=np.zeros((h, w), np.float32), cover=np.ones((h, w), np.float32), trust=1.0)
    parse["neck"][270:420, 90:270] = 1.0                       # the parser's neck below the jaw
    old = RE.NECK_UP_PARSED
    try:
        RE.NECK_UP_PARSED = False
        m0 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
        RE.NECK_UP_PARSED = True
        m1 = RE.neck_mask(lm, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
        tall = _lm_short(brows_y=90.0)
        RE.NECK_UP_PARSED = False
        t0 = RE.neck_mask(tall, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
        RE.NECK_UP_PARSED = True
        t1 = RE.neck_mask(tall, None, None, (h, w), 1.0, 1.0, prob, None, 1.5, None, None, parse)
    finally:
        RE.NECK_UP_PARSED = old
    assert float(m0.max()) < 1e-6, "before: a face this short had no neck at all"
    assert float(m1[300:360, 150:210].mean()) > 0.5, "now: the parser's sure neck is the neck"
    assert float(m1[:255, :].max()) < 0.05, "nothing above the jaw"
    assert float(np.abs(t1 - t0).max()) < 1e-6, "a face of the usual height is as before"


def _lips_picture(h=200, w=300):
    Y = np.full((h, w), 0.35, np.float32)
    Y[80:100, 60:240] = 0.70                                       # the teeth (in the mouth's shade)
    for x in range(70, 230, 20):
        Y[100:108, x:x + 6] = 0.48                                 # their tips, by the lower lip's inner edge
    Y[150:200, :] = 0.45                                           # the skin under the lip
    flakes = [(100, 130), (150, 125), (200, 135), (120, 114)]
    for (cx, cy) in flakes:
        cv2.circle(Y, (cx, cy), 3, 0.43, -1)
    Y = cv2.GaussianBlur(Y, (0, 0), 0.8)
    z = np.zeros((h, w), np.float32)
    z[103:152, 40:260] = 1                                         # the zone: a few pixels into the tips (the inner lip points' error)
    z = cv2.GaussianBlur(z, (0, 0), 1.2)
    return Y, z, flakes


def test_lip_flakes_leave_the_teeth_and_take_the_flakes():
    Y, z, flakes = _lips_picture()
    h, w = Y.shape
    Cr = np.full((h // 2, w // 2), 0.6, np.float32)
    zc = cv2.resize(z, (w // 2, h // 2))
    old = RE.LIP_FLAKES_TOP
    try:
        RE.LIP_FLAKES_TOP = None
        y0, _ = RE._lip_flakes(Y, Cr, z, zc, 600.0, 0.5, 1.0)
        RE.LIP_FLAKES_TOP = old
        y1, _ = RE._lip_flakes(Y, Cr, z, zc, 600.0, 0.5, 1.0)
    finally:
        RE.LIP_FLAKES_TOP = old
    d0, d1 = y0 - Y, y1 - Y
    assert float(d0[98:110, 68:232].min()) < -0.05, "before: the teeth's tips were darkened"
    assert float(d1[98:110, 68:232].min()) > -0.015, "now: they stay"
    for (cx, cy) in flakes:
        a, b = float(d0[cy - 1:cy + 2, cx - 1:cx + 2].mean()), float(d1[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        assert b < -0.03 and abs(a - b) < 0.005, "the flakes on the lip are taken as before"


def test_lip_zone_keeps_an_open_mouths_opening_out():
    lm = np.zeros((68, 2), np.float32)
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    lm[48:60] = np.c_[150 + 70 * np.cos(ang), 100 + 45 * np.sin(ang)]
    ang8 = np.linspace(0, 2 * np.pi, 9)[:-1]
    lm[60:68] = np.c_[150 + 40 * np.cos(ang8), 100 + 12 * np.sin(ang8)]
    wf = 600.0
    old = RE.LIP_OPEN_MARGIN
    try:
        RE.LIP_OPEN_MARGIN = 0.0
        z0 = RE.lip_zone(lm, (200, 300), wf)
        RE.LIP_OPEN_MARGIN = old
        z1 = RE.lip_zone(lm, (200, 300), wf)
    finally:
        RE.LIP_OPEN_MARGIN = old
    assert z0 is not None and z1 is not None
    gap = 100 + 12 + int(0.6 * old * wf)                             # inside the margin, below the opening
    assert float(z0[gap, 150]) > 0.5 and float(z1[gap, 150]) < 0.5, "the margin round the opening is out of the zone now"
    assert float(z1[100 + 33, 150]) > 0.5, "the lip itself is still in"
