"""What the analysis models see in dark or colour-lit frames.

Detectors and pose models are trained mostly on ordinary daylight photos. A stage lit in deep blue or red, a dark
club set or an underexposed log grade makes them miss people. For analysis only (never for the output), dark frames
are lifted with a tone curve, and under one strongly coloured light the picture is turned into a clean black and
white one from the channels that carry the light, so the models get a normal looking picture.
Ordinary frames pass through untouched."""
from __future__ import annotations

import cv2
import numpy as np

DARK_TOP = 0.45      # when even the brightest 1 % of the frame is below this luma (0..1), the frame is lifted
TARGET_TOP = 0.70    # ... until they reach this
CAST = 1.6           # channel ratio beyond which the light counts as coloured


def analysis_view(rgb: np.ndarray) -> np.ndarray:
    small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    luma = small @ np.array([0.299, 0.587, 0.114], np.float32)
    # judged on the lit part of the picture: a black stage with well lit dancers is not a dark frame
    lit = luma >= np.percentile(luma, 70)
    m = small[lit].mean(0) + 1e-3
    mono = m.max() / m.min() > CAST and float(luma[lit].mean()) > 0.03
    if mono:
        # one coloured light (deep blue club light, red stage): the weak channels hold almost nothing but
        # noise, and balancing them up paints false colours and blotches. A black and white picture made
        # from the channels that carry the light (each weighted by how much it carries) is clean and looks
        # like an ordinary photo to the models.
        w = (m / m.sum()).astype(np.float32)
        f = rgb.astype(np.float32) @ w
    else:
        f = None
    sm = (small @ (m / m.sum())) if mono else luma
    y = float(np.percentile(sm, 99))
    if y < DARK_TOP:
        # the brightest parts are dim: the whole frame is underexposed. Lift so those reach a normal level,
        # in floating point (an 8-bit curve this steep would leave visible steps) and after a touch of blur
        # where the lift is strong (it lifts the sensor noise too)
        y = max(y, 0.02)
        gamma = float(np.clip(np.log(TARGET_TOP) / np.log(y), 0.3, 1.0))
        if f is None:
            f = rgb.astype(np.float32)
        if gamma < 0.7:
            f = cv2.GaussianBlur(f, (0, 0), 0.6 + 1.2 * (0.7 - gamma))
        f = 255.0 * np.power(np.clip(f / 255.0, 0, 1), gamma)
    if f is None:
        return rgb
    out = np.clip(f + 0.5, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = np.dstack([out, out, out])
    return out
