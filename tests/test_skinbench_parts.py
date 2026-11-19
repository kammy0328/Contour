"""Skin benchmark on the default MediaPipe parser (instruction 2026-11-19 §1): it runs to the end with the app's
own parser, crop, landmark rules and occluders; a change inside a protected part raises its leak; a part that is not
there is 'unmeasured', never a leak of 0. feature_mask (what the app retouches around) is unchanged."""
import cv2
import numpy as np
import pytest

from contour.learn import skinbench as SB
from contour.track import faceparse as FP


def face68(cx=320.0, cy=260.0, w=180.0):
    """A plausible 68-point face (dlib order) of width w centred at (cx, cy)."""
    lm = np.zeros((68, 3), np.float64)
    a = np.linspace(np.pi * 1.0, 0.0, 17)
    lm[:17, 0] = cx - 0.5 * w * np.cos(a)
    lm[:17, 1] = cy + 0.15 * w + 0.55 * w * np.sin(a) * (np.linspace(0, 1, 17) * 0 + 1) - 0.25 * w * (1 - np.sin(a))
    lm[17:22, 0] = np.linspace(cx - 0.38 * w, cx - 0.08 * w, 5)
    lm[22:27, 0] = np.linspace(cx + 0.08 * w, cx + 0.38 * w, 5)
    lm[17:27, 1] = cy - 0.28 * w
    lm[27:31, 0] = cx
    lm[27:31, 1] = np.linspace(cy - 0.2 * w, cy + 0.05 * w, 4)
    lm[31:36, 0] = np.linspace(cx - 0.1 * w, cx + 0.1 * w, 5)
    lm[31:36, 1] = cy + 0.1 * w
    for base, ex in ((36, cx - 0.22 * w), (42, cx + 0.22 * w)):
        t = np.linspace(0, 2 * np.pi, 7)[:6]
        lm[base:base + 6, 0] = ex + 0.09 * w * np.cos(t)
        lm[base:base + 6, 1] = cy - 0.15 * w + 0.035 * w * np.sin(t)
    t = np.linspace(0, 2 * np.pi, 13)[:12]
    lm[48:60, 0] = cx + 0.2 * w * np.cos(t)
    lm[48:60, 1] = cy + 0.28 * w + 0.07 * w * np.sin(t)
    t = np.linspace(0, 2 * np.pi, 9)[:8]
    lm[60:68, 0] = cx + 0.13 * w * np.cos(t)
    lm[60:68, 1] = cy + 0.28 * w + 0.03 * w * np.sin(t)
    lm[:, 2] = 1.0
    return lm


def _picture():
    rng = np.random.default_rng(0)
    img = np.full((540, 640, 3), (60, 70, 80), np.uint8)
    cv2.ellipse(img, (320, 270), (100, 130), 0, 0, 360, (190, 150, 130), -1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def _fresh_parser():
    SB._P[0] = None
    yield
    SB._P[0] = None


def test_default_parser_is_mediapipe_and_the_benchmark_parses_with_it():
    assert isinstance(FP.make_parser(), FP.MPFaceParser)
    maps = SB.parse_maps(_picture(), face68())
    assert maps is not None
    for k in ("eyes", "brows", "lips", "hair", "skin", "neck", "cover"):
        assert k in maps and maps[k].shape == (540, 640)
    assert maps["eyes"].max() > 0.6 and maps["lips"].max() > 0.6
    assert maps["source"]["eyes"] == "app"


def test_feature_parts_are_feature_mask_split():
    lmc = face68(256, 230, 300)
    whole = FP.feature_mask(lmc)
    parts = FP.feature_parts(lmc)
    union = np.maximum.reduce([parts[k] for k in ("eyes", "brows", "lips")])
    assert np.abs((whole > 0.5).astype(int) - (union > 0.5).astype(int)).sum() <= 0.002 * whole.size


def test_feature_mask_unchanged():
    """The rules as they were before the split (copied from 1.22.1)."""
    def old(lmc, size=FP.SIZE):
        m = np.zeros((size, size), np.uint8)
        L = np.asarray(lmc, np.float64)[:, :2]
        if not np.isfinite(L[17:68]).all():
            return m.astype(np.float32)
        wf = float(np.hypot(*(L[16] - L[0]))) if np.isfinite(L[[0, 16]]).all() else size * 0.45
        for a, b in ((36, 42), (42, 48)):
            c = L[a:b].mean(0)
            poly = c + (L[a:b] - c) * 1.35
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        th = max(2, int(round(0.035 * wf)))
        for a, b in ((17, 22), (22, 27)):
            cv2.polylines(m, [np.round(L[a:b]).astype(np.int32)], False, 1, thickness=th)
        cv2.fillPoly(m, [np.round(L[48:60]).astype(np.int32)], 1)
        k = max(1, int(round(0.01 * wf)))
        m = cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.8, 0.006 * wf)), 0, 1)
    for w in (120, 300, 420):
        lmc = face68(256, 240, w)
        assert np.array_equal(FP.feature_mask(lmc), old(lmc))
    bad = face68()
    bad[40] = np.nan
    assert np.array_equal(FP.feature_mask(bad), old(bad))
    assert FP.feature_parts(bad) is None


def _skin(maps):
    """The cheek / forehead area of the drawn face, without the protected features (the parser need not see a
    real face in a drawn picture: only the measuring is tested here)."""
    yy, xx = np.mgrid[0:540, 0:640]
    oval = ((xx - 320) / 100.0) ** 2 + ((yy - 270) / 130.0) ** 2 <= 1.0
    return oval & ~(maps["eyes"] > 0.3) & ~(maps["lips"] > 0.3) & ~(maps["brows"] > 0.3)


def _retouched(src, skin, extra=None, amount=12.0):
    out = src.astype(np.float32)
    out[skin] += amount
    if extra is not None:
        out[extra] += amount
    return np.clip(out, 0, 255).astype(np.uint8)


def test_a_change_inside_a_protected_part_raises_its_leak():
    src = _picture()
    maps = SB.parse_maps(src, face68())
    skin = _skin(maps)
    m = skin.astype(np.float32)
    clean = SB.measure(src, _retouched(src, skin), m, maps)
    eyes = maps["eyes"] > 0.6
    leaky = SB.measure(src, _retouched(src, skin, eyes), m, maps)
    assert clean["leak_eyes"] < 0.1
    assert leaky["leak_eyes"] > 0.8
    assert "eyes" not in leaky["unmeasured"]


def test_a_missing_protection_is_unmeasured_not_zero():
    src = _picture()
    maps = SB.parse_maps(src, face68())
    skin = _skin(maps)
    m = skin.astype(np.float32)
    out = _retouched(src, skin)
    for k in ("eyes", "lips", "prop"):
        maps.pop(k, None)
    r = SB.measure(src, out, m, maps)
    assert "leak_eyes" not in r and "leak_lips" not in r
    assert {"eyes", "lips", "prop"} <= set(r["unmeasured"])
    s = SB.summarize([dict(r, level="std", cut="c", pid=1, t=0)])
    assert s["std"]["all"]["unmeasured"]["eyes"] == 1          # counted as not checked, not as a pass
    r0 = SB.measure(src, out, m, None)
    assert set(r0["unmeasured"]) == set(SB.PARTS) and not any(k.startswith("leak_") for k in r0)


def test_hands_and_manual_marks():
    src = _picture()
    lm = face68()
    occ = [(np.array([[250.0, 330.0], [300.0, 300.0], [340.0, 280.0]]), 14.0)]   # a forearm across the cheek
    mic = np.zeros(src.shape[:2], np.float32)
    mic[330:380, 300:330] = 1.0
    maps = SB.parse_maps(src, lm, occ, manual=dict(prop=mic))
    assert maps["source"]["hand"] == "app" and maps["hand"].max() > 0.5
    assert maps["source"]["prop"] == "manual"
    skin = _skin(maps) & ~(mic > 0.5)
    r = SB.measure(src, _retouched(src, skin, mic > 0.5), skin.astype(np.float32), maps)
    assert r["src_prop"] == "manual" and r["leak_prop"] > 0.5
