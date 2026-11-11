"""Retouch style learned from hand retouched results.

When a plate comes with the retoucher's own result (TEST004.mov + TEST004_AFTER.mov), the pair says exactly
how far this person wants skin to go. After the plate is analysed, a few frames with large, well tracked
faces are retouched with different skin settings and compared with the hand result on the face skin only
(eyes, brows and lips left out). The settings that come closest are kept as one sample per pair in
Contour_Data/learning/retouch_style.json, and every new pair updates the shared preset `학습된_스타일`
(weighted median of all samples, weight = how much closer than doing nothing the fit got).

Measures, per frame and face (luma in 0..1, x1000):
- tone error: RMS of the 3 px blurred luma difference to the hand result (blotches, shading, spots)
- colour error: same for R - Y (redness, colour blotches)
- fine texture: std of the finest detail band relative to the hand result (grain added or smoothed away)
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core import media as M
from ..core.settings import settings

AFTER_RE = re.compile(r"^(?P<base>.+?)[ _\-.]*(after|retouched|수정본|보정본)$", re.I)
VIDEO_EXT = (".mov", ".mp4", ".mxf", ".avi", ".mkv", ".m4v")
PRESET_NAME = "학습된_스타일"
K = np.array([0.2126, 0.7152, 0.0722], np.float32)

# Search space: one slider at a time, in this order, twice for the main ones. Values in slider units.
SEARCH: List[Tuple[str, List[float]]] = [
    ("skin", [0, 15, 30, 45, 60, 80]),
    ("_detail", [15, 25, 45, 65]),
    ("blemish", [0, 20, 40, 70]),
    ("redness", [0, 25, 50]),
    ("dark_circle", [0, 25, 50]),
    ("nasolabial", [0, 25, 50]),
    ("shine", [0, 15, 30, 45, 60, 80]),
    ("mouth_lines", [0, 25, 50]),
    ("neck_lines", [0, 30, 60]),
    ("shine", None),
    ("skin", None),          # refine around the current value
    ("_detail", None),
]
LEARNED_KEYS = ["skin", "blemish", "redness", "dark_circle", "nasolabial", "shine", "mouth_lines", "neck_lines", "grain"]


class StyleError(Exception):
    pass


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-.]+", "", s).lower()


def split_after(name: str) -> Optional[str]:
    """'TEST004_AFTER' -> 'TEST004', else None."""
    m = AFTER_RE.match(Path(name).stem)
    return m.group("base") if m and m.group("base") else None


def find_after(path: str) -> Optional[str]:
    """A hand retouched result sitting next to the plate (same folder, name + _AFTER, any video type)."""
    p = Path(path)
    if p.suffix == "" or not p.parent.exists():
        return None
    want = _norm(p.stem)
    try:
        for q in p.parent.iterdir():
            if q == p or q.suffix.lower() not in VIDEO_EXT:
                continue
            b = split_after(q.name)
            if b is not None and _norm(b) == want:
                return str(q)
    except OSError:
        return None
    return None


def pair_paths(paths: List[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Split dropped files into plates and hand results. Returns (plates, {plate: after}, orphan afters)."""
    afters = {}
    plates = []
    for p in paths:
        b = split_after(Path(p).name)
        if b is not None and Path(p).suffix.lower() in VIDEO_EXT:
            afters[p] = _norm(b)
        else:
            plates.append(p)
    pairs = {}
    used = set()
    for p in plates:
        k = _norm(Path(p).stem)
        hit = next((a for a, b in afters.items() if b == k), None)
        if hit is None:
            hit = find_after(p)
        if hit is not None:
            pairs[p] = hit
            used.add(hit)
    orphans = [a for a in afters if a not in used]
    return plates, pairs, orphans


# ------------------------------------------------------------------ face skin regions
def skin_region(lm: np.ndarray, shape) -> np.ndarray:
    """Face skin from 68 landmarks: forehead, under the eyes, nose, cheeks, upper lip skin and chin.
    Eyes, brows, lips and nostrils are left out (the hand result never changes them)."""
    L = lm[:, :2].astype(np.float64)
    m = np.zeros(shape, np.uint8)

    def poly(pts):
        cv2.fillPoly(m, [np.round(np.asarray(pts)).astype(np.int32)], 1)

    wf = float(np.hypot(*(L[16] - L[0])))
    up = L[27] - L[8]
    up = up / max(1e-6, np.linalg.norm(up))
    brow = L[17:27]
    poly(np.vstack([brow, brow[::-1] + up * 0.3 * wf]))
    for eye in (L[36:42], L[42:48]):
        lo = eye[[0, 5, 4, 3]]
        poly(np.vstack([lo, lo[::-1] - up * 0.13 * wf]))
    poly(np.vstack([L[27:31], L[31:36][::-1]]))
    poly([L[1], L[2], L[3], L[48], L[31], L[39], L[40], L[41], L[36]])
    poly([L[15], L[14], L[13], L[54], L[35], L[42], L[47], L[46], L[45]])
    poly([L[48], L[31], L[33], L[35], L[54], L[53], L[52], L[51], L[50], L[49]])
    poly([L[5], L[6], L[7], L[8], L[9], L[10], L[11], L[54], L[55], L[56], L[57], L[58], L[59], L[48]])
    # the throat under the jaw (neck lines, chin shading): a band as deep as 0.45 face widths
    down = -up
    jaw = L[3:14]
    poly(np.vstack([jaw, (jaw + down * 0.45 * wf)[::-1]]))
    cut = np.zeros(shape, np.uint8)
    for grp in (L[36:42], L[42:48], L[48:60]):
        cv2.fillPoly(cut, [np.round(grp).astype(np.int32)], 1)
    for grp in (L[17:22], L[22:27]):
        cv2.polylines(cut, [np.round(grp).astype(np.int32)], False, 1, max(2, int(0.03 * wf)))
    cut = cv2.dilate(cut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.02 * wf) + 1,) * 2))
    return (m > 0) & (cut == 0)


# ------------------------------------------------------------------ results from other programs
def _gfeats(X: np.ndarray) -> np.ndarray:
    r, g, b = X[:, 0], X[:, 1], X[:, 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b, r ** 3, g ** 3, b ** 3], 1)


def fit_grade(o: np.ndarray, a: np.ndarray, box=None) -> Optional[np.ndarray]:
    """Colour change the other program made over the whole picture (a grade, curves, a LUT on export).

    A cubic RGB polynomial from original to hand result, fitted on blurred pictures (a skin retouch is
    small and local, a grade is everywhere and smooth), with every brightness band weighted alike so a dark
    background does not decide the skin tones. Includes the face: on a graded copy of a hand result this
    left almost exactly the hand retouch (residual 2.86 / 3.36 against 2.82 / 3.24 without any grade).
    None when there is no grade."""
    H, W = o.shape[:2]
    st = max(1, int(np.sqrt(H * W / 80000)))
    ob = cv2.GaussianBlur(o.astype(np.float32), (0, 0), 3)[::st, ::st].reshape(-1, 3).astype(np.float64)
    ab = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3)[::st, ::st].reshape(-1, 3).astype(np.float64)
    ok = (ob.max(1) < 0.98) & (ab.max(1) < 0.98) & (ob.min(1) > 0.003)
    X, Y = ob[ok], ab[ok]
    if len(X) < 1000 or float(np.abs(X - Y).mean()) < 0.003:
        return None
    lum = X @ K
    bins = np.clip((lum * 16).astype(int), 0, 15)
    cnt = np.bincount(bins, minlength=16).astype(np.float64)
    w = np.sqrt(1.0 / np.maximum(cnt[bins], 1))[:, None]
    F = _gfeats(X)
    C, *_ = np.linalg.lstsq(F * w, Y * w, rcond=None)
    e_id = float(np.abs(Y - X).mean())
    e_g = float(np.abs(Y - F @ C).mean())
    return C if e_g < 0.8 * e_id else None


def apply_grade(G: Optional[np.ndarray], rgb: np.ndarray) -> np.ndarray:
    if G is None:
        return rgb
    out = _gfeats(rgb.reshape(-1, 3).astype(np.float64)) @ G
    return out.reshape(rgb.shape).astype(rgb.dtype)


def align_crop(o: np.ndarray, a: np.ndarray, m: np.ndarray):
    """If the other program also reshaped the face (liquify, a face warp), its pixels moved: warp the hand
    result back onto the original's geometry before comparing skin, and leave out what moved a lot.
    Returns (aligned hand crop, mask, median shift px)."""
    g0 = np.clip(o @ K * 255, 0, 255).astype(np.uint8)
    g1 = np.clip(a @ K * 255, 0, 255).astype(np.uint8)
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        F = dis.calc(g0, g1, None)
    except Exception:
        return a, m, 0.0
    mag = np.hypot(F[..., 0], F[..., 1])
    med = float(np.median(mag[m])) if m.any() else 0.0
    if float(np.quantile(mag[m], 0.9)) < 0.6 if m.any() else True:
        return a, m, med
    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    al = cv2.remap(a.astype(np.float32), gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)
    return al.astype(a.dtype), m & (mag < 3.0), med


# ------------------------------------------------------------------ fitting
def _pick(st, n_frames: int = 4):
    """[(person index, frame)] with the largest, best tracked faces, spread over the clip."""
    if not getattr(st, "has_face", False):
        raise StyleError("얼굴이 추적되지 않은 컷입니다")
    P, T = st.present.shape
    cand = []
    for p in range(P):
        for t in range(T):
            if not st.present[p, t]:
                continue
            f = st.face[p, t, :, :2]
            if not np.isfinite(f).all() or float(st.face_vis[p, t]) < 0.5:
                continue
            w = float(np.hypot(*(f[16] - f[0])))
            if w < 120:
                continue
            cand.append((w * float(st.face_vis[p, t]), p, t))
    if not cand:
        raise StyleError("수작업과 비교할 만큼 크게 잡힌 얼굴이 없습니다 (얼굴 폭 120px 이상)")
    cand.sort(reverse=True)
    best_p = {}
    for s, p, t in cand:
        best_p[p] = max(best_p.get(p, 0.0), s)
    persons = sorted(best_p, key=lambda p: -best_p[p])[:2]
    picks = []
    for p in persons:
        ts = sorted(t for s, q, t in cand if q == p)
        k = max(1, n_frames // len(persons))
        idx = np.linspace(0, len(ts) - 1, min(k, len(ts))).round().astype(int)
        picks += [(p, ts[i]) for i in sorted(set(idx.tolist()))]
    return picks[:n_frames]


def _pick_at(st, t: int, n: int = 3):
    """The people with a usable face at frame t, largest first."""
    out = []
    for p in range(st.present.shape[0]):
        if not st.present[p, t]:
            continue
        f = st.face[p, t, :, :2]
        if not np.isfinite(f).all() or float(st.face_vis[p, t]) < 0.4:
            continue
        w = float(np.hypot(*(f[16] - f[0])))
        if w >= 60:
            out.append((w, p))
    out.sort(reverse=True)
    if not out:
        raise StyleError("레퍼런스 프레임에서 얼굴을 찾지 못했습니다")
    return [(p, t) for _, p in out[:n]]


def _read(info, t: int, pf: str):
    rd = M.FrameReader(info, t, 1, pf, prefetch=1)
    try:
        pl = rd.read()
        return None if pl is None else [np.array(x, copy=True) for x in pl]
    finally:
        rd.close()


def _fine(Y: np.ndarray, m: np.ndarray) -> float:
    return float((Y - cv2.GaussianBlur(Y, (0, 0), 0.8))[m].std())


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _still(path: str, W: int, H: int) -> np.ndarray:
    """A reference still (an exported frame of the retouched picture) as RGB 0..1."""
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise StyleError("레퍼런스 이미지를 읽지 못했습니다")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = im[..., :3]
    mx = 65535.0 if im.dtype == np.uint16 else 255.0
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / mx
    if rgb.shape[:2] != (H, W):
        raise StyleError(f"레퍼런스 이미지 크기가 다릅니다 ({rgb.shape[1]}x{rgb.shape[0]} / {W}x{H})")
    return rgb


def find_still_frame(cut, path: str, progress: Callable = None) -> Tuple[int, float]:
    """Which frame of the cut a reference still was taken from: (frame, mean difference)."""
    info = cut.info
    ref = cv2.resize(_still(path, info.width, info.height), (256, 144), interpolation=cv2.INTER_AREA)
    best = (0, 9.0)
    W = 256
    H = 144
    with M.FrameReader(info, 0, info.frames, "rgb24", size=(W, H), rgb_convert=True, prefetch=4) as rd:
        for t in range(info.frames):
            pl = rd.read()
            if pl is None:
                break
            d = float(np.abs(pl[0].astype(np.float32) / 255 - ref).mean())
            if d < best[1]:
                best = (t, d)
    return best


def fit_pair(cut, tracker, after_path: str, progress: Callable = None, cancelled: Callable = None,
             at_frame: Optional[int] = None) -> dict:
    """after_path: the retoucher's result of this cut as a video, or one retouched still of it (then
    at_frame says which frame; found automatically when left out)."""
    from ..retouch import engine as RE
    from ..warp import render as R
    from ..track.analyze import Cancelled

    progress = progress or (lambda f, m="": None)
    cancelled = cancelled or (lambda: False)
    st = tracker.store
    if st is None:
        raise StyleError("먼저 분석을 실행하세요")
    info = cut.info
    still = None
    if Path(after_path).suffix.lower() in IMAGE_EXT:
        still = _still(after_path, info.width, info.height)
        if at_frame is None:
            at_frame, dd = find_still_frame(cut, after_path)
            if dd > 0.05:
                raise StyleError("레퍼런스 이미지와 같은 장면을 이 컷에서 찾지 못했습니다")
        ainfo = info
    else:
        ainfo = M.probe(after_path)
    if (ainfo.width, ainfo.height) != (info.width, info.height):
        raise StyleError(f"해상도가 다릅니다 ({info.width}x{info.height} / {ainfo.width}x{ainfo.height})")
    # a hand result of the whole edited timeline: this cut is the same stretch of it
    if getattr(info, "offset", 0) and ainfo.frames >= info.offset + info.frames and ainfo.frames > info.frames + 2:
        ainfo.offset = int(info.offset)
        ainfo.frames = int(info.frames)
    pf = R._proc_format(cut, cut.output)
    layout = M.plane_layout(pf, info.width, info.height)
    mtx = R._matrix_name(info)
    rgb = lambda pl: M.yuv_to_rgb(pl, layout, mtx, info.color_range)

    if still is not None:
        picks = _pick_at(st, int(at_frame))
    else:
        picks = _pick(st)
    progress(0.02, "프레임 읽는 중")
    frames = {}
    for i, (p, t) in enumerate(picks):
        if cancelled():
            raise Cancelled()
        pl = _read(info, t, pf)
        if pl is None:
            continue
        o = rgb(pl)
        lm = st.face[p, t]
        f = lm[:, :2]
        c = f.mean(0)
        r = 1.1 * float(np.ptp(f[:, 0]))
        x0, y0 = max(0, int(c[0] - r)), max(0, int(c[1] - 1.2 * r))
        x1, y1 = min(info.width, int(c[0] + r)), min(info.height, int(c[1] + 1.3 * r))
        loc = lm.copy()
        loc[:, 0] -= x0
        loc[:, 1] -= y0
        m = skin_region(loc, (y1 - y0, x1 - x0))
        if m.sum() < 500:
            continue
        # the hand result may be off by a frame or two (handles, a re-export): take the closest of t-2..t+2
        best = None
        for d in ((0,) if still is not None else (0, -1, 1, -2, 2)):
            if not 0 <= t + d < max(1, ainfo.frames or 10 ** 9):
                continue
            if still is not None:
                a = still
            else:
                ap = _read(ainfo, t + d, pf)
                if ap is None:
                    continue
                a = rgb(ap)
            G = fit_grade(o, a, (x0, y0, x1, y1))
            e = float(np.abs(((apply_grade(G, o[::4, ::4]) - a[::4, ::4]) @ K)).mean())
            if best is None or e < best[0]:
                best = (e, a, G)
            if e < 0.004:
                break
        if best is None:
            continue
        if best[0] > 0.06:
            raise StyleError("수작업 결과가 원본과 다른 장면입니다 (프레임 번호 또는 파일 확인)")
        G = best[2]
        oc = o[y0:y1, x0:x1]
        if still is not None:
            # a reference still: only the faces the retoucher worked on teach (the others were left alone,
            # usually on purpose: out of focus, at the back)
            dd = cv2.GaussianBlur(((best[1][y0:y1, x0:x1] - apply_grade(G, oc)) @ K).astype(np.float32), (0, 0), 3)
            if float((np.abs(dd[m]) > 0.004).mean()) < 0.05:
                continue
        ac, m, wpx = align_crop(apply_grade(G, oc), best[1][y0:y1, x0:x1], m)
        frames[(p, t)] = dict(planes=pl, o=oc, a=ac, box=(x0, y0, x1, y1), m=m, cache={}, G=G, warp=wpx)
        progress(0.02 + 0.1 * (i + 1) / len(picks), "프레임 읽는 중")
    if not frames:
        raise StyleError("비교할 프레임을 찾지 못했습니다")

    # work on a copy of the roles and sliders; the cut is put back exactly as it was
    roles = {q.id: q.role for q in cut.persons}
    params = copy.deepcopy(cut.params)
    chroma0 = RE.SKIN_CHROMA
    by_p = {}
    for (p, t) in frames:
        by_p.setdefault(p, []).append(t)

    def evaluate(vals: dict) -> Tuple[float, float, float]:
        E, C, F = [], [], []
        for (p, t), fr in frames.items():
            pid = st.pids[p]
            for q in cut.persons:
                q.role = "target" if q.id == pid else "protect"
            pp = cut.params_for(pid)
            for k in pp.amounts:
                pp.amounts[k].base = 0.0
                pp.amounts[k].keys = {}
            pp.skin_detail = float(vals.get("_detail", 45.0))
            for k, v in vals.items():
                if not k.startswith("_") and k in pp.amounts:
                    pp.amounts[k].base = float(v)
            rf = RE.prepare(cut, tracker, t, cache=fr["cache"])
            if rf is None:
                out = fr["planes"]
            else:
                out = RE.apply_planes([x.copy() for x in fr["planes"]], layout.subsample, layout.max_value, pf, rf, 1.0)
            x0, y0, x1, y1 = fr["box"]
            b = apply_grade(fr["G"], rgb(out)[y0:y1, x0:x1])
            a, m = fr["a"], fr["m"]
            dl = cv2.GaussianBlur((b - a) @ K, (0, 0), 3)
            E.append(float(np.sqrt((dl[m] ** 2).mean())) * 1000)
            cr = lambda Z: cv2.GaussianBlur(Z[..., 0] - Z @ K, (0, 0), 3)
            C.append(float(np.sqrt(((cr(b) - cr(a))[m] ** 2).mean())) * 1000)
            F.append(_fine((b @ K).astype(np.float32), m) / max(1e-6, _fine((a @ K).astype(np.float32), m)))
        return float(np.mean(E)), float(np.mean(C)), float(np.mean(F))

    def score(e, c):
        return e + 0.5 * c

    try:
        RE.SKIN_CHROMA = chroma0
        e0, c0, f0 = evaluate({})
        cur = {"_detail": 45.0}
        ce, cc, cf = e0, c0, f0
        n = sum(len(v) if v else 3 for _, v in SEARCH) + 4
        done = 0
        for key, vals in SEARCH:
            if vals is None:
                v0 = cur.get(key, 0.0)
                step = 10.0 if key == "skin" else 8.0
                vals = [v0 - step, v0 + step]
            for v in vals:
                if cancelled():
                    raise Cancelled()
                done += 1
                progress(0.12 + 0.8 * done / n, f"수작업과 비교 중 · {key.lstrip('_')} {v:.0f}")
                lo, hi = (10.0, 90.0) if key == "_detail" else (0.0, 100.0)
                v = float(np.clip(v, lo, hi))
                if abs(cur.get(key, 45.0 if key == "_detail" else 0.0) - v) < 1e-6:
                    continue
                trial = dict(cur)
                trial[key] = v
                e, c, f = evaluate(trial)
                if score(e, c) < score(ce, cc) - 0.005:
                    cur, ce, cc, cf = trial, e, c, f
        # grain: the hand result often puts fine grain back on the smoothed skin; match the finest band
        if cf < 0.97:
            for g in (10, 20, 35, 50):
                if cancelled():
                    raise Cancelled()
                trial = dict(cur)
                trial["grain"] = float(g)
                e, c, f = evaluate(trial)
                if abs(f - 1) < abs(cf - 1) and score(e, c) < score(ce, cc) + 0.05:
                    cur, ce, cc, cf = trial, e, c, f
        progress(0.95, "저장 중")
    finally:
        for q in cut.persons:
            if q.id in roles:
                q.role = roles[q.id]
        cut.params = params
        RE.SKIN_CHROMA = chroma0

    # what the hand did, measured directly (for the summary)
    tex = []
    for fr in frames.values():
        o, a, m = apply_grade(fr["G"], fr["o"]) @ K, fr["a"] @ K, fr["m"]
        band = lambda Y: (Y - cv2.GaussianBlur(Y, (0, 0), 3))[m].std()
        tex.append(float(band(a.astype(np.float32)) / max(1e-6, band(o.astype(np.float32)))))
    sample = dict(
        clip=cut.name, source=str(getattr(cut, "source", "")), after=str(after_path),
        date=time.strftime("%Y-%m-%d %H:%M"), frames=[[int(st.pids[p]), int(t)] for (p, t) in frames],
        values={k: round(float(v), 1) for k, v in cur.items() if not k.startswith("_")},
        detail=round(float(cur.get("_detail", 45.0)), 1),
        err_before=round(e0, 3), err_after=round(ce, 3), colour_before=round(c0, 3), colour_after=round(cc, 3),
        hand_texture_kept=round(float(np.mean(tex)), 3), fine_ratio=round(cf, 3),
        grade=sum(1 for fr in frames.values() if fr["G"] is not None),
        warp_px=round(float(np.mean([fr["warp"] for fr in frames.values()])), 2),
    )
    progress(1.0, "")
    return sample


# ------------------------------------------------------------------ storage and the shared preset
def _store_path() -> Path:
    p = settings().data_path / "learning"
    p.mkdir(parents=True, exist_ok=True)
    return p / "retouch_style.json"


def load_samples() -> List[dict]:
    f = _store_path()
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return list(d.get("samples", []))
    except Exception:
        return []


def _wmedian(x: List[float], w: List[float]) -> float:
    o = np.argsort(x)
    x = np.asarray(x, float)[o]
    w = np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def aggregate(samples: List[dict]) -> Optional[dict]:
    ok = [s for s in samples if s.get("err_after", 1e9) < s.get("err_before", 0) - 1e-3]
    if not ok:
        return None
    w = [max(0.05, (s["err_before"] - s["err_after"]) / max(1e-6, s["err_before"])) for s in ok]
    vals = {k: round(_wmedian([float(s["values"].get(k, 0.0)) for s in ok], w)) for k in LEARNED_KEYS}
    detail = round(_wmedian([float(s.get("detail", 45.0)) for s in ok], w))
    return dict(values=vals, detail=detail, n=len(ok))


def add_sample(sample: dict) -> Optional[dict]:
    """Stores the sample (one per plate: a re-learn of the same plate replaces it) and rewrites the preset."""
    from ..core import presets as PR
    samples = [s for s in load_samples() if _norm(Path(s.get("source") or s.get("clip", "")).stem)
               != _norm(Path(sample.get("source") or sample.get("clip", "")).stem)]
    samples.append(sample)
    f = _store_path()
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(version=1, samples=samples), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, f)
    agg = aggregate(samples)
    if agg is None:
        return None
    PR.save(dict(version=PR.VERSION, name=PRESET_NAME, created=time.strftime("%Y-%m-%d %H:%M"), groups=["skin"],
                 values={k: float(v) for k, v in agg["values"].items()},
                 skin=dict(detail=float(agg["detail"]), scope="face"),
                 note=f"수작업 리터칭 {agg['n']}개 영상에서 학습"))
    return agg


def forget(clip_key: str) -> None:
    samples = [s for s in load_samples() if _norm(Path(s.get("source") or s.get("clip", "")).stem) != _norm(clip_key)]
    _store_path().write_text(json.dumps(dict(version=1, samples=samples), ensure_ascii=False, indent=1), encoding="utf-8")
