"""Skin retouch benchmark: the basic adjustment (표준, 강하게) scored on analysed cuts of any kind of footage.

Each sample is one frame of one cut where a beauty target's face is well seen. It is tagged with what makes footage
hard (log, dark scene, coloured light, motion blur, small or large face, skin tone) and measured:

- colour:   mean change of the skin's colour (CIELAB dE, dL, da, db) inside the skin the retouch touched; dL < 0 is
            skin made darker (never wanted), da / db a colour cast
- strength: mean |change of luma| on that skin (0..255)
- grain:    fine detail after / before in three bands (finest = grain, middle = pores, coarse = blotches); the finest
            band must stay near 1
- leaks:    change inside the eyes, brows, lips (face parser) and in hair over the face, relative to the change on
            the skin: should be near 0
- flicker:  how much the correction itself changes from frame to frame where the face does not (optical flow on
            the source), relative to its size
- render:   the same frame retouched as the render does it (full resolution, native planes, log carried back),
            shrunk to the preview size and compared with the preview's correction (strength ratio, difference)

compare_hand(): DIAP's levels against a retoucher's own result of the same frames (tone / colour error after a
global grade fit, fine detail against the hand result's), with doing nothing as the yardstick.
Used by the PC test jobs; nothing here changes a project (the cut's sliders are put back as they were)."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

LEVELS = ("std", "strong")
BANDS = ((0.6, 1.2), (1.2, 2.4), (2.4, 4.8))          # sigma px of a 1080-line picture
EXTRA = ("mouth_lines", "nasolabial", "blemish")
_P = [None]


# ------------------------------------------------------------------ sliders
def set_level(cut, level: Optional[str]) -> Callable[[], None]:
    """The beauty targets' basic adjustment at `level` (None: nothing); returns the function that puts back what was."""
    from ..core.params import default as pdef, skin_basic_values
    from ..core.project import Curve
    saved = copy.deepcopy(cut.params)
    vals = skin_basic_values(level, []) if level else {}
    keys = set(skin_basic_values("strong", []).keys()) | set(EXTRA)
    for q in cut.persons:
        if q.role != "target":
            continue
        pp = copy.deepcopy(cut.params_for(q.id))
        for k in keys:
            if k in pp.amounts:
                pp.amounts[k] = Curve(base=float(pdef(k)))
        for k, v in vals.items():
            pp.amounts[k] = Curve(base=float(v))
        cut.params[q.id] = pp

    def restore():
        cut.params = saved
    return restore


# ------------------------------------------------------------------ pictures
def luma(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def lab(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) * np.array([100 / 255, 1, 1], np.float32) \
        - np.array([0, 128, 128], np.float32)


def band(Y: np.ndarray, s0: float, s1: float) -> np.ndarray:
    k = Y.shape[0] / 1080.0
    return cv2.GaussianBlur(Y, (0, 0), max(0.3, s0 * k)) - cv2.GaussianBlur(Y, (0, 0), max(0.5, s1 * k))


def _rms(x: np.ndarray, m: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.any() else float("nan")


def parse_maps(rgb_u8: np.ndarray, lm: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Per class probability maps of the face parser on the image (eyes, brows, lips with mouth, hair, skin)."""
    from ..track import faceparse as FP
    if _P[0] is None:
        _P[0] = FP.FaceParser() if FP.ready() else False
    P = _P[0]
    if not P:
        return None
    M = P.crop_affine(np.asarray(lm, np.float64))
    if M is None:
        return None
    crop = cv2.warpAffine(rgb_u8, M, (FP.SIZE, FP.SIZE), flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    x = (crop.astype(np.float32) / 255.0 - FP.MEAN) / FP.STD
    lo = P.s.run(None, {P.inp: np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0][0]
    lo = lo - lo.max(0, keepdims=True)
    e = np.exp(lo)
    pr = e / e.sum(0, keepdims=True)
    shape = rgb_u8.shape[:2]

    def cls(ids):
        m = cv2.resize(pr[list(ids)].sum(0), (FP.KEEP, FP.KEEP), interpolation=cv2.INTER_LINEAR)
        return FP.to_image(m, M, shape)
    return dict(eyes=cls((4, 5)), brows=cls((6, 7)), lips=cls((10, 11, 12)), hair=cls(FP.HAIR), skin=cls(FP.SKIN),
                cover=FP.to_image(np.ones((FP.KEEP, FP.KEEP), np.float32), M, shape))


def measure(src: np.ndarray, out: np.ndarray, m: np.ndarray, maps: Optional[dict]) -> dict:
    sk = m > 0.5
    if sk.sum() < 200:
        return dict(skip="small mask")
    Y0, Y1 = luma(src), luma(out)
    dY = Y1 - Y0
    L0, L1 = lab(src), lab(out)
    d = L1[sk].mean(0) - L0[sk].mean(0)
    dE = np.linalg.norm(L1 - L0, axis=2)
    strength = float(np.abs(dY[sk]).mean())
    r = dict(dE=round(float(np.linalg.norm(d)), 3), dL=round(float(d[0]), 3), da=round(float(d[1]), 3),
             db=round(float(d[2]), 3), dE_p95=round(float(np.percentile(dE[sk], 95)), 2),
             strength=round(strength, 3), skin_px=int(sk.sum()))
    for i, (s0, s1) in enumerate(BANDS):
        r[f"grain{i}"] = round(_rms(band(Y1, s0, s1), sk) / max(1e-6, _rms(band(Y0, s0, s1), sk)), 3)
    if maps is not None:
        er = np.ones((3, 3), np.uint8)
        for k in ("eyes", "brows", "lips"):
            fm = cv2.erode((maps[k] > 0.6).astype(np.uint8), er).astype(bool)
            if fm.sum() >= 20:
                v = float(np.abs(dY[fm]).mean())
                r[f"leak_{k}"] = round(v / max(strength, 0.05), 3)
                r[f"leak_{k}_abs"] = round(v, 3)
        face = cv2.dilate((maps["skin"] > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        hm = (maps["hair"] > 0.7) & face & (maps["cover"] > 0.5)
        hm = cv2.erode(hm.astype(np.uint8), er).astype(bool)
        if hm.sum() >= 50:
            v = float(np.abs(dY[hm]).mean())
            r["leak_hair"] = round(v / max(strength, 0.05), 3)
            r["leak_hair_abs"] = round(v, 3)
    return r


def flicker(frames: List[tuple]) -> Optional[dict]:
    """frames: (src, out, mask) of consecutive frames. The correction (out - src) carried along the source's own motion
    to the next frame against the correction there, relative to its size; and the frame to frame spread of its mean."""
    if len(frames) < 3:
        return None
    rel, means = [], []
    for (s0, o0, m0), (s1, o1, m1) in zip(frames, frames[1:]):
        sk = (m0 > 0.5) & (m1 > 0.5)
        if sk.sum() < 200:
            continue
        ys, xs = np.nonzero(sk)
        pad = 12
        y0, y1 = max(0, ys.min() - pad), min(sk.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(sk.shape[1], xs.max() + pad)
        g0 = cv2.cvtColor(s0[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(s1[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        gx, gy = np.meshgrid(np.arange(x1 - x0, dtype=np.float32), np.arange(y1 - y0, dtype=np.float32))
        D0 = (luma(o0) - luma(s0))[y0:y1, x0:x1]
        D1 = (luma(o1) - luma(s1))[y0:y1, x0:x1]
        D0w = cv2.remap(D0, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        k = sk[y0:y1, x0:x1]
        size = float(np.abs(D1[k]).mean())
        if size < 0.05:
            continue
        rel.append(float(np.abs(D1[k] - D0w[k]).mean()) / size)
        means.append(size)
    if not rel:
        return None
    return dict(flicker_px=round(float(np.median(rel)), 3),
                flicker_mean=round(float(np.std(means) / max(1e-6, np.mean(means))), 3))


def render_frame(cut, t: int, rf, proxy_wh) -> Optional[dict]:
    """The frame as the render makes it: native planes at full size, retouched like render_cut does, then both
    pictures in the display view, at full size and shrunk to the preview size."""
    from ..core import media as MD
    from ..core.colour import PASSTHROUGH, manager
    from ..retouch import engine as RE
    from ..warp import render as R
    info = cut.info
    pf = R._proc_format(cut, cut.output)
    layout = MD.plane_layout(pf, info.width, info.height)
    cm = manager()
    vs = R._vs_for(cut)
    disp_ = None
    if (cut.input_cs or PASSTHROUGH) != PASSTHROUGH:
        v_ = R._vs_for(cut)
        v_.exposure = 0.0
        disp_ = (cm, v_, info.color_range)
    with MD.FrameReader(info, int(t), 1, pf) as rd:
        planes = next(iter(rd), None)
    if planes is None:
        return None
    out = RE.apply_planes([p.copy() for p in planes], layout.subsample, layout.max_value, pf, rf, 1.0, disp_,
                           info.color_range)

    def disp(pl):
        if pf.startswith("yuv"):
            rgb = MD.yuv_to_rgb(pl, layout, R._matrix_name(info), info.color_range)
        else:
            rgb = MD.planes_to_rgb_float(pl, layout)
        return cm.to_display_u8(rgb, vs)
    s_full, o_full = disp(planes), disp(out)
    w, h = proxy_wh
    small = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    return dict(src=s_full, out=o_full, src_small=small(s_full), out_small=small(o_full))


def render_check(rend: dict, src_p: np.ndarray, out_p: np.ndarray, m: np.ndarray) -> dict:
    sk = m > 0.5
    dP = luma(out_p) - luma(src_p)
    dR = luma(rend["out_small"]) - luma(rend["src_small"])
    a = float(np.abs(dP[sk]).mean())
    r = dict(strength_ratio=round(float(np.abs(dR[sk]).mean()) / max(a, 0.05), 3),
             diff=round(float(np.abs(dR[sk] - dP[sk]).mean()) / max(a, 0.05), 3))
    # grain at full resolution (the finest band of a 4K picture is finer than any of the preview's)
    H, W = rend["src"].shape[:2]
    mk = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    Y0, Y1 = luma(rend["src"]), luma(rend["out"])
    for i, (s0, s1) in enumerate(BANDS):
        r[f"full_grain{i}"] = round(_rms(band(Y1, s0, s1), mk) / max(1e-6, _rms(band(Y0, s0, s1), mk)), 3)
    L0, L1 = lab(rend["src"]), lab(rend["out"])
    d = L1[mk].mean(0) - L0[mk].mean(0)
    r.update(full_dL=round(float(d[0]), 3), full_dE=round(float(np.linalg.norm(d)), 3))
    return r


# ------------------------------------------------------------------ what kind of footage
def tags(cut, st, p: int, t: int, src: np.ndarray, m: np.ndarray, px) -> dict:
    Y = luma(src)
    sk = m > 0.5
    L = lab(src)
    skinL = float(L[sk][:, 0].mean()) if sk.any() else float("nan")
    # light colour: chroma of the brightest quarter of the picture (a coloured stage light tints everything lit)
    lit = Y >= np.percentile(Y, 75)
    ab = L[lit][:, 1:].mean(0)
    F = st.face[p, t, :, :2] if np.isfinite(st.face[p, t, :, :2]).all() else None
    fw = float(np.hypot(*(F[16] - F[0]))) if F is not None else float("nan")
    cs = (cut.input_cs or "").lower()
    return dict(log="log" in cs, scene_y=round(float(Y.mean()), 1), light_chroma=round(float(np.hypot(*ab)), 1),
                skin_L=round(skinL, 1), face_w_src=round(fw, 1), face_w_1080=round(fw * 1080.0 / max(1, cut.info.height), 1),
                blur=round(float(st.blur[p, t]), 2), res=f"{cut.info.width}x{cut.info.height}")


def pick_frames(st, cut, n: int = 2, run: int = 4) -> List[tuple]:
    """(p, t0) for targets: frames whose face is well seen, spread over the cut, each with `run` frames after it
    (for the flicker)."""
    a, b = cut.work_range
    out = []
    targets = [q.id for q in cut.persons if q.role == "target"]
    for pid in targets:
        if pid not in st.pids:
            continue
        p = st.pids.index(pid)
        ok = st.present[p] & (st.face_vis[p] >= 0.7) & np.isfinite(st.face[p, :, :, :2]).all((1, 2))
        good = [t for t in range(a, b - run) if ok[t:t + run + 1].all()]
        if not good:
            continue
        for t in np.linspace(0, len(good) - 1, min(n, len(good))).astype(int):
            out.append((p, good[t]))
    return out


# ------------------------------------------------------------------ one cut
def bench_cut(cut, tr, px, levels=LEVELS, n: int = 2, run: int = 4, render: bool = True) -> List[dict]:
    from ..retouch import engine as RE
    st = tr.store
    rows = []
    if st is None or not getattr(st, "has_face", False):
        return rows
    picks = pick_frames(st, cut, n, run)
    for (p, t0) in picks:
        pid = int(st.pids[p])
        base = dict(cut=cut.name, pid=pid, t=int(t0))
        for level in levels:
            restore = set_level(cut, level)
            try:
                seq = []
                rf0 = None
                for t in range(t0, t0 + run + 1):
                    src = np.asarray(px.frame(t))
                    rf = RE.prepare(cut, tr, t, {})
                    if rf is None or rf.empty():
                        seq.append((src, src.copy(), np.zeros(src.shape[:2], np.float32)))
                        continue
                    out = RE.apply_rgb(src, rf, px.sx)
                    m = rf.union()
                    m = cv2.resize(m.astype(np.float32), (src.shape[1], src.shape[0])) if m.shape != src.shape[:2] else m
                    seq.append((src, out, m))
                    if t == t0:
                        rf0 = rf
                src, out, m = seq[0]
                if rf0 is None:
                    rows.append(dict(base, level=level, skip="no retouch"))
                    continue
                lm = st.face[p, t0].copy()
                lm[:, 0] *= px.sx
                lm[:, 1] *= px.sy
                maps = parse_maps(src, lm)
                r = dict(base, level=level, **measure(src, out, m, maps))
                r.update(tags(cut, st, p, t0, src, m, px))
                fl = flicker(seq)
                if fl:
                    r.update(fl)
                if render and level == levels[-1]:
                    try:
                        rend = render_frame(cut, t0, rf0, (px.w, px.h))
                        if rend is not None:
                            r.update(render_check(rend, src, out, m))
                    except Exception as e:
                        r["render_error"] = str(e)[:200]
                r["_tile"] = tile(src, out, m, f"{cut.name[:22]} t{t0} {level}")
                rows.append(r)
            finally:
                restore()
    return rows


# ------------------------------------------------------------------ summary
CHECKS = (
    ("darker skin (dL < -0.5)", lambda r: r.get("dL", 0) < -0.5),
    ("colour cast (|da| or |db| > 1)", lambda r: abs(r.get("da", 0)) > 1.0 or abs(r.get("db", 0)) > 1.0),
    ("grain lost (finest band < 0.85)", lambda r: r.get("grain0", 1) < 0.85),
    ("eye/brow/lip leak (> 0.25)", lambda r: max(r.get("leak_eyes", 0), r.get("leak_brows", 0), r.get("leak_lips", 0)) > 0.25),
    ("hair leak (> 0.35)", lambda r: r.get("leak_hair", 0) > 0.35),
    ("flicker (> 0.6)", lambda r: r.get("flicker_px", 0) > 0.6),
    ("render differs from preview (strength off by > 25 %)", lambda r: "strength_ratio" in r and abs(r["strength_ratio"] - 1) > 0.25),
    ("4K grain lost (finest band < 0.85)", lambda r: r.get("full_grain0", 1) < 0.85),
)
GROUPS = (
    ("log", lambda r: r.get("log")), ("not log", lambda r: not r.get("log")),
    ("dark scene (Y < 60)", lambda r: r.get("scene_y", 128) < 60),
    ("coloured light (chroma > 18)", lambda r: r.get("light_chroma", 0) > 18),
    ("blur", lambda r: r.get("blur", 1) < 0.6),
    ("small face (< 90 px @1080)", lambda r: r.get("face_w_1080", 999) < 90),
    ("large face (> 250 px @1080)", lambda r: r.get("face_w_1080", 0) > 250),
    ("darker skin (L < 55)", lambda r: r.get("skin_L", 100) < 55),
    ("all", lambda r: True),
)


def summarize(rows: List[dict]) -> dict:
    ok = [r for r in rows if "skip" not in r and "dE" in r]
    out = {}
    for lv in sorted({r["level"] for r in ok}):
        R = [r for r in ok if r["level"] == lv]
        g = {}
        for gname, gf in GROUPS:
            S = [r for r in R if gf(r)]
            if not S:
                continue
            row = dict(n=len(S))
            for k in ("dE", "dL", "strength", "grain0", "grain1", "grain2", "leak_eyes", "leak_brows", "leak_lips",
                      "leak_hair", "flicker_px", "strength_ratio", "diff", "full_grain0"):
                v = [r[k] for r in S if k in r and np.isfinite(r[k])]
                if v:
                    row[k] = round(float(np.median(v)), 3)
            row["fails"] = {cn: sum(1 for r in S if cf(r)) for cn, cf in CHECKS if any(cf(r) for r in S)}
            g[gname] = row
        out[lv] = g
    return out


def tile(src: np.ndarray, out: np.ndarray, m: np.ndarray, label: str = "") -> Optional[np.ndarray]:
    """Before | after | change x12 around the retouched skin, 240 px each (BGR, for a sheet)."""
    ys, xs = np.nonzero(m > 0.5)
    if not len(ys):
        return None
    pad = int(0.25 * max(np.ptp(ys), np.ptp(xs)))
    y0, y1 = max(0, ys.min() - pad), min(src.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(src.shape[1], xs.max() + pad)
    cs = lambda x: cv2.resize(x[y0:y1, x0:x1], (240, 240), interpolation=cv2.INTER_AREA)
    d = np.clip(np.abs(luma(out) - luma(src)) * 12, 0, 255).astype(np.uint8)
    dcol = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)[..., ::-1]
    row = cv2.cvtColor(np.hstack([cs(src), cs(out), cs(dcol)]), cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(row, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return row


def sheet(rows: List[dict], path: str, n: int = 12, key: Callable = None) -> None:
    """The tiles of the worst samples (by key), one under the other, with the measures written on them."""
    R = [r for r in rows if r.get("_tile") is not None]
    if key is not None:
        R = sorted(R, key=key, reverse=True)
    tiles = []
    for r in R[:n]:
        t_ = r["_tile"].copy()
        txt = f"dL{r.get('dL')} g{r.get('grain0')} fl{r.get('flicker_px')} eye{r.get('leak_eyes')} hair{r.get('leak_hair')}"
        cv2.putText(t_, txt, (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(t_)
    if tiles:
        cv2.imwrite(path, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


def strip(rows: List[dict]) -> List[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


# ------------------------------------------------------------------ against a retoucher's own result
def compare_hand(cut, tr, after_path: str, levels=("light", "std", "strong"), at_frame: Optional[int] = None) -> dict:
    """DIAP's levels and doing nothing against the hand result (style.py's measures): tone error, colour error (x1000,
    after a global grade fit so a different grade does not count), fine detail against the hand result's."""
    from ..core import media as MD
    from ..retouch import engine as RE
    from ..warp import render as R
    from . import style as SL
    st = tr.store
    info = cut.info
    still = None
    if Path(after_path).suffix.lower() in SL.IMAGE_EXT:
        still = SL._still(after_path, info.width, info.height)
        if at_frame is None:
            at_frame, dd = SL.find_still_frame(cut, after_path)
        ainfo = info
    else:
        ainfo = MD.probe(after_path)
    pf = R._proc_format(cut, cut.output)
    layout = MD.plane_layout(pf, info.width, info.height)
    mtx = R._matrix_name(info)
    rgb = lambda pl: MD.yuv_to_rgb(pl, layout, mtx, info.color_range) if pf.startswith("yuv") else MD.planes_to_rgb_float(pl, layout)
    picks = SL._pick_at(st, int(at_frame)) if still is not None else SL._pick(st, 6)
    frames = {}
    for (p, t) in picks:
        pl = SL._read(info, t, pf)
        if pl is None:
            continue
        o = rgb(pl)
        lm = st.face[p, t]
        if not np.isfinite(lm[:, :2]).all():
            continue
        f = lm[:, :2]
        c = f.mean(0)
        rr = 1.1 * float(np.ptp(f[:, 0]))
        x0, y0 = max(0, int(c[0] - rr)), max(0, int(c[1] - 1.2 * rr))
        x1, y1 = min(info.width, int(c[0] + rr)), min(info.height, int(c[1] + 1.3 * rr))
        loc = lm.copy()
        loc[:, 0] -= x0
        loc[:, 1] -= y0
        m = SL.skin_region(loc, (y1 - y0, x1 - x0))
        if m.sum() < 500:
            continue
        if still is not None:
            a = still
        else:
            ap = SL._read(ainfo, t, pf)
            if ap is None:
                continue
            a = rgb(ap)
        G = SL.fit_grade(o, a, (x0, y0, x1, y1))
        oc = o[y0:y1, x0:x1]
        ac, m2, _ = SL.align_crop(SL.apply_grade(G, oc), a[y0:y1, x0:x1], m)
        frames[(p, t)] = dict(planes=pl, box=(x0, y0, x1, y1), a=ac, m=m2, G=G)
    if not frames:
        return dict(error="no frames")
    K = SL.K

    roles = {q.id: q.role for q in cut.persons}

    def errs(level):
        try:
            E, C, F = [], [], []
            for (p, t), fr in frames.items():
                pid = int(st.pids[p])
                for q in cut.persons:                      # the face compared is the one retouched
                    q.role = "target" if q.id == pid else "protect"
                restore = set_level(cut, level)
                try:
                    rf = RE.prepare(cut, tr, t, {})
                    out = fr["planes"] if rf is None else RE.apply_planes([x.copy() for x in fr["planes"]], layout.subsample,
                                                                          layout.max_value, pf, rf, 1.0, None,
                                                                          info.color_range)
                finally:
                    restore()
                x0, y0, x1, y1 = fr["box"]
                b = SL.apply_grade(fr["G"], rgb(out)[y0:y1, x0:x1])
                a, m = fr["a"], fr["m"]
                dl = cv2.GaussianBlur(((b - a) @ K).astype(np.float32), (0, 0), 3)
                E.append(float(np.sqrt((dl[m] ** 2).mean())) * 1000)
                cr = lambda Z: cv2.GaussianBlur((Z[..., 0] - Z @ K).astype(np.float32), (0, 0), 3)
                C.append(float(np.sqrt(((cr(b) - cr(a))[m] ** 2).mean())) * 1000)
                F.append(SL._fine((b @ K).astype(np.float32), m) / max(1e-6, SL._fine((a @ K).astype(np.float32), m)))
            return dict(tone=round(float(np.mean(E)), 2), colour=round(float(np.mean(C)), 2), fine=round(float(np.mean(F)), 3))
        finally:
            for q in cut.persons:
                q.role = roles.get(q.id, q.role)
    out = dict(frames=[[int(st.pids[p]), int(t)] for (p, t) in frames], none=errs(None))
    for lv in levels:
        out[lv] = errs(lv)
    return out
