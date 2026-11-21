"""A drawn scene for the body skin tests (1.23): a close-up with no face and one shoulder, bare arm and thigh over a
top of nearly the skin's colour, things lying on the skin, marks on it, another person's arm across, a cover passing,
a light change. Everything is known pixel by pixel (kind()), so every test says exactly what may change.

The picture is made at the 'original' size (W x H); the analysis picture is half of it (the 4K original with a
1920 analysis picture, small). The body parser is stood in for by the scene's own truth (truth_parser): the tests
are about what the engine does with what it is told, not about a model."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, N = 960, 720, 8
AW, AH = W // 2, H // 2                 # the analysis picture
MW, MH = AW // 2, AH // 2               # the matte
SKIN = (0.74, 0.57, 0.48)
TOP = (0.77, 0.61, 0.52)                # a top of nearly the skin's colour
OTHER = (0.70, 0.52, 0.44)              # the other person's skin
BG, SKIN_K, CLOTH, STRAP, HAIR, OTHER_K, COVER = 0, 1, 2, 3, 4, 5, 6
# the person: one shoulder, elbow, wrist, hip and knee (source px at frame 0); it drifts by DRIFT a frame
SH, EL, WR, HIP, KN = (250.0, 150.0), (430.0, 310.0), (650.0, 430.0), (300.0, 540.0), (590.0, 640.0)
ARM_HW, FORE_HW, THIGH_HW = 40.0, 32.0, 70.0
DRIFT = (1.5, 0.8)
# marks on the forearm, in the forearm's own measure (along from elbow to wrist, across in its lengths)
PIMPLES = [(0.30, 0.03), (0.62, -0.04)]         # red, small: go
DARKS = [(0.46, 0.05)]                          # a faint dark blemish: goes
MOLE = (0.80, 0.02)                             # deep dark: stays
FRECKLES = [(0.10 + 0.032 * (i % 4), -0.070 + 0.034 * (i // 4)) for i in range(8)]     # many together: stay
BIG = (0.38, -0.05)                             # a red patch well larger than the largest spot of the tests' size: stays
THIGH_SPOT = (0.45, 0.02)                       # a pimple on the thigh (in the thigh's measure)
FLICKER_FRAME = 3                               # THIGH_SPOT is half as strong on this frame
BRIGHT_FRAME = 4                                # the light is a fifth brighter on this frame
COVER_FRAMES = (5, 6)                           # a prop lies over the middle of the forearm
SPOT_SIGMA = 1.5                                # a blemish: about 5 source px across, 4 % of the body's size (2.5 px
                                                # on the analysis picture)
SPOT_SIZE = 70.0                                # the 최대 잡티 크기 the tests use (largest spot: about 6 source px)


def shift(t: int):
    return np.array([DRIFT[0] * t, DRIFT[1] * t])


def local(a, b, uv, t: int = 0):
    """A point given in a limb's own measure (along, across in its lengths), in source px of frame t."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ab = b - a
    L = float(np.hypot(*ab))
    u = ab / L
    n = np.array([-u[1], u[0]])
    return a + (uv[0] * u + uv[1] * n) * L + shift(t)


def _capsule(m, a, b, r, val):
    cv2.line(m, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), int(val), int(round(2 * r)), cv2.LINE_8)
    for q in (a, b):
        cv2.circle(m, (int(round(q[0])), int(round(q[1]))), int(round(r)), int(val), -1, cv2.LINE_8)


def kind(t: int, size=(W, H)) -> np.ndarray:
    """What every pixel of frame t is (BG, SKIN_K: the person's bare skin, CLOTH, STRAP, HAIR, OTHER_K: the other
    person, COVER), uint8 at size (w, h)."""
    k = np.zeros((H, W), np.uint8)
    s = shift(t)
    P = lambda q: np.asarray(q) + s
    # the top: from the shoulder down to the hip, and the thigh below it
    cv2.fillPoly(k, [np.round(np.array([P((170, 60)), P((330, 110)), P((390, 560)), P((120, 600))])).astype(np.int32)], CLOTH)
    _capsule(k, P(HIP), P(KN), THIGH_HW, SKIN_K)
    cv2.fillPoly(k, [np.round(np.array([P((120, 470)), P((400, 470)), P((400, 560)), P((120, 600))])).astype(np.int32)], CLOTH)   # the hem over the thigh's top
    _capsule(k, P(SH), P(EL), ARM_HW, SKIN_K)
    _capsule(k, P(EL), P(WR), FORE_HW, SKIN_K)
    _capsule(k, P(WR), P(WR) + (np.asarray(WR) - np.asarray(EL)) * 0.28, FORE_HW * 1.05, SKIN_K)         # the hand
    # a strap across the upper arm, hair over the shoulder
    a_, b_ = local(SH, EL, (0.45, -0.22), t), local(SH, EL, (0.45, 0.22), t)
    cv2.line(k, tuple(np.round(a_).astype(int)), tuple(np.round(b_).astype(int)), STRAP, 7)
    for i in range(5):
        h0, h1 = P((225 + 9 * i, 70)), P((245 + 11 * i, 190 + 6 * i))
        cv2.line(k, tuple(np.round(h0).astype(int)), tuple(np.round(h1).astype(int)), HAIR, 3)
    # the other person's arm, across the upper arm, in front; it moves the other way
    o = np.array([-1.2 * t, 0.6 * t])
    _capsule(k, np.array((900.0, 40.0)) + o, np.array((420.0, 205.0)) + o, 30.0, OTHER_K)
    if t in COVER_FRAMES:
        c = local(EL, WR, (0.50, 0.0), t)
        cv2.rectangle(k, (int(c[0] - 55), int(c[1] - 70)), (int(c[0] + 55), int(c[1] + 70)), COVER, -1)
    if tuple(size) != (W, H):
        k = cv2.resize(k, tuple(size), interpolation=cv2.INTER_NEAREST)
    return k


def _spot(f, c, sigma, dy=0.0, dcr=0.0):
    """A soft round mark: luma down by dy (relative), redder by dcr."""
    x0, x1 = max(0, int(c[0] - 5 * sigma)), min(W, int(c[0] + 5 * sigma) + 1)
    y0, y1 = max(0, int(c[1] - 5 * sigma)), min(H, int(c[1] + 5 * sigma) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * sigma ** 2))
    f[y0:y1, x0:x1] *= (1 - dy * g)[..., None]
    f[y0:y1, x0:x1, 0] += 1.5748 * dcr * g
    f[y0:y1, x0:x1, 1] -= 0.4681 * dcr * g


def frame(t: int, marks: bool = True, tex: float = 1.0) -> np.ndarray:
    """Frame t at the original's size, uint8 RGB."""
    k = kind(t)
    rng = np.random.default_rng(1230)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.empty((H, W, 3), np.float32)
    f[..., 0] = 0.16 + 0.05 * xx / W
    f[..., 1] = 0.20 + 0.04 * yy / H
    f[..., 2] = 0.27 + 0.03 * np.sin(xx / 23.0)
    # the skin: its colour, the limb's own shading, a texture that moves with the person and grain that does not
    s = shift(t)
    base_tex = cv2.GaussianBlur(rng.normal(0, 1.0, (H + 64, W + 64)).astype(np.float32), (0, 0), 1.3)
    base_tex /= float(base_tex.std())
    A = np.array([[1, 0, s[0]], [0, 1, s[1]]], np.float32)
    texm = cv2.warpAffine(base_tex, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shade = 1.0 + 0.10 * np.sin((xx - s[0]) / 90.0) * np.cos((yy - s[1]) / 120.0)
    for kk, col, tx in ((SKIN_K, SKIN, 0.012 * tex), (OTHER_K, OTHER, 0.012 * tex)):
        sel = k == kk
        for c in range(3):
            f[..., c][sel] = (col[c] * shade * (1.0 + tx * texm))[sel]
    sel = k == CLOTH
    stripes = 0.05 * (((xx - s[0]) // 6) % 2)
    for c in range(3):
        f[..., c][sel] = (TOP[c] * (1.0 + stripes))[sel]
    f[k == STRAP] = (0.07, 0.07, 0.08)
    f[k == HAIR] = (0.10, 0.07, 0.05)
    f[k == COVER] = (0.35, 0.36, 0.40)
    if marks:
        L = float(np.hypot(*(np.asarray(WR) - np.asarray(EL))))
        for uv in PIMPLES:
            _spot(f, local(EL, WR, uv, t), SPOT_SIGMA, dy=0.06, dcr=0.05)
        for uv in DARKS:
            _spot(f, local(EL, WR, uv, t), SPOT_SIGMA, dy=0.14)
        _spot(f, local(EL, WR, MOLE, t), 1.7, dy=0.60)
        for uv in FRECKLES:
            _spot(f, local(EL, WR, uv, t), 1.1, dy=0.17)
        _spot(f, local(EL, WR, BIG, t), 8.5, dy=0.03, dcr=0.045)
        _spot(f, local(HIP, KN, THIGH_SPOT, t), SPOT_SIGMA, dy=0.06, dcr=(0.024 if t == FLICKER_FRAME else 0.05))
        # ink: a dark blue drawing on the thigh
        c = local(HIP, KN, (0.70, -0.05), t)
        ink = np.zeros((H, W), np.uint8)
        cv2.ellipse(ink, (int(c[0]), int(c[1])), (34, 20), 25, 0, 360, 255, 3)
        cv2.line(ink, (int(c[0] - 30), int(c[1] - 12)), (int(c[0] + 30), int(c[1] + 14)), 255, 3)
        inkf = cv2.GaussianBlur(ink.astype(np.float32) / 255.0, (0, 0), 0.8)[..., None] * (k == SKIN_K)[..., None]
        f = f * (1 - inkf) + np.array((0.12, 0.16, 0.30), np.float32) * inkf
        # the things painted last are theirs again
        f[k == COVER] = (0.35, 0.36, 0.40)
    f = cv2.GaussianBlur(f, (0, 0), 0.6)                         # the lens
    f += np.random.default_rng(1230 + t).normal(0, 0.006, (H, W, 1)).astype(np.float32)      # grain, new on every frame
    if t == BRIGHT_FRAME:
        f *= 1.2
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def ink_mask(t: int) -> np.ndarray:
    c = local(HIP, KN, (0.70, -0.05), t)
    ink = np.zeros((H, W), np.uint8)
    cv2.ellipse(ink, (int(c[0]), int(c[1])), (34, 20), 25, 0, 360, 255, 9)
    cv2.line(ink, (int(c[0] - 30), int(c[1] - 12)), (int(c[0] + 30), int(c[1] + 14)), 255, 9)
    return ink > 0


def write_source(path: Path, marks: bool = True, tex: float = 1.0) -> Path:
    from contour.core import media as MD
    MD.configure_ffmpeg()
    cmd = [MD.ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv444p", "-qp", "0", "-color_range", "tv", "-colorspace",
           "bt709", str(path)]
    subprocess.run(cmd, input=b"".join(frame(t, marks, tex).tobytes() for t in range(N)), check=True)
    return path


def make_project(folder: Path, marks: bool = True, tex: float = 1.0, jump_at: int = -1):
    """The scene as an analysed project: the person (id 0, target) with the joints a close-up gives (no head, one
    shoulder), the other person (id 1), matte and owners as an analysis leaves them. The analysis picture is half
    the original. jump_at: the picture jumps between that frame and the next (a cut inside the clip)."""
    from contour.core import media as MD
    from contour.core.project import Person, Project
    from contour.core.settings import settings
    from contour.track import skeleton as S
    from contour.track.store import TrackStore
    from contour.warp.body import SEG_INDEX
    src = write_source(folder / "body.mkv", marks, tex)
    settings().proxy_long_side = AW
    prj = Project.create(str(folder), "body")
    cut = prj.add_cut(MD.probe(str(src)), "c1")
    cut.analyzed = True
    cut.analysis_version = 2
    cut.cs_analysed = ""
    cut.persons = [Person(id=0, name="A", color="#ff8800", role="target"),
                   Person(id=1, name="B", color="#00aaff", role="protect")]
    st = TrackStore(N, [0, 1])
    st.has_vis = True
    st.has_face = False
    for t in range(N):
        s = shift(t)
        for j, q in ((S.L_SH, SH), (S.L_EL, EL), (S.L_WR, WR), (S.L_HIP, HIP), (S.L_KN, KN)):
            st.final[0, t, j, :2] = np.asarray(q) + s
            st.final[0, t, j, 2] = 0.9
        st.vis[0, t] = 0
        st.vis[0, t, [S.L_SH, S.L_EL, S.L_WR, S.L_HIP, S.L_KN]] = 1
        st.present[0, t] = True
        st.boxes[0, t] = [120 + s[0], 60 + s[1], 720 + s[0], 715]
        o = np.array([-1.2 * t, 0.6 * t])
        st.final[1, t, S.R_SH, :2] = np.array((900.0, 40.0)) + o
        st.final[1, t, S.R_EL, :2] = np.array((420.0, 205.0)) + o
        st.final[1, t, [S.R_SH, S.R_EL], 2] = 0.9
        st.vis[1, t] = 0
        st.vis[1, t, [S.R_SH, S.R_EL]] = 1
        st.present[1, t] = True
        st.boxes[1, t] = [380 + o[0], 0, 940 + o[0], 250 + o[1]]
    st.scale[:] = np.nan
    st.conf[:] = 0.9
    st.widths[0, :, SEG_INDEX["upper_arm_l"]] = ARM_HW
    st.widths[0, :, SEG_INDEX["forearm_l"]] = FORE_HW
    st.widths[0, :, SEG_INDEX["thigh_l"]] = THIGH_HW
    if jump_at >= 0:
        st.jump[jump_at] = True
    st.save(prj.cut_dir(cut) / "tracks.npz")
    cache = prj.cache_dir(cut)
    mm = np.memmap(cache / f"matte_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    lb = np.memmap(cache / f"seglab_{MW}x{MH}.u8", np.uint8, "w+", shape=(N, MH, MW))
    for t in range(N):
        k = kind(t, (MW, MH))
        mine = np.isin(k, (SKIN_K, CLOTH, STRAP, HAIR))
        mm[t][mine | (k == OTHER_K)] = 255
        lb[t][mine] = 1
        lb[t][k == OTHER_K] = 2
    mm.flush()
    lb.flush()
    del mm, lb
    (cache / "seg_meta.json").write_text(json.dumps(dict(model="SAM 3.1", made=1795000000.0, refreshed=[],
                                                         range=[0, N])), encoding="utf-8")
    prj.save()
    return prj, cut


def truth_parser(monkeypatch, wrong=None):
    """The body parser stood in for by the scene's own truth: skin where the person's or the other's skin is, cloth
    on the top, the strap and a cover, hair on the hair. wrong(t, kind map) -> kind map: what the parser gets wrong
    on a frame (a hole, a patch of cloth called skin)."""
    from contour.retouch import bodyskin as BS
    calls = []

    def parse(tracker, p, t, rgb_u8, own):
        calls.append((int(p), int(t)))
        h, w = rgb_u8.shape[:2]
        k = kind(int(t), (w, h))
        if wrong is not None:
            k = wrong(int(t), k)
        ys = np.nonzero(own.max(1) > 0.3)[0]
        xs = np.nonzero(own.max(0) > 0.3)[0]
        if len(ys) < 4 or len(xs) < 4:
            return None
        x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
        sz = 256

        def m(sel):
            a = cv2.resize(sel[y0:y1, x0:x1].astype(np.float32), (sz, sz), interpolation=cv2.INTER_AREA)
            return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
        return dict(box=(x0, y0, x1, y1), skin=m(np.isin(k, (SKIN_K, OTHER_K))), hair=m(k == HAIR),
                    cloth=m(np.isin(k, (CLOTH, STRAP, COVER))))
    monkeypatch.setattr(BS, "_parse_cached", parse)
    return calls


def open_cut(prj, cut):
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    fr = frames_for(prj, cut)
    return fr, CutTracker(prj, cut, fr)


def body_on(cut, pid: int = 0, skin: float = 0.0, blemish: float = 0.0, scope: str = "all", detail: float = 60.0,
            size: float = 40.0):
    from contour.core.project import Curve
    pp = cut.params_for(pid)
    pp.amounts["body_skin"] = Curve(base=float(skin))
    pp.amounts["body_blemish"] = Curve(base=float(blemish))
    pp.body_scope, pp.body_detail, pp.body_blemish_size = scope, float(detail), float(size)
    return pp


def preview(cut, tr, t: int, cache=None):
    """Frame t of the preview: (before, after) uint8 at the analysis size."""
    from contour.retouch import engine as RE
    px = tr.proxy
    rgb = np.asarray(px.frame(t))
    rf = RE.prepare(cut, tr, t, cache if cache is not None else {})
    return rgb, (RE.apply_rgb(rgb, rf, px.sx) if rf is not None else rgb), rf


def luma(rgb_u8: np.ndarray) -> np.ndarray:
    f = rgb_u8.astype(np.float32) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def contrast(rgb_u8: np.ndarray, c, r: float = 4.0, ring: float = 9.0, red: bool = False) -> float:
    """How far a mark at c (this picture's px) stands off the skin round it: luma (or redness) of the middle
    against a ring."""
    f = rgb_u8.astype(np.float32) / 255.0
    v = (f[..., 0] - f[..., 1]) if red else luma(rgb_u8)
    yy, xx = np.mgrid[0:rgb_u8.shape[0], 0:rgb_u8.shape[1]].astype(np.float32)
    d = np.hypot(xx - c[0], yy - c[1])
    inner, outer = d <= r, (d > ring) & (d <= ring + 4)
    return float(v[outer].mean() - v[inner].mean()) * (-1.0 if red else 1.0)
