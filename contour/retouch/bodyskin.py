"""Body skin (1.23): where a person's visible skin outside the face is, part by part, with or without a face.

What the face's retouch knows of the body (engine.body_regions) hangs on both shoulders (every size there is a share
of the shoulder width) and on the face's colour model: a close-up of an arm, of legs or of a back has neither. Here:

  parts   each limb from its own two joints and its measured width, the neck and the shoulders from whichever of
          them are there, the head left out. The trunk is what is left of the person, so it needs no joint at all
          (a back filling the frame is all trunk).
  skin    a body parser (MediaPipe Selfie Multiclass on the person's own piece of the picture: body skin, hair,
          clothes, accessories) says what kind of thing a place is; the skin's colour is learned from what it calls
          skin on this very frame and decides pixel by pixel; the person's matte and the owner of each pixel say
          whose it is; garments the analysis found (track.clothing) are left out. Whatever is not surely this
          person's skin stays as it is.
  sizes   every radius and spot size follows the face width a body of this size has (BodyInfo.ref), from the limbs'
          lengths and widths and the face when it is there, never from the frame's size.

The masks are made on the analysis picture like every other mask of the engine; refine() then fits their edge to
the picture that is being worked on (the original's planes at a render), so a 4K export does not get an enlarged
proxy mask.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import skin as SK
from ..track import skeleton as S
from ..warp.body import SEG_INDEX, SEGMENTS

# ------------------------------------------------------------------------------------------------- parts
# name, region, joint a, joint b, length over the face width, full width over the face width (adult build)
LIMBS = [
    ("upper_arm_l", "arm", S.L_SH, S.L_EL, 2.15, 0.63),
    ("upper_arm_r", "arm", S.R_SH, S.R_EL, 2.15, 0.63),
    ("forearm_l", "arm", S.L_EL, S.L_WR, 1.85, 0.52),
    ("forearm_r", "arm", S.R_EL, S.R_WR, 1.85, 0.52),
    ("thigh_l", "leg", S.L_HIP, S.L_KN, 3.10, 1.18),
    ("thigh_r", "leg", S.R_HIP, S.R_KN, 3.10, 1.18),
    ("calf_l", "leg", S.L_KN, S.L_AN, 2.95, 0.80),
    ("calf_r", "leg", S.R_KN, S.R_AN, 2.95, 0.80),
]
SHOULDERS_REF = 2.45        # shoulder joint to shoulder joint over the face width
TRUNK_REF = 3.55            # shoulder line to hip line
HIPS_REF = 1.55             # hip joint to hip joint
HEIGHT_REF = 0.082          # the face width over the body height (track.skeleton.person_scale)
REF_MIN = 12.0              # source px: below this nothing can be said about skin texture
REF_FRAMES = 4              # frames each side the size is taken over (median, same shot)
LIMB_REACH = 1.25           # a limb's region reaches this many half widths from its axis
HAND_LEN, HAND_R = 0.75, 0.24     # the hand beyond the wrist, of the forearm's length
FOOT_LEN, FOOT_R = 0.22, 0.26     # the foot beyond the ankle, of the calf's length
CANVAS = 512                # the regions are drawn on a picture at most this long (they are soft shapes)

VIS_HIDDEN = 0


@dataclass
class Part:
    name: str
    region: str             # neck | arm | torso | leg
    a: np.ndarray           # (2,) source px
    b: np.ndarray
    hw: float               # half width, source px


@dataclass
class BodyInfo:
    """One person's body on one frame, as far as the joints tell."""
    parts: Dict[str, Part] = field(default_factory=dict)
    ref: float = 0.0                       # the face width a body of this size has, source px
    head: Optional[Tuple[str, np.ndarray, float]] = None    # ("poly", points, face width) or ("circle", centre, radius)
    model: Optional[tuple] = None          # the body skin's colour model of this frame (skin.build_skin_model)
    parsed: bool = False                   # the body parser's classes were used
    box: Optional[Tuple[float, float, float, float]] = None  # the person's piece of the analysis picture
    allow: Optional[np.ndarray] = None     # where this frame itself does not rule bare skin out (see skin_conf)


def _joints(st, p: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Joints that can place a body part, source px (NaN where there is none), and which of them are hidden."""
    kp = st.final[p, t]
    xy = kp[:, :2].astype(np.float64).copy()
    ok = np.isfinite(xy).all(1)
    if kp.shape[1] > 2:
        ok &= np.nan_to_num(kp[:, 2]) > 0.2
    xy[~ok] = np.nan
    hid = np.zeros(len(xy), bool)
    if getattr(st, "has_vis", False) and getattr(st, "vis", None) is not None:
        hid = np.asarray(st.vis[p, t]) == VIS_HIDDEN
    return xy, hid


def _shot(st, p: int, t: int, r: int) -> List[int]:
    n = st.final.shape[1]
    return [s for s in range(max(0, t - r), min(n, t + r + 1)) if st.present[p, s] and st.connected(s, t)]


def _half_width(st, p: int, frames: List[int], seg: str, length: float) -> float:
    """A segment's half width (source px): what the matte measured, median over the frames round it, kept within
    reach of what a limb of this length has (a sleeve, an arm lying against the trunk measure wide)."""
    i = SEG_INDEX[seg]
    fb = float(SEGMENTS[i].fallback) * float(length)
    w = None
    if getattr(st, "widths", None) is not None and frames:
        w = np.asarray(st.widths[p, frames, i], np.float64)
    if w is None or not np.isfinite(w).any():
        return fb
    return float(np.clip(np.nanmedian(w), 0.5 * fb, 2.0 * fb))


def _ref_one(st, p: int, t: int) -> float:
    """The face width (source px) a body of this frame's size has, from whatever of it is seen: each limb's length
    and width, the shoulder and hip lines, the trunk, the face itself. The median of them: a limb pointing at the
    camera is short, a sleeve is wide, a face in profile is narrow, and none of them alone decides."""
    xy, _hid = _joints(st, p, t)
    est = []

    def d(a, b):
        if np.isfinite(xy[[a, b]]).all():
            v = float(np.hypot(*(xy[a] - xy[b])))
            return v if v > 2.0 else None
        return None
    for name, _reg, a, b, kl, kw in LIMBS:
        L = d(a, b)
        if L is None:
            continue
        est.append(L / kl)
        if getattr(st, "widths", None) is not None:
            w = np.asarray(st.widths[p, t, SEG_INDEX[name]], np.float64)
            if np.isfinite(w).any():
                est.append(2.0 * float(np.nanmean(w)) / kw)
    v = d(S.L_SH, S.R_SH)
    if v:
        est.append(v / SHOULDERS_REF)
    v = d(S.L_HIP, S.R_HIP)
    if v:
        est.append(v / HIPS_REF)
    for a, b in ((S.L_SH, S.L_HIP), (S.R_SH, S.R_HIP)):
        v = d(a, b)
        if v:
            est.append(v / TRUNK_REF)
    if getattr(st, "has_face", False) and np.isfinite(st.face[p, t, [0, 16], :2]).all() and float(st.face_vis[p, t]) > 0.5:
        wf = float(np.hypot(*(st.face[p, t, 16, :2] - st.face[p, t, 0, :2])))
        if wf > 4:
            est += [wf, wf]                     # the face itself counts twice
    if not est:
        sc = float(st.scale[p, t]) if np.isfinite(st.scale[p, t]) else 0.0
        return HEIGHT_REF * sc if sc > 0 else 0.0
    return float(np.median(est))


def ref_width(st, p: int, t: int) -> float:
    """BodyInfo.ref: _ref_one held steady over the frames round t (a size that changes from frame to frame makes
    the smoothing's bands breathe)."""
    vals = [v for v in (_ref_one(st, p, s) for s in _shot(st, p, t, REF_FRAMES)) if v > 0]
    return float(np.median(vals)) if vals else 0.0


def body_info(st, p: int, t: int) -> BodyInfo:
    """The parts of person p on frame t that the joints place, the head to leave out and the size."""
    xy, hid = _joints(st, p, t)
    frames = _shot(st, p, t, REF_FRAMES)
    info = BodyInfo(ref=ref_width(st, p, t))
    P = info.parts

    def seg(a, b):
        if not np.isfinite(xy[[a, b]]).all() or (hid[a] and hid[b]):
            return None                     # a limb with both ends hidden is behind something
        L = float(np.hypot(*(xy[b] - xy[a])))
        return L if L > 2.0 else None
    for name, reg, a, b, _kl, _kw in LIMBS:
        L = seg(a, b)
        if L is not None:
            P[name] = Part(name, reg, xy[a].copy(), xy[b].copy(), _half_width(st, p, frames, name, L))
    for side, fa, wr in (("l", "forearm_l", S.L_WR), ("r", "forearm_r", S.R_WR)):
        f = P.get(fa)
        if f is not None and not hid[wr]:
            u = (f.b - f.a) / max(1e-6, float(np.hypot(*(f.b - f.a))))
            L = float(np.hypot(*(f.b - f.a)))
            P[f"hand_{side}"] = Part(f"hand_{side}", "arm", f.b.copy(), f.b + u * HAND_LEN * L, HAND_R * L)
    for side, ca, an in (("l", "calf_l", S.L_AN), ("r", "calf_r", S.R_AN)):
        c = P.get(ca)
        if c is not None and not hid[an]:
            u = (c.b - c.a) / max(1e-6, float(np.hypot(*(c.b - c.a))))
            L = float(np.hypot(*(c.b - c.a)))
            P[f"foot_{side}"] = Part(f"foot_{side}", "leg", c.b.copy(), c.b + u * FOOT_LEN * L, FOOT_R * L)
    # the neck and the shoulders
    ls, rs = xy[S.L_SH], xy[S.R_SH]
    head = [xy[j] for j in (S.NOSE, S.L_EYE, S.R_EYE, S.L_EAR, S.R_EAR) if np.isfinite(xy[j]).all()]
    hc = np.mean(head, 0) if head else None
    if np.isfinite(ls).all() and np.isfinite(rs).all():
        sw = float(np.hypot(*(ls - rs)))
        if sw > 4:
            mid = (ls + rs) / 2
            P["shoulders"] = Part("shoulders", "neck", ls.copy(), rs.copy(), 0.30 * sw)
            if hc is not None:
                L = float(np.hypot(*(hc - mid)))
                if L > 2:
                    P["neck"] = Part("neck", "neck", mid, hc.copy(), _half_width(st, p, frames, "neck", L))
    else:
        for s_, ua in ((ls, "upper_arm_l"), (rs, "upper_arm_r")):
            if np.isfinite(s_).all() and info.ref > 0:
                # one shoulder in the picture: the round of the shoulder itself
                P["shoulders"] = Part("shoulders", "neck", s_.copy(), s_.copy(), 0.55 * info.ref)
    # the trunk: only as something a hand mark can be fixed to (its region is what the other parts leave)
    hips = [xy[j] for j in (S.L_HIP, S.R_HIP) if np.isfinite(xy[j]).all()]
    shs = [q for q in (ls, rs) if np.isfinite(q).all()]
    if hips and shs:
        a_, b_ = np.mean(shs, 0), np.mean(hips, 0)
        L = float(np.hypot(*(b_ - a_)))
        if L > 4:
            P["trunk"] = Part("trunk", "torso", a_, b_, 0.35 * L)
    # the head: never the body's
    lm = None
    if getattr(st, "has_face", False) and np.isfinite(st.face[p, t, :, :2]).all():
        lm = st.face[p, t]
    if lm is not None:
        poly = SK.face_polygon(lm, SK.FOREHEAD_PARSED)
        if poly is not None:
            info.head = ("poly", np.asarray(poly, np.float64), float(np.hypot(*(lm[16, :2] - lm[0, :2]))))
    if info.head is None and hc is not None and len(head) >= 2:
        spread = float(np.max(np.linalg.norm(np.asarray(head) - hc, axis=1)))
        info.head = ("circle", hc, max(1.9 * spread, 0.75 * info.ref))
    return info


def _capsule(m: np.ndarray, a, b, r: float) -> None:
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or r < 0.5:
        return
    pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
    cv2.line(m, pa, pb, 255, max(1, int(round(2 * r))), cv2.LINE_AA)
    for q in (pa, pb):
        cv2.circle(m, q, max(1, int(round(r))), 255, -1, cv2.LINE_AA)


def region_maps(info: BodyInfo, shape: Tuple[int, int], sx: float, sy: float,
                hands: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Where each body region is on a picture of `shape` (the analysis picture; sx, sy: its pixels per source
    pixel), float32 0..1: 'arm', 'leg', 'neck', 'head' and 'torso'. The trunk is what the others leave. Soft
    shapes: what really is skin there, and whose, is skin_conf's to say. hands: (42, >=2) tracked hand points."""
    h, w = shape
    q = min(1.0, CANVAS / float(max(h, w)))
    ch, cw = max(8, int(round(h * q))), max(8, int(round(w * q)))
    k = np.array([sx * cw / w, sy * ch / h])
    maps = {n: np.zeros((ch, cw), np.uint8) for n in ("arm", "leg", "neck", "head")}
    soft = max(1.0, 0.05 * info.ref * float(k.mean())) if info.ref > 0 else 2.0
    for part in info.parts.values():
        if part.region not in maps:
            continue
        m = maps[part.region]
        if part.name == "shoulders":
            if np.allclose(part.a, part.b):
                cv2.circle(m, (int(round(part.a[0] * k[0])), int(round(part.a[1] * k[1]))),
                           max(1, int(round(part.hw * k.mean()))), 255, -1, cv2.LINE_AA)
                continue
            ls, rs = part.a * k, part.b * k
            sw = float(np.hypot(*(ls - rs)))
            side = (ls - rs) / max(sw, 1e-6)
            down = np.array([-side[1], side[0]])
            hips = [info.parts[n].a for n in ("thigh_l", "thigh_r") if n in info.parts]
            if hips:
                dv = np.mean(hips, 0) * k - (ls + rs) / 2
                if float(dv @ down) < 0:
                    down = -down
            elif "neck" in info.parts:
                dv = info.parts["neck"].b * k - (ls + rs) / 2
                if float(dv @ down) > 0:
                    down = -down
            elif down[1] < 0:
                down = -down
            pts = [ls + side * 0.18 * sw - down * 0.12 * sw, rs - side * 0.18 * sw - down * 0.12 * sw,
                   rs - side * 0.10 * sw + down * 0.45 * sw, ls + side * 0.10 * sw + down * 0.45 * sw]
            cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255, cv2.LINE_AA)
        else:
            _capsule(m, part.a * k, part.b * k, LIMB_REACH * part.hw * float(k.mean()))
    if hands is not None and info.parts:
        H = np.asarray(hands, np.float64)[:, :2] * k
        okh = np.isfinite(H).all(1)
        for i0 in (0, 21):
            pts = H[i0:i0 + 21][okh[i0:i0 + 21]]
            if len(pts) >= 5:
                cv2.fillConvexPoly(maps["arm"], cv2.convexHull(np.round(pts).astype(np.int32)), 255, cv2.LINE_AA)
    if info.head is not None:
        if info.head[0] == "poly":
            wf = info.head[2] * float(k.mean())
            hm = maps["head"]
            cv2.fillPoly(hm, [np.round(info.head[1][:, :2] * k).astype(np.int32)], 255, cv2.LINE_AA)
            kd = max(3, int(round(0.10 * wf)) | 1)
            maps["head"] = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        else:
            c, r = info.head[1] * k, info.head[2] * float(k.mean())
            cv2.circle(maps["head"], (int(round(c[0])), int(round(c[1]))), max(1, int(round(r))), 255, -1, cv2.LINE_AA)
    out = {}
    for n, m in maps.items():
        f = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), soft) if m.any() else np.zeros((ch, cw), np.float32)
        out[n] = f
    hd = np.clip(out["head"] * 1.6, 0, 1)
    for n in ("arm", "leg", "neck"):
        out[n] = out[n] * (1.0 - hd)
    out["head"] = hd
    out["torso"] = np.clip(1.0 - np.maximum.reduce([out["arm"], out["leg"], out["neck"]]) * 1.5, 0, 1) * (1.0 - hd)
    if (ch, cw) != (h, w):
        out = {n: cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for n, m in out.items()}
    return out


def chosen(maps: Dict[str, np.ndarray], regions: set) -> np.ndarray:
    """The chosen regions as one map. All four: everything but the head (전신: also where no joint places a part)."""
    if {"neck", "arm", "torso", "leg"} <= set(regions):
        return (1.0 - maps["head"]).astype(np.float32)
    out = None
    for n in ("neck", "arm", "torso", "leg"):
        if n in regions:
            out = maps[n] if out is None else np.maximum(out, maps[n])
    if out is None:
        return np.zeros_like(maps["head"])
    return np.clip(out, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------- the parser
PARSE_ON = True             # False: colour, matte and garments alone say what is skin
PARSE_PAD = 0.06            # the person's box is widened by this share before it is parsed
PARSE_ASPECT = 1.6          # and its long side is at most this many times its short side
PARSE_LRU = 96              # person-frames kept (each four 256 x 256 maps)
_PLOCK = threading.Lock()
_PARSER = [None]            # the session, False once it cannot be made
_PSTATE = dict(cpu=False, error="")


class _Parser:
    """MediaPipe Selfie Multiclass (Apache-2.0, shipped with the app: track/data) on a person's piece of the
    picture: background, hair, body skin, face skin, clothes, accessories."""

    def __init__(self, cpu: bool = False):
        from ..track import faceparse as FP
        from ..track import models as M
        self.size = FP.MP_SIZE
        if cpu:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.s = ort.InferenceSession(str(FP.MP_FILE), sess_options=so, providers=["CPUExecutionProvider"])
        else:
            self.s = M.session(FP.MP_FILE)
        self.inp = self.s.get_inputs()[0].name
        self.key = "mp@" + FP.file_fp(FP.MP_FILE)

    def probs(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (self.size, self.size),
                       interpolation=cv2.INTER_AREA if max(crop.shape[:2]) > self.size else cv2.INTER_LINEAR)
        lo = self.s.run(None, {self.inp: (x.astype(np.float32) / 255.0)[None]})[0][0].transpose(2, 0, 1)
        lo = lo - lo.max(0, keepdims=True)
        e = np.exp(lo)
        return e / e.sum(0, keepdims=True)


def parser_state() -> dict:
    """For the diagnosis: whether the body parser runs, on what, and why not."""
    return dict(on=bool(PARSE_ON and _PARSER[0]), tried=_PARSER[0] is not None, **_PSTATE)


def _probs(crop: np.ndarray) -> Optional[np.ndarray]:
    """The parser's class probabilities of a piece of picture, or None when the parser cannot run. Short of GPU
    memory it runs on the CPU from then on (a small model): the same answer on every frame matters more than speed."""
    if not PARSE_ON:
        return None
    with _PLOCK:
        if _PARSER[0] is None:
            try:
                _PARSER[0] = _Parser()
            except Exception as e:
                _PSTATE["error"] = repr(e)[:200]
                try:
                    _PARSER[0] = _Parser(cpu=True)
                    _PSTATE["cpu"] = True
                except Exception as e2:
                    _PSTATE["error"] = repr(e2)[:200]
                    _PARSER[0] = False
        if not _PARSER[0]:
            return None
        try:
            return _PARSER[0].probs(crop)
        except Exception as e:
            _PSTATE["error"] = repr(e)[:200]
            if _PSTATE["cpu"]:
                _PARSER[0] = False
                return None
            try:
                _PARSER[0] = _Parser(cpu=True)
                _PSTATE["cpu"] = True
                return _PARSER[0].probs(crop)
            except Exception as e2:
                _PSTATE["error"] = repr(e2)[:200]
                _PARSER[0] = False
                return None


def _widen(a0: int, a1: int, want: int, limit: int) -> Tuple[int, int]:
    """[a0, a1) grown to `want` long about its middle, kept inside [0, limit)."""
    want = min(want, limit)
    c = (a0 + a1) // 2
    b0 = max(0, min(c - want // 2, limit - want))
    return b0, b0 + want


def parse_person(rgb_u8: np.ndarray, own: np.ndarray) -> Optional[dict]:
    """The parser's view of one person: {'box': (x0, y0, x1, y1) in this picture, 'skin', 'hair', 'cloth': uint8
    maps of the box at the parser's size}. own: the person's own pixels (0..1) on this picture."""
    ys = np.nonzero(own.max(1) > 0.3)[0]
    xs = np.nonzero(own.max(0) > 0.3)[0]
    if len(ys) < 4 or len(xs) < 4:
        return None
    h, w = own.shape
    py, px_ = PARSE_PAD * (ys[-1] - ys[0] + 1), PARSE_PAD * (xs[-1] - xs[0] + 1)
    y0, y1 = max(0, int(ys[0] - py)), min(h, int(ys[-1] + 1 + py))
    x0, x1 = max(0, int(xs[0] - px_)), min(w, int(xs[-1] + 1 + px_))
    # the parser sees its piece as a square: a long thin piece (one arm at the frame's edge) is squeezed out of
    # shape and it no longer knows a limb when it sees one. The short side takes in more of the picture
    bh, bw = y1 - y0, x1 - x0
    if bw * PARSE_ASPECT < bh:
        x0, x1 = _widen(x0, x1, int(np.ceil(bh / PARSE_ASPECT)), w)
    elif bh * PARSE_ASPECT < bw:
        y0, y1 = _widen(y0, y1, int(np.ceil(bw / PARSE_ASPECT)), h)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    pr = _probs(np.ascontiguousarray(rgb_u8[y0:y1, x0:x1]))
    if pr is None:
        return None
    from ..track import faceparse as FP
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    return dict(box=(x0, y0, x1, y1), skin=u8(pr[FP.MP_BODY] + pr[FP.MP_FACE]), hair=u8(pr[FP.MP_HAIR]),
                cloth=u8(pr[FP.MP_CLOTHES] + pr[FP.MP_OTHERS]))


def _place(parse: dict, name: str, shape: Tuple[int, int]) -> np.ndarray:
    """One of the parser's maps on the whole picture (float32 0..1; 0 outside the person's box)."""
    x0, y0, x1, y1 = parse["box"]
    out = np.zeros(shape, np.float32)
    out[y0:y1, x0:x1] = cv2.resize(parse[name], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return out


# ------------------------------------------------------------------------------------------------- skin
SKIN_SURE = (0.55, 0.80)    # the parser's skin share from which a pixel teaches the colour model .. fully
NOT_SKIN = (0.35, 0.65)     # the parser's clothes / accessories / hair share from which a pixel is left out .. fully
SUPPORT = 0.035             # of the person's size: skin the parser misses counts this near to skin it found
SUPPORT_GAIN = 3.0
OTHERS_NEAR = 0.012         # of the frame's height: this near to another person's pixels the owner is not sure
CONF = (0.30, 0.62)         # the joined confidence from which a pixel is retouched .. in full
COLOUR_ONLY = 0.75          # without the parser the colour alone must be this sure (0..1 of skin_probability)
MODEL_MIN = 150             # skin pixels needed to learn the colour from


def _ss(e0: float, e1: float, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def fit_model(rgb_u8: np.ndarray, weight: np.ndarray) -> Optional[tuple]:
    """The colour of the skin where weight says skin is: (mean (Cr, Cb), inverse covariance, median luma), as
    skin.build_skin_model makes it from a face."""
    sel = weight > 0.5
    n = int(sel.sum())
    if n < MODEL_MIN:
        return None
    ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
    px = ycc[sel].astype(np.float32)
    px = px[(px[:, 0] > 12) & (px[:, 0] < 245)]
    if len(px) < MODEL_MIN:
        return None
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    yref = float(np.median(px[:, 0]))
    X = px[:, 1:3]
    med = np.median(X, 0)
    d = np.linalg.norm(X - med, axis=1)
    keep = d <= np.percentile(d, 85)
    if keep.sum() >= 20:
        X = X[keep]
    cov = np.cov(X.T) + np.eye(2) * 12.0
    return X.mean(0), np.linalg.inv(cov), yref


def skin_conf(rgb_u8: np.ndarray, own: np.ndarray, others: Optional[np.ndarray], parse: Optional[dict],
              garment: Optional[np.ndarray], prior: Optional[tuple],
              size: float) -> Tuple[np.ndarray, Optional[tuple], bool, np.ndarray]:
    """How sure each pixel is this person's bare skin (float32 0..1), the colour model that said so, whether the
    parser was used, and where this frame does not rule skin out. own: the person's matte where the pixel is
    theirs; others: everybody else's; garment: 1 on clothes the analysis found; prior: a colour model to start from
    when there is no parser (the face's); size: the person's size on this picture (px) for the distances.

    The last map is what the frame knows for sure whatever the colour says: the pixel is the person's own, the
    parser does not call it clothes or hair, the analysis found no garment on it, no other person is beside it.
    Where it is 0 something lies over the skin on this frame (a sleeve, hair, a prop, somebody's hand), and no
    neighbouring frame's skin is carried there."""
    h, w = own.shape
    model = None
    not_other = np.ones((h, w), np.float32)
    support = np.ones((h, w), np.float32)
    parsed = parse is not None
    if parsed:
        sk = _place(parse, "skin", (h, w))
        no = np.maximum(_place(parse, "cloth", (h, w)), _place(parse, "hair", (h, w)))
        not_other = 1.0 - _ss(NOT_SKIN[0], NOT_SKIN[1], no)
        teach = _ss(SKIN_SURE[0], SKIN_SURE[1], sk) * (own > 0.8)
        if garment is not None:
            teach = teach * (garment < 0.3)
        model = fit_model(rgb_u8, teach)
        sg = max(1.5, SUPPORT * size)
        support = np.clip(cv2.GaussianBlur(sk, (0, 0), sg) * SUPPORT_GAIN, 0, 1)
    if model is None:
        model = prior
    if model is None:
        return np.zeros((h, w), np.float32), None, parsed, np.zeros((h, w), np.float32)
    col = SK.skin_probability(rgb_u8, model)
    if not parsed:
        col = _ss(COLOUR_ONLY - 0.25, COLOUR_ONLY, col)
    allow = own * not_other
    if garment is not None:
        allow = allow * (1.0 - np.clip(garment, 0, 1))
    if others is not None and float(others.max()) > 0.05:
        sg = max(1.0, OTHERS_NEAR * h)
        allow = allow * (1.0 - np.clip(cv2.GaussianBlur(others, (0, 0), sg) * 2.5, 0, 1))
    c = col * allow * support
    return _ss(CONF[0], CONF[1], c), model, parsed, allow.astype(np.float32)


# ------------------------------------------------------------------------------------------------- one frame's mask
def ownership(tracker, t: int, p: int, size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """(the person's own pixels, everybody else's) on a picture of size (w, h), float32 0..1: the matte where
    the pixel's owner is this person / another tracked person. A matte pixel nobody owns is neither: it stays
    as it is."""
    a = tracker.matte.get(t)
    lab = tracker.labels(t)
    if a is None or lab is None:
        return None, None
    af = np.asarray(a).astype(np.float32) / 255.0
    own = cv2.resize(af * (lab == p), size, interpolation=cv2.INTER_LINEAR)
    oth = (lab != p) & (lab >= 0)
    others = cv2.resize(af * oth, size, interpolation=cv2.INTER_LINEAR) if oth.any() else None
    return own, others


def garments(tracker, t: int, p: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Clothes, shoes, hats and bags the analysis found on this frame (float32 0..1 at size (w, h)), or None when
    it made no clothing masks or they are of another matte (stale: not used until they are made again)."""
    try:
        from ..track import clothing as CL
        if not tracker.cloth.exists() or CL.stale_at(tracker, t):
            return None
        cm = tracker.cloth.get(t)
        if cm is None:
            return None
        g = np.isin(np.asarray(cm).astype(np.int16) - 1, list(CL.ALL_CLOTHES)).astype(np.uint8)
        if not g.any():
            return None
        g = cv2.dilate(g, np.ones((3, 3), np.uint8))
        return cv2.GaussianBlur(cv2.resize(g.astype(np.float32), size, interpolation=cv2.INTER_LINEAR), (0, 0), 1.0)
    except Exception:
        return None


_CLOCK = threading.Lock()


def _parse_cached(tracker, p: int, t: int, rgb_u8: np.ndarray, own: np.ndarray) -> Optional[dict]:
    """parse_person, kept per person and frame for this analysis of the cut."""
    key = (int(p), int(t))
    sig = (id(tracker.store), getattr(getattr(tracker, "cut", None), "analysis_version", 0), rgb_u8.shape[:2])
    with _CLOCK:
        lru = tracker.__dict__.get("_bparse")
        if lru is None or tracker.__dict__.get("_bparse_sig") != sig:
            lru = tracker.__dict__["_bparse"] = OrderedDict()
            tracker.__dict__["_bparse_sig"] = sig
        if key in lru:
            lru.move_to_end(key)
            return lru[key]
    pr = parse_person(rgb_u8, own)
    with _CLOCK:
        lru[key] = pr
        while len(lru) > PARSE_LRU:
            lru.popitem(last=False)
    return pr


def body_mask(tracker, p: int, t: int, rgb_u8: np.ndarray, regions: set, prior: Optional[tuple] = None,
              face_lm: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], BodyInfo]:
    """Where the body skin effects go for person p (store index) on frame t: float32 0..1 on the analysis picture
    rgb_u8, and what is known of the body. regions: body regions chosen (project.body_scope_set)."""
    st = tracker.store
    px = tracker.proxy
    info = body_info(st, p, t)
    h, w = rgb_u8.shape[:2]
    own, others = ownership(tracker, t, p, (w, h))
    if own is None or float(own.max()) < 0.3 or info.ref < REF_MIN:
        return None, info
    hands = st.hand[p, t] if getattr(st, "has_hands", False) and np.isfinite(st.hand[p, t][:, :2]).any() else None
    maps = region_maps(info, (h, w), px.sx, px.sy, hands)
    area = chosen(maps, regions)
    if float(area.max()) <= 0.01:
        return None, info
    ys = np.nonzero(own.max(1) > 0.3)[0]
    xs = np.nonzero(own.max(0) > 0.3)[0]
    info.box = (float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)) if len(xs) and len(ys) else None
    parse = _parse_cached(tracker, p, t, rgb_u8, own)
    size = float(np.sqrt(max(1.0, float((own > 0.3).sum()))))
    conf, model, parsed, allow = skin_conf(rgb_u8, own, others, parse, garments(tracker, t, p, (w, h)), prior, size)
    info.model, info.parsed = model, parsed
    info.allow = allow * np.clip(area, 0, 1)
    m = conf * area
    if float(m.max()) <= 0.01:
        return None, info
    sig = max(0.6, min(0.0025 * h, 0.008 * info.ref * px.sx))
    return np.clip(cv2.GaussianBlur(m, (0, 0), sig), 0, 1).astype(np.float32), info


# ------------------------------------------------------------------------------------------------- at the picture's own size
REFINE = (0.25, 0.60)       # the colour's say at the worked picture's own pixels: from where a pixel counts .. in full
REFINE_SOFT = 0.8           # px of that picture: how softly the edge is laid


def plane_colour(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, model: tuple, box) -> np.ndarray:
    """skin.skin_probability on a piece (y0, y1, x0, x1) of a picture given as Y 0..1 and Cb, Cr centred on 0
    (BT.709, any size each): the same measure the mask was made with, taken at this picture's own pixels."""
    y0, y1, x0, x1 = box
    hY, wY = Y.shape
    y = Y[y0:y1, x0:x1].astype(np.float32)

    def at(c):
        if c.shape != (hY, wY):
            fy, fx = c.shape[0] / hY, c.shape[1] / wY
            c0, c1 = int(np.floor(y0 * fy)), min(c.shape[0], int(np.ceil(y1 * fy)) + 1)
            d0, d1 = int(np.floor(x0 * fx)), min(c.shape[1], int(np.ceil(x1 * fx)) + 1)
            A = np.array([[1.0 / fx, 0, d0 / fx - x0 + 0.5 / fx - 0.5], [0, 1.0 / fy, c0 / fy - y0 + 0.5 / fy - 0.5]], np.float32)
            return cv2.warpAffine(c[c0:c1, d0:d1].astype(np.float32), A, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        return c[y0:y1, x0:x1].astype(np.float32)
    cb, cr = at(Cb), at(Cr)
    r = np.clip(y + 1.5748 * cr, 0, 1) * 255.0
    b = np.clip(y + 1.8556 * cb, 0, 1) * 255.0
    g = np.clip((y - 0.2126 * (y + 1.5748 * cr) - 0.0722 * (y + 1.8556 * cb)) / 0.7152, 0, 1) * 255.0
    y6 = 0.299 * r + 0.587 * g + 0.114 * b
    sig = max(0.8, 0.0012 * hY)
    c_r = cv2.GaussianBlur((r - y6) * 0.713 + 128.0, (0, 0), sig)
    c_b = cv2.GaussianBlur((b - y6) * 0.564 + 128.0, (0, 0), sig)
    mean, icov = model[0], model[1]
    yref = model[2] if len(model) > 2 else None
    d0_, d1_ = c_r - mean[0], c_b - mean[1]
    m2 = icov[0, 0] * d0_ * d0_ + 2 * icov[0, 1] * d0_ * d1_ + icov[1, 1] * d1_ * d1_
    prob = np.exp(-0.5 * np.maximum(np.sqrt(np.maximum(m2, 0)) - 1.6, 0) ** 2)
    if yref is not None and yref > 8:
        prob *= np.clip((y6 - 0.30 * yref) / (0.25 * yref), 0, 1)
    else:
        prob *= np.clip((y6 - 25) / 25, 0, 1)
    return prob.astype(np.float32)


def pack_mask(m: np.ndarray) -> Optional[dict]:
    """A mask of the frame's size as the caches keep it: the piece that is not empty, at half precision. A body's
    area is a part of the frame; kept whole for every person and frame, the masks of a 4K cut with several people
    filled the caches with zeros. {'box': (x0, y0, x1, y1), 'm': float16 piece, 'shape': (h, w)}"""
    ys = np.nonzero(m.max(1) > 0.004)[0]
    xs = np.nonzero(m.max(0) > 0.004)[0]
    if not len(ys) or not len(xs):
        return None
    x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    return dict(box=(x0, y0, x1, y1), m=np.ascontiguousarray(m[y0:y1, x0:x1]).astype(np.float16), shape=tuple(m.shape))


def pack_allow(a: Optional[np.ndarray]) -> Optional[dict]:
    """BodyInfo.allow as the caches keep it: yes or no for each pixel of its piece that is not empty, a bit each (a
    person's piece of a 1920-wide picture: some 60 KB)."""
    if a is None:
        return None
    yes = a > 0.5
    ys = np.nonzero(yes.any(1))[0]
    xs = np.nonzero(yes.any(0))[0]
    if not len(ys) or not len(xs):
        return None
    x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    return dict(box=(x0, y0, x1, y1), m=np.packbits(yes[y0:y1, x0:x1], axis=1), shape=tuple(a.shape))


def unpack_allow(pa: Optional[dict]) -> Optional[np.ndarray]:
    """pack_allow's map at the frame's size (float32, 0 or 1), or None."""
    if pa is None:
        return None
    x0, y0, x1, y1 = pa["box"]
    out = np.zeros(pa["shape"], np.float32)
    out[y0:y1, x0:x1] = np.unpackbits(pa["m"], axis=1, count=x1 - x0)
    return out


def unpack_mask(pm: Optional[dict], shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """pack_mask's mask again at the frame's size (float32), or laid on a picture of another `shape` (h, w)."""
    if pm is None:
        return None
    h, w = pm["shape"]
    x0, y0, x1, y1 = pm["box"]
    if shape is None or tuple(shape) == (h, w):
        out = np.zeros((h, w), np.float32)
        out[y0:y1, x0:x1] = pm["m"]
        return out
    H, W = shape
    fx, fy = W / float(w), H / float(h)
    X0, Y0 = int(np.floor(x0 * fx)), int(np.floor(y0 * fy))
    X1, Y1 = min(W, int(np.ceil(x1 * fx)) + 1), min(H, int(np.ceil(y1 * fy)) + 1)
    out = np.zeros((H, W), np.float32)
    if X1 - X0 < 1 or Y1 - Y0 < 1:
        return out
    # the frame's pixel centres laid on the other picture's (as cv2.resize of the whole mask would)
    A = np.array([[fx, 0, x0 * fx - X0 + 0.5 * fx - 0.5], [0, fy, y0 * fy - Y0 + 0.5 * fy - 0.5]], np.float32)
    out[Y0:Y1, X0:X1] = cv2.warpAffine(pm["m"].astype(np.float32), A, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def refine(mask, model: Optional[tuple], Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """The body mask on the picture being worked on (float32, Y's shape). mask: of the analysis picture, whole or
    as pack_mask keeps it. Made larger for a 4K original its edge would be a blur a few pixels wide lying over
    hair, cloth and background. The colour that made the mask is asked again at this picture's own pixels and the
    edge laid there. The same rule on the preview (where the two pictures are one size) and on an export."""
    hY, wY = Y.shape
    if isinstance(mask, dict):
        m = unpack_mask(mask, (hY, wY))
    else:
        m = mask if mask.dtype == np.float32 else mask.astype(np.float32)
        if m.shape != (hY, wY):
            m = cv2.resize(m, (wY, hY), interpolation=cv2.INTER_LINEAR)
    if model is None:
        return m
    ys = np.nonzero(m.max(1) > 0.004)[0]
    xs = np.nonzero(m.max(0) > 0.004)[0]
    if not len(ys) or not len(xs):
        return m
    pad = 4
    y0, y1 = max(0, int(ys[0]) - pad), min(hY, int(ys[-1]) + 1 + pad)
    x0, x1 = max(0, int(xs[0]) - pad), min(wY, int(xs[-1]) + 1 + pad)
    col = plane_colour(Y, Cb, Cr, model, (y0, y1, x0, x1))
    out = m.copy()
    out[y0:y1, x0:x1] = cv2.GaussianBlur(m[y0:y1, x0:x1] * _ss(REFINE[0], REFINE[1], col), (0, 0), REFINE_SOFT)
    return out


# ------------------------------------------------------------------------------------------------- motion between frames
FLOW_LONG = 512             # the person's piece is this long at most when its motion is measured
FLOW_PAD = 0.10
FLOW_SAME = 0.08            # brightness difference (0..1) from which a neighbour's picture is not this frame's (something
                            # in front, another person, a jump): it has no say there


FLOW_TENSOR = 1.2           # px of the reduced piece: the window a pixel's structure is taken over
FLOW_SCALES = ((2.5, 1.0), (8.0, 0.3), (25.0, 0.1), (80.0, 0.03))      # (width px, weight): how far structure has a say
FLOW_FIRM = (2.0, 8.0)      # times the piece's median structure: from where a pixel's structure counts .. in full
                            # (below it is the picture's grain and skin's faint texture: motion measured on those
                            # slides, and many such pixels together would outvote the few that are firm)
FLOW_REG = 1e-7
FLOW_KEEP = 256             # the motion is kept on a grid this long at most (it is smooth by its making)
FLOW_BACK = (0.5, 1.5)      # px of the measuring grid: going to the other frame and back a point may miss itself by
                            # the first without a word; by the second more its neighbour has hardly a say there


def _wide(img: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur; a wide one is made on a smaller picture (its result has nothing fine in it)."""
    if sigma <= 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    q = sigma / 2.5
    ws, hs = max(2, int(round(w / q))), max(2, int(round(h / q)))
    sm = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma * ws / float(w), sigmaY=sigma * hs / float(h))
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def motion(gray_t: np.ndarray, gray_s: np.ndarray, box, weight: Optional[np.ndarray] = None) -> Optional[dict]:
    """One way only (see motion_pair): where each point of frame t's piece is on frame s."""
    return _keep(_flow(gray_t, gray_s, box, weight))


def motion_pair(gray_t: np.ndarray, gray_s: np.ndarray, box_t, box_s=None, weight_t: Optional[np.ndarray] = None,
                weight_s: Optional[np.ndarray] = None) -> Tuple[Optional[dict], Optional[dict]]:
    """The motion of frame t onto frame s and of s onto t (each as _flow gives it, on its own frame's piece), each
    checked against the other: where something lies over the skin on s, the covered skin of t has no place on s,
    and what the measuring finds for it is the skin beside the cover (skin is like skin: the pictures agree). Going
    on from there with the motion of s onto t does not come back to where it started; there s has no say on t."""
    a = _flow(gray_t, gray_s, box_t, weight_t)
    b = _flow(gray_s, gray_t, box_s if box_s is not None else box_t, weight_s)
    if a is not None and b is not None:
        sa, sb = _there_and_back(a, b), _there_and_back(b, a)
        a["same"], b["same"] = sa, sb
    return _keep(a), _keep(b)


def _there_and_back(a: dict, b: dict) -> np.ndarray:
    """a's 'same', taken down where b does not bring a's points back to themselves (and where they leave b's piece)."""
    x0, y0, x1, y1 = a["box"]
    h, w = a["same"].shape
    X0, Y0, X1, Y1 = b["box"]
    H, W = b["same"].shape
    u = ((a["map"][..., 0] - X0 + 0.5) * (W / float(X1 - X0)) - 0.5).astype(np.float32)
    v = ((a["map"][..., 1] - Y0 + 0.5) * (H / float(Y1 - Y0)) - 0.5).astype(np.float32)
    back = cv2.remap(b["map"], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ax = x0 + (gx + 0.5) * ((x1 - x0) / float(w)) - 0.5
    ay = y0 + (gy + 0.5) * ((y1 - y0) / float(h)) - 0.5
    err = np.hypot(back[..., 0] - ax, back[..., 1] - ay) * (w / float(x1 - x0))
    keep = np.exp(-np.square(np.maximum(err - FLOW_BACK[0], 0.0) / FLOW_BACK[1])) * inside
    return (a["same"] * keep).astype(np.float32)


def _keep(mo: Optional[dict]) -> Optional[dict]:
    """A motion as it is kept: on a grid of FLOW_KEEP at most."""
    if mo is None:
        return None
    h, w = mo["same"].shape
    q = FLOW_KEEP / float(max(h, w))
    if q >= 1.0:
        return mo
    ws, hs = max(8, int(round(w * q))), max(8, int(round(h * q)))
    return dict(box=mo["box"], map=cv2.resize(mo["map"], (ws, hs), interpolation=cv2.INTER_AREA),
                same=cv2.resize(mo["same"], (ws, hs), interpolation=cv2.INTER_AREA))


def _flow(gray_t: np.ndarray, gray_s: np.ndarray, box, weight: Optional[np.ndarray] = None) -> Optional[dict]:
    """Where each point of frame t's piece (box x0, y0, x1, y1 in analysis pixels) is on frame s, measured on the
    picture itself (dense optical flow), and how alike the two pictures are once s is carried onto t. A body has no
    landmarks to follow as a face has: its skin moves with the picture. {'box', 'map' (h, w, 2) float32 in analysis
    pixels of s, 'same' (h, w) 0..1}, at the size the motion was measured at.

    Bare skin has little to hold on to: along a limb the measured motion slides (the same skin lies everywhere along
    it), and under something that covers the limb on s it lands on skin beside the cover. So a pixel has a say only
    in the direction its own structure fixes (across an edge; both ways at a corner, a hem, a mark), by how strong
    that structure is over the picture's grain, and the motion of the plain skin between comes from the structure
    round it, near before far. weight (analysis pixels, 0..1): whose structure counts (the person's own pixels: what
    is behind them or another person moves another way)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = gray_t.shape
    px_, py_ = int(FLOW_PAD * (x1 - x0)) + 4, int(FLOW_PAD * (y1 - y0)) + 4
    x0, y0, x1, y1 = max(0, x0 - px_), max(0, y0 - py_), min(W, x1 + px_), min(H, y1 + py_)
    if x1 - x0 < 16 or y1 - y0 < 16 or gray_s.shape != gray_t.shape:
        return None
    q = min(1.0, FLOW_LONG / float(max(x1 - x0, y1 - y0)))
    w, h = max(16, int(round((x1 - x0) * q))), max(16, int(round((y1 - y0) * q)))
    a = cv2.resize(gray_t[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray_s[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
    wq = None
    if weight is not None and weight.shape == gray_t.shape:
        wq = cv2.resize(weight[y0:y1, x0:x1].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        if float(wq.max()) < 0.3:
            wq = None
    # the light may change from one frame to the next (a flash): s is brought to this frame's level
    sel = wq > 0.5 if wq is not None else None
    ma = float(np.median(a[sel])) if sel is not None and int(sel.sum()) > 50 else float(np.median(a))
    mb = float(np.median(b[sel])) if sel is not None and int(sel.sum()) > 50 else float(np.median(b))
    gain = float(np.clip(ma / max(mb, 1.0), 0.5, 2.0))
    if abs(gain - 1.0) > 0.02:
        b = np.clip(b.astype(np.float32) * gain + 0.5, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(a, b, None)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0
    # each pixel's structure (which way it is fixed, how firmly) ...
    dx = cv2.Sobel(af, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    dy = cv2.Sobel(af, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    T = cv2.GaussianBlur(np.dstack([dx * dx, dx * dy, dy * dy]), (0, 0), FLOW_TENSOR)
    tr = T[..., 0] + T[..., 2]
    med = float(np.median(tr[sel]) if sel is not None and int(sel.sum()) > 50 else np.median(tr)) + 1e-9
    # ... counts where the measured motion brings the same picture, and whose pixel it is
    bw0 = cv2.remap(bf, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ok = np.exp(-np.square(cv2.GaussianBlur(np.abs(bw0 - af), (0, 0), 1.5) / FLOW_SAME))
    k = ok * np.square(_ss(FLOW_FIRM[0] * med, FLOW_FIRM[1] * med, tr)) / (tr + med)
    if wq is not None:
        k = k * wq
    Tn = (T * k[..., None]).astype(np.float32)
    Bv = np.dstack([Tn[..., 0] * f[..., 0] + Tn[..., 1] * f[..., 1], Tn[..., 1] * f[..., 0] + Tn[..., 2] * f[..., 1]])
    A = np.zeros((h, w, 3), np.float64)
    B = np.zeros((h, w, 2), np.float64)
    for sg, ak in FLOW_SCALES:
        A += ak * _wide(Tn, sg)
        B += ak * _wide(Bv, sg)
    firm = (Tn[..., 0] + Tn[..., 2]) > 0.5
    f0 = np.median(f[firm], axis=0) if int(firm.sum()) > 20 else np.zeros(2, np.float32)
    a11, a12, a22 = A[..., 0] + FLOW_REG, A[..., 1], A[..., 2] + FLOW_REG
    b1, b2 = B[..., 0] + FLOW_REG * float(f0[0]), B[..., 1] + FLOW_REG * float(f0[1])
    det = np.maximum(a11 * a22 - a12 * a12, 1e-30)
    fx = ((a22 * b1 - a12 * b2) / det).astype(np.float32)
    fy = ((a11 * b2 - a12 * b1) / det).astype(np.float32)
    mx, my = gx + fx, gy + fy
    bw = cv2.remap(bf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    dif = cv2.GaussianBlur(np.abs(bw - af), (0, 0), 1.5)
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    same = np.exp(-np.square(dif / FLOW_SAME)) * inside
    sx_, sy_ = (x1 - x0) / float(w), (y1 - y0) / float(h)
    mp = np.stack([x0 + (mx + 0.5) * sx_ - 0.5, y0 + (my + 0.5) * sy_ - 0.5], -1).astype(np.float32)
    return dict(box=(x0, y0, x1, y1), map=mp, same=same.astype(np.float32))


def carry(mo: dict, m_s: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """A map of frame s (analysis pixels, any float type) carried onto frame t along `mo` (motion): (the map on a
    picture of `shape`, how much it can be trusted there 0..1). Outside the measured piece both are 0."""
    x0, y0, x1, y1 = mo["box"]
    h, w = shape
    bw, bh = x1 - x0, y1 - y0
    mp = mo["map"]
    if mp.shape[:2] != (bh, bw):
        mp = cv2.resize(mp, (bw, bh), interpolation=cv2.INTER_LINEAR)
        same = cv2.resize(mo["same"], (bw, bh), interpolation=cv2.INTER_LINEAR)
    else:
        same = mo["same"]
    src = m_s if m_s.dtype == np.float32 else m_s.astype(np.float32)
    piece = cv2.remap(src, mp[..., 0], mp[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.zeros((h, w), np.float32)
    conf = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = piece
    conf[y0:y1, x0:x1] = same
    return out, conf


# ------------------------------------------------------------------------------------------------- hand marks
def marks_key(marks) -> tuple:
    """What of the hand marks the masks and the detection depend on (the engine's cache signature)."""
    return tuple((m.get("id"), m.get("kind"), int(m.get("t", 0)), round(float(m.get("r", 0)), 2), m.get("part", ""),
                  tuple(tuple(round(float(v), 2) for v in q) for q in m.get("pts") or ())) for m in (marks or ()))


MARK_REACH = 1.8            # a mark is fixed to a body part when it lies within this many half widths of its axis


def _local(part: Part, pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Points (n, 2) in a part's own measure: along its axis and across it, in lengths of the axis."""
    ab = part.b - part.a
    L = max(1e-6, float(np.hypot(*ab)))
    u = ab / L
    n = np.array([-u[1], u[0]])
    q = np.asarray(pts, np.float64) - part.a
    return np.stack([q @ u / L, q @ n / L], 1), L


def anchor_mark(st, p: int, t: int, mark: dict) -> dict:
    """A mark drawn on frame t (pts and r in source px) fixed to the body part it lies on: 'part', 'uv' and 'ru' are
    added, so it follows that part through the cut. Where no part is near it stays where it was drawn."""
    out = dict(mark)
    out.pop("uv", None)
    out.pop("ru", None)
    out["part"] = ""
    pts = np.asarray(mark.get("pts") or [], np.float64).reshape(-1, 2)
    if not len(pts) or st is None:
        return out
    info = body_info(st, p, t)
    c = pts.mean(0)
    best, bd = None, MARK_REACH
    for part in info.parts.values():
        if np.allclose(part.a, part.b):
            continue
        ab = part.b - part.a
        tt = float(np.clip(((c - part.a) @ ab) / max(1e-9, float(ab @ ab)), 0, 1))
        d = float(np.hypot(*(c - (part.a + tt * ab)))) / max(part.hw, 1.0)
        if part.name == "trunk":
            d += 0.6                            # a limb lying over the trunk takes the mark first
        if d < bd:
            best, bd = part, d
    if best is None:
        return out
    uv, L = _local(best, pts)
    out["part"] = best.name
    out["uv"] = [[float(a), float(b)] for a, b in uv]
    out["ru"] = float(mark.get("r", 6.0)) / L
    return out


def marks_at(st, p: int, t: int, marks) -> list:
    """The person's hand marks as they lie on frame t: [(kind, points (n, 2) source px, radius source px)]. A mark
    fixed to a part the joints do not place on this frame is not there."""
    out = []
    if not marks:
        return out
    info = None
    for m in marks:
        if m.get("part") and m.get("uv"):
            if info is None:
                info = body_info(st, p, t)
            part = info.parts.get(m["part"])
            if part is None:
                continue
            ab = part.b - part.a
            L = max(1e-6, float(np.hypot(*ab)))
            u = ab / L
            n = np.array([-u[1], u[0]])
            uv = np.asarray(m["uv"], np.float64)
            pts = part.a + (uv[:, :1] * u + uv[:, 1:2] * n) * L
            out.append((m["kind"], pts, float(m.get("ru", 0.0)) * L))
        else:
            out.append((m["kind"], np.asarray(m["pts"], np.float64).reshape(-1, 2), float(m.get("r", 6.0))))
    return out
