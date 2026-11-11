"""Face and head shape changes.

Everything here is driven by the tracked 68 face landmarks (and, for the head, the person's matte):

* A dense face contour (jaw sampled at 49 points + a forehead arc) splits the picture into "face" and
  "around the face". Inside, an interior map moves the pixels (scale, stretch); around the face, the
  contour's own movement fades out over a short band, so hair close to the face takes up the change and
  the background further away stays where it is.
* Edits that only belong to the outline (jaw line, jaw correction, sculpted contour points) move the
  contour and fade inwards and outwards from it, following its curve. That is what keeps a slimmer jaw
  round instead of cutting it straight.
* Nose, mouth, nose bridge and sculpted inner points are small smooth local maps.
* The head (with hair) uses the same idea one level up: an outline found in the matte around the face.

All maps are built as forward displacements V (where a source pixel goes) and inverted into the
backward map the renderer needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

JAW_SUB = 3                       # points per jaw landmark interval
N_JAW = 16 * JAW_SUB + 1          # 49
N_FORE = 11
# dense point layout (also the sculpt editing points)
D_JAW = slice(0, N_JAW)                          # 0..48
D_BROW = slice(N_JAW, N_JAW + 10)                # lm 17..26
D_NOSE = slice(N_JAW + 10, N_JAW + 19)           # lm 27..35
D_EYE = slice(N_JAW + 19, N_JAW + 31)            # lm 36..47
D_MOUTH = slice(N_JAW + 31, N_JAW + 51)          # lm 48..67
D_FORE = slice(N_JAW + 51, N_JAW + 51 + N_FORE)  # forehead arc
N_DENSE = N_JAW + 51 + N_FORE                    # 111
DENSE_GROUPS = [("턱·윤곽", D_JAW), ("눈썹", D_BROW), ("코", D_NOSE), ("눈", D_EYE), ("입", D_MOUTH), ("이마", D_FORE)]


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _smoother(e0, e1, x):
    """C2 fade (no visible edge where it starts or ends)."""
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-9), 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def catmull(pts: np.ndarray, n: int) -> np.ndarray:
    """Catmull-Rom through pts, n samples per interval (end point included)."""
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    s = np.linspace(0, 1, n, endpoint=False)[:, None]
    s2, s3 = s * s, s * s * s
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(pts[-1:])
    return np.vstack(out)


# --------------------------------------------------------------------------- geometry

@dataclass
class FaceGeom:
    lm: np.ndarray        # (68,2) source px
    jaw: np.ndarray       # (49,2)
    fore: np.ndarray      # (11,2) forehead arc, from the lm16 side to the lm0 side
    O: np.ndarray         # eye midpoint
    ex: np.ndarray        # unit, image left eye -> image right eye
    ey: np.ndarray        # unit, towards the chin
    iod: float            # distance between the eye centres
    Wf: float             # jaw width (lm0-lm16)
    chin: np.ndarray

    @staticmethod
    def from_landmarks(lm: Optional[np.ndarray]) -> Optional["FaceGeom"]:
        if lm is None:
            return None
        pts = np.asarray(lm[:, :2], np.float64)
        if not np.isfinite(pts).all():
            return None
        e1, e2 = pts[36:42].mean(0), pts[42:48].mean(0)
        iod = float(np.hypot(*(e2 - e1)))
        Wf = float(np.hypot(*(pts[16] - pts[0])))
        if iod < 4 or Wf < 8:
            return None
        ex = (e2 - e1) / iod
        ey = np.array([-ex[1], ex[0]])
        O = (e1 + e2) / 2
        if (pts[8] - O) @ ey < 0:
            ey = -ey
        jaw = catmull(pts[0:17], JAW_SUB)
        # forehead: brows lifted towards the hairline, spanning the jaw top points
        up = -ey
        brows = np.vstack([pts[0], pts[17:27], pts[16]])
        lift = np.r_[0.10, np.full(10, 0.36), 0.10] * Wf
        arc = brows + up[None] * lift[:, None]
        arc = catmull(arc, 2)
        idx = np.linspace(0, len(arc) - 1, N_FORE).round().astype(int)
        fore = arc[idx][::-1]
        return FaceGeom(pts, jaw, fore, O, ex, ey, iod, Wf, pts[8].copy())

    def v(self, P: np.ndarray) -> np.ndarray:
        return (P[..., 0] - self.O[0]) * self.ey[0] + (P[..., 1] - self.O[1]) * self.ey[1]

    def u(self, P: np.ndarray) -> np.ndarray:
        return (P[..., 0] - self.O[0]) * self.ex[0] + (P[..., 1] - self.O[1]) * self.ex[1]

    def contour(self) -> np.ndarray:
        """Closed face outline: jaw (lm0 -> lm16) then the forehead arc back."""
        return np.vstack([self.jaw, self.fore[1:-1]])

    def dense(self) -> np.ndarray:
        L = self.lm
        return np.vstack([self.jaw, L[17:27], L[27:36], L[36:48], L[48:68], self.fore])

    # face units: x along the eye line, y towards the chin, 1 = eye distance
    def to_units(self, P: np.ndarray) -> np.ndarray:
        d = np.asarray(P, np.float64) - self.O
        return np.stack([d @ self.ex, d @ self.ey], -1) / self.iod

    def from_units_vec(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, np.float64)
        return (uv[..., :1] * self.ex + uv[..., 1:2] * self.ey) * self.iod


@dataclass
class HeadGeom:
    outline: np.ndarray   # (M,2) source px, from the lm0 side over the top to the lm16 side
    center: np.ndarray
    R: float
    top_v: float          # v (face frame) of the head top

    def polygon(self, face: FaceGeom) -> np.ndarray:
        # jaw from lm16 back to lm0, then the outline over the top
        return np.vstack([face.jaw[::-1], self.outline])


def head_outline(alpha: np.ndarray, lab: Optional[np.ndarray], pidx: int, sx: float, sy: float,
                 face: FaceGeom, n_rays: int = 27) -> Optional[np.ndarray]:
    """Radii of the head silhouette in face units for rays over the top of the head (from the lm0 side
    to the lm16 side), measured in the matte. None if the matte has no head there."""
    h, w = alpha.shape
    centre = face.O - face.ey * 0.15 * face.Wf
    c = np.array([centre[0] * sx, centre[1] * sy])
    up = -face.ey
    ang = np.linspace(-np.pi * 0.62, np.pi * 0.62, n_rays)
    side = face.ex
    rmax = 1.6 * face.Wf * sx
    steps = np.arange(1.0, rmax, 1.0)
    out = np.full(n_rays, np.nan)
    for i, a in enumerate(ang):
        d = up * np.cos(a) + side * np.sin(a)        # a<0: towards the lm0 side (image left of the face)
        d = np.array([d[0] * sx, d[1] * sy])
        d /= max(np.hypot(*d), 1e-9)
        pts = c[None] + steps[:, None] * d[None]
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        oob = (xi < 0) | (xi >= w) | (yi < 0) | (yi >= h)
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)
        inside = alpha[yi, xi] >= 110
        if lab is not None:
            inside &= (lab[yi, xi] == pidx) | (lab[yi, xi] == -1)
        inside &= ~oob
        if not inside[: max(3, int(0.15 * face.Wf * sx))].any():
            continue
        k = int(np.argmax(~inside)) if (~inside).any() else len(steps) - 1
        out[i] = steps[k] / sx / face.iod
    if np.isfinite(out).sum() < n_rays * 0.6:
        return None
    return out


def head_from_radii(face: FaceGeom, radii: np.ndarray) -> Optional[HeadGeom]:
    if radii is None or not np.isfinite(radii).any():
        return None
    n = len(radii)
    ang = np.linspace(-np.pi * 0.62, np.pi * 0.62, n)
    r = np.array(radii, np.float64)
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return None
    r = np.interp(np.arange(n), np.nonzero(ok)[0], r[ok])
    centre = face.O - face.ey * 0.15 * face.Wf
    up = -face.ey
    dirs = up[None] * np.cos(ang)[:, None] + face.ex[None] * np.sin(ang)[:, None]
    pts = centre[None] + dirs * (r * face.iod)[:, None]
    # never inside the face outline: the head is at least the forehead arc
    fore_r = np.hypot(*(face.fore - centre).T).max()
    pts = centre[None] + dirs * np.maximum(r * face.iod, 0.8 * fore_r)[:, None]
    # join the jaw ends smoothly
    pts = np.vstack([face.jaw[0], pts, face.jaw[-1]])
    R = float(np.median(r) * face.iod)
    top_v = float(face.v(pts).min())
    return HeadGeom(pts[1:-1], centre, R, top_v)


# --------------------------------------------------------------------------- grid helpers

class _Grid:
    """Regular grid window in source px (X, Y from compute_field) with raster helpers."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X, self.Y = X, Y
        self.x0, self.y0 = float(X[0, 0]), float(Y[0, 0])
        self.hx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
        self.hy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
        self.shape = X.shape

    def to_g(self, P: np.ndarray) -> np.ndarray:
        return np.stack([(P[..., 0] - self.x0) / self.hx, (P[..., 1] - self.y0) / self.hy], -1)

    def inside(self, poly: np.ndarray, sup: int = 2) -> np.ndarray:
        """Soft (anti-aliased) inside mask of a polygon."""
        h, w = self.shape
        m = np.zeros((h * sup, w * sup), np.uint8)
        g = (self.to_g(poly) + 0.5) * sup - 0.5
        cv2.fillPoly(m, [np.round(g * 4).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=2)
        if sup > 1:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
        return m.astype(np.float32) / 255.0

    def nearest_on(self, poly: np.ndarray, closed: bool = True):
        """Distance (source px) to a polyline and, per grid point, the nearest polyline parameter
        (fractional vertex index) and point."""
        h, w = self.shape
        P = np.vstack([poly, poly[:1]]) if closed else poly
        seg = np.diff(P, axis=0)
        L = np.hypot(seg[:, 0] / self.hx, seg[:, 1] / self.hy)
        samples, params = [], []
        for i in range(len(seg)):
            n = max(2, int(np.ceil(L[i] * 2)))
            s = np.linspace(0, 1, n, endpoint=False)
            samples.append(P[i] + s[:, None] * seg[i])
            params.append(i + s)
        samples.append(P[-1:])
        params.append(np.array([len(seg)], float))
        S_ = np.vstack(samples)
        Q = np.concatenate(params)
        G = np.round(self.to_g(S_)).astype(int)
        keep = (G[:, 0] >= 0) & (G[:, 0] < w) & (G[:, 1] >= 0) & (G[:, 1] < h)
        if keep.sum() < 2:
            return None
        G, S_, Q = G[keep], S_[keep], Q[keep]
        img = np.ones((h, w), np.uint8)
        img[G[:, 1], G[:, 0]] = 0
        qimg = np.zeros((h, w), np.float64)
        ximg = np.zeros((h, w, 2), np.float64)
        qimg[G[:, 1], G[:, 0]] = Q
        ximg[G[:, 1], G[:, 0]] = S_
        dist, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        zeros = np.argwhere(img == 0)          # same (raster) order as the labels
        zz = zeros[np.clip(lab - 1, 0, len(zeros) - 1)]
        q = qimg[zz[..., 0], zz[..., 1]]
        near = ximg[zz[..., 0], zz[..., 1]]
        d = np.hypot(self.X - near[..., 0], self.Y - near[..., 1])
        return d, q, near


def _densify(poly: np.ndarray, vals: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a closed polyline (and per-vertex values) to ~`step` px spacing."""
    P = np.vstack([poly, poly[:1]])
    Vv = np.vstack([vals, vals[:1]])
    out_p, out_v = [], []
    for i in range(len(P) - 1):
        n = max(1, int(np.ceil(np.hypot(*(P[i + 1] - P[i])) / step)))
        s = np.linspace(0, 1, n, endpoint=False)[:, None]
        out_p.append(P[i] + s * (P[i + 1] - P[i]))
        out_v.append(Vv[i] + s * (Vv[i + 1] - Vv[i]))
    return np.vstack(out_p), np.vstack(out_v)


def _smooth_extend(X, Y, pts: np.ndarray, vals: np.ndarray, sig: float) -> np.ndarray:
    """Kernel-weighted average of contour values at every grid point (smooth everywhere, equal to the
    nearby contour value close to the contour, and to the nearest part of the contour far from it).

    The weights are taken relative to the nearest contour point: plain Gaussian weights underflow a few
    kernel widths away, which dropped the field to zero there; the fade-out of a big head or hair change
    then ended in a hard edge (the crown looked cut off, the hair as if it hit a wall)."""
    Xf = X.astype(np.float32).ravel()
    Yf = Y.astype(np.float32).ravel()
    inv = np.float32(1.0 / (2 * sig * sig))
    P = pts.astype(np.float32)
    Vv = vals.astype(np.float32)
    B = 24
    d2min = np.full(Xf.size, np.inf, np.float32)
    for i in range(0, len(P), B):
        dx = Xf[:, None] - P[None, i:i + B, 0]
        dy = Yf[:, None] - P[None, i:i + B, 1]
        d2min = np.minimum(d2min, (dx * dx + dy * dy).min(1))
    num = np.zeros((Xf.size, vals.shape[1]), np.float32)
    den = np.zeros(Xf.size, np.float32)
    for i in range(0, len(P), B):
        dx = Xf[:, None] - P[None, i:i + B, 0]
        dy = Yf[:, None] - P[None, i:i + B, 1]
        g = np.exp(-(dx * dx + dy * dy - d2min[:, None]) * inv)
        num += g @ Vv[i:i + B]
        den += g.sum(1)
    return (num / np.maximum(den, 1e-12)[:, None]).reshape(X.shape + (vals.shape[1],)).astype(np.float64)


def _interp_param(values: np.ndarray, q: np.ndarray, closed: bool) -> np.ndarray:
    """values (M, C) at integer vertex params; q fractional params -> (…, C)."""
    M = len(values)
    if closed:
        V = np.vstack([values, values[:1]])
        q = np.mod(q, M)
    else:
        V = values
        q = np.clip(q, 0, M - 1)
    i0 = np.floor(q).astype(int)
    i1 = np.minimum(i0 + 1, len(V) - 1)
    f = (q - i0)[..., None]
    return V[i0] * (1 - f) + V[i1] * f


def invert(V: np.ndarray, grid: _Grid, iters: int = 4) -> np.ndarray:
    """Backward map D with D(p) = -V(p + D(p)) (V forward displacement sampled on the same grid)."""
    if not np.abs(V).max() > 1e-6:
        return np.zeros_like(V)
    Vf = V.astype(np.float32)
    gx = np.arange(V.shape[1], dtype=np.float32)[None, :].repeat(V.shape[0], 0)
    gy = np.arange(V.shape[0], dtype=np.float32)[:, None].repeat(V.shape[1], 1)
    D = -Vf.copy()
    for _ in range(iters):
        mx = gx + D[..., 0] / grid.hx
        my = gy + D[..., 1] / grid.hy
        D = -cv2.remap(Vf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return D.astype(np.float64)


# --------------------------------------------------------------------------- the face map

def _jaw_weight(n: int) -> np.ndarray:
    """0 at the ears, peak around the jaw angle / lower cheek, 0 at the chin (per jaw sample)."""
    q = np.linspace(0, 1, n)
    s = np.minimum(q, 1 - q) * 2            # 0 at the ears, 1 at the chin
    return np.sin(np.pi * np.clip(s, 0, 1)) ** 1.15 * (1 - 0.25 * s)


def jaw_targets(face: FaceGeom) -> np.ndarray:
    """Jaw line for 'jaw correction': the square corners and bumps shaved down to a smooth curve.

    Only points that stick out beyond the smoothed line move, and only inwards, so the outline never gets
    a dent and the chin and the ears stay put. Left-right evening out is added only for a face looking
    straight at the camera: on a turned face the two sides differ because of the turn, and mirroring them
    pulled one cheek in (the face looked crumpled)."""
    J = face.jaw
    n = len(J)
    uv = np.stack([face.u(J), face.v(J)], -1)
    mid_u = float(np.median(face.u(face.lm[27:31])))
    uv[:, 0] -= mid_u
    # how frontal: the nose line halfway between the two jaw sides
    span_l = float(-uv[: n // 2, 0].min()) if n else 0.0
    span_r = float(uv[n // 2 + 1:, 0].max()) if n else 0.0
    yaw = abs(span_l - span_r) / max(1e-6, span_l + span_r)
    frontal = float(np.clip(1.0 - (yaw - 0.04) / 0.08, 0.0, 1.0))
    ref = uv.copy()
    if frontal > 0:
        mir = uv[::-1] * np.array([-1, 1])
        ref = uv + 0.5 * frontal * (mir - uv)
    # strong smoothing along the contour, ends pinned
    k = np.exp(-0.5 * (np.arange(-7, 8) / 3.0) ** 2)
    k /= k.sum()
    pad = np.vstack([2 * ref[:1] - ref[7:0:-1], ref, 2 * ref[-1:] - ref[-2:-9:-1]])
    sm = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(2)], -1)
    # inward only: compare along the outward normal of the original outline
    tang = np.gradient(uv, axis=0)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], -1)
    nrm /= np.maximum(np.hypot(*nrm.T), 1e-9)[:, None]
    cen = np.array([0.0, float(uv[:, 1].min())])
    flip = ((uv - cen) * nrm).sum(1) < 0
    nrm[flip] *= -1
    out = ((uv - sm) * nrm).sum(1)                      # > 0: the original sticks out
    # the jaw angle itself is rounded off a little more (along the inward normal: no dent beside it)
    q = np.linspace(0, 1, n)
    corner = np.exp(-0.5 * ((np.minimum(q, 1 - q) * 2 - 0.42) / 0.16) ** 2)
    move = -(np.clip(out, 0, None) + 0.03 * face.Wf * corner)[:, None] * nrm
    if frontal > 0:
        # the evening out may move a side either way, kept small
        sym = ref - uv
        move = move + np.clip(sym, -0.03 * face.Wf, 0.03 * face.Wf) * frontal
    move *= _jaw_weight(n)[:, None] / max(_jaw_weight(n).max(), 1e-6)
    # smooth the move itself along the contour
    k2 = np.exp(-0.5 * (np.arange(-4, 5) / 1.6) ** 2)
    k2 /= k2.sum()
    padm = np.vstack([np.repeat(move[:1], 4, 0), move, np.repeat(move[-1:], 4, 0)])
    move = np.stack([np.convolve(padm[:, c], k2, mode="valid") for c in range(2)], -1)
    res = uv + move
    res[:, 0] += mid_u
    return face.O[None] + res[:, :1] * face.ex[None] + res[:, 1:2] * face.ey[None]


def sculpt_vectors(face: FaceGeom, sculpt: Dict[int, List[float]]) -> np.ndarray:
    """(N_DENSE, 2) forward displacement of every dense point, source px. A dragged contour point
    carries its neighbours along the contour so the outline stays smooth."""
    out = np.zeros((N_DENSE, 2))
    if not sculpt:
        return out
    raw = np.zeros((N_DENSE, 2))
    for k, uv in sculpt.items():
        if 0 <= int(k) < N_DENSE:
            raw[int(k)] = face.from_units_vec(np.array(uv[:2], float))
    # contour (jaw + forehead) spread along the outline
    cont = np.r_[np.arange(N_JAW), np.arange(D_FORE.start + 1, D_FORE.stop - 1)]
    n = len(cont)
    vals = raw[cont]
    if np.abs(vals).max() > 0:
        idx = np.arange(n)
        sig = 2.2
        Wm = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / sig) ** 2)
        # only spread from points the user moved
        moved = np.abs(vals).sum(1) > 1e-9
        Wm = Wm[:, moved]
        spread = Wm @ vals[moved] / np.maximum(1.0, Wm.sum(1, keepdims=True))
        spread = np.where(moved[:, None], vals, spread)
        out[cont] = spread
    inner = np.r_[np.arange(D_BROW.start, D_MOUTH.stop)]
    out[inner] = raw[inner]
    return out


def face_forward(grid: _Grid, face: FaceGeom, head: Optional[HeadGeom], A: Dict[str, float],
                 sculpt: Dict[int, List[float]], w: float, feather: float = 1.0, parts=None) -> np.ndarray:
    """Forward displacement (gh, gw, 2) in source px. A: amounts as fractions (−0.1 = 10 % smaller).
    w: overall weight (strength x face usability). feather: how far changes fade out (1 = default)."""
    f = float(np.clip(feather, 0.4, 3.0))
    X, Y = grid.X, grid.Y
    P = np.stack([X, Y], -1)
    V = np.zeros(X.shape + (2,))
    if w < 1e-3:
        return V
    Wf, iod = face.Wf, face.iod
    L = face.lm
    k_face = A.get("face", 0.0) * w
    k_len = A.get("face_length", 0.0) * w
    k_low = A.get("lower_face", 0.0) * w
    k_jaw = A.get("jaw", 0.0) * w
    s_fix = A.get("jaw_fix", 0.0) * w
    s_sc = A.get("sculpt", 0.0) * w
    k_cb = A.get("cheekbone", 0.0) * w
    k_chin = A.get("chin", 0.0) * w
    k_cw = A.get("chin_width", 0.0) * w
    sc = sculpt_vectors(face, sculpt) * s_sc if (sculpt and abs(s_sc) > 1e-4) else None

    # ---- face layer: interior map + contour-only edits
    face_on = any(abs(x) > 1e-4 for x in (k_face, k_len, k_low, k_jaw, s_fix, k_cb, k_chin, k_cw)) or sc is not None
    if face_on:
        cont = face.contour()
        Cf = 0.55 * L[30] + 0.45 * L[29]
        v_nb = float(face.v(L[33:34])[0])

        v_brow_f = float(face.v(L[17:27]).mean())

        def interior(Q):
            vq = face.v(Q)
            out = (Q - Cf) * k_face
            # above the brows the face size works mostly sideways: the hairline (and the hair above it) stays
            # nearly where it is instead of being pulled down into the forehead
            up = (v_brow_f - vq) / max(1.0, 0.35 * Wf)
            keep_up = 0.65 * _smoother(0.0, 1.0, up)
            dv = (out[..., 0] * face.ey[0] + out[..., 1] * face.ey[1]) * keep_up
            out = out - face.ey * dv[..., None]
            out = out + face.ey * (k_len * np.maximum(vq, 0.0))[..., None]
            out = out + face.ey * (k_low * np.maximum(vq - v_nb, 0.0))[..., None]
            return out

        # contour-only displacement per contour vertex
        nj = N_JAW
        dc = np.zeros((len(cont), 2))
        if abs(k_jaw) > 1e-4:
            uj = face.u(face.jaw) - float(np.median(face.u(L[27:31])))
            inward = -np.sign(uj)[:, None] * face.ex[None]
            # mix in the local inward normal so the move follows the curve
            tang = np.gradient(face.jaw, axis=0)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], -1)
            nrm /= np.maximum(np.hypot(*nrm.T), 1e-9)[:, None]
            cen = face.jaw.mean(0)
            flip = ((face.jaw - cen) * nrm).sum(1) > 0
            nrm[flip] *= -1
            dirv = 0.7 * inward + 0.3 * nrm
            dirv /= np.maximum(np.hypot(*dirv.T), 1e-9)[:, None]
            dc[:nj] += dirv * (-k_jaw * 0.5 * Wf * 0.9 * _jaw_weight(nj))[:, None]
        if abs(s_fix) > 1e-4:
            dfix = jaw_targets(face) - face.jaw
            # a badly seen jaw (turned face, hand in front) must never throw the outline around
            mag = np.hypot(dfix[:, 0], dfix[:, 1])
            cap = 0.06 * Wf
            dfix *= np.minimum(1.0, cap / np.maximum(mag, 1e-6))[:, None]
            dc[:nj] += dfix * min(1.0, s_fix)
        mid_u = float(np.median(face.u(L[27:31])))
        qj = np.linspace(0, 1, nj)
        if abs(k_cb) > 1e-4:
            # cheekbones: the outline at the height between the eyes and the nose tip, pushed sideways
            v_cb = 0.55 * float(face.v(L[30:31])[0])
            vc = face.v(cont)
            uc = face.u(cont) - mid_u
            wcb = np.exp(-0.5 * ((vc - v_cb) / (0.14 * Wf)) ** 2)
            dc += (np.sign(uc) * k_cb * 0.18 * Wf * wcb)[:, None] * face.ex[None]
        if abs(k_chin) > 1e-4:
            # chin tip only: the lowest part of the jaw moves along the face axis
            wch = np.exp(-0.5 * ((qj - 0.5) / 0.10) ** 2)
            dc[:nj] += (k_chin * 0.22 * Wf * wch)[:, None] * face.ey[None]
        if abs(k_cw) > 1e-4:
            # chin width: the chin squeezed towards the middle (zero on the centre line, so it stays smooth)
            uj = face.u(face.jaw) - mid_u
            wcw = np.exp(-0.5 * ((qj - 0.5) / 0.13) ** 2)
            dc[:nj] += (k_cw * 0.6 * uj * wcw)[:, None] * face.ex[None]
        if sc is not None:
            dc[:nj] += sc[D_JAW]
            dc[nj:] += sc[D_FORE.start + 1:D_FORE.stop - 1]
        amp = float(np.abs(dc).max()) if dc.size else 0.0
        amp_in = float(np.abs(interior(cont)).max())
        nr = grid.nearest_on(cont, closed=True)
        if nr is not None:
            d = nr[0]
            inside = grid.inside(cont)
            L_in = 0.32 * Wf * (0.85 + 0.15 * f)       # inside: the softness should not change how much moves
            # outside the face only a short band follows (hair and neck stay put); long enough not to fold
            L_out = max(0.18 * Wf, min(3.0 * max(amp, amp_in), max(0.36 * Wf, 2.3 * max(amp, amp_in)))) * f
            step = max(1.5, 0.035 * Wf)
            cp, cv_extra = _densify(cont, dc, step)
            c_all = interior(cp) + cv_extra
            sig = max(2.0, 0.06 * Wf * (0.7 + 0.3 * f))
            ext = _smooth_extend(X, Y, cp, np.hstack([cv_extra, c_all]), sig)
            fin = 1.0 - _smoother(0, L_in, d)
            V_in = interior(P) + ext[..., :2] * fin[..., None]
            fout = 1.0 - _smoother(0, L_out, d)
            V_out = ext[..., 2:] * fout[..., None]
            V += V_in * inside[..., None] + V_out * (1 - inside[..., None])

    # ---- local features
    bf = 0.9 + 0.1 * f

    def bump(c, ax, ay_up, ay_dn, k, keep=None):
        """Smooth local scaling about c inside an egg shape (ax along ex; ay_up towards the brows,
        ay_dn towards the chin). keep: weight map that must stay still (0..1, 1 = keep)."""
        if abs(k) < 1e-4:
            return
        du = (X - c[0]) * face.ex[0] + (Y - c[1]) * face.ex[1]
        dv = (X - c[0]) * face.ey[0] + (Y - c[1]) * face.ey[1]
        ay = np.where(dv > 0, ay_dn, ay_up)
        r2 = (du / ax) ** 2 + (dv / ay) ** 2
        win = np.clip(1 - r2, 0, None) ** 2
        if keep is not None:
            win = win * (1 - keep)
        V[..., 0] += (X - c[0]) * k * win
        V[..., 1] += (Y - c[1]) * k * win

    def soft_poly(poly, grow, feather_px):
        c = poly.mean(0)
        m = grid.inside(c + (poly - c) * grow)
        sg = feather_px / max(grid.hx, 1e-6)
        if sg > 0.3:
            m = np.clip(cv2.GaussianBlur(m, (0, 0), sg) * 1.5, 0, 1)
        return m

    v_of = lambda q: float(face.v(np.asarray(q)[None])[0])
    k_nose = A.get("nose", 0.0) * w
    if abs(k_nose) > 1e-4:
        nw_ = float(np.hypot(*(L[35] - L[31])))
        c = 0.4 * L[30] + 0.6 * L[33]
        gap = max(1.0, v_of(L[51]) - v_of(c))            # nose centre to the top of the upper lip
        up = 0.85 * max(1.0, v_of(c) - v_of(L[27]))
        mouth_keep = soft_poly(L[48:60], 1.15, 0.05 * iod)
        bump(c, 1.25 * nw_ * bf, up * bf, 0.75 * gap, k_nose, mouth_keep)
    k_mouth = A.get("mouth", 0.0) * w
    if abs(k_mouth) > 1e-4:
        mw = float(np.hypot(*(L[54] - L[48])))
        c = L[48:60].mean(0)
        to_nose = max(1.0, v_of(c) - v_of(L[33]))
        to_chin = max(1.0, v_of(face.chin) - v_of(c))
        nose_keep = soft_poly(L[31:36], 1.3, 0.05 * iod)
        bump(c, 0.95 * mw * bf, 0.9 * to_nose, 0.75 * to_chin * bf, k_mouth, nose_keep)
    k_br = A.get("nose_bridge", 0.0) * w
    if abs(k_br) > 1e-4:
        a, b = L[27], 0.5 * L[30] + 0.5 * L[33]
        ax_ = b - a
        Lb = float(np.hypot(*ax_))
        if Lb > 2:
            t = ((X - a[0]) * ax_[0] + (Y - a[1]) * ax_[1]) / (Lb * Lb)
            nrm = np.array([-ax_[1], ax_[0]]) / Lb
            ub = (X - a[0]) * nrm[0] + (Y - a[1]) * nrm[1]
            B = 0.30 * iod * bf
            wt = _smoother(-0.25, 0.1, t) * (1 - _smoother(0.85, 1.15, t))
            prof = np.clip(1 - (ub / B) ** 2, 0, None) ** 2
            m = k_br * ub * prof * wt
            V[..., 0] += m * nrm[0]
            V[..., 1] += m * nrm[1]
    # mouth corners up / down
    k_mc = A.get("mouth_corner", 0.0) * w
    if abs(k_mc) > 1e-4:
        mw = float(np.hypot(*(L[54] - L[48])))
        sg = 0.26 * mw * bf
        for cn in (L[48], L[54]):
            g = np.exp(-((X - cn[0]) ** 2 + (Y - cn[1]) ** 2) / (2 * sg * sg))
            V[..., 0] -= face.ey[0] * k_mc * 0.24 * mw * g
            V[..., 1] -= face.ey[1] * k_mc * 0.24 * mw * g
    # philtrum: the lips move as one piece towards / away from the nose; nose and chin tip stay
    k_ph = A.get("philtrum", 0.0) * w
    if abs(k_ph) > 1e-4:
        v33, v51, v57 = v_of(L[33]), v_of(L[51]), v_of(L[57])
        v8 = v_of(face.chin)
        gap = max(1.0, v51 - v33)
        mw = float(np.hypot(*(L[54] - L[48])))
        vq = face.v(P)
        uq = face.u(P) - float(face.u(L[48:60].mean(0)[None])[0])
        prof = _smoother(v33, v51, vq) * (1 - _smoother(v57, v57 + 0.8 * max(1.0, v8 - v57), vq))
        lat = 1 - _smoother(0.6 * mw * bf, 1.15 * mw * bf, np.abs(uq))
        d = k_ph * gap * prof * lat
        V[..., 0] += face.ey[0] * d
        V[..., 1] += face.ey[1] * d
    # eyes: distance, outer corner tilt, height
    k_ed = A.get("eye_dist", 0.0) * w
    k_et = A.get("eye_tilt", 0.0) * w
    k_eh = A.get("eye_height", 0.0) * w
    if any(abs(x) > 1e-4 for x in (k_ed, k_et, k_eh)):
        mid = float(face.u(L[27:31]).mean())
        for idx, inner_i, outer_i in ((slice(36, 42), 39, 36), (slice(42, 48), 42, 45)):
            ce = L[idx].mean(0)
            ci_, co_ = L[inner_i], L[outer_i]
            ew = float(np.hypot(*(co_ - ci_)))
            if ew < 2:
                continue
            du = (X - ce[0]) * face.ex[0] + (Y - ce[1]) * face.ex[1]
            dv = (X - ce[0]) * face.ey[0] + (Y - ce[1]) * face.ey[1]
            if abs(k_ed) > 1e-4:
                r2 = (du / (1.0 * ew * bf)) ** 2 + (dv / (0.8 * ew * bf)) ** 2
                win = np.clip(1 - r2, 0, None) ** 2
                out_sign = np.sign(float(face.u(ce[None])[0]) - mid)
                m = out_sign * k_ed * 0.22 * iod * win
                V[..., 0] += face.ex[0] * m
                V[..., 1] += face.ex[1] * m
            if abs(k_et) > 1e-4:
                a = co_ - ci_
                sp = ((X - ci_[0]) * a[0] + (Y - ci_[1]) * a[1]) / (ew * ew)
                pv = dv / (0.75 * ew * bf)
                win = _smoother(0.1, 1.0, sp) * (1 - _smoother(1.0, 1.7, sp)) * np.clip(1 - pv * pv, 0, None) ** 2
                m = -k_et * 0.45 * ew * win
                V[..., 0] += face.ey[0] * m
                V[..., 1] += face.ey[1] * m
            if abs(k_eh) > 1e-4:
                r2 = (du / (0.9 * ew * bf)) ** 2 + (dv / (0.7 * ew * bf)) ** 2
                win = np.clip(1 - r2, 0, None) ** 2
                m = k_eh * 1.6 * dv * win
                V[..., 0] += face.ey[0] * m
                V[..., 1] += face.ey[1] * m
    if sc is not None:
        inner = np.r_[np.arange(D_BROW.start, D_MOUTH.stop)]
        dense = face.dense()
        sig = 0.16 * iod
        for i in inner:
            v = sc[i]
            if abs(v).max() < 1e-6:
                continue
            g = np.exp(-((X - dense[i, 0]) ** 2 + (Y - dense[i, 1]) ** 2) / (2 * sig * sig))
            V[..., 0] += v[0] * g
            V[..., 1] += v[1] * g

    # ---- head layer
    if head is not None:
        k_head = A.get("head", 0.0) * w
        k_crown = A.get("crown", 0.0) * w
        k_hair = A.get("hair_volume", 0.0) * w
        if any(abs(x) > 1e-4 for x in (k_head, k_crown, k_hair)):
            poly = head.polygon(face)
            # hair band weight: 0 on the face outline, 1 on the head outline (radial, smooth)
            hair_w, hair_bw = _radial_band(X, Y, head.center, face.contour(), head.polygon(face), True)

            hw_poly, bw_poly = _radial_band(poly[:, 0], poly[:, 1], head.center, face.contour(), poly, True)
            outline_disp = head_move(face, head, poly, hw_poly, bw_poly, k_head, k_crown, k_hair)
            amp = float(np.abs(outline_disp).max())
            reach = head_reach(head, amp, f)
            # one smooth map inside and around the head: the head's own movement, faded out with the distance
            # beyond the head outline (per direction, from a smoothed outline radius). A nearest-outline-point
            # extension creased the background where the outline turns (beside the ears) and reached the
            # shoulders; this fades evenly in every direction.
            raw = _polar_radius(poly, head.center, 360, smooth=1)
            ro = np.max([np.roll(raw, k) for k in range(-8, 9)], axis=0)       # never inside the outline
            ro = np.convolve(np.r_[ro[-10:], ro, ro[:10]], np.ones(21) / 21, mode="valid")
            ang = (np.arctan2(Y - head.center[1], X - head.center[0]) + np.pi) / (2 * np.pi) * 360
            b_ = np.clip(ang.astype(int), 0, 359)
            dist = np.hypot(X - head.center[0], Y - head.center[1])
            t_out = np.clip((dist - ro[b_]) / max(reach, 1.0), 0.0, 1.0)
            fade = 1.0 - _smoother(0.0, 1.0, t_out)
            if parts is not None and abs(k_hair) > 1e-4:
                # hair found by segmentation: the volume goes on the hair only; face skin and ears stay, and hair
                # that hangs down beside the face gets volume too instead of fading out below the eyes
                keep = parts.sample("keep", X, Y)
                hairm = np.clip(parts.sample("hair", X, Y) * 1.6, 0.0, 1.0)
                Vb = head_move(face, head, P, hair_w, hair_bw, k_head, k_crown, 0.0)
                Vh = head_move(face, head, P, hair_w, hair_bw, 0.0, 0.0, k_hair, ear_fade=False) - \
                    head_move(face, head, P, hair_w, hair_bw, 0.0, 0.0, 0.0)
                # outside the hair (background just beyond it) the move fades with the head layer as before;
                # inside the head, only where there is hair
                inside = np.clip(parts.sample("hair", X, Y) + keep, 0, 1)
                wh = (hairm + (1 - inside) * _hair_fallback(face, head, P)) * (1 - keep)
                V += (Vb + Vh * wh[..., None]) * fade[..., None]
            else:
                V += head_move(face, head, P, hair_w, hair_bw, k_head, k_crown, k_hair) * fade[..., None]

    sig = max(0.8, 0.014 * Wf * f / max(grid.hx, 1e-6))
    if np.abs(V).max() > 1e-6:
        V = np.stack([cv2.GaussianBlur(V[..., c].astype(np.float32), (0, 0), sig) for c in range(2)], -1).astype(np.float64)
    return V


def head_reach(head: "HeadGeom", amp: float, feather: float) -> float:
    """How far outside the head outline its edits fade out (source px)."""
    return max(0.32 * head.R, 3.0 * amp) * feather


def _hair_fallback(face: FaceGeom, head: "HeadGeom", Q: np.ndarray) -> np.ndarray:
    """Where the masks have nothing (beyond the head outline): the old height fade, brows -> eyes."""
    L = face.lm
    v_brow = float(face.v(L[17:27]).mean())
    v_eye = float(face.v(L[36:48]).mean())
    lo = v_brow - 0.25 * max(1.0, v_eye - float(head.top_v))
    return 1.0 - _smoother(lo, v_eye, face.v(Q))


def head_move(face: FaceGeom, head: "HeadGeom", Q: np.ndarray, hw: np.ndarray, bw: np.ndarray,
              k_head: float, k_crown: float, k_hair: float, ear_fade: bool = True) -> np.ndarray:
    """Forward displacement of head points Q (…, 2) for head size, crown and hair volume.
    hw: 0 on the face outline .. 1 on the head outline; bw: band thickness there."""
    L = face.lm
    Wf = face.Wf
    pivot = face.chin + face.ey * 0.12 * Wf
    v_brow = float(face.v(L[17:27]).mean())
    v_eye = float(face.v(L[36:48]).mean())
    v_top = float(head.top_v)
    Hh = max(1.0, float(face.v(face.chin[None])[0]) - v_top)
    span = max(1.0, v_brow - v_top)
    vq = face.v(Q)
    # scaled about a point just below the chin; below it (neck, collarbones, shoulders) nothing is pushed down
    v_piv = float(face.v(pivot[None])[0])
    neck_keep = 1.0 - _smoother(v_piv - 0.02 * Hh, v_piv + 0.22 * Hh, vq)
    out = (Q - pivot) * (k_head * neck_keep)[..., None]
    # crown: the part above the brows stretches upwards, most at the top
    cw = np.clip((v_brow - vq) / span, 0.0, 1.0)
    out = out - face.ey * (k_crown * 0.45 * Hh * cw * cw * (3 - 2 * cw))[..., None]
    if abs(k_hair) > 1e-4:
        # hair grows outwards in proportion to how much hair there is in that direction; from the brows down
        # to the eyes it fades out, so the ears (beside the eyes) and hair hanging below them stay where they are
        rad = Q - head.center
        rn = np.maximum(np.hypot(rad[..., 0], rad[..., 1]), 1e-6)[..., None]
        lo = v_brow - 0.25 * max(1.0, v_eye - v_top)
        ear_keep = 1.0 - _smoother(lo, v_eye, vq) if ear_fade else 1.0
        out = out + rad / rn * (k_hair * 1.1 * bw * hw * ear_keep)[..., None]
    return out


def _polar_radius(poly: np.ndarray, c: np.ndarray, nb: int = 360, smooth: int = 7) -> np.ndarray:
    """Outer radius of a closed polygon around c per angle bin (densified, max per bin, gaps filled)."""
    P = np.vstack([poly, poly[:1]])
    pts = []
    for i in range(len(P) - 1):
        n = max(2, int(np.hypot(*(P[i + 1] - P[i])) / 2))
        s = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(P[i] + s * (P[i + 1] - P[i]))
    Q = np.vstack(pts) - c
    ang = (np.arctan2(Q[:, 1], Q[:, 0]) + np.pi) / (2 * np.pi) * nb
    b = np.clip(ang.astype(int), 0, nb - 1)
    r = np.hypot(Q[:, 0], Q[:, 1])
    out = np.full(nb, np.nan)
    np.fmax.at(out, b, r)
    ok = np.isfinite(out)
    if ok.sum() < 3:
        return np.full(nb, np.nanmax(r) if len(r) else 1.0)
    idx = np.arange(nb)
    out = np.interp(idx, idx[ok], out[ok], period=nb)
    h = smooth // 2
    k = np.ones(2 * h + 1) / (2 * h + 1)
    return np.convolve(np.r_[out[-h:], out, out[:h]], k, mode="valid")


def _radial_band(X, Y, c, inner_poly, outer_poly, with_width: bool = False):
    """0 inside the inner outline, rising smoothly to 1 at the outer outline (per direction from c).
    with_width: also the band thickness (outer - inner radius) in that direction, smoothed over ±20 degrees:
    hair volume moves the outline by an amount that follows this thickness, and a thickness that jumps
    between neighbouring directions (a bun, a ponytail) would shear and fold the picture."""
    nb = 360
    ri = _polar_radius(inner_poly, c, nb)
    ro = np.maximum(_polar_radius(outer_poly, c, nb), ri + 1.0)
    ang = (np.arctan2(Y - c[1], X - c[0]) + np.pi) / (2 * np.pi) * nb
    b = np.clip(ang.astype(int), 0, nb - 1)
    r = np.hypot(X - c[0], Y - c[1])
    t = np.clip((r - ri[b]) / (ro[b] - ri[b]), 0, 1)
    t = t * t * (3 - 2 * t)
    if with_width:
        bw = ro - ri
        h = 20
        bw = np.convolve(np.r_[bw[-h:], bw, bw[:h]], np.ones(2 * h + 1) / (2 * h + 1), mode="valid")
        return t, bw[b]
    return t


def eye_bulge(X, Y, face: FaceGeom, amount: float, moved: Optional[np.ndarray] = None):
    """Backward-map eye enlargement (radial magnification, centred on each eye)."""
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    if abs(amount) < 1e-4:
        return dx, dy
    av = float(np.clip(amount, -0.35, 0.35)) * 0.9
    L = face.lm
    for k, (idx, (c0, c1)) in enumerate(((list(range(36, 42)), (36, 39)), (list(range(42, 48)), (42, 45)))):
        ep = L[idx]
        ew = float(np.hypot(*(L[c1] - L[c0])))
        if ew < 3:
            continue
        c = ep.mean(0) + (moved[k] if moved is not None else 0.0)
        R = ew * 1.15
        x0, x1 = c[0] - R, c[0] + R
        y0, y1 = c[1] - R, c[1] + R
        sel = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        if not sel.any():
            continue
        Xs, Ys = X[sel] - c[0], Y[sel] - c[1]
        r2 = (Xs * Xs + Ys * Ys) / (R * R)
        kk = np.clip(1 - r2, 0, None) ** 2 * av
        dx[sel] -= Xs * kk
        dy[sel] -= Ys * kk
    return dx, dy


def face_backward(X: np.ndarray, Y: np.ndarray, face: FaceGeom, head: Optional[HeadGeom], A: Dict[str, float],
                  sculpt: Dict[int, List[float]], w: float, feather: float = 1.0,
                  parts=None) -> Tuple[np.ndarray, np.ndarray]:
    """Backward map (dx, dy) on the grid window for all face/head edits of one person."""
    grid = _Grid(X, Y)
    V = face_forward(grid, face, head, A, sculpt, w, feather, parts)
    D = invert(V, grid) if np.abs(V).max() > 1e-6 else np.zeros(X.shape + (2,))
    e = A.get("eyes", 0.0) * w
    if abs(e) > 1e-4:
        # eyes are enlarged where the other edits put them
        moved = []
        for idx in (slice(36, 42), slice(42, 48)):
            c = face.lm[idx].mean(0)
            gi = grid.to_g(c[None])[0]
            gx_, gy_ = int(np.clip(round(gi[0]), 0, X.shape[1] - 1)), int(np.clip(round(gi[1]), 0, X.shape[0] - 1))
            moved.append(V[gy_, gx_])
        ex_, ey_ = eye_bulge(X + D[..., 0], Y + D[..., 1], face, e, np.array(moved))
        D[..., 0] += ex_
        D[..., 1] += ey_
    return D[..., 0], D[..., 1]


def face_extent(face: FaceGeom, head: Optional[HeadGeom], A: Dict[str, float]) -> np.ndarray:
    """Points that bound everything the face/head edits can touch (for the grid window). The window must hold
    the whole fade-out: a field cut off at the window edge shows as a hard edge (the top of an enlarged head or
    a big hair volume looked sliced off, as if it hit a wall)."""
    pts = [face.contour(), face.lm]
    f = float(np.clip(A.get("_feather", 1.0), 0.4, 3.0))
    big = max(abs(A.get(k, 0.0)) for k in ("face", "face_length", "lower_face", "jaw", "jaw_fix", "cheekbone",
                                             "chin", "chin_width"))
    amp_f = big * 0.6 * face.Wf
    m = max(0.45 * face.Wf, 3.0 * amp_f * f + amp_f) * max(1.0, f) + 0.05 * face.Wf
    if head is not None and any(abs(A.get(k, 0.0)) > 1e-4 for k in ("head", "crown", "hair_volume")):
        poly = head.polygon(face)
        pts.append(poly)
        hw, bw = _radial_band(poly[:, 0], poly[:, 1], head.center, face.contour(), poly, True)
        d = head_move(face, head, poly, hw, bw, A.get("head", 0.0), A.get("crown", 0.0), A.get("hair_volume", 0.0))
        amp = float(np.abs(d).max())
        m = max(m, head_reach(head, amp, f) + amp + 0.1 * head.R)
    P = np.vstack(pts)
    return np.vstack([P.min(0) - m, P.max(0) + m])


def face_mask_poly(face: FaceGeom) -> np.ndarray:
    return face.contour()
