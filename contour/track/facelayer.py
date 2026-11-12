"""Per-frame face layer, computed once at analysis: what is skin and neck around each face, and the skin's
colour, both held steady through time.

Without it every rendered frame decided on its own (colour model re-centred on that frame, neck length from
that frame's shoulders), and the mask edges shook by a few pixels from frame to frame even on a still face.

For every person and frame with a face:
- face parsing (faceparse.py) in a crop that follows the stabilised landmarks, so in crop space the face stands
  still and the maps of neighbouring frames line up; they are averaged over a few frames (only within a shot,
  only between frames whose crop moved little relative to the face) -> skin, neck maps without flicker;
- how much the parser itself can be trusted on that frame (dark, blurred or turned-away faces it reads poorly);
- the face skin's colour centre (Cr, Cb) and brightness, measured on the parser's skin (so hair, a hand or a
  light stick in front do not pull it) and median-filtered over the shot.
Saved as cuts/<id>/cache/facelayer.npz. The renderer falls back to the old per-frame path when it is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from . import faceparse as FP

VERSION = 1
MIN_FACE_PX = 48       # proxy px (1920 wide): smaller faces are not parsed
T_SIGMA = 1.3          # frames, temporal averaging of the parse maps
T_R = 3
ADD_MAX = 0.2          # how much skin the neighbours may add to a frame's own parse
REL_R = 6              # frames each side, parser trust
COLOR_R = 3            # frames each side, colour centre median


ALIGN_STEP = 0.085     # of the face width in the crop: search grid for a landmark set that sits beside the face
ALIGN_N = 3            # grid -3..3 steps each way (about 0.25 face widths)
ALIGN_ON = False       # off: at full size the landmarks of the checked cuts agreed with the parser; kept for later
ALIGN_GAIN = 0.10      # IoU the shifted outline must gain before landmarks are moved


def _align(r, lm):
    """In fast motion or blur the tracked landmarks can lie beside the face (the mask then sits on hair, neck or
    the background). The parser sees the real face in the crop: slide the landmark outline to where it covers
    the parser's skin best. Returns the shift in image px (x, y), or None when the landmarks are fine."""
    from ..retouch.skin import face_polygon
    K = FP.KEEP
    s = K / FP.SIZE
    L = lm.copy()
    L[:, :2] = lm[:, :2] @ r["M"][:, :2].T + r["M"][:, 2]
    L[:, :2] *= s
    poly = face_polygon(L)
    if poly is None:
        return None
    wf = float(np.hypot(*(L[16, :2] - L[0, :2])))
    skin = (r["skin"].astype(np.float32) > 0.5)
    if skin.sum() < 40 or wf < 10:
        return None
    # skin close to the outline only: a neighbour's face further off must not pull it over
    near = np.zeros((K, K), np.uint8)
    cv2.fillPoly(near, [np.round(poly).astype(np.int32)], 1)
    k = max(3, int(0.45 * wf) | 1)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sk = skin & (near > 0)
    st_ = ALIGN_STEP * wf
    best, best_d, base = -1.0, (0.0, 0.0), 0.0
    for iy in range(-ALIGN_N, ALIGN_N + 1):
        for ix in range(-ALIGN_N, ALIGN_N + 1):
            d = np.array([ix * st_, iy * st_])
            m = np.zeros((K, K), np.uint8)
            cv2.fillPoly(m, [np.round(poly + d).astype(np.int32)], 1)
            m = m > 0
            u = (m | sk).sum()
            iou = float((m & sk).sum()) / max(1, u)
            if ix == 0 and iy == 0:
                base = iou
            if iou > best + 1e-6 or (abs(iou - best) <= 1e-6 and np.hypot(*d) < np.hypot(*best_d)):
                best, best_d = iou, (d[0], d[1])
    if best - base < ALIGN_GAIN or best < 0.30:
        return None
    A = r["M"][:, :2].astype(np.float64)
    v = np.linalg.solve(A, np.array(best_d) / s)
    return v


def _crop_lm(lm, M):
    out = lm.copy()
    out[:, :2] = lm[:, :2] @ M[:, :2].T + M[:, 2]
    return out


def path_for(tracker) -> Path:
    return Path(tracker.project.cache_dir(tracker.cut)) / "facelayer.npz"


def _reliability(skin: np.ndarray, lm_crop: np.ndarray) -> float:
    """Share of the inner face (landmark outline minus eyes, brows, mouth) that the parser calls skin."""
    from ..retouch.skin import face_polygon, _features_mask
    K = FP.KEEP
    s = K / FP.SIZE
    L = lm_crop.copy()
    L[:, :2] *= s
    poly = face_polygon(L)
    if poly is None:
        return 0.0
    reg = np.zeros((K, K), np.uint8)
    cv2.fillPoly(reg, [np.round(poly).astype(np.int32)], 255)
    reg = cv2.erode(reg, np.ones((5, 5), np.uint8))
    feat = _features_mask(L, (K, K))
    sel = (reg > 0) & (feat == 0)
    if sel.sum() < 30:
        return 0.0
    return float(skin.astype(np.float32)[sel].mean())


def build(tracker, progress: Optional[Callable[[float, str], None]] = None,
          cancel: Optional[Callable[[], bool]] = None) -> Optional[Path]:
    if not FP.available():
        return None
    st = tracker.store
    px = tracker.proxy
    if st is None or not getattr(st, "has_face", False):
        return None
    P, N = st.face.shape[0], st.face.shape[1]
    def _wf(p, t):
        F = st.face[p, t, :, :2]
        return float(np.hypot((F[16, 0] - F[0, 0]) * px.sx, (F[16, 1] - F[0, 1]) * px.sy))
    def _wf_arr(F):
        return float(np.hypot((F[16, 0] - F[0, 0]) * px.sx, (F[16, 1] - F[0, 1]) * px.sy))
    # faces too small to parse usefully (and to retouch with any detail) are left to the colour model
    still = {q.id for q in getattr(tracker.cut, "persons", []) if getattr(q, "still", False)}
    todo = [(p, t) for p in range(P) for t in range(N)
            if int(st.pids[p]) not in still and st.present[p, t] and np.isfinite(st.face[p, t, :, :2]).all() and st.face_vis[p, t] >= 0.08
            and _wf(p, t) >= MIN_FACE_PX]
    # frames where the tracker lost the landmarks (hand, hair or a turn over the face) but the person is there and
    # was seen a few frames away: the face is carried over from the nearest seen frame, and the parser decides
    # from the pixels what is skin. Its trust (rel) is measured as usual.
    virt: Dict = {}
    have_lm = {(p, t) for p, t in todo}
    for p in range(P):
        if int(st.pids[p]) in still:
            continue
        seen = np.nonzero(st.face_vis[p] >= 0.3)[0]
        if not len(seen):
            continue
        for t in range(N):
            if (p, t) in have_lm or not st.present[p, t] or np.min(np.abs(seen - t)) > 12:
                continue
            try:
                v = tracker._virtual_face(p, t)
            except Exception:
                v = None
            if v is None or not np.isfinite(v).all():
                continue
            lm = np.zeros((v.shape[0], 3), np.float32)
            lm[:, :2] = v
            lm[:, 0] /= px.sx
            lm[:, 1] /= px.sy
            if _wf_arr(lm) >= MIN_FACE_PX:
                virt[(p, t)] = lm
                todo.append((p, t))
    if not todo:
        return None
    parser = FP.FaceParser()
    K = FP.KEEP
    M = np.full((P, N, 2, 3), np.nan, np.float32)
    skin = np.zeros((P, N, K, K), np.float16)
    neck = np.zeros((P, N, K, K), np.float16)
    tex = np.ones((P, N, K, K), np.float16)
    rel = np.zeros((P, N), np.float32)
    col = np.full((P, N, 3), np.nan, np.float32)            # Cr, Cb, Y
    have = np.zeros((P, N), bool)
    shift = np.zeros((P, N, 2), np.float32)        # landmark correction found by the parser, proxy px
    from ..retouch.skin import _features_mask, _texture_gate
    last_t, rgb = -1, None
    for i, (p, t) in enumerate(sorted(todo, key=lambda x: (x[1], x[0]))):
        if cancel and cancel():
            return None
        if t != last_t:
            rgb = np.ascontiguousarray(np.asarray(px.frame(t)))
            ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            last_t = t
        lm = (virt[(p, t)] if (p, t) in virt else st.face[p, t]).astype(np.float64).copy()
        lm[:, 0] *= px.sx
        lm[:, 1] *= px.sy
        r = parser.parse(rgb, lm)
        if r is None:
            continue
        if ALIGN_ON and (p, t) not in virt:
            sh = _align(r, lm)
            if sh is not None:
                lm2 = lm.copy()
                lm2[:, :2] += sh
                r2 = parser.parse(rgb, lm2)
                if r2 is not None:
                    lm, r = lm2, r2
                    shift[p, t] = sh
        M[p, t] = r["M"]
        # a dark, blurred or backlit face is read in patches (holes, specks): the retouch then comes out patchy.
        # Close the gaps (about 6 to 12 % of the face width); eyes, brows and mouth are cut out separately later
        rel_now = _reliability(r["skin"], _crop_lm(lm, r["M"]))
        kk = 5 if rel_now >= 0.55 else 9
        r["skin"] = cv2.morphologyEx(r["skin"].astype(np.float32), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))).astype(np.float16)
        skin[p, t] = r["skin"]
        neck[p, t] = r["neck"]
        # the hair/skin texture gate, in the same crop so it can be held steady through time too
        tg = _texture_gate(rgb, lm)
        if tg is not None:
            S = np.array([[K / FP.SIZE, 0, 0], [0, K / FP.SIZE, 0], [0, 0, 1]])
            A = (S @ np.vstack([r["M"].astype(np.float64), [0, 0, 1]]))[:2]
            tex[p, t] = cv2.warpAffine(tg, A, (K, K), flags=cv2.INTER_AREA, borderValue=1.0).astype(np.float16)
        lmc = lm.copy()
        lmc[:, :2] = lm[:, :2] @ r["M"][:, :2].T + r["M"][:, 2]
        rel[p, t] = _reliability(r["skin"], lmc)
        have[p, t] = True
        # colour centre on confident parser skin, eyes/brows/mouth out
        sk = FP.to_image(r["skin"], r["M"], rgb.shape[:2])
        feat = _features_mask(lm, rgb.shape[:2]) > 0
        sel = (sk > 0.8) & ~feat & (ycc[..., 0] > 12) & (ycc[..., 0] < 245)
        if sel.sum() > 60:
            v = ycc[sel].astype(np.float32)
            col[p, t] = [np.median(v[:, 1]), np.median(v[:, 2]), np.median(v[:, 0])]
        if progress and i % 10 == 0:
            progress(i / len(todo), "얼굴 영역 분석")
    # landmarks that sat beside the face are moved onto it (steadied over a few frames of the same shot), so the
    # geometry masks (face outline, neck, folds) follow the face and not the drifted tracking
    if np.abs(shift).sum() > 0:
        connected = st.connected if hasattr(st, "connected") else (lambda a, b: True)
        fixed = shift.copy()
        for p in range(P):
            for t in range(N):
                if not have[p, t]:
                    continue
                win = [u for u in range(max(0, t - 2), min(N, t + 3)) if have[p, u] and connected(u, t)]
                fixed[p, t] = np.median(shift[p, win], 0) if len(win) >= 3 else shift[p, t]
                if np.abs(shift[p, t]).sum() > 0 or np.abs(fixed[p, t]).sum() > 0:
                    st.face[p, t, :, 0] += fixed[p, t, 0] / px.sx
                    st.face[p, t, :, 1] += fixed[p, t, 1] / px.sy
        try:
            st.face_aligned = int((np.abs(shift).sum(2) > 0).sum())
        except Exception:
            pass
    # ---- temporal: within a shot, between frames whose face crop stays put
    conn = st.connected if hasattr(st, "connected") else (lambda a, b: True)
    skin_s = skin.copy()
    neck_s = neck.copy()
    tex_s = tex.copy()
    col_s = col.copy()
    for p in range(P):
        for t in range(N):
            if not have[p, t]:
                continue
            acc_s = skin[p, t].astype(np.float32)
            acc_n = neck[p, t].astype(np.float32)
            acc_t = tex[p, t].astype(np.float32)
            wsum = 1.0
            for d in range(1, T_R + 1):
                for s in (t - d, t + d):
                    if not (0 <= s < N) or not have[p, s]:
                        continue
                    if not conn(s, t):
                        continue
                    # the crops follow the face: neighbouring maps line up unless the crop's scale jumped
                    sc = np.hypot(*M[p, s, 0, :2]) / max(1e-6, np.hypot(*M[p, t, 0, :2]))
                    if not 0.92 < sc < 1.08:
                        continue
                    # nor across a light flicker: a frame gone dark sees no hair texture and little skin, and
                    # would wash out what the lit frame next to it sees clearly
                    ya, yb = col[p, s, 2], col[p, t, 2]
                    if np.isfinite(ya) and np.isfinite(yb) and abs(ya - yb) > 0.35 * max(ya, yb) + 6:
                        continue
                    w = float(np.exp(-0.5 * (d / T_SIGMA) ** 2))
                    acc_s += w * skin[p, s].astype(np.float32)
                    acc_n += w * neck[p, s].astype(np.float32)
                    acc_t += w * tex[p, s].astype(np.float32)
                    wsum += w
            # neighbours may take skin away (an edge that shook outwards) but add only a little: a hand or a
            # sleeve sweeping over the face is in this frame, and the frames around must not paint skin onto it
            own_s, own_n = skin[p, t].astype(np.float32), neck[p, t].astype(np.float32)
            skin_s[p, t] = (own_s + np.clip(acc_s / wsum - own_s, -1, ADD_MAX)).astype(np.float16)
            neck_s[p, t] = (own_n + np.clip(acc_n / wsum - own_n, -1, ADD_MAX)).astype(np.float16)
            tex_s[p, t] = (acc_t / wsum).astype(np.float16)
            # colour centre: median over the shot neighbourhood
            seg = [s for s in range(max(0, t - COLOR_R), min(N, t + COLOR_R + 1))
                   if np.isfinite(col[p, s]).all() and conn(s, t)]
            if seg:
                col_s[p, t] = np.median(col[p, seg], 0)
    # trust: a face half covered by a hand or a sleeve has little visible skin, which is exactly when the parser
    # is needed; a frame takes the best trust of its shot neighbourhood
    rel_s = rel.copy()
    for p in range(P):
        for t in range(N):
            if not have[p, t]:
                continue
            # ...but only from frames that look alike (same light on the face, about the same size): a face
            # going dark (into silhouette) is a parser that cannot see, not a hand in front of it
            def alike(s_):
                ya, yb = col[p, s_, 2], col[p, t, 2]
                if not (np.isfinite(ya) and np.isfinite(yb)):
                    return False
                if abs(ya - yb) > 0.25 * max(ya, yb) + 4:
                    return False
                sc = np.hypot(*M[p, s_, 0, :2]) / max(1e-6, np.hypot(*M[p, t, 0, :2]))
                return 0.85 < sc < 1.18
            nb = [s for s in range(max(0, t - REL_R), min(N, t + REL_R + 1))
                  if have[p, s] and conn(s, t) and (s == t or alike(s))]
            rel_s[p, t] = max(rel[p, nb]) if nb else rel[p, t]
    rel = rel_s
    out = path_for(tracker)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, version=VERSION, pids=np.array(st.pids), M=M, skin=skin_s, neck=neck_s, tex=tex_s, rel=rel,
                        col=col_s, have=have)
    if progress:
        progress(1.0, "얼굴 영역 분석")
    return out


class FaceLayer:
    def __init__(self, path: Path):
        d = np.load(path)
        self.pids = [int(x) for x in d["pids"]]
        self.M, self.skin, self.neck = d["M"], d["skin"], d["neck"]
        self.rel, self.col, self.have = d["rel"], d["col"], d["have"]
        self.tex = d["tex"] if "tex" in d.files else None

    def index(self, pid: int) -> Optional[int]:
        return self.pids.index(pid) if pid in self.pids else None

    def maps(self, pid: int, t: int, shape) -> Optional[Dict[str, np.ndarray]]:
        """skin, neck and the crop's coverage on the image (h, w); trust 0..1."""
        p = self.index(pid)
        if p is None or t >= self.have.shape[1] or not self.have[p, t]:
            return None
        Mt = self.M[p, t]
        cover = FP.to_image(np.ones((FP.KEEP, FP.KEEP), np.float32), Mt, shape)
        tex = None
        if self.tex is not None:
            tex = FP.to_image(self.tex[p, t], Mt, shape) * cover + (1.0 - cover)
        return dict(tex=tex, skin=FP.to_image(self.skin[p, t], Mt, shape), neck=FP.to_image(self.neck[p, t], Mt, shape),
                    cover=cover, trust=float(np.clip((self.rel[p, t] - 0.35) / 0.25, 0, 1)))

    def colour(self, pid: int, t: int) -> Optional[np.ndarray]:
        p = self.index(pid)
        if p is None or t >= self.col.shape[1]:
            return None
        c = self.col[p, t]
        return c if np.isfinite(c).all() else None


def load(tracker) -> Optional[FaceLayer]:
    """Cached on the tracker; reloaded when the file changes."""
    f = path_for(tracker)
    if not f.exists():
        tracker._facelayer = None
        return None
    mt = f.stat().st_mtime
    cur = getattr(tracker, "_facelayer", None)
    if cur is not None and getattr(tracker, "_facelayer_mt", None) == mt:
        return cur
    try:
        fl = FaceLayer(f)
    except Exception:
        fl = None
    tracker._facelayer = fl
    tracker._facelayer_mt = mt
    return fl
