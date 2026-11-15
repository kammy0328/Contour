"""Person matte with SAM 2: one exact silhouette per tracked person.

Every frame is encoded once by SAM 2.1; each tracked person is then decoded from tracking prompts (box, joints
on the body as positive points, other people's joints as negative points). People that are small in the frame
get their own close-up encoding so their outline is as detailed as a close shot.

The result is written into the regular matte store (so everything that reads the matte benefits) plus a label
store with the owner of every person pixel (person id + 1, 0 = nobody). RVM still runs alongside: away from
the tracked people it keeps untracked people, extras and props in the matte as before."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .exposure import analysis_view

from . import skeleton as S

POS_TORSO = [S.NOSE, S.L_SH, S.R_SH, S.L_HIP, S.R_HIP]
POS_LIMBS = [S.L_EL, S.R_EL, S.L_KN, S.R_KN]
SMALL_PERSON = 0.33          # box height share of the frame under which a person gets a close-up encoding
MAX_CLOSEUPS = 6


def model_for(accuracy: str) -> str:
    """SAM 3.1 for every accuracy (1.9.4). run() falls back to SAM 2.1 if it cannot be loaded."""
    return "seg_sam31"


def _kp_for(st, p: int, t: int, use_final: bool) -> Optional[np.ndarray]:
    """(17,3) source px + confidence (0..1) for prompting."""
    if use_final and np.isfinite(st.final[p, t, :, :2]).any():
        kp = st.final[p, t].copy()
        vis = st.vis[p, t] if st.vis is not None else None
        conf = np.where(np.isfinite(kp[:, 2]), kp[:, 2], 0.0)
        if vis is not None:
            from .store import VIS_VISIBLE
            conf = np.where(vis == VIS_VISIBLE, np.maximum(conf, 0.7), 0.0)
        kp[:, 2] = conf
        return kp
    arrs = [v[p, t] for k, v in st.raw.items() if k.startswith("pose_") and "@" not in k and v.shape[2] == S.K]
    arrs = [a for a in arrs if np.isfinite(a[:, :2]).any()]
    if not arrs:
        return None
    A = np.stack(arrs)
    kp = np.full((S.K, 3), np.nan, np.float32)
    kp[:, :2] = np.nanmean(A[:, :, :2], 0)
    kp[:, 2] = np.nanmean(np.where(np.isfinite(A[:, :, 2]), A[:, :, 2], 0.0), 0)
    return kp


def prompts(st, t: int, sx: float, sy: float, W: int, H: int, use_final: bool):
    """[(p, box (4,) proxy px, pos (N,2), neg (M,2))] for the people present on frame t."""
    people = []
    for p in np.nonzero(st.present[:, t])[0]:
        box = st.boxes[p, t]
        kp = _kp_for(st, p, t, use_final)
        if kp is None and not np.isfinite(box).all():
            continue
        if kp is not None:
            kp = kp.copy()
            kp[:, 0] *= sx
            kp[:, 1] *= sy
        if np.isfinite(box).all():
            bx = box * [sx, sy, sx, sy]
        else:
            ok = np.isfinite(kp[:, :2]).all(1)
            if ok.sum() < 3:
                continue
            x0, y0 = kp[ok, :2].min(0)
            x1, y1 = kp[ok, :2].max(0)
            bx = np.array([x0, y0, x1, y1])
        mirror = []
        if kp is not None:
            okk = np.isfinite(kp[:, :2]).all(1) & (kp[:, 2] >= 0.3)
            head = [kp[j, 1] for j in (S.NOSE, S.L_EYE, S.R_EYE) if okk[j]]
            hips = [kp[j, 1] for j in (S.L_HIP, S.R_HIP) if okk[j]]
            if okk.sum() >= 5 and head and hips and np.mean(head) < np.mean(hips):
                top, bot = kp[okk, 1].min(), kp[okk, 1].max()
                # nothing of a standing person is far below the feet: a box reaching there holds the floor
                # (a mirror image on a glossy floor, a shadow)
                bx = bx.copy()
                bx[3] = min(bx[3], bot + 0.10 * (bot - top))
                for kn, an in ((S.L_KN, S.L_AN), (S.R_KN, S.R_AN)):
                    if okk[kn] and okk[an] and kp[an, 1] > kp[kn, 1]:
                        q = np.array([kp[kn, 0], kp[an, 1] + 1.1 * (kp[an, 1] - kp[kn, 1])])
                        if 0 <= q[0] < W and 0 <= q[1] < H:
                            mirror.append(q)          # where the knee's mirror image would be
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        bx = np.array([bx[0] - 0.04 * bw, bx[1] - 0.04 * bh, bx[2] + 0.04 * bw, bx[3] + 0.04 * bh])
        bx = np.clip(bx, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
        pos = []
        if kp is not None:
            for j in POS_TORSO:
                if np.isfinite(kp[j, :2]).all() and kp[j, 2] >= 0.35:
                    pos.append(kp[j, :2])
            for j in POS_LIMBS:
                if np.isfinite(kp[j, :2]).all() and kp[j, 2] >= 0.6:
                    pos.append(kp[j, :2])
            sh = [kp[j, :2] for j in (S.L_SH, S.R_SH) if np.isfinite(kp[j, :2]).all()]
            hp = [kp[j, :2] for j in (S.L_HIP, S.R_HIP) if np.isfinite(kp[j, :2]).all()]
            if sh and hp:
                pos.append(0.6 * np.mean(sh, 0) + 0.4 * np.mean(hp, 0))      # chest
        pos = np.array([q for q in pos if bx[0] <= q[0] <= bx[2] and bx[1] <= q[1] <= bx[3]], np.float32).reshape(-1, 2)
        people.append([p, bx, pos, kp, mirror])
    out = []
    for p, bx, pos, kp, mirror in people:
        bh = bx[3] - bx[1]
        neg = list(mirror)
        for q, obx, opos, okp, _m in people:
            if q == p:
                continue
            for pt in opos[:6]:
                if not (bx[0] <= pt[0] <= bx[2] and bx[1] <= pt[1] <= bx[3]):
                    continue
                if len(pos) and np.hypot(*(pos - pt).T).min() < 0.06 * bh:
                    continue                     # too close to this person's own body: conflicting prompt
                neg.append(pt)
        out.append((p, bx, pos, np.array(neg, np.float32).reshape(-1, 2)))
    return out


def _closeup_rect(bx, W, H):
    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
    side = max(bw, bh) * 1.3
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    x0, y0 = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
    x1, y1 = min(float(W), cx + side / 2), min(float(H), cy + side / 2)
    return x0, y0, x1, y1


def _fill_small_holes(hard: np.ndarray, frac: float = 0.002) -> np.ndarray:
    """Close pin holes inside a silhouette (decoder texture), keep real gaps (arm against the waist)."""
    inv = (~hard).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    area = max(1, int(hard.sum()))
    h, w = hard.shape
    out = hard.copy()
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if a < frac * area and x > 0 and y > 0 and x + bw < w and y + bh < h:
            out[lab == i] = True
    return out


HEAD_PULL = 5.0             # logit units: a person's own visible face over another person's overlapping mask


def head_priors(st, t: int, mw: int, mh: int, W: int, H: int, sx: float, sy: float, use_final: bool) -> dict:
    """pid -> (cx, cy, r, w) at matte size: where each person's face is, how big, how sure (visible face points)."""
    out = {}
    kx, ky = mw / W, mh / H
    for p in np.nonzero(st.present[:, t])[0]:
        kp = _kp_for(st, p, t, use_final)
        if kp is None:
            continue
        hd = kp[[S.NOSE, S.L_EYE, S.R_EYE]]
        ok = np.isfinite(hd[:, :2]).all(1) & (np.nan_to_num(hd[:, 2]) >= 0.3)
        if ok.sum() < 2:
            continue
        pts = hd[ok, :2] * [sx * kx, sy * ky]
        c = pts.mean(0)
        ears = [kp[j, :2] * [sx * kx, sy * ky] for j in (S.L_EAR, S.R_EAR)
                if np.isfinite(kp[j, :2]).all() and np.nan_to_num(kp[j, 2]) >= 0.2]
        span = max([float(np.hypot(*(q - c))) for q in list(pts) + ears] + [0.0])
        sh = [kp[j, :2] * [sx * kx, sy * ky] for j in (S.L_SH, S.R_SH) if np.isfinite(kp[j, :2]).all()]
        if len(sh) == 2:
            span = max(span, 0.22 * float(np.hypot(*(sh[0] - sh[1]))))
        r = max(3.0, 1.1 * span)
        wgt = float(np.clip((np.mean(np.nan_to_num(hd[ok, 2])) - 0.3) / 0.4, 0.0, 1.0))
        if wgt > 0:
            out[int(st.pids[p])] = (float(c[0]), float(c[1]), r, wgt)
    return out


DEPTH_PULL = 8.0            # logit units taken from a person whose own depth does not fit the pixel
DEPTH_TAU = 0.10            # depth misfit (share of the people's depth range) at which the pull is full
DEPTH_SIGMA = 0.12          # of sqrt(mask area): how far a person's own depth is carried
CORE_LOGIT = 1.5            # a pixel this far inside a mask counts as that person's solid body
SUPPORT_LOGIT = 1.0         # what a person's skeleton is worth on a pixel another person's mask took
SUPPORT_LIMB = 0.045        # half width of a limb, of the person's height
OWN_CLIP = 6.0              # mask logits are capped at this when owners are compared


def skeleton_support(kp: np.ndarray, height: float, shape) -> np.ndarray:
    """Where a person's body is by their own VISIBLE joints (kp (17,3) in matte px, confidence 0 = not visible):
    limbs as thick lines, the torso as a polygon, the head as a disc. Deliberately thinner than the body."""
    h, w = shape
    img = np.zeros((h, w), np.uint8)
    ok = np.isfinite(kp[:, :2]).all(1) & (np.nan_to_num(kp[:, 2]) >= 0.35)
    th = max(2, int(round(2 * SUPPORT_LIMB * height)))
    P = kp[:, :2]
    pt = lambda j: (int(round(float(P[j, 0]))), int(round(float(P[j, 1]))))
    for a, b in S.LIMB_BONES:
        if ok[a] and ok[b]:
            cv2.line(img, pt(a), pt(b), 255, th)
    tors = [j for j in (S.L_SH, S.R_SH, S.R_HIP, S.L_HIP) if ok[j]]
    if len(tors) >= 3:
        cv2.fillConvexPoly(img, np.round(P[tors]).astype(np.int32), 255)
    hd = [j for j in (S.NOSE, S.L_EYE, S.R_EYE) if ok[j]]
    if hd:
        c = P[hd].mean(0)
        cv2.circle(img, (int(round(float(c[0]))), int(round(float(c[1])))), max(2, int(round(0.05 * height))), 255, -1)
    return img > 0


def depth_fit(L: np.ndarray, depth: np.ndarray, cand: np.ndarray, support: Optional[List[Optional[np.ndarray]]] = None
              ) -> Optional[np.ndarray]:
    """(n,h,w) misfit 0..1 of every person's own depth to the picture's depth, on the pixels two or more people
    can own (`cand`).

    A person's own depth near a pixel = the depth of that person's sure pixels (well inside the mask, no other mask
    stronger, and on the person's own skeleton where one is known), carried over by a normalised blur. A pixel whose
    depth is the front person's belongs to the front person, wherever the masks leak: a back mask that takes in the
    occluder, a front mask that spills on the body behind, an arm in front of a torso that is itself behind. People at
    the same depth (side by side, touching) get the same fit: nothing changes."""
    n, h, w = L.shape
    over = cand.sum(0) >= 2
    if n < 2 or not over.any():
        return None
    union = (L > 0).any(0)
    if union.sum() < 32:
        return None
    lo, hi = np.percentile(depth[union], [5, 95])
    rng = float(max(hi - lo, 1e-6))
    Lmax = L.max(0)
    fit = np.zeros((n, h, w), np.float32)
    for i in range(n):
        sure = (L[i] > CORE_LOGIT) & (L[i] >= Lmax)
        if support is not None and support[i] is not None and int((sure & support[i]).sum()) >= 16:
            sure &= support[i]
        if int(sure.sum()) < 16:
            fit[i] = 0.5                         # nothing sure of this person: says nothing
            continue
        ex = sure.astype(np.float32)
        area = float((L[i] > 0).sum())
        sig = float(np.clip(DEPTH_SIGMA * np.sqrt(area), 3.0, 0.08 * max(h, w)))
        den = cv2.GaussianBlur(ex, (0, 0), sig)
        num = cv2.GaussianBlur(ex * depth, (0, 0), sig)
        ok = den > 0.02
        dexp = np.where(ok, num / np.maximum(den, 1e-6), float(np.median(depth[sure])))
        fit[i] = np.clip(np.abs(depth - dexp) / rng / DEPTH_TAU, 0.0, 1.0)
    fit[:, ~over] = 0.0
    return fit


def combine(logits: List[Tuple[int, np.ndarray]], rvm: Optional[np.ndarray], shape,
            heads: Optional[dict] = None, depth: Optional[np.ndarray] = None,
            skel: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per person mask logits at matte size -> (alpha uint8, labels uint8 = pid + 1).

    Who owns a person pixel:
      * a person whose mask covers it, or whose visible face is there, or (with `depth`) whose own skeleton
        (`skel`: pid -> (kp (17,3) matte px with confidence 0 for joints that are not visible, height px)) runs
        through it, can own it;
      * with `depth` (matte size, larger = nearer) the person whose own depth fits the pixel is preferred: the
        person in front gets what is in front, pixel by pixel (depth_fit);
      * around each visible face its own person is pulled up and everyone else down;
      * otherwise the larger (slightly smoothed) logit.
    Two near-equal logits used to split a face into stripes of both owners (the decoder's texture), and a mask that
    took in the face of the person in front won it outright on some frames, so that face lost its retouch."""
    h, w = shape
    alpha = np.zeros((h, w), np.float32)
    lab = np.zeros((h, w), np.uint8)
    if logits:
        sig = max(1.0, 0.002 * max(h, w))
        L = np.stack([cv2.GaussianBlur(l, (0, 0), sig) for _, l in logits])
        n = len(logits)
        bumps = {}
        if heads and n > 1:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            for pid, (cx, cy, r, wg) in heads.items():
                x0, x1 = int(max(0, cx - 3 * r)), int(min(w, cx + 3 * r + 1))
                y0, y1 = int(max(0, cy - 3 * r)), int(min(h, cy + 3 * r + 1))
                if x1 <= x0 or y1 <= y0:
                    continue
                b = np.zeros((h, w), np.float32)
                b[y0:y1, x0:x1] = wg * np.exp(-((xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2) / (2 * r * r))
                bumps[pid] = b
        Lmax = L.max(0)
        hard = Lmax > 0
        cand = L > 0
        La = np.clip(L, -OWN_CLIP, OWN_CLIP) if n > 1 else L
        support = None
        if depth is not None and n > 1:
            if skel:
                support = []
                for i, (pid, _l) in enumerate(logits):
                    sp = None
                    if pid in skel and skel[pid][0] is not None:
                        sp = skeleton_support(skel[pid][0], float(skel[pid][1]), (h, w))
                        claim = sp & hard & (L[i] <= 0)          # on a pixel another person's mask took
                        if claim.any():
                            cand[i] |= claim
                            La[i] = np.where(claim, SUPPORT_LOGIT, La[i])
                    support.append(sp)
            dfit = depth_fit(L, depth, cand, support)
            if dfit is not None:
                La = La - DEPTH_PULL * dfit
        if bumps:
            tot = sum(bumps.values())
            for i, (pid, _l) in enumerate(logits):
                own = bumps.get(pid)
                others = tot - own if own is not None else tot
                # only where this mask is in play at all (a pull never makes a person out of background)
                inplay = L[i] > -6
                adj = HEAD_PULL * ((own if own is not None else 0.0) - np.clip(others, 0, 1))
                La[i] = La[i] + np.where(inplay, adj, 0.0)
                if own is not None:
                    cand[i] |= inplay & (own > 0.3)
        arg = np.where(cand.any(0), np.where(cand, La, -1e9).argmax(0), La.argmax(0))
        pids = np.array([pid for pid, _ in logits])
        solid = _fill_small_holes(hard)
        a = np.clip(0.5 + Lmax / 3.0, 0.0, 1.0)
        inner = cv2.erode(solid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        a = np.where(inner | (solid & ~hard), 1.0, a)
        alpha = a.astype(np.float32)
        lab = np.where(solid, pids[arg] + 1, 0).astype(np.uint8)
        if rvm is not None:
            r = max(3, int(0.025 * max(h, w)))
            near = cv2.dilate(solid.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            near = cv2.GaussianBlur(near.astype(np.float32) / 255.0, (0, 0), r / 2.0)
            alpha = np.maximum(alpha, rvm * (1.0 - np.clip(near, 0, 1)))
    elif rvm is not None:
        alpha = rvm
    return np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8), lab


def skeletons(st, t: int, mw: int, mh: int, W: int, H: int, sx: float, sy: float, use_final: bool) -> dict:
    """pid -> (kp (17,3) in matte px, confidence 0 where the joint is not visible; person height in matte px)."""
    out = {}
    kx, ky = sx * mw / W, sy * mh / H
    for p in np.nonzero(st.present[:, t])[0]:
        kp = _kp_for(st, p, t, use_final)
        if kp is None:
            continue
        kp = kp.copy()
        kp[:, 0] *= kx
        kp[:, 1] *= ky
        box = st.boxes[p, t]
        hgt = float((box[3] - box[1]) * ky) if np.isfinite(box).all() else float("nan")
        sc = S.person_scale(kp)
        if np.isfinite(sc):
            hgt = max(sc, 0.0) if not np.isfinite(hgt) else max(hgt, sc)
        if np.isfinite(hgt) and hgt > 4:
            out[int(st.pids[p])] = (kp, hgt)
    return out


def _canvas(lg: np.ndarray, rect, W: int, H: int, mw: int, mh: int) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = rect
    canvas = np.full((mh, mw), -20.0, np.float32)
    mx0, my0 = int(round(x0 * mw / W)), int(round(y0 * mh / H))
    mx1, my1 = int(round(x1 * mw / W)), int(round(y1 * mh / H))
    if mx1 - mx0 < 2 or my1 - my0 < 2:
        return None
    canvas[my0:my1, mx0:mx1] = cv2.resize(lg, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_LINEAR)
    return canvas


def decode_frame(seg, rgb, st, t: int, sx: float, sy: float, W: int, H: int, mw: int, mh: int,
                 use_final: bool) -> dict:
    """First decode of everybody on frame t: pid -> dict(p, bx, pos, neg, enc, canvas (matte size mask logits))."""
    pr = prompts(st, t, sx, sy, W, H, use_final)
    dec = {}
    if not pr:
        return dec
    full = None
    small = sorted([i for i, x in enumerate(pr) if (x[1][3] - x[1][1]) < SMALL_PERSON * H],
                   key=lambda i: pr[i][1][3] - pr[i][1][1])[:MAX_CLOSEUPS]
    for i, (p, bx, pos, neg) in enumerate(pr):
        if i in small:
            enc = seg.encode(rgb, _closeup_rect(bx, W, H))
        else:
            if full is None:
                full = seg.encode(rgb)
            enc = full
        lg, q = seg.segment(enc, bx, pos, neg)
        if lg is None:
            continue
        cv = _canvas(lg, enc["rect"], W, H, mw, mh)
        if cv is None:
            continue
        dec[int(st.pids[p])] = dict(p=p, bx=bx, pos=pos, neg=neg, enc=enc, canvas=cv)
    return dec


def run(st, proxy, matte, seglab, a: int, b: int, seg_key: str, use_final: bool,
        progress: Optional[Callable[[float, str], None]] = None, cancel: Optional[Callable[[], bool]] = None,
        matter=None) -> None:
    """Fill the matte (and the person labels) for frames a..b."""
    from .infer import make_segmenter
    try:
        seg = make_segmenter(seg_key)
    except Exception as e:
        if seg_key != "seg_sam31":
            raise
        import sys
        sys.stderr.write(f"SAM 3.1 could not be loaded ({str(e)[:200]}), using SAM 2.1\n")
        seg = make_segmenter("seg_sam2_b" if "max" in str(seg_key) else "seg_sam2_s")
    mw, mh = matte.w, matte.h
    W, H = proxy.w, proxy.h
    sx, sy = proxy.sx, proxy.sy
    depth_net = None
    if np.count_nonzero(st.present[:, a:b].sum(0) >= 2):
        try:
            from .infer import DepthEstimator
            depth_net = DepthEstimator()
        except Exception as e:                  # depth is a refinement: never stop the matte for it
            import sys
            sys.stderr.write(f"depth model could not be loaded ({str(e)[:200]}): owners by mask only\n")
    mm = matte.open_rw()
    lm = seglab.open_rw()
    if matter is not None:
        matter.reset()
        for t in range(a, min(b, a + 4)):
            matter(cv2.resize(analysis_view(np.asarray(proxy.frame(t))), (mw, mh), interpolation=cv2.INTER_AREA))
        matter.reset()
    try:
        for t in range(a, b):
            if cancel and cancel():
                break
            # dark or colour-lit frames: what SAM and RVM see only (never the output)
            rgb = analysis_view(np.asarray(proxy.frame(t)))
            rvm = None
            if matter is not None:
                rvm = matter(cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_AREA))
            dec = decode_frame(seg, rgb, st, t, sx, sy, W, H, mw, mh, use_final)
            logits = [(pid, d["canvas"]) for pid, d in dec.items()]
            depth = skel = None
            if depth_net is not None and len(logits) > 1:
                try:
                    depth = depth_net(rgb, (mw, mh))
                    skel = skeletons(st, t, mw, mh, W, H, sx, sy, use_final)
                except Exception:
                    depth = skel = None
            al, lb = combine(logits, rvm, (mh, mw), head_priors(st, t, mw, mh, W, H, sx, sy, use_final), depth, skel)
            mm[t] = al
            lm[t] = lb
            if progress:
                progress((t - a + 1) / max(1, b - a), f"SAM 인물 분할 {t - a + 1}/{b - a}")
    finally:
        mm.flush()
        lm.flush()
        del mm, lm
        matte.close()
        seglab.close()
