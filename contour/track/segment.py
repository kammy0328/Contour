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


HAIR = True                 # loose strands from RVM around visible heads (outside the mask only)
HAIR_REACH = 3.0            # head radii around a visible head that count as hair
HAIR_OUT = 0.6              # head radii outside the mask a strand may reach
RVM_TRUST = 0.85            # RVM's mean alpha on the person's sure body for its strands to be used


def refine_hair(alpha: np.ndarray, lab: np.ndarray, rvm: np.ndarray, heads: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Loose strands around visible heads. SAM's masks are 288 px over the frame: strands that fly off the head are
    not in them. RVM's alpha (on the frame at matte size, good at hair) has them; it is taken outside the mask, in
    a band along the head, only for a person RVM clearly holds (high alpha on their body), and never lowers
    anything: the mask, the face and the body stay as they are. Strands take the label of the head's person."""
    h, w = alpha.shape
    out, lab = alpha.copy(), lab.copy()
    cap = 0.06 * max(h, w)                      # close-ups: a band of a few % of the frame, not of the head
    for pid, (cx, cy, r, wg) in heads.items():
        if wg < 0.4 or r < 3:
            continue
        r = min(r, cap)
        R = HAIR_REACH * r
        x0, x1 = int(max(0, cx - R)), int(min(w, cx + R + 1))
        y0, y1 = int(max(0, cy - R)), int(min(h, cy + R + 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        own = lab[sl] == pid + 1
        if own.sum() < 20:
            continue
        core = cv2.erode(own.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
        if core.sum() < 20 or float(np.mean(rvm[sl][core])) < RVM_TRUST:
            continue
        ro = max(2, int(HAIR_OUT * r))
        band = cv2.dilate(own.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1))).astype(bool)
        band &= ~own & (lab[sl] == 0)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        near = np.clip(1.5 - np.hypot(xx - cx, yy - cy) / R * 1.5, 0.0, 1.0)
        add = np.where(band, rvm[sl] * near, 0.0)
        out[sl] = np.maximum(out[sl], add)
        lab[sl][band & (add >= 0.25)] = np.uint8(pid + 1)
    return out, lab


def combine(logits: List[Tuple[int, np.ndarray]], rvm: Optional[np.ndarray], shape,
            heads: Optional[dict] = None, extra: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per person mask logits at matte size -> (alpha uint8, labels uint8 = pid + 1).

    Where two people's masks overlap the owner is the larger logit. Two near-equal logits used to split a face
    into stripes of both owners (the decoder's texture), and a mask that took in the face of the person in front
    won it outright on some frames, so that face lost its retouch on those frames. Logits are now smoothed a
    little, and around each visible face its own person is pulled up and everyone else down."""
    h, w = shape
    alpha = np.zeros((h, w), np.float32)
    lab = np.zeros((h, w), np.uint8)
    if logits:
        sig = max(1.0, 0.002 * max(h, w))
        L = np.stack([cv2.GaussianBlur(l, (0, 0), sig) for _, l in logits])
        La = L                                  # for who owns a pixel; L alone says whether it is a person
        bumps = {}
        if heads and len(logits) > 1:
            La = L.copy()
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            for pid, (cx, cy, r, wg) in heads.items():
                x0, x1 = int(max(0, cx - 3 * r)), int(min(w, cx + 3 * r + 1))
                y0, y1 = int(max(0, cy - 3 * r)), int(min(h, cy + 3 * r + 1))
                if x1 <= x0 or y1 <= y0:
                    continue
                b = np.zeros((h, w), np.float32)
                b[y0:y1, x0:x1] = wg * np.exp(-((xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2) / (2 * r * r))
                bumps[pid] = b
            if bumps:
                tot = sum(bumps.values())
                for i, (pid, _l) in enumerate(logits):
                    own = bumps.get(pid)
                    others = tot - own if own is not None else tot
                    # only where this mask is in play at all (a pull never makes a person out of background)
                    inplay = L[i] > -6
                    adj = HEAD_PULL * ((own if own is not None else 0.0) - np.clip(others, 0, 1))
                    La[i] = np.where(inplay, L[i] + adj, L[i])
        Lmax = L.max(0)
        arg = La.argmax(0)
        pids = np.array([pid for pid, _ in logits])
        hard = Lmax > 0
        solid = _fill_small_holes(hard)
        a = np.clip(0.5 + Lmax / 3.0, 0.0, 1.0)
        inner = cv2.erode(solid.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        a = np.where(inner | (solid & ~hard), 1.0, a)
        alpha = a.astype(np.float32)
        lab = np.where(solid, pids[arg] + 1, 0).astype(np.uint8)
        if HAIR and rvm is not None and heads:
            try:
                alpha, lab = refine_hair(alpha, lab, rvm, heads)
            except Exception:
                import traceback
                traceback.print_exc()
        if rvm is not None:
            r = max(3, int(0.025 * max(h, w)))
            near = cv2.dilate(solid.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            near = cv2.GaussianBlur(near.astype(np.float32) / 255.0, (0, 0), r / 2.0)
            alpha = np.maximum(alpha, rvm * (1.0 - np.clip(near, 0, 1)))
    elif rvm is not None:
        alpha = rvm
    # people the finder saw that tracking does not follow (extras): in the matte, and marked UNTRACKED in the labels
    # so that nobody's retouch takes them (the label lookup gives an unlabelled matte pixel to the nearest skeleton)
    for e in (extra or []):
        e, untracked = e if isinstance(e, tuple) else (e, True)
        ea = np.clip(0.5 + e / 3.0, 0.0, 1.0)
        alpha = np.maximum(alpha, np.where(lab == 0, ea, 0.0))
        if untracked:
            lab = np.where((lab == 0) & (e > 0), np.uint8(UNTRACKED), lab).astype(np.uint8)
    return np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8), lab


def _canvas(lg: np.ndarray, rect, W: int, H: int, mw: int, mh: int) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = rect
    canvas = np.full((mh, mw), -20.0, np.float32)
    mx0, my0 = int(round(x0 * mw / W)), int(round(y0 * mh / H))
    mx1, my1 = int(round(x1 * mw / W)), int(round(y1 * mh / H))
    if mx1 - mx0 < 2 or my1 - my0 < 2:
        return None
    canvas[my0:my1, mx0:mx1] = cv2.resize(lg, (mx1 - mx0, my1 - my0), interpolation=cv2.INTER_LINEAR)
    return canvas


UNTRACKED = 255             # label of a person the finder saw that tracking does not follow
REPOSE_MIN = 0.40           # finder score an instance needs to be a body a borrowed skeleton is re-estimated on
EXTRA_MIN_H = 0.07          # an untracked person this tall (share of the frame) can become a tracked one
EXTRA_OVERLAP = 0.40        # ... unless this much of it is on tracked people's masks (a part of one of them)
DEBUG: Optional[list] = None  # a list here collects why borrowed skeletons were (not) re-estimated (tests)
FIND_MIN = 0.30             # a finder instance this sure can be a tracked person (tracking vouches for it)
FIND_SURE = 0.50            # this sure, it is a person even when tracking does not follow it
MATCH_MIN = 0.35            # joints inside + box overlap a person needs with an instance
GHOST_KEEP = 0.25           # a click mask of which less than this is left outside the found people is dropped


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ua = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)
    return ix * iy / ua


def _inside(pos, inst) -> float:
    """Share of a person's visible joints that fall in a finder instance's mask."""
    if len(pos) == 0:
        return 0.0
    x0, y0, x1, y1 = inst["rect"]
    lg = inst["lg"]
    gx = ((pos[:, 0] - x0) / max(x1 - x0, 1e-6) * lg.shape[1]).astype(int)
    gy = ((pos[:, 1] - y0) / max(y1 - y0, 1e-6) * lg.shape[0]).astype(int)
    ok = (gx >= 0) & (gx < lg.shape[1]) & (gy >= 0) & (gy < lg.shape[0])
    return float((lg[gy[ok], gx[ok]] > 0).sum()) / len(pos)


def _match_score(bx, pos, inst) -> float:
    """How well a finder instance is this tracked person: the share of the person's visible joints inside the
    instance's mask, and the overlap of the two boxes."""
    biou = _box_iou(bx, inst["box"])
    if len(pos) == 0:
        return 0.7 * biou
    return 0.6 * _inside(pos, inst) + 0.4 * biou


def _merged(pr, insts) -> dict:
    """instance index -> [indices in pr] for an instance that holds two or more tracked people (the finder took a
    tight pair for one person): each of them has most visible joints inside it, has no better instance of their own,
    and is clearly smaller than it. Such an instance cannot say who is who: its people are told apart by clicks,
    inside it."""
    if len(pr) < 2 or not insts:
        return {}
    best = []
    for p, bx, pos, neg in pr:
        sc = [_match_score(bx, pos, ins) for ins in insts]
        best.append(int(np.argmax(sc)))
    out = {}
    for k, ins in enumerate(insts):
        b_ = ins["box"]
        area = max(1e-6, float((b_[2] - b_[0]) * (b_[3] - b_[1])))
        who = [i for i, (p, bx, pos, neg) in enumerate(pr)
               if best[i] == k and len(pos) >= 3 and _inside(pos, ins) >= 0.6
               and float((bx[2] - bx[0]) * (bx[3] - bx[1])) < 0.75 * area]
        if len(who) >= 2:
            out[k] = who
    return out


def _assign(pr, insts) -> dict:
    """index in pr -> index in insts, one instance per person at most, best total score."""
    if not pr or not insts:
        return {}
    S_ = np.zeros((len(pr), len(insts)), np.float32)
    for i, (p, bx, pos, neg) in enumerate(pr):
        for k, ins in enumerate(insts):
            S_[i, k] = _match_score(bx, pos, ins)
    out = {}
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-S_)
        for i, k in zip(r, c):
            if S_[i, k] >= MATCH_MIN:
                out[int(i)] = int(k)
    except Exception:
        used = set()
        for i, k in sorted(((i, k) for i in range(len(pr)) for k in range(len(insts))), key=lambda ik: -S_[ik]):
            if S_[i, k] < MATCH_MIN:
                break
            if i not in out and k not in used:
                out[i] = k
                used.add(k)
    return out


def record(inst, W: int, H: int, side: int = 128) -> Optional[dict]:
    """A compact copy of a finder instance: box (proxy px), score, mask over the box (bool, long side <= side)."""
    x0, y0, x1, y1 = [float(v) for v in inst["box"]]
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    rx0, ry0, rx1, ry1 = inst["rect"]
    lg = inst["lg"]
    k = side / max(x1 - x0, y1 - y0)
    cw, ch = max(4, int(round((x1 - x0) * k))), max(4, int(round((y1 - y0) * k)))
    xs = ((x0 + (np.arange(cw) + 0.5) * (x1 - x0) / cw) - rx0) / max(rx1 - rx0, 1e-6) * lg.shape[1] - 0.5
    ys = ((y0 + (np.arange(ch) + 0.5) * (y1 - y0) / ch) - ry0) / max(ry1 - ry0, 1e-6) * lg.shape[0] - 0.5
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    m = cv2.remap(lg.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-20.0) > 0
    if not m.any():
        return None
    return dict(box=np.array([x0, y0, x1, y1], np.float32), score=float(inst["score"]), mask=m)


def record_mask(rec: dict) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """(x0, y0, x1, y1) int proxy px and the bool mask at that box's size."""
    x0, y0, x1, y1 = [int(round(float(v))) for v in rec["box"]]
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    m = cv2.resize(rec["mask"].astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0
    return (x0, y0, x1, y1), m


def _grid(ins, W: int, H: int, gw: int = 160) -> np.ndarray:
    """An instance's mask on a coarse grid over the whole frame (for overlaps between instances)."""
    gh = max(4, int(round(gw * H / W)))
    cv = _canvas(ins["lg"], ins["rect"], W, H, gw, gh)
    return np.zeros((gh, gw), bool) if cv is None else cv > 0


def _borrowed(st, t, pr, insts, assigned, merged, sx, sy, use_final, W: int, H: int) -> List[Tuple[int, int]]:
    """Tracks whose skeleton is on someone else (a face found behind a dancer, given the dancer's body by the pose
    model; a track that stayed on the body it followed while its box and face went to the person behind): two
    tracks with most of their joints inside one person's instance. The one whose box fits that instance is its
    owner; the other one, if its own box is on another instance (not a part of the first), is a borrower.
    -> [(index in pr, instance index)]. `assigned` is updated: the borrower takes that instance, the owner the one
    the joints are in."""
    out = []
    if len(pr) < 2 or not insts:
        return out
    inside = np.zeros((len(pr), len(insts)), np.float32)
    for i, (p, bx, pos, neg) in enumerate(pr):
        # all the sure joints, not only those in the box (the box may have gone over to someone else)
        kp = _kp_for(st, p, t, use_final)
        if kp is None:
            continue
        ok = np.isfinite(kp[:, :2]).all(1) & (np.nan_to_num(kp[:, 2]) >= 0.3)
        if ok.sum() < 4:
            continue
        jp = (kp[ok, :2] * [sx, sy]).astype(np.float32)
        for k, ins in enumerate(insts):
            inside[i, k] = _inside(jp, ins)
    grids = {}

    def grid(k):
        if k not in grids:
            grids[k] = _grid(insts[k], W, H)
        return grids[k]

    for kS in range(len(insts)):
        if kS in merged:
            continue
        holders = [i for i in range(len(pr)) if inside[i, kS] >= 0.5]
        if len(holders) < 2:
            continue
        fit = {i: _box_iou(pr[i][1], insts[kS]["box"]) for i in holders}
        owner = max(holders, key=lambda i: (fit[i], inside[i, kS]))
        for borrower in holders:
            if borrower == owner or any(borrower == q for q, _k in out):
                continue
            bbx = pr[borrower][1]
            cur = assigned.get(borrower)
            used = set(assigned.values()) | set(merged) | {kS}
            best, kb = 0.25, -1
            for k, ins in enumerate(insts):
                if k == kS or ins["score"] < REPOSE_MIN or (k in used and k != cur):
                    continue
                v = _box_iou(bbx, ins["box"])
                if v <= best:
                    continue
                gk = grid(k)
                # a part of the person the joints are on (an arm, the hair) is not somebody else
                if gk.any() and float((gk & grid(kS)).sum()) >= 0.5 * float(gk.sum()):
                    continue
                best, kb = v, k
            if DEBUG is not None:
                DEBUG.append(dict(t=t, inst=kS, owner=int(st.pids[pr[owner][0]]), borrower=int(st.pids[pr[borrower][0]]),
                                  inside=round(float(inside[borrower, kS]), 2), kb=kb,
                                  cands=[(k, round(_box_iou(bbx, ins["box"]), 2), round(ins["score"], 2)) for k, ins in enumerate(insts)]))
            if kb < 0:
                continue
            assigned[borrower] = kb
            holders_ = {v for q, v in assigned.items() if q != owner}
            if assigned.get(owner) is None and kS not in holders_:
                assigned[owner] = kS
            out.append((borrower, kb))
    return out


def decode_frame(seg, rgb, st, t: int, sx: float, sy: float, W: int, H: int, mw: int, mh: int,
                 use_final: bool, finder=None, heads: Optional[dict] = None, info: Optional[dict] = None,
                 repose: Optional[Callable] = None) -> Tuple[dict, List[np.ndarray]]:
    """Everybody on frame t: (pid -> dict(p, bx, pos, neg, canvas (matte size mask logits), src), extra).

    With `finder` (SAM 3.1's "person" finder): the frame is first split into the people in it, each a mask of their
    own, by a model trained to tell overlapping people apart (who is in front, whose arm it is). Every tracked person
    takes the instance their visible joints and box fall in (src "find"); people small in the frame take theirs from
    a close-up. Only a tracked person the finder has nothing for is decoded from clicks as before (src "click"), and
    then never over a found person; one that is left with next to nothing (a track sitting on someone else) is
    dropped. extra: the people the finder is sure of that tracking does not follow.

    info (a dict, filled when given): "repose": [(pid, p, record)] tracks that carried someone else's skeleton and
    now have the instance their own box is on (the caller re-estimates their pose there); "extras": [record] the
    people nobody tracks, tall enough and not a part of a tracked person."""
    pr = prompts(st, t, sx, sy, W, H, use_final)
    dec, extra = {}, []
    if not pr:
        return dec, extra
    full = seg.encode(rgb)
    insts = []
    if finder is not None:
        try:
            insts = finder.find(full, FIND_MIN)
        except Exception:
            insts = []
    merged = _merged(pr, insts)
    shared = {i: k for k, who in merged.items() for i in who}        # person -> the instance they share
    if merged:
        keep = [k for k in range(len(insts)) if k not in merged]
        sub = _assign(pr, [insts[k] for k in keep])
        assigned = {i: keep[k] for i, k in sub.items() if i not in shared}
    else:
        assigned = _assign(pr, insts)
    if heads and insts:
        # a track whose body joints sit on someone else (two people tight together) but whose own face is seen:
        # the instance that face is in, if nobody took it
        free = [k for k in range(len(insts)) if k not in assigned.values() and k not in merged]
        for i, (p, bx, pos, neg) in enumerate(pr):
            hp = heads.get(int(st.pids[p]))
            if i in assigned or i in shared or not hp or hp[3] < 0.5 or not free:
                continue
            cx, cy = hp[0] * W / mw, hp[1] * H / mh
            best = None
            for k in free:
                x0, y0, x1, y1 = insts[k]["rect"]
                lg = insts[k]["lg"]
                gx = int((cx - x0) / max(x1 - x0, 1e-6) * lg.shape[1])
                gy = int((cy - y0) / max(y1 - y0, 1e-6) * lg.shape[0])
                if 0 <= gx < lg.shape[1] and 0 <= gy < lg.shape[0] and lg[gy, gx] > 0 \
                        and (best is None or insts[k]["score"] > insts[best]["score"]):
                    best = k
            if best is not None:
                assigned[i] = best
                free.remove(best)
    reposed = {}
    if repose is not None and insts and len(pr) > 1:
        before = dict(assigned)
        for i, k in _borrowed(st, t, pr, insts, assigned, merged, sx, sy, use_final, W, H):
            rec = record(insts[k], W, H)
            ok = False
            if rec is not None:
                try:
                    ok = bool(repose(t, int(pr[i][0]), rgb, rec))
                except Exception:
                    import traceback
                    traceback.print_exc()
            if DEBUG is not None:
                DEBUG.append(dict(t=t, repose=int(st.pids[pr[i][0]]), inst=k, ok=ok))
            if ok:
                reposed[i] = k
                if info is not None:
                    info.setdefault("repose", []).append((int(st.pids[pr[i][0]]), int(pr[i][0]), rec))
            else:                                   # no pose of its own there: as before
                if before.get(i) is None:
                    assigned.pop(i, None)
                else:
                    assigned[i] = before[i]
        if reposed:
            # the owners keep what the skeleton is in; drop any instance given twice by the steps above
            seen = {}
            for i in sorted(assigned, key=lambda q: (q not in reposed, q)):
                k = assigned[i]
                if k in seen:
                    assigned.pop(i)
                else:
                    seen[k] = i
    small = sorted([i for i, x in enumerate(pr) if (x[1][3] - x[1][1]) < SMALL_PERSON * H and i not in reposed],
                   key=lambda i: pr[i][1][3] - pr[i][1][1])[:MAX_CLOSEUPS]
    taken = set(assigned.values()) | set(merged)
    inst_cv = [None] * len(insts)

    def icv(k):
        if inst_cv[k] is None:
            inst_cv[k] = _canvas(insts[k]["lg"], insts[k]["rect"], W, H, mw, mh)
        return inst_cv[k]

    for i, (p, bx, pos, neg) in enumerate(pr):
        pid = int(st.pids[p])
        cv, src = None, None
        enc = full
        if i in small:
            enc = seg.encode(rgb, _closeup_rect(bx, W, H))
            if finder is not None:
                try:
                    cins = finder.find(enc, FIND_MIN)
                except Exception:
                    cins = []
                best, bk = MATCH_MIN, -1
                for k, ins in enumerate(cins):
                    sc = _match_score(bx, pos, ins)
                    if i in assigned:                      # the same person the whole frame gave this track
                        sc *= 0.5 + 0.5 * min(1.0, _box_iou(ins["box"], insts[assigned[i]]["box"]) / 0.5)
                    if sc > best:
                        best, bk = sc, k
                if bk >= 0:
                    cv, src = _canvas(cins[bk]["lg"], cins[bk]["rect"], W, H, mw, mh), "find"
        if cv is None and i in assigned:
            cv, src = icv(assigned[i]), "find"
        if cv is None:
            lg, q = seg.segment(enc, bx, pos, neg)
            if lg is None:
                continue
            cv, src = _canvas(lg, enc["rect"], W, H, mw, mh), "click"
            if cv is not None and i in shared and icv(shared[i]) is not None:
                cv, src = np.minimum(cv, icv(shared[i]) + 2.0), "pair"      # told apart by clicks, inside the pair's mask
        if cv is None:
            continue
        dec[pid] = dict(p=p, bx=bx, pos=pos, neg=neg, canvas=cv, src=src)
    if insts:
        # a click mask never over a found person (someone else's, or one nobody tracks)
        found = [d["canvas"] for d in dec.values() if d["src"] in ("find", "pair")]
        fmask = np.max(np.stack(found), 0) > 0 if found else None
        for k, ins in enumerate(insts):
            if k not in taken and ins["score"] >= FIND_SURE and icv(k) is not None:
                # an instance a tracked person's box or joints point to is that person (or a part of them) even
                # when the matching gave them something else: in the matte, but not marked as nobody's
                tracked = any(_box_iou(bx, ins["box"]) >= 0.5 or (len(pos) >= 3 and _inside(pos, ins) >= 0.6)
                              for p_, bx, pos, neg in pr)
                extra.append((icv(k), not tracked))
                if info is not None and not tracked and (ins["box"][3] - ins["box"][1]) >= EXTRA_MIN_H * H:
                    m = icv(k) > 0
                    n_ = int(m.sum())
                    if n_ and (fmask is None or float((m & fmask).sum()) <= EXTRA_OVERLAP * n_):
                        rec = record(ins, W, H)
                        if rec is not None:
                            info.setdefault("extras", []).append(rec)
        others = found + [e for e, _u in extra]
        if others:
            omax = np.max(np.stack(others), 0)
            for pid in list(dec):
                d = dec[pid]
                if d["src"] != "click":
                    continue
                before = int((d["canvas"] > 0).sum())
                cut = np.minimum(d["canvas"], -omax)
                hp = heads.get(pid) if heads else None
                if hp and hp[3] >= 0.5:
                    # this track's own face is seen (two tracks on one person, or a face the finder gave to the
                    # body in front): around it the click mask stays, so the face keeps its owner and its retouch
                    yy, xx = np.ogrid[0:mh, 0:mw]
                    disc = (xx - hp[0]) ** 2 + (yy - hp[1]) ** 2 <= (1.5 * hp[2]) ** 2
                    d["canvas"] = np.where(disc, d["canvas"], cut)
                    continue
                d["canvas"] = cut
                if before and int((d["canvas"] > 0).sum()) < GHOST_KEEP * before:
                    del dec[pid]
    return dec, extra


def run(st, proxy, matte, seglab, a: int, b: int, seg_key: str, use_final: bool,
        progress: Optional[Callable[[float, str], None]] = None, cancel: Optional[Callable[[], bool]] = None,
        matter=None, repose: Optional[Callable] = None, collect: Optional[dict] = None) -> dict:
    """Fill the matte (and the person labels) for frames a..b. Returns pid -> {"find", "pair", "click", "none"}:
    on how many frames the person's mask came from the finder, from clicks inside a pair, from clicks alone, or was
    left out (a track the finder has nobody for is not a person: a shadow, a reflection).

    repose(t, p, rgb, record) -> bool: called for a track that carried someone else's skeleton, to estimate its
    pose again on the instance it now has (analysis only). collect: filled with "src" {t: {pid: source}} and
    "extras" {t: [record]} (the people nobody tracks) for the caller's fixes after the matte."""
    stats: dict = {}
    from .infer import make_segmenter
    try:
        seg = make_segmenter(seg_key)
    except Exception as e:
        if seg_key != "seg_sam31":
            raise
        import sys
        sys.stderr.write(f"SAM 3.1 could not be loaded ({str(e)[:200]}), using SAM 2.1\n")
        seg = make_segmenter("seg_sam2_b" if "max" in str(seg_key) else "seg_sam2_s")
    finder = None
    if type(seg).__name__ == "Segmenter31":
        try:
            from .infer import Finder31
            finder = Finder31()
        except Exception as e:                  # the finder is a refinement: without it, clicks as before
            import sys
            sys.stderr.write(f"SAM 3.1 finder could not be loaded ({str(e)[:200]}): masks from tracking prompts only\n")
    mw, mh = matte.w, matte.h
    W, H = proxy.w, proxy.h
    sx, sy = proxy.sx, proxy.sy
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
            small = cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_AREA)
            if matter is not None:
                rvm = matter(small)
            heads = head_priors(st, t, mw, mh, W, H, sx, sy, use_final)
            info = {} if (finder is not None and (repose is not None or collect is not None)) else None
            dec, extra = decode_frame(seg, rgb, st, t, sx, sy, W, H, mw, mh, use_final, finder, heads, info,
                                      repose if finder is not None else None)
            if info and info.get("repose"):
                heads = head_priors(st, t, mw, mh, W, H, sx, sy, use_final)
                if collect is not None:
                    collect.setdefault("reposed", []).extend((t, pid) for pid, p, rec in info["repose"])
            logits = [(pid, d["canvas"]) for pid, d in dec.items()]
            if finder is not None:
                fr = {}
                for p in np.nonzero(st.present[:, t])[0]:
                    pid = int(st.pids[p])
                    s_ = stats.setdefault(pid, {"find": 0, "pair": 0, "click": 0, "none": 0})
                    src = dec[pid]["src"] if pid in dec else "none"
                    s_[src] += 1
                    fr[pid] = src
                if collect is not None:
                    collect.setdefault("src", {})[t] = fr
                    collect.setdefault("extras", {})[t] = list((info or {}).get("extras", []))
            al, lb = combine(logits, rvm, (mh, mw), heads, extra)
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
    return stats


def stats_from(src: dict) -> dict:
    """pid -> counts per mask source, from per-frame sources {t: {pid: source}}."""
    out: dict = {}
    for fr in (src or {}).values():
        for pid, s_ in fr.items():
            c = out.setdefault(int(pid), {"find": 0, "pair": 0, "click": 0, "none": 0})
            c[s_] = c.get(s_, 0) + 1
    return out


def unconfirmed(stats: dict, min_frames: int = 8, share: float = 0.2) -> set:
    """Tracked people the finder saw as a person on fewer than `share` of their frames."""
    out = set()
    for pid, s_ in (stats or {}).items():
        n = sum(s_.values())
        if n >= min_frames and (s_["find"] + s_["pair"]) < share * n:
            out.add(int(pid))
    return out
