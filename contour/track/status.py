"""Beauty status of a cut, decided from its analysis.

none   리터칭 불필요   nobody, or nobody whose face is seen (back, far away, a hand over it all along)
small  얼굴 작음       faces seen, but all smaller than SMALL_FACE px wide (a retouch would not show)
good   적용 양호       every retouched face followed most of the time with confident tracking
bad    인식 불량       a retouched face lost often, tracking unsure, or (transitions) two shots mixed

The user can override it (status["user"]) and mark the cut as checked (status["reviewed"]).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

SMALL_FACE = 60          # px, face width (jaw to jaw) on a 1920 wide source (scaled for larger ones)
FACE_OK = 0.8            # share of a person's frames with a usable face
TRACK_OK = 0.3           # share of a person's frames with confident body tracking (neck, shoulders)
LABEL = {"none": "리터칭 불필요", "small": "얼굴 작음", "good": "적용 양호", "bad": "인식 불량", "pending": "분석 전",
         "skip": "제외"}


def body_edited(cut, pids=None) -> bool:
    """Body or proportion edits (which follow the body tracking) on these people (any person: None)."""
    try:
        from ..core.params import BODY_KEYS, SHAPE_KEYS, default
    except Exception:
        return False
    keys = set(BODY_KEYS) | set(SHAPE_KEYS)
    for pid, pp in (cut.params or {}).items():
        try:
            if pids is not None and int(pid) not in pids:
                continue
        except (TypeError, ValueError):
            pass
        for k, c in pp.amounts.items():
            if k not in keys:
                continue
            d = default(k)
            if abs(c.base - d) > 1e-6 or any(abs(v - d) > 1e-6 for v in c.keys.values()):
                return True
        if getattr(pp, "body_sculpt", None):
            return True
    return False


def effective(cut) -> str:
    st = cut.status or {}
    if st.get("work") == "skip":
        return "none"                       # marked 작업 불필요 by the user
    v = st.get("user") or st.get("auto") or ("none" if cut.tl.get("kind") in ("gap",) else "pending")
    # the body tracking is unsure for someone: a problem only once that body is being reshaped
    if v == "good" and not st.get("user") and st.get("body_weak") and body_edited(cut, set(st["body_weak"])):
        return "bad"
    return v


def reason(cut) -> str:
    st = cut.status or {}
    if effective(cut) == "bad" and st.get("auto") == "good" and st.get("body_weak"):
        return st.get("body_reason", "몸 추적 약함")
    return st.get("reason", "")


STILL_RES = 0.008       # body points' deviation from one rigid motion (p90, share of the body's size)
STILL_FRAMES = 20       # seen at least this long before a figure can be called a picture


def still_pictures(st) -> set:
    """Ids of figures that never move on their own: people on posters, photos, screens. The person detector
    finds them like anyone else; their body points follow one rigid motion (the camera's) through the whole
    shot, within the models' own repeatability, which a living person (breathing, swaying, talking) does not."""
    out = set()
    if st is None:
        return out
    K = st.final[:, :, :17, :]
    for p, pid in enumerate(st.pids):
        pres = np.nonzero(st.present[p])[0]
        ref = None
        res = []
        for t in pres:
            if t > 0 and ref is not None and (st.jump[t - 1] if t - 1 < len(st.jump) else False):
                ref = None
            k = K[p, t]
            ok = np.isfinite(k[:, :2]).all(1) & (np.nan_to_num(k[:, 2]) > 0.5)
            if ok.sum() < 4:
                continue
            if ref is None:
                ref = (k, ok)
                continue
            m = ok & ref[1]
            if m.sum() < 4:
                continue
            A, _ = cv2.estimateAffinePartial2D(ref[0][m, :2].astype(np.float32), k[m, :2].astype(np.float32))
            if A is None:
                continue
            q = ref[0][m, :2] @ A[:, :2].T + A[:, 2]
            size = max(float(np.ptp(k[m, :2], 0).max()), 1.0)
            res.append(float(np.median(np.linalg.norm(q - k[m, :2], axis=1)) / size))
        if len(res) + 1 >= STILL_FRAMES and float(np.quantile(res, 0.9)) < STILL_RES:
            out.add(int(pid))
    return out


def face_widths(st, p: int) -> np.ndarray:
    f = st.face[p, :, :, :2]
    ok = st.present[p] & np.isfinite(f).all((1, 2)) & (st.face_vis[p] >= 0.5)
    w = np.full(st.n, np.nan, np.float32)
    if ok.any():
        w[ok] = np.hypot(*(f[ok, 16] - f[ok, 0]).T)
    return w


def evaluate(cut, tracker, make_targets: bool = False) -> dict:
    """Sets cut.status["auto"/"reason"/"metrics"]; with make_targets every person whose face is seen well
    becomes a beauty target (group shots: all members)."""
    st = tracker.store if tracker is not None else None
    res = dict(auto="none", reason="", metrics={})
    face_ok, small_face, track_ok = FACE_OK, SMALL_FACE, TRACK_OK
    try:
        from ..learn.statuscal import rules
        r_ = rules()
        if r_:
            face_ok, small_face, track_ok = r_["face_ok"], r_["small"], r_["track_ok"]
    except Exception:
        pass
    # face size in source pixels, judged against an HD frame: on a 4K source a 60 px face is half as large on
    # screen, and its retouch shows as little
    small_face *= max(1.0, float(getattr(cut.info, "width", 1920) or 1920) / 1920.0)
    kind = cut.tl.get("kind", "clip")
    if kind == "gap":
        res["reason"] = "검정/빈 구간"
    elif st is None or not st.present.any():
        res["reason"] = "인물 없음"
    elif not getattr(st, "has_face", False):
        res.update(auto="bad", reason="얼굴 분석 없음 (다시 분석 필요)")
    else:
        a, b = cut.work_range
        span = max(1, b - a)
        rows = []
        still = {q.id for q in cut.persons if getattr(q, "still", False)}
        yn = getattr(st, "raw", {}).get("face_yunet")
        for p, pid in enumerate(st.pids):
            if int(pid) in still:
                continue                    # a poster or photo: not someone the shot is about
            pres = st.present[p]
            n = int(pres.sum())
            if n < max(3, 0.15 * span):
                continue
            w = face_widths(st, p)
            seen = np.isfinite(w)
            med = float(np.nanmedian(w)) if seen.any() else 0.0
            # frames the face should have been followed in: where it was, or where the separate face detector
            # (YuNet) saw one. A face behind a cushion, a hand or a turned head is not a miss.
            exp = seen.copy()
            if yn is not None and p < yn.shape[0]:
                det = pres & (np.nan_to_num(yn[p, :, 0, 2]) > 0)
                # only a face on this person's own head (the detector's search box also reaches a neighbour's)
                hd = st.final[p, :, :5, :2]
                ok_h = np.isfinite(hd).all(2) & (np.nan_to_num(st.final[p, :, :5, 2]) > 0.3)
                for t in np.nonzero(det)[0]:
                    pts = hd[t][ok_h[t]]
                    if len(pts) >= 2:
                        spread = max(float(np.ptp(pts, 0).max()), 1.0)
                        if np.linalg.norm(pts.mean(0) - yn[p, t, 0, :2]) > 1.2 * spread:
                            det[t] = False
                exp |= det
            rows.append(dict(pid=int(pid), frames=n, face_share=float(seen.sum() / n), face_px=round(med, 1),
                             face_hit=float(seen.sum() / max(1, int(exp.sum()))),
                             track=float((st.auto_conf[p][pres] >= cut.tracking.conf_threshold).mean())))
        res["metrics"] = dict(persons=rows)
        faces = [r for r in rows if r["face_share"] >= 0.2]
        if not faces:
            res["reason"] = "얼굴이 보이는 인물 없음"
        elif all(r["face_px"] < small_face for r in faces):
            res.update(auto="small", reason=f"얼굴 폭 {max(r['face_px'] for r in faces):.0f}px 이하")
        else:
            main = [r for r in faces if r["face_px"] >= small_face]
            if make_targets:
                ids = {r["pid"] for r in main}
                for q in cut.persons:
                    if q.id in ids and q.role != "ignore" and not getattr(q, "still", False):
                        q.role = "target"
            tg = {q.id for q in cut.persons if q.role == "target"}
            judged = [r for r in main if r["pid"] in tg] or main
            # the people the shot is about decide: the largest faces, or anyone on screen most of the shot.
            # A dancer at the back who is followed badly is noted, but does not fail the cut.
            big = max(r["face_px"] for r in judged)
            major = [r for r in judged if r["face_px"] >= 0.6 * big or r["frames"] >= 0.6 * span]
            hit = lambda r: r.get("face_hit", r["face_share"])
            # the face decides the retouch; the body tracking only matters for body edits (effective() turns
            # body_weak into a warning once the body of that person is reshaped)
            weak = [r for r in major if hit(r) < face_ok]
            minor_weak = [r for r in judged if r not in major and hit(r) < face_ok]
            body_weak = [r for r in major if r["track"] < track_ok]
            if body_weak:
                r = min(body_weak, key=lambda r: r["track"])
                res["body_weak"] = [r_["pid"] for r_ in body_weak]
                res["body_reason"] = f"인물 {r['pid'] + 1}: 몸 추적 신뢰 {r['track'] * 100:.0f}% (몸매 보정 확인)"
            if kind == "transition":
                res.update(auto="bad", reason="트랜지션 (두 샷이 겹침)")
            elif weak:
                r = min(weak, key=lambda r: hit(r))
                why = [f"얼굴 인식 {hit(r) * 100:.0f}%"]
                if r["track"] < track_ok:
                    why.append(f"추적 신뢰 {r['track'] * 100:.0f}%")
                res.update(auto="bad", reason=f"인물 {r['pid'] + 1}: " + ", ".join(why))
            else:
                res.update(auto="good", reason=f"얼굴 {len(judged)}명 · 주요 인물 최소 인식 "
                                               f"{min(hit(r) for r in major) * 100:.0f}%"
                                               + (f" · 작은 인물 {len(minor_weak)}명 약함" if minor_weak else ""))
    s = dict(cut.status or {})
    s.pop("body_weak", None)
    s.pop("body_reason", None)
    s.update(res)
    cut.status = s
    return s


def set_user(cut, value: Optional[str]) -> None:
    s = dict(cut.status or {})
    if value:
        s["user"] = value
    else:
        s.pop("user", None)
    cut.status = s


def set_reviewed(cut, on: bool) -> None:
    s = dict(cut.status or {})
    s["reviewed"] = bool(on)
    cut.status = s
