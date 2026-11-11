"""Beauty status of a cut, decided from its analysis.

none   리터칭 불필요   nobody, or nobody whose face is seen (back, far away, a hand over it all along)
small  얼굴 작음       faces seen, but all smaller than SMALL_FACE px wide (a retouch would not show)
good   적용 양호       every retouched face followed most of the time with confident tracking
bad    인식 불량       a retouched face lost often, tracking unsure, or (transitions) two shots mixed

The user can override it (status["user"]) and mark the cut as checked (status["reviewed"]).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

SMALL_FACE = 60          # source px, face width (jaw to jaw)
FACE_OK = 0.8            # share of a person's frames with a usable face
TRACK_OK = 0.3           # share of a person's frames with confident body tracking (neck, shoulders)
LABEL = {"none": "리터칭 불필요", "small": "얼굴 작음", "good": "적용 양호", "bad": "인식 불량", "pending": "분석 전",
         "skip": "제외"}


def effective(cut) -> str:
    st = cut.status or {}
    return st.get("user") or st.get("auto") or ("none" if cut.tl.get("kind") in ("gap",) else "pending")


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
        for p, pid in enumerate(st.pids):
            pres = st.present[p]
            n = int(pres.sum())
            if n < max(3, 0.15 * span):
                continue
            w = face_widths(st, p)
            seen = np.isfinite(w)
            med = float(np.nanmedian(w)) if seen.any() else 0.0
            rows.append(dict(pid=int(pid), frames=n, face_share=float(seen.sum() / n), face_px=round(med, 1),
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
                    if q.id in ids and q.role != "ignore":
                        q.role = "target"
            tg = {q.id for q in cut.persons if q.role == "target"}
            judged = [r for r in main if r["pid"] in tg] or main
            # the people the shot is about decide: the largest faces, or anyone on screen most of the shot.
            # A dancer at the back who is followed badly is noted, but does not fail the cut.
            big = max(r["face_px"] for r in judged)
            major = [r for r in judged if r["face_px"] >= 0.6 * big or r["frames"] >= 0.6 * span]
            weak = [r for r in major if r["face_share"] < face_ok or r["track"] < track_ok]
            minor_weak = [r for r in judged if r not in major and (r["face_share"] < face_ok or r["track"] < track_ok)]
            if kind == "transition":
                res.update(auto="bad", reason="트랜지션 (두 샷이 겹침)")
            elif weak:
                r = min(weak, key=lambda r: min(r["face_share"], r["track"]))
                why = []
                if r["face_share"] < face_ok:
                    why.append(f"얼굴 인식 {r['face_share'] * 100:.0f}%")
                if r["track"] < track_ok:
                    why.append(f"추적 신뢰 {r['track'] * 100:.0f}%")
                res.update(auto="bad", reason=f"인물 {r['pid'] + 1}: " + ", ".join(why))
            else:
                res.update(auto="good", reason=f"얼굴 {len(judged)}명 · 주요 인물 최소 인식 "
                                               f"{min(r['face_share'] for r in major) * 100:.0f}%"
                                               + (f" · 작은 인물 {len(minor_weak)}명 약함" if minor_weak else ""))
    s = dict(cut.status or {})
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
