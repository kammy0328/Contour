"""Learning the beauty-status rules from the user's own review.

Each time a cut's automatic status is overridden (or a cut is marked reviewed without a change, which
confirms it), the cut's measurements and the user's verdict are kept. With enough of them the two numbers
that decide most verdicts, the face-recognition share a main person needs (FACE_OK) and the smallest face
worth retouching (SMALL_FACE), are re-chosen to agree with the user as often as possible. The new values
are used only when they agree with the user on clearly more cuts than the shipped ones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..core.settings import settings

MIN_LABELS = 15


def _f() -> Path:
    d = settings().learning_path
    d.mkdir(parents=True, exist_ok=True)
    return d / "status_feedback.jsonl"


def _active() -> Path:
    return settings().learning_path / "status_rules.json"


def record(cut, verdict: str) -> None:
    if verdict not in ("good", "bad", "small", "none"):
        return
    m = (cut.status or {}).get("metrics", {})
    if not m.get("persons"):
        return
    row = dict(t=time.time(), cut=cut.id, verdict=verdict, auto=(cut.status or {}).get("auto"), metrics=m,
               kind=cut.tl.get("kind", "clip"))
    try:
        with open(_f(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _judge(m: dict, face_ok: float, small: float, track_ok: float) -> str:
    rows = m.get("persons", [])
    faces = [r for r in rows if r["face_share"] >= 0.2]
    if not faces:
        return "none"
    if all(r["face_px"] < small for r in faces):
        return "small"
    main = [r for r in faces if r["face_px"] >= small]
    big = max(r["face_px"] for r in main)
    span = max(r["frames"] for r in rows) or 1
    major = [r for r in main if r["face_px"] >= 0.6 * big or r["frames"] >= 0.6 * span]
    weak = [r for r in major if r["face_share"] < face_ok or r["track"] < track_ok]
    return "bad" if weak else "good"


def refit() -> Optional[dict]:
    from ..track import status as ST
    rows = []
    try:
        for ln in _f().read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(ln))
    except (OSError, ValueError):
        return None
    last = {}
    for r in rows:
        if r.get("kind", "clip") == "clip":
            last[r["cut"]] = r                 # the latest verdict per cut
    rows = list(last.values())
    if len(rows) < MIN_LABELS:
        return None
    base = dict(face_ok=0.8, small=60.0, track_ok=0.3)

    def agree(p):
        return sum(1 for r in rows if _judge(r["metrics"], p["face_ok"], p["small"], p["track_ok"]) == r["verdict"])

    a0 = agree(base)
    best, ab = base, a0
    for fo in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        for sm in (40.0, 50.0, 60.0, 70.0, 80.0, 100.0):
            for to in (0.2, 0.3, 0.4, 0.5):
                p = dict(face_ok=fo, small=sm, track_ok=to)
                a = agree(p)
                if a > ab:
                    best, ab = p, a
    res = dict(labels=len(rows), agree_default=a0, agree_learned=ab, rules=best, active=False)
    if ab >= a0 + max(2, 0.1 * len(rows)):
        _active().write_text(json.dumps(dict(res, active=True, created=time.time()), ensure_ascii=False), encoding="utf-8")
        res["active"] = True
    return res


def rules() -> Optional[dict]:
    if not getattr(settings(), "self_learning", True):
        return None
    try:
        d = json.loads(_active().read_text(encoding="utf-8"))
        return d.get("rules") if d.get("active") else None
    except (OSError, ValueError):
        return None
