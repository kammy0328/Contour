"""Where each cut stands in the work, and its versions.

States (one per cut, shown in the cut list, the timeline strip and used by the filters and the renders):
  pre   분석 전      not analysed yet
  todo  작업 전      analysed, nothing adjusted yet
  doing 작업 중      something adjusted
  done  완료         marked finished by the user
  skip  작업 불필요   no one to retouch (found automatically) or marked by the user
Versions: every cut starts at v1. "버전 올리기" keeps a snapshot of the current work as that version and goes on
from it as the next one (nothing is reset); "이전 버전으로" brings a kept snapshot back.
"""
from __future__ import annotations

import copy
import time
from typing import List, Tuple

STATES: List[Tuple[str, str]] = [("pre", "분석 전"), ("todo", "작업 전"), ("doing", "작업 중"), ("done", "완료"),
                                 ("skip", "작업 불필요")]
LABEL = dict(STATES)
COLOR = {"pre": "#6F6F6F", "todo": "#A6A6A6", "doing": "#4F8BFF", "done": "#5CC8A0", "skip": "#5A5A5A"}


def _active(cut) -> bool:
    from .params import default
    for pp in cut.params.values():
        for k, c in pp.amounts.items():
            d = default(k)
            if abs(c.base - d) > 1e-6 or any(abs(v - d) > 1e-6 for v in c.keys.values()):
                return True
        if pp.sculpt or pp.body_sculpt:
            return True
    return False


def state(cut) -> str:
    st = cut.status or {}
    w = st.get("work", "")
    if w == "skip":
        return "skip"
    if w == "done":
        return "done"
    if not w and (st.get("user") == "skip" or (st.get("auto") == "none" and not st.get("user"))):
        return "skip"
    if not cut.analyzed:
        return "pre"
    return "doing" if _active(cut) else "todo"


def set_state(cut, s: str) -> None:
    """'done' / 'skip' are kept; '' goes back to automatic (pre / todo / doing / auto skip)."""
    st = dict(cut.status or {})
    if s in ("done", "skip"):
        st["work"] = s
    else:
        st.pop("work", None)
        if st.get("user") == "skip":
            st.pop("user", None)
    if s == "" and st.get("auto") == "none":
        st["work"] = "unskip"          # the user said this one does need work after all
    cut.status = st


def _snapshot(cut) -> dict:
    d = cut.to_dict()
    keep = ("params", "persons", "mode", "protect")
    return dict({k: copy.deepcopy(d[k]) for k in keep}, time=time.time())


def version_up(cut) -> int:
    cut.versions[str(cut.version)] = _snapshot(cut)
    cut.version += 1
    return cut.version


def version_back(cut) -> int:
    """Back to the newest kept version below the current one (the current work is kept as its version)."""
    older = sorted(int(v) for v in cut.versions if int(v) < cut.version)
    if not older:
        return cut.version
    cut.versions[str(cut.version)] = _snapshot(cut)
    v = older[-1]
    snap = cut.versions[str(v)]
    from .project import Cut
    tmp = Cut.from_dict(dict(cut.to_dict(), **{k: snap[k] for k in ("params", "persons", "mode", "protect")}))
    cut.params, cut.persons, cut.mode, cut.protect = tmp.params, tmp.persons, tmp.mode, tmp.protect
    cut.version = v
    return v
