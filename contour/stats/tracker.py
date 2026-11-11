"""Work statistics: active editing time, corrections, analysis and render events.

Events are appended to <data>/stats/events.jsonl so the dashboard can aggregate across projects even when
project folders move.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..core.settings import settings

IDLE_LIMIT = 90.0


def _log_path() -> Path:
    return settings().stats_path / "events.jsonl"


def log_event(kind: str, project: str, cut: str, **data) -> None:
    rec = dict(t=time.time(), kind=kind, project=project, cut=cut, **data)
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_events() -> List[dict]:
    p = _log_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


class ActivityClock:
    """Counts seconds of active work on the current cut (gaps over IDLE_LIMIT are not counted)."""

    def __init__(self):
        self.project = ""
        self.cut = ""
        self.cut_obj = None
        self._last = None
        self._pending = 0.0

    def set_cut(self, project_name: str, cut) -> None:
        self.flush()
        self.project = project_name
        self.cut_obj = cut
        self.cut = cut.name if cut else ""
        self._last = None

    def touch(self) -> None:
        now = time.time()
        if self._last is not None:
            gap = now - self._last
            if gap <= IDLE_LIMIT:
                self._pending += gap
                if self.cut_obj is not None:
                    self.cut_obj.stats.active_seconds += gap
        self._last = now
        if self._pending > 60:
            self.flush()

    def flush(self) -> None:
        if self._pending > 0.5 and self.cut:
            log_event("active", self.project, self.cut, seconds=round(self._pending, 1))
        self._pending = 0.0


# --------------------------------------------------------------------------- aggregation

def aggregate(events: Optional[List[dict]] = None) -> dict:
    ev = events if events is not None else read_events()
    per_cut: Dict[tuple, dict] = defaultdict(lambda: dict(active=0.0, corrections=0, frames=0, auto_ok=None,
                                                           renders=0, render_s=0.0, analyses=0, last=0.0))
    for e in ev:
        key = (e.get("project", ""), e.get("cut", ""))
        r = per_cut[key]
        r["last"] = max(r["last"], e.get("t", 0))
        k = e.get("kind")
        if k == "active":
            r["active"] += float(e.get("seconds", 0))
        elif k == "correction":
            r["corrections"] += int(e.get("count", 1))
        elif k == "analysis":
            r["analyses"] += 1
            r["frames"] = int(e.get("frames", r["frames"]))
            if e.get("auto_ok") is not None and e.get("auto_ok") >= 0:
                r["auto_ok"] = float(e["auto_ok"])
        elif k == "render":
            r["renders"] += 1
            r["render_s"] += float(e.get("seconds", 0))
            r["frames"] = max(r["frames"], int(e.get("frames", 0)))
    per_project: Dict[str, dict] = defaultdict(lambda: dict(cuts=0, active=0.0, corrections=0, frames=0, renders=0,
                                                            auto_ok=[], last=0.0))
    for (pj, ct), r in per_cut.items():
        pp = per_project[pj]
        pp["cuts"] += 1
        pp["active"] += r["active"]
        pp["corrections"] += r["corrections"]
        pp["frames"] += r["frames"]
        pp["renders"] += r["renders"]
        pp["last"] = max(pp["last"], r["last"])
        if r["auto_ok"] is not None:
            pp["auto_ok"].append(r["auto_ok"])
    for pp in per_project.values():
        pp["auto_ok"] = sum(pp["auto_ok"]) / len(pp["auto_ok"]) if pp["auto_ok"] else None
    cuts = [dict(project=k[0], cut=k[1], **v) for k, v in per_cut.items()]
    projects = [dict(project=k, **v) for k, v in per_project.items()]
    total_frames = sum(c["frames"] for c in cuts)
    total_active = sum(c["active"] for c in cuts)
    total_corr = sum(c["corrections"] for c in cuts)
    oks = [c["auto_ok"] for c in cuts if c["auto_ok"] is not None]
    worked = [c for c in cuts if c["active"] > 0]
    return dict(
        cuts=sorted(cuts, key=lambda c: -c["last"]),
        projects=sorted(projects, key=lambda p: -p["last"]),
        total_cuts=len(cuts),
        total_frames=total_frames,
        total_active=total_active,
        avg_cut_minutes=(sum(c["active"] for c in worked) / len(worked) / 60.0) if worked else None,
        auto_ok=(sum(oks) / len(oks)) if oks else None,
        corrections_per_100=(100.0 * total_corr / total_frames) if total_frames else None,
        total_renders=sum(c["renders"] for c in cuts),
    )


def export_csv(path: str, agg: Optional[dict] = None) -> None:
    agg = agg or aggregate()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["프로젝트", "컷", "작업 시간(분)", "수동 수정", "프레임", "자동 추적 성공률(%)", "렌더 횟수",
                    "렌더 시간(분)", "마지막 작업"])
        for c in agg["cuts"]:
            w.writerow([c["project"], c["cut"], round(c["active"] / 60, 1), c["corrections"], c["frames"],
                        "" if c["auto_ok"] is None else round(c["auto_ok"] * 100, 1), c["renders"],
                        round(c["render_s"] / 60, 1), time.strftime("%Y-%m-%d %H:%M", time.localtime(c["last"]))])
