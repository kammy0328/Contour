"""Turning a baked timeline video + its editing file into one cut per shot."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import media as M
from .project import Project, safe_name
from .timeline import Timeline, align, picture_change, read_timeline

KIND_NAME = {"transition": "트랜지션", "gap": "검정"}


def thumbs_dir(project: Project) -> Path:
    d = project.folder / "timeline" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_thumb(project: Project, cut_id: str, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        (thumbs_dir(project) / f"{cut_id}.jpg").write_bytes(buf.tobytes())


def load_thumb(project: Project, cut_id: str) -> Optional[np.ndarray]:
    f = thumbs_dir(project) / f"{cut_id}.jpg"
    if not f.exists():
        return None
    im = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def import_timeline(project: Project, edit_path: str, baked_path: str, progress: Callable = None,
                    cancelled: Callable = None, replace: bool = True, offset: Optional[int] = None) -> dict:
    progress = progress or (lambda f, m="": None)
    info = M.probe(baked_path)
    tl: Timeline = read_timeline(edit_path, info.fps)
    warn = list(tl.warnings)
    if abs(tl.fps - info.fps) > 0.01:
        warn.append(f"프레임레이트가 다릅니다: 편집 {tl.fps:.3f} / 영상 {info.fps:.3f}")
    if tl.width and info.width and (tl.width, tl.height) != (info.width, info.height):
        warn.append(f"해상도가 다릅니다: 편집 {tl.width}x{tl.height} / 영상 {info.width}x{info.height}")
    diff, thumbs = picture_change(info, lambda f, m="": progress(0.9 * f, m), cancelled)
    n = len(diff) or info.frames
    baked_tc = M.tc_to_frames(info.timecode, info.fps) if info.timecode else None
    al = align(tl, diff, baked_tc)
    off = int(al["offset"] if offset is None else offset)
    if al["total"] and al["matched"] < 0.6 * al["total"]:
        warn.append(f"편집 경계 {al['total']}개 중 {al['matched']}개만 영상의 화면 전환과 맞습니다. "
                    f"편집 파일 버전이나 영상 시작 위치를 확인하세요")
    if off:
        warn.append(f"영상이 시퀀스보다 {off:+d}프레임 어긋나 있어 맞췄습니다 ({al['method']})")
    if al["extra"]:
        warn.append(f"편집 파일에 없는 큰 화면 변화 {len(al['extra'])}곳 (플래시 또는 빠진 컷): 프레임 " +
                    ", ".join(str(x) for x in al["extra"][:8]) + (" …" if len(al["extra"]) > 8 else ""))
    if replace:
        project.cuts = [c for c in project.cuts if not c.tl]
    shots = []
    for s in tl.shots:
        a, b = s.rec_in + off, s.rec_out + off
        a, b = max(0, a), min(n, b)
        if b - a >= 1:
            shots.append((a, b, s))
    if shots and shots[-1][1] < n:
        from .timeline import Shot
        shots.append((shots[-1][1], n, Shot(shots[-1][1] - off, n - off, "gap", note="편집 파일 범위 밖")))
    if shots and shots[0][0] > 0:
        from .timeline import Shot
        shots.insert(0, (0, shots[0][0], Shot(-off, shots[0][0] - off, "gap", note="편집 파일 범위 밖")))
    if tl.duration + off > n:
        warn.append(f"영상이 편집보다 {tl.duration + off - n}프레임 짧습니다 (끝부분 컷 일부 없음)")
    made = []
    for k, (a, b, s) in enumerate(shots):
        ci = copy.deepcopy(info)
        ci.offset = int(a)
        ci.frames = int(b - a)
        ci.duration = ci.frames / max(1e-6, ci.fps)
        label = KIND_NAME.get(s.kind) or Path(s.name).stem or "shot"
        cut = project.add_cut(ci, f"{k + 1:03d}_{safe_name(label)[:40]}")
        cut.tl = dict(index=k, kind=s.kind, rec_in=int(s.rec_in), rec_out=int(s.rec_out), base=int(a),
                      name=s.name, note=s.note, src_in=int(s.src_in), speed=float(s.speed))
        cut.edit_points = []
        if s.kind == "gap":
            cut.status = dict(auto="none", reason="검정/빈 구간")
        elif s.kind == "transition":
            cut.status = dict(auto="none", reason="트랜지션 (두 샷이 겹침, 기본은 원본 그대로)")
        if thumbs:
            save_thumb(project, cut.id, thumbs[min(len(thumbs) - 1, (a + b) // 2)])
        made.append(cut)
    project.timeline = dict(edit=str(edit_path), baked=str(baked_path), fmt=tl.fmt, name=tl.name, fps=info.fps,
                            fps_num=info.fps_num, fps_den=info.fps_den, start_tc=tl.start_tc,
                            seq_start=int(tl.start_frames), drop=tl.drop, offset=off, frames=int(n),
                            width=info.width, height=info.height, align=al, warnings=warn, created=time.time(),
                            shots=len(made))
    project.save()
    progress(1.0, "")
    return project.timeline
