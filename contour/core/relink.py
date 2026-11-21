"""Media that is somewhere else now (the project opened on another computer or system, a volume with another name or
letter, a folder moved): found again and connected, without touching what was analysed (instruction 2026-11-20 §5).

A project names its media by the path it had where it was added: C:\\\\job\\\\A001.mov or /Volumes/RAID/job/A001.mov. The
format and the keys of the project are the same on every system; only these paths are the computer's own. Given a
folder, every cut whose original is not found is looked for in it by file name (also inside its folders; names are
compared without regard to case, a Windows path's name is read on a Mac and the other way round), and connected
when the file there is the same footage as far as the saved description tells: size, frame rate, and the frames
the cut uses. A file of the same name that is something else is not taken (and said). The analysis, the mattes
and the edits belong to the cut, not to the path: they stay."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import media as MD

MAX_FILES = 300000          # a folder walk gives up past this many files (a whole disk chosen by mistake)


def base_name(path: str) -> str:
    """The file name of a path saved on any system (a Windows path keeps its backslashes on a Mac)."""
    return re.split(r"[\\/]", str(path or "").rstrip("\\/"))[-1]


def _present(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except (OSError, ValueError):
        return False


def _source_file(info) -> str:
    """The file whose presence stands for the cut's original: the video, or a sequence's first frame."""
    if getattr(info, "kind", "video") == "sequence":
        try:
            return info.seq_pattern % info.seq_start
        except (TypeError, ValueError):
            return str(info.seq_pattern)
    return str(info.path)


def missing(project) -> List[dict]:
    """What the project names and this computer does not have: [{kind, path, name, cuts: [id]}] with kind
    'source' (a cut's original), 'lut', 'retouch' (the hand result a cut is compared with)."""
    out: Dict[tuple, dict] = {}
    for c in project.cuts:
        f = _source_file(c.info)
        if f and not _present(f):
            out.setdefault(("source", f), dict(kind="source", path=f, name=base_name(f), cuts=[]))["cuts"].append(c.id)
        for kind, p in (("lut", c.lut_path), ("retouch", c.retouch_ref)):
            if p and not _present(p):
                out.setdefault((kind, p), dict(kind=kind, path=p, name=base_name(p), cuts=[]))["cuts"].append(c.id)
    return list(out.values())


def index_folder(folder, wanted: Optional[set] = None, cancel: Optional[Callable[[], bool]] = None) -> Dict[str, List[str]]:
    """file name (lower case) -> the paths under the folder with that name. wanted: only these names."""
    out: Dict[str, List[str]] = {}
    n = 0
    want = {w.lower() for w in wanted} if wanted is not None else None
    for root, dirs, files in os.walk(str(folder)):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("ai_review", "autosave"))
        for f in files:
            n += 1
            if n > MAX_FILES or (cancel and cancel()):
                return out
            k = f.lower()
            if want is None or k in want:
                out.setdefault(k, []).append(os.path.join(root, f))
    return out


def same_footage(info, new_file: str, probe=None) -> str:
    """'' when the file can stand for the cut's original as its saved description tells, else why not."""
    probe = probe or MD.probe
    try:
        got = probe(new_file)
    except Exception as e:
        return f"읽지 못함 ({str(e)[:80]})"
    if (int(got.width), int(got.height)) != (int(info.width), int(info.height)):
        return f"크기가 다름 ({got.width}x{got.height}, 저장된 것 {info.width}x{info.height})"
    if abs(float(got.fps) - float(info.fps)) > 0.01:
        return f"프레임 속도가 다름 ({got.fps:.3f}, 저장된 것 {info.fps:.3f})"
    off = int(getattr(info, "offset", 0) or 0)
    if off:
        if int(got.frames) < off + int(info.frames):
            return f"길이가 모자람 ({got.frames}프레임, 이 컷은 {off}~{off + int(info.frames) - 1} 사용)"
    elif int(got.frames) != int(info.frames):
        return f"프레임 수가 다름 ({got.frames}, 저장된 것 {info.frames})"
    return ""


def _connect(cut, old_file: str, new_file: str) -> None:
    info = cut.info
    if getattr(info, "kind", "video") == "sequence":
        # the pattern keeps its file name part, in the folder where the first frame was found
        pat = os.path.join(os.path.dirname(new_file), base_name(info.seq_pattern))
        if str(cut.source) == str(info.seq_pattern) or base_name(str(cut.source)) == base_name(info.seq_pattern):
            cut.source = pat
        info.seq_pattern = pat
        if base_name(info.path) == base_name(old_file) or not _present(info.path):
            info.path = new_file
    else:
        if str(cut.source) == str(info.path) or base_name(str(cut.source)) == base_name(old_file):
            cut.source = new_file
        info.path = new_file


def relink(project, folder, probe=None, progress: Optional[Callable[[float, str], None]] = None,
           cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Connects what is missing to the files of the same name under the folder.
    -> {"linked": [{kind, name, cuts}], "different": [{name, why}], "not_found": [{kind, name, cuts}]}.
    The project is changed in memory (the caller saves it); nothing on the disk is touched."""
    miss = missing(project)
    res = dict(linked=[], different=[], not_found=[])
    if not miss:
        return res
    idx = index_folder(folder, {m["name"] for m in miss}, cancel)
    by_id = {c.id: c for c in project.cuts}
    for i, m in enumerate(miss):
        if cancel and cancel():
            break
        if progress:
            progress(i / len(miss), f"원본 찾는 중: {m['name']}")
        cands = idx.get(m["name"].lower(), [])
        if not cands:
            res["not_found"].append(dict(kind=m["kind"], name=m["name"], cuts=m["cuts"]))
            continue
        if m["kind"] != "source":
            for cid in m["cuts"]:
                c = by_id[cid]
                if m["kind"] == "lut":
                    c.lut_path = cands[0]
                else:
                    c.retouch_ref = cands[0]
            res["linked"].append(dict(kind=m["kind"], name=m["name"], cuts=m["cuts"]))
            continue
        done, why_not = [], ""
        for cid in m["cuts"]:
            c = by_id[cid]
            hit = None
            for cand in cands:
                why = same_footage(c.info, cand, probe)
                if not why:
                    hit = cand
                    break
                why_not = why
            if hit is None:
                continue
            _connect(c, m["path"], hit)
            done.append(cid)
        if done:
            res["linked"].append(dict(kind="source", name=m["name"], cuts=done))
            tl = getattr(project, "timeline", None) or {}
            if tl.get("baked") and base_name(tl["baked"]) == m["name"] and not _present(tl["baked"]):
                tl["baked"] = by_id[done[0]].info.path          # the timeline's own note of its bake
        left = [c for c in m["cuts"] if c not in done]
        if left:
            res["different"].append(dict(name=m["name"], why=why_not, cuts=left))
    if progress:
        progress(1.0, "원본 연결 끝")
    return res


def summary(res: dict) -> str:
    n_l = sum(len(x["cuts"]) for x in res["linked"] if x["kind"] == "source")
    parts = [f"원본 {n_l}컷 연결"] if n_l else []
    other = [x for x in res["linked"] if x["kind"] != "source"]
    if other:
        parts.append(f"LUT·수작업 결과 {len(other)}개 연결")
    if res["different"]:
        d = res["different"][0]
        parts.append(f"이름은 같지만 다른 영상 {len(res['different'])}개 ({d['name']}: {d['why']})")
    if res["not_found"]:
        parts.append(f"찾지 못함 {len(res['not_found'])}개 ({', '.join(x['name'] for x in res['not_found'][:3])}"
                     + (" 외" if len(res["not_found"]) > 3 else "") + ")")
    return " · ".join(parts) or "연결할 것이 없습니다 (모든 원본이 제자리에 있습니다)"
