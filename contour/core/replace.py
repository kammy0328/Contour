"""A cut's pictures taken from another file of the same shot (the user, 2026-11-27, LNGSHOT BEEP BEEP: the cuts sent to
cleaning come back as files of their own and take the place of the stretches of the graded film; "Replace한다고 해서 보정
값이 달라지면 안돼"): right-click a cut > 소스 교체.

What the cut is stays: its people, their values, the tracks, the mattes, the face layer, its place in the timeline, its
name and version. Only where its pictures are read from changes (Cut.source, Cut.info: the new file, and where in it the
cut's frames lie), so the same retouch is made on the new pictures. The proxy of the old pictures is let go (a
timeline cut no longer reads the baked film's shared proxy).

Where the cut's frames lie in the new file, in this order:
1. by timecode, when the cut and the new file both have one and the cut's whole range lies in the file (a cleaned cut
   exported with the timeline's timecode: its first frame; a new export of the whole film: where the cut was);
2. a file exactly as long as the cut: from its first frame;
3. a file exactly as long as the file the cut reads now: at the same place (a new version of the whole film).
A file of another size or frame rate, or one in which the cut's frames cannot be placed, is not taken (and said why).

The step before can be taken back (Cut.replaced: the source and place of before)."""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import media as MD

SIM_WARN = 0.35          # below this the old and the new pictures are not the same shot as far as a few frames tell


def _tc_start(info) -> Optional[int]:
    """The timecode of the cut's (or file's) first frame in frames; None without a timecode."""
    tc = str(getattr(info, "timecode", "") or "")
    if not tc:
        return None
    try:
        return int(MD.tc_to_frames(tc, info.fps)) + int(getattr(info, "offset", 0) or 0)
    except Exception:
        return None


def check(cut, new_file: str, probe=None) -> dict:
    """Can the cut read its pictures from new_file? {ok, why, info, offset, how, note}: info is the cut's new
    description (the file's, with the cut's place and length), how says in a few words how the place was found."""
    probe = probe or MD.probe
    old = cut.info
    n = int(old.frames)
    try:
        got = probe(new_file)
    except Exception as e:
        return dict(ok=False, why=f"파일을 읽지 못했습니다 ({str(e)[:100]})")
    if getattr(got, "kind", "video") != getattr(old, "kind", "video"):
        return dict(ok=False, why="영상 파일과 이미지 시퀀스는 서로 교체할 수 없습니다")
    if (int(got.width), int(got.height)) != (int(old.width), int(old.height)):
        return dict(ok=False, why=f"크기가 다릅니다 (새 파일 {got.width}x{got.height}, 이 컷 {old.width}x{old.height}). "
                                  "분석이 화면 좌표에 묶여 있어 같은 크기만 교체할 수 있습니다")
    if abs(float(got.fps) - float(old.fps)) > 0.01:
        return dict(ok=False, why=f"프레임 속도가 다릅니다 (새 파일 {got.fps:.3f}, 이 컷 {old.fps:.3f})")
    total = int(got.frames)
    off, how, note = None, "", ""
    t_cut, t_new = _tc_start(old), _tc_start(got)
    if t_cut is not None and t_new is not None:
        d = t_cut - t_new
        if 0 <= d and d + n <= total:
            off, how = d, "타임코드"
        else:
            note = "타임코드가 이 컷의 구간과 맞지 않습니다"
    if off is None and total == n:
        off, how = 0, "같은 길이"
    if off is None and int(getattr(old, "offset", 0) or 0) > 0:
        try:
            was = int(probe(str(old.path)).frames)
        except Exception:
            was = -1
        if was == total and int(old.offset) + n <= total:
            off, how = int(old.offset), "같은 위치"
    if off is None:
        return dict(ok=False, why=f"이 컷의 프레임을 새 파일에서 찾을 수 없습니다 (새 파일 {total}프레임, 이 컷 {n}프레임"
                                  + (", " + note if note else "") + "). 컷과 같은 길이의 파일이거나 같은 타임코드를 가진 파일이어야 합니다")
    info = copy.deepcopy(got)
    info.offset = int(off)
    info.frames = n
    info.duration = n / max(1e-6, info.fps)
    return dict(ok=True, why="", info=info, offset=int(off), how=how, note=note if how != "타임코드" else "")


def _outline(rgb: np.ndarray) -> np.ndarray:
    """A small picture of where the frame's edges are, whatever its colours and brightness (the same shot graded or
    cleaned gives nearly the same; another shot does not)."""
    import cv2
    g = cv2.resize(np.asarray(rgb, np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32), (160, 90),
                   interpolation=cv2.INTER_AREA)
    l = np.log(np.maximum(g, 0.0) + 0.01)
    m = np.hypot(cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3))
    m = m / (cv2.GaussianBlur(m, (0, 0), 6) + 0.02)
    m = m - m.mean()
    return m / (float(m.std()) + 1e-6)


def similarity(cut, new_info, read: Optional[Callable] = None) -> Optional[float]:
    """How alike the cut's pictures of now and of the new file are (the least over its first, middle and last frame;
    about 1 for the same shot, about 0 for another). None when it cannot be told (the old file is not there).
    read(info, t) -> float RGB (h, w, 3) of frame t of a cut description."""
    if read is None:
        def read(info, t):
            pf = "yuv444p10le"
            rd = MD.FrameReader(info, int(t), 1, pf, prefetch=1)
            try:
                pl = rd.read()
            finally:
                rd.close()
            if pl is None:
                raise MD.MediaError("프레임을 읽지 못했습니다")
            m = MD.swscale_matrix(info)
            return MD.yuv_to_rgb(pl, MD.plane_layout(pf, int(info.width), int(info.height)),
                                 m if m in ("bt709", "bt601", "bt2020") else "bt709", info.color_range)
    n = int(cut.info.frames)
    worst = None
    for t in sorted({0, n // 2, max(0, n - 1)}):
        try:
            a, b = _outline(read(cut.info, t)), _outline(read(new_info, t))
        except Exception:
            return None
        s = float((a * b).mean())
        worst = s if worst is None else min(worst, s)
    return worst


def _drop_proxy(project, cut) -> None:
    """The proxy of the pictures the cut read before: the link to a timeline's shared proxy and a proxy of its own."""
    from .proxy import SHARED
    dirs = []
    try:
        dirs.append(Path(project.proxy_dir(cut)))
    except Exception:
        pass
    try:
        dirs.append(Path(project.folder) / "cuts" / cut.id / "cache")
    except Exception:
        pass
    for d in dirs:
        try:
            for f in [d / SHARED, d / "proxy_meta.json"] + list(d.glob("proxy_*.u8")):
                try:
                    f.unlink()
                except OSError:
                    pass
        except OSError:
            pass


def apply(project, cut, new_info, how: str = "") -> dict:
    """The cut reads its pictures from new_info (check()'s 'info'). Returns what it read before."""
    before = dict(source=str(cut.source), info=cut.info.to_dict(), at=time.time(), how=how)
    _drop_proxy(project, cut)
    cut.replaced = before
    cut.source = str(new_info.path)
    cut.info = new_info
    return before


def can_undo(cut) -> bool:
    r = getattr(cut, "replaced", None) or {}
    return bool(r.get("info"))


def undo(project, cut) -> bool:
    """Back to the source of before the last replace (kept in the cut). False when there is none."""
    if not can_undo(cut):
        return False
    r = cut.replaced
    now = dict(source=str(cut.source), info=cut.info.to_dict(), at=time.time(), how="되돌리기")
    _drop_proxy(project, cut)
    cut.source = str(r.get("source") or r["info"].get("path", ""))
    cut.info = MD.MediaInfo.from_dict(r["info"])
    cut.replaced = now                     # taking it back can be taken back
    return True


def find_in_folder(cuts, folder: str, probe=None, cancel: Optional[Callable[[], bool]] = None) -> List[dict]:
    """For each cut the file of the folder (not its subfolders) that holds its frames by timecode: [{cut, ok, file,
    info, offset, how, others} | {cut, ok: False, why}]. A file as long as the cut that begins at the cut's timecode is
    taken before a longer one that contains it; of several such the newest (others: the rest). Files the cut reads
    now are left out. Without timecodes nothing is matched (a folder has no other way to tell the cuts apart)."""
    probe = probe or MD.probe
    files = []
    try:
        names = sorted(os.listdir(folder))
    except OSError as e:
        return [dict(cut=c, ok=False, why=f"폴더를 읽지 못했습니다 ({str(e)[:80]})") for c in cuts]
    for nm in names:
        if cancel is not None and cancel():
            break
        p = os.path.join(folder, nm)
        if nm.startswith(".") or Path(nm).suffix.lower() not in MD.VIDEO_EXTS or not os.path.isfile(p):
            continue
        try:
            files.append((p, probe(p), os.path.getmtime(p)))
        except Exception:
            continue
    out = []
    for c in cuts:
        t_cut = _tc_start(c.info)
        if t_cut is None:
            out.append(dict(cut=c, ok=False, why="이 컷에 타임코드가 없어 폴더에서 맞는 파일을 고를 수 없습니다 (파일을 직접 골라 교체하세요)"))
            continue
        cand = []
        for p, got, mt in files:
            try:
                if os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(str(c.info.path))):
                    continue
            except Exception:
                pass
            t_new = _tc_start(got)
            if t_new is None:
                continue
            r = check(c, p, probe=lambda _p, got=got: got)
            if not r["ok"] or r["how"] != "타임코드":
                continue
            exact = r["offset"] == 0 and int(got.frames) == int(c.info.frames)
            cand.append((0 if exact else 1, -mt, p, r))
        if not cand:
            out.append(dict(cut=c, ok=False, why="타임코드가 맞는 파일이 폴더에 없습니다"))
            continue
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        best = cand[0]
        r = best[3]
        out.append(dict(cut=c, ok=True, file=best[2], info=r["info"], offset=r["offset"], how=r["how"],
                        others=[x[2] for x in cand[1:] if x[0] == best[0]]))
    return out
