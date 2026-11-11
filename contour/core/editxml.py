"""Edit points from an editing XML (Premiere Pro / Final Cut Pro 7 xmeml).

A plate handed over for beauty work is often a stretch of footage with the edit already in it (jump cuts on
the snare, punch-ins). The editor's sequence knows exactly where each shot starts: every clip item that uses
a file says which source frames it covers. Those boundaries are where tracking must restart, and only there
(a flash frame inside a shot is then never mistaken for a cut)."""
from __future__ import annotations

import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, List
from urllib.parse import unquote


def _key(name: str) -> str:
    """File name compared loosely: no folder, no extension, case, spaces, underscores and dashes ignored
    (a file renamed from '스네어 플레이트.mov' to '스네어_플레이트.mov' is the same plate)."""
    base = os.path.basename(unquote(name).replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = unicodedata.normalize("NFC", base).lower()
    return re.sub(r"[\s_\-]+", "", base)


def _int(el, tag, default=None):
    v = el.findtext(tag)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def read_edit_points(path: str) -> Dict[str, List[int]]:
    """{file key: sorted source frames where a new shot starts}. Only boundaries strictly inside the used
    footage are returned (the first frame of a file is not a cut)."""
    root = ET.parse(path).getroot()
    names: Dict[str, str] = {}
    for f in root.iter("file"):
        fid = f.get("id")
        nm = f.findtext("name") or f.findtext("pathurl") or ""
        if fid and nm:
            names[fid] = nm
    spans: Dict[str, List[tuple]] = {}
    for video in root.iter("video"):
        for track in video.findall("track"):
            for ci in track.findall("clipitem"):
                f = ci.find("file")
                if f is None:
                    continue
                nm = names.get(f.get("id") or "", "") or ci.findtext("name") or ""
                a, b = _int(ci, "in"), _int(ci, "out")
                if not nm or a is None or b is None or b <= a:
                    continue
                spans.setdefault(_key(nm), []).append((a, b))
    out: Dict[str, List[int]] = {}
    for k, sp in spans.items():
        lo = min(a for a, _ in sp)
        hi = max(b for _, b in sp)
        pts = sorted({v for a, b in sp for v in (a, b) if lo < v < hi})
        out[k] = pts
    return out


def apply_to_project(project, path: str) -> List[tuple]:
    """Sets edit_points on every cut whose source file the XML uses. Returns [(cut name, points)]."""
    eps = read_edit_points(path)
    done = []
    for cut in project.cuts:
        pts = eps.get(_key(cut.source))
        if pts is None:
            continue
        n = cut.info.frames or 0
        cut.edit_points = [p for p in pts if 0 < p < n] if n else list(pts)
        done.append((cut.name, list(cut.edit_points)))
    return done
