"""The projects the project manager lists: what is known about each without opening it, renaming, the proxy
cache. Project folders end in .dpprj (.contour before 1.13; both open the same)."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from .project import PROJECT_EXT, PROJECT_EXTS, safe_name
from .settings import settings


def norm(path: str) -> str:
    p = Path(path)
    return str(p.parent if p.is_file() else p)


def known() -> List[str]:
    """Every project the manager lists: the known list, then recent ones not in it (no duplicates)."""
    s = settings()
    out: List[str] = []
    for p in list(s.known_projects) + list(s.recent_projects) + ([s.last_project] if s.last_project else []):
        p = norm(p)
        if p and p not in out:
            out.append(p)
    return out


def remember(path: str) -> None:
    s = settings()
    path = norm(path)
    if path in s.known_projects:
        s.known_projects.remove(path)
    s.known_projects.insert(0, path)
    s.save()


def forget(path: str) -> None:
    s = settings()
    path = norm(path)
    s.known_projects = [p for p in known() if p != path]
    s.recent_projects = [p for p in s.recent_projects if norm(p) != path]
    if norm(s.last_project or "") == path:
        s.last_project = ""
    s.save()


def _replace_known(old: str, new: str) -> None:
    s = settings()
    s.known_projects = [new if p == old else p for p in known()]
    s.recent_projects = [new if norm(p) == old else p for p in s.recent_projects]
    if norm(s.last_project or "") == old:
        s.last_project = new
    s.save()


def info(path: str) -> Dict:
    """Read from project.json only (fast): name, client, times, cuts, analysed cuts, done cuts."""
    f = Path(path) / "project.json"
    d: Dict = {"path": path, "exists": f.exists(), "name": Path(path).stem}
    if not f.exists():
        return d
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        d["broken"] = True
        return d
    cuts = [c for c in j.get("cuts", []) if (c.get("tl") or {}).get("kind", "clip") == "clip"]
    d.update(name=j.get("name") or d["name"], client=j.get("client", ""),
             created=float(j.get("created", 0) or 0), modified=f.stat().st_mtime,
             cuts=len(cuts), analyzed=sum(1 for c in cuts if c.get("analyzed")),
             done=sum(1 for c in cuts if (c.get("status") or {}).get("work") == "done"),
             first_cut=(cuts[0].get("id") if cuts else ""))
    return d


def local_cache(path: str) -> Optional[Path]:
    """The project's folder in this computer's proxy cache (1.16), if there is one."""
    try:
        j = json.loads((Path(path) / "project.json").read_text(encoding="utf-8"))
        key = f"{int(float(j.get('created', 0) or 0))}"
        from .settings import settings
        root = settings().cache_path
        return next((p for p in root.glob(key + "_*") if p.is_dir()), None)
    except Exception:
        return None


def _proxy_files(path: str) -> List[Path]:
    root = Path(path)
    out = [p for p in root.glob("cuts/*/cache/proxy_*.u8")] + [p for p in root.glob("cuts/*/cache/proxy_meta.json")]
    out += [p for p in root.glob("cuts/*/cache/proxy_shared.json")]
    bc = root / "bake_cache"
    if bc.exists():
        out += [p for p in bc.rglob("*") if p.is_file()]
    lc = local_cache(path)
    if lc is not None:
        out += [p for p in lc.rglob("*") if p.is_file()]
    return out


def sizes(path: str) -> Dict[str, int]:
    """Bytes in the project folder, and of that the proxies (made again when a cut is opened)."""
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    prox = 0
    for p in _proxy_files(path):
        try:
            prox += p.stat().st_size
        except OSError:
            pass
    return {"total": total, "proxy": prox}


def clear_proxies(path: str) -> int:
    """Deletes the proxy frames (analysis, mattes and edits stay); opening a cut makes its proxy again."""
    freed = 0
    for p in _proxy_files(path):
        try:
            n = p.stat().st_size
            p.unlink()
            freed += n
        except OSError:
            pass
    bc = Path(path) / "bake_cache"
    if bc.exists():
        shutil.rmtree(bc, ignore_errors=True)
    lc = local_cache(path)
    if lc is not None:
        shutil.rmtree(lc, ignore_errors=True)
    return freed


def cache_usage() -> Dict:
    """This computer's proxy cache: folder, bytes used, free space on its drive, and the folders of projects that are
    no longer in the project list."""
    from .settings import settings
    st = settings()
    root = st.cache_path
    used = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                used += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    keys = set()
    for p in list(st.known_projects) + list(st.recent_projects):
        try:
            j = json.loads((Path(p) / "project.json").read_text(encoding="utf-8"))
            keys.add(f"{int(float(j.get('created', 0) or 0))}")
        except Exception:
            pass
    orphans = [d for d in root.iterdir() if d.is_dir() and d.name.split("_", 1)[0] not in keys] if root.exists() else []
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        free = -1
    return dict(root=str(root), used=used, free=free, orphans=orphans)


def clear_orphans(dirs) -> int:
    freed = 0
    for d in dirs:
        for dp, _, fs in os.walk(d):
            for f in fs:
                try:
                    freed += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        shutil.rmtree(d, ignore_errors=True)
    return freed


def rename(path: str, new_name: str) -> str:
    """The project's name and its folder (<name>.dpprj, beside the old one). Returns the new folder."""
    old = Path(norm(path))
    nm = safe_name(new_name)
    dst = old.parent / f"{nm}{PROJECT_EXT}"
    if dst.exists() and dst.resolve() != old.resolve():
        raise FileExistsError(f"같은 이름의 프로젝트 폴더가 이미 있습니다: {dst.name}")
    f = old / "project.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    j["name"] = nm
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)
    if dst != old:
        os.replace(old, dst)          # same drive: a rename, nothing copied
    _replace_known(str(old), str(dst))
    return str(dst)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit in ("B", "KB") else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def when(t: float) -> str:
    if not t:
        return ""
    d = time.time() - t
    if d < 60:
        return "방금"
    if d < 3600:
        return f"{int(d // 60)}분 전"
    if d < 86400:
        return f"{int(d // 3600)}시간 전"
    if d < 86400 * 7:
        return f"{int(d // 86400)}일 전"
    return time.strftime("%Y-%m-%d", time.localtime(t))


def is_project_dir(path: str) -> bool:
    return (Path(path) / "project.json").exists()


def ext_ok(path: str) -> bool:
    return Path(path).suffix.lower() in PROJECT_EXTS
