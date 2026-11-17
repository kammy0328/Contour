"""Updates from the public release page (the source stays private).

The app asks GitHub for the latest release of RELEASE_REPO. A newer version is only offered: nothing is
downloaded or installed until the user says so. The installer is checked against the SHA-256 published with
the release before it runs, and it is kept (the last few) so an older version can be installed again.
Projects, learned data, presets and models live in the data folder, outside the installed app, so an update
never touches them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .settings import APP_VERSION, settings

RELEASE_REPO = "kammy0328/Contour-Releases"
API = f"https://api.github.com/repos/{RELEASE_REPO}/releases/latest"
KEEP = 3


@dataclass
class Release:
    version: str
    notes: str
    setup_url: str
    setup_name: str
    sha256: str
    size: int


def _ver(s: str):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3]) or (0,)


def is_newer(v: str, than: str = APP_VERSION) -> bool:
    return _ver(v) > _ver(than)


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"DIAP/{APP_VERSION}",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest() -> Optional[Release]:
    d = json.loads(_get(API).decode("utf-8"))
    tag = d.get("tag_name", "")
    assets = {a["name"]: a for a in d.get("assets", [])}
    setup = next((a for n, a in assets.items() if n.lower().endswith(".exe") and "setup" in n.lower()), None)
    if setup is None:
        return None
    sha = ""
    side = assets.get(setup["name"] + ".sha256")
    if side is not None:
        sha = _get(side["browser_download_url"]).decode("utf-8", "ignore").split()[0].strip().lower()
    if not sha:
        m = re.search(r"SHA256[^0-9a-fA-F]*([0-9a-fA-F]{64})", d.get("body") or "")
        sha = m.group(1).lower() if m else ""
    return Release(version=tag.lstrip("vV"), notes=d.get("body") or "", setup_url=setup["browser_download_url"],
                   setup_name=setup["name"], sha256=sha, size=int(setup.get("size") or 0))


def check() -> Optional[Release]:
    """The latest release when it is newer than this app and not skipped by the user, else None."""
    r = latest()
    if r is None or not is_newer(r.version):
        return None
    if getattr(settings(), "skip_version", "") == r.version:
        return None
    return r


def updates_dir() -> Path:
    p = settings().data_path / "updates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(r: Release, progress: Optional[Callable[[float, str], None]] = None,
             cancelled: Optional[Callable[[], bool]] = None) -> Path:
    dst = updates_dir() / r.setup_name
    if dst.exists() and r.sha256 and _sha(dst) == r.sha256:
        return dst
    req = urllib.request.Request(r.setup_url, headers={"User-Agent": f"DIAP/{APP_VERSION}"})
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, os.fdopen(fd, "wb") as f:
            total = int(resp.headers.get("Content-Length") or r.size or 0)
            got = 0
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("취소됨")
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if progress and total:
                    progress(got / total, f"업데이트 받는 중 {got / 1e6:.0f}/{total / 1e6:.0f} MB")
        if not r.sha256:
            raise RuntimeError("릴리스에 SHA-256 값이 없어 설치 파일을 확인할 수 없습니다")
        if h.hexdigest().lower() != r.sha256:
            raise RuntimeError("받은 설치 파일이 릴리스의 SHA-256과 다릅니다 (손상되었거나 바뀐 파일)")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    _prune()
    return dst


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().lower()


def _prune():
    files = sorted(updates_dir().glob("*.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass


def installed_dir() -> Optional[Path]:
    """{app} of an installed build (…\\Diap\\app\\app.py next to …\\Diap\\python), else None (source run)."""
    here = Path(__file__).resolve().parents[2]
    root = here.parent
    if (root / "python" / "python.exe").exists() and here.name == "app":
        return root
    return None


def launch_installer(setup: Path) -> None:
    """Runs the installer silently once this app has closed; it starts Contour again when it is done."""
    args = [str(setup), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/UPDATE=1"]
    # no /DIR: the installer keeps an earlier DIAP install's folder, and moves a 1.8 per-user install to
    # Program Files\Diap
    if sys.platform != "win32":
        subprocess.Popen(args, close_fds=True)
        return
    # a few seconds' wait first: this app's Python must have exited before its files are replaced.
    # 'start' goes through ShellExecute: the installer (Program Files, needs admin) gets its UAC prompt.
    # The command line is given to cmd as one string (cmd /s /c "...") so its quotes reach cmd as they are:
    # passed as a list, Python escapes them as \" which cmd does not read (1.9.0: Windows said '\\' not found).
    # CREATE_NO_WINDOW without DETACHED_PROCESS: cmd and its ping share one hidden console, nothing shows.
    inner = 'ping -n 5 127.0.0.1 >nul & start "" ' + " ".join(f'"{a}"' for a in args)
    line = f'cmd.exe /d /s /c "{inner}"'
    flags = 0x00000200 | 0x08000000                   # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    subprocess.Popen(line, close_fds=True, creationflags=flags)
