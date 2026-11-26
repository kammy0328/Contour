"""How much of the computer a render may use side by side, from its memory and cores (1.26.3).

The PC batches of 2611 IWALY rendered with the people of a frame prepared side by side (engine PREP_THREADS 6, MASK_THREADS 6),
ffmpeg's readers decoding on one thread each and the ProRes writer on three; the app kept all of it at one, and a 4K render of
LNGSHOT 024 (4 faces) took 17.9 s a frame where the batch's settings gave 11.2 s, frame for frame the same file (j2847, j2848).
Side by side costs memory (024: 9.7 -> 12.5 GB at the peak), so how far a computer goes follows its RAM:

    RAM >= 28 GB: 4 people at once, a person's masks 2 to 4 at once (12 // people)
    RAM >= 20 GB: 2 people, 2 masks
    less: one after the other, as before 1.26.3

CONTOUR_PREP_THREADS / CONTOUR_MASK_THREADS override (a batch, a test); settings().render_parallel = "off" keeps one after the other."""
from __future__ import annotations

import os
import sys
from typing import Optional

_RAM: list = []


def ram_gb() -> float:
    """The computer's physical memory in GB (0 when it cannot be read)."""
    if _RAM:
        return _RAM[0]
    v = 0.0
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                v = m.ullTotalPhys / 2 ** 30
        elif sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            v = int(r.stdout.strip() or 0) / 2 ** 30
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        v = int(line.split()[1]) / 2 ** 20
                        break
    except Exception:
        v = 0.0
    _RAM.append(float(v))
    return float(v)


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name, "")
    try:
        return int(v) if v.strip() else None
    except ValueError:
        return None


def _off() -> bool:
    try:
        from .settings import settings
        return str(getattr(settings(), "render_parallel", "auto") or "auto") == "off"
    except Exception:
        return False


def prep_threads(ram: Optional[float] = None, cores: Optional[int] = None) -> int:
    """People of one frame prepared side by side (engine.PREP_THREADS when it is not set)."""
    e = _env_int("CONTOUR_PREP_THREADS")
    if e is not None:
        return max(1, e)
    if _off():
        return 1
    ram = ram_gb() if ram is None else float(ram)
    cores = int(cores or os.cpu_count() or 1)
    if ram >= 28 and cores >= 8:
        return 4
    if ram >= 20 and cores >= 4:
        return 2
    return 1


def mask_threads(people: int, ram: Optional[float] = None, cores: Optional[int] = None) -> int:
    """One person's masks made side by side (engine.MASK_THREADS when it is not set); 0 or 1: one after the other."""
    e = _env_int("CONTOUR_MASK_THREADS")
    if e is not None:
        return max(0, e)
    if _off():
        return 0
    ram = ram_gb() if ram is None else float(ram)
    cores = int(cores or os.cpu_count() or 1)
    if ram >= 28 and cores >= 8:
        return max(2, min(4, 12 // max(1, int(people))))
    if ram >= 20 and cores >= 4:
        return 2
    return 0


def ff_decode_threads() -> int:
    """Threads of one ffmpeg reader (-threads before -i). Many readers run at once in a render (the frame, a piece of each face and
    body): left to themselves each takes every core and they crowd the retouch out (the batch: 1)."""
    e = _env_int("CONTOUR_FF_DEC_THREADS")
    return max(1, e) if e is not None else 1


def ff_encode_threads() -> int:
    """Threads of the ProRes writer (prores_ks -threads) while the retouch runs beside it (the batch: 3)."""
    e = _env_int("CONTOUR_FF_ENC_THREADS")
    return max(1, e) if e is not None else 3


def summary() -> str:
    return f"RAM {ram_gb():.0f} GB, 코어 {os.cpu_count()}: 사람 {prep_threads()}명 동시"
