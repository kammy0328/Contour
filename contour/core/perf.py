"""How much of the computer a render may use side by side, from its memory and cores (1.26.3).

The PC batches of 2611 IWALY rendered with the people of a frame prepared side by side (engine PREP_THREADS 6, MASK_THREADS 6),
ffmpeg's readers decoding on one thread each and the ProRes writer on three; the app kept all of it at one, and a 4K render of
LNGSHOT 024 (4 faces) took 17.9 s a frame where the batch's settings gave 11.2 s, frame for frame the same file (j2847, j2848).
Side by side costs memory (024: 9.7 -> 12.5 GB at the peak), so how far a computer goes follows its RAM:

    RAM >= 28 GB: 4 people at once, a person's masks 2 to 4 at once (12 // people)
    RAM >= 20 GB: 2 people, 2 masks
    less: one after the other, as before 1.26.3

CONTOUR_PREP_THREADS / CONTOUR_MASK_THREADS override (a batch, a test); settings().render_parallel = "off" keeps one after the other.
(1.27) raw_ahead: the next frames' own masks made ahead of the render's serial stage, where there is memory and cores for it."""
from __future__ import annotations

import os
import sys
from typing import Optional

_RAM: list = []


def _memory() -> tuple:
    """(total, available) physical memory in GB, asked now ((0, -1) when it cannot be read)."""
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
                return m.ullTotalPhys / 2 ** 30, m.ullAvailPhys / 2 ** 30
        elif sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            total = int(r.stdout.strip() or 0) / 2 ** 30
            avail = -1.0
            try:
                v = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
                page = 4096
                n = 0
                for line in v.splitlines():
                    if "page size of" in line:
                        page = int("".join(ch for ch in line.split("page size of")[1] if ch.isdigit()) or 4096)
                    elif line.split(":")[0].strip() in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
                        n += int(line.split(":")[1].strip().rstrip("."))
                avail = n * page / 2 ** 30
            except Exception:
                avail = -1.0
            return total, avail
        else:
            total, avail = 0.0, -1.0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 2 ** 20
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 2 ** 20
            return total, avail
    except Exception:
        pass
    return 0.0, -1.0


def ram_gb() -> float:
    """The computer's physical memory in GB (0 when it cannot be read)."""
    if _RAM:
        return _RAM[0]
    v = float(_memory()[0])
    _RAM.append(v)
    return v


_FREE: list = [0.0, -1.0]          # when asked (monotonic), the answer


def free_gb(max_age: float = 1.0) -> float:
    """The physical memory nothing uses right now, in GB (-1 when it cannot be read): what the other programs leave, which
    the installed memory does not tell (LNGSHOT 024 on the 31 GB PC, 16 GB of it in use by other programs: a render with
    frames made ahead took the computer to 30.7 GB, j2895b). Asked at most once a max_age seconds."""
    import time
    now = time.monotonic()
    if _FREE[1] < 0 or now - _FREE[0] >= max_age:
        _FREE[0], _FREE[1] = now, float(_memory()[1])
    return _FREE[1]


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


RAW_AHEAD_MIN_FREE_GB = 4.0        # frames are made ahead only while this much physical memory is free (0: always)
RENDER_MIN_FREE_GB = 2.0           # with less free, a render starts no further frame beside the ones in the making (0: never held)


def short_of_memory(need: Optional[float] = None) -> bool:
    """The computer has less than `need` GB (RENDER_MIN_FREE_GB) of physical memory free right now. False when that
    cannot be read. CONTOUR_MIN_FREE_GB overrides (0: never)."""
    e = os.environ.get("CONTOUR_MIN_FREE_GB", "").strip()
    try:
        need = float(e) if e else (RENDER_MIN_FREE_GB if need is None else float(need))
    except ValueError:
        need = RENDER_MIN_FREE_GB
    if need <= 0:
        return False
    free = free_gb(max_age=0.5)
    return 0 <= free < need


def raw_ahead(people: int = 1, ram: Optional[float] = None, cores: Optional[int] = None) -> tuple:
    """(frames, threads): the next frames' own masks made ahead of the render's serial stage (retouch/rawahead.py, 1.27).
    A frame in the making takes about a gigabyte and as many threads as it has people, so only where both are there:

        RAM >= 28 GB and 16 cores or more: 3 frames ahead on 2 threads
        less: none, as before 1.27

    LNGSHOT 024 (4 faces, 4K LogC4, 36 frames, the 18-core 31 GB PC, 11/26): none 9.41 s a frame, 3 on 2 threads 7.90, every
    frame of the file the same (j2895). 2 on 1 thread 9.98: one thread makes a frame no faster than the render itself and
    takes cores from the frames being finished, so the serial stage waited on it; there is no tier with one thread.
    The computer's memory went to 30.7 of 31 GB in that run (16 GB of it other programs'), so while the render runs frames are
    made ahead only as long as RAW_AHEAD_MIN_FREE_GB of memory is free (rawahead.RawAhead.take).

    CONTOUR_RAW_AHEAD "frames" or "frames:threads" overrides (0: none); render_parallel off: none."""
    v = os.environ.get("CONTOUR_RAW_AHEAD", "").strip()
    if v:
        try:
            a, _, b = v.partition(":")
            n = max(0, int(a))
            return (n, max(1, int(b)) if b.strip() else 2) if n else (0, 0)
        except ValueError:
            pass
    if _off():
        return 0, 0
    ram = ram_gb() if ram is None else float(ram)
    cores = int(cores or os.cpu_count() or 1)
    if ram >= 28 and cores >= 16:
        return 3, 2
    return 0, 0


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
    n, k = raw_ahead()
    return f"RAM {ram_gb():.0f} GB, 코어 {os.cpu_count()}: 사람 {prep_threads()}명 동시" + (f", 다음 {n}프레임 미리" if n else "")
