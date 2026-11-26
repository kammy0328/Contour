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
RENDER_RESERVE_GB = 3.0            # what a render leaves free to the rest of the computer when it decides how many frames to retouch at once (0: no gate)


def process_gb() -> float:
    """The memory this process holds right now, in GB (Windows: its private bytes; elsewhere: what of it is in memory);
    -1 when it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.PrivateUsage / 2 ** 30
        elif sys.platform == "darwin":
            import ctypes

            class TV(ctypes.Structure):
                _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]

            class INFO(ctypes.Structure):
                _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64), ("resident_size_max", ctypes.c_uint64),
                            ("user_time", TV), ("system_time", TV), ("policy", ctypes.c_int), ("suspend_count", ctypes.c_int)]
            lib = ctypes.CDLL(None)
            info = INFO()
            n = ctypes.c_uint32(ctypes.sizeof(INFO) // 4)
            task = ctypes.c_uint32.in_dll(lib, "mach_task_self_")
            if lib.task_info(task, 20, ctypes.byref(info), ctypes.byref(n)) == 0:      # MACH_TASK_BASIC_INFO
                return info.resident_size / 2 ** 30
        else:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except Exception:
        pass
    return -1.0


class FrameGate:
    """How many frames a render retouches at once, by the memory the computer has left for them (1.27).

    Retouching a frame (engine.apply_planes) takes gigabytes for its time: with a body effect the whole 4K frame is worked on,
    3 to 4 GB a frame, and four at once took the 31 GB PC, 12 GB of it in use by other programs, to its last megabyte (IWALY
    c004, 6 people with body skin and knees: 6.4 GB between frames, +3.6, +4.7, +2.4, +3.0 for the first to the fourth frame
    retouched at once, 21.9 GB at the peak, under 2 GB free a sixth of the time; j2896a, j2903). One frame alone there takes
    13 s where each of four at once takes 32 s: the frames at once buy little time and cost four times the memory.

    So the render asks here before it retouches a frame. `slots` frames may be retouched at once (and one more may be
    prepared meanwhile): as many as the memory free when the first frame is prepared has room for (hint: what the render
    holds between frames and what a frame of this kind takes are guessed from the people, the frame's size and whether the
    whole frame is worked on), never more than the render's workers, never less than one. While it runs a watcher looks at
    the free memory four times a second: under LOW_GB a slot is taken away (the frames being retouched finish, the next
    one waits); after GROW_S seconds in which the least free memory stayed above the reserve and one and a half frames'
    worth, a slot is given back. Nothing changes in a frame: the same frames in the same order, fewer at once.

        with gate.frame():        # around the making of one frame, on its worker thread
            rf = prepare(...)
            gate.hint(whole_frame)
            with gate.apply():
                apply_planes(...)

    CONTOUR_FRAME_GATE=0 switches it off; CONTOUR_RESERVE_GB overrides RENDER_RESERVE_GB."""
    LOW_GB = 1.5            # under this much free memory a slot is taken away
    GROW_S = 20.0           # seconds of plenty before a slot is given back
    PER_MP = (0.25, 0.42)   # GB a megapixel that retouching a frame takes: on the people's pieces, on the whole frame

    def __init__(self, workers: int, people: int = 1, megapixels: float = 8.3, free=None, held=None, clock=None):
        import threading
        import time
        self.workers = max(1, int(workers))
        self.people = max(1, int(people))
        self.mp = max(0.1, float(megapixels))
        self._free = free or (lambda: free_gb(max_age=0.2))
        self._proc = held or process_gb
        self._clock = clock or time.monotonic
        self.base = self._proc()
        self.per = None                     # GB a frame's retouch is taken to need (set by hint)
        self.frames = 0
        self.applies = 0
        self.most = 0
        self.waited = 0
        self.shrunk = 0
        self.grown = 0
        self._hinted = False
        self._cond = threading.Condition()
        self._thread = None
        self._stop = False
        self._changed = self._clock()
        self._least = float("inf")          # the least free memory since the slots last changed
        e = os.environ.get("CONTOUR_RESERVE_GB", "").strip()
        try:
            self.reserve = float(e) if e else float(RENDER_RESERVE_GB)
        except ValueError:
            self.reserve = float(RENDER_RESERVE_GB)
        self.on = self.reserve > 0 and os.environ.get("CONTOUR_FRAME_GATE", "1") != "0" and self.workers > 1
        self.slots = 1 if self.on else self.workers          # one until the first frame says what kind they are
        self.first = self.slots

    def hint(self, whole: bool) -> None:
        """Called once a frame is prepared: `whole`: the retouch works on the whole frame (a body effect), not on the
        people's pieces. Sets the slots this computer has room for now."""
        if not self.on or self._hinted:
            return
        with self._cond:
            if self._hinted:
                return
            self._hinted = True
            self.per = self.PER_MP[1 if whole else 0] * self.mp
            between = 4.0 + 0.4 * self.people               # what the render holds between frames (pools, the masks kept, frames read ahead)
            free, p = self._free(), self._proc()
            if free < 0:
                n = self.workers
            else:
                took = max(0.0, p - self.base) if (p >= 0 and self.base >= 0) else 0.0
                room = free + took - self.reserve - between  # (what the render has taken so far is part of `between`)
                n = int(max(1, min(self.workers, room // self.per)))
            self.slots = self.first = n
            self._changed, self._least = self._clock(), float("inf")
            self._cond.notify_all()

    def look(self) -> None:
        """The watcher's step (also callable): a slot taken away when memory is short, given back after a time of plenty."""
        if not self.on or not self._hinted:
            return
        free = self._free()
        if free < 0:
            return
        now = self._clock()
        with self._cond:
            self._least = min(self._least, free)
            if free < self.LOW_GB and self.slots > 1 and now - self._changed >= 3.0:
                self.slots -= 1
                self.shrunk += 1
                self._changed, self._least = now, float("inf")
            elif (self.slots < self.workers and now - self._changed >= self.GROW_S
                  and self._least >= self.reserve + 1.5 * float(self.per or 0.0) and self.applies >= self.slots):
                self.slots += 1
                self.grown += 1
                self._changed, self._least = now, float("inf")
                self._cond.notify_all()

    def _watch(self) -> None:
        import time
        idle = 0
        while not self._stop and idle < 120:        # (ends by itself half a minute after the last frame)
            self.look()
            idle = idle + 1 if self.frames == 0 else 0
            time.sleep(0.25)

    def close(self) -> None:
        self._stop = True

    def _enter(self, what: str) -> None:
        if not self.on:
            return
        th = self._thread
        if th is None or not th.is_alive():
            import threading
            self._stop = False
            self._thread = threading.Thread(target=self._watch, name="frame-gate", daemon=True)
            self._thread.start()
        with self._cond:
            first = True
            while True:
                n = self.frames if what == "frames" else self.applies
                # frames: the ones retouched and one more being prepared (no limit before the first frame said what
                # kind they are: frames with nothing to retouch pass through)
                cap = (self.slots + 1 if self._hinted else self.workers) if what == "frames" else self.slots
                if n < max(1, cap):
                    break
                if first and what == "applies":
                    self.waited += 1
                first = False
                self._cond.wait(0.5)
            if what == "frames":
                self.frames += 1
            else:
                self.applies += 1
                self.most = max(self.most, self.applies)

    def _exit(self, what: str) -> None:
        if not self.on:
            return
        with self._cond:
            if what == "frames":
                self.frames -= 1
            else:
                self.applies -= 1
            self._cond.notify_all()

    def frame(self):
        return _Held(self, "frames")

    def apply(self):
        return _Held(self, "applies")

    def summary(self) -> str:
        if not self.on or not self._hinted or (self.first >= self.workers and not self.shrunk):
            return ""
        t = f"메모리에 맞춰 한 번에 {self.most}프레임까지 보정"
        if self.shrunk or self.grown:
            t += f", 도중에 {self.shrunk}번 줄이고 {self.grown}번 늘림"
        return t


class _Held:
    __slots__ = ("g", "what")

    def __init__(self, g, what):
        self.g, self.what = g, what

    def __enter__(self):
        self.g._enter(self.what)
        return self

    def __exit__(self, *exc):
        self.g._exit(self.what)
        return False


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
