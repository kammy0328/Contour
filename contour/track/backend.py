"""Where a model runs (instruction 2026-11-20 §5): decided in one place for every model by the system, the device,
the model and what 설정 > 추론 장치 asks; checked by what a run actually did, not by the name of a provider; and
told with its reason (settings, the diagnostics report, the AI review report).

- The backends of a system, in the order they are tried: Windows CUDA -> DirectML -> CPU, macOS Core ML -> CPU,
  Linux CUDA -> CPU; only those the installed onnxruntime offers. Each model's own list narrows that: a model
  package says which backends it is for (modelpkg), Core ML takes only the models listed in COREML_OK.
- A session's list of providers is what it may use. What it did use is read from the first run's own record of
  which provider ran each operator (placement): 'CUDA' with every operator on the CPU is the CPU.
- A model that ends on the CPU although an accelerator was asked for has a reason: the driver, the libraries that
  did not load, no memory, a model that is not for that backend, operators the backend does not have.

Nothing here opens a model: models.session does, and reports to this module."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RTX_EP = "NvTensorRTRTXExecutionProvider"       # NVIDIA's TensorRT for RTX: a plugin of onnxruntime, used beside CUDA (see rtx() below)
PROVIDER = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "coreml": "CoreMLExecutionProvider",
            "cpu": "CPUExecutionProvider", "rtx": RTX_EP}
BACKEND = {v: k for k, v in PROVIDER.items()}
LABEL = {"cuda": "CUDA", "dml": "DirectML", "coreml": "Core ML", "cpu": "CPU", "rtx": "TensorRT RTX"}
ORDER = {"win32": ("cuda", "dml", "cpu"), "darwin": ("coreml", "cpu")}
ORDER_OTHER = ("cuda", "cpu")

# The combination the Windows build is fixed to and checked with (requirements.txt pins the runtime; its CUDA and
# cuDNN libraries come with it as pip packages, not from the system):
#   onnxruntime-gpu 1.24.4  =  CUDA 12.8 + cuDNN 9.x      (ONNX Runtime's CUDA provider requirements)
#   NVIDIA driver: 528.33 is the least a CUDA 12.x program runs with on Windows (525.60.13 on Linux); 572.30 is the
#   driver CUDA 12.8 came out with (NVIDIA CUDA 12.8 release notes). Below the first CUDA does not start; between
#   the two it runs but is a combination nobody checked here.
ORT_PINNED = "1.24.4"
CUDA_RUNTIME, CUDNN_RUNTIME = "12.8", "9"
DRIVER_MIN = {"win32": (528, 33), "linux": (525, 60)}
DRIVER_BUILT = {"win32": (572, 30), "linux": (570, 117)}

# Core ML: with 추론 장치 '자동' only the models listed here are given to it, each with the options it was checked
# with. A model is listed after tools/compare_backends.py ran it on a real Apple Silicon Mac with the same input on
# Core ML and on the CPU and the outputs agreed within the tolerance of the tests and it was faster. None is
# listed yet (no Mac was at hand: 미검증), so a Mac runs every model on the CPU unless 추론 장치 is set to
# 'Core ML', which tries it for every model that is not in COREML_NO and falls back per model.
COREML_OK: Dict[str, dict] = {}
# models Core ML is not tried for at all: their graphs use operators or shapes it does not take (dynamic
# prompts, external weights over its size limits), found by trying or known from the export
COREML_NO = ("seg_sam31", "seg_sam31_find", "seg_sam2_s", "seg_sam2_b", "track_sam2v")
COREML_OPTIONS = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "0",
                  "EnableOnSubgraphs": "0"}


def coreml_files() -> Dict[str, dict]:
    """Model FILES given to Core ML even when their model is in COREML_NO or not in COREML_OK (environment
    DIAP_COREML_FILES: file names separated by commas, e.g. "image-encoder-gpu.onnx"; DIAP_COREML_UNITS: ALL,
    CPUAndGPU, CPUAndNeuralEngine; DIAP_COREML_CACHE: a folder for the compiled model). Set by the Mac batch after
    the file was checked on that Mac against the CPU (iwaly_batch/coreml.json). Empty by default: nothing changes."""
    names = [n.strip() for n in os.environ.get("DIAP_COREML_FILES", "").split(",") if n.strip()]
    if not names or sys.platform != "darwin":
        return {}
    o: dict = {}
    if os.environ.get("DIAP_COREML_UNITS"):
        o["MLComputeUnits"] = os.environ["DIAP_COREML_UNITS"]
    if os.environ.get("DIAP_COREML_FORMAT"):
        o["ModelFormat"] = os.environ["DIAP_COREML_FORMAT"]
    if os.environ.get("DIAP_COREML_CACHE"):
        o["ModelCacheDirectory"] = os.environ["DIAP_COREML_CACHE"]
    return {n: dict(o) for n in names}

# ------------------------------------------------------------------------------------------------- TensorRT for RTX
# (1.27) NVIDIA's TensorRT for RTX execution provider: a plugin of onnxruntime (pip onnxruntime-ep-nv-tensorrt-rtx-cu12, put in by
# tools/setup_env.py on a computer whose GPU it runs on: RTX 30 series and later), used beside the CUDA runtime, never instead of it. Only
# the model FILES listed here are opened on it, each after it was run on both with the same input and what the analysis makes of the outputs
# agreed. A file not listed, no plugin, no device, a session that does not open or a run that fails: CUDA, as before.
#   image-encoder-gpu.onnx  SAM 3.1's image encoder (42 % of an analysis): 0.49 -> 0.076 s a picture on an RTX 4070 Ti, the people's labels of a
#                           whole SAM pass IoU 0.9994 to 0.9999 against CUDA's, the matte more than 32/255 off on 0.03 % of the people's pixels,
#                           GPU memory of the pass 10 -> 5.7 GB (LNGSHOT 012 and 024, j2875 / j2885, 11/26). TensorRT 10's own fp16 attention
#                           is wrong on this graph (j2861..j2867): that provider is not used.
# Off unless the user switches it on (settings().rtx, 설정 > TensorRT RTX 가속): the mattes agree to IoU 0.9995, but what the analysis decides
# from them where it is in doubt does not always (LNGSHOT 024, j2897: a face at the frame's edge known on 26 frames instead of 32, joints a
# couple of pixels off on a quarter of the points, far off where a limb is hidden; 012: faces the same). Every model on it ("*") was slower
# than the encoder alone (024: 103 s against 88, 012: 108 against 85): only the encoder is listed.
RTX_OK: Dict[str, dict] = {"image-encoder-gpu.onnx": {}}
RTX_PACKAGE = "onnxruntime-ep-nv-tensorrt-rtx-cu12"
RTX_VERSION = "0.5.0"                   # the version checked (setup_env.py installs this one)
_rtx: Optional[dict] = None


def rtx_files() -> Dict[str, dict]:
    """The model files TensorRT RTX is used for: RTX_OK, or what DIAP_RTX_FILES names (file names separated by commas,
    '*' for every model: for trying a model on it before it is listed)."""
    v = os.environ.get("DIAP_RTX_FILES", "").strip()
    if not v:
        return RTX_OK
    if v == "*":
        return {"*": {}}
    return {n.strip(): {} for n in v.split(",") if n.strip()}


def rtx(refresh: bool = False) -> dict:
    """The TensorRT RTX plugin on this computer, looked at once: {"devices": the GPUs it runs on ([]: not used), "why": the reason
    when there is none, "version": the plugin's, "name": the provider's name}. Nothing is opened here; the plugin's library is
    given to onnxruntime (once a process) and asked which devices it takes."""
    global _rtx
    with _lock:
        if _rtx is not None and not refresh:
            return _rtx
        d = dict(devices=[], why="", version="", name=RTX_EP, state="off")     # state: ok | off | no_plugin | no_device | error
        try:
            if system() == "darwin":
                d["why"] = "macOS에는 없음"
            elif os.environ.get("DIAP_RTX", "1") == "0":
                d["why"] = "꺼짐 (DIAP_RTX=0)"
            else:
                import onnxruntime as ort
                if not hasattr(ort, "register_execution_provider_library") or not hasattr(ort, "get_ep_devices"):
                    d["why"] = f"ONNX Runtime {ort.__version__}에는 플러그인 실행 장치가 없음"
                    d["state"] = "error"
                else:
                    try:
                        import onnxruntime_ep_nv_tensorrt_rtx as ep
                        lib_of, name_of = ep.get_library_path, ep.get_ep_name
                    except (ImportError, AttributeError):
                        ep = None
                        d["state"] = "no_plugin"
                        d["why"] = "TensorRT RTX 구성요소가 설치되어 있지 않음 (설치 파일을 다시 실행하면 RTX 30 이상에서 설치됩니다)"
                    if ep is not None:
                        from . import models as M
                        M._preload_cuda()                       # the plugin uses the CUDA runtime's libraries (pip's nvidia-* wheels)
                        lib, name = str(lib_of()), str(name_of())
                        d["name"] = name
                        try:
                            from importlib import metadata
                            d["version"] = metadata.version(RTX_PACKAGE)
                        except Exception:
                            d["version"] = str(getattr(ep, "__version__", ""))
                        if os.name == "nt":
                            try:
                                os.add_dll_directory(os.path.dirname(lib))
                            except Exception:
                                pass
                            os.environ["PATH"] = os.path.dirname(lib) + os.pathsep + os.environ.get("PATH", "")
                        try:
                            ort.register_execution_provider_library(name, lib)
                        except Exception as e:
                            if "already" not in str(e).lower():
                                raise
                        d["devices"] = [x for x in ort.get_ep_devices() if x.ep_name == name]
                        PROVIDER["rtx"] = name
                        BACKEND[name] = "rtx"
                        d["state"] = "ok" if d["devices"] else "no_device"
                        if not d["devices"]:
                            d["why"] = "이 GPU는 TensorRT RTX 대상이 아님 (RTX 30 시리즈 이상)"
        except Exception as e:
            d["devices"] = []
            d["state"] = "error"
            d["why"] = f"TensorRT RTX를 불러오지 못함 ({str(e)[:160]})"
        _rtx = d
        return d


def rtx_brief() -> str:
    """A short line for the settings, without loading anything: whether the plugin is installed, and what an analysis of this
    run has found of it ('' on a Mac: nothing to say). 진단 tells the rest."""
    if system() == "darwin":
        return ""
    r = _rtx
    if r is not None:
        if r.get("devices"):
            return f"사용 가능 (TensorRT RTX {r.get('version', '')})".replace(" )", ")")
        return str(r.get("why") or "쓰지 않음")
    try:
        import importlib.util
        have = importlib.util.find_spec("onnxruntime_ep_nv_tensorrt_rtx") is not None
    except Exception:
        have = False
    if not have:
        return "구성요소 없음 (RTX 30 이상: 설치 파일을 다시 실행하면 설치됩니다)"
    return "설치됨 · 분석할 때 이 GPU에서 쓸 수 있는지 확인합니다"


def rtx_for(path, want: str = "auto", allow: str = "auto") -> Tuple[list, str]:
    """(devices, why not) for one model file: the TensorRT RTX devices its session is given when the file is one checked on it
    (rtx_files) and the settings allow it: 추론 장치 자동 or CUDA, 'TensorRT RTX 가속' not switched off. ([], "") for a file that is
    not listed: nothing to say."""
    files = rtx_files()
    name = Path(str(path)).name
    if "*" not in files and name not in files:
        return [], ""
    if want not in ("auto", "cuda"):
        return [], ""
    if str(allow or "auto") == "off":
        return [], "TensorRT RTX 가속이 설정에서 꺼져 있음"
    r = rtx()
    return list(r["devices"]), ("" if r["devices"] else r["why"])


# models that are not moved to the CPU when the GPU has no memory left for them: one run there takes minutes (the
# SAM 3.1 encoder: 206 s a picture on 16 threads), so the analysis says what to do instead of seeming to hang
CPU_TOO_SLOW = ("image-encoder", "finder", "vision_encoder", "memory_attention")

_lock = threading.RLock()
_records: Dict[str, dict] = {}            # model file name -> what was asked, what ran, why
_gpu: Optional[dict] = None


# ------------------------------------------------------------------------------------------------- the system
def system() -> str:
    return "win32" if sys.platform == "win32" else "darwin" if sys.platform == "darwin" else "linux"


def offered() -> List[str]:
    """The backends the installed onnxruntime can be asked for, in this system's order."""
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        avail = set()
    order = ORDER.get(system(), ORDER_OTHER)
    return [b for b in order if PROVIDER[b] in avail or b == "cpu"]


def runtime_packages() -> List[Tuple[str, str]]:
    """The onnxruntime distributions installed in this environment. More than one (onnxruntime beside
    onnxruntime-gpu or -directml) share one module folder and break each other: told in the diagnostics."""
    out = []
    try:
        from importlib import metadata
        for d in metadata.distributions():
            n = (d.metadata.get("Name") or "").lower()
            if n == "onnxruntime" or n.startswith("onnxruntime-") or n.startswith("onnxruntime_"):
                if n.replace("_", "-") not in ("onnxruntime-extensions", "onnxruntime-tools", "onnxruntime-genai"):
                    out.append((n.replace("_", "-"), d.version))
    except Exception:
        pass
    return sorted(set(out))


def _ver(text: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(text))[:3])


def gpu(refresh: bool = False) -> dict:
    """The graphics hardware as far as it can be read without opening a model: NVIDIA name, memory (MB) and
    driver through nvidia-smi; on a Mac the chip and the memory it shares. {} when nothing answers."""
    global _gpu
    with _lock:
        if _gpu is not None and not refresh:
            return dict(_gpu)
        d: dict = {}
        if system() == "darwin":
            try:
                chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                      timeout=5).stdout.strip()
                mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                         timeout=5).stdout.strip() or 0)
                d = dict(vendor="apple", name=chip, memory_mb=mem // (1 << 20), driver=platform.mac_ver()[0],
                         arch=platform.machine())
            except Exception:
                d = {}
        else:
            try:
                kw = dict(capture_output=True, text=True, timeout=8)
                if os.name == "nt":
                    kw["creationflags"] = 0x08000000           # no console window
                r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                                    "--format=csv,noheader,nounits"], **kw)
                rows = [[c.strip() for c in ln.split(",")] for ln in (r.stdout or "").splitlines() if ln.strip()]
                if r.returncode == 0 and rows and len(rows[0]) >= 4:
                    d = dict(vendor="nvidia", name=rows[0][0], memory_mb=int(float(rows[0][1])),
                             used_mb=int(float(rows[0][2])), driver=rows[0][3], count=len(rows))
            except Exception:
                d = {}
        _gpu = d
        return dict(d)


def driver_state(info: Optional[dict] = None) -> Tuple[str, str]:
    """('ok' | 'old' | 'unchecked' | 'unknown', text) for the NVIDIA driver against the fixed combination."""
    info = gpu() if info is None else info
    if info.get("vendor") != "nvidia" or not info.get("driver"):
        return "unknown", ""
    have = _ver(info["driver"])[:2]
    lo, built = DRIVER_MIN.get(system()), DRIVER_BUILT.get(system())
    if lo and have < lo:
        return "old", (f"NVIDIA 드라이버 {info['driver']}: CUDA {CUDA_RUNTIME} 실행에는 {lo[0]}.{lo[1]:02d} 이상이 필요합니다. "
                       "드라이버를 업데이트하면 GPU로 돌아갑니다")
    if built and have < built:
        return "unchecked", (f"NVIDIA 드라이버 {info['driver']}: 동작은 하지만 확인된 조합({built[0]}.{built[1]:02d} 이상)이 "
                             "아닙니다")
    return "ok", ""


# ------------------------------------------------------------------------------------------------- per model
def model_name(path) -> str:
    """How a model is named in the records and the lists above: its key when it is one of models.MODELS (found by
    its folder or file name), else its file name."""
    p = Path(str(path))
    try:
        from . import models as M
        for cand in (p.stem, p.parent.name, p.parent.parent.name):
            if cand in M.MODELS:
                return cand
    except Exception:
        pass
    return p.name


def package_backends(path) -> Tuple[str, ...]:
    """The backends a model package says its model is for (() when it says nothing or is no package)."""
    try:
        from . import modelpkg as MP
        own = MP.store().owns(path)
        if own:
            pkg = MP.store().package(*own)
            return tuple(pkg.backends) if pkg else ()
    except Exception:
        pass
    return ()


def for_model(path, base: List, want: str = "auto") -> Tuple[List, List[str]]:
    """The providers this model's session is asked for, from the list decided for the system (base, as
    models.providers gives it), and why a backend was taken out. The CPU is always the last one."""
    name = model_name(path)
    allowed = package_backends(path)
    out, why = [], []
    for p in base:
        pname, opts = (p, None) if isinstance(p, str) else (p[0], p[1])
        b = BACKEND.get(pname)
        if b is None or b == "cpu":
            continue
        if allowed and b not in allowed:
            why.append(f"{LABEL[b]}: 이 모델 패키지의 대상이 아님 ({', '.join(LABEL.get(x, x) for x in allowed)})")
            continue
        if b == "coreml":
            forced = coreml_files().get(Path(str(path)).name)
            if forced is not None:
                out.append((pname, {**COREML_OPTIONS, **(opts or {}), **forced}))
                continue
            if name in COREML_NO:
                why.append("Core ML: 이 모델은 지원하지 않는 연산·입력이 있어 CPU로 실행")
                continue
            if want != "coreml" and name not in COREML_OK:
                why.append("Core ML: 이 모델은 품질·속도 확인 전이라 CPU로 실행 (설정에서 Core ML을 고르면 시도)")
                continue
            p = (pname, {**COREML_OPTIONS, **(opts or {}), **(COREML_OK.get(name) or {})})   # the model's own last
        out.append(p)
    out.append("CPUExecutionProvider")
    return out, why


# ------------------------------------------------------------------------------------------------- what ran
def placement(profile_file) -> Dict[str, dict]:
    """From the record onnxruntime writes of a run (session profiling): for each provider how many operators it
    ran and how long they took, {provider: {"nodes": n, "ms": t}}. {} when the file cannot be read. (A part of
    the graph an accelerator takes whole, as Core ML does, is one operator here: the time tells its weight.)"""
    try:
        ev = json.loads(Path(profile_file).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for e in ev if isinstance(ev, list) else []:
        a = e.get("args") if isinstance(e, dict) else None
        if not isinstance(a, dict) or e.get("cat") != "Node" or not str(e.get("name", "")).endswith("_kernel_time"):
            continue
        p = a.get("provider") or ""
        if p:
            d = out.setdefault(p, dict(nodes=0, ms=0.0))
            d["nodes"] += 1
            d["ms"] = round(d["ms"] + float(e.get("dur") or 0) / 1000.0, 3)
    return out


CPU_HEAVY = 0.5        # more of a run's operator time than this on the CPU, with an accelerator asked: said


def ran_on(place: Dict[str, dict]) -> Tuple[str, float]:
    """(the backend the run was really on, the share of its operator time spent on the CPU). A session whose
    operators all ran on the CPU is 'cpu' whatever its provider list says. CPU operators beside an accelerator
    (shape arithmetic, an operator the accelerator lacks) leave it on the accelerator; the share tells their
    weight."""
    total = sum(float(v.get("ms", 0)) for v in place.values())
    if not place:
        return "", 0.0
    acc = {BACKEND.get(p, p): v for p, v in place.items() if BACKEND.get(p, "cpu") != "cpu"}
    if not acc:
        return "cpu", 1.0
    best = max(acc, key=lambda b: (acc[b].get("ms", 0), acc[b].get("nodes", 0)))
    cpu_ms = sum(float(v.get("ms", 0)) for p, v in place.items() if BACKEND.get(p, "cpu") == "cpu")
    return best, (cpu_ms / total if total > 0 else 0.0)


def note(path, **fields) -> dict:
    """Adds to what is known of a model's session (asked, session_providers, placement, backend, reason, ms...)."""
    name = Path(str(path)).name
    with _lock:
        r = _records.setdefault(name, dict(file=name, model=model_name(path), t=time.time(), ran=False))
        for k, v in fields.items():
            if k == "reasons":
                r["reasons"] = list(dict.fromkeys(list(r.get("reasons") or []) + [x for x in v if x]))
            else:
                r[k] = v
        return dict(r)


def records() -> List[dict]:
    with _lock:
        return [dict(r) for r in sorted(_records.values(), key=lambda r: r.get("t", 0))]


def forget() -> None:
    with _lock:
        _records.clear()


def effective(r: dict) -> str:
    """The backend of one record: what its first run was on, or before a run the session's own first provider."""
    if r.get("ran") and r.get("backend"):
        return r["backend"]
    for p in r.get("session") or []:
        return BACKEND.get(p, "cpu")
    return ""


def summary_label(default: str) -> str:
    """For the status line: the device, told by what the models that asked for an accelerator got. As decided for
    the system while none was opened; '(일부 CPU)' when some of them ended on the CPU; 'CPU' when all did."""
    rs = [r for r in records() if any(a != "cpu" for a in r.get("asked") or [])]
    eff = [effective(r) for r in rs]
    eff = [e for e in eff if e]
    if not eff:
        return default
    acc = sorted({e for e in eff if e != "cpu"})
    if not acc:
        return "CPU"
    # (1.27) CUDA with the models that run on TensorRT RTX beside it: both named
    names = [LABEL.get(a, a) for a in acc] if set(acc) == {"cuda", "rtx"} else [LABEL.get(acc[0], acc[0])]
    return " + ".join(names) + (" (일부 CPU)" if "cpu" in eff else "")


def diagnostics() -> dict:
    """Everything the settings and the reports show: the system, the runtime, what it offers, the device, the
    fixed combination and how this computer stands against it, and each model session opened so far."""
    try:
        import onnxruntime as ort
        ort_v, avail = str(ort.__version__), list(ort.get_available_providers())
    except Exception as e:
        ort_v, avail = f"오류: {e}", []
    try:
        from ..core.settings import settings
        want = settings().gpu
    except Exception:
        want = "auto"
    g = gpu()
    dstate, dtext = driver_state(g)
    pk = runtime_packages()
    warn = []
    if len(pk) > 1:
        warn.append("ONNX Runtime 패키지가 여러 개 설치되어 있습니다 (" + ", ".join(f"{n} {v}" for n, v in pk)
                    + "): 하나만 남기세요. 서로의 파일을 덮어써 GPU 실행이 깨질 수 있습니다")
    if dtext:
        warn.append(dtext)
    if system() == "win32" and "cuda" not in offered() and g.get("vendor") == "nvidia":
        warn.append("NVIDIA GPU가 있지만 설치된 ONNX Runtime에 CUDA가 없습니다 (DirectML 또는 CPU로 실행)")
    try:
        r_ = rtx() if (want in ("auto", "cuda") and "cuda" in offered()) else dict(devices=[], why="", version="")
        rtx_d = dict(devices=len(r_.get("devices") or []), why=r_.get("why", ""), version=r_.get("version", ""),
                     files=sorted(rtx_files()))
    except Exception as e:
        rtx_d = dict(devices=0, why=str(e)[:160], version="", files=[])
    return dict(system=f"{platform.system()} {platform.release()} {platform.machine()}", python=platform.python_version(),
                onnxruntime=ort_v, packages=pk, available=avail, offered=offered(), setting=want, gpu=g, rtx=rtx_d,
                driver=dstate, pinned=dict(onnxruntime=ORT_PINNED, cuda=CUDA_RUNTIME, cudnn=CUDNN_RUNTIME,
                                           driver_min=".".join(str(x) for x in DRIVER_MIN.get(system(), ())),
                                           driver_checked=".".join(str(x) for x in DRIVER_BUILT.get(system(), ()))),
                coreml_models=sorted(COREML_OK), warnings=warn, models=records())


def text(d: Optional[dict] = None) -> str:
    """The diagnostics for a person to read (설정 > 추론 장치 진단, the log)."""
    d = diagnostics() if d is None else d
    g = d.get("gpu") or {}
    L = [f"시스템: {d['system']} · Python {d['python']} · ONNX Runtime {d['onnxruntime']}",
         "설치된 런타임 패키지: " + (", ".join(f"{n} {v}" for n, v in d.get("packages") or []) or "확인 못 함"),
         "런타임이 제공하는 실행 장치: " + (", ".join(d.get("available") or []) or "없음"),
         f"이 시스템에서 쓰는 순서: {' → '.join(LABEL.get(b, b) for b in d.get('offered') or [])} · 설정: {d.get('setting')}"]
    if g.get("vendor") == "nvidia":
        L.append(f"GPU: {g.get('name')} · VRAM {g.get('memory_mb', 0)} MB (사용 중 {g.get('used_mb', 0)} MB) · 드라이버 "
                 f"{g.get('driver')}" + (f" · GPU {g['count']}개" if g.get("count", 1) > 1 else ""))
    elif g.get("vendor") == "apple":
        L.append(f"칩: {g.get('name')} ({g.get('arch')}) · 메모리 {g.get('memory_mb', 0)} MB · macOS {g.get('driver')}")
    else:
        L.append("GPU: 확인되지 않음 (nvidia-smi 응답 없음)")
    p = d.get("pinned") or {}
    if "cuda" in (d.get("offered") or []) or g.get("vendor") == "nvidia":
        L.append(f"확인된 조합: ONNX Runtime {p.get('onnxruntime')} + CUDA {p.get('cuda')} + cuDNN {p.get('cudnn')}.x, "
                 f"NVIDIA 드라이버 {p.get('driver_checked')} 이상 (최소 {p.get('driver_min')})")
    rx = d.get("rtx") or {}
    if rx.get("devices"):
        L.append(f"TensorRT RTX {rx.get('version') or ''}: 사용 ({', '.join(rx.get('files') or [])})".replace("  ", " "))
    elif rx.get("why") and g.get("vendor") == "nvidia":
        L.append(f"TensorRT RTX: 쓰지 않음 ({rx['why']})")
    if d.get("system", "").startswith("Darwin"):
        L.append("Core ML로 실행하는 모델: " + (", ".join(d.get("coreml_models") or []) or "없음 (확인된 모델이 아직 없어 CPU로 실행)"))
    for w in d.get("warnings") or []:
        L.append("주의: " + w)
    ms = d.get("models") or []
    L.append("")
    L.append("모델별 실제 실행 (이 실행에서 연 모델):" if ms else "모델별 실제 실행: 아직 연 모델이 없습니다 (분석이나 보정을 한 번 하면 채워집니다)")
    for r in ms:
        asked = " → ".join(LABEL.get(a, a) for a in r.get("asked") or []) or "-"
        place = r.get("placement") or {}
        if place:
            pl = ", ".join(f"{LABEL.get(BACKEND.get(k, k), k)} {v.get('nodes', 0)}개({v.get('ms', 0):.0f} ms)"
                           for k, v in sorted(place.items(), key=lambda kv: -kv[1].get("ms", 0)))
        else:
            pl = "첫 실행 전" if not r.get("ran") else "배치 기록 없음"
        ran = LABEL.get(effective(r), effective(r) or "확인 전") + ("" if r.get("ran") else " (첫 실행 전)")
        L.append(f"- {r.get('model')} ({r.get('file')}): 요청 {asked} · 세션 {', '.join(LABEL.get(BACKEND.get(p, p), p) for p in r.get('session') or []) or '-'}"
                 f" · 실제 {ran}" + (f" · 연산 배치 {pl}" if pl else "")
                 + (f" · 첫 실행 {r['first_ms']:.0f} ms" if r.get("first_ms") else ""))
        for why in r.get("reasons") or []:
            L.append(f"    사유: {why}")
    return "\n".join(L)
